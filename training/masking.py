"""Token ids and an assistant-only loss mask for one dialogue.

The policy gradient must touch only the tokens the model itself produced: not the
system prompt, the client's messages, or tool observations. Chat templates differ,
so instead of hard-coding one, this finds each assistant span by rendering the
dialogue twice per assistant message with the tokenizer's own template:

    prefix = render(messages[:i], add_generation_prompt=True)
    full   = render(messages[:i + 1])

The assistant's tokens are the part of ``full`` after the header it shares with
``prefix`` (and with a rendering of an empty assistant turn, as a cross-check). That includes the end-of-turn token, so the model also learns
when to stop.

Works with any object exposing Hugging Face's ``apply_chat_template``. It has been
tested against a stand-in tokenizer only. Check it once on the real Qwen3 tokenizer
during the pilot (``describe_mask`` prints what is and is not trained on), in
particular with thinking disabled: Qwen3's generation prompt then ends with an
empty think block that the template does not write back into past assistant turns.
"""

from __future__ import annotations


def _render(tokenizer, messages, tools, add_generation_prompt: bool, template_kwargs: dict) -> list[int]:
    out = tokenizer.apply_chat_template(messages, tools=tools, tokenize=True,
                                        add_generation_prompt=add_generation_prompt, **template_kwargs)
    if hasattr(out, "keys") and "input_ids" in out:      # newer versions may return an encoding
        out = out["input_ids"]
    return list(out)


def _common_prefix(a: list[int], b: list[int]) -> int:
    n = 0
    for x, y in zip(a, b):
        if x != y:
            break
        n += 1
    return n


def assistant_mask(tokenizer, messages: list[dict], tools: list[dict] | None = None,
                   max_length: int | None = None, **template_kwargs) -> tuple[list[int], list[int]]:
    """Return ``(input_ids, loss_mask)``; ``loss_mask[t]`` is 1 where token ``t`` was produced by the model.

    Raises ``ValueError`` if the template does not render prefixes consistently, or if
    the dialogue is longer than ``max_length`` (truncating would cut assistant turns
    out of a trajectory whose reward depends on them, so the caller should drop it).
    """
    while messages and messages[-1]["role"] != "assistant":
        messages = messages[:-1]                           # trailing tool output is never trained on
    input_ids = _render(tokenizer, messages, tools, False, template_kwargs)
    if max_length is not None and len(input_ids) > max_length:
        raise ValueError(f"dialogue is {len(input_ids)} tokens, over max_length={max_length}")
    mask = [0] * len(input_ids)
    for i, message in enumerate(messages):
        if message["role"] != "assistant":
            continue
        prefix = _render(tokenizer, messages[:i], tools, True, template_kwargs)
        full = _render(tokenizer, messages[: i + 1], tools, False, template_kwargs)
        # Two estimates of where the model's own tokens begin; either can overshoot when the
        # reply happens to start with the same tokens as what follows the header in the other
        # rendering (an empty think block, or the end-of-turn token), so take the earlier one.
        empty = _render(tokenizer, messages[:i] + [{"role": "assistant", "content": ""}], tools, False,
                        template_kwargs)
        start = min(_common_prefix(prefix, full), _common_prefix(empty, full))
        before = _render(tokenizer, messages[:i], tools, False, template_kwargs) if i else []
        if start < _common_prefix(before, full) or input_ids[: len(full)] != full:
            raise ValueError(f"chat template is not prefix-consistent at message {i}")
        for t in range(start, len(full)):
            mask[t] = 1
    return input_ids, mask


def describe_mask(tokenizer, input_ids: list[int], mask: list[int]) -> str:
    """Human-readable dump: trained spans in [[double brackets]], everything else plain."""
    parts, start = [], 0
    for t in range(1, len(input_ids) + 1):
        if t == len(input_ids) or mask[t] != mask[start]:
            text = tokenizer.decode(input_ids[start:t])
            parts.append(f"[[{text}]]" if mask[start] else text)
            start = t
    return "".join(parts)
