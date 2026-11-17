"""Token ids and an assistant-only loss mask for one dialogue.

The policy gradient must touch only the tokens the model itself produced: not the
system prompt, the client's messages, or tool observations. And each of those
tokens should sit in the context the model actually had when it produced them.

That second point rules out simply rendering the finished dialogue with the chat
template. Qwen3's template is not prefix-consistent: with thinking disabled the
generation prompt ends in an empty ``<think>`` block, the template writes that
block only into the *last* assistant turn of a dialogue, and drops it from every
earlier one. So the sequence is assembled turn by turn instead:

    for each assistant turn i:
        prompt_i  = render(messages[:i], add_generation_prompt=True)   # what the model was shown
        reply_i   = the tokens of turn i, taken from a rendering where it is a past turn
        sequence += (the part of prompt_i that is new since turn i-1) + reply_i

The result reads exactly like inference at every point where the model had to
predict: history as the template renders past turns, then the generation prompt,
then the model's own tokens through the end-of-turn token. Earlier turns keep
their generation-prompt tokens (for Qwen3 with thinking off, the empty think
block), which at inference the template would have dropped from history; that is
a few tokens per turn and the price of using one sequence per dialogue. For a
template that is prefix-consistent this reduces to the plain rendering.

Works with any object exposing Hugging Face's ``apply_chat_template``. Limits:
only the final ``content`` and tool calls of each turn are trained on, so with
thinking *enabled* the model's reasoning tokens are not in the sequence unless
the rollout stored them; the planned runs use thinking off.
"""

from __future__ import annotations

import json


def normalize_tool_calls(messages: list[dict]) -> list[dict]:
    """Copy of ``messages`` with tool-call arguments as dicts.

    The OpenAI wire format carries arguments as a JSON string; Hugging Face chat
    templates (Qwen3's included) expect a dict and would otherwise quote it twice.
    Arguments that are not valid JSON (a malformed model output) are left as text.
    """
    out = []
    for m in messages:
        if m.get("tool_calls"):
            calls = []
            for call in m["tool_calls"]:
                fn = dict(call["function"])
                if isinstance(fn.get("arguments"), str):
                    try:
                        fn["arguments"] = json.loads(fn["arguments"])
                    except json.JSONDecodeError:
                        pass
                calls.append({**call, "function": fn})
            m = {**m, "tool_calls": calls}
        out.append(m)
    return out


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

    Raises ``ValueError`` if past turns do not render the same way from one prompt to
    the next, or if the dialogue is longer than ``max_length`` (truncating would cut
    assistant turns out of a trajectory whose reward depends on them, so the caller
    should drop it).
    """
    messages = normalize_tool_calls(messages)
    while messages and messages[-1]["role"] != "assistant":
        messages = messages[:-1]                           # trailing tool output is never trained on
    turns = [i for i, m in enumerate(messages) if m["role"] == "assistant"]
    if not turns:
        raise ValueError("dialogue has no assistant turn")
    dummy = {"role": "user", "content": "x"}

    def past(k: int, extra: int = 1) -> list[int]:
        """messages[:k] followed by dummy user turns, so every assistant turn renders as a past turn."""
        return _render(tokenizer, messages[:k] + [dummy] * extra, tools, False, template_kwargs)

    dummy_len = len(past(turns[0] + 1, 2)) - len(past(turns[0] + 1))

    input_ids: list[int] = []
    mask: list[int] = []
    prev_end, prev_past = 0, []
    for i in turns:
        prompt = _render(tokenizer, messages[:i], tools, True, template_kwargs)
        rendered = past(i + 1)
        end = len(rendered) - dummy_len                    # just past this turn's end-of-turn tokens
        # Where the model's own tokens begin: after the assistant header. Two estimates, either
        # of which can overshoot when the reply starts with the same tokens as what follows the
        # header in the other rendering, so take the earlier one.
        empty = _render(tokenizer, messages[:i] + [{"role": "assistant", "content": ""}, dummy],
                        tools, False, template_kwargs)
        start = min(_common_prefix(prompt, rendered), _common_prefix(empty, rendered))
        if not (prev_end <= start < end) or prompt[:prev_end] != prev_past[:prev_end] \
                or rendered[:prev_end] != prev_past[:prev_end]:
            raise ValueError(f"chat template renders history inconsistently at message {i}")
        reply = rendered[start:end]
        new_context = prompt[prev_end:]
        input_ids += new_context + reply
        mask += [0] * len(new_context) + [1] * len(reply)
        prev_end, prev_past = end, rendered
    if max_length is not None and len(input_ids) > max_length:
        raise ValueError(f"dialogue is {len(input_ids)} tokens, over max_length={max_length}")
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
