"""Pilot check 1: the assistant-only loss mask on the real tokenizer. CPU is enough.

    python -m training.check_mask --model Qwen/Qwen3-0.6B

Prints one dialogue with the trained spans in [[double brackets]] for you to read,
and fails loudly if the mask is wrong in a way a program can detect.
"""

from __future__ import annotations

import argparse
import json

from baselines.prompted import system_prompt
from baselines.scripted import HeuristicAgent
from brokerdesk.policy import default_policy
from brokerdesk.tasks import load_split
from brokerdesk.tools import openai_tools
from eval.rollout import run_episode
from training.masking import assistant_mask, describe_mask
from training.rollouts import messages_from_transcript


def main(argv=None) -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--model", default="Qwen/Qwen3-0.6B")
    p.add_argument("--n", type=int, default=20, help="dialogues to check")
    args = p.parse_args(argv)

    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(args.model)
    policy = default_policy()
    system = system_prompt(policy)
    tasks = load_split("train", policy)[: args.n]

    for thinking in (False, True):
        lengths, fracs = [], []
        for k, task in enumerate(tasks):
            ep = run_episode(task, HeuristicAgent("always_ask", policy), policy)
            messages = messages_from_transcript(system, ep["transcript"])
            ids, mask = assistant_mask(tok, messages, tools=openai_tools(), enable_thinking=thinking)
            trained = tok.decode([i for i, m in zip(ids, mask) if m])
            untrained = tok.decode([i for i, m in zip(ids, mask) if not m])
            calls = [m["tool_calls"][0]["function"] for m in messages if m["role"] == "assistant"]
            # Every call the agent made must be inside the trained text, with its arguments ...
            for call in calls:
                assert call["name"] in trained, (call, "tool name missing from trained tokens")
                for value in json.loads(call["arguments"]).values():
                    if isinstance(value, str):
                        assert value in trained, (value, "argument missing from trained tokens")
            # ... and nothing the environment said may be.
            assert task.request not in trained and task.request in untrained
            assert "book_value" not in trained and "Desk policy" not in trained
            assert sum(mask) > 0 and trained.count("<|im_end|>") == len(calls), "one end-of-turn per assistant turn"
            lengths.append(len(ids))
            fracs.append(sum(mask) / len(ids))
            if k == 0:
                text = describe_mask(tok, ids, mask)
                start = text.find(task.request)        # skip the system prompt and tool schemas
                print(f"\n===== thinking={thinking}: first dialogue after the system prompt =====\n")
                print(text[max(start - 40, 0):][:2500])
        print(f"\nthinking={thinking}: {len(tasks)} dialogues OK; tokens per dialogue "
              f"min {min(lengths)} / mean {sum(lengths) // len(lengths)} / max {max(lengths)}; "
              f"assistant share {100 * sum(fracs) / len(fracs):.1f}%")
    print("\nMASK CHECK PASSED. Read the dump above once: only the agent's own tool calls should be bracketed.")


if __name__ == "__main__":
    main()
