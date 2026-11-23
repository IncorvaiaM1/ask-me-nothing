"""Collect one batch of grouped rollouts and write it as JSONL.

Dry run with no model (cycles the scripted agents inside each group, so groups
have reward spread and the whole pipeline can be exercised on a laptop):

    python -m training.collect --agent mixed --split train --n 8 --group-size 4 --out runs/rollouts/dry.jsonl

Sampling from the current policy served by vLLM:

    python -m training.collect --agent llm --model Qwen/Qwen3-0.6B --base-url http://localhost:8000/v1 \
        --split train --n 16 --group-size 8 --temperature 0.8 --workers 8 --no-thinking \
        --reward full --out runs/rollouts/step0000.jsonl

Each line is one dialogue: chat messages, reward, group-relative advantage, and
the full per-episode score.
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

from baselines.prompted import system_prompt
from baselines.scripted import HeuristicAgent
from brokerdesk.env import MAX_TURNS
from brokerdesk.policy import default_policy
from brokerdesk.tasks import SPLITS, generate_tasks
from training.config import REWARD_CONFIGS
from training.rollouts import batch_stats, collect_batch, write_jsonl

_MIX = ("need_to_know", "never_ask", "always_ask")


def main(argv=None) -> dict:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--agent", default="mixed", choices=["mixed", "llm", *_MIX])
    p.add_argument("--split", default="train", choices=sorted(SPLITS))
    p.add_argument("--n", type=int, default=8, help="number of tasks (groups) in the batch")
    p.add_argument("--offset", type=int, default=0, help="index of the first task in the split")
    p.add_argument("--underspecified-frac", type=float, default=None,
                   help="curriculum: regenerate the split with this share of tasks that need a question")
    p.add_argument("--group-size", type=int, default=8)
    p.add_argument("--reward", default="full", choices=sorted(REWARD_CONFIGS))
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--max-turns", type=int, default=MAX_TURNS)
    p.add_argument("--workers", type=int, default=1, help="concurrent rollouts inside a group")
    p.add_argument("--model", default=None)
    p.add_argument("--base-url", default=None)
    p.add_argument("--api-key-env", default="OPENAI_API_KEY")
    p.add_argument("--temperature", type=float, default=0.8)
    p.add_argument("--no-thinking", action="store_true")
    p.add_argument("--out", required=True)
    args = p.parse_args(argv)

    policy = default_policy()
    count, split_seed, held_out = SPLITS[args.split]
    frac = {} if args.underspecified_frac is None else {"underspecified_frac": args.underspecified_frac}
    tasks = generate_tasks(count, split_seed, policy=policy, held_out_wording=held_out,
                           **frac)[args.offset: args.offset + args.n]
    system = system_prompt(policy, args.max_turns)

    if args.agent == "llm":
        from baselines.prompted import LLMAgent
        if not args.model:
            raise SystemExit("--model is required for the llm agent")
        extra = {"chat_template_kwargs": {"enable_thinking": False}} if args.no_thinking else None

        def make_agent(i):
            return LLMAgent(args.model, base_url=args.base_url, api_key=os.environ.get(args.api_key_env),
                            policy=policy, max_turns=args.max_turns, temperature=args.temperature,
                            extra_body=extra)
    elif args.agent == "mixed":
        def make_agent(i):
            return HeuristicAgent(_MIX[i % len(_MIX)], policy)
    else:
        def make_agent(i):
            return HeuristicAgent(args.agent, policy)

    batch = collect_batch(tasks, make_agent, args.group_size, system, seed=args.seed,
                          weights=REWARD_CONFIGS[args.reward], policy=policy,
                          max_turns=args.max_turns, workers=args.workers)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    write_jsonl(batch, args.out)
    stats = batch_stats(batch)
    print(f"wrote {args.out}: " + ", ".join(f"{k}={v:.3f}" if isinstance(v, float) else f"{k}={v}"
                                              for k, v in stats.items()))
    return stats


if __name__ == "__main__":
    main()
