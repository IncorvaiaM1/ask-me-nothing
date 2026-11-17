"""Run baselines on a BrokerDesk split and print the milestone table.

Scripted references (no GPU, runs anywhere):

    python -m eval.run_baseline --agents need_to_know never_ask always_ask --split val

Prompted small model, served by vLLM inside a PACE-ICE GPU session:

    vllm serve Qwen/Qwen3-1.7B --enable-auto-tool-choice --tool-call-parser hermes --port 8000
    python -m eval.run_baseline --agents llm --model Qwen/Qwen3-1.7B \
        --base-url http://localhost:8000/v1 --split val --no-thinking

Prompted frontier reference (reads OPENAI_API_KEY, or pass --api-key-env):

    python -m eval.run_baseline --agents llm --model <model name> --split val

Add --injection paired to run every task twice, clean and with attacker text in the
untrusted tool outputs, and print the paired attack table (the injected run is saved
as results/<label>+inj__<split>.*).

Each run writes results/<label>__<split>.jsonl (one episode per line, with the full
transcript) and results/<label>__<split>.summary.json.
"""

from __future__ import annotations

import argparse
import json
import os
import re
from pathlib import Path

from baselines.scripted import MODES, HeuristicAgent
from brokerdesk.injection import assign_injection
from brokerdesk.policy import default_policy
from brokerdesk.tasks import SPLITS, load_split
from eval.metrics import format_attack_table, format_table, summarize
from eval.rollout import run_episode


def build_agent(name: str, args, policy):
    if name in MODES:
        return HeuristicAgent(name, policy)
    if name == "llm":
        from baselines.prompted import LLMAgent
        if not args.model:
            raise SystemExit("--model is required for the llm agent")
        extra = {"chat_template_kwargs": {"enable_thinking": False}} if args.no_thinking else None
        return LLMAgent(args.model, base_url=args.base_url, api_key=os.environ.get(args.api_key_env),
                        policy=policy, max_turns=args.max_turns, privacy_aware=not args.plain_prompt,
                        temperature=args.temperature, extra_body=extra)
    raise SystemExit(f"unknown agent {name!r}; choose from {MODES + ('llm',)}")


def main(argv=None) -> dict:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--agents", nargs="+", default=list(MODES))
    p.add_argument("--split", default="val", choices=sorted(SPLITS))
    p.add_argument("--n", type=int, default=None, help="use only the first n tasks of the split")
    p.add_argument("--rollouts", type=int, default=1, help="rollouts per task (5 for pass^1 and pass^3)")
    p.add_argument("--seed", type=int, default=0, help="base seed for the price paths")
    p.add_argument("--max-turns", type=int, default=12)
    p.add_argument("--model", default=None)
    p.add_argument("--base-url", default=None)
    p.add_argument("--api-key-env", default="OPENAI_API_KEY")
    p.add_argument("--temperature", type=float, default=0.0)
    p.add_argument("--no-thinking", action="store_true", help="disable Qwen3 thinking mode (vLLM)")
    p.add_argument("--plain-prompt", action="store_true", help="drop the need-to-know prompt section")
    p.add_argument("--label", default=None, help="name for the llm run in tables and file names")
    p.add_argument("--injection", default="none", choices=["none", "paired", "only"],
                   help="paired: run clean and injected; only: injected runs alone")
    p.add_argument("--out", default="results")
    args = p.parse_args(argv)

    policy = default_policy()
    tasks = load_split(args.split, policy)[: args.n]
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    conditions = {"none": [False], "paired": [False, True], "only": [True]}[args.injection]
    rows, injected_rows = {}, {}
    for name in args.agents:
        label = name if name != "llm" else (args.label or re.sub(r"[^\w.-]+", "_", args.model or "llm"))
        for inject in conditions:
            agent = build_agent(name, args, policy)
            stem = f"{label}{'+inj' if inject else ''}__{args.split}"
            scores = []
            with open(out_dir / f"{stem}.jsonl", "w", encoding="utf-8") as fh:
                for task in tasks:
                    injection = assign_injection(task.task_id) if inject else None
                    for r in range(args.rollouts):
                        ep = run_episode(task, agent, policy, seed=args.seed + r, max_turns=args.max_turns,
                                         injection=injection)
                        scores.append(ep["score"])
                        fh.write(json.dumps(ep) + "\n")
            summary = summarize(scores)
            summary["agent"] = label
            summary["split"] = args.split
            summary["config"] = {k: v for k, v in vars(args).items() if k != "agents"}
            if hasattr(agent, "usage"):
                summary["token_usage"] = agent.usage
            (out_dir / f"{stem}.summary.json").write_text(json.dumps(summary, indent=2))
            (injected_rows if inject else rows)[label] = summary

    print(f"split={args.split}  tasks={len(tasks)}  rollouts={args.rollouts}  seed={args.seed}\n")
    if rows:
        print(format_table(rows))
    if injected_rows:
        print("\nWith injected tool outputs:\n")
        print(format_table(injected_rows))
    if rows and injected_rows:
        print("\nPaired clean vs injected:\n")
        print(format_attack_table(rows, injected_rows))
    return {"clean": rows, "injected": injected_rows}


if __name__ == "__main__":
    main()
