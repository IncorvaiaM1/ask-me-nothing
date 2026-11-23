"""Evaluate a model, with or without a LoRA adapter, on BrokerDesk. Needs a GPU and vLLM.

Every model condition in the paper goes through this one script, so prompts, tools,
turn limits and parsing are identical across rows.

    # prompted base model, validation, clean and injected (development attacks)
    python -m eval.eval_adapter --label qwen1.7b-prompted --split val --injection paired

    # a trained adapter on the test split, five sampled rollouts per task for pass^k
    python -m eval.eval_adapter --adapter runs/grpo/tiered-s0/final --label tiered-s0 \\
        --split test --rollouts 5 --temperature 0.7

    # held-out wording, held-out attacks
    python -m eval.eval_adapter --adapter ... --split test_paraphrase --injection paired --attack-set heldout

    # LLM-simulated clients (serve the client model separately, e.g. on port 8001)
    python -m eval.eval_adapter --adapter ... --split test --client llm --persona all \\
        --client-model Qwen/Qwen3-8B --client-url http://localhost:8001/v1

    # steerability: show fixed collection costs in the prompt
    python -m eval.eval_adapter --adapter runs/grpo/steer-s0/final --steer-costs 0.001,0.002,0.5 ...

Writes results/<label>__<split>[+suffix].jsonl (episodes with transcripts) and
.summary.json, in the same format as eval.run_baseline.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from baselines.prompted import system_prompt
from brokerdesk.env import MAX_TURNS, RewardWeights
from brokerdesk.injection import assign_injection
from brokerdesk.policy import default_policy
from brokerdesk.tasks import SPLITS, load_split
from eval.metrics import format_attack_table, format_table, summarize
from training.batch_rollout import Job, run_jobs
from training.config import REWARD_CONFIGS, describe_costs


def build_jobs(tasks, args, inject: bool, persona: str | None, system: str, weights) -> list[Job]:
    factory = None
    if persona is not None:
        from brokerdesk.clients import LLMClient, openai_chat
        chat = openai_chat(args.client_model, args.client_url, seed=args.seed)
        factory = lambda task, seed, p=persona: LLMClient(task, seed, p, chat)       # noqa: E731
    jobs = []
    for t in tasks:
        injection = assign_injection(t.task_id, attack_set=args.attack_set) if inject else None
        for r in range(args.rollouts):
            jobs.append(Job(task=t, system=system, seed=args.seed + r, weights=weights, injection=injection,
                            group_id=t.task_id, index=r, client_factory=factory))
    return jobs


def main(argv=None) -> dict:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--model", default="Qwen/Qwen3-1.7B")
    p.add_argument("--adapter", default=None)
    p.add_argument("--label", required=True)
    p.add_argument("--split", default="val", choices=sorted(SPLITS))
    p.add_argument("--n", type=int, default=None)
    p.add_argument("--rollouts", type=int, default=1)
    p.add_argument("--temperature", type=float, default=0.0)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--thinking", action="store_true")
    p.add_argument("--max-tokens", type=int, default=512)
    p.add_argument("--plain-prompt", action="store_true", help="drop the need-to-know prompt section")
    p.add_argument("--arm", default="tiered", choices=sorted(REWARD_CONFIGS), help="weights for scoring")
    p.add_argument("--steer-costs", default=None, help="low,sensitive,restricted costs shown in the prompt")
    p.add_argument("--injection", default="none", choices=["none", "paired", "only"])
    p.add_argument("--attack-set", default="dev", choices=["dev", "heldout"])
    p.add_argument("--client", default="scripted", choices=["scripted", "llm"])
    p.add_argument("--persona", default="all", help="persona name, or 'all'")
    p.add_argument("--client-model", default="Qwen/Qwen3-8B")
    p.add_argument("--client-url", default="http://localhost:8001/v1")
    p.add_argument("--gpu-mem", type=float, default=0.5)
    p.add_argument("--out", default="results")
    args = p.parse_args(argv)

    from training.batch_rollout import VLLMEngine
    policy = default_policy()
    tasks = load_split(args.split, policy)[: args.n]
    weights = REWARD_CONFIGS[args.arm]
    system = system_prompt(policy, MAX_TURNS, privacy_aware=not args.plain_prompt)
    if args.steer_costs:
        costs = tuple(float(x) for x in args.steer_costs.split(","))
        weights = RewardWeights(privacy_cost=costs)
        system += "\n\n" + describe_costs(weights)
    if args.thinking:
        args.max_tokens = max(args.max_tokens, 3072)

    engine = VLLMEngine(args.model, gpu_memory_utilization=args.gpu_mem, thinking=args.thinking, seed=args.seed)
    engine.set_adapter(args.adapter)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    personas = [None]
    if args.client == "llm":
        from brokerdesk.clients import PERSONAS
        personas = list(PERSONAS) if args.persona == "all" else [args.persona]
    conditions = {"none": [False], "paired": [False, True], "only": [True]}[args.injection]
    clean, injected = {}, {}
    for persona in personas:
        for inject in conditions:
            suffix = (f"+llm-{persona}" if persona else "") + (f"+inj-{args.attack_set}" if inject else "")
            stem = f"{args.label}{suffix}__{args.split}"
            eps = run_jobs(engine, build_jobs(tasks, args, inject, persona, system, weights), policy,
                           temperature=args.temperature, max_tokens=args.max_tokens)
            with open(out / f"{stem}.jsonl", "w", encoding="utf-8") as fh:
                for e in eps:
                    fh.write(json.dumps({"score": {**e.score, "rollout_seed": e.job.seed},
                                         "transcript": e.transcript}) + "\n")
            summary = summarize([e.score for e in eps])
            summary.update(agent=args.label + suffix, split=args.split, config=vars(args))
            (out / f"{stem}.summary.json").write_text(json.dumps(summary, indent=2))
            (injected if inject else clean)[args.label + (f"+llm-{persona}" if persona else "")] = summary

    print(f"split={args.split}  tasks={len(tasks)}  rollouts={args.rollouts}\n")
    if clean:
        print(format_table(clean))
    if injected:
        print("\nInjected:\n")
        print(format_table(injected))
    if clean and injected:
        print("\nPaired:\n")
        print(format_attack_table(clean, injected))
    return {"clean": clean, "injected": injected}


if __name__ == "__main__":
    main()
