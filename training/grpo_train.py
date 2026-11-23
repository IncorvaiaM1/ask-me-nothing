"""Multi-turn GRPO on BrokerDesk: collect, update, reload. Needs a GPU, vLLM, torch, peft.

    python -m training.grpo_train --init-adapter checkpoints/sft-always_ask --arm tiered \\
        --seed 0 --out runs/grpo/tiered-s0

One process holds both an in-process vLLM engine (rollouts) and the Hugging Face model
being trained (LoRA). After every update the adapter is saved and vLLM switches to it,
so each batch is sampled from the current policy.

Arms (``training/config.py``): tiered, flat, shuffled, no_collection_cost. ``--steerable``
samples tier costs per group and writes them into the system prompt instead.
``--advantage turn`` credits each cost to the call that incurred it (turn-level
ablation); the default ``dialogue`` gives every token of a dialogue the same advantage.

Interrupted runs resume from the last saved step with ``--resume``.
"""

from __future__ import annotations

import argparse
import json
import os
import random
import statistics
import time
from pathlib import Path

from baselines.prompted import system_prompt
from brokerdesk.env import MAX_TURNS
from brokerdesk.policy import default_policy
from brokerdesk.tasks import load_split
from brokerdesk.tools import openai_tools
from eval.metrics import summarize
from training.batch_rollout import Episode, Job, run_jobs
from training.config import REWARD_CONFIGS, describe_costs, sample_steerable
from training.masking import assistant_turns
from training.rollouts import group_advantages

MIN_STD = 0.05


# ------------------------------------------------------------------ advantages


def compute_advantages(episodes: list[Episode], mode: str = "dialogue") -> None:
    """Fill ``advantage`` (dialogue mode) or ``turn_advantages`` (turn mode) in place."""
    groups: dict[str, list[Episode]] = {}
    for e in episodes:
        groups.setdefault(e.job.group_id, []).append(e)
    for group in groups.values():
        rewards = [e.score["reward"] for e in group]
        for e, a in zip(group, group_advantages(rewards)):
            e.advantage = a
        if mode != "turn":
            continue
        std = statistics.pstdev(rewards)
        if std < 1e-6:
            for e in group:
                e.turn_advantages = [0.0] * e.score["turns"]
            continue
        scale = max(std, MIN_STD)
        outcome = [float(e.score["goal_met"]) for e in group]
        total_cost = [o - r for o, r in zip(outcome, rewards)]
        mean_o, mean_c = statistics.fmean(outcome), statistics.fmean(total_cost)
        for e, o in zip(group, outcome):
            costs = {int(k): v for k, v in e.score["turn_costs"].items()}
            adv = []
            for call in range(1, e.score["turns"] + 1):        # assistant turn k is tool call k + 1
                to_go = sum(v for t, v in costs.items() if t >= call)
                adv.append((o - mean_o) / scale - (to_go - mean_c) / scale)
            e.turn_advantages = adv


# ------------------------------------------------------------------ batch + step


def build_batch(tokenizer, episodes: list[Episode], mode: str = "dialogue", max_length: int = 12288):
    """Token ids, mask and per-token advantages for every episode with a learning signal."""
    batch, skipped = [], 0
    for e in episodes:
        if mode == "dialogue" and e.advantage == 0.0:
            continue
        if mode == "turn" and not any(e.turn_advantages):
            continue
        try:
            ids, turns = assistant_turns(tokenizer, e.messages, tools=openai_tools(), max_length=max_length,
                                         enable_thinking=False)
        except ValueError:
            skipped += 1
            continue
        mask = [int(t >= 0) for t in turns]
        if mode == "dialogue":
            adv = [e.advantage] * len(ids)
        else:
            adv = [e.turn_advantages[t] if 0 <= t < len(e.turn_advantages) else 0.0 for t in turns]
        batch.append((ids, mask, adv))
    return batch, skipped


def train_step(model, optimizer, batch, device, clip: float = 1.0) -> dict:
    """One optimizer step over the batch, one dialogue at a time (gradient accumulation)."""
    import torch

    from training.grpo import grpo_loss
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer.zero_grad(set_to_none=True)
    total, tokens = 0.0, 0
    for ids, mask, adv in batch:
        x = torch.tensor([ids], device=device)
        loss = grpo_loss(model(input_ids=x).logits, x, torch.tensor([mask], device=device),
                         torch.tensor([adv], device=device)) / len(batch)
        loss.backward()
        total += loss.item()
        tokens += len(ids)
    grad_norm = float(torch.nn.utils.clip_grad_norm_(params, clip)) if batch else 0.0
    if batch:
        optimizer.step()
    return {"loss": total, "tokens": tokens, "dialogues": len(batch), "grad_norm": grad_norm}


# ------------------------------------------------------------------ jobs


def make_jobs(tasks, group_size: int, step: int, arm: str, steerable: bool, rng: random.Random,
              base_system: str) -> list[Job]:
    jobs = []
    for t in tasks:
        weights = sample_steerable(rng) if steerable else REWARD_CONFIGS[arm]
        system = base_system + ("\n\n" + describe_costs(weights) if steerable else "")
        gid = f"{t.task_id}:{step}"
        jobs += [Job(task=t, system=system, seed=step, weights=weights, group_id=gid, index=i)
                 for i in range(group_size)]
    return jobs


def episode_stats(episodes: list[Episode]) -> dict:
    s = summarize([e.score for e in episodes])
    groups: dict[str, list[Episode]] = {}
    for e in episodes:
        groups.setdefault(e.job.group_id, []).append(e)
    s["groups_with_signal"] = sum(len({e.score["reward"] for e in g}) > 1 for g in groups.values())
    s["groups_with_success_and_failure"] = sum(len({e.score["success"] for e in g}) > 1 for g in groups.values())
    return {k: s[k] for k in ("success", "secure_success", "reward", "required_recall",
                              "unneeded_questions_per_ep", "needed_questions_per_ep",
                              "unneeded_requests_per_ep_by_tier", "unneeded_by_field",
                              "violations_per_ep", "timeout_rate", "invalid_calls_per_ep",
                              "groups_with_signal", "groups_with_success_and_failure")}


# ------------------------------------------------------------------ main


def main(argv=None) -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--model", default="Qwen/Qwen3-1.7B")
    p.add_argument("--init-adapter", required=True, help="SFT adapter to start from")
    p.add_argument("--arm", default="tiered", choices=sorted(REWARD_CONFIGS))
    p.add_argument("--steerable", action="store_true", help="sample tier costs per group and show them")
    p.add_argument("--advantage", default="dialogue", choices=["dialogue", "turn"])
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--steps", type=int, default=60)
    p.add_argument("--tasks-per-step", type=int, default=16)
    p.add_argument("--group-size", type=int, default=8)
    p.add_argument("--temperature", type=float, default=1.0)
    p.add_argument("--lr", type=float, default=2e-5)
    p.add_argument("--eval-every", type=int, default=10)
    p.add_argument("--eval-n", type=int, default=60, help="validation tasks per evaluation")
    p.add_argument("--gpu-mem", type=float, default=0.45, help="vLLM share of GPU memory")
    p.add_argument("--out", required=True)
    p.add_argument("--resume", action="store_true")
    args = p.parse_args(argv)

    import torch
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer

    from training.batch_rollout import VLLMEngine

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    log_path = out / "log.jsonl"
    start, adapter = 0, args.init_adapter
    if args.resume:
        steps = sorted(int(d.name.split("_")[1]) for d in out.glob("step_*") if (d / "adapter_config.json").exists())
        if steps:
            start, adapter = steps[-1], str(out / f"step_{steps[-1]:04d}")
            print(f"resuming from {adapter}")
    (out / "config.json").write_text(json.dumps(vars(args), indent=2))

    random.seed(args.seed)
    torch.manual_seed(args.seed)
    policy = default_policy()
    train_tasks = load_split("train", policy)
    val_tasks = load_split("val", policy)[: args.eval_n]
    base_system = system_prompt(policy, MAX_TURNS)

    engine = VLLMEngine(args.model, gpu_memory_utilization=args.gpu_mem, seed=args.seed)
    engine.set_adapter(adapter)
    tok = AutoTokenizer.from_pretrained(args.model)
    base = AutoModelForCausalLM.from_pretrained(args.model, torch_dtype=torch.bfloat16).cuda()
    model = PeftModel.from_pretrained(base, adapter, is_trainable=True)
    model.gradient_checkpointing_enable()
    model.enable_input_require_grads()
    optimizer = torch.optim.AdamW([q for q in model.parameters() if q.requires_grad], lr=args.lr)

    def evaluate(step: int) -> None:
        weights = REWARD_CONFIGS["tiered" if args.steerable else args.arm]
        system = base_system + ("\n\n" + describe_costs(weights) if args.steerable else "")
        eps = run_jobs(engine, [Job(task=t, system=system, seed=10_000, weights=weights, group_id=t.task_id)
                                for t in val_tasks], policy, temperature=0.0)
        row = {"step": step, "eval": episode_stats(eps)}
        print(json.dumps(row))
        with open(log_path, "a") as fh:
            fh.write(json.dumps(row) + "\n")

    if start == 0:
        evaluate(0)
    for step in range(start + 1, args.steps + 1):
        t0 = time.time()
        rng = random.Random(f"{args.seed}:{step}")
        tasks = rng.sample(train_tasks, args.tasks_per_step)
        episodes = run_jobs(engine, make_jobs(tasks, args.group_size, step, args.arm, args.steerable, rng,
                                              base_system), policy, temperature=args.temperature)
        t_roll = time.time() - t0
        compute_advantages(episodes, args.advantage)
        batch, skipped = build_batch(tok, episodes, args.advantage)
        model.train()
        upd = train_step(model, optimizer, batch, "cuda")
        path = out / f"step_{step:04d}"
        model.save_pretrained(path)
        engine.set_adapter(str(path))
        row = {"step": step, "train": episode_stats(episodes), "update": upd, "skipped": skipped,
               "seconds_rollout": round(t_roll, 1), "seconds_total": round(time.time() - t0, 1)}
        print(json.dumps({k: row[k] for k in ("step", "update", "seconds_total")}
                         | {"success": row["train"]["success"], "reward": row["train"]["reward"],
                            "unneeded_q": row["train"]["unneeded_questions_per_ep"]}))
        with open(log_path, "a") as fh:
            fh.write(json.dumps(row) + "\n")
        old = out / f"step_{step - 2:04d}"            # keep the last two steps and every eval step
        if old.exists() and (step - 2) % args.eval_every:
            for f in old.iterdir():
                f.unlink()
            old.rmdir()
        if step % args.eval_every == 0 or step == args.steps:
            evaluate(step)
    model.save_pretrained(out / "final")
    print(f"done; adapter in {out / 'final'}, log in {log_path}")


if __name__ == "__main__":
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    main()
