"""Supervised warm start: LoRA fine-tuning on scripted-teacher dialogues. Needs a GPU.

    python -m training.sft --model Qwen/Qwen3-1.7B --teacher always_ask --out checkpoints/sft-always_ask

Why a warm start, and why this teacher: the prompted model completes no tasks, so
every rollout in a GRPO group would fail and the only reward differences would be
the question and privacy costs. The default teacher is the *over-asking* scripted
agent: it trades correctly but asks for everything. A model fine-tuned on it can do
the job and still has the habit the paper is about, which leaves RL the question it
is meant to answer. The need-to-know agent stays the reference, not the teacher.

Dialogues are kept only when the teacher succeeded with no violations, rendered
exactly as for RL (``training/masking.py``), and trained with cross-entropy on the
assistant tokens only. The adapter is saved for vLLM's ``--lora-modules``.

The training loop in this file had not been run when it was written.
"""

from __future__ import annotations

import argparse
import json
import random
import time
from pathlib import Path

from baselines.prompted import system_prompt
from baselines.scripted import MODES, HeuristicAgent
from brokerdesk.policy import Policy, default_policy
from brokerdesk.tasks import Task, load_split
from eval.rollout import run_episode
from training.rollouts import messages_from_transcript

LORA_TARGETS = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]


def build_dialogues(tasks: list[Task], teacher: str, policy: Policy, seeds: int = 2,
                    max_turns: int = 12) -> tuple[list[list[dict]], dict]:
    """Teacher dialogues as chat messages, one per (task, price-path seed), successes only."""
    system = system_prompt(policy, max_turns)
    dialogues, tried = [], 0
    for task in tasks:
        for seed in range(seeds):
            tried += 1
            ep = run_episode(task, HeuristicAgent(teacher, policy), policy, seed=seed, max_turns=max_turns)
            if ep["score"]["success"] and ep["score"]["n_leaks"] == 0:
                dialogues.append(messages_from_transcript(system, ep["transcript"]))
    return dialogues, {"teacher": teacher, "attempted": tried, "kept": len(dialogues)}


def main(argv=None) -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--model", default="Qwen/Qwen3-1.7B")
    p.add_argument("--teacher", default="always_ask", choices=[m for m in MODES if m != "gullible"])
    p.add_argument("--out", required=True)
    p.add_argument("--epochs", type=int, default=3)
    p.add_argument("--seeds-per-task", type=int, default=2, help="price paths per training task")
    p.add_argument("--lr", type=float, default=2e-4)
    p.add_argument("--accumulate", type=int, default=8, help="dialogues per optimizer step")
    p.add_argument("--lora-rank", type=int, default=16)
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args(argv)

    import torch
    from peft import LoraConfig, get_peft_model
    from transformers import AutoModelForCausalLM, AutoTokenizer

    from brokerdesk.tools import openai_tools
    from training.masking import assistant_mask

    assert torch.cuda.is_available(), "no GPU visible"
    torch.manual_seed(args.seed)
    gpu = torch.cuda.get_device_name(0)
    print("GPU:", gpu)
    policy = default_policy()
    tok = AutoTokenizer.from_pretrained(args.model)

    def encode(dialogues):
        out = []
        for messages in dialogues:
            ids, mask = assistant_mask(tok, messages, tools=openai_tools(), enable_thinking=False)
            labels = [t if m else -100 for t, m in zip(ids, mask)]
            out.append((torch.tensor([ids]), torch.tensor([labels])))
        return out

    train_dialogues, info = build_dialogues(load_split("train", policy), args.teacher, policy, args.seeds_per_task)
    val_dialogues, _ = build_dialogues(load_split("val", policy), args.teacher, policy, 1)
    train, val = encode(train_dialogues), encode(val_dialogues)
    tokens = sum(x.shape[1] for x, _ in train)
    print(f"teacher={args.teacher}: kept {info['kept']} of {info['attempted']} training dialogues, "
          f"{tokens:,} tokens; {len(val)} validation dialogues")

    model = AutoModelForCausalLM.from_pretrained(args.model, torch_dtype=torch.bfloat16).cuda()
    model = get_peft_model(model, LoraConfig(r=args.lora_rank, lora_alpha=2 * args.lora_rank,
                                             lora_dropout=0.05, task_type="CAUSAL_LM",
                                             target_modules=LORA_TARGETS))
    model.print_trainable_parameters()
    params = [q for q in model.parameters() if q.requires_grad]
    opt = torch.optim.AdamW(params, lr=args.lr, weight_decay=0.0)
    total_steps = max(1, args.epochs * len(train) // args.accumulate)
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: max(0.0, 1.0 - s / total_steps))

    def val_loss() -> float:
        model.eval()
        with torch.no_grad():
            losses = [model(input_ids=x.cuda(), labels=y.cuda()).loss.item() for x, y in val]
        model.train()
        return sum(losses) / len(losses)

    print(f"before training: validation loss {val_loss():.4f}")
    rng = random.Random(args.seed)
    history, t0 = [], time.time()
    for epoch in range(args.epochs):
        rng.shuffle(train)
        running, n = 0.0, 0
        opt.zero_grad(set_to_none=True)
        for k, (x, y) in enumerate(train, 1):
            loss = model(input_ids=x.cuda(), labels=y.cuda()).loss
            (loss / args.accumulate).backward()
            running, n = running + loss.item(), n + 1
            if k % args.accumulate == 0 or k == len(train):
                torch.nn.utils.clip_grad_norm_(params, 1.0)
                opt.step()
                sched.step()
                opt.zero_grad(set_to_none=True)
        vl = val_loss()
        history.append({"epoch": epoch + 1, "train_loss": running / n, "val_loss": vl})
        print(f"epoch {epoch + 1}: train loss {running / n:.4f}  validation loss {vl:.4f}  "
              f"({time.time() - t0:.0f}s elapsed)")

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(out)
    (out / "sft_run.json").write_text(json.dumps({
        "args": vars(args), "data": info, "train_tokens": tokens, "history": history,
        "gpu": gpu, "seconds": round(time.time() - t0), "lora_targets": LORA_TARGETS,
        "peak_gpu_gb": round(torch.cuda.max_memory_allocated() / 2**30, 1)}, indent=2))
    print(f"saved adapter to {out} ({time.time() - t0:.0f}s, "
          f"peak {torch.cuda.max_memory_allocated() / 2**30:.1f} GB)")


if __name__ == "__main__":
    main()
