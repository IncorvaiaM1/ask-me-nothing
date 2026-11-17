"""Pilot check 2: one LoRA gradient step of the reference loss, with timing. Needs a GPU.

    python -m training.check_loss --model Qwen/Qwen3-0.6B
    python -m training.check_loss --model Qwen/Qwen3-1.7B

Uses scripted-agent dialogues as stand-in rollouts, so no inference server is
needed. Reports tokens per second and peak memory for the training half of a GRPO
step, which is what the compute plan needs. This script had never been run when it
was written; if it fails, the traceback is the finding.
"""

from __future__ import annotations

import argparse
import time

from baselines.prompted import system_prompt
from baselines.scripted import HeuristicAgent
from brokerdesk.policy import default_policy
from brokerdesk.tasks import load_split
from brokerdesk.tools import openai_tools
from training.grpo import grpo_loss
from training.masking import assistant_mask
from training.rollouts import collect_batch

_MIX = ("need_to_know", "never_ask", "always_ask")


def main(argv=None) -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--model", default="Qwen/Qwen3-0.6B")
    p.add_argument("--groups", type=int, default=4)
    p.add_argument("--group-size", type=int, default=6)
    p.add_argument("--lora-rank", type=int, default=16)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--episodes-per-run", type=int, default=7200,
                   help="for the extrapolation: 300 tasks x group of 8 x 3 epochs")
    args = p.parse_args(argv)

    import torch
    from peft import LoraConfig, get_peft_model
    from transformers import AutoModelForCausalLM, AutoTokenizer

    assert torch.cuda.is_available(), "no GPU visible"
    print("GPU:", torch.cuda.get_device_name(0))
    tok = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(args.model, torch_dtype=torch.bfloat16).cuda()
    model = get_peft_model(model, LoraConfig(
        r=args.lora_rank, lora_alpha=2 * args.lora_rank, lora_dropout=0.0, task_type="CAUSAL_LM",
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"]))
    model.print_trainable_parameters()
    params = [q for q in model.parameters() if q.requires_grad]
    opt = torch.optim.AdamW(params, lr=args.lr)

    policy = default_policy()
    tasks = [t for t in load_split("train", policy) if t.underspecified][: args.groups]
    batch = collect_batch(tasks, lambda i: HeuristicAgent(_MIX[i % 3], policy), args.group_size,
                          system_prompt(policy), policy=policy)
    batch = [t for t in batch if t.advantage != 0.0]
    print(f"{len(batch)} dialogues with non-zero advantage from {args.groups} groups")

    def step() -> tuple[float, int]:
        """One optimizer step over the whole batch, one dialogue at a time (gradient accumulation)."""
        opt.zero_grad(set_to_none=True)
        total, tokens = 0.0, 0
        for traj in batch:
            ids, mask = assistant_mask(tok, traj.messages, tools=openai_tools(), enable_thinking=False)
            input_ids = torch.tensor([ids], device="cuda")
            loss_mask = torch.tensor([mask], device="cuda")
            adv = torch.tensor([traj.advantage], device="cuda")
            logits = model(input_ids=input_ids).logits
            loss = grpo_loss(logits, input_ids, loss_mask, adv) / len(batch)
            loss.backward()
            total += loss.item()
            tokens += len(ids)
        torch.nn.utils.clip_grad_norm_(params, 1.0)
        opt.step()
        return total, tokens

    torch.cuda.reset_peak_memory_stats()
    losses = []
    for k in range(4):
        torch.cuda.synchronize()
        t0 = time.time()
        loss, tokens = step()
        torch.cuda.synchronize()
        dt = time.time() - t0
        losses.append(loss)
        print(f"step {k}: loss {loss:+.4f}  {tokens} tokens  {dt:.2f}s  {tokens / dt:,.0f} tok/s")
    assert all(l == l and abs(l) < 1e4 for l in losses), "loss is NaN or exploding"
    # Advantage-weighted likelihood ascent on a fixed batch: the surrogate loss should fall.
    print(f"loss moved {losses[0]:+.4f} -> {losses[-1]:+.4f} "
          f"({'decreasing, as expected' if losses[-1] < losses[0] else 'NOT decreasing: investigate'})")
    peak = torch.cuda.max_memory_allocated() / 2**30
    rate = tokens / dt
    per_episode = tokens / len(batch)
    hours = args.episodes_per_run * per_episode / rate / 3600
    print(f"\npeak GPU memory {peak:.1f} GB at batch size 1, {per_episode:,.0f} tokens per dialogue")
    print(f"training half of one run ({args.episodes_per_run} episodes): about {hours:.1f} GPU-hours "
          f"at this rate; rollout generation comes on top")


if __name__ == "__main__":
    main()
