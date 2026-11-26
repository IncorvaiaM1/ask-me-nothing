# Run plan to December 8 (BrokerDesk v1)

Everything below runs on PACE-ICE inside an interactive H100 session, started inside
`tmux` so a dropped connection does not kill it. Each script is resumable; rerun it with
the stage number or `--resume`. Push `results/` after each block so the paper tables
(`python -m eval.milestone_tables`, or the final-table script) can be rebuilt anywhere.

Time estimates come from the November 17 measurements (LoRA step about 13,000 tokens/s
on Qwen3-1.7B; dialogues about 2,600 tokens) and are not yet measured for v1. The pilot
in block 1 replaces them with real numbers.

## Block 1: v1 baselines, warm start, pilot (about 2 hours)

    git pull
    bash training/pace/setup_env.sh          # only if the env changed; it pins versions
    bash training/pace/v1_pipeline.sh

Read before going on:
- `runs/v1/5_sft_eval.log`: the warm-started model's success. The point of `size_orders`
  is that this should now be well above the old 8 percent. If it is not, stop and send the
  log; RL on a model that cannot trade learns nothing useful.
- `runs/grpo/pilot-Qwen3-1.7B/log.jsonl`: `groups_with_signal` should be most of the 16
  groups, `seconds_total` per step sets the budget below, and unneeded questions should
  start to fall by step 10.

## Block 2: the central comparison (9 runs; about 1 hour each at 60 steps)

v1.1 (Nov 26): GRPO now has a KL penalty to the SFT adapter (`--kl-beta 0.04`, default),
validates every 5 steps, and evaluates `best/` (highest validation reward) instead of
`final/`. The first tiered seed-0 run, without KL, collapsed; see
`results/v1.0_nokl_collapse/README.md`. Before rerunning, move its directory aside or
`--resume` will pick it up at step 60:

    mv runs/grpo/tiered-Qwen3-1.7B-s0 runs/grpo/v1.0-nokl-tiered-Qwen3-1.7B-s0

Watch `kl` in the training lines and `required_recall` in the eval lines of `log.jsonl`;
recall should stay near 1.0. If it falls below 0.9 at any eval, stop and send the log.

    for seed in 0 1 2; do
      for arm in tiered flat shuffled; do
        bash training/pace/grpo_run.sh $arm $seed
      done
    done

Each run trains, then evaluates on the test split with five sampled rollouts per task,
clean and with held-out attacks. Run several in parallel if PACE gives you more than one
GPU; they are independent.

## Block 3: ablations and extensions (about 6 runs)

    bash training/pace/grpo_run.sh no_collection_cost 0
    bash training/pace/grpo_run.sh turn 0                 # turn-level credit assignment
    bash training/pace/grpo_run.sh steerable 0            # costs in the prompt
    bash training/pace/steer_eval.sh runs/grpo/steerable-Qwen3-1.7B-s0/best steer-s0
    MODEL=Qwen/Qwen3-4B bash training/pace/v1_pipeline.sh 4          # 4B warm start + pilot
    MODEL=Qwen/Qwen3-4B bash training/pace/grpo_run.sh tiered 0

## Block 4: transfer and robustness (per model condition)

For the prompted base model, the SFT model, and the seed-0 tiered, flat and shuffled models:

    bash training/pace/final_eval.sh qwen-prompted
    bash training/pace/final_eval.sh sft checkpoints/sft-v1-Qwen3-1.7B
    bash training/pace/final_eval.sh tiered-s0 runs/grpo/tiered-Qwen3-1.7B-s0/best
    ...

This covers held-out wording, held-out attacks, the unseen wire family and four LLM
client personas with the drift audit.

    bash training/pace/tau2_eval.sh base
    bash training/pace/tau2_eval.sh tiered-s0 runs/grpo/tiered-Qwen3-1.7B-s0/best

τ²-bench retail and airline test splits plus intake (τ-Entity, scaffolded text), with a
local Qwen3-8B user simulator.

## Order of priority if time runs short

1. Block 1 and seed 0 of block 2: the minimum for the poster on December 2.
2. Seeds 1 and 2 of block 2: needed for intervals in the paper.
3. Block 4 for seed-0 models; then block 3; then τ²-bench.

## Decisions already taken

- BrokerDesk v1 changes the tasks, so milestone numbers are archived in
  `results/milestone_v0/` and every v1 number is rerun.
- Validation is 60 tasks and test 120 (20 and 40 per family), up from 50 and 100, so each
  of three families has the same count.
- The budget is 14 tool calls, up from 12: with `size_orders` the over-asking teacher
  needs up to 13 calls on rebalances.
