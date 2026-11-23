#!/bin/bash
# BrokerDesk v1: baselines, warm start and an RL pilot. Run INSIDE an interactive GPU
# session (inside tmux), from the repo root, after `git pull`:
#
#     bash training/pace/v1_pipeline.sh            # everything, in order
#     bash training/pace/v1_pipeline.sh 4          # start from stage 4
#
# Stages:
#   1  tests; scripted agents on every split, clean and with both attack sets (CPU)
#   2  prompted Qwen3 (MODEL) on validation, clean and injected, thinking off
#   3  the same with thinking on
#   4  supervised warm start on the over-asking teacher (uses size_orders)
#   5  the warm-started model on validation, clean and injected
#   6  RL pilot: 10 GRPO steps of the tiered arm, to check learning signal and timing
# Logs: runs/v1/; results: results/; adapters: checkpoints/, runs/grpo/.
set -euo pipefail
START="${1:-1}"
MODEL="${MODEL:-Qwen/Qwen3-1.7B}"
TAG="$(basename "$MODEL")"
ENV_DIR="${ENV_DIR:-$HOME/scratch/envs/amn}"
export HF_HOME="${HF_HOME:-$HOME/scratch/hf_cache}"
module load anaconda3 2>/dev/null || true
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "$ENV_DIR"
mkdir -p runs/v1
stage() { [ "$START" -le "$1" ]; }
SFT="checkpoints/sft-v1-$TAG"

if stage 1; then
    pytest -q 2>&1 | tee runs/v1/1_tests.log
    for split in val test test_paraphrase test_unseen; do
        for set in dev heldout; do
            python -m eval.run_baseline --agents need_to_know never_ask always_ask gullible --split "$split" \
                --injection paired --attack-set "$set" --out "results/scripted-$set" 2>&1 | tail -20
        done
    done | tee runs/v1/1_scripted.log
fi
if stage 2; then
    python -m eval.eval_adapter --model "$MODEL" --label "$TAG-prompted" --split val --injection paired \
        2>&1 | tee runs/v1/2_prompted.log
fi
if stage 3; then
    python -m eval.eval_adapter --model "$MODEL" --label "$TAG-prompted-thinking" --split val --thinking \
        2>&1 | tee runs/v1/3_thinking.log
fi
if stage 4; then
    python -m training.sft --model "$MODEL" --teacher always_ask --out "$SFT" 2>&1 | tee runs/v1/4_sft.log
fi
if stage 5; then
    python -m eval.eval_adapter --model "$MODEL" --adapter "$SFT" --label "$TAG-sft" --split val \
        --injection paired 2>&1 | tee runs/v1/5_sft_eval.log
fi
if stage 6; then
    python -m training.grpo_train --model "$MODEL" --init-adapter "$SFT" --arm tiered --seed 0 --steps 10 \
        --eval-every 5 --out "runs/grpo/pilot-$TAG" --resume 2>&1 | tee runs/v1/6_pilot.log
fi
echo "done; logs in runs/v1/"
