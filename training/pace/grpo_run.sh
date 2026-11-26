#!/bin/bash
# One RL run plus its evaluation. Inside a GPU session (tmux), from the repo root:
#
#     bash training/pace/grpo_run.sh ARM SEED [extra grpo_train args...]
#
#   ARM: tiered | flat | shuffled | no_collection_cost | steerable | turn
#     steerable = tiered arm with costs sampled per group and shown in the prompt
#     turn      = tiered arm with turn-level credit assignment
# Resumes automatically if the run directory already has steps. After training, the
# best-on-validation adapter is evaluated on validation (greedy) and on the test split with five
# sampled rollouts per task, clean and with held-out attacks.
set -euo pipefail
ARM="${1:?arm}"; SEED="${2:?seed}"; shift 2
MODEL="${MODEL:-Qwen/Qwen3-1.7B}"
TAG="$(basename "$MODEL")"
SFT="${SFT:-checkpoints/sft-v1-$TAG}"
ENV_DIR="${ENV_DIR:-$HOME/scratch/envs/amn}"
export HF_HOME="${HF_HOME:-$HOME/scratch/hf_cache}"
module load anaconda3 2>/dev/null || true
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "$ENV_DIR"

case "$ARM" in
    steerable) EXTRA=(--arm tiered --steerable) ;;
    turn)      EXTRA=(--arm tiered --advantage turn) ;;
    *)         EXTRA=(--arm "$ARM") ;;
esac
NAME="$ARM-$TAG-s$SEED"
OUT="runs/grpo/$NAME"
mkdir -p "$OUT"
python -m training.grpo_train --model "$MODEL" --init-adapter "$SFT" --seed "$SEED" --out "$OUT" --resume \
    "${EXTRA[@]}" "$@" 2>&1 | tee -a "$OUT/train.log"
case "$ARM" in flat|shuffled|no_collection_cost) EVAL_ARM=$ARM ;; *) EVAL_ARM=tiered ;; esac
python -m eval.eval_adapter --model "$MODEL" --adapter "$OUT/best" --label "$NAME" --split test \
    --rollouts 5 --temperature 0.7 --arm "$EVAL_ARM" --injection paired --attack-set heldout 2>&1 | tee "$OUT/eval_test.log"
