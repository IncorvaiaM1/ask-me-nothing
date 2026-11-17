#!/bin/bash
# Milestone rows 2 and 3. Run INSIDE an interactive GPU session, from the repo root:
#
#     bash training/pace/warm_start.sh            # everything, in order
#     bash training/pace/warm_start.sh 2          # start from stage 2
#
# Stages:
#   1  prompted Qwen3-1.7B with thinking ON (is the failure a reasoning problem?)
#   2  supervised warm start on the over-asking scripted teacher
#   3  the warm-started model on validation, clean and injected
# Logs go to runs/warm_start/, results to results/, the adapter to checkpoints/.
set -euo pipefail
START="${1:-1}"
MODEL="${MODEL:-Qwen/Qwen3-1.7B}"
TEACHER="${TEACHER:-always_ask}"
ADAPTER_DIR="checkpoints/sft-$TEACHER"
ENV_DIR="${ENV_DIR:-$HOME/scratch/envs/amn}"
export HF_HOME="${HF_HOME:-$HOME/scratch/hf_cache}"

module load anaconda3 2>/dev/null || true
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "$ENV_DIR"
mkdir -p runs/warm_start
stage() { [ "$START" -le "$1" ]; }

if stage 1; then
    echo "=== stage 1: prompted $MODEL, thinking on ==="
    SECONDS=0
    THINKING=1 bash training/pace/prompted_baseline.sh "$MODEL" val 2>&1 | tee runs/warm_start/1_thinking.log
    echo "stage 1 took ${SECONDS}s" | tee -a runs/warm_start/1_thinking.log
fi

if stage 2; then
    echo "=== stage 2: supervised warm start, teacher=$TEACHER ==="
    python -m training.sft --model "$MODEL" --teacher "$TEACHER" --out "$ADAPTER_DIR" \
        2>&1 | tee runs/warm_start/2_sft.log
fi

if stage 3; then
    echo "=== stage 3: warm-started model on validation ==="
    ADAPTER="$PWD/$ADAPTER_DIR" LABEL="$(basename "$MODEL")-sft-$TEACHER" \
        bash training/pace/prompted_baseline.sh "$MODEL" val 2>&1 | tee runs/warm_start/3_sft_eval.log
fi
echo "done; logs in runs/warm_start/, results in results/"
