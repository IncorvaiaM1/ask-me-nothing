#!/bin/bash
# Steerability: evaluate a cost-conditioned adapter at cost settings it saw in training
# and at settings outside its training range. Inside a GPU session, repo root:
#
#     bash training/pace/steer_eval.sh runs/grpo/steerable-Qwen3-1.7B-s0/best steer-s0
set -euo pipefail
ADAPTER="${1:?adapter}"; LABEL="${2:?label}"
MODEL="${MODEL:-Qwen/Qwen3-1.7B}"
ENV_DIR="${ENV_DIR:-$HOME/scratch/envs/amn}"
export HF_HOME="${HF_HOME:-$HOME/scratch/hf_cache}"
module load anaconda3 2>/dev/null || true
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "$ENV_DIR"
mkdir -p runs/steer
# training draws low in [0.001, 0.05], sensitive up to 20x low, restricted in [0.1, 1]
for costs in 0.001,0.002,0.2 0.01,0.05,0.5 0.03,0.3,1.0 0.0,0.0,0.0 0.2,1.0,2.0; do
    python -m eval.eval_adapter --model "$MODEL" --adapter "$ADAPTER" --label "$LABEL-c${costs//,/_}" \
        --split val --steer-costs "$costs" 2>&1 | tail -6 | tee -a "runs/steer/$LABEL.log"
done
