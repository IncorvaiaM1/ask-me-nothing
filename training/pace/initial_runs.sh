#!/bin/bash
# Initial runs on PACE-ICE. Run INSIDE an interactive GPU session, from the repo root,
# after training/pace/setup_env.sh:
#
#     bash training/pace/initial_runs.sh            # everything, in order
#     bash training/pace/initial_runs.sh 3          # start from stage 3
#
# Stages:
#   1  simulator tests and scripted reference agents (no GPU)
#   2  assistant-token mask on the real Qwen3 tokenizer
#   3  one LoRA gradient step with timing, 0.6B then 1.7B
#   4  prompted Qwen3-1.7B baseline on validation, clean and injected (the milestone result)
# Everything is logged under runs/initial/.
set -euo pipefail
START="${1:-1}"
ENV_DIR="${ENV_DIR:-$HOME/scratch/envs/amn}"
export HF_HOME="${HF_HOME:-$HOME/scratch/hf_cache}"

module load anaconda3 2>/dev/null || true
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "$ENV_DIR"
mkdir -p runs/initial "$HF_HOME"
stage() { [ "$START" -le "$1" ]; }

if stage 1; then
    echo "=== stage 1: tests and scripted agents ==="
    pytest -q 2>&1 | tee runs/initial/1_tests.log
    python -m eval.run_baseline --agents need_to_know never_ask always_ask gullible \
        --split val --injection paired 2>&1 | tee runs/initial/1_scripted.log
fi

if stage 2; then
    echo "=== stage 2: mask on the real tokenizer ==="
    python -m training.check_mask --model Qwen/Qwen3-0.6B 2>&1 | tee runs/initial/2_mask.log
fi

if stage 3; then
    echo "=== stage 3: one gradient step, timed ==="
    nvidia-smi 2>&1 | tee runs/initial/3_gpu.log
    python -m training.check_loss --model Qwen/Qwen3-0.6B 2>&1 | tee runs/initial/3_loss_0.6B.log
    python -m training.check_loss --model Qwen/Qwen3-1.7B 2>&1 | tee runs/initial/3_loss_1.7B.log
fi

if stage 4; then
    echo "=== stage 4: prompted Qwen3-1.7B baseline, clean and injected ==="
    SECONDS=0
    bash training/pace/prompted_baseline.sh Qwen/Qwen3-1.7B val 2>&1 | tee runs/initial/4_baseline.log
    echo "stage 4 took ${SECONDS}s for 100 episodes" | tee -a runs/initial/4_baseline.log
fi
echo "done; logs in runs/initial/, results in results/"
