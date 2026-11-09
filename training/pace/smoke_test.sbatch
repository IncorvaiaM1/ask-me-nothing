#!/bin/bash
# GPU smoke test on PACE-ICE. Edit the partition/GPU line to match what `pace-check-queue`
# or the ICE docs list for our course allocation, then: sbatch training/pace/smoke_test.sbatch
#SBATCH -J amn-smoke
#SBATCH -N 1
#SBATCH --gres=gpu:1
#SBATCH -t 00:30:00
#SBATCH -o slurm-%j.out

module load anaconda3 2>/dev/null || true
cd "$SLURM_SUBMIT_DIR"

nvidia-smi
python -c "import torch; print('torch', torch.__version__, 'cuda', torch.cuda.is_available())"
# TODO: replace with a Qwen3-0.6B GRPO step on fixed tasks once the RL library is chosen.
