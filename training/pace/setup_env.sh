#!/bin/bash
# One-time environment setup on PACE-ICE. Run from the repo root on a login node or
# inside your interactive session:   bash training/pace/setup_env.sh
# Creates a separate env so nothing in your existing envs changes.
set -euo pipefail
ENV_DIR="${ENV_DIR:-$HOME/scratch/envs/amn}"

module load anaconda3 2>/dev/null || true
if [ ! -d "$ENV_DIR" ]; then
    conda create -y -p "$ENV_DIR" python=3.11
fi
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "$ENV_DIR"

pip install --upgrade pip
pip install -e ".[train,api,dev]"

python - <<'PY'
import torch, transformers, peft, vllm
print("torch", torch.__version__, "| transformers", transformers.__version__,
      "| peft", peft.__version__, "| vllm", vllm.__version__)
PY
pip freeze > training/pace/requirements.lock.txt
echo "env ready at $ENV_DIR; versions written to training/pace/requirements.lock.txt (commit it)"
