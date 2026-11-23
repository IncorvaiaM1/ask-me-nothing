#!/bin/bash
# Transfer and robustness for one model condition. Inside a GPU session, repo root:
#
#     bash training/pace/final_eval.sh LABEL [ADAPTER_DIR]
#
# Runs, for the base model or an adapter:
#   test_paraphrase (held-out wording), clean and with held-out attacks
#   test_unseen (the wire family, never trained on)
#   LLM-simulated clients (Qwen3-8B served on port 8001), all four personas, test split
# Every condition writes results/<LABEL>...; the client audit is in each summary.
set -euo pipefail
LABEL="${1:?label}"; ADAPTER="${2:-}"
MODEL="${MODEL:-Qwen/Qwen3-1.7B}"
CLIENT_MODEL="${CLIENT_MODEL:-Qwen/Qwen3-8B}"
ENV_DIR="${ENV_DIR:-$HOME/scratch/envs/amn}"
export HF_HOME="${HF_HOME:-$HOME/scratch/hf_cache}"
module load anaconda3 2>/dev/null || true
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "$ENV_DIR"
mkdir -p runs/final
AD=(); [ -n "$ADAPTER" ] && AD=(--adapter "$ADAPTER")

python -m eval.eval_adapter --model "$MODEL" "${AD[@]}" --label "$LABEL" --split test_paraphrase \
    --injection paired --attack-set heldout 2>&1 | tee "runs/final/$LABEL-paraphrase.log"
python -m eval.eval_adapter --model "$MODEL" "${AD[@]}" --label "$LABEL" --split test_unseen \
    2>&1 | tee "runs/final/$LABEL-unseen.log"

vllm serve "$CLIENT_MODEL" --port 8001 --max-model-len 16384 --gpu-memory-utilization 0.35 \
    > "runs/final/vllm-client.log" 2>&1 &
CLIENT=$!
trap 'kill $CLIENT 2>/dev/null || true' EXIT
until curl -sf http://localhost:8001/v1/models > /dev/null; do
    kill -0 $CLIENT 2>/dev/null || { echo "client server exited; see runs/final/vllm-client.log"; exit 1; }
    sleep 5
done
python -m eval.eval_adapter --model "$MODEL" "${AD[@]}" --label "$LABEL" --split test --client llm \
    --persona all --client-model "$CLIENT_MODEL" --gpu-mem 0.45 2>&1 | tee "runs/final/$LABEL-llmclient.log"
