#!/bin/bash
# Prompted Qwen3 baseline on BrokerDesk. Run this INSIDE an interactive GPU session on
# PACE-ICE, from the repo root:   bash training/pace/prompted_baseline.sh [model] [split]
# Needs an env with vllm and openai installed:  pip install -e ".[train,api]"
set -euo pipefail
MODEL="${1:-Qwen/Qwen3-1.7B}"
SPLIT="${2:-val}"
PORT="${PORT:-8000}"

nvidia-smi
vllm serve "$MODEL" --enable-auto-tool-choice --tool-call-parser hermes \
    --port "$PORT" --max-model-len 16384 > "vllm-$PORT.log" 2>&1 &
SERVER=$!
trap 'kill $SERVER 2>/dev/null || true' EXIT

echo "waiting for vLLM (log: vllm-$PORT.log)"
until curl -sf "http://localhost:$PORT/v1/models" > /dev/null; do
    kill -0 $SERVER 2>/dev/null || { echo "vLLM exited; see vllm-$PORT.log"; exit 1; }
    sleep 5
done

# Thinking off first (fast); rerun without --no-thinking for the thinking-on diagnostic.
python -m eval.run_baseline --agents llm --model "$MODEL" --base-url "http://localhost:$PORT/v1" \
    --split "$SPLIT" --no-thinking --injection paired --label "$(basename "$MODEL")-prompted"
