#!/bin/bash
# Prompted baseline on BrokerDesk, clean and injected. Run this INSIDE an interactive GPU
# session on PACE-ICE, from the repo root:
#
#     bash training/pace/prompted_baseline.sh [model] [split]
#
# Optional environment variables:
#     THINKING=1          leave Qwen3 thinking on (default: off)
#     ADAPTER=path        serve a LoRA adapter on top of the model and evaluate that
#     LABEL=name          row name in tables and result files
# Needs an env with vllm and openai installed:  pip install -e ".[train,api]"
set -euo pipefail
MODEL="${1:-Qwen/Qwen3-1.7B}"
SPLIT="${2:-val}"
PORT="${PORT:-8000}"
THINKING="${THINKING:-0}"
ADAPTER="${ADAPTER:-}"

SERVE_ARGS=()
SERVED="$MODEL"
DEFAULT_LABEL="$(basename "$MODEL")-prompted"
if [ -n "$ADAPTER" ]; then
    SERVE_ARGS+=(--enable-lora --lora-modules "adapter=$ADAPTER" --max-lora-rank 64)
    SERVED="adapter"
    DEFAULT_LABEL="$(basename "$MODEL")-$(basename "$ADAPTER")"
fi
RUN_ARGS=(--no-thinking)
if [ "$THINKING" = "1" ]; then
    RUN_ARGS=(--max-tokens 3072)
    DEFAULT_LABEL="$DEFAULT_LABEL-thinking"
fi
LABEL="${LABEL:-$DEFAULT_LABEL}"

vllm serve "$MODEL" --enable-auto-tool-choice --tool-call-parser hermes \
    --port "$PORT" --max-model-len 32768 "${SERVE_ARGS[@]}" > "vllm-$PORT.log" 2>&1 &
SERVER=$!
trap 'kill $SERVER 2>/dev/null || true; wait $SERVER 2>/dev/null || true' EXIT

echo "waiting for vLLM (log: vllm-$PORT.log)"
until curl -sf "http://localhost:$PORT/v1/models" > /dev/null; do
    kill -0 $SERVER 2>/dev/null || { echo "vLLM exited; see vllm-$PORT.log"; tail -20 "vllm-$PORT.log"; exit 1; }
    sleep 5
done

python -m eval.run_baseline --agents llm --model "$SERVED" --base-url "http://localhost:$PORT/v1" \
    --split "$SPLIT" --injection paired --label "$LABEL" "${RUN_ARGS[@]}"
