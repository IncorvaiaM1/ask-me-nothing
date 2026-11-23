#!/bin/bash
# External transfer on τ²-bench (sierra-research/tau2-bench). Run INSIDE a GPU session,
# from the ask-me-nothing repo root, after training/pace/setup_env.sh.
#
#     bash training/pace/tau2_eval.sh LABEL [ADAPTER_DIR]
#
# Serves the agent (Qwen3-1.7B, optionally with a LoRA adapter) and a local user
# simulator (Qwen3-8B) with vLLM, then runs three text evaluations:
#   retail test split (40 tasks) and airline test split (20): general tool-agent-user skill
#   intake base split (200, τ-Entity, scaffolded text): exact capture of a missing value
# No API keys are used; both models are local. Results are not comparable to the public
# leaderboard, which uses a frontier user simulator; compare rows to each other.
#
# τ²-bench needs Python 3.12+, so it gets its own env at $TAU2_ENV. This script creates it
# on first use.
set -euo pipefail
LABEL="${1:?label}"
ADAPTER="${2:-}"
MODEL="${MODEL:-Qwen/Qwen3-1.7B}"
USER_MODEL="${USER_MODEL:-Qwen/Qwen3-8B}"
AMN_ENV="${ENV_DIR:-$HOME/scratch/envs/amn}"
TAU2_ENV="${TAU2_ENV:-$HOME/scratch/envs/tau2}"
TAU2_DIR="${TAU2_DIR:-$HOME/scratch/tau2-bench}"
TRIALS="${TRIALS:-1}"
export HF_HOME="${HF_HOME:-$HOME/scratch/hf_cache}"

module load anaconda3 2>/dev/null || true
source "$(conda info --base)/etc/profile.d/conda.sh"
if [ ! -d "$TAU2_ENV" ]; then
    conda create -y -p "$TAU2_ENV" python=3.12
    conda activate "$TAU2_ENV"
    [ -d "$TAU2_DIR" ] || git clone https://github.com/sierra-research/tau2-bench.git "$TAU2_DIR"
    pip install -e "$TAU2_DIR"
    (cd "$TAU2_DIR" && git rev-parse HEAD) > training/pace/tau2_commit.txt
fi
mkdir -p runs/tau2

conda activate "$AMN_ENV"
SERVED="$MODEL"; LORA=()
if [ -n "$ADAPTER" ]; then SERVED="agent"; LORA=(--enable-lora --lora-modules "agent=$ADAPTER" --max-lora-rank 64); fi
vllm serve "$MODEL" --port 8000 --enable-auto-tool-choice --tool-call-parser hermes --max-model-len 32768 \
    --gpu-memory-utilization 0.35 "${LORA[@]}" > "runs/tau2/vllm-agent-$LABEL.log" 2>&1 &
AGENT=$!
vllm serve "$USER_MODEL" --port 8001 --max-model-len 32768 --gpu-memory-utilization 0.45 \
    > "runs/tau2/vllm-user-$LABEL.log" 2>&1 &
USERSIM=$!
trap 'kill $AGENT $USERSIM 2>/dev/null || true' EXIT
for port in 8000 8001; do
    until curl -sf "http://localhost:$port/v1/models" > /dev/null; do
        kill -0 $AGENT 2>/dev/null && kill -0 $USERSIM 2>/dev/null || { echo "a vLLM server exited; see runs/tau2/"; exit 1; }
        sleep 5
    done
done

conda activate "$TAU2_ENV"
NOTHINK='"extra_body": {"chat_template_kwargs": {"enable_thinking": false}}'
AGENT_ARGS="{\"temperature\": 0.0, \"api_base\": \"http://localhost:8000/v1\", $NOTHINK}"
USER_ARGS="{\"temperature\": 0.0, \"api_base\": \"http://localhost:8001/v1\", $NOTHINK}"
cd "$TAU2_DIR"
run() {   # domain split extra...
    local domain=$1 split=$2; shift 2
    tau2 run --domain "$domain" --task-split-name "$split" --num-trials "$TRIALS" --max-concurrency 8 \
        --agent-llm "hosted_vllm/$SERVED" --agent-llm-args "$AGENT_ARGS" \
        --user-llm "hosted_vllm/$USER_MODEL" --user-llm-args "$USER_ARGS" \
        --save-to "amn-$LABEL-$domain" --auto-resume "$@"
}
# Smoke test first: two retail tasks. Read one agent turn in the output to check that
# no <think> block appears; if it does, the no-thinking argument did not reach vLLM.
run retail test --num-tasks 2 --save-to "amn-$LABEL-smoke"
run retail test
run airline test
run intake base --elicitation-strategy scaffolded
cd - > /dev/null

conda activate "$AMN_ENV"
python -m eval.tau2_summary "$TAU2_DIR"/data/simulations/amn-$LABEL-{retail,airline,intake}/results.json \
    | tee "runs/tau2/summary-$LABEL.md"
