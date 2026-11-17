# training

Planned multi-turn GRPO with LoRA on Qwen3-1.7B, with Qwen3-0.6B for smoke tests. Owner: RL workstream. No trainer is implemented yet.

- **Library:** OpenPipe ART, verifiers, or verl; choose during the pilot by Nov 15 after checking multi-turn rollout and assistant-token masking support on PACE-ICE.
- **Model:** check the base model's license before releasing adapters (Qwen3 is Apache 2.0; some Qwen2.5 sizes are research-only).
- **Curriculum:** fully specified tasks first, then underspecified.
- **Seeds:** 0, 1, 2 for every reported run.
- **Compute log:** record GPU type and hours per run here; the paper needs a compute statement.
- **Comparison:** initialize SFT+GRPO from the same distilled checkpoint used by the SFT baseline. Group complete dialogues for the same task and update assistant tokens only.
- **Core budget:** main method plus privacy-cost-off and question-cost-off ablations, each over three seeds. At an unmeasured 5-10 GPU-hours/run this is 45-90 GPU-hours, excluding SFT and evaluation. Profile first; shrink to 0.6B if needed. Thinking/tool-removal diagnostics are optional.

## What exists

- `rollouts.py`, `collect.py`: grouped rollouts (same task, same price path), one scalar reward per dialogue, group-normalized advantages, written as JSONL of chat messages. `--agent mixed` is a no-model dry run.
- `masking.py`: token ids plus a loss mask covering only assistant tokens, derived from the tokenizer's own chat template. Tested on a stand-in tokenizer; run `describe_mask` on the real Qwen3 tokenizer in the pilot.
- `grpo.py`: reference on-policy loss. Never executed; it states what the chosen library must compute.
- `config.py`: reward weights for the main method and the two ablations.

Whichever library is chosen only has to consume the JSONL (messages and advantage) or call `collect_group` as its rollout function.

`pace/prompted_baseline.sh` runs the prompted baseline inside an interactive GPU session. `pace/smoke_test.sbatch` checks that a GPU job runs and torch sees CUDA.
