# training

Planned multi-turn GRPO with LoRA on Qwen3-1.7B, with Qwen3-0.6B for smoke tests. Owner: RL workstream. No trainer is implemented yet.

- **Library:** OpenPipe ART, verifiers, or verl; choose during the pilot by Nov 15 after checking multi-turn rollout and assistant-token masking support on PACE-ICE.
- **Model:** check the base model's license before releasing adapters (Qwen3 is Apache 2.0; some Qwen2.5 sizes are research-only).
- **Curriculum:** fully specified tasks first, then underspecified.
- **Seeds:** 0, 1, 2 for every reported run.
- **Compute log:** record GPU type and hours per run here; the paper needs a compute statement.
- **Comparison:** initialize SFT+GRPO from the same distilled checkpoint used by the SFT baseline. Group complete dialogues for the same task and update assistant tokens only.
- **Core budget:** main method plus privacy-cost-off and question-cost-off ablations, each over three seeds. At an unmeasured 5-10 GPU-hours/run this is 45-90 GPU-hours, excluding SFT and evaluation. Profile first; shrink to 0.6B if needed. Thinking/tool-removal diagnostics are optional.

`pace/smoke_test.sbatch` checks that a GPU job runs and torch sees CUDA.
