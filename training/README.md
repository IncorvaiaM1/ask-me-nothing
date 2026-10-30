# training

Multi-turn GRPO with LoRA on a 1.5 to 3B model. Owner: RL workstream.

- **Library:** OpenPipe ART, verifiers, or verl; choose by Nov 1 after a smoke test on PACE-ICE.
- **Model:** check the base model's license before releasing adapters (Qwen3 is Apache 2.0; some Qwen2.5 sizes are research-only).
- **Curriculum:** fully specified tasks first, then underspecified.
- **Seeds:** 0, 1, 2 for every reported run.
- **Compute log:** record GPU type and hours per run here; the paper needs a compute statement.

`pace/smoke_test.sbatch` checks that a GPU job runs and torch sees CUDA.
