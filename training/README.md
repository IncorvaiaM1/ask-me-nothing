# training

Multi-turn GRPO with LoRA on Qwen3-1.7B (Qwen3-0.6B for smoke tests, Qwen3-4B for the size check). The full run order is in `docs/RUNBOOK.md`.

## Pieces

- `sft.py`: supervised warm start on the scripted over-asking teacher (`always_ask`), successful and finished dialogues only, cross-entropy on assistant tokens.
- `grpo_train.py`: the RL loop. One process holds an in-process vLLM engine for rollouts and the Hugging Face model being trained; after each update the adapter is saved and vLLM switches to it. Resumable with `--resume`.
- `batch_rollout.py`: every active dialogue's next turn in one `generate` call; the model's reply is kept verbatim as the assistant message.
- `config.py`: reward arms. `tiered` (the method), `flat` (uniform cost matched to tiered on the teacher's collection), `shuffled` (tier costs on the wrong fields; the control showing the model learns which data is sensitive), `no_collection_cost`; `sample_steerable` for the cost-conditioned variant.
- `masking.py`: token ids, assistant-only loss mask and per-token turn index, built turn by turn so each reply follows the prompt the model actually saw (Qwen3's template drops the empty think block from past turns). Verified on the real tokenizer on Nov 17.
- `grpo.py`: on-policy GRPO loss with per-dialogue or per-token advantages. Verified on PACE on Nov 17.
- `rollouts.py`, `collect.py`: the earlier HTTP-based rollout harness, kept for dry runs.

## Conventions

- Seeds 0, 1 and 2 for every reported run; same SFT initialization for every arm.
- Record GPU type and hours per run (the paper needs a compute statement). Nov 17: one H100 80GB; LoRA step about 13,450 tokens/s on 1.7B, 18 GB peak at batch size 1.
- Qwen3 is Apache 2.0; check the base model's license before releasing adapters.

`pace/` holds the scripts that run all of this inside an interactive GPU session.
