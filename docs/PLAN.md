# Project plan

This repository plan is aligned with `proposal/proposal.tex` as reviewed on November 8, 2026. Reconcile any differences with the separately shared team document before implementation; that document was not available during the repository review.

## Thesis

Test whether multi-turn RL with question costs and sensitivity-weighted unnecessary-request costs improves need-to-know clarification without sacrificing task success. Qwen3-1.7B with LoRA is the primary model; Qwen3-0.6B is the pilot/fallback. Reduced leakage is a hypothesis, not an established result or privacy guarantee.

## Research questions

| RQ | Question | Refuted if |
| --- | --- | --- |
| RQ1 | Can multi-turn GRPO improve policy-compliant success over prompting and SFT distillation? | No gain over SFT distillation, or gains only with more policy violations |
| RQ2 | Does pricing sensitive data teach data minimization? | Over-collection falls only because mandatory items are skipped or more tasks fail |
| RQ3 | Are redundant questions reduced while necessary questions are retained? Thinking allocation is exploratory. | Gains come from one shortcut, such as never asking |
| RQ4 | Does it hold up off the training distribution? | Gains vanish with LLM clients, or injected text extracts protected data |

## Deadlines

| Date | Deliverable |
| --- | --- |
| Nov 5 | Proposal (2 pages, ACL format) |
| Nov 9 | Minimal BrokerDesk running end to end (internal); landed Nov 16 |
| Nov 19 | Milestone (4 pages, running baseline, one real result) |
| Nov 21 | BrokerDesk v1.0 frozen (internal) |
| Dec 2 | Poster PDF |
| Dec 3 | Poster session |
| Dec 8 | Final paper and code (cannot move) |

## Regulatory anchors

- FINRA Rule 2111 motivates investment-profile checks where applicable; section .08 excludes recommendations subject to Regulation Best Interest.
- SEC Regulation S-P (2024 amendments) motivates safeguarding customer information and incident response. The synthetic desk policy does not establish real-world compliance.

## BrokerDesk v1 (November 23) and experiment scope

- Families: cash raising, rebalancing and tax-aware cash raising, 300/60/120 train/validation/test tasks plus `test_paraphrase` (held-out wording) and `test_unseen` (wire transfers, never trained on). Sixty percent of tasks need at least one question.
- The tax bracket is the key field: sensitive, needed in tax-aware tasks (stated, on file, or client-only), irrelevant everywhere else. Without it, every penalty would teach the same thing and the tiered-versus-flat comparison could show nothing.
- `size_orders` computes compliant orders from goal parameters. It removes the share arithmetic the fine-tuned model failed at, and it trusts its inputs, so guessed information still fails the task.
- Arms: tiered, flat (matched expected penalty), shuffled tiers, no collection cost; three seeds for the first three. Extensions: cost-conditioned steerability, turn-level credit assignment, Qwen3-4B.
- Metrics: success; secure success (also no leak, no successful attack); required-field recall; unneeded requests by field and by tier; pass^1 and pass^3 on five rollouts; task-bootstrap intervals.
- Robustness and transfer: held-out wording, nine held-out attacks, four LLM-client personas with a reply audit, the unseen wire family, τ²-bench retail, airline and intake with a local user simulator.
- Leak checks: canary markers plus raw values (digits normalized), so a policy cannot hide a leak by dropping the marker.
- Not built: AirGap-style minimizer, frontier-model reference (needs an API budget decision), margin rules.
