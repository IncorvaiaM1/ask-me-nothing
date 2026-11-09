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
| Nov 9 | Minimal BrokerDesk running end to end (internal) |
| Nov 19 | Milestone (4 pages, running baseline, one real result) |
| Nov 21 | BrokerDesk v1.0 frozen (internal) |
| Dec 2 | Poster PDF |
| Dec 3 | Poster session |
| Dec 8 | Final paper and code (cannot move) |

## Regulatory anchors

- FINRA Rule 2111 motivates investment-profile checks where applicable; section .08 excludes recommendations subject to Regulation Best Interest.
- SEC Regulation S-P (2024 amendments) motivates safeguarding customer information and incident response. The synthetic desk policy does not establish real-world compliance.

## Implementation and experiment scope

- Implemented: interfaces/data classes, sensitivity tiers, deterministic canary matching, three privacy unit tests, and GPU availability script. The environment, task generation, policy checker, client replies, schemas, training, and evaluation remain stubs or plans.
- Core tasks: cash-raising and rebalancing. Hedging and tax-aware trimming are extensions.
- Data: approximately 300 training, 50 validation, and 100 test tasks, separating clients, seeds, and goal/portfolio combinations. No tuning on test tasks.
- Structured field requests support deterministic scoring. Mandatory means required and still missing; information already obtained is redundant. Repeated unnecessary requests incur repeated costs.
- Train SFT+GRPO from the same distilled checkpoint as the SFT baseline. Score complete dialogues; mask user/tool tokens from training updates.
- Core ablations: privacy cost off and question cost off. Thinking on/off and tool removal are optional diagnostics, not required extra training runs.
- Three seeds for each of three core configurations means nine runs: an unmeasured 45-90 GPU-hour estimate at 5-10 hours per run, plus SFT/evaluation. Profile 0.6B by Nov 15 and choose/pin the RL stack then.
- Five test rollouts per task support pass^1/pass^3; use paired task-bootstrap intervals and report training-seed variability separately. See `../eval/README.md` for metric definitions.
- Existing `find_leaks` only matches markers and exempts the allowed client's low-tier fields. The environment must route only outbound messages through the check and implement the desk's recipient/field authorization policy. It cannot detect paraphrases.
- API cap, exact model versions, queue availability, and dependency lockfile are still pending. Set the cap before API generation and log actual spend.
