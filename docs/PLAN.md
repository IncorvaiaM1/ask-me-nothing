# Project plan

The living version of this plan is the team doc **"Need to Know: ConvAI Final Project Plan"** (shared separately). This file is a snapshot of the parts the code depends on; if they disagree, the doc wins and this file gets updated.

## Thesis

In financial dialogue every clarifying question is also a request for client data. A small model trained with multi-turn RL, where questions carry a time cost and sensitive data a privacy cost, learns a need-to-know policy: it looks up or derives what it can, asks only what the rules require or the client alone holds, never leaks what it was told, and still completes the task as often as prompted or distilled baselines.

## Research questions

| RQ | Question | Refuted if |
| --- | --- | --- |
| RQ1 | Can multi-turn GRPO teach a 1.5 to 3B model when to ask, look up, think, or act? | No gain over SFT distillation, or gains only with more policy violations |
| RQ2 | Does pricing sensitive data teach data minimization? | Over-collection falls only because mandatory items are skipped or more tasks fail |
| RQ3 | What did training change? | Gains come from one shortcut, such as never asking |
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

- FINRA Rule 2111 (suitability): reasonable diligence to ascertain the customer's investment profile.
- SEC Regulation S-P (2024 amendments): safeguarding customer information, incident response, 30-day notice.
