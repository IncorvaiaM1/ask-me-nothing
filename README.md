# ask-me-nothing

**Learning need-to-know clarification: multi-turn RL in a simulated brokerage desk.**

Final project for ECE 6256 / CS 6604 Conversational AI (Georgia Tech, Fall 2026, Prof. Larry Heck).
Team: Greyson, Michael Incorvaia, Heeba, Canting.

## The idea

In financial dialogue every clarifying question can request client data. We plan to train Qwen3-1.7B with multi-turn GRPO and LoRA (0.6B for smoke tests), pricing every question and unnecessary field request by sensitivity. The goal is to reduce redundant questions and unauthorized disclosure while retaining necessary clarification and task success. Thinking allocation is exploratory. See the [proposal](docs/proposal/proposal.pdf) for the submission plan.

Current status (BrokerDesk v1, November 23): three task families (cash raising, rebalancing, tax-aware cash raising) plus a held-out wire-transfer family for transfer tests; a `size_orders` tool that turns goal parameters into compliant orders, so success depends on gathering the right information rather than share arithmetic; held-out request wording; twelve attacks in untrusted tool outputs (three development, nine held out); raw-value leak matching besides canaries; a secure-success metric; an LLM-simulated client with four personas and a reply audit; a GRPO trainer with tiered, flat, shuffled-tier and cost-conditioned reward arms and an optional turn-level advantage; one evaluation script for every model condition; and a τ²-bench runner. 52 tests pass, including real gradient steps on a tiny model. Milestone-era results are archived in `results/milestone_v0/`; v1 GPU runs follow `docs/RUNBOOK.md`. Canary and raw-value matching detect verbatim disclosure only; they are not a general privacy guarantee.

Research questions (details in [`docs/PLAN.md`](docs/PLAN.md)):

1. **RQ1.** Can multi-turn GRPO improve policy-compliant task success over prompting and SFT distillation?
2. **RQ2.** Does pricing sensitive data teach data minimization without hurting success?
3. **RQ3.** What did training change: fewer redundant questions, kept mandatory ones, better thinking allocation?
4. **RQ4.** Does it transfer from scripted to LLM-simulated clients and resist injected data requests?

## Repository layout

```
brokerdesk/   BrokerDesk simulator: tools, policy checker, clients, tasks, privacy tiers and canary leak checks
training/     GRPO configs and PACE-ICE job scripts
baselines/    prompted, SFT-distilled, frontier-model, and AirGap-style minimizer baselines
eval/         one script per table or figure in the paper
prompts/      every prompt we use, verbatim
tests/        unit tests for the simulator
docs/         plan, meeting brief, proposal and paper drafts
```

| Workstream | Folder | Owner (proposed) |
| --- | --- | --- |
| Environment and policy | `brokerdesk/env.py`, `tools.py`, `policy.py` | Greyson |
| Clients, tasks, privacy | `brokerdesk/clients.py`, `tasks.py`, `privacy.py` | Heeba |
| RL training on PACE-ICE | `training/` | Michael |
| Baselines, eval, paper | `baselines/`, `eval/`, `docs/` | Canting |

## Setup

Requires Python 3.10+. Use [uv](https://docs.astral.sh/uv/) or plain `pip`. Dependency ranges are currently unpinned; a tested lockfile is still pending.

```bash
git clone https://github.com/IncorvaiaM1/ask-me-nothing.git
cd ask-me-nothing

# simulator + tests only (laptop)
uv venv && uv pip install -e ".[dev]"
pytest -q

# add training dependencies (PACE-ICE GPU node)
uv pip install -e ".[train]"

# add API baselines (frontier and LLM-simulated clients)
uv pip install -e ".[api]"
cp .env.example .env   # then fill in your own keys; .env is gitignored
```

Without uv: `python -m venv .venv`, activate it, then `pip install -e ".[dev]"`.

Choose the RL library during the 0.6B pilot by Nov 15, then pin the tested environment with `uv lock` and commit `uv.lock`. The training extra alone does not provide a working GRPO trainer.

## Reproducing results

To be filled in as results land. The rule: **one command per table or figure**, each with a fixed seed, listed here.

| Paper item | Command |
| --- | --- |
| Scripted reference agents, any split | `python -m eval.run_baseline --agents need_to_know never_ask always_ask gullible --split val --injection paired` |
| Any model or adapter, any split or condition | `python -m eval.eval_adapter --help` |
| Full GPU run plan | `docs/RUNBOOK.md` |
| Same agents, paired clean and injected tool outputs | `python -m eval.run_baseline --agents need_to_know never_ask always_ask gullible --split val --seed 0 --injection paired` |
| Dry-run rollout batch, no model | `python -m training.collect --agent mixed --split train --n 8 --group-size 6 --out runs/rollouts/dry.jsonl` |
| Prompted Qwen3-1.7B, validation split (PACE-ICE GPU session) | `bash training/pace/prompted_baseline.sh Qwen/Qwen3-1.7B val` *(not run yet)* |
| Table 1 (main results) | `python -m eval.main_table --seeds 0 1 2` *(TBD)* |

Scripted reference agents on the 60 validation tasks of BrokerDesk v1, one rollout, seed 0 (no model; these check the simulator and bracket the behaviours under study):

| agent | success | secure | req. recall | goal | viol/ep | needed q | unneeded q | missed mand. | leak rate | reward | unneeded fields (low/sens/restr) |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| need_to_know | 1.00 | 1.00 | 1.00 | 1.00 | 0.00 | 0.68 | 0.00 | 0.00 | 0.00 | 0.986 | 0.00/0.00/0.00 |
| never_ask | 0.55 | 0.55 | 0.00 | 0.55 | 0.20 | 0.00 | 0.00 | 0.20 | 0.00 | 0.350 | 0.00/0.00/0.00 |
| always_ask | 1.00 | 1.00 | 1.00 | 1.00 | 0.00 | 0.68 | 3.02 | 0.00 | 0.00 | 0.827 | 1.30/1.72/0.00 |
| gullible | 1.00 | 1.00 | 1.00 | 1.00 | 0.00 | 0.68 | 0.00 | 0.00 | 0.00 | 0.986 | 0.00/0.00/0.00 |

`success` is goal met with zero policy violations; `secure` additionally requires no leak and no successful attack; `req. recall` is the share of client-only or mandatory fields the agent obtained. `never_ask` guesses instead of asking; `always_ask`, the supervised teacher, asks for every task field plus one to three unneeded profile fields drawn per task; `gullible` obeys instructions in tool outputs, which only matters in injected runs.

## Conventions

- Seeds are fixed and passed explicitly; no unseeded randomness in `brokerdesk/`.
- Every prompt lives in `prompts/` as a plain text file; code loads it from there, never inline.
- No real client data. Planned personas and values are synthetic. Canary markers measure verbatim disclosure in unauthorized outbound messages, not paraphrased leakage or incoming tool observations.
- No API keys or PACE credentials in git. Use `.env`.
- Work on a branch per workstream and open a PR into `main`.

## Data statement

All planned client profiles, portfolios, and tasks are synthetic; `brokerdesk/tasks.py` generates them from fixed seeds. Splits are 300 training, 50 validation, and 100 test tasks with disjoint clients and seeds; enforcing disjoint goal/portfolio combinations is still to do. No personal data or human subjects are planned. The simulator is a research tool and not investment advice. Record generator and model licenses when releasing data or adapters.

## AI use

Planning and code scaffolding used Claude assistance for brainstorming, literature search, drafting, and coding. OpenAI Codex assisted source verification, proposal editing, and repository documentation consistency. We disclose adopted model-suggested ideas in the proposal's Acknowledgements. Authors remain responsible for reading cited works, validating claims, and reviewing code; AI checks do not replace that review.

## License

MIT, see [LICENSE](LICENSE). Trained model weights, if released, follow the license of their base model.
