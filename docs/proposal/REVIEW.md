# Proposal review - November 8, 2026

The revised proposal covers the handout's required sections and is scoped as a research plan, not an implemented system. This review does not establish instructor acceptance, deadline eligibility, human reading of every paper, or access to the separately shared team plan.

## Requirement coverage

| Requirement | Revised proposal |
| --- | --- |
| Title, four team members, lecture anchors | Title block and opening line |
| Open problem and at least five prior-work citations | Section 1: ten related research works; narrower gap against existing clarification RL |
| Answerable research questions | Section 2, with refutation conditions and primary/secondary scope |
| Method and system | Section 3: tools, information classes, reward, multi-turn training |
| Data, fair baselines, metrics, multi-turn evaluation | Section 4: split isolation, SFT comparison, full-episode evaluation, variability and error analysis |
| Feasible compute plan | Section 5: nine core runs, explicitly unmeasured estimate, pilot and fallback |
| Risks and paper-worthy fallback | Section 5: controlled comparisons and negative results without unsupported learning claims |
| Division of labor | Section 6, all four members |
| AI disclosure | Acknowledgements and README; authors must confirm this matches actual use |

The handout specifies November 5 at 11:59 pm, with at most two slip days. November 8 is beyond that stated allowance; confirm any extension directly with the instructor. No submission was made.

## Source checks and starting reading set

The bibliography now includes 14 sources: eleven research papers, two regulatory sources, and one official model card. Primary-source abstracts, publication records, and relevant official guidance were checked for the claims retained. This is not a full-text literature review; team members must read the papers they cite, as the handout requires.

| Source | Use and review finding |
| --- | --- |
| [CollabLLM](https://proceedings.mlr.press/v267/wu25i.html) | Main multi-turn training precedent. ICML 2025 metadata verified. Removed reductive claims that it merely trains agents to ask more or only uses proxy outcomes. |
| [UserBench](https://arxiv.org/abs/2507.22034) | Preference elicitation with tools. Replaced unsupported NeurIPS 2025 attribution with the verified 2025 arXiv record. |
| [Ask or Assume?](https://arxiv.org/abs/2603.26233) | Clarification decisions in coding. Title, authors, and arXiv identifier verified. |
| [Clarify or Answer](https://arxiv.org/abs/2601.16400) | Added missing close precedent: GRPO for clarification in VQA. Rules out claiming clarification RL itself as new. |
| [tau-bench](https://arxiv.org/abs/2406.12045) | Tool-user-agent structure, end-state scoring, pass^k. ICLR 2025 publication also checked. |
| [AirGapAgent](https://arxiv.org/abs/2405.05175) | Data minimization baseline design; CCS 2024 record verified. |
| [AgentDAM](https://papers.nips.cc/paper_files/paper/2025/hash/c9826b9ea5e1b49b256329934a578d83-Abstract-Datasets_and_Benchmarks_Track.html) | Data-minimization evaluation and prompting defenses; NeurIPS 2025 metadata verified. Avoided a blanket utility-loss claim. |
| [AgentRaft](https://arxiv.org/abs/2603.07557) | Cross-tool over-exposure detection; verified title/authors/identifier. |
| [PrivacyLens](https://arxiv.org/abs/2409.00138) | Contextual privacy evaluation; NeurIPS 2024 record verified. |
| [AgentDojo](https://arxiv.org/abs/2406.13352) | Prompt injection via untrusted tool outputs. |
| [DeepSeekMath](https://arxiv.org/abs/2402.03300) | GRPO origin; does not itself establish a working multi-turn trainer. |
| [Qwen3-1.7B model card](https://huggingface.co/Qwen/Qwen3-1.7B) | Added implementation source for model and thinking modes. Pin a compatible training stack after the GPU pilot. |
| [FINRA Rule 2111](https://www.finra.org/rules-guidance/rulebooks/finra-rules/2111) | Added the Regulation Best Interest exclusion; simulator rules are not legal-compliance claims. |
| [Regulation S-P advisory](https://www.finra.org/rules-guidance/guidance/cybersecurity-advisory-sec-amends-regulation-s-p) | Supports safeguarding motivation, not an invented dollar-valued privacy penalty. |

Start with CollabLLM, Clarify or Answer, tau-bench, AirGapAgent, and AgentDAM; these most directly determine method, novelty, metrics, and baselines. The source count is sufficient for the proposal. More citations are less urgent than reading these closely and implementing the environment.

## Repository consistency and remaining implementation work

The repo has three working privacy tests, but `reset`, `step`, `score`, `check`, `generate_tasks`, and `ScriptedClient.reply` raise `NotImplementedError`; tool schemas are empty. No trained model, evaluation result, or dependency lockfile exists. The revised proposal and READMEs distinguish planned behavior from this current state.

Canary matching is only a verbatim-marker measure. Do not score incoming client replies or profile lookups as leaks. The existing helper's authorization exception covers only the allowed client's low-tier fields; recipient-aware checks must be completed before broader privacy claims. Policy checks also need per-action enforcement, since a final portfolio alone cannot reveal every transient violation.

Before experiments: confirm PACE allocation/queue and API budget; implement and validate the two core task families; freeze disjoint splits; pin dependencies/models/prompts; verify the scripted baseline end to end. No simulation/training feature was implemented during this document review.

Before submission: each author should confirm names, roles, AI disclosure, and source reading. If code or ideas from earlier course work are reused, disclose the actual reuse and the new contribution; this review did not establish such reuse.

## Validation

- Rebuilt with the repository's ACL style: two pages of proposal text plus one page of references.
- No undefined citations or overfull boxes in the final LaTeX build; all three rendered pages visually checked.
- Existing repository privacy tests: 3 passed. This validates the existing helper tests, not an end-to-end simulator.
