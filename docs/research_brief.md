# Research brief: novelty and literature for "Ask Me Nothing"

Paste everything below the line into Claude Science.

---

I'm a graduate student at Georgia Tech working on a team research project for a Conversational AI course (ECE 6256 / CS 6604, Prof. Larry Heck, Fall 2026). The course asks for a paper of at least workshop quality, and if the results hold up we want to submit it to a real venue. I need help with the literature: confirming what is genuinely new, finding every closely related paper, and anticipating what a reviewer would say.

Treat this as a literature review, not a writing task. Please don't draft paper text for us.

## The project in one paragraph

In financial dialogue, a clarifying question is also an act of data collection. If a broker's assistant asks a client for their income in order to raise cash, it has collected sensitive data it had no use for. We study whether multi-turn reinforcement learning can teach a small model (Qwen3-1.7B with LoRA, GRPO over whole dialogues) to complete a policy-constrained tool task while asking only for information that is missing and necessary. Each unnecessary question is priced by the sensitivity of the field it requests, and some questions are mandatory under a suitability rule, so never asking is itself a policy violation.

Code: https://github.com/IncorvaiaM1/ask-me-nothing (public). The proposal is in `docs/proposal/proposal.tex`, the current milestone draft in `docs/milestone/milestone.tex`, and the plan in `docs/PLAN.md`.

## Research questions as proposed

1. Does multi-turn GRPO improve policy-compliant task success over prompting and supervised distillation?
2. Does pricing unnecessary sensitive requests reduce over-collection and leakage without reducing success?
3. Are gains explained by fewer redundant questions while necessary ones are kept, rather than a never-ask shortcut?
4. Do gains transfer to LLM-simulated clients and hold up against instructions injected into tool outputs?

## What we have built and found so far

- **BrokerDesk**, a simulated brokerage desk: two task families (raising cash, rebalancing to a model portfolio), 300/50/100 train/validation/test tasks, nine tools including a structured `ask_client(field, question)`, a written desk policy with exact end-state checks, and a 12-turn budget with prices moving each turn.
- **Exact scoring of every question.** Each field in each task is labeled derivable, on file, mandatory, client-only or irrelevant, so each question is scored as needed or unneeded with no judge model.
- **Privacy measurement.** Client fields fall in three sensitivity tiers. Every value carries a unique canary string, so a leak is an exact match in an outbound message.
- **Prompt injection.** Attacker instructions are placed in untrusted tool outputs, paired with clean runs (AgentDojo threat model).
- **Reward:** goal met, minus a per-question cost, minus a tier-weighted cost per unneeded request, minus penalties for policy violations and leaks.
- **First result:** a prompted Qwen3-1.7B completes 0 of 50 validation tasks. It loops on pre-trade checks, claims trades it never made, and 1.30 of its 1.76 questions per episode are unneeded. Under injection, 8% of attacks succeed.
- **Plan:** supervised warm start on a scripted agent that over-asks, then GRPO. The central comparison is the full reward against a flat question cost, which is the prior-work-style baseline.

## What I already know is close

Please verify these, read what you can, and tell me if any of them already does what we're claiming:

- **Clarification with RL:** CollabLLM (Wu et al., ICML 2025); SpeakRL (Acikgoz et al., arXiv 2512.13159); Structured Uncertainty guided Clarification / ClarifyBench (Suri et al., Findings of ACL 2026); AskBench (Zhao et al., arXiv 2602.11199); Learning Steerable Clarification Policies with Collaborative Self-play (Berant et al., arXiv 2512.04068), which puts an explicit cost on each question and is our closest method; Clarify or Answer (arXiv 2601.16400), GRPO for clarification in visual question answering; UserBench; Ask or Assume? (arXiv 2603.26233); CIGAsk (arXiv 2609.24290); InfoPO (arXiv 2603.00656).
- **Privacy in agents:** Contextual Integrity in LLMs via Reasoning and RL (Lan et al., NeurIPS 2025); ToolPrivacyBench (arXiv 2606.28061); Do Phone-Use Agents Respect Your Privacy? / MyPhoneBench (arXiv 2604.00986), which measures agents asking for personal data they don't need but does no training; AirGapAgent; AgentDAM; AgentRaft; PrivacyLens; Contextualized Privacy Defense for LLM Agents (arXiv 2603.02983); AgentLeak (arXiv 2602.11510).
- **Evaluation and attacks:** τ-bench and pass^k; AgentDojo.

## What I need from you

1. **Novelty verdict.** Our claim is that no prior work trains an agent where a clarifying question's cost depends on the sensitivity of what it collects, under a policy that also makes some questions mandatory. Search hard for anything that contradicts this, including work outside NLP: privacy-aware dialogue systems, privacy-preserving or minimal-disclosure preference elicitation, data minimization in recommender or survey design, medical history-taking agents, and RL or bandits with information-acquisition costs. Tell me plainly whether the claim survives, and if not, what the narrower defensible claim is.
2. **Missing related work.** Find papers we haven't listed in these areas:
   - RL or preference training for clarification and information seeking in multi-turn or tool-use settings, especially 2025 and 2026;
   - training (not just evaluating) agents for data minimization, contextual integrity or need-to-know disclosure;
   - financial or banking agent benchmarks that involve client data, suitability or compliance;
   - multi-turn GRPO and credit assignment for tool-using agents, including turn-level rewards and loss masking of user and tool tokens;
   - reward hacking in clarification RL, especially agents learning never to ask or always to ask;
   - user simulators for evaluating agents, including goal drift in LLM-simulated users.
3. **The closest five papers in depth.** For each: what they do, which of our four research questions they overlap with, and the one-sentence difference we should state in our paper.
4. **Likely reviewer objections,** given that the environment is synthetic, the client is scripted, the requests are templated, and the model is small. For each, say whether published work has addressed the same objection and how.
5. **Experiments that would most strengthen the paper,** judged against what the closest papers report. We already plan a cost sweep, ablations without the privacy cost and without the question cost, held-out attacks, and LLM-simulated clients.
6. **Venues.** Workshops and conferences where this fits (dialogue, privacy in NLP, agents, finance and NLP), with current deadlines checked on the official call pages. Note any that close before mid-December 2026 or fall in January to March 2027.

## Ground rules

- Every paper you cite must be real and linked: arXiv, ACL Anthology, OpenReview or the publisher page. Give title, authors, year, and venue if published. If you can't confirm a paper exists, leave it out rather than guess.
- Say whether each paper is peer-reviewed or a preprint.
- Tell me which papers you read in full and which you're summarizing from the abstract.
- Prioritize 2025 and 2026 work, but include older foundational papers where they define the problem.
- Deliver a report with a section per item above, plus a BibTeX file for everything you recommend we cite.
