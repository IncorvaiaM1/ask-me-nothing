# baselines

None of these are designed to lose. Owner: baselines and eval workstream.

1. **Prompted:** the same small model with a careful, privacy-aware prompt.
2. **SFT-distilled:** the same model fine-tuned on filtered frontier-model trajectories.
3. **Frontier reference:** a prompted frontier model through the API; not a guaranteed upper bound.
4. **AirGap-style minimizer:** a separate minimizer selects the client data the task needs before the prompted model sees it (after Bagdasarian et al., 2024).

Status: `scripted.py` holds three scripted reference agents (need-to-know, never-ask, always-ask) that run with no model. `prompted.py` implements baselines 1 and 3 through any OpenAI-compatible endpoint; it is tested against a fake client but has not been run on a real model yet. Baselines 2 and 4 are planned, not implemented. Use the same information, tools, and turn limits across methods; report minimizer overhead. SFT+GRPO starts from the SFT baseline's checkpoint. Tune prompts and filters on validation only, and set an API spending cap before data generation.
