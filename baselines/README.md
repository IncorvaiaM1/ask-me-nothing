# baselines

None of these are designed to lose. Owner: baselines and eval workstream.

1. **Prompted:** the same small model with a careful, privacy-aware prompt.
2. **SFT-distilled:** the same model fine-tuned on filtered frontier-model trajectories.
3. **Frontier reference:** a prompted frontier model through the API; not a guaranteed upper bound.
4. **AirGap-style minimizer:** a separate minimizer selects the client data the task needs before the prompted model sees it (after Bagdasarian et al., 2024).

These are planned, not implemented. Use the same information, tools, and turn limits across methods; report minimizer overhead. SFT+GRPO starts from the SFT baseline's checkpoint. Tune prompts and filters on validation only, and set an API spending cap before data generation.
