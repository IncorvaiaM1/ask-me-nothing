# baselines

None of these are designed to lose. Owner: baselines and eval workstream.

1. **Prompted:** the same small model with a careful, privacy-aware prompt.
2. **SFT-distilled:** the same model fine-tuned on filtered frontier-model trajectories.
3. **Frontier ceiling:** a prompted frontier model through the API.
4. **AirGap-style minimizer:** a separate minimizer selects the client data the task needs before the prompted model sees it (after Bagdasarian et al., 2024).
