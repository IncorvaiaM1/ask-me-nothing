# Tiered arm, seed 0, before the KL penalty (v1.0 environment)

GRPO from the SFT adapter, 60 steps, no KL term, final checkpoint, test split, 5 rollouts.
The policy collapsed to one fixed script: ask `cash_amount` (even when stated), never ask
`keep_ticker` or `tax_bracket`, guess a 32% bracket. Required-field recall 0.51, tax_cash
success 0.50 (a coin flip on the bracket), overall success 0.82.

Two causes, both fixed in v1.1:
1. In v1.0 the protected holding was a random position, so guessing that nothing was
   protected still succeeded in ~86% of tasks. Asking for it bought nothing, and the
   habit of not asking spread to `tax_bracket`. The protected holding is now the one an
   unconstrained sale hits first, so guessing always fails.
2. Once all samples in a group follow the same script, the group's rewards are equal,
   the advantage is zero, and nothing pushes the policy back. v1.1 adds a KL penalty to
   the SFT adapter (beta 0.04) and evaluates the best checkpoint on validation reward.

Kept as the "no KL" ablation; note it also predates the environment fix.
