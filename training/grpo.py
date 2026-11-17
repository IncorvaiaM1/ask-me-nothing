"""Reference policy-gradient loss for grouped multi-turn rollouts.

NOT YET RUN: neither PACE nor the development machine had torch available when
this was written. Treat it as the specification of what the chosen RL library
must compute, and check it in the 0.6B pilot before trusting any number from it.

This is the on-policy form of the GRPO objective (importance ratio equal to 1,
no clipping, no KL term), which is what one gradient step per freshly sampled
batch reduces to. A library trainer that reuses a batch for several steps adds
the ratio and the clip.
"""

from __future__ import annotations


def grpo_loss(logits, input_ids, loss_mask, advantages):
    """Advantage-weighted log-likelihood of the assistant tokens.

    logits      [batch, time, vocab]  model outputs for ``input_ids``
    input_ids   [batch, time]
    loss_mask   [batch, time]  1 on assistant tokens (training/masking.py), 0 elsewhere and on padding
    advantages  [batch]        group-normalized advantages (training/rollouts.py)

    Log-probabilities are averaged over each dialogue's assistant tokens, so a long
    dialogue does not outweigh a short one.
    """
    import torch

    logp = torch.log_softmax(logits[:, :-1].float(), dim=-1)
    token_logp = logp.gather(-1, input_ids[:, 1:].unsqueeze(-1)).squeeze(-1)
    mask = loss_mask[:, 1:].to(token_logp.dtype)
    per_dialogue = (token_logp * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1)
    return -(advantages.to(per_dialogue.dtype) * per_dialogue).mean()
