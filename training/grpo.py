"""Policy-gradient loss for grouped multi-turn rollouts.

This is the on-policy form of the GRPO objective (importance ratio equal to 1, no
clipping, no KL term): one gradient step per freshly sampled batch. Verified on PACE
(Nov 17): runs on Qwen3-0.6B and 1.7B and falls on a fixed batch as it should.
"""

from __future__ import annotations


def grpo_loss(logits, input_ids, loss_mask, advantages):
    """Advantage-weighted log-likelihood of the assistant tokens.

    logits      [batch, time, vocab]  model outputs for ``input_ids``
    input_ids   [batch, time]
    loss_mask   [batch, time]  1 on assistant tokens, 0 elsewhere and on padding
    advantages  [batch] for one advantage per dialogue, or [batch, time] for per-token
                (turn-level) advantages aligned with ``input_ids``

    Log-probabilities are averaged over each dialogue's assistant tokens, so a long
    dialogue does not outweigh a short one.
    """
    import torch

    logp = torch.log_softmax(logits[:, :-1].float(), dim=-1)
    token_logp = logp.gather(-1, input_ids[:, 1:].unsqueeze(-1)).squeeze(-1)
    mask = loss_mask[:, 1:].to(token_logp.dtype)
    adv = advantages.to(token_logp.dtype)
    if adv.dim() == 1:
        adv = adv[:, None].expand_as(token_logp)
    else:
        adv = adv[:, 1:]
    per_dialogue = (token_logp * adv * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1)
    return -per_dialogue.mean()
