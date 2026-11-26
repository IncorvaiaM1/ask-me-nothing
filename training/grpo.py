"""Policy-gradient loss for grouped multi-turn rollouts.

This is the on-policy form of the GRPO objective (importance ratio equal to 1, no
clipping): one gradient step per freshly sampled batch. An optional KL penalty to a
frozen reference policy (the SFT adapter) uses the k3 estimator of Schulman (2020),
as in DeepSeekMath's GRPO. Without it the first v1 run (tiered, seed 0) collapsed to
one fixed script that stopped asking for client-only fields.
"""

from __future__ import annotations


def token_logps(logits, input_ids):
    """Log-probability of each next token: [batch, time - 1]."""
    import torch

    logp = torch.log_softmax(logits[:, :-1].float(), dim=-1)
    return logp.gather(-1, input_ids[:, 1:].unsqueeze(-1)).squeeze(-1)


def grpo_loss(logits, input_ids, loss_mask, advantages, ref_logp=None, beta: float = 0.0,
              return_kl: bool = False):
    """Advantage-weighted log-likelihood of the assistant tokens.

    logits      [batch, time, vocab]  model outputs for ``input_ids``
    input_ids   [batch, time]
    loss_mask   [batch, time]  1 on assistant tokens, 0 elsewhere and on padding
    advantages  [batch] for one advantage per dialogue, or [batch, time] for per-token
                (turn-level) advantages aligned with ``input_ids``

    ref_logp    [batch, time - 1]  reference log-probabilities from ``token_logps``
                (no gradient); with ``beta`` > 0 adds beta * KL(policy || reference)

    Log-probabilities are averaged over each dialogue's assistant tokens, so a long
    dialogue does not outweigh a short one. With ``return_kl`` the result is
    ``(loss, mean_kl)``.
    """
    import torch

    token_logp = token_logps(logits, input_ids)
    mask = loss_mask[:, 1:].to(token_logp.dtype)
    adv = advantages.to(token_logp.dtype)
    if adv.dim() == 1:
        adv = adv[:, None].expand_as(token_logp)
    else:
        adv = adv[:, 1:]
    denom = mask.sum(dim=1).clamp(min=1)
    per_dialogue = (token_logp * adv * mask).sum(dim=1) / denom
    loss = -per_dialogue.mean()
    kl = torch.zeros((), device=token_logp.device)
    if ref_logp is not None:
        d = ref_logp.to(token_logp.dtype) - token_logp
        kl = (((torch.exp(d) - d - 1) * mask).sum(dim=1) / denom).mean()
        if beta:
            loss = loss + beta * kl
    return (loss, kl.detach()) if return_kl else loss
