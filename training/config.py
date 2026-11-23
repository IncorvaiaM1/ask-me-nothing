"""Reward configurations: the arms of the central comparison.

- ``tiered``: the method. Unneeded requests cost by the sensitivity of the field.
- ``flat``: prior-work-style uniform cost per unneeded request, set so that the over-asking
  teacher's expected collection penalty is the same as under ``tiered`` (0.031 per
  request; see ``matched_flat_cost``). Same per-question cost.
- ``shuffled``: the same tier costs, but assigned to the wrong fields by a fixed
  permutation among the low and sensitive profile fields. Restricted fields keep their
  tier. If ``tiered`` beats ``shuffled``, the model learned which data is sensitive,
  not merely that requests cost something graded.
- ``no_collection_cost``: question cost only, no charge for what is collected.

``sample_steerable`` draws per-group tier costs for the cost-conditioned variant; the
draw is written into the system prompt so the policy can condition on it.
"""

from __future__ import annotations

import random

from brokerdesk.env import RewardWeights
from brokerdesk.tasks import PROFILE_TIERS

TIER_COSTS = (0.01, 0.05, 0.5)

# Fixed permutation of tiers among the low and sensitive profile fields, keeping the same
# set of prices (two low, three sensitive). Both low fields become sensitive and two of
# the three sensitive fields become low; one (net worth) must keep its tier. The key
# field, the tax bracket, becomes cheap.
SHUFFLED_TIERS = {
    "risk_tolerance": 1, "time_horizon": 1, "income": 0, "net_worth": 1, "tax_bracket": 0,
    "ssn": 2, "account_number": 2,
}


def matched_flat_cost() -> float:
    """Uniform cost with the same expected penalty as tiered on the teacher's collection.

    The teacher's unneeded requests on the training split average 1.46 low and 1.67
    sensitive per episode (measured with ``python -m training.config``); restricted 0.
    """
    low, sens = 1.46, 1.67
    return round((low * TIER_COSTS[0] + sens * TIER_COSTS[1]) / (low + sens), 3)


REWARD_CONFIGS: dict[str, RewardWeights] = {
    "tiered": RewardWeights(privacy_cost=TIER_COSTS),
    "flat": RewardWeights(privacy_cost=(matched_flat_cost(),) * 3),
    "shuffled": RewardWeights(privacy_cost=TIER_COSTS, field_tier=SHUFFLED_TIERS),
    "no_collection_cost": RewardWeights(privacy_cost=(0.0, 0.0, 0.0)),
}
REWARD_CONFIGS["full"] = REWARD_CONFIGS["tiered"]          # name used before the v1 arms


def sample_steerable(rng: random.Random) -> RewardWeights:
    """Per-group costs for the cost-conditioned policy: log-uniform per tier, ordered."""
    low = 10 ** rng.uniform(-3, -1.3)
    sens = low * 10 ** rng.uniform(0, 1.3)
    restr = max(sens, 10 ** rng.uniform(-1, 0))
    return RewardWeights(privacy_cost=(round(low, 4), round(sens, 4), round(restr, 4)))


def describe_costs(w: RewardWeights) -> str:
    """The sentence a cost-conditioned policy sees in its system prompt."""
    return ("Collection costs this session, charged for every client field you request that the task "
            f"does not need: low-sensitivity {w.privacy_cost[0]:g}, sensitive {w.privacy_cost[1]:g}, "
            f"restricted {w.privacy_cost[2]:g}. Each question also costs {w.question_cost:g}.")


if __name__ == "__main__":
    from baselines.scripted import HeuristicAgent
    from brokerdesk.policy import default_policy
    from brokerdesk.tasks import load_split
    from eval.rollout import run_episode

    pol = default_policy()
    scores = [run_episode(t, HeuristicAgent("always_ask", pol), pol)["score"] for t in load_split("train")]
    per = [sum(s["unneeded_requests_by_tier"][i] for s in scores) / len(scores) for i in range(3)]
    print("teacher unneeded requests per episode by tier:", [round(x, 2) for x in per])
    assert set(SHUFFLED_TIERS) == set(PROFILE_TIERS)
