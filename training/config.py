"""Reward configurations for the main method and the two core ablations."""

from __future__ import annotations

from brokerdesk.env import RewardWeights

REWARD_CONFIGS: dict[str, RewardWeights] = {
    "full": RewardWeights(),
    "no_privacy_cost": RewardWeights(privacy_cost=(0.0, 0.0, 0.0)),
    "no_question_cost": RewardWeights(question_cost=0.0),
}
