"""The BrokerDesk episode loop.

Owner: environment and policy workstream. Critical path: a minimal version that
runs end to end is due Nov 9 so baselines and RL can start.

Reward (see docs/PLAN.md):
    task success
    - time cost per question (prices move between turns)
    - privacy cost per unneeded field, scaled by tier
    - large penalties for policy violations and leaks
"""

from __future__ import annotations

from dataclasses import dataclass

from brokerdesk.policy import Policy
from brokerdesk.tasks import Task


@dataclass
class RewardWeights:
    question_cost: float = 0.02
    privacy_cost: tuple[float, float, float] = (0.01, 0.05, 0.5)   # by Tier: LOW, SENSITIVE, RESTRICTED
    violation_penalty: float = 1.0
    leak_penalty: float = 1.0


class BrokerDeskEnv:
    def __init__(self, task: Task, policy: Policy, seed: int,
                 weights: RewardWeights | None = None, max_turns: int = 12):
        self.task = task
        self.policy = policy
        self.seed = seed
        self.weights = weights or RewardWeights()
        self.max_turns = max_turns

    def reset(self) -> str:
        """Start an episode; return the client's opening message. TODO."""
        raise NotImplementedError

    def step(self, tool_name: str, arguments: dict) -> tuple[str, bool]:
        """Run one tool call; return (observation, done). TODO."""
        raise NotImplementedError

    def score(self) -> dict:
        """Final reward and every metric in the paper for this episode. TODO."""
        raise NotImplementedError
