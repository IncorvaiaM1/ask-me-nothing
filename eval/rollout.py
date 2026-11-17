"""Run one agent through one BrokerDesk episode and keep the full transcript."""

from __future__ import annotations

from brokerdesk.env import BrokerDeskEnv, RewardWeights
from brokerdesk.policy import Policy, default_policy
from brokerdesk.tasks import Task


def run_episode(task: Task, agent, policy: Policy | None = None, weights: RewardWeights | None = None,
                seed: int = 0, max_turns: int = 12, injection=None) -> dict:
    """``agent`` needs ``reset(opening)`` and ``act(observation) -> (tool_name, arguments)``."""
    env = BrokerDeskEnv(task, policy or default_policy(), seed, weights, max_turns, injection)
    opening = env.reset()
    agent.reset(opening)
    transcript = [{"role": "client", "content": opening}]
    observation, done = None, False
    while not done:
        tool, args = agent.act(observation)
        observation, done = env.step(tool, args)
        transcript.append({"role": "agent", "tool": tool, "arguments": args})
        transcript.append({"role": "tool", "content": observation})
    score = env.score()
    score["rollout_seed"] = seed
    return {"score": score, "transcript": transcript}
