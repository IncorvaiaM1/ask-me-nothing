"""Grouped rollouts for multi-turn GRPO, independent of the RL library.

One *group* is ``group_size`` complete dialogues on the same task with the same
price path, so the only thing that differs inside a group is what the policy did.
Each dialogue gets one scalar reward from ``BrokerDeskEnv.score`` and a
group-relative advantage. A trainer then needs only two things from here: the chat
messages of each dialogue and its advantage (see ``training/masking.py`` for
turning messages into token ids with an assistant-only loss mask).
"""

from __future__ import annotations

import json
import statistics
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, dataclass, field
from typing import Callable

from brokerdesk.env import MAX_TURNS, RewardWeights
from brokerdesk.policy import Policy, default_policy
from brokerdesk.tasks import Task
from eval.rollout import run_episode


@dataclass
class Trajectory:
    task_id: str
    group_id: str
    index: int                       # position inside the group
    messages: list[dict]             # OpenAI chat format: system, user, assistant(tool_calls), tool, ...
    reward: float
    advantage: float = 0.0
    score: dict = field(default_factory=dict)


def group_advantages(rewards: list[float], normalize: bool = True, eps: float = 1e-6) -> list[float]:
    """GRPO advantage: reward minus the group mean, divided by the group std.

    A group whose rewards are all equal carries no learning signal and gets zeros.
    """
    mean = statistics.fmean(rewards)
    centered = [r - mean for r in rewards]
    if not normalize:
        return centered
    std = statistics.pstdev(rewards)
    if std < eps:
        return [0.0] * len(rewards)
    return [c / (std + eps) for c in centered]


def messages_from_transcript(system: str, transcript: list[dict]) -> list[dict]:
    """Rebuild chat messages from an ``eval.rollout`` transcript.

    Used for agents that do not keep their own message list (the scripted ones).
    A model-backed agent's own ``messages`` are preferred, because they hold the
    exact text the model produced, including malformed calls.
    """
    messages = [{"role": "system", "content": system}]
    call_id = None
    for n, turn in enumerate(transcript):
        if turn["role"] == "client":
            messages.append({"role": "user", "content": turn["content"]})
        elif turn["role"] == "agent":
            call_id = f"call_{n}"
            messages.append({"role": "assistant", "content": "", "tool_calls": [{
                "id": call_id, "type": "function",
                "function": {"name": turn["tool"], "arguments": json.dumps(turn["arguments"])}}]})
        else:
            messages.append({"role": "tool", "tool_call_id": call_id, "content": turn["content"]})
    return messages


def collect_group(task: Task, make_agent: Callable[[int], object], group_size: int, system: str,
                  seed: int = 0, weights: RewardWeights | None = None, policy: Policy | None = None,
                  max_turns: int = MAX_TURNS, injection=None, workers: int = 1,
                  normalize: bool = True) -> list[Trajectory]:
    """Roll out one group. ``make_agent(i)`` returns a fresh agent for group member ``i``."""
    policy = policy or default_policy()

    def one(i: int) -> Trajectory:
        agent = make_agent(i)
        ep = run_episode(task, agent, policy, weights, seed=seed, max_turns=max_turns, injection=injection)
        messages = getattr(agent, "messages", None)
        if messages is None:
            messages = messages_from_transcript(system, ep["transcript"])
        return Trajectory(task_id=task.task_id, group_id=f"{task.task_id}:{seed}", index=i,
                          messages=list(messages), reward=ep["score"]["reward"], score=ep["score"])

    if workers > 1:
        with ThreadPoolExecutor(max_workers=workers) as pool:
            group = list(pool.map(one, range(group_size)))
    else:
        group = [one(i) for i in range(group_size)]
    for traj, adv in zip(group, group_advantages([t.reward for t in group], normalize)):
        traj.advantage = adv
    return group


def collect_batch(tasks: list[Task], make_agent: Callable[[int], object], group_size: int, system: str,
                  **kwargs) -> list[Trajectory]:
    batch: list[Trajectory] = []
    for task in tasks:
        batch.extend(collect_group(task, make_agent, group_size, system, **kwargs))
    return batch


def batch_stats(batch: list[Trajectory]) -> dict:
    groups: dict[str, list[Trajectory]] = {}
    for t in batch:
        groups.setdefault(t.group_id, []).append(t)
    return {
        "trajectories": len(batch),
        "groups": len(groups),
        "mean_reward": statistics.fmean(t.reward for t in batch),
        "success": statistics.fmean(bool(t.score.get("success")) for t in batch),
        # Groups with identical rewards contribute nothing to the update.
        "groups_with_signal": sum(any(t.advantage != 0.0 for t in g) for g in groups.values()),
    }


def write_jsonl(batch: list[Trajectory], path) -> None:
    with open(path, "w", encoding="utf-8") as fh:
        for t in batch:
            fh.write(json.dumps(asdict(t)) + "\n")


def read_jsonl(path) -> list[Trajectory]:
    with open(path, encoding="utf-8") as fh:
        return [Trajectory(**json.loads(line)) for line in fh if line.strip()]
