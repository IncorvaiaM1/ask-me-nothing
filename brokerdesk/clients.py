"""Simulated clients.

Owner: clients, tasks, privacy workstream.

- ScriptedClient: fast and deterministic; answers truthfully from the hidden goal.
  Used for RL training.
- LLMClient and noisy variants (terse, verbose, once mistaken, pushy): used for
  evaluation only, to test transfer (RQ4).
"""

from __future__ import annotations

from brokerdesk.tasks import Task


class ScriptedClient:
    def __init__(self, task: Task, seed: int):
        self.task = task
        self.seed = seed

    def reply(self, question: str) -> str:
        """Answer one agent question from the hidden goal. TODO (Nov 9 minimal env)."""
        raise NotImplementedError
