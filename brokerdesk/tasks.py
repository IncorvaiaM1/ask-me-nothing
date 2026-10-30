"""Task generation with labeled information classes.

Owner: clients, tasks, privacy workstream.

Each task is a client request (rebalance, hedge, raise cash, tax-aware trim) plus a
hidden goal. Every piece of information the agent might need is tagged with where
it lives, so every question the agent asks can be scored as needed or unneeded.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum


class InfoClass(str, Enum):
    DERIVABLE = "derivable"        # inferable from what the agent already has
    ON_FILE = "on_file"            # available through a look-up tool
    MANDATORY = "mandatory"        # suitability profile item not on file: must ask
    CLIENT_ONLY = "client_only"    # a preference only the client holds: asking is justified
    IRRELEVANT = "irrelevant"      # not needed for this task


@dataclass
class InfoItem:
    key: str
    info_class: InfoClass
    value: str


@dataclass
class Task:
    task_id: str
    client_id: str
    request: str                         # what the client says on turn 1
    hidden_goal: dict                    # target end state, checked exactly
    info: list[InfoItem] = field(default_factory=list)


def generate_tasks(n: int, seed: int) -> list[Task]:
    """Generate ``n`` tasks deterministically from ``seed``. TODO (Nov 9 minimal env)."""
    raise NotImplementedError
