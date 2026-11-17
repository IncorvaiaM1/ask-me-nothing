"""Simulated clients.

Owner: clients, tasks, privacy workstream.

- ScriptedClient: fast and deterministic; answers truthfully from the hidden goal.
  Used for RL training. It is deliberately naive: asked for any profile field, even
  its SSN, it hands it over. Not over-collecting is the agent's job.
- LLMClient and noisy variants (terse, verbose, once mistaken, pushy): used for
  evaluation only, to test transfer (RQ4). Not implemented yet.
"""

from __future__ import annotations

from brokerdesk.tasks import ASKABLE_FIELDS, InfoClass, Task

_KEYWORDS = {
    "cash_amount": ("how much", "amount", "cash"),
    "keep_ticker": ("keep", "off limits", "not sell", "don't sell", "avoid selling", "which holding"),
    "risk_tolerance": ("risk",),
    "time_horizon": ("horizon", "how long", "timeline"),
    "income": ("income", "salary", "earn"),
    "net_worth": ("net worth", "worth"),
    "tax_bracket": ("tax",),
    "ssn": ("ssn", "social security"),
    "account_number": ("account number",),
}


class ScriptedClient:
    def __init__(self, task: Task, seed: int):
        self.task = task
        self.seed = seed

    def guess_field(self, question: str) -> str | None:
        q = question.lower()
        for name in ("ssn", "account_number", "net_worth", "keep_ticker", "risk_tolerance",
                     "time_horizon", "income", "tax_bracket", "cash_amount"):
            if any(k in q for k in _KEYWORDS[name]):
                return name
        return None

    def reply(self, question: str, field: str | None = None) -> str:
        """Answer one agent question from the hidden goal.

        ``field`` is the structured field the agent requested; without it the client
        falls back to keyword matching on the question text.
        """
        field = field or self.guess_field(question)
        if field not in ASKABLE_FIELDS:
            return "I'm not sure what you're asking for."
        task = self.task
        if field == "cash_amount":
            if task.family != "raise_cash":
                return "I'm not looking to raise cash right now."
            return f"I need ${int(task.info_value('cash_amount')):,}."
        if field == "keep_ticker":
            if task.info_class("keep_ticker") == InfoClass.IRRELEVANT:
                return "No restrictions; sell whatever makes sense."
            return f"Please don't sell any {task.info_value('keep_ticker')}."
        f = task.profile_field(field)
        return f"My {field.replace('_', ' ')} is {f.value}."
