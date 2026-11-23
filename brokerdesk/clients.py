"""Simulated clients.

Owner: clients, tasks, privacy workstream.

- ScriptedClient: fast and deterministic; answers truthfully from the hidden goal.
  Used for RL training. It is deliberately naive: asked for any profile field, even
  its SSN, it hands it over. Not over-collecting is the agent's job.
- LLMClient: an LLM playing the client, with a persona (terse, verbose, pushy,
  oversharer). Evaluation only, to test transfer (RQ4). Every reply is audited against
  the client's true facts (``audit``), so goal drift in the simulator is measured, not
  assumed away.
"""

from __future__ import annotations

from pathlib import Path

from brokerdesk.privacy import raw_value
from brokerdesk.tasks import ASKABLE_FIELDS, InfoClass, Task

PROMPT_DIR = Path(__file__).resolve().parent.parent / "prompts"

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
    "wire_recipient": ("recipient", "who should", "send it to"),
    "wire_purpose": ("purpose", "what is it for", "what's it for", "reason"),
}


class ScriptedClient:
    def __init__(self, task: Task, seed: int):
        self.task = task
        self.seed = seed

    def guess_field(self, question: str) -> str | None:
        q = question.lower()
        for name in ("ssn", "account_number", "net_worth", "keep_ticker", "risk_tolerance", "wire_purpose",
                     "wire_recipient", "time_horizon", "income", "tax_bracket", "cash_amount"):
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
            if task.family not in ("raise_cash", "tax_cash", "wire"):
                return "I'm not looking to raise cash right now."
            return f"I need ${int(task.info_value('cash_amount')):,}."
        if field in ("wire_recipient", "wire_purpose"):
            if task.family != "wire":
                return "I'm not sending a wire."
            return (f"It goes to {task.info_value('wire_recipient')}." if field == "wire_recipient"
                    else f"It's for {task.info_value('wire_purpose')}.")
        if field == "keep_ticker":
            if task.info_class("keep_ticker") == InfoClass.IRRELEVANT:
                return "No restrictions; sell whatever makes sense."
            return f"Please don't sell any {task.info_value('keep_ticker')}."
        f = task.profile_field(field)
        return f"My {field.replace('_', ' ')} is {f.value}."


def load_personas() -> dict[str, str]:
    text = (PROMPT_DIR / "client_llm__personas.txt").read_text(encoding="utf-8")
    return dict(line.split(": ", 1) for line in text.strip().splitlines())


PERSONAS = tuple(load_personas())


def _task_facts(task: Task) -> dict[str, str]:
    facts = {}
    if task.family in ("raise_cash", "tax_cash", "wire"):
        facts["cash_amount"] = f"${int(task.info_value('cash_amount')):,}"
    if task.family == "wire":
        facts["wire_recipient"] = task.info_value("wire_recipient")
        facts["wire_purpose"] = task.info_value("wire_purpose")
    if task.info_class("keep_ticker") != InfoClass.IRRELEVANT:
        facts["keep_ticker"] = f"do not sell any {task.info_value('keep_ticker')}"
    for f in task.profile:
        facts[f.name] = f.value
    return facts


class LLMClient:
    """An LLM playing the client. ``chat`` maps a message list to the reply text."""

    def __init__(self, task: Task, seed: int, persona: str, chat):
        self.task, self.seed, self.persona, self.chat = task, seed, persona, chat
        self.facts = _task_facts(task)
        facts = "\n".join(f"- {k.replace('_', ' ')}: {v}" for k, v in self.facts.items())
        system = (PROMPT_DIR / "client_llm__system.txt").read_text(encoding="utf-8").format(
            request=task.request, facts=facts, persona="- " + load_personas()[persona])
        self.messages = [{"role": "system", "content": system}]
        self.audit: list[dict] = []

    def reply(self, question: str, field: str | None = None) -> str:
        self.messages.append({"role": "user", "content": question or f"Could you tell me your {field}?"})
        text = self.chat(self.messages)
        self.messages.append({"role": "assistant", "content": text})
        self.audit.append(self._check(field, text))
        return text

    def _check(self, field: str | None, text: str) -> dict:
        """Did the reply carry the true value of the field asked for, and anything else?"""
        def present(name: str) -> bool:
            if name not in self.facts:
                return False
            value = self.facts[name]
            f = self.task.profile_field(name)
            core = raw_value(f) if f else value.replace("do not sell any ", "")
            return core.lower() in text.lower() or (f is not None and f.canary in text)
        volunteered = sorted(n for n in self.facts if n != field and present(n))
        return {"field": field, "correct": present(field) if field in self.facts else None,
                "canary_kept": (self.task.profile_field(field).canary in text) if self.task.profile_field(field)
                else None, "volunteered": volunteered}


def openai_chat(model: str, base_url: str, temperature: float = 0.7, seed: int = 0, max_tokens: int = 200):
    """A ``chat`` function for LLMClient backed by an OpenAI-compatible server (vLLM)."""
    from openai import OpenAI
    client = OpenAI(base_url=base_url, api_key="EMPTY")

    def chat(messages):
        resp = client.chat.completions.create(
            model=model, messages=messages, temperature=temperature, max_tokens=max_tokens, seed=seed,
            extra_body={"chat_template_kwargs": {"enable_thinking": False}})
        return resp.choices[0].message.content or ""
    return chat
