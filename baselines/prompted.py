"""Prompted baseline: any chat model behind an OpenAI-compatible endpoint.

The same class runs the small model (Qwen3 served by vLLM on PACE-ICE) and a
frontier reference (a hosted API), so both get identical prompts, tools, and turn
limits. Prompts are loaded from ``prompts/`` and never defined inline.

Serving Qwen3 with vLLM for tool calling:

    vllm serve Qwen/Qwen3-1.7B --enable-auto-tool-choice --tool-call-parser hermes

If the server returns no structured tool call, the agent falls back to parsing a
``<tool_call>{...}</tool_call>`` block or a bare JSON object out of the text.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

from brokerdesk.env import MAX_TURNS
from brokerdesk.market import UNIVERSE
from brokerdesk.policy import Policy, default_policy
from brokerdesk.tools import openai_tools

PROMPT_DIR = Path(__file__).resolve().parent.parent / "prompts"
_THINK = re.compile(r"<think>.*?</think>", re.S)
_OPEN_THINK = re.compile(r"^\s*<think>.*", re.S)     # thinking cut off by the token limit


def strip_thinking(text: str) -> str:
    """Drop reasoning from a reply before it goes into the history.

    Qwen3's guidance for multi-turn use is to keep only the final output of past turns.
    Kept, a few thousand thinking tokens per turn fill a 32k context within 12 turns.
    """
    return _OPEN_THINK.sub("", _THINK.sub("", text or "")).strip()


_TOOL_CALL = re.compile(r"<tool_call>\s*(\{.*?\})\s*</tool_call>", re.S)


def load_prompt(name: str) -> str:
    return (PROMPT_DIR / name).read_text(encoding="utf-8").strip()


def system_prompt(policy: Policy | None = None, max_turns: int = MAX_TURNS, privacy_aware: bool = True) -> str:
    policy = policy or default_policy()
    text = load_prompt("agent__system.txt").format(
        equities=", ".join(t for t, c in UNIVERSE.items() if c == "equity"),
        bonds=", ".join(t for t, c in UNIVERSE.items() if c == "bond"),
        restricted=", ".join(sorted(policy.restricted)) or "none",
        max_position_pct=round(policy.max_position_frac * 100),
        max_order_notional=f"{policy.max_order_notional:,.0f}",
        model_table="; ".join(f"{k} {round(v * 100)}%" for k, v in policy.model_equity_frac.items()),
        rebalance_tol_pct=round(policy.rebalance_tol * 100),
        max_cash_pct=round(policy.max_cash_frac * 100),
        cash_overshoot_pct=round(policy.cash_overshoot * 100),
        tax_high_pct=f"{policy.tax_high_bracket_pct:g}",
        wire_threshold=f"{policy.wire_purpose_threshold:,.0f}",
        max_turns=max_turns,
    )
    if privacy_aware:
        text += "\n\n" + load_prompt("baseline__privacy_aware.txt")
    return text


def parse_text_tool_call(text: str) -> tuple[str, dict]:
    """Best-effort parse of a tool call written as text. Returns ("invalid", {}) on failure."""
    text = _THINK.sub("", text or "")
    m = _TOOL_CALL.search(text)
    candidates = [m.group(1)] if m else []
    start, end = text.find("{"), text.rfind("}")
    if start != -1 and end > start:
        candidates.append(text[start:end + 1])
    for cand in candidates:
        try:
            obj = json.loads(cand)
        except json.JSONDecodeError:
            continue
        if not isinstance(obj, dict):
            continue
        name = obj.get("name") or obj.get("tool")
        args = obj.get("arguments", obj.get("parameters", {}))
        if isinstance(args, str):
            try:
                args = json.loads(args)
            except json.JSONDecodeError:
                continue
        if isinstance(name, str) and isinstance(args, dict):
            return name, args
    return "invalid", {}


class LLMAgent:
    def __init__(self, model: str, base_url: str | None = None, api_key: str | None = None,
                 policy: Policy | None = None, max_turns: int = MAX_TURNS, privacy_aware: bool = True,
                 temperature: float = 0.0, max_tokens: int = 1024, extra_body: dict | None = None,
                 client=None):
        if client is None:
            from openai import OpenAI          # imported here so the simulator has no hard dependency
            client = OpenAI(base_url=base_url, api_key=api_key or "EMPTY")
        self.client = client
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.extra_body = extra_body
        self.system = system_prompt(policy, max_turns, privacy_aware)
        self.usage = {"prompt_tokens": 0, "completion_tokens": 0, "calls": 0, "context_overflows": 0}

    def reset(self, opening: str) -> None:
        self.messages = [{"role": "system", "content": self.system},
                         {"role": "user", "content": opening}]
        self._call_id = None

    def act(self, observation: str | None) -> tuple[str, dict]:
        if observation is not None:
            if self._call_id is not None:
                self.messages.append({"role": "tool", "tool_call_id": self._call_id, "content": observation})
            else:
                self.messages.append({"role": "user", "content": f"Tool result: {observation}"})
        kwargs = dict(model=self.model, messages=self.messages, tools=openai_tools(),
                      tool_choice="auto", temperature=self.temperature, max_tokens=self.max_tokens)
        if self.extra_body:
            kwargs["extra_body"] = self.extra_body
        try:
            resp = self.client.chat.completions.create(**kwargs)
        except Exception as exc:                # openai.BadRequestError; not imported at module level
            if "context length" not in str(exc):
                raise
            # The dialogue no longer fits. End the episode as a failure instead of crashing
            # the run; the count is reported in the summary's token_usage.
            self.usage["context_overflows"] += 1
            return "finish", {"message": ""}
        usage = getattr(resp, "usage", None)
        if usage is not None:
            self.usage["prompt_tokens"] += getattr(usage, "prompt_tokens", 0) or 0
            self.usage["completion_tokens"] += getattr(usage, "completion_tokens", 0) or 0
        self.usage["calls"] += 1

        msg = resp.choices[0].message
        calls = getattr(msg, "tool_calls", None) or []
        if calls:
            call = calls[0]                     # one tool call per turn; extra calls are dropped
            self._call_id = call.id
            self.messages.append({"role": "assistant", "content": strip_thinking(msg.content),
                                  "tool_calls": [{"id": call.id, "type": "function",
                                                  "function": {"name": call.function.name,
                                                               "arguments": call.function.arguments}}]})
            try:
                args = json.loads(call.function.arguments or "{}")
            except json.JSONDecodeError:
                return "invalid", {}
            return call.function.name, args if isinstance(args, dict) else {}
        self._call_id = None
        self.messages.append({"role": "assistant", "content": strip_thinking(msg.content)})
        return parse_text_tool_call(msg.content or "")
