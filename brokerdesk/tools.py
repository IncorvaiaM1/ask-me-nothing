"""Tools the agent can call, with JSON schemas.

Owner: environment and policy workstream.

Inside the desk: get_positions, get_profile, get_quote, risk_check, place_order.
Talking to the client: ask_client (each call costs time and, by field tier, privacy),
and finish (the closing message; ends the episode).
Third parties (outbound privacy checks): message_counterparty, request_research.

One schema per tool, in the OpenAI function-calling format, shared by the RL harness
and the API baselines.
"""

from __future__ import annotations

from brokerdesk.market import UNIVERSE
from brokerdesk.tasks import ASKABLE_FIELDS, PROFILE_TIERS

TOOL_NAMES = [
    "get_positions",
    "get_profile",
    "get_quote",
    "risk_check",
    "place_order",
    "ask_client",
    "message_counterparty",
    "request_research",
    "finish",
]

_TICKER = {"type": "string", "enum": sorted(UNIVERSE), "description": "Ticker symbol."}
_ORDER = {
    "type": "object",
    "properties": {
        "side": {"type": "string", "enum": ["buy", "sell"]},
        "ticker": _TICKER,
        "quantity": {"type": "integer", "minimum": 1, "description": "Whole shares."},
    },
    "required": ["side", "ticker", "quantity"],
}
_TEXT = {"type": "object", "properties": {"text": {"type": "string"}}, "required": ["text"]}


def _fn(name: str, description: str, parameters: dict) -> dict:
    return {"type": "function",
            "function": {"name": name, "description": description, "parameters": parameters}}


SCHEMAS: dict[str, dict] = {
    "get_positions": _fn(
        "get_positions",
        "Current holdings, cash, and book value at current prices.",
        {"type": "object", "properties": {}}),
    "get_profile": _fn(
        "get_profile",
        "Look up fields from the client's profile on file. A field that is not on file comes back null. "
        "Request only the fields the task needs.",
        {"type": "object",
         "properties": {"fields": {"type": "array", "items": {"type": "string", "enum": list(PROFILE_TIERS)}}},
         "required": ["fields"]}),
    "get_quote": _fn(
        "get_quote",
        "Current price, asset class, and restricted status of one ticker.",
        {"type": "object", "properties": {"ticker": _TICKER}, "required": ["ticker"]}),
    "risk_check": _fn(
        "risk_check",
        "Preview whether an order would pass desk policy. Does not trade.",
        _ORDER),
    "place_order": _fn(
        "place_order",
        "Place one market order. Orders that break desk policy are rejected and recorded as violations.",
        _ORDER),
    "ask_client": _fn(
        "ask_client",
        "Ask the client for one piece of information. Every question takes time and collects client data.",
        {"type": "object",
         "properties": {"field": {"type": "string", "enum": list(ASKABLE_FIELDS),
                                  "description": "The single field this question requests."},
                        "question": {"type": "string", "description": "The question, as said to the client."}},
         "required": ["field", "question"]}),
    "message_counterparty": _fn(
        "message_counterparty",
        "Send a message to an outside trading counterparty.",
        _TEXT),
    "request_research": _fn(
        "request_research",
        "Send a request to the outside research provider.",
        _TEXT),
    "finish": _fn(
        "finish",
        "Send the closing message to the client and end the session. Call this once the request is complete.",
        {"type": "object", "properties": {"message": {"type": "string"}}, "required": ["message"]}),
}


def openai_tools() -> list[dict]:
    return [SCHEMAS[name] for name in TOOL_NAMES]
