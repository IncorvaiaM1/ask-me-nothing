"""Tools the agent can call, with JSON schemas.

Owner: environment and policy workstream.

Inside the desk: get_positions, get_profile, get_quote, risk_check, place_order.
Talking to the client: ask_client (each call costs time and, by field tier, privacy).
Third parties (outbound privacy checks): message_counterparty, request_research.
"""

TOOL_NAMES = [
    "get_positions",
    "get_profile",
    "get_quote",
    "risk_check",
    "place_order",
    "ask_client",
    "message_counterparty",
    "request_research",
]

# TODO: one JSON schema per tool, shared by the RL harness and the API baselines.
SCHEMAS: dict[str, dict] = {}
