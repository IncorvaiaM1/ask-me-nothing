"""Order sizing: turns goal parameters into a list of policy-compliant orders.

This is the arithmetic the prompted and fine-tuned models could not do. Moving it
into a tool keeps the research question about information: every input is something
the agent has to have obtained (the amount, the protected holding, the risk
tolerance, the tax bracket), and a guessed input produces orders that miss the goal.
The tool never places orders and never checks suitability; ``place_order`` does.
"""

from __future__ import annotations

import math
import re

from brokerdesk.market import UNIVERSE
from brokerdesk.policy import Policy, book_value

_PCT = re.compile(r"(\d+(?:\.\d+)?)\s*%?")
_SELL_MARGIN = 1.03        # aim 3% over a cash target so price drift cannot leave it short
_ORDER_FRAC = 0.95         # stay 5% under the order cap
_SLACK = 0.015             # stay 1.5 points under the position limit on buys


def parse_bracket(value) -> float | None:
    if value is None:
        return None
    if isinstance(value, (int, float)):
        return float(value)
    m = _PCT.search(str(value))
    return float(m.group(1)) if m else None


def _sell_to_raise(candidates: list[tuple[str, int, float]], target: float, cap: float) -> list[dict]:
    """Sell from the largest holdings first until ``target`` dollars are covered."""
    orders, raised = [], 0.0
    for ticker, qty, price in sorted(candidates, key=lambda c: -c[1] * c[2]):
        while raised < target and qty > 0:
            n = min(qty, math.ceil((target - raised) / price), int(cap / price))
            if n <= 0:
                break
            orders.append({"side": "sell", "ticker": ticker, "quantity": n})
            raised += n * price
            qty -= n
        if raised >= target:
            break
    return orders


def plan_raise_cash(positions: dict[str, int], prices: dict[str, float], amount: float,
                    policy: Policy, keep=(), tax_bracket=None, cost_basis=None) -> tuple[list[dict], list[str]]:
    notes = []
    keep = set(keep or [])
    candidates = [(t, q, prices[t]) for t, q in positions.items() if q > 0 and t not in keep]
    bracket = parse_bracket(tax_bracket)
    if bracket is not None:
        if cost_basis is None:
            raise ValueError("cost basis unavailable")
        losses = bracket >= policy.tax_high_bracket_pct
        candidates = [c for c in candidates if (c[2] < cost_basis[c[0]]) == losses]
        notes.append(f"tax rule applied for a {bracket:g}% bracket: selling only holdings with unrealized "
                     f"{'losses' if losses else 'gains'}")
    orders = _sell_to_raise(candidates, amount * _SELL_MARGIN, policy.max_order_notional * _ORDER_FRAC)
    if sum(o["quantity"] * prices[o["ticker"]] for o in orders) < amount:
        notes.append("not enough eligible holdings to cover the amount")
    return orders, notes


def plan_rebalance(positions: dict[str, int], cash: float, prices: dict[str, float], target_equity: float,
                   policy: Policy) -> tuple[list[dict], list[str]]:
    book = book_value(positions, cash, prices)
    value = {t: q * prices[t] for t, q in positions.items()}
    equity = sum(v for t, v in value.items() if UNIVERSE[t] == "equity")
    delta = target_equity * book - equity
    buy_class, sell_class = ("equity", "bond") if delta > 0 else ("bond", "equity")
    cap = policy.max_order_notional * _ORDER_FRAC
    orders, notes = [], []

    remaining = abs(delta)
    for t in sorted((t for t in positions if UNIVERSE[t] == sell_class and positions[t] > 0),
                    key=lambda t: -value[t]):
        sellable = value[t] * 0.95                  # leave a sliver of every holding
        while remaining >= 0.002 * book and sellable > prices[t]:
            n = int(min(remaining, sellable, cap) / prices[t])
            if n <= 0:
                break
            orders.append({"side": "sell", "ticker": t, "quantity": n})
            remaining -= n * prices[t]
            sellable -= n * prices[t]
            value[t] -= n * prices[t]
        if remaining < 0.002 * book:
            break

    budget = abs(delta) - remaining + cash - policy.max_cash_frac * book * 0.5
    budget = min(budget, abs(delta) - remaining + cash)
    for t in sorted((t for t in positions if UNIVERSE[t] == buy_class and t not in policy.restricted),
                    key=lambda t: value[t]):
        room = (policy.max_position_frac - _SLACK) * book - value[t]
        while budget >= 0.002 * book and room > prices[t]:
            n = int(min(budget, room, cap) / prices[t])
            if n <= 0:
                break
            orders.append({"side": "buy", "ticker": t, "quantity": n})
            budget -= n * prices[t]
            room -= n * prices[t]
            value[t] += n * prices[t]
    if remaining >= 0.002 * book:
        notes.append("could not sell enough within the order and holding limits")
    return orders, notes
