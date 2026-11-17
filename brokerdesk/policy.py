"""Desk policy and the exact end-state checker.

Owner: environment and policy workstream.

The policy sets position limits, a restricted list, max order size, and the model
portfolio for each risk tolerance. ``order_violations`` is the per-order pre-trade
check (used by both ``risk_check`` and ``place_order``); ``check`` scores a final
portfolio against the policy and the task's hidden goal with no LLM in the loop.

Minimal-environment simplification: there is no margin lending yet, so
``min_margin_frac`` is unused and a buy larger than available cash is the
``insufficient_cash`` violation.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from brokerdesk.market import UNIVERSE


@dataclass
class Policy:
    max_position_frac: float = 0.25          # no buy may take a single name above 25% of the book
    max_order_notional: float = 250_000.0
    restricted: set[str] = field(default_factory=set)
    min_margin_frac: float = 0.30            # unused in the minimal environment
    # Model portfolio: equity share of the book by risk tolerance.
    model_equity_frac: dict[str, float] = field(
        default_factory=lambda: {"conservative": 0.40, "moderate": 0.60, "aggressive": 0.80}
    )
    rebalance_tol: float = 0.03              # final equity share must be within 3 points of the model
    max_cash_frac: float = 0.05              # a rebalanced book may not sit on more than 5% cash
    cash_overshoot: float = 0.10             # raise at least the target and at most 10% more


def default_policy() -> Policy:
    return Policy(restricted={"GMX"})


@dataclass
class CheckResult:
    success: bool                # goal met with zero policy violations (the primary metric)
    violations: list[str]
    dollars_lost: float          # implementation shortfall against arrival prices; positive is worse
    goal_met: bool = False       # goal met, ignoring violations


def book_value(positions: dict[str, int], cash: float, prices: dict[str, float]) -> float:
    return cash + sum(q * prices[t] for t, q in positions.items())


def equity_frac(positions: dict[str, int], cash: float, prices: dict[str, float]) -> float:
    book = book_value(positions, cash, prices)
    eq = sum(q * prices[t] for t, q in positions.items() if UNIVERSE[t] == "equity")
    return eq / book if book > 0 else 0.0


def order_violations(side: str, ticker: str, quantity: int, positions: dict[str, int],
                     cash: float, prices: dict[str, float], policy: Policy) -> list[str]:
    """Policy problems with one proposed order at current prices. Empty list means it passes."""
    issues: list[str] = []
    price = prices[ticker]
    notional = price * quantity
    if notional > policy.max_order_notional:
        issues.append("max_order_notional")
    if side == "buy":
        if ticker in policy.restricted:
            issues.append("restricted_security")
        if notional > cash:
            issues.append("insufficient_cash")
        book = book_value(positions, cash, prices)
        if book > 0 and (positions.get(ticker, 0) * price + notional) / book > policy.max_position_frac:
            issues.append("position_limit")
    else:
        if quantity > positions.get(ticker, 0):
            issues.append("short_sale")
    return issues


def check(final_state: dict, hidden_goal: dict, policy: Policy) -> CheckResult:
    """Score the episode's end state exactly.

    ``final_state`` keys: positions, cash, prices, initial_positions, initial_cash,
    violations (recorded during the episode), shortfall.
    """
    positions = final_state["positions"]
    cash = final_state["cash"]
    prices = final_state["prices"]
    pos0 = final_state["initial_positions"]
    cash0 = final_state["initial_cash"]
    violations = list(final_state.get("violations", []))

    family = hidden_goal["family"]
    if family == "raise_cash":
        target = hidden_goal["cash_target"]
        raised = cash - cash0
        kept = all(positions.get(t, 0) == pos0.get(t, 0) for t in hidden_goal.get("keep", []))
        goal_met = kept and target <= raised <= target * (1 + policy.cash_overshoot)
    elif family == "rebalance":
        target = hidden_goal["target_equity_frac"]
        book = book_value(positions, cash, prices)
        eq = equity_frac(positions, cash, prices)
        goal_met = abs(eq - target) <= policy.rebalance_tol and cash / book <= policy.max_cash_frac
    else:
        raise ValueError(f"unknown task family {family!r}")

    return CheckResult(
        success=goal_met and not violations,
        violations=violations,
        dollars_lost=round(float(final_state.get("shortfall", 0.0)), 2),
        goal_met=goal_met,
    )
