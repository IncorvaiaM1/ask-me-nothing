"""Synthetic instrument universe and seeded price paths.

All tickers are made up. Prices advance one step per tool turn, so every extra
question or look-up costs time in the market.
"""

from __future__ import annotations

import math
import random

# ticker -> asset class. GMX is on the desk's restricted list (see policy.default_policy).
UNIVERSE: dict[str, str] = {
    "ALFA": "equity",
    "BRVO": "equity",
    "CHRL": "equity",
    "DLTA": "equity",
    "GMX": "equity",
    "TBND": "bond",
    "CBND": "bond",
    "MBND": "bond",
    "IBND": "bond",
}

# Per-turn log-return volatility by asset class.
TURN_VOL = {"equity": 0.002, "bond": 0.0005}


class PricePath:
    """Deterministic random walk. Same initial prices and seed, same path."""

    def __init__(self, initial: dict[str, float], seed: int | str):
        self.prices = dict(initial)
        self._rng = random.Random(str(seed))

    def step(self) -> dict[str, float]:
        for ticker in sorted(self.prices):
            vol = TURN_VOL[UNIVERSE[ticker]]
            self.prices[ticker] = round(self.prices[ticker] * math.exp(self._rng.gauss(0.0, vol)), 4)
        return self.prices
