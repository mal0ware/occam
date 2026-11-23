"""The rolling-sample test of DeMiguel, Garlappi and Uppal, section 2.

At the end of each month the previous ``M`` months of excess returns are the only input to
a rule. The rule returns weights, and those weights earn the next month's returns. The
window then moves forward one month. A series of ``T`` months gives ``T - M`` out-of-sample
returns.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from occam import estimators as est
from occam import metrics as met
from occam import optimizers as opt

Array = NDArray[np.float64]

#: A rule maps (window of returns with shape (M, N), index of the market asset) to N weights.
Rule = Callable[[Array, int], Array]


def _ew(window: Array, market: int) -> Array:
    return opt.equal_weight(window.shape[1])


def _mv(window: Array, market: int) -> Array:
    return opt.mean_variance(est.sample_mean(window), est.sample_cov(window))


def _min(window: Array, market: int) -> Array:
    return opt.min_variance(est.sample_cov(window))


def _min_c(window: Array, market: int) -> Array:
    return opt.min_variance_long_only(est.sample_cov(window))


def _vw(window: Array, market: int) -> Array:
    w = np.zeros(window.shape[1])
    w[market] = 1.0
    return w


def _lw_min(window: Array, market: int) -> Array:
    return opt.min_variance(est.ledoit_wolf(window).covariance)


def _erc(window: Array, market: int) -> Array:
    return opt.equal_risk_contribution(est.sample_cov(window))


def _hrp(window: Array, market: int) -> Array:
    return opt.hierarchical_risk_parity(est.sample_cov(window))


def _bl(window: Array, market: int) -> Array:
    return opt.black_litterman_neutral(est.sample_cov(window), _vw(window, market))


#: Rules in table order. The first five are rows of the paper's tables; the last four are added.
RULES: dict[str, Rule] = {
    "ew": _ew,
    "mv": _mv,
    "min": _min,
    "min-c": _min_c,
    "vw": _vw,
    "lw-min": _lw_min,
    "erc": _erc,
    "hrp": _hrp,
    "bl": _bl,
}

PAPER_RULES = ("ew", "mv", "min", "min-c", "vw")
ADDED_RULES = ("lw-min", "erc", "hrp", "bl")

DESCRIPTIONS = {
    "ew": "1/N, rebalanced monthly",
    "mv": "sample mean-variance",
    "min": "minimum variance",
    "min-c": "minimum variance, no short sales",
    "vw": "the market portfolio",
    "lw-min": "minimum variance on the Ledoit-Wolf covariance",
    "erc": "risk parity (equal risk contribution)",
    "hrp": "hierarchical risk parity",
    "bl": "Black-Litterman with a neutral view",
}


@dataclass(frozen=True)
class Backtest:
    """Output of one rule on one dataset."""

    weights: Array  # (T - M + 1, N): the last row is the portfolio held after the sample ends
    returns: Array  # (T - M,) out-of-sample excess returns before costs
    trades: Array  # (T - M,) the terms of eq. (15)

    @property
    def turnover(self) -> float:
        """Eq. (15): average absolute trade per month, as a fraction of wealth."""
        return float(self.trades.mean())


def rolling_backtest(returns: Array, window: int, rule: Rule, market: int) -> Backtest:
    """Run one rule over a ``(T, N)`` array of excess returns with estimation window ``M``.

    Weights with index ``k`` use rows ``k`` to ``k + M - 1`` and earn row ``k + M``. The rule
    never sees the row it is evaluated on. The trade at the end of month ``k + M`` is the
    distance between the next weights and the current ones after they drift with that
    month's returns.
    """
    t, n = returns.shape
    if t <= window:
        raise ValueError(f"need more than {window} months, got {t}")
    weights = np.empty((t - window + 1, n))
    for k in range(t - window + 1):
        weights[k] = rule(returns[k : k + window], market)
    realised = returns[window:]
    port = np.einsum("ij,ij->i", weights[:-1], realised)
    trades = np.array(
        [met.trade_size(weights[k], realised[k], weights[k + 1]) for k in range(t - window)]
    )
    return Backtest(weights=weights, returns=port, trades=trades)


def in_sample_mean_variance(returns: Array) -> Array:
    """Returns of the mean-variance portfolio estimated on the whole sample (eq. 13, ``M = T``)."""
    w = opt.mean_variance(est.sample_mean(returns), est.sample_cov(returns))
    return np.asarray(returns @ w, dtype=np.float64)
