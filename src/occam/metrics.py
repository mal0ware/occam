"""Out-of-sample performance measures and tests from DeMiguel, Garlappi and Uppal, section 2."""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

Array = NDArray[np.float64]


def sharpe_ratio(excess_returns: Array) -> float:
    """Eq. (12): sample mean of excess returns over their sample standard deviation."""
    return float(excess_returns.mean() / excess_returns.std(ddof=1))


def certainty_equivalent(excess_returns: Array, gamma: float = 1.0) -> float:
    """Eq. (14): ``mean - (gamma / 2) * variance``."""
    return float(excess_returns.mean() - 0.5 * gamma * excess_returns.var(ddof=1))


def drifted_weights(w: Array, asset_returns: Array) -> Array:
    """Weights just before rebalancing: ``w_j (1 + R_j) / (1 + w'R)``.

    This is the ``w_{t+}`` of eq. (15). The paper does not write the formula out; this is the
    usual one, applied to the returns the dataset holds.
    """
    return np.asarray(w * (1.0 + asset_returns) / (1.0 + w @ asset_returns), dtype=np.float64)


def trade_size(w: Array, asset_returns: Array, w_next: Array) -> float:
    """One term of eq. (15): ``sum_j |w_{j,t+1} - w_{j,t+}|``."""
    return float(np.abs(w_next - drifted_weights(w, asset_returns)).sum())


def net_returns(gross: Array, trades: Array, cost: float = 0.005) -> Array:
    """Eq. (16): ``(1 + R_p) (1 - c * trade) - 1`` with proportional cost ``c``."""
    return np.asarray((1.0 + gross) * (1.0 - cost * trades) - 1.0, dtype=np.float64)


def return_loss(benchmark_net: Array, strategy_net: Array) -> float:
    """Eq. (17): extra monthly return the strategy needs to match the benchmark's Sharpe ratio."""
    sr_bench = benchmark_net.mean() / benchmark_net.std(ddof=1)
    return float(sr_bench * strategy_net.std(ddof=1) - strategy_net.mean())


@dataclass(frozen=True)
class TestResult:
    """A test statistic and its two-sided p-value.

    For the two tests in this module the p-value is ``2 (1 - Phi(|z|))``. No p-value in the
    paper's tables is above 0.50, so the paper's own figures look one-sided in the direction
    of the observed difference. Half of the two-sided value is the comparable number, and
    ``one_sided`` returns it.
    """

    __test__ = False  # not a pytest class

    statistic: float
    p_value: float

    @property
    def one_sided(self) -> float:
        """P-value against the alternative in the direction of the observed difference."""
        return 0.5 * self.p_value


def _two_sided(z: float) -> TestResult:
    return TestResult(statistic=z, p_value=math.erfc(abs(z) / math.sqrt(2.0)))


def sharpe_difference_test(r_i: Array, r_n: Array) -> TestResult:
    """Jobson-Korkie test with Memmel's correction, as written in the paper's footnote 16.

    ``z = (s_n m_i - s_i m_n) / sqrt(theta)`` with

    ``theta = (1/T) (2 s_i^2 s_n^2 - 2 s_i s_n s_in + m_i^2 s_n^2 / 2 + m_n^2 s_i^2 / 2
    - m_i m_n s_in^2 / (s_i s_n))``.

    Positive when the first series has the higher Sharpe ratio.
    """
    t = r_i.shape[0]
    m_i, m_n = float(r_i.mean()), float(r_n.mean())
    cov = np.cov(r_i, r_n)
    s_i, s_n, s_in = math.sqrt(cov[0, 0]), math.sqrt(cov[1, 1]), float(cov[0, 1])
    theta = (
        2.0 * s_i**2 * s_n**2
        - 2.0 * s_i * s_n * s_in
        + 0.5 * m_i**2 * s_n**2
        + 0.5 * m_n**2 * s_i**2
        - m_i * m_n / (s_i * s_n) * s_in**2
    ) / t
    num = s_n * m_i - s_i * m_n
    if theta <= 0.0 or num == 0.0:
        return TestResult(statistic=0.0, p_value=1.0)
    return _two_sided(num / math.sqrt(theta))


def ceq_difference_test(r_i: Array, r_n: Array, gamma: float = 1.0) -> TestResult:
    """Delta-method test of a difference in certainty equivalents, the paper's footnote 18.

    With ``v = (m_i, m_n, s_i^2, s_n^2)`` and ``f(v) = (m_i - gamma s_i^2 / 2) - (m_n - gamma
    s_n^2 / 2)``, ``sqrt(T) (f(v_hat) - f(v))`` is asymptotically normal with variance
    ``grad' Theta grad``, where ``Theta`` has blocks ``[[s_i^2, s_in], [s_in, s_n^2]]`` and
    ``[[2 s_i^4, 2 s_in^2], [2 s_in^2, 2 s_n^4]]``.
    """
    t = r_i.shape[0]
    cov = np.cov(r_i, r_n)
    v_i, v_n, c_in = float(cov[0, 0]), float(cov[1, 1]), float(cov[0, 1])
    theta = np.array(
        [
            [v_i, c_in, 0.0, 0.0],
            [c_in, v_n, 0.0, 0.0],
            [0.0, 0.0, 2.0 * v_i**2, 2.0 * c_in**2],
            [0.0, 0.0, 2.0 * c_in**2, 2.0 * v_n**2],
        ]
    )
    grad = np.array([1.0, -1.0, -0.5 * gamma, 0.5 * gamma])
    diff = (float(r_i.mean()) - 0.5 * gamma * v_i) - (float(r_n.mean()) - 0.5 * gamma * v_n)
    var = float(grad @ theta @ grad) / t
    if var <= 0.0 or diff == 0.0:
        return TestResult(statistic=0.0, p_value=1.0)
    return _two_sided(diff / math.sqrt(var))
