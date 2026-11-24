"""Sharpe-difference tests that allow fat tails and serial dependence.

The method is Ledoit and Wolf (2008), "Robust performance hypothesis testing with the Sharpe
ratio", Journal of Empirical Finance 15(5). Two tests of ``H0: SR_i = SR_n`` are built on the
same delta-method standard error:

* ``hac_sharpe_test``: a normal test with a kernel (HAC) estimate of the long-run covariance.
* ``bootstrap_sharpe_test``: a studentised circular block bootstrap, which the authors
  recommend for sample sizes like the ones here.

Write ``v = (mu_i, mu_n, g_i, g_n)`` with ``g = E[r^2]``. The difference in Sharpe ratios is
``f(v) = mu_i / sqrt(g_i - mu_i^2) - mu_n / sqrt(g_n - mu_n^2)`` and its standard error is
``sqrt(grad' Psi grad / T)``, where ``Psi`` is the long-run covariance of
``y_t = (r_ti - mu_i, r_tn - mu_n, r_ti^2 - g_i, r_tn^2 - g_n)``.

One choice here is simpler than the paper's. The kernel estimate uses the Parzen kernel with
the automatic bandwidth of Andrews (1991) and no prewhitening; the authors prewhiten.
PAPER.md lists this with the other differences.
"""

from __future__ import annotations

import math

import numpy as np
from numpy.typing import NDArray

from occam.metrics import TestResult

Array = NDArray[np.float64]

#: Block lengths, in months, that the bootstrap is reported for. The middle one is the default.
BLOCKS = (1, 3, 6, 10)
BLOCK = 6
DRAWS = 4999


def sharpe_difference(r_i: Array, r_n: Array) -> tuple[float, Array, Array]:
    """The estimate ``f(v_hat)``, its gradient, and the ``(T, 4)`` array of ``y_t``.

    Each series is first divided by its own standard deviation. A Sharpe ratio does not change
    with leverage, and after this step neither does the automatic bandwidth, which otherwise
    weights the four columns of ``y`` by their scale.
    """
    r_i = r_i / r_i.std()
    r_n = r_n / r_n.std()
    mu_i, mu_n = float(r_i.mean()), float(r_n.mean())
    g_i, g_n = float((r_i**2).mean()), float((r_n**2).mean())
    v_i, v_n = g_i - mu_i**2, g_n - mu_n**2
    delta = mu_i / math.sqrt(v_i) - mu_n / math.sqrt(v_n)
    grad = np.array(
        [g_i / v_i**1.5, -g_n / v_n**1.5, -0.5 * mu_i / v_i**1.5, 0.5 * mu_n / v_n**1.5]
    )
    y = np.column_stack([r_i - mu_i, r_n - mu_n, r_i**2 - g_i, r_n**2 - g_n])
    return delta, grad, y


def parzen(x: float) -> float:
    """The Parzen kernel."""
    a = abs(x)
    if a <= 0.5:
        return 1.0 - 6.0 * a**2 + 6.0 * a**3
    if a <= 1.0:
        return 2.0 * (1.0 - a) ** 3
    return 0.0


def andrews_bandwidth(y: Array) -> float:
    """Andrews (1991) automatic bandwidth for the Parzen kernel, from AR(1) fits to each column.

    ``S = 2.6614 (alpha T)^(1/5)`` with ``alpha = sum 4 rho^2 s^4 / (1 - rho)^8 / sum s^4 /
    (1 - rho)^4``, where ``rho`` and ``s^2`` are the AR(1) coefficient and innovation variance
    of a column and every column has weight one.
    """
    t = y.shape[0]
    num = den = 0.0
    for col in y.T:
        lag, lead = col[:-1], col[1:]
        rho = float(lag @ lead / (lag @ lag))
        rho = min(max(rho, -0.97), 0.97)
        s2 = float(((lead - rho * lag) ** 2).mean())
        num += 4.0 * rho**2 * s2**2 / (1.0 - rho) ** 8
        den += s2**2 / (1.0 - rho) ** 4
    return float(2.6614 * (num / den * t) ** 0.2)


def hac_covariance(y: Array, bandwidth: float | None = None) -> Array:
    """Parzen-kernel estimate of the long-run covariance of the mean-zero columns of ``y``.

    ``Psi = T / (T - 4) sum_j k(j / S) Gamma(j)`` with ``Gamma(j) = (1/T) sum_t y_t y_{t-j}'``.
    A bandwidth of zero keeps only ``Gamma(0)``.
    """
    t = y.shape[0]
    s = andrews_bandwidth(y) if bandwidth is None else bandwidth
    psi = y.T @ y / t
    j = 1
    while s > 0.0 and j < t and j / s < 1.0:
        gamma = y[j:].T @ y[:-j] / t
        psi = psi + parzen(j / s) * (gamma + gamma.T)
        j += 1
    return np.asarray(psi * t / (t - 4), dtype=np.float64)


def _two_sided_normal(z: float) -> float:
    return math.erfc(abs(z) / math.sqrt(2.0))


def hac_standard_error(r_i: Array, r_n: Array, bandwidth: float | None = None) -> float:
    """Standard error of the Sharpe difference with the kernel estimate of ``Psi``."""
    _, grad, y = sharpe_difference(r_i, r_n)
    return math.sqrt(max(float(grad @ hac_covariance(y, bandwidth) @ grad), 0.0) / y.shape[0])


def hac_sharpe_test(r_i: Array, r_n: Array, bandwidth: float | None = None) -> TestResult:
    """Two-sided normal test of equal Sharpe ratios with a HAC standard error.

    The statistic is positive when the first series has the higher Sharpe ratio.
    """
    if np.array_equal(r_i, r_n):
        return TestResult(statistic=0.0, p_value=1.0)
    delta, _, _ = sharpe_difference(r_i, r_n)
    se = hac_standard_error(r_i, r_n, bandwidth)
    if se <= 0.0:
        return TestResult(statistic=0.0, p_value=1.0)
    z = delta / se
    return TestResult(statistic=z, p_value=_two_sided_normal(z))


def bootstrap_sharpe_test(
    r_i: Array,
    r_n: Array,
    rng: np.random.Generator,
    block: int = BLOCK,
    draws: int = DRAWS,
) -> TestResult:
    """Two-sided studentised circular block bootstrap test of equal Sharpe ratios.

    The observed statistic is ``d = |delta| / se`` with the HAC standard error. Each bootstrap
    sample joins ``l = floor(T / block)`` blocks of ``block`` consecutive pairs, wrapping round
    the end of the sample, and gives ``d* = |delta* - delta| / se*``. Its ``se*`` uses
    ``Psi* = (1/l) sum_j z_j z_j'``, where ``z_j`` is the sum of ``y*`` over block ``j``
    divided by ``sqrt(block)``. The p-value is ``(#{d* >= d} + 1) / (draws + 1)``.

    The returned statistic is the signed ``delta / se``, positive when the first series has
    the higher Sharpe ratio.
    """
    if np.array_equal(r_i, r_n):
        return TestResult(statistic=0.0, p_value=1.0)
    t = r_i.shape[0]
    delta, _, _ = sharpe_difference(r_i, r_n)
    se = hac_standard_error(r_i, r_n)
    if se <= 0.0:
        return TestResult(statistic=0.0, p_value=1.0)
    d = abs(delta) / se

    n_blocks = t // block
    length = n_blocks * block
    starts = rng.integers(0, t, size=(draws, n_blocks))
    index = ((starts[:, :, None] + np.arange(block)[None, None, :]) % t).reshape(draws, length)
    x = np.column_stack([r_i, r_n])[index]  # (draws, length, 2)
    mu = x.mean(axis=1)
    g = (x**2).mean(axis=1)
    var = g - mu**2
    delta_star = mu[:, 0] / np.sqrt(var[:, 0]) - mu[:, 1] / np.sqrt(var[:, 1])
    grad = np.column_stack(
        [
            g[:, 0] / var[:, 0] ** 1.5,
            -g[:, 1] / var[:, 1] ** 1.5,
            -0.5 * mu[:, 0] / var[:, 0] ** 1.5,
            0.5 * mu[:, 1] / var[:, 1] ** 1.5,
        ]
    )
    y = np.concatenate([x - mu[:, None, :], x**2 - g[:, None, :]], axis=2)  # (draws, length, 4)
    zeta = y.reshape(draws, n_blocks, block, 4).sum(axis=2) / math.sqrt(block)
    psi = np.einsum("mlj,mlk->mjk", zeta, zeta) / n_blocks
    se_star = np.sqrt(np.einsum("mj,mjk,mk->m", grad, psi, grad) / length)
    d_star = np.abs(delta_star - delta) / se_star
    p = (int(np.count_nonzero(d_star >= d)) + 1) / (draws + 1)
    return TestResult(statistic=delta / se, p_value=float(p))
