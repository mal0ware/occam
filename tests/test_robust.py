"""The Ledoit-Wolf (2008) Sharpe tests: known answers first, then size and power."""

from __future__ import annotations

import math

import numpy as np
import pytest

from occam import metrics as met
from occam import pipeline, robust, size


def pair(seed: int, months: int = 300) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    a = rng.normal(0.010, 0.05, months)
    b = 0.6 * a + rng.normal(0.002, 0.03, months)
    return a, b


def test_difference_is_the_difference_of_population_sharpe_ratios() -> None:
    a, b = pair(1)
    delta, _, y = robust.sharpe_difference(a, b)
    assert np.isclose(delta, a.mean() / a.std(ddof=0) - b.mean() / b.std(ddof=0))
    assert y.shape == (300, 4)
    assert np.allclose(y.mean(axis=0), 0.0)


def test_gradient_matches_finite_differences() -> None:
    a, b = pair(2)
    _, grad, _ = robust.sharpe_difference(a, b)
    a, b = a / a.std(), b / b.std()  # the function works on series with unit variance
    v = np.array([a.mean(), b.mean(), (a**2).mean(), (b**2).mean()])

    def f(v: np.ndarray) -> float:
        return float(v[0] / math.sqrt(v[2] - v[0] ** 2) - v[1] / math.sqrt(v[3] - v[1] ** 2))

    for k in range(4):
        step = np.zeros(4)
        step[k] = 1e-7 * max(abs(v[k]), 1e-3)
        numeric = (f(v + step) - f(v - step)) / (2 * step[k])
        assert np.isclose(grad[k], numeric, rtol=1e-5)


def test_parzen_kernel_values() -> None:
    assert robust.parzen(0.0) == 1.0
    assert np.isclose(robust.parzen(0.25), 1 - 6 / 16 + 6 / 64)
    assert np.isclose(robust.parzen(0.5), 0.25)  # both branches agree at one half
    assert np.isclose(robust.parzen(-0.75), 2 * 0.25**3)
    assert robust.parzen(1.0) == 0.0 and robust.parzen(3.0) == 0.0


def test_hac_covariance_written_out_with_loops() -> None:
    rng = np.random.default_rng(3)
    y = rng.standard_normal((40, 4))
    y -= y.mean(axis=0)
    t, s = 40, 3.5
    expected = np.zeros((4, 4))
    for j in range(-t + 1, t):
        gamma = np.zeros((4, 4))
        for k in range(t):
            if 0 <= k - j < t:
                gamma += np.outer(y[k], y[k - j])
        expected += robust.parzen(j / s) * gamma / t
    expected *= t / (t - 4)
    assert np.allclose(robust.hac_covariance(y, bandwidth=s), expected)
    lag0 = y.T @ y / t * t / (t - 4)
    assert np.allclose(robust.hac_covariance(y, bandwidth=0.0), lag0)


def test_andrews_bandwidth_is_the_formula_for_one_ar1_column() -> None:
    rng = np.random.default_rng(4)
    x = np.empty(500)
    x[0] = 0.0
    for k in range(1, 500):
        x[k] = 0.5 * x[k - 1] + rng.standard_normal()
    lag, lead = x[:-1], x[1:]
    rho = lag @ lead / (lag @ lag)
    alpha = 4 * rho**2 / (1 - rho) ** 4  # the innovation variance cancels with one column
    assert np.isclose(robust.andrews_bandwidth(x[:, None]), 2.6614 * (alpha * 500) ** 0.2)


def test_hac_test_agrees_with_jobson_korkie_on_long_iid_normal_series() -> None:
    # With independent normal returns both statistics estimate the same asymptotic variance.
    a, b = pair(5, months=20000)
    jk = met.sharpe_difference_test(a, b)
    hac = robust.hac_sharpe_test(a, b)
    assert np.isclose(hac.statistic, jk.statistic, rtol=0.05)


def test_tests_ignore_leverage_and_flip_sign_with_the_order() -> None:
    a, b = pair(6)
    z = robust.hac_sharpe_test(a, b).statistic
    assert np.isclose(z, robust.hac_sharpe_test(3.0 * a, 0.2 * b).statistic)
    assert np.isclose(z, -robust.hac_sharpe_test(b, a).statistic)


def test_identical_series_are_not_rejected() -> None:
    a, _ = pair(7)
    rng = np.random.default_rng(0)
    assert robust.hac_sharpe_test(a, a.copy()).p_value == 1.0
    assert robust.bootstrap_sharpe_test(a, a.copy(), rng).p_value == 1.0


def test_bootstrap_is_reproducible_and_bounded() -> None:
    a, b = pair(8)
    one = robust.bootstrap_sharpe_test(a, b, np.random.default_rng(11), block=6, draws=199)
    two = robust.bootstrap_sharpe_test(a, b, np.random.default_rng(11), block=6, draws=199)
    assert one == two
    assert 1 / 200 <= one.p_value <= 1.0
    assert np.isclose(one.statistic, robust.hac_sharpe_test(a, b).statistic)


def test_bootstrap_rejects_a_large_true_difference() -> None:
    rng = np.random.default_rng(9)
    common = rng.normal(0.0, 0.04, 360)
    a = 0.012 + common + rng.normal(0.0, 0.01, 360)  # Sharpe ratio near 0.29
    b = 0.000 + common + rng.normal(0.0, 0.01, 360)  # Sharpe ratio near 0
    result = robust.bootstrap_sharpe_test(a, b, rng, block=6, draws=999)
    assert result.statistic > 0 and result.p_value < 0.01


@pytest.mark.parametrize("process", ["iid normal", "iid t(5)"])
def test_bootstrap_and_hac_hold_their_size_under_a_true_null(process: str) -> None:
    # 400 simulations: the standard error of a 5% rejection rate is 0.011, so the band
    # below is about three standard errors either side.
    row = size.rejection_rates(process, months=240, simulations=400, seed=12, draws=199)
    assert 0.015 < row.bootstrap < 0.085
    assert 0.015 < row.hac < 0.085


def test_serial_correlation_inflates_jobson_korkie_more_than_the_bootstrap() -> None:
    row = size.rejection_rates(
        "AR(1) 0.2 with GARCH(1,1), t(5) shocks", months=240, simulations=300, seed=13, draws=199
    )
    assert row.jk > 0.08  # the test the paper uses rejects a true null too often here
    assert row.bootstrap < row.jk


def test_simulated_series_share_a_sharpe_ratio() -> None:
    rng = np.random.default_rng(14)
    for process in size.PROCESSES:
        x = size.simulate(process, 200_000, rng)
        sr = x.mean(axis=0) / x.std(axis=0)
        assert abs(sr[0] - sr[1]) < 0.01, process


def test_holm_step_down() -> None:
    # Thresholds for four hypotheses are 0.0125, 0.0167, 0.025 and 0.05.
    assert pipeline.holm([0.001, 0.04, 0.012, 0.03]) == [True, False, True, False]
    assert pipeline.holm([0.02, 0.02, 0.02, 0.02]) == [False] * 4
    assert pipeline.holm([0.01, 0.011, 0.012, 0.04]) == [True] * 4
