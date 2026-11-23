"""Performance measures and the two significance tests of DeMiguel, Garlappi and Uppal."""

from __future__ import annotations

import math

import numpy as np

from occam import metrics as met


def test_sharpe_and_ceq_hand_solved() -> None:
    r = np.array([0.01, 0.03, -0.01, 0.05])
    mean, var = 0.02, (0.01**2 + 0.01**2 + 0.03**2 + 0.03**2) / 3
    assert np.isclose(met.sharpe_ratio(r), mean / math.sqrt(var))  # eq. (12)
    assert np.isclose(met.certainty_equivalent(r, gamma=1.0), mean - var / 2)  # eq. (14)
    assert np.isclose(met.certainty_equivalent(r, gamma=4.0), mean - 2 * var)


def test_drifted_weights_hand_solved() -> None:
    w = np.array([0.5, 0.5])
    r = np.array([0.10, 0.0])
    assert np.allclose(met.drifted_weights(w, r), [0.55 / 1.05, 0.50 / 1.05])


def test_trade_size_of_rebalancing_to_equal_weight() -> None:
    # |0.5 - 0.5238| + |0.5 - 0.4762| = 0.047619
    w = np.array([0.5, 0.5])
    trade = met.trade_size(w, np.array([0.10, 0.0]), w)
    assert np.isclose(trade, 2 * (0.55 / 1.05 - 0.5))


def test_no_trade_when_all_assets_move_together() -> None:
    w = np.array([0.2, 0.3, 0.5])
    assert np.isclose(met.trade_size(w, np.full(3, 0.07), w), 0.0)


def test_single_asset_portfolio_never_trades() -> None:
    w = np.array([0.0, 1.0, 0.0])
    assert met.trade_size(w, np.array([0.3, -0.2, 0.1]), w) == 0.0


def test_net_returns_eq_16() -> None:
    gross = np.array([0.02, -0.01])
    trades = np.array([0.10, 0.40])
    net = met.net_returns(gross, trades, cost=0.005)
    assert np.allclose(net, [1.02 * (1 - 0.0005) - 1, 0.99 * (1 - 0.002) - 1])


def test_return_loss_eq_17() -> None:
    rng = np.random.default_rng(0)
    bench = rng.normal(0.01, 0.04, 400)
    assert np.isclose(met.return_loss(bench, bench), 0.0)
    worse = bench - 0.002  # same volatility, mean lower by 20 basis points
    assert np.isclose(met.return_loss(bench, worse), 0.002)


def jk_by_hand(mi: float, mn: float, si: float, sn: float, sin: float, t: int) -> float:
    theta = (
        2 * si**2 * sn**2
        - 2 * si * sn * sin
        + 0.5 * mi**2 * sn**2
        + 0.5 * mn**2 * si**2
        - mi * mn / (si * sn) * sin**2
    ) / t
    return (sn * mi - si * mn) / math.sqrt(theta)


def test_sharpe_test_matches_footnote_16() -> None:
    rng = np.random.default_rng(1)
    a = rng.normal(0.010, 0.05, 300)
    b = 0.6 * a + rng.normal(0.002, 0.03, 300)
    cov = np.cov(a, b)
    expected = jk_by_hand(
        a.mean(), b.mean(), math.sqrt(cov[0, 0]), math.sqrt(cov[1, 1]), cov[0, 1], 300
    )
    got = met.sharpe_difference_test(a, b)
    assert np.isclose(got.statistic, expected)
    assert np.isclose(got.p_value, 0.5 * math.erfc(abs(expected) / math.sqrt(2)))


def test_sharpe_test_is_antisymmetric_and_null_on_identical_series() -> None:
    rng = np.random.default_rng(2)
    a = rng.normal(0.01, 0.05, 200)
    b = rng.normal(0.00, 0.04, 200)
    ab, ba = met.sharpe_difference_test(a, b), met.sharpe_difference_test(b, a)
    assert np.isclose(ab.statistic, -ba.statistic)
    assert np.isclose(ab.p_value, ba.p_value)
    same = met.sharpe_difference_test(a, a.copy())
    assert same.statistic == 0.0 and same.p_value == 0.5


def test_sharpe_test_ignores_leverage() -> None:
    # A Sharpe ratio does not change with scale, so neither should the test.
    rng = np.random.default_rng(3)
    a = rng.normal(0.01, 0.05, 250)
    b = 0.5 * a + rng.normal(0.004, 0.02, 250)
    z1 = met.sharpe_difference_test(a, b).statistic
    z2 = met.sharpe_difference_test(3.0 * a, 0.2 * b).statistic
    assert np.isclose(z1, z2)


def test_ceq_test_matches_footnote_18() -> None:
    rng = np.random.default_rng(4)
    a = rng.normal(0.010, 0.05, 300)
    b = 0.6 * a + rng.normal(0.002, 0.03, 300)
    gamma = 2.0
    cov = np.cov(a, b)
    vi, vn, cin = cov[0, 0], cov[1, 1], cov[0, 1]
    theta = np.array(
        [
            [vi, cin, 0, 0],
            [cin, vn, 0, 0],
            [0, 0, 2 * vi**2, 2 * cin**2],
            [0, 0, 2 * cin**2, 2 * vn**2],
        ]
    )
    grad = np.array([1.0, -1.0, -gamma / 2, gamma / 2])
    f = (a.mean() - gamma / 2 * vi) - (b.mean() - gamma / 2 * vn)
    expected = f / math.sqrt(grad @ theta @ grad / 300)
    got = met.ceq_difference_test(a, b, gamma=gamma)
    assert np.isclose(got.statistic, expected)


def test_both_tests_have_unit_variance_under_the_null() -> None:
    # Two iid normal series with the same mean and variance and correlation 0.5: the statistics
    # should be close to standard normal. 4000 draws put the standard error of the sample
    # standard deviation near 0.011.
    rng = np.random.default_rng(5)
    chol = np.linalg.cholesky(np.array([[1.0, 0.5], [0.5, 1.0]]))
    z_sr, z_ceq = [], []
    for _ in range(4000):
        x = 0.01 + 0.05 * rng.standard_normal((360, 2)) @ chol.T
        z_sr.append(met.sharpe_difference_test(x[:, 0], x[:, 1]).statistic)
        z_ceq.append(met.ceq_difference_test(x[:, 0], x[:, 1]).statistic)
    for z in (np.array(z_sr), np.array(z_ceq)):
        assert abs(z.mean()) < 0.06
        assert abs(z.std() - 1.0) < 0.05
        assert abs((np.abs(z) > 1.96).mean() - 0.05) < 0.015
