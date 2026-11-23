"""The rolling-sample machinery: timing, shapes and the no-look-ahead property."""

from __future__ import annotations

import numpy as np
import pytest

from occam import backtest as bt
from occam import estimators as est
from occam import optimizers as opt


def panel(t: int = 90, n: int = 5, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    common = rng.normal(0.005, 0.04, (t, 1))
    return common + rng.normal(0.002, 0.03, (t, n)) * np.linspace(0.6, 1.4, n)


def test_shapes_follow_the_paper() -> None:
    r = panel()
    run = bt.rolling_backtest(r, 24, bt.RULES["min"], 4)
    assert run.returns.shape == (90 - 24,)  # T - M out-of-sample returns
    assert run.trades.shape == (90 - 24,)  # eq. (15) sums T - M trades
    assert run.weights.shape == (90 - 24 + 1, 5)


def test_equal_weight_returns_are_the_cross_sectional_mean() -> None:
    r = panel()
    run = bt.rolling_backtest(r, 24, bt.RULES["ew"], 4)
    assert np.allclose(run.returns, r[24:].mean(axis=1))


def test_market_rule_returns_the_market_column_and_never_trades() -> None:
    r = panel()
    for name in ("vw", "bl"):
        run = bt.rolling_backtest(r, 24, bt.RULES[name], 2)
        assert np.allclose(run.returns, r[24:, 2])
        assert np.allclose(run.trades, 0.0, atol=1e-12)


def test_first_weights_use_exactly_the_first_window() -> None:
    r = panel()
    run = bt.rolling_backtest(r, 24, bt.RULES["mv"], 4)
    expected = opt.mean_variance(est.sample_mean(r[:24]), est.sample_cov(r[:24]))
    assert np.allclose(run.weights[0], expected)
    assert np.isclose(run.returns[0], expected @ r[24])
    last = opt.mean_variance(est.sample_mean(r[-24:]), est.sample_cov(r[-24:]))
    assert np.allclose(run.weights[-1], last)


@pytest.mark.parametrize("name", list(bt.RULES))
def test_no_look_ahead(name: str) -> None:
    # Changing returns from month 60 onward must not change any weight chosen before month 60
    # or any return earned before it.
    r = panel(seed=3)
    changed = r.copy()
    changed[60:] = panel(seed=99)[60:]
    a = bt.rolling_backtest(r, 24, bt.RULES[name], 4)
    b = bt.rolling_backtest(changed, 24, bt.RULES[name], 4)
    assert np.array_equal(a.weights[: 60 - 24 + 1], b.weights[: 60 - 24 + 1])
    assert np.array_equal(a.returns[: 60 - 24], b.returns[: 60 - 24])


@pytest.mark.parametrize("name", list(bt.RULES))
def test_weights_sum_to_one_in_absolute_value(name: str) -> None:
    run = bt.rolling_backtest(panel(seed=4), 24, bt.RULES[name], 4)
    assert np.allclose(np.abs(run.weights.sum(axis=1)), 1.0)


@pytest.mark.parametrize("name", ["ew", "min-c", "vw", "erc", "hrp"])
def test_long_only_rules_stay_long_only(name: str) -> None:
    run = bt.rolling_backtest(panel(seed=5), 24, bt.RULES[name], 4)
    assert (run.weights >= 0).all()


def test_equal_weight_turnover_hand_solved() -> None:
    # Two assets, one month after the window: returns (10%, 0%) drift 50/50 to
    # (0.55, 0.50) / 1.05, and trading back costs 2 * (0.55 / 1.05 - 0.5).
    r = np.zeros((4, 2))
    r[3] = [0.10, 0.0]
    run = bt.rolling_backtest(r, 3, bt.RULES["ew"], 1)
    assert np.isclose(run.turnover, 2 * (0.55 / 1.05 - 0.5))


def test_window_must_be_shorter_than_sample() -> None:
    with pytest.raises(ValueError, match="need more than"):
        bt.rolling_backtest(panel(t=24), 24, bt.RULES["ew"], 0)


def test_in_sample_mean_variance_has_the_maximum_sharpe_ratio() -> None:
    # Eq. (13): with M = T the Sharpe ratio is sqrt(mu' Sigma^-1 mu), the highest attainable.
    r = panel(t=200, seed=6)
    ins = bt.in_sample_mean_variance(r)
    mu, sigma = est.sample_mean(r), est.sample_cov(r)
    best = float(np.sqrt(mu @ np.linalg.solve(sigma, mu)))
    assert np.isclose(abs(ins.mean() / ins.std(ddof=1)), best)
    for name in ("ew", "min"):
        w = bt.RULES[name](r, 4)
        other = r @ w
        assert other.mean() / other.std(ddof=1) <= best + 1e-12
