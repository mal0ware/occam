"""Closed forms and hand-solved cases for every portfolio rule."""

from __future__ import annotations

import numpy as np
import pytest

from occam import optimizers as opt


def cov2(s1: float, s2: float, rho: float) -> np.ndarray:
    return np.array([[s1 * s1, rho * s1 * s2], [rho * s1 * s2, s2 * s2]])


def random_cov(n: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    a = rng.standard_normal((4 * n, n)) * rng.uniform(0.02, 0.08, n)
    a += rng.standard_normal((4 * n, 1)) * 0.04  # a common factor, so correlations are positive
    return np.cov(a, rowvar=False)


# --- equal weight -----------------------------------------------------------------------------


def test_equal_weight() -> None:
    w = opt.equal_weight(7)
    assert w.shape == (7,)
    assert np.allclose(w, 1 / 7)


# --- sample mean-variance, DeMiguel-Garlappi-Uppal eq. (3) -------------------------------------


def test_mean_variance_hand_solved_two_assets() -> None:
    # Sigma^-1 mu = (1/0.0035) * (0.0015, 0.0010), so the relative weights are (0.6, 0.4).
    sigma = np.array([[0.04, 0.01], [0.01, 0.09]])
    mu = np.array([0.02, 0.03])
    assert np.allclose(opt.mean_variance(mu, sigma), [0.6, 0.4], atol=1e-14)


def test_mean_variance_reproduces_paper_footnote_8() -> None:
    # Footnote 8: two assets, vol 20%, correlation 0.99, true mean 8%. Estimating the first
    # mean as 9% moves the weights from 50/50 to 635% and -535%.
    sigma = cov2(0.20, 0.20, 0.99)
    assert np.allclose(opt.mean_variance(np.array([0.08, 0.08]), sigma), [0.5, 0.5])
    w = opt.mean_variance(np.array([0.09, 0.08]), sigma)
    assert np.round(100 * w).tolist() == [635.0, -535.0]


def test_mean_variance_is_scale_free_in_risk_aversion_and_covariance() -> None:
    sigma = random_cov(5, 1)
    mu = np.array([0.01, 0.004, 0.007, 0.002, 0.009])
    w = opt.mean_variance(mu, sigma)
    assert np.allclose(opt.mean_variance(3.0 * mu, sigma), w)
    assert np.allclose(opt.mean_variance(mu, 7.0 * sigma), w)


def test_normalise_keeps_direction_when_weights_sum_negative() -> None:
    # Eq. (1): dividing by |1'x| keeps the sign of every position.
    x = np.array([1.0, -3.0])
    w = opt.normalise(x)
    assert np.allclose(w, [0.5, -1.5])
    assert np.isclose(w.sum(), -1.0)


def test_mean_variance_proportional_means_gives_minimum_variance() -> None:
    # Section 1.4.1: minimum variance is the limit of eq. (3) when mu is proportional to 1.
    sigma = random_cov(6, 2)
    assert np.allclose(opt.mean_variance(np.full(6, 0.01), sigma), opt.min_variance(sigma))


# --- minimum variance, eq. (6) -----------------------------------------------------------------


def test_min_variance_two_asset_closed_form() -> None:
    sigma = np.array([[0.04, 0.01], [0.01, 0.09]])
    w1 = (0.09 - 0.01) / (0.04 + 0.09 - 2 * 0.01)
    assert np.allclose(opt.min_variance(sigma), [w1, 1 - w1], atol=1e-14)


def test_min_variance_first_order_condition() -> None:
    # At the optimum Sigma w is the same for every asset (the Lagrange multiplier).
    sigma = random_cov(8, 3)
    w = opt.min_variance(sigma)
    grad = sigma @ w
    assert np.isclose(w.sum(), 1.0)
    assert np.allclose(grad, grad[0])


def test_min_variance_beats_perturbations() -> None:
    sigma = random_cov(5, 4)
    w = opt.min_variance(sigma)
    rng = np.random.default_rng(0)
    for _ in range(200):
        d = rng.standard_normal(5) * 0.05
        d -= d.mean()
        v = w + d
        assert v @ sigma @ v >= w @ sigma @ w - 1e-15


# --- long-only minimum variance -----------------------------------------------------------------


def test_long_only_equals_unconstrained_when_interior() -> None:
    sigma = np.array([[0.04, 0.01], [0.01, 0.09]])
    assert np.allclose(opt.min_variance_long_only(sigma), opt.min_variance(sigma))


def test_long_only_corner_two_assets() -> None:
    # Unconstrained w1 = (0.04 - 0.018) / (0.05 - 0.036) = 1.571, so the constraint binds at (1, 0).
    sigma = cov2(0.10, 0.20, 0.9)
    assert opt.min_variance(sigma)[1] < 0
    assert np.allclose(opt.min_variance_long_only(sigma), [1.0, 0.0])


@pytest.mark.parametrize("seed", range(12))
def test_long_only_satisfies_kkt(seed: int) -> None:
    sigma = random_cov(9, 100 + seed)
    w = opt.min_variance_long_only(sigma)
    grad = sigma @ w
    lam = w @ grad
    assert np.isclose(w.sum(), 1.0)
    assert (w >= 0).all()
    held = w > 0
    assert np.allclose(grad[held], lam, rtol=1e-9)
    assert (grad[~held] >= lam * (1 - 1e-9)).all()


def test_long_only_matches_brute_force_three_assets() -> None:
    sigma = random_cov(3, 7)
    w = opt.min_variance_long_only(sigma)
    grid = np.linspace(0, 1, 201)
    best = min(
        (np.array([a, b, 1 - a - b]) @ sigma @ np.array([a, b, 1 - a - b]))
        for a in grid
        for b in grid
        if a + b <= 1 + 1e-12
    )
    assert w @ sigma @ w <= best + 1e-12


# --- equal risk contribution, Maillard-Roncalli-Teiletche ---------------------------------------


@pytest.mark.parametrize("rho", [-0.5, 0.0, 0.4, 0.95])
def test_erc_two_assets_is_inverse_volatility_for_any_correlation(rho: float) -> None:
    # Section 3.1: x = (1/s1, 1/s2) / (1/s1 + 1/s2), and rho drops out.
    w = opt.equal_risk_contribution(cov2(0.1, 0.3, rho))
    assert np.allclose(w, [0.75, 0.25], atol=1e-10)


def test_erc_equal_correlation_is_inverse_volatility() -> None:
    # Eq. (3) of the paper.
    vols = np.array([0.05, 0.10, 0.20, 0.40])
    corr = np.full((4, 4), 0.3) + 0.7 * np.eye(4)
    sigma = corr * np.outer(vols, vols)
    expected = (1 / vols) / (1 / vols).sum()
    assert np.allclose(opt.equal_risk_contribution(sigma), expected, atol=1e-10)


@pytest.mark.parametrize("seed", range(8))
def test_erc_equalises_risk_contributions(seed: int) -> None:
    sigma = random_cov(10, 200 + seed)
    w = opt.equal_risk_contribution(sigma)
    rc = opt.risk_contributions(w, sigma)
    assert np.isclose(w.sum(), 1.0)
    assert (w > 0).all()
    assert np.isclose(rc.sum(), np.sqrt(w @ sigma @ w))  # Euler decomposition
    assert np.allclose(rc, rc.mean(), rtol=1e-8)


def test_erc_volatility_lies_between_min_variance_and_equal_weight() -> None:
    # Section 3.4 of the paper: sigma_mv <= sigma_erc <= sigma_1/n.
    sigma = random_cov(7, 9)

    def vol(w: np.ndarray) -> float:
        return float(np.sqrt(w @ sigma @ w))

    v_erc = vol(opt.equal_risk_contribution(sigma))
    assert vol(opt.min_variance_long_only(sigma)) <= v_erc + 1e-12
    assert v_erc <= vol(opt.equal_weight(7)) + 1e-12


# --- hierarchical risk parity, Lopez de Prado ----------------------------------------------------

PAPER_CORR = np.array([[1.0, 0.7, 0.2], [0.7, 1.0, -0.2], [0.2, -0.2, 1.0]])


def test_hrp_correlation_distance_matches_paper_example_1() -> None:
    d = opt.correlation_distance(PAPER_CORR)
    expected = np.array([[0, 0.3873, 0.6325], [0.3873, 0, 0.7746], [0.6325, 0.7746, 0]])
    assert np.allclose(d, expected, atol=5e-5)


def test_hrp_distance_of_distances_matches_paper_example_2() -> None:
    dd = opt.distance_of_distances(opt.correlation_distance(PAPER_CORR))
    expected = np.array([[0, 0.5659, 0.9747], [0.5659, 0, 1.1225], [0.9747, 1.1225, 0]])
    assert np.allclose(dd, expected, atol=5e-5)


def test_hrp_first_cluster_matches_paper_example_3() -> None:
    link = opt.single_linkage(opt.correlation_distance(PAPER_CORR))
    assert sorted(int(i) for i in link[0, :2]) == [0, 1]  # the paper's u[1] = (1, 2)
    assert np.isclose(link[0, 2], 0.5659, atol=5e-5)
    assert np.isclose(link[1, 2], 0.9747, atol=5e-5)  # nearest-point update, example 4
    assert sorted(opt.quasi_diagonal_order(link)) == [0, 1, 2]


def test_hrp_two_assets_is_inverse_variance() -> None:
    sigma = cov2(0.1, 0.2, 0.3)
    assert np.allclose(opt.hierarchical_risk_parity(sigma), [0.8, 0.2])


@pytest.mark.parametrize("n", [2, 3, 5, 8, 13])
def test_hrp_on_diagonal_covariance_is_inverse_variance(n: int) -> None:
    # Appendix A.2: inverse-variance weights are optimal for a diagonal matrix, and the
    # bisection reproduces them at every level.
    var = np.linspace(0.01, 0.09, n)
    w = opt.hierarchical_risk_parity(np.diag(var))
    assert np.allclose(w, (1 / var) / (1 / var).sum())


def test_recursive_bisection_hand_solved_three_assets() -> None:
    # Order (0, 1, 2) splits into {0} and {1, 2}. Cluster {1, 2} uses inverse-variance weights
    # (0.8, 0.2): variance 0.64*0.01 + 0.04*0.04 + 2*0.8*0.2*0.004 = 0.00928.
    sigma = np.array([[0.02, 0.0, 0.0], [0.0, 0.01, 0.004], [0.0, 0.004, 0.04]])
    alpha = 1 - 0.02 / (0.02 + 0.00928)
    w = opt.recursive_bisection(sigma, [0, 1, 2])
    assert np.allclose(w, [alpha, (1 - alpha) * 0.8, (1 - alpha) * 0.2])


@pytest.mark.parametrize("seed", range(6))
def test_hrp_weights_are_a_long_only_portfolio(seed: int) -> None:
    w = opt.hierarchical_risk_parity(random_cov(11, 300 + seed))
    assert np.isclose(w.sum(), 1.0)
    assert (w > 0).all()


def test_hrp_is_invariant_to_asset_order() -> None:
    sigma = random_cov(9, 11)
    perm = np.random.default_rng(5).permutation(9)
    w = opt.hierarchical_risk_parity(sigma)
    w_perm = opt.hierarchical_risk_parity(sigma[np.ix_(perm, perm)])
    assert np.allclose(np.sort(w), np.sort(w_perm))


# --- Black-Litterman -----------------------------------------------------------------------------


def test_bl_implied_returns_definition() -> None:
    # Appendix item 6: Pi = delta * Sigma * W.
    sigma = random_cov(4, 12)
    w_ref = np.array([0.4, 0.3, 0.2, 0.1])
    assert np.allclose(opt.implied_returns(sigma, w_ref, 2.5), 2.5 * sigma @ w_ref)


def test_bl_without_views_returns_equilibrium_and_reference_weights() -> None:
    sigma = random_cov(5, 13)
    w_ref = np.array([0.35, 0.25, 0.2, 0.15, 0.05])
    pi = opt.implied_returns(sigma, w_ref, 3.0)
    post = opt.black_litterman_mean(sigma, pi, tau=0.05)
    assert np.allclose(post, pi)
    assert np.allclose(opt.mean_variance(post, sigma), w_ref)
    assert np.allclose(opt.black_litterman_neutral(sigma, w_ref), w_ref)


def test_bl_neutral_weights_do_not_depend_on_delta_or_covariance() -> None:
    w_ref = np.array([0.0, 0.0, 1.0])
    for seed in (1, 2, 3):
        sigma = random_cov(3, seed)
        for delta in (0.5, 2.5, 10.0):
            assert np.allclose(opt.black_litterman_neutral(sigma, w_ref, delta=delta), w_ref)


def test_bl_certain_view_is_met_and_vague_view_is_ignored() -> None:
    sigma = random_cov(4, 14)
    pi = opt.implied_returns(sigma, np.full(4, 0.25), 2.5)
    P = np.array([[1.0, -1.0, 0.0, 0.0]])
    Q = np.array([0.02])
    sure = opt.black_litterman_mean(sigma, pi, tau=0.05, P=P, Q=Q, omega=np.array([[1e-14]]))
    assert np.allclose(P @ sure, Q, atol=1e-8)
    vague = opt.black_litterman_mean(sigma, pi, tau=0.05, P=P, Q=Q, omega=np.array([[1e12]]))
    assert np.allclose(vague, pi, atol=1e-10)


def test_bl_one_asset_is_precision_weighted_average() -> None:
    # Appendix item 8 with n = k = 1: (pi/(tau s2) + q/omega) / (1/(tau s2) + 1/omega).
    s2, tau, pi, q, om = 0.04, 0.5, 0.01, 0.03, 0.01
    expected = (pi / (tau * s2) + q / om) / (1 / (tau * s2) + 1 / om)
    got = opt.black_litterman_mean(
        np.array([[s2]]), np.array([pi]), tau=tau, P=np.eye(1), Q=np.array([q]),
        omega=np.array([[om]]),
    )
    assert np.isclose(got[0], expected)


# --- shared invariants ---------------------------------------------------------------------------


@pytest.mark.parametrize("seed", range(5))
def test_every_rule_sums_to_one(seed: int) -> None:
    sigma = random_cov(6, 400 + seed)
    mu = np.random.default_rng(seed).uniform(0.002, 0.01, 6)
    rules = [
        opt.equal_weight(6),
        opt.mean_variance(mu, sigma),
        opt.min_variance(sigma),
        opt.min_variance_long_only(sigma),
        opt.equal_risk_contribution(sigma),
        opt.hierarchical_risk_parity(sigma),
        opt.black_litterman_neutral(sigma, opt.equal_weight(6)),
    ]
    for w in rules:
        assert np.isclose(abs(w.sum()), 1.0)
