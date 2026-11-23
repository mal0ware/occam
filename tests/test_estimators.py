"""Sample moments and the Ledoit-Wolf (2004) estimator."""

from __future__ import annotations

import numpy as np
import pytest

from occam import estimators as est


def sample(n: int, p: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    scale = rng.uniform(0.02, 0.08, p)
    return rng.standard_normal((n, p)) * scale + rng.standard_normal((n, 1)) * 0.03


def test_sample_moments_match_numpy() -> None:
    x = sample(60, 5, 0)
    assert np.allclose(est.sample_mean(x), x.mean(axis=0))
    assert np.allclose(est.sample_cov(x), np.cov(x, rowvar=False))


def test_sample_cov_hand_solved() -> None:
    x = np.array([[1.0, 2.0], [3.0, 2.0], [5.0, 8.0]])
    # means (3, 4); deviations (-2,-2), (0,-2), (2,4); divide by n - 1 = 2.
    assert np.allclose(est.sample_cov(x), [[4.0, 6.0], [6.0, 12.0]])


def lw_by_definition(x: np.ndarray) -> tuple[np.ndarray, float]:
    """Ledoit-Wolf section 3.3 written out with explicit loops and the norm ||A||^2 = tr(AA')/p."""
    n, p = x.shape
    x = x - x.mean(axis=0)
    s = x.T @ x / n

    def norm2(a: np.ndarray) -> float:
        return float(np.trace(a @ a.T) / p)

    m = float(np.trace(s) / p)  # Lemma 3.2
    d2 = norm2(s - m * np.eye(p))  # Lemma 3.3
    bbar2 = sum(norm2(np.outer(x[k], x[k]) - s) for k in range(n)) / n**2  # Lemma 3.4
    b2 = min(bbar2, d2)
    a2 = d2 - b2  # Lemma 3.5
    return b2 / d2 * m * np.eye(p) + a2 / d2 * s, b2 / d2  # eq. (14)


@pytest.mark.parametrize(("n", "p", "seed"), [(120, 11, 1), (60, 21, 2), (30, 24, 3), (20, 40, 4)])
def test_ledoit_wolf_matches_the_definition(n: int, p: int, seed: int) -> None:
    x = sample(n, p, seed)
    expected, rho = lw_by_definition(x)
    got = est.ledoit_wolf(x)
    assert np.allclose(got.covariance, expected, rtol=1e-12, atol=1e-18)
    assert np.isclose(got.shrinkage, rho)


def test_ledoit_wolf_intensity_between_zero_and_one_and_trace_preserved() -> None:
    for seed in range(10):
        x = sample(40, 15, seed)
        got = est.ledoit_wolf(x)
        assert 0.0 <= got.shrinkage <= 1.0
        s = np.cov(x, rowvar=False, bias=True)
        assert np.isclose(np.trace(got.covariance), np.trace(s))


def test_ledoit_wolf_pulls_eigenvalues_toward_their_mean() -> None:
    x = sample(30, 20, 5)
    s = np.cov(x, rowvar=False, bias=True)
    shrunk = est.ledoit_wolf(x).covariance
    ev_s, ev_l = np.linalg.eigvalsh(s), np.linalg.eigvalsh(shrunk)
    assert ev_l.min() >= ev_s.min() and ev_l.max() <= ev_s.max()
    assert np.linalg.cond(shrunk) < np.linalg.cond(s)


def test_ledoit_wolf_is_invertible_when_sample_matrix_is_singular() -> None:
    x = sample(10, 25, 6)  # more assets than observations
    s = np.cov(x, rowvar=False, bias=True)
    assert np.linalg.matrix_rank(s) < 25
    shrunk = est.ledoit_wolf(x).covariance
    assert np.linalg.eigvalsh(shrunk).min() > 0


def test_ledoit_wolf_shrinks_less_with_more_data() -> None:
    rng = np.random.default_rng(7)
    scale = np.linspace(0.02, 0.10, 10)
    small = est.ledoit_wolf(rng.standard_normal((30, 10)) * scale).shrinkage
    large = est.ledoit_wolf(rng.standard_normal((3000, 10)) * scale).shrinkage
    assert large < small
    assert large < 0.05


def test_ledoit_wolf_spherical_data_shrinks_fully() -> None:
    # If the true covariance is a multiple of the identity, all dispersion of the sample
    # eigenvalues is error, and the estimator should move almost all the way to m*I.
    rng = np.random.default_rng(8)
    got = est.ledoit_wolf(rng.standard_normal((50, 30)) * 0.05)
    assert got.shrinkage > 0.85
