"""Portfolio rules. Each returns relative weights in the risky assets.

Sources, with the equation each function implements:

* DeMiguel, Garlappi and Uppal (2009), "DGU": equal weight (section 1.1), sample
  mean-variance (eq. 3), minimum variance (eq. 6), and its short-sale-constrained version
  (section 1.5).
* Maillard, Roncalli and Teiletche (2010): equal risk contribution (their eq. 1 and 7).
* Lopez de Prado (2016): hierarchical risk parity (stages 1 to 3).
* Black and Litterman (1992): implied returns and the posterior mean (appendix, items 6-8).
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray
from scipy.cluster.hierarchy import linkage
from scipy.spatial.distance import squareform

Array = NDArray[np.float64]


def normalise(x: Array) -> Array:
    """DGU eq. (1): ``w = x / |1'x|``. The absolute value keeps the sign of each position."""
    return np.asarray(x / abs(float(x.sum())), dtype=np.float64)


def equal_weight(n: int) -> Array:
    """The 1/N rule, DGU section 1.1."""
    return np.full(n, 1.0 / n)


def mean_variance(mu: Array, sigma: Array) -> Array:
    """Sample-based mean-variance weights, DGU eq. (3): ``Sigma^-1 mu`` normalised by eq. (1).

    Risk aversion cancels in the normalisation, so it is not an argument.
    """
    return normalise(np.linalg.solve(sigma, mu))


def min_variance(sigma: Array) -> Array:
    """Global minimum-variance weights, DGU eq. (6): ``Sigma^-1 1 / (1' Sigma^-1 1)``."""
    x = np.linalg.solve(sigma, np.ones(sigma.shape[0]))
    return np.asarray(x / x.sum(), dtype=np.float64)


def min_variance_long_only(sigma: Array, tol: float = 1e-12) -> Array:
    """Minimum variance with ``w >= 0`` (DGU section 1.5, "min-c"), by an active-set method.

    The loop solves eq. (6) on the assets currently held, drops the most negative weight if
    there is one, and otherwise checks the Karush-Kuhn-Tucker condition for the excluded
    assets: ``(Sigma w)_i >= w' Sigma w``. An excluded asset that violates it is put back.
    The problem is convex, so a point that passes both checks is the optimum.
    """
    n = sigma.shape[0]
    held = np.ones(n, dtype=bool)
    for _ in range(20 * n + 20):
        idx = np.flatnonzero(held)
        y = np.linalg.solve(sigma[np.ix_(idx, idx)], np.ones(idx.size))
        w = np.zeros(n)
        w[idx] = y / y.sum()
        worst = int(np.argmin(w))
        if w[worst] < -tol:
            held[worst] = False
            continue
        grad = sigma @ w
        lam = float(w @ grad)
        slack = np.where(held, 0.0, lam - grad)
        entering = int(np.argmax(slack))
        if slack[entering] > tol * max(lam, 1.0):
            held[entering] = True
            continue
        w = np.clip(w, 0.0, None)
        return np.asarray(w / w.sum(), dtype=np.float64)
    raise RuntimeError("long-only minimum variance did not converge")


def risk_contributions(w: Array, sigma: Array) -> Array:
    """Total risk contributions ``w_i (Sigma w)_i / sqrt(w' Sigma w)``; they sum to the volatility.

    Maillard, Roncalli and Teiletche, section 2.1.
    """
    marginal = sigma @ w
    return np.asarray(w * marginal / np.sqrt(w @ marginal), dtype=np.float64)


def equal_risk_contribution(sigma: Array, tol: float = 1e-13, max_sweeps: int = 100_000) -> Array:
    """Long-only equal-risk-contribution weights, Maillard, Roncalli and Teiletche eq. (1).

    Their problem (7) minimises ``sqrt(y' Sigma y)`` subject to ``sum ln y_i >= c``; its
    first-order condition is ``y_i (Sigma y)_i = constant`` and the portfolio is ``y / sum(y)``.
    The paper suggests sequential quadratic programming. Here the condition is solved one
    coordinate at a time, where it is a quadratic with one positive root:

        sigma_ii y_i^2 + b_i y_i - 1/n = 0,   b_i = sum_{j != i} sigma_ij y_j.

    The result is checked against the defining condition (1), not against a solver.
    """
    n = sigma.shape[0]
    diag = np.diag(sigma).copy()
    c = 1.0 / n
    y = 1.0 / np.sqrt(diag * n)
    for _ in range(max_sweeps):
        for i in range(n):
            b = float(sigma[i] @ y) - diag[i] * y[i]
            y[i] = (-b + np.sqrt(b * b + 4.0 * diag[i] * c)) / (2.0 * diag[i])
        if float(np.max(np.abs(y * (sigma @ y) - c))) < tol:
            return np.asarray(y / y.sum(), dtype=np.float64)
    raise RuntimeError("equal risk contribution did not converge")


# --- hierarchical risk parity ---------------------------------------------------------------------


def correlation_distance(corr: Array) -> Array:
    """Lopez de Prado stage 1, first step: ``d_ij = sqrt((1 - rho_ij) / 2)``."""
    return np.asarray(np.sqrt(np.clip(0.5 * (1.0 - corr), 0.0, None)), dtype=np.float64)


def distance_of_distances(dist: Array) -> Array:
    """Stage 1, second step: Euclidean distance between the columns of the distance matrix."""
    diff = dist[:, :, None] - dist[:, None, :]
    return np.asarray(np.sqrt(np.sum(diff * diff, axis=0)), dtype=np.float64)


def single_linkage(dist: Array) -> Array:
    """Stage 1, steps 3 to 6: nearest-point (single) linkage on the distance of distances.

    Returns the ``(N-1) x 4`` linkage matrix described in the paper, in scipy's layout.
    """
    dd = distance_of_distances(dist)
    dd = 0.5 * (dd + dd.T)
    np.fill_diagonal(dd, 0.0)
    return np.asarray(linkage(squareform(dd, checks=False), method="single"), dtype=np.float64)


def quasi_diagonal_order(link: Array) -> list[int]:
    """Stage 2: replace each cluster by its two constituents, in order, until only items remain.

    Same result as the paper's code snippet 1.
    """
    n = link.shape[0] + 1

    def expand(node: int) -> list[int]:
        if node < n:
            return [node]
        row = link[node - n]
        return expand(int(row[0])) + expand(int(row[1]))

    return expand(2 * n - 2)


def _cluster_variance(cov: Array, items: list[int]) -> float:
    """Stage 3, step 3b: variance of the inverse-variance portfolio of a cluster."""
    sub = cov[np.ix_(items, items)]
    ivp = 1.0 / np.diag(sub)
    ivp /= ivp.sum()
    return float(ivp @ sub @ ivp)


def recursive_bisection(cov: Array, order: list[int]) -> Array:
    """Stage 3: split each list at ``int(len/2)`` and share weight by inverse cluster variance.

    ``alpha = 1 - V1 / (V1 + V2)`` goes to the first half (step 3c). Paper's code snippet 2.
    """
    w = np.ones(cov.shape[0])
    clusters = [order]
    while clusters:
        nxt: list[list[int]] = []
        for items in clusters:
            if len(items) < 2:
                continue
            half = len(items) // 2
            left, right = items[:half], items[half:]
            v_left, v_right = _cluster_variance(cov, left), _cluster_variance(cov, right)
            alpha = 1.0 - v_left / (v_left + v_right)
            w[left] *= alpha
            w[right] *= 1.0 - alpha
            nxt += [left, right]
        clusters = nxt
    return w


def hierarchical_risk_parity(cov: Array) -> Array:
    """Hierarchical risk parity of Lopez de Prado (2016): cluster, reorder, bisect."""
    sd = np.sqrt(np.diag(cov))
    corr = cov / np.outer(sd, sd)
    if cov.shape[0] == 1:
        return np.ones(1)
    order = quasi_diagonal_order(single_linkage(correlation_distance(corr)))
    return recursive_bisection(cov, order)


# --- Black-Litterman ------------------------------------------------------------------------------


def implied_returns(sigma: Array, w_ref: Array, delta: float) -> Array:
    """Equilibrium risk premiums ``Pi = delta * Sigma * W`` (Black-Litterman appendix, item 6)."""
    return np.asarray(delta * sigma @ w_ref, dtype=np.float64)


def black_litterman_mean(
    sigma: Array,
    pi: Array,
    tau: float,
    P: Array | None = None,
    Q: Array | None = None,
    omega: Array | None = None,
) -> Array:
    """Posterior mean of expected excess returns (Black-Litterman appendix, item 8).

    ``E[R] = [(tau Sigma)^-1 + P' Omega^-1 P]^-1 [(tau Sigma)^-1 Pi + P' Omega^-1 Q]``

    With no views (``P`` is ``None``) the bracketed terms reduce to the prior and the result
    is ``Pi`` itself.
    """
    prior_precision = np.linalg.inv(tau * sigma)
    if P is None or Q is None or omega is None:
        return np.asarray(np.linalg.solve(prior_precision, prior_precision @ pi), dtype=np.float64)
    view_precision = np.linalg.inv(omega)
    lhs = prior_precision + P.T @ view_precision @ P
    rhs = prior_precision @ pi + P.T @ view_precision @ Q
    return np.asarray(np.linalg.solve(lhs, rhs), dtype=np.float64)


def black_litterman_neutral(
    sigma: Array, w_ref: Array, delta: float = 2.5, tau: float = 0.05
) -> Array:
    """Black-Litterman with no view: mean-variance weights on the equilibrium premiums.

    ``Sigma^-1 (delta Sigma W)`` is ``delta W``, so the answer is the reference portfolio
    whatever ``delta``, ``tau`` and the covariance estimate are. The function still goes
    through the full calculation so that the identity is tested, not assumed.
    """
    pi = implied_returns(sigma, w_ref, delta)
    return mean_variance(black_litterman_mean(sigma, pi, tau), sigma)
