"""Moment estimators: sample mean and covariance, and Ledoit-Wolf (2004) shrinkage.

Every function takes a ``(n, p)`` array with one row per month and one column per asset.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

Array = NDArray[np.float64]


def sample_mean(returns: Array) -> Array:
    """Sample mean of each column."""
    return np.asarray(returns.mean(axis=0), dtype=np.float64)


def sample_cov(returns: Array) -> Array:
    """Unbiased sample covariance matrix (divisor ``n - 1``).

    The divisor does not affect any portfolio in this project: every rule normalises its
    weights, and a scalar multiple of the covariance matrix cancels.
    """
    x = returns - returns.mean(axis=0)
    return np.asarray(x.T @ x / (returns.shape[0] - 1), dtype=np.float64)


@dataclass(frozen=True)
class Shrinkage:
    """A shrunk covariance matrix and the weight placed on the target."""

    covariance: Array
    shrinkage: float


def ledoit_wolf(returns: Array) -> Shrinkage:
    """Ledoit and Wolf (2004) shrinkage toward a scaled identity, their equation (14).

    With the norm ``||A||^2 = tr(A A') / p`` of their section 2.1 and ``S = X'X / n``:

    * ``m = tr(S) / p``                                      (Lemma 3.2)
    * ``d2 = ||S - m I||^2``                                 (Lemma 3.3)
    * ``bbar2 = (1/n^2) sum_k ||x_k x_k' - S||^2``, ``b2 = min(bbar2, d2)``  (Lemma 3.4)
    * ``a2 = d2 - b2``                                       (Lemma 3.5)
    * ``S* = (b2/d2) m I + (a2/d2) S``                       (equation 14)

    The paper assumes mean-zero observations. Here each column is demeaned first and the
    divisor stays ``n``; this is the one departure from the source.
    """
    n, p = returns.shape
    x = returns - returns.mean(axis=0)
    s = x.T @ x / n
    m = float(np.trace(s)) / p
    d2 = float(np.sum((s - m * np.eye(p)) ** 2)) / p
    if d2 == 0.0:
        return Shrinkage(covariance=np.asarray(s, dtype=np.float64), shrinkage=0.0)
    # ||x x' - S||^2 * p = (x'x)^2 - 2 x'Sx + tr(S^2), summed over observations.
    sq_norms = np.einsum("ij,ij->i", x, x)
    quad = np.einsum("ij,jk,ik->i", x, s, x)
    bbar2 = float(np.sum(sq_norms**2 - 2.0 * quad + np.sum(s * s))) / p / n**2
    b2 = min(bbar2, d2)
    a2 = d2 - b2
    cov = (b2 / d2) * m * np.eye(p) + (a2 / d2) * s
    return Shrinkage(covariance=np.asarray(cov, dtype=np.float64), shrinkage=b2 / d2)
