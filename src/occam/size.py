"""How often each Sharpe test rejects a true null, by simulation.

Two series with the same Sharpe ratio are drawn many times and each test is run at the 5%
level. A test with the right size rejects about 5% of the time.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from occam import metrics as met
from occam import robust

Array = NDArray[np.float64]

CORRELATION = 0.5
MEAN, VOL = 0.006, 0.045  # roughly a monthly equity portfolio
#: name -> (Student t shocks with 5 degrees of freedom, GARCH(1,1) variance, AR(1) coefficient)
PROCESSES: dict[str, tuple[bool, bool, float]] = {
    "iid normal": (False, False, 0.0),
    "iid t(5)": (True, False, 0.0),
    "GARCH(1,1), normal shocks": (False, True, 0.0),
    "GARCH(1,1), t(5) shocks": (True, True, 0.0),
    "AR(1) 0.2 with GARCH(1,1), t(5) shocks": (True, True, 0.2),
}
GARCH_A, GARCH_B = 0.10, 0.85
BURN = 200


def simulate(process: str, months: int, rng: np.random.Generator) -> Array:
    """One ``(months, 2)`` sample of two series with equal Sharpe ratios.

    Both columns have the same mean and the same variance under every process, so their
    Sharpe ratios are equal in population. Shocks have correlation 0.5. With GARCH each
    column has its own variance process.
    """
    fat, garch, phi = PROCESSES[process]
    chol = np.linalg.cholesky(np.array([[1.0, CORRELATION], [CORRELATION, 1.0]]))
    n = months + BURN
    dof = 5.0
    raw = (
        rng.standard_t(dof, size=(n, 2)) * np.sqrt((dof - 2.0) / dof)
        if fat
        else rng.standard_normal((n, 2))
    )
    e = raw @ chol.T
    if garch:
        h = np.ones(2)
        for k in range(n):
            e[k] = np.sqrt(h) * e[k]
            h = (1.0 - GARCH_A - GARCH_B) + GARCH_A * e[k] ** 2 + GARCH_B * h
    if phi != 0.0:
        for k in range(1, n):
            e[k] = phi * e[k - 1] + e[k]
    return np.asarray(MEAN + VOL * e[BURN:], dtype=np.float64)


@dataclass(frozen=True)
class SizeRow:
    process: str
    months: int
    simulations: int
    jk: float
    hac: float
    bootstrap: float


def rejection_rates(
    process: str,
    months: int,
    simulations: int,
    seed: int,
    block: int = robust.BLOCK,
    draws: int = 499,
    level: float = 0.05,
) -> SizeRow:
    """Share of simulations in which each two-sided test rejects at ``level``."""
    rng = np.random.default_rng(seed)
    hits = np.zeros(3)
    for _ in range(simulations):
        x = simulate(process, months, rng)
        a, b = x[:, 0], x[:, 1]
        hits[0] += met.sharpe_difference_test(a, b).p_value < level
        hits[1] += robust.hac_sharpe_test(a, b).p_value < level
        hits[2] += robust.bootstrap_sharpe_test(a, b, rng, block, draws).p_value < level
    rates = hits / simulations
    return SizeRow(process, months, simulations, float(rates[0]), float(rates[1]), float(rates[2]))


def size_md(rows: list[SizeRow], block: int, draws: int, level: float = 0.05) -> str:
    """Rejection rates as a table."""
    n = rows[0].simulations
    se = (level * (1.0 - level) / n) ** 0.5
    lines = [
        "# Rejection rates of the Sharpe tests when the null is true\n",
        f"Two series with the same Sharpe ratio and {rows[0].months} months, the length of "
        f"the paper's out-of-sample period, drawn {n} times for each process. Each test is "
        f"two-sided at {level:g}. A correctly sized test rejects {level:.0%} of the time. With "
        f"{n} simulations the standard error of a rate near that is {se:.3f}. The bootstrap "
        f"uses blocks of {block} months and {draws} draws for each simulation.\n",
        "| process | Jobson-Korkie with Memmel | Ledoit-Wolf HAC | Ledoit-Wolf bootstrap |",
        "|---|---|---|---|",
    ]
    lines += [f"| {r.process} | {r.jk:.3f} | {r.hac:.3f} | {r.bootstrap:.3f} |" for r in rows]
    return "\n".join(lines) + "\n"
