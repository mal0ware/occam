"""How far small changes to the input returns move each column of the paper's Table 3.

The 2005 files the paper used are not available, so the size of the revisions since then is
unknown. This module adds independent normal noise of a stated size to every monthly return,
reruns the rolling test, and reports the spread of the Sharpe ratio. A rule whose spread is
wide at a small noise level cannot be compared with the paper cell by cell. A rule whose
spread is narrow should match, and a gap there needs another explanation.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from occam import backtest as bt
from occam import metrics as met
from occam import paper
from occam.data import Dataset

#: Standard deviation of the noise, in basis points per month. One basis point is the
#: precision the library publishes to. The larger sizes are assumptions, not measurements.
SCALES_BP = (1.0, 5.0, 20.0)
DRAWS = 100
RULES = ("ew", "mv-is", "mv", "min")
SEED = 20261124


@dataclass(frozen=True)
class Spread:
    dataset: str
    rule: str
    scale_bp: float
    base: float
    low: float  # 5th percentile
    median: float
    high: float  # 95th percentile
    minimum: float
    maximum: float
    published: float


def _sharpe(returns: np.ndarray, market: int, rule: str, window: int) -> float:
    if rule == "mv-is":
        return met.sharpe_ratio(bt.in_sample_mean_variance(returns))
    return met.sharpe_ratio(bt.rolling_backtest(returns, window, bt.RULES[rule], market).returns)


def spreads(
    datasets: dict[str, Dataset],
    draws: int = DRAWS,
    scales_bp: tuple[float, ...] = SCALES_BP,
    window: int = paper.WINDOW,
) -> list[Spread]:
    """Spread of each rule's Sharpe ratio on the paper's sample under noise of each size."""
    out = []
    for d, key in enumerate(paper.DATASETS):
        if key not in datasets:
            continue
        data = datasets[key].window(paper.SAMPLE_START, paper.SAMPLE_END)
        base = {r: _sharpe(data.returns, data.market_index, r, window) for r in RULES}
        for scale in scales_bp:
            rng = np.random.default_rng([SEED, d, round(scale * 100)])
            sims: dict[str, list[float]] = {r: [] for r in RULES}
            for _ in range(draws):
                noisy = data.returns + rng.normal(0.0, scale * 1e-4, data.returns.shape)
                for r in RULES:
                    sims[r].append(_sharpe(noisy, data.market_index, r, window))
            for r in RULES:
                x = np.array(sims[r])
                published = paper.SHARPE[r][paper.DATASETS.index(key)]
                assert published is not None
                out.append(
                    Spread(
                        dataset=key,
                        rule=r,
                        scale_bp=scale,
                        base=base[r],
                        low=float(np.quantile(x, 0.05)),
                        median=float(np.median(x)),
                        high=float(np.quantile(x, 0.95)),
                        minimum=float(x.min()),
                        maximum=float(x.max()),
                        published=published,
                    )
                )
    return out


def spreads_csv(rows: list[Spread]) -> str:
    lines = ["dataset,rule,noise_bp,sharpe,p05,median,p95,min,max,paper,paper_inside_min_max"]
    for s in rows:
        inside = s.minimum <= s.published <= s.maximum
        lines.append(
            f"{s.dataset},{s.rule},{s.scale_bp:g},{s.base:.4f},{s.low:.4f},{s.median:.4f},"
            f"{s.high:.4f},{s.minimum:.4f},{s.maximum:.4f},{s.published:.4f},"
            f"{'yes' if inside else 'no'}"
        )
    return "\n".join(lines) + "\n"


def spreads_md(rows: list[Spread], labels: dict[str, str], draws: int) -> str:
    """One table per noise size: the 5th to 95th percentile of each Sharpe ratio."""
    parts = [
        "# Sensitivity of the paper-sample Sharpe ratios to small changes in the data\n",
        "Independent normal noise is added to every monthly return of a dataset and the "
        f"rolling test is rerun, {draws} times for each noise size. A cell shows the Sharpe "
        "ratio on the files as downloaded, then the 5th to 95th percentile over the noisy "
        "copies in brackets, then the paper's figure. A star marks a paper figure outside the "
        "full range of the noisy copies. One basis point is the precision the library "
        "publishes to. The larger sizes are assumptions about how large revisions might be. "
        "They were not measured.\n",
    ]
    keys = list(dict.fromkeys(s.dataset for s in rows))
    for scale in dict.fromkeys(s.scale_bp for s in rows):
        parts.append(f"## Noise with a standard deviation of {scale:g} basis points a month\n")
        parts.append("| rule | " + " | ".join(labels[k] for k in keys) + " |")
        parts.append("|" + "|".join(["---"] * (len(keys) + 1)) + "|")
        for rule in RULES:
            cells = []
            for key in keys:
                s = next(x for x in rows if (x.dataset, x.rule, x.scale_bp) == (key, rule, scale))
                star = "" if s.minimum <= s.published <= s.maximum else "*"
                cells.append(
                    f"{s.base:.4f} [{s.low:.4f}, {s.high:.4f}] paper {s.published:.4f}{star}"
                )
            parts.append(f"| {rule} | " + " | ".join(cells) + " |")
        parts.append("")
    return "\n".join(parts)
