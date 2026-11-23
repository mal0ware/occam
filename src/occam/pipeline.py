"""Turn datasets into the result tables. Pure functions: same input, same text out."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from occam import backtest as bt
from occam import metrics as met
from occam import paper
from occam.data import Dataset

GAMMA = 1.0  # risk aversion in the certainty equivalent, as in the paper's tables
COST = 0.005  # 50 basis points per unit traded, as in the paper's Table 5
ALPHA = 0.05  # level used when counting "significant" cells


@dataclass(frozen=True)
class Sample:
    """A stretch of months. Evaluation starts ``window`` months after ``start``."""

    key: str
    label: str
    start: int
    end: int


@dataclass(frozen=True)
class Cell:
    """One rule on one dataset in one sample."""

    sample: str
    dataset: str
    rule: str
    months: int
    first_month: int
    last_month: int
    sharpe: float
    sharpe_z: float
    sharpe_p: float
    ceq: float
    ceq_z: float
    ceq_p: float
    turnover: float
    turnover_rel: float
    return_loss: float


def evaluate(dataset: Dataset, sample: Sample, window: int, rules: tuple[str, ...]) -> list[Cell]:
    """Run every rule on one dataset and measure it against equal weight.

    The first cell is ``mv-is``, the mean-variance portfolio estimated on the whole sample
    (eq. 13). It has no turnover and is not tested.
    """
    data = dataset.window(sample.start, sample.end)
    if np.isnan(data.returns).any():
        raise ValueError(f"{dataset.key}: missing returns inside {sample.key}")
    first, last = int(data.months[window]), int(data.months[-1])
    n_oos = data.returns.shape[0] - window
    runs = {
        name: bt.rolling_backtest(data.returns, window, bt.RULES[name], data.market_index)
        for name in rules
    }
    bench = runs["ew"]
    bench_net = met.net_returns(bench.returns, bench.trades, COST)

    ins = bt.in_sample_mean_variance(data.returns)
    nan = float("nan")
    cells = [
        Cell(
            sample.key,
            dataset.key,
            "mv-is",
            n_oos,
            first,
            last,
            met.sharpe_ratio(ins),
            nan,
            nan,
            met.certainty_equivalent(ins, GAMMA),
            nan,
            nan,
            nan,
            nan,
            nan,
        )
    ]
    for name, run in runs.items():
        sr = met.sharpe_difference_test(run.returns, bench.returns)
        ce = met.ceq_difference_test(run.returns, bench.returns, GAMMA)
        net = met.net_returns(run.returns, run.trades, COST)
        cells.append(
            Cell(
                sample=sample.key,
                dataset=dataset.key,
                rule=name,
                months=n_oos,
                first_month=first,
                last_month=last,
                sharpe=met.sharpe_ratio(run.returns),
                sharpe_z=sr.statistic,
                sharpe_p=sr.p_value,
                ceq=met.certainty_equivalent(run.returns, GAMMA),
                ceq_z=ce.statistic,
                ceq_p=ce.p_value,
                turnover=run.turnover,
                turnover_rel=run.turnover / bench.turnover,
                return_loss=met.return_loss(bench_net, net),
            )
        )
    return cells


def run_all(
    datasets: dict[str, Dataset],
    samples: tuple[Sample, ...],
    window: int = paper.WINDOW,
    rules: tuple[str, ...] = tuple(bt.RULES),
) -> list[Cell]:
    """Every sample, dataset and rule, in a fixed order."""
    return [
        cell
        for sample in samples
        for dataset in datasets.values()
        for cell in evaluate(dataset, sample, window, rules)
    ]


# --- text output ----------------------------------------------------------------------------------


def _num(x: float, digits: int = 4) -> str:
    return "" if np.isnan(x) else f"{x:.{digits}f}"


def metrics_csv(cells: list[Cell]) -> str:
    """Long table with one line per cell."""
    head = (
        "sample,dataset,rule,months,first_month,last_month,sharpe,sharpe_z,sharpe_p,"
        "ceq,ceq_z,ceq_p,turnover,turnover_rel,return_loss"
    )
    lines = [head]
    for c in cells:
        lines.append(
            ",".join(
                [
                    c.sample,
                    c.dataset,
                    c.rule,
                    str(c.months),
                    str(c.first_month),
                    str(c.last_month),
                    _num(c.sharpe, 6),
                    _num(c.sharpe_z, 4),
                    _num(c.sharpe_p, 4),
                    _num(c.ceq, 6),
                    _num(c.ceq_z, 4),
                    _num(c.ceq_p, 4),
                    _num(c.turnover, 6),
                    _num(c.turnover_rel, 4),
                    _num(c.return_loss, 6),
                ]
            )
        )
    return "\n".join(lines) + "\n"


def _lookup(cells: list[Cell], sample: str) -> dict[tuple[str, str], Cell]:
    return {(c.dataset, c.rule): c for c in cells if c.sample == sample}


def _md_table(header: list[str], rows: list[list[str]]) -> str:
    out = ["| " + " | ".join(header) + " |", "|" + "|".join(["---"] * len(header)) + "|"]
    out += ["| " + " | ".join(row) + " |" for row in rows]
    return "\n".join(out) + "\n"


def sample_tables_md(cells: list[Cell], sample: Sample, datasets: dict[str, Dataset]) -> str:
    """Three tables in the layout of the paper's Tables 3, 4 and 5 for one sample."""
    table = _lookup(cells, sample.key)
    keys = [k for k in datasets if (k, "ew") in table]
    rules = [r for r in ("mv-is", *bt.RULES) if (keys[0], r) in table]
    any_cell = table[(keys[0], "ew")]
    header = ["rule", *[f"{datasets[k].label} (N={len(datasets[k].assets)})" for k in keys]]

    def body(fmt: object) -> list[list[str]]:
        assert callable(fmt)
        return [[r, *[fmt(table[(k, r)]) for k in keys]] for r in rules]

    def sharpe(c: Cell) -> str:
        return _num(c.sharpe) if c.rule in ("ew", "mv-is") else f"{c.sharpe:.4f} ({c.sharpe_p:.2f})"

    def ceq(c: Cell) -> str:
        return _num(c.ceq) if c.rule in ("ew", "mv-is") else f"{c.ceq:.4f} ({c.ceq_p:.2f})"

    def turnover(c: Cell) -> str:
        if c.rule == "mv-is":
            return "-"
        return f"{c.turnover:.4f}" if c.rule == "ew" else f"{c.turnover_rel:.2f}"

    def loss(c: Cell) -> str:
        return "-" if c.rule in ("ew", "mv-is") else f"{c.return_loss:.4f}"

    parts = [
        f"# {sample.label}\n",
        f"Out-of-sample months: {any_cell.first_month} to {any_cell.last_month} "
        f"({any_cell.months} months). All figures are monthly. In parentheses: one-sided "
        "p-value of the difference from `ew`.\n",
        "## Sharpe ratio\n",
        _md_table(header, body(sharpe)),
        f"## Certainty-equivalent return (risk aversion {GAMMA:g})\n",
        _md_table(header, body(ceq)),
        "## Turnover (`ew` in absolute terms, other rules relative to `ew`)\n",
        _md_table(header, body(turnover)),
        f"## Return-loss against `ew` per month, {COST * 1e4:.0f} basis points of cost\n",
        _md_table(header, body(loss)),
    ]
    return "\n".join(parts)


_PAPER_METRICS = (
    ("sharpe", paper.SHARPE, 4),
    ("sharpe_p", paper.SHARPE_P, 2),
    ("ceq", paper.CEQ, 4),
    ("ceq_p", paper.CEQ_P, 2),
    ("turnover", paper.TURNOVER, 4),
    ("return_loss", paper.RETURN_LOSS, 4),
)


def _ours(cell: Cell, metric: str) -> float:
    if metric == "turnover":
        return cell.turnover if cell.rule == "ew" else cell.turnover_rel
    return float(getattr(cell, metric))


def paper_comparison_csv(cells: list[Cell], sample_key: str) -> str:
    """Every published cell this project can compute, next to the figure computed here."""
    table = _lookup(cells, sample_key)
    lines = ["metric,rule,dataset,paper,occam,difference"]
    for metric, published, digits in _PAPER_METRICS:
        for rule, row in published.items():
            for dataset, value in zip(paper.DATASETS, row, strict=True):
                if value is None or (dataset, rule) not in table:
                    continue
                ours = _ours(table[(dataset, rule)], metric)
                lines.append(
                    f"{metric},{rule},{dataset},{value:.{digits}f},{ours:.{digits}f},"
                    f"{ours - value:.{digits}f}"
                )
    return "\n".join(lines) + "\n"


def summary_md(cells: list[Cell], samples: tuple[Sample, ...]) -> str:
    """Counts behind the headline: how often each rule beats equal weight, and how often at 5%."""
    lines = [
        "# Scoreboard against equal weight\n",
        f"A cell is one rule on one dataset. `sig` means one-sided p below {ALPHA:g}. "
        "`mv-is` is left out: it is not an investable rule.\n",
    ]
    for sample in samples:
        mine = [c for c in cells if c.sample == sample.key and c.rule not in ("ew", "mv-is")]
        if not mine:
            continue
        rules = list(dict.fromkeys(c.rule for c in mine))
        n_data = len({c.dataset for c in mine})
        rows = []
        for rule in rules:
            rc = [c for c in mine if c.rule == rule]
            rows.append(
                [
                    rule,
                    str(sum(c.sharpe_z > 0 for c in rc)),
                    str(sum(c.sharpe_z > 0 and c.sharpe_p < ALPHA for c in rc)),
                    str(sum(c.sharpe_z < 0 and c.sharpe_p < ALPHA for c in rc)),
                    str(sum(c.ceq_z > 0 and c.ceq_p < ALPHA for c in rc)),
                    str(sum(c.ceq_z < 0 and c.ceq_p < ALPHA for c in rc)),
                    str(sum(c.turnover_rel < 1 for c in rc)),
                    str(sum(c.return_loss < 0 for c in rc)),
                ]
            )
        any_cell = mine[0]
        lines.append(
            f"## {sample.label}: {any_cell.first_month} to {any_cell.last_month}, "
            f"{n_data} datasets\n"
        )
        lines.append(
            _md_table(
                [
                    "rule",
                    "Sharpe above ew",
                    "sig above",
                    "sig below",
                    "CEQ sig above",
                    "CEQ sig below",
                    "turnover below ew",
                    "beats ew after costs",
                ],
                rows,
            )
        )
    return "\n".join(lines)


def write_tables(
    out_dir: Path, cells: list[Cell], samples: tuple[Sample, ...], datasets: dict[str, Dataset]
) -> list[Path]:
    """Write every table. Returns the paths in the order written."""
    out_dir.mkdir(parents=True, exist_ok=True)
    files: dict[str, str] = {
        "metrics.csv": metrics_csv(cells),
        "summary.md": summary_md(cells, samples),
    }
    for sample in samples:
        files[f"tables_{sample.key}.md"] = sample_tables_md(cells, sample, datasets)
    if any(s.key == "paper" for s in samples):
        files["paper_comparison.csv"] = paper_comparison_csv(cells, "paper")
    paths = []
    for name, text in files.items():
        path = out_dir / name
        path.write_text(text, encoding="utf-8", newline="\n")
        paths.append(path)
    return paths
