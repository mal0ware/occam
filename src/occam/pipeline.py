"""Turn datasets into the result tables. Pure functions: same input, same text out."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from zlib import crc32

import numpy as np

from occam import backtest as bt
from occam import metrics as met
from occam import paper, robust
from occam.data import Dataset

GAMMA = 1.0  # risk aversion in the certainty equivalent, as in the paper's tables
COST = 0.005  # 50 basis points per unit traded, as in the paper's Table 5
ALPHA = 0.05  # level used when counting "significant" cells; every test is two-sided
SEED = 20261123  # base seed of the bootstrap
OPTIMIZERS = ("mv", "min", "min-c", "lw-min", "erc", "hrp")  # the rules counted in the headline


@dataclass(frozen=True)
class Sample:
    """A stretch of months. Evaluation starts ``window`` months after ``start``."""

    key: str
    label: str
    start: int
    end: int
    window: int = paper.WINDOW

    @property
    def tag(self) -> str:
        """Name used in file names: the key, with the window added when it is not the paper's."""
        return self.key if self.window == paper.WINDOW else f"{self.key}_m{self.window}"

    @property
    def title(self) -> str:
        return f"{self.label}, {self.window}-month window"


@dataclass(frozen=True)
class Cell:
    """One rule on one dataset in one sample.

    Every p-value is two-sided. ``jk`` is the Jobson-Korkie test with Memmel's correction that
    the paper uses; ``hac`` and ``boot_p`` are the two Ledoit-Wolf (2008) tests. ``boot_p``
    holds one bootstrap p-value per block length in ``robust.BLOCKS``.
    """

    sample: str  # the sample's tag
    window: int
    dataset: str
    rule: str
    months: int
    first_month: int
    last_month: int
    sharpe: float
    sharpe_net: float
    jk_z: float
    jk_p: float
    hac_z: float
    hac_p: float
    boot_p: tuple[float, ...]
    ceq: float
    ceq_z: float
    ceq_p: float
    turnover: float
    turnover_rel: float
    return_loss: float

    @property
    def boot(self) -> float:
        """Bootstrap p-value at the default block length."""
        return self.boot_p[robust.BLOCKS.index(robust.BLOCK)]


def evaluate(
    dataset: Dataset,
    sample: Sample,
    rules: tuple[str, ...],
    draws: int = robust.DRAWS,
) -> list[Cell]:
    """Run every rule on one dataset and measure it against equal weight.

    The first cell is ``mv-is``, the mean-variance portfolio estimated on the whole sample
    (eq. 13). It has no turnover and is not tested.

    The bootstrap seed depends on the sample, the dataset and the block length, not on the
    rule, so every rule on a dataset is resampled with the same blocks.
    """
    window = sample.window
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
            sample=sample.tag,
            window=window,
            dataset=dataset.key,
            rule="mv-is",
            months=n_oos,
            first_month=first,
            last_month=last,
            sharpe=met.sharpe_ratio(ins),
            sharpe_net=nan,
            jk_z=nan,
            jk_p=nan,
            hac_z=nan,
            hac_p=nan,
            boot_p=tuple(nan for _ in robust.BLOCKS),
            ceq=met.certainty_equivalent(ins, GAMMA),
            ceq_z=nan,
            ceq_p=nan,
            turnover=nan,
            turnover_rel=nan,
            return_loss=nan,
        )
    ]
    for name, run in runs.items():
        jk = met.sharpe_difference_test(run.returns, bench.returns)
        hac = robust.hac_sharpe_test(run.returns, bench.returns)
        boot = []
        for block in robust.BLOCKS:
            seed = [SEED, crc32(f"{sample.tag}/{dataset.key}/{block}".encode())]
            rng = np.random.default_rng(seed)
            boot.append(
                robust.bootstrap_sharpe_test(run.returns, bench.returns, rng, block, draws).p_value
            )
        ce = met.ceq_difference_test(run.returns, bench.returns, GAMMA)
        net = met.net_returns(run.returns, run.trades, COST)
        cells.append(
            Cell(
                sample=sample.tag,
                window=window,
                dataset=dataset.key,
                rule=name,
                months=n_oos,
                first_month=first,
                last_month=last,
                sharpe=met.sharpe_ratio(run.returns),
                sharpe_net=met.sharpe_ratio(net),
                jk_z=jk.statistic,
                jk_p=jk.p_value,
                hac_z=hac.statistic,
                hac_p=hac.p_value,
                boot_p=tuple(boot),
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
    rules: tuple[str, ...] = tuple(bt.RULES),
    draws: int = robust.DRAWS,
) -> list[Cell]:
    """Every sample, dataset and rule, in a fixed order."""
    return [
        cell
        for sample in samples
        for dataset in datasets.values()
        for cell in evaluate(dataset, sample, rules, draws)
    ]


# --- text output ----------------------------------------------------------------------------------


def _num(x: float, digits: int = 4) -> str:
    return "" if np.isnan(x) else f"{x:.{digits}f}"


def metrics_csv(cells: list[Cell]) -> str:
    """Long table with one line per cell. All p-values are two-sided."""
    head = (
        "sample,window,dataset,rule,months,first_month,last_month,sharpe,sharpe_net,"
        "jk_z,jk_p,hac_z,hac_p,"
        + ",".join(f"boot_p_b{b}" for b in robust.BLOCKS)
        + ",ceq,ceq_z,ceq_p,turnover,turnover_rel,return_loss"
    )
    lines = [head]
    for c in cells:
        lines.append(
            ",".join(
                [
                    c.sample,
                    str(c.window),
                    c.dataset,
                    c.rule,
                    str(c.months),
                    str(c.first_month),
                    str(c.last_month),
                    _num(c.sharpe, 6),
                    _num(c.sharpe_net, 6),
                    _num(c.jk_z, 4),
                    _num(c.jk_p, 4),
                    _num(c.hac_z, 4),
                    _num(c.hac_p, 4),
                    *[_num(p, 4) for p in c.boot_p],
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
    """Tables in the layout of the paper's Tables 3, 4 and 5 for one sample."""
    table = _lookup(cells, sample.tag)
    keys = [k for k in datasets if (k, "ew") in table]
    rules = [r for r in ("mv-is", *bt.RULES) if (keys[0], r) in table]
    any_cell = table[(keys[0], "ew")]
    header = ["rule", *[f"{datasets[k].label} (N={len(datasets[k].assets)})" for k in keys]]

    def body(fmt: object) -> list[list[str]]:
        assert callable(fmt)
        return [[r, *[fmt(table[(k, r)]) for k in keys]] for r in rules]

    def sharpe(c: Cell) -> str:
        if c.rule in ("ew", "mv-is"):
            return _num(c.sharpe)
        return f"{c.sharpe:.4f} ({c.jk_p:.2f}, {c.boot:.2f})"

    def sharpe_net(c: Cell) -> str:
        return "-" if c.rule == "mv-is" else f"{c.sharpe_net:.4f}"

    def ceq(c: Cell) -> str:
        return _num(c.ceq) if c.rule in ("ew", "mv-is") else f"{c.ceq:.4f} ({c.ceq_p:.2f})"

    def turnover(c: Cell) -> str:
        if c.rule == "mv-is":
            return "-"
        return f"{c.turnover:.4f}" if c.rule == "ew" else f"{c.turnover_rel:.2f}"

    def loss(c: Cell) -> str:
        return "-" if c.rule in ("ew", "mv-is") else f"{c.return_loss:.4f}"

    parts = [
        f"# {sample.title}\n",
        f"Out-of-sample months: {any_cell.first_month} to {any_cell.last_month} "
        f"({any_cell.months} months). All figures are monthly. All p-values are two-sided "
        "and test the difference from `ew`.\n",
        "## Sharpe ratio\n",
        "In parentheses: the p-value of the Jobson-Korkie test with Memmel's correction, then "
        f"the p-value of the Ledoit-Wolf studentised bootstrap with blocks of {robust.BLOCK} "
        "months.\n",
        _md_table(header, body(sharpe)),
        f"## Sharpe ratio net of {COST * 1e4:.0f} basis points of cost\n",
        _md_table(header, body(sharpe_net)),
        f"## Certainty-equivalent return (risk aversion {GAMMA:g})\n",
        "In parentheses: the p-value of the delta-method test in the paper's footnote 18.\n",
        _md_table(header, body(ceq)),
        "## Turnover (`ew` in absolute terms, other rules relative to `ew`)\n",
        _md_table(header, body(turnover)),
        f"## Return-loss against `ew` per month, {COST * 1e4:.0f} basis points of cost\n",
        _md_table(header, body(loss)),
    ]
    return "\n".join(parts)


_PAPER_METRICS = (
    ("sharpe", paper.SHARPE, 4),
    ("sharpe_p_one_sided", paper.SHARPE_P, 2),
    ("ceq", paper.CEQ, 4),
    ("ceq_p_one_sided", paper.CEQ_P, 2),
    ("turnover", paper.TURNOVER, 4),
    ("return_loss", paper.RETURN_LOSS, 4),
)


def _ours(cell: Cell, metric: str) -> float:
    if metric == "turnover":
        return cell.turnover if cell.rule == "ew" else cell.turnover_rel
    if metric == "sharpe_p_one_sided":
        return 0.5 * cell.jk_p
    if metric == "ceq_p_one_sided":
        return 0.5 * cell.ceq_p
    return float(getattr(cell, metric))


def paper_comparison_csv(cells: list[Cell], sample_tag: str) -> str:
    """Every published cell this project can compute, next to the figure computed here.

    The paper's p-values are read as one-sided, so the p-values here are halved for this file.
    """
    table = _lookup(cells, sample_tag)
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


def holm(p_values: list[float], alpha: float = ALPHA) -> list[bool]:
    """Holm's step-down rule: which hypotheses are rejected with family-wise level ``alpha``."""
    order = sorted(range(len(p_values)), key=lambda i: p_values[i])
    rejected = [False] * len(p_values)
    for rank, i in enumerate(order):
        if p_values[i] > alpha / (len(p_values) - rank):
            break
        rejected[i] = True
    return rejected


def _count(cells: list[Cell], keep: list[bool]) -> tuple[int, int]:
    above = sum(k and c.jk_z > 0 for c, k in zip(cells, keep, strict=True))
    below = sum(k and c.jk_z < 0 for c, k in zip(cells, keep, strict=True))
    return above, below


def headline_rows(cells: list[Cell]) -> list[list[str]]:
    """Counts of significant Sharpe differences among the optimizers, one row per test."""
    mine = [c for c in cells if c.rule in OPTIMIZERS]
    tests: list[tuple[str, list[float]]] = [
        ("Jobson-Korkie with Memmel's correction", [c.jk_p for c in mine]),
        ("Ledoit-Wolf HAC", [c.hac_p for c in mine]),
    ]
    for i, block in enumerate(robust.BLOCKS):
        name = f"Ledoit-Wolf bootstrap, blocks of {block}"
        tests.append((name, [c.boot_p[i] for c in mine]))
    rows = []
    for name, ps in tests:
        above, below = _count(mine, [p < ALPHA for p in ps])
        rows.append([name, str(above), str(below)])
    boot = [c.boot for c in mine]
    above, below = _count(mine, holm(boot))
    rows.append([f"Ledoit-Wolf bootstrap, blocks of {robust.BLOCK}, Holm", str(above), str(below)])
    return rows


def summary_md(cells: list[Cell], samples: tuple[Sample, ...]) -> str:
    """Counts behind the headline: how often each rule beats equal weight, and how often at 5%."""
    lines = [
        "# Scoreboard against equal weight\n",
        f"A cell is one rule on one dataset. Every test is two-sided at the {ALPHA:g} level. "
        "`JK` is the Jobson-Korkie test with Memmel's correction, which assumes independent "
        "normal returns. `boot` is the Ledoit-Wolf studentised circular block bootstrap with "
        f"blocks of {robust.BLOCK} months and {robust.DRAWS} draws in the full run. "
        "`mv-is` is left out: it is not an investable rule.\n",
    ]
    for sample in samples:
        mine = [c for c in cells if c.sample == sample.tag and c.rule not in ("ew", "mv-is")]
        if not mine:
            continue
        n_data = len({c.dataset for c in mine})
        n_opt = sum(c.rule in OPTIMIZERS for c in mine)
        any_cell = mine[0]
        lines.append(
            f"## {sample.title}: {any_cell.first_month} to {any_cell.last_month}, "
            f"{n_data} datasets\n"
        )
        lines.append(
            f"Optimizers only ({', '.join(f'`{r}`' for r in OPTIMIZERS)}): {n_opt} comparisons "
            "with `ew`. If every null were true and the comparisons were independent, "
            f"{n_opt * ALPHA:.1f} would be significant by chance, {n_opt * ALPHA / 2:.1f} in "
            "each direction. The Holm row controls the chance of any false rejection among the "
            f"{n_opt} at {ALPHA:g}.\n"
        )
        lines.append(
            _md_table(
                ["test of the Sharpe difference", "significantly above ew", "significantly below"],
                headline_rows(mine),
            )
        )
        rows = []
        for rule in dict.fromkeys(c.rule for c in mine):
            rc = [c for c in mine if c.rule == rule]
            rows.append(
                [
                    rule,
                    str(sum(c.jk_z > 0 for c in rc)),
                    str(sum(c.jk_z > 0 and c.jk_p < ALPHA for c in rc)),
                    str(sum(c.jk_z < 0 and c.jk_p < ALPHA for c in rc)),
                    str(sum(c.jk_z > 0 and c.boot < ALPHA for c in rc)),
                    str(sum(c.jk_z < 0 and c.boot < ALPHA for c in rc)),
                    str(sum(c.ceq_z > 0 and c.ceq_p < ALPHA for c in rc)),
                    str(sum(c.ceq_z < 0 and c.ceq_p < ALPHA for c in rc)),
                    str(sum(c.turnover_rel < 1 for c in rc)),
                    str(sum(c.return_loss < 0 for c in rc)),
                ]
            )
        lines.append("By rule, out of the datasets:\n")
        lines.append(
            _md_table(
                [
                    "rule",
                    "Sharpe above ew",
                    "JK sig above",
                    "JK sig below",
                    "boot sig above",
                    "boot sig below",
                    "CEQ sig above",
                    "CEQ sig below",
                    "turnover below ew",
                    "net Sharpe above ew",
                ],
                rows,
            )
        )
    return "\n".join(lines)


def factor_weights_md(datasets: dict[str, Dataset], samples: tuple[Sample, ...]) -> str:
    """Where the FF-4-factor portfolios put their money, and what the factors earned.

    For each sample: the mean monthly return of MKT, SMB, HML and UMD over the out-of-sample
    months, and each rule's average total weight in the three long-short factors.
    """
    data = datasets["ff4"]
    factors = ("MKT", "SMB", "HML", "UMD")
    zero_cost = [data.assets.index(a) for a in factors[1:]]
    rules = [r for r in bt.RULES if r not in ("vw", "bl")]
    means: list[list[str]] = []
    shares: dict[str, list[str]] = {r: [r] for r in rules}
    for sample in samples:
        cut = data.window(sample.start, sample.end)
        oos = cut.returns[sample.window :]
        first, last = int(cut.months[sample.window]), int(cut.months[-1])
        means.append(
            [
                f"{sample.title}: {first} to {last}",
                *[f"{100 * oos[:, cut.assets.index(a)].mean():.2f}" for a in factors],
            ]
        )
        for rule in rules:
            run = bt.rolling_backtest(cut.returns, sample.window, bt.RULES[rule], cut.market_index)
            held = run.weights[:-1][:, zero_cost].sum(axis=1)
            shares[rule].append(f"{100 * held.mean():.1f}")
    parts = [
        "# FF-4-factor: factor returns and factor weights\n",
        "## Mean monthly return over the out-of-sample months, percent\n",
        _md_table(["sample", *factors], means),
        "## Average share of the portfolio in SMB, HML and UMD together, percent\n",
        "The average over the out-of-sample months of the summed weights in the three "
        "long-short factors. `vw` and `bl` hold only the market, so their share is zero.\n",
        _md_table(["rule", *[s.title for s in samples]], list(shares.values())),
    ]
    return "\n".join(parts)


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
        files[f"tables_{sample.tag}.md"] = sample_tables_md(cells, sample, datasets)
    if any(s.tag == "paper" for s in samples):
        files["paper_comparison.csv"] = paper_comparison_csv(cells, "paper")
    if "ff4" in datasets:
        files["factor_weights.md"] = factor_weights_md(datasets, samples)
    paths = []
    for name, text in files.items():
        path = out_dir / name
        path.write_text(text, encoding="utf-8", newline="\n")
        paths.append(path)
    return paths
