"""Figures. One claim per chart."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.axes import Axes
from matplotlib.patches import Rectangle

from occam import backtest as bt
from occam.data import Dataset
from occam.pipeline import ALPHA, Cell, Sample

SURFACE = "#fcfcfb"
INK = "#0b0b0b"
MUTED = "#52514e"
GRID = "#e4e3df"
SERIES = ("#2a78d6", "#eb6834")  # first two categorical slots: paper sample, later sample


def _span(cell: Cell) -> str:
    """Out-of-sample months of a cell as ``YYYY-MM to YYYY-MM``."""
    a, b = cell.first_month, cell.last_month
    return f"{a // 100}-{a % 100:02d} to {b // 100}-{b % 100:02d}"


def _style(ax: Axes) -> None:
    ax.set_facecolor(SURFACE)
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    ax.spines["bottom"].set_color(GRID)
    ax.tick_params(colors=MUTED, length=0, labelsize=8)
    ax.grid(axis="x", color=GRID, linewidth=0.6)
    ax.set_axisbelow(True)


def sharpe_figure(
    path: Path, cells: list[Cell], samples: tuple[Sample, ...], datasets: dict[str, Dataset]
) -> None:
    """Sharpe ratio of each rule minus that of equal weight, one panel per dataset.

    Bars to the right of zero beat 1/N. A solid bar is significant at the 5% level in the
    two-sided Ledoit-Wolf bootstrap test; a pale bar is not.
    """
    rules = [r for r in bt.RULES if r not in ("ew", "bl")]
    table = {(c.sample, c.dataset, c.rule): c for c in cells}
    keys = [k for k in datasets if (samples[0].tag, k, "ew") in table]
    fig, axes = plt.subplots(1, len(keys), figsize=(11, 3.9), sharey=True, facecolor=SURFACE)
    axes = [axes] if len(keys) == 1 else list(axes)
    height = 0.8 / len(samples)
    for ax, key in zip(axes, keys, strict=True):
        _style(ax)
        for s, sample in enumerate(samples):
            base = table[(sample.tag, key, "ew")].sharpe
            for i, rule in enumerate(rules):
                cell = table[(sample.tag, key, rule)]
                y = i + (s - (len(samples) - 1) / 2) * height
                ax.barh(
                    y,
                    cell.sharpe - base,
                    height=height * 0.88,
                    color=SERIES[s],
                    alpha=1.0 if cell.boot < ALPHA else 0.35,
                    linewidth=0,
                )
        ax.axvline(0, color=INK, linewidth=0.8)
        ax.set_title(
            f"{datasets[key].label}  (N = {len(datasets[key].assets)})",
            fontsize=9,
            color=INK,
            loc="left",
        )
        ax.set_xlim(-0.26, 0.26)
        ax.set_xticks([-0.2, -0.1, 0, 0.1, 0.2])
        ax.set_yticks(range(len(rules)))
        ax.set_yticklabels(rules, fontsize=9, color=INK)
    axes[0].invert_yaxis()
    handles = []
    for s, sample in enumerate(samples):
        c0 = table[(sample.tag, keys[0], "ew")]
        label = f"{sample.label}, {_span(c0)}"
        handles.append(
            Rectangle((0, 0), 1, 1, color=SERIES[s], label=f"{label}, bootstrap p < {ALPHA:g}")
        )
        handles.append(
            Rectangle((0, 0), 1, 1, color=SERIES[s], alpha=0.35, label=f"{label}, not significant")
        )
    fig.legend(
        handles=handles,
        loc="lower center",
        ncol=2,
        frameon=False,
        fontsize=8,
        labelcolor=MUTED,
    )
    fig.suptitle(
        "Monthly out-of-sample Sharpe ratio minus the Sharpe ratio of 1/N",
        fontsize=11,
        color=INK,
        x=0.01,
        ha="left",
    )
    fig.supxlabel(
        "difference in monthly Sharpe ratio (right of zero beats 1/N)",
        fontsize=8.5,
        color=MUTED,
        y=0.125,
    )
    fig.tight_layout(rect=(0, 0.13, 1, 1.0))
    fig.savefig(path, dpi=160, facecolor=SURFACE, metadata={"Software": None})
    plt.close(fig)


def turnover_figure(
    path: Path, cells: list[Cell], samples: tuple[Sample, ...], datasets: dict[str, Dataset]
) -> None:
    """Monthly turnover of each rule on a log scale, one panel per dataset."""
    rules = [r for r in bt.RULES if r not in ("vw", "bl")]
    table = {(c.sample, c.dataset, c.rule): c for c in cells}
    keys = [k for k in datasets if (samples[0].tag, k, "ew") in table]
    fig, axes = plt.subplots(1, len(keys), figsize=(11, 3.9), sharey=True, facecolor=SURFACE)
    axes = [axes] if len(keys) == 1 else list(axes)
    height = 0.8 / len(samples)
    for ax, key in zip(axes, keys, strict=True):
        _style(ax)
        for s, sample in enumerate(samples):
            for i, rule in enumerate(rules):
                y = i + (s - (len(samples) - 1) / 2) * height
                ax.barh(
                    y,
                    table[(sample.tag, key, rule)].turnover - 1e-3,
                    height=height * 0.88,
                    color=SERIES[s],
                    linewidth=0,
                    left=1e-3,
                )
        ax.set_xscale("log")
        ax.set_xlim(1e-3, 1e3)
        ax.set_title(
            f"{datasets[key].label}  (N = {len(datasets[key].assets)})",
            fontsize=9,
            color=INK,
            loc="left",
        )
        ax.set_yticks(range(len(rules)))
        ax.set_yticklabels(rules, fontsize=9, color=INK)
    axes[0].invert_yaxis()
    handles = []
    for s, sample in enumerate(samples):
        c0 = table[(sample.tag, keys[0], "ew")]
        handles.append(
            Rectangle(
                (0, 0),
                1,
                1,
                color=SERIES[s],
                label=f"{sample.label}, {_span(c0)}",
            )
        )
    fig.legend(
        handles=handles,
        loc="lower center",
        ncol=len(handles),
        frameon=False,
        fontsize=8,
        labelcolor=MUTED,
    )
    fig.suptitle(
        "Average monthly turnover (1 = the whole portfolio traded each month)",
        fontsize=11,
        color=INK,
        x=0.01,
        ha="left",
    )
    fig.supxlabel(
        "fraction of wealth traded per month, log scale", fontsize=8.5, color=MUTED, y=0.085
    )
    fig.tight_layout(rect=(0, 0.09, 1, 1.0))
    fig.savefig(path, dpi=160, facecolor=SURFACE, metadata={"Software": None})
    plt.close(fig)
