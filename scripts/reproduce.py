"""Rebuild every table and figure in ``results/`` from the files in ``data/``.

Usage:
    uv run python scripts/fetch_data.py
    uv run python scripts/reproduce.py            # everything, into results/
    uv run python scripts/reproduce.py --fast     # smoke run for CI, into results-fast/

The fast run uses the two smallest datasets, the paper's sample and far fewer bootstrap and
simulation draws. It exercises the same code path and finishes in a few seconds. The full
run takes a few minutes, nearly all of it in the bootstrap and the two simulations.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path

from occam import data, paper, pipeline, plots, robust, sensitivity, size

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"

FIRST_AFTER = 200412  # the month after the paper's data end
WINDOWS = (paper.WINDOW, 60)  # the paper's main window and the one in its appendix
FAST_DATASETS = ("industry", "mkt_smb_hml")
SIZE_SIMULATIONS = 500
SIZE_DRAWS = 499
SIZE_SEED = 20261125


def months_before(month: int, count: int) -> int:
    """The YYYYMM that lies ``count`` months before ``month``."""
    index = (month // 100) * 12 + (month % 100 - 1) - count
    return (index // 12) * 100 + index % 12 + 1


def samples_for(window: int) -> tuple[pipeline.Sample, pipeline.Sample]:
    """The paper's sample and the months after it, for one estimation window.

    In the second sample estimation starts ``window`` months before December 2004, so every
    out-of-sample return in it is one the authors could not have seen.
    """
    return (
        pipeline.Sample("paper", "Paper's sample", paper.SAMPLE_START, paper.SAMPLE_END, window),
        pipeline.Sample(
            "after",
            "After the paper's sample",
            months_before(FIRST_AFTER, window),
            999912,
            window,
        ),
    )


def vintage(datasets: dict[str, data.Dataset]) -> dict[str, object]:
    """What the tables were built from: file hashes, the CRSP vintage line, the last month."""
    files = {}
    for name in (data.FACTORS_FILE, data.MOMENTUM_FILE, data.INDUSTRY_FILE, data.SIZE_BM_FILE):
        path = DATA_DIR / name
        files[name] = {
            "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            "first_line": data.vintage_line(data.read_zip_text(path)),
        }
    first = next(iter(datasets.values()))
    return {
        "source": "Kenneth R. French Data Library",
        "first_month": int(first.months[0]),
        "last_month": int(first.months[-1]),
        "files": files,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--fast", action="store_true", help="small smoke run into results-fast/")
    parser.add_argument("--out", type=Path, default=None, help="output directory")
    args = parser.parse_args()

    if not (DATA_DIR / data.FACTORS_FILE).exists():
        print("data/ is empty. Run: uv run python scripts/fetch_data.py", file=sys.stderr)
        return 1

    started = time.perf_counter()
    datasets = data.build_datasets(DATA_DIR)
    samples = tuple(s for window in WINDOWS for s in samples_for(window))
    main_samples = samples[:2]
    draws, size_sims, noise_draws = robust.DRAWS, SIZE_SIMULATIONS, sensitivity.DRAWS
    if args.fast:
        datasets = {key: datasets[key] for key in FAST_DATASETS}
        samples = main_samples = samples[:1]
        draws, size_sims, noise_draws = 199, 10, 3
    out_dir = args.out or (ROOT / ("results-fast" if args.fast else "results"))

    cells = pipeline.run_all(datasets, samples, draws=draws)
    written = pipeline.write_tables(out_dir, cells, samples, datasets)
    plots.sharpe_figure(out_dir / "fig_sharpe.png", cells, main_samples, datasets)
    plots.turnover_figure(out_dir / "fig_turnover.png", cells, main_samples, datasets)

    def text(name: str, body: str) -> None:
        (out_dir / name).write_text(body, encoding="utf-8", newline="\n")
        written.append(out_dir / name)

    spread = sensitivity.spreads(datasets, draws=noise_draws)
    labels = {key: d.label for key, d in datasets.items()}
    text("revision_sensitivity.csv", sensitivity.spreads_csv(spread))
    text("revision_sensitivity.md", sensitivity.spreads_md(spread, labels, noise_draws))

    months = paper.SAMPLE_MONTHS - paper.WINDOW
    rates = [
        size.rejection_rates(name, months, size_sims, SIZE_SEED + i, robust.BLOCK, SIZE_DRAWS)
        for i, name in enumerate(size.PROCESSES)
    ]
    text("test_size.md", size.size_md(rates, robust.BLOCK, SIZE_DRAWS))
    text("data_vintage.json", json.dumps(vintage(datasets), indent=2) + "\n")

    for path in [*written, out_dir / "fig_sharpe.png", out_dir / "fig_turnover.png"]:
        print(f"wrote {path.relative_to(ROOT) if path.is_relative_to(ROOT) else path}")
    print(f"done in {time.perf_counter() - started:.1f} s")
    return 0


if __name__ == "__main__":
    sys.exit(main())
