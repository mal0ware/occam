"""Rebuild every table and figure in ``results/`` from the files in ``data/``.

Usage:
    uv run python scripts/fetch_data.py
    uv run python scripts/reproduce.py            # everything, into results/
    uv run python scripts/reproduce.py --fast     # smoke run for CI, into results-fast/

The fast run uses the two smallest datasets and the paper's sample only. It exercises the
same code path and finishes in a few seconds. The full run takes well under a minute.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path

from occam import data, paper, pipeline, plots

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"

SAMPLES = (
    pipeline.Sample("paper", "Paper's sample", paper.SAMPLE_START, paper.SAMPLE_END),
    # Estimation starts 120 months before the first month after the paper's sample ends, so
    # every out-of-sample return in this block is one the authors could not have seen.
    pipeline.Sample("after", "After the paper's sample", 199412, 999912),
)
FAST_DATASETS = ("industry", "mkt_smb_hml")


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
    samples = SAMPLES
    if args.fast:
        datasets = {key: datasets[key] for key in FAST_DATASETS}
        samples = SAMPLES[:1]
    out_dir = args.out or (ROOT / ("results-fast" if args.fast else "results"))

    cells = pipeline.run_all(datasets, samples)
    written = pipeline.write_tables(out_dir, cells, samples, datasets)
    plots.sharpe_figure(out_dir / "fig_sharpe.png", cells, samples, datasets)
    plots.turnover_figure(out_dir / "fig_turnover.png", cells, samples, datasets)
    (out_dir / "data_vintage.json").write_text(
        json.dumps(vintage(datasets), indent=2) + "\n", encoding="utf-8", newline="\n"
    )

    for path in [
        *written,
        out_dir / "fig_sharpe.png",
        out_dir / "fig_turnover.png",
        out_dir / "data_vintage.json",
    ]:
        print(f"wrote {path.relative_to(ROOT) if path.is_relative_to(ROOT) else path}")
    print(f"done in {time.perf_counter() - started:.1f} s")
    return 0


if __name__ == "__main__":
    sys.exit(main())
