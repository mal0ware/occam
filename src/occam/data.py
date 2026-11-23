"""Read Kenneth R. French Data Library files and assemble the datasets of the paper.

The library ships each file as a zip holding one CSV. A CSV holds several tables one after
another, each introduced by a title line (the factor files have no title for the first
table). Returns are in percent; this module converts them to decimals.
"""

from __future__ import annotations

import io
import zipfile
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from numpy.typing import NDArray

Array = NDArray[np.float64]

FACTORS_FILE = "F-F_Research_Data_Factors_CSV.zip"
MOMENTUM_FILE = "F-F_Momentum_Factor_CSV.zip"
INDUSTRY_FILE = "10_Industry_Portfolios_CSV.zip"
SIZE_BM_FILE = "25_Portfolios_5x5_CSV.zip"
MONTHLY_VW = "Average Value Weighted Returns -- Monthly"

MISSING = (-99.99, -999.0)


@dataclass(frozen=True)
class Table:
    """One monthly table from a French file. ``months`` holds integers of the form YYYYMM."""

    months: NDArray[np.int64]
    columns: tuple[str, ...]
    values: Array  # decimals, NaN where the file marks a value as missing

    def column(self, name: str) -> Array:
        return np.asarray(self.values[:, self.columns.index(name)], dtype=np.float64)


@dataclass(frozen=True)
class Dataset:
    """Monthly excess returns on N risky assets, the last of which may be the market."""

    key: str
    label: str
    months: NDArray[np.int64]
    assets: tuple[str, ...]
    returns: Array  # shape (T, N), decimals
    market_index: int

    def window(self, start: int, end: int) -> Dataset:
        """Rows with ``start <= month <= end`` (both YYYYMM)."""
        keep = (self.months >= start) & (self.months <= end)
        return Dataset(
            self.key,
            self.label,
            self.months[keep],
            self.assets,
            self.returns[keep],
            self.market_index,
        )


def read_zip_text(path: Path) -> str:
    """Text of the single CSV inside a French zip file."""
    with zipfile.ZipFile(path) as archive:
        name = archive.namelist()[0]
        with archive.open(name) as handle:
            return io.TextIOWrapper(handle, encoding="latin-1").read()


def vintage_line(text: str) -> str:
    """The first line of a file, which names the CRSP database it was built from."""
    return text.splitlines()[0].strip()


def parse_table(text: str, title: str | None = None) -> Table:
    """Parse one monthly table.

    The table starts at the first header line (a line beginning with a comma) after
    ``title``, or after the top of the file when ``title`` is ``None``. It ends at the first
    line whose first field is not a six-digit month.
    """
    lines = text.splitlines()
    start = 0
    if title is not None:
        matches = [i for i, line in enumerate(lines) if line.strip() == title]
        if not matches:
            raise ValueError(f"table title not found: {title!r}")
        start = matches[0] + 1
    header_at = next((i for i in range(start, len(lines)) if lines[i].startswith(",")), None)
    if header_at is None:
        raise ValueError("no header line found")
    columns = tuple(field.strip() for field in lines[header_at].split(",")[1:])
    months: list[int] = []
    rows: list[list[float]] = []
    for line in lines[header_at + 1 :]:
        fields = [field.strip() for field in line.split(",")]
        if len(fields[0]) != 6 or not fields[0].isdigit():
            break
        if len(fields) != len(columns) + 1:
            raise ValueError(
                f"row {fields[0]} has {len(fields) - 1} values, expected {len(columns)}"
            )
        months.append(int(fields[0]))
        rows.append([float(field) for field in fields[1:]])
    if not rows:
        raise ValueError("table has no monthly rows")
    values = np.array(rows, dtype=np.float64)
    for marker in MISSING:
        values[values == marker] = np.nan
    return Table(np.array(months, dtype=np.int64), columns, values / 100.0)


def _align(tables: list[Table]) -> tuple[NDArray[np.int64], list[Array]]:
    """Restrict tables to the months they all share, in order."""
    common = tables[0].months
    for table in tables[1:]:
        common = np.intersect1d(common, table.months)
    aligned = [table.values[np.isin(table.months, common)] for table in tables]
    return common, aligned


def build_datasets(data_dir: Path) -> dict[str, Dataset]:
    """The four datasets of DGU Table 2 that come from the French library, at full length.

    * ``industry``: ten industry portfolios plus the market (their #2, N = 11)
    * ``mkt_smb_hml``: SMB, HML and the market (their #4, N = 3)
    * ``ff1``: twenty size and book-to-market portfolios plus the market (their #5, N = 21)
    * ``ff4``: the same twenty plus MKT, SMB, HML and UMD (their #7, N = 24)

    Portfolio returns are in excess of the one-month bill rate from the factor file. SMB, HML
    and UMD are zero-cost and enter as they are. As in the paper, the five portfolios of the
    largest size quintile are left out of the 25.
    """
    factors = parse_table(read_zip_text(data_dir / FACTORS_FILE))
    momentum = parse_table(read_zip_text(data_dir / MOMENTUM_FILE))
    industry = parse_table(read_zip_text(data_dir / INDUSTRY_FILE), MONTHLY_VW)
    size_bm = parse_table(read_zip_text(data_dir / SIZE_BM_FILE), MONTHLY_VW)

    if len(industry.columns) != 10 or len(size_bm.columns) != 25:
        raise ValueError("unexpected number of portfolios in a French file")
    largest = size_bm.columns[20:]
    if not all(name.startswith(("BIG", "ME5")) for name in largest):
        raise ValueError(f"expected the last five columns to be the largest firms, got {largest}")
    twenty = Table(size_bm.months, size_bm.columns[:20], size_bm.values[:, :20])

    months, (fac, mom, ind, sbm) = _align([factors, momentum, industry, twenty])
    cols = factors.columns
    mkt = fac[:, cols.index("Mkt-RF")]
    smb = fac[:, cols.index("SMB")]
    hml = fac[:, cols.index("HML")]
    rf = fac[:, cols.index("RF")]
    umd = mom[:, 0]

    def make(key: str, label: str, assets: tuple[str, ...], returns: Array, market: int) -> Dataset:
        return Dataset(key, label, months, assets, np.ascontiguousarray(returns), market)

    ind_x = ind - rf[:, None]
    sbm_x = sbm - rf[:, None]
    return {
        "industry": make(
            "industry", "Industry", (*industry.columns, "MKT"), np.column_stack([ind_x, mkt]), 10
        ),
        "mkt_smb_hml": make(
            "mkt_smb_hml", "MKT/SMB/HML", ("SMB", "HML", "MKT"), np.column_stack([smb, hml, mkt]), 2
        ),
        "ff1": make(
            "ff1", "FF-1-factor", (*twenty.columns, "MKT"), np.column_stack([sbm_x, mkt]), 20
        ),
        "ff4": make(
            "ff4",
            "FF-4-factor",
            (*twenty.columns, "MKT", "SMB", "HML", "UMD"),
            np.column_stack([sbm_x, mkt, smb, hml, umd]),
            20,
        ),
    }
