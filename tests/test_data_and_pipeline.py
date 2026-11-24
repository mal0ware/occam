"""The French-file parser, the determinism of the tables, and checks against the paper.

The last group needs the downloaded files. When ``data/`` is empty those tests are skipped
and ``conftest.py`` prints a notice at the top and bottom of the run. With
``OCCAM_REQUIRE_DATA=1`` the run fails instead.

The tolerances in the last group were set after seeing the results, to the size of the gaps
between this project and the paper. They are regression guards on today's files. They do not
measure how close a replication ought to be.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from occam import data, paper, pipeline, robust, sensitivity

FRENCH_TEXT = """This file was created using the 202609 CRSP database.
It contains made-up numbers for a test.

  Average Value Weighted Returns -- Monthly
,Cnsmr,Manuf
196307,   1.00,  -2.00
196308,   3.50,  -99.99
196309,  -0.25,   0.75

  Average Equal Weighted Returns -- Monthly
,Cnsmr,Manuf
196307,   9.00,   9.00

Annual Factors: January-December
,Cnsmr,Manuf
1963,   5.00,   6.00
"""


def test_parse_named_table() -> None:
    table = data.parse_table(FRENCH_TEXT, data.MONTHLY_VW)
    assert table.months.tolist() == [196307, 196308, 196309]
    assert table.columns == ("Cnsmr", "Manuf")
    assert np.allclose(table.values[0], [0.01, -0.02])
    assert np.isnan(table.values[1, 1])  # -99.99 marks a missing value
    assert np.isclose(table.column("Cnsmr")[2], -0.0025)


def test_parse_stops_before_the_next_table_and_before_annual_rows() -> None:
    second = data.parse_table(FRENCH_TEXT, "Average Equal Weighted Returns -- Monthly")
    assert second.months.tolist() == [196307]
    untitled = data.parse_table(FRENCH_TEXT)
    assert untitled.months.tolist() == [196307, 196308, 196309]


def test_parse_errors() -> None:
    with pytest.raises(ValueError, match="title not found"):
        data.parse_table(FRENCH_TEXT, "No Such Table")
    with pytest.raises(ValueError, match="expected 2"):
        data.parse_table(",a,b\n196307, 1.0\n")


def test_vintage_line() -> None:
    assert data.vintage_line(FRENCH_TEXT) == "This file was created using the 202609 CRSP database."


def synthetic_datasets() -> dict[str, data.Dataset]:
    rng = np.random.default_rng(20261123)
    months = np.array([196301 + 100 * (i // 12) + i % 12 for i in range(200)], dtype=np.int64)
    out = {}
    for key, n in (("a", 4), ("b", 7)):
        r = rng.normal(0.004, 0.03, (200, 1)) + rng.normal(0.002, 0.02, (200, n))
        out[key] = data.Dataset(
            key, key.upper(), months, tuple(f"x{i}" for i in range(n)), r, n - 1
        )
    return out


SAMPLES = (
    pipeline.Sample("first", "First", 196301, 197212, 36),
    pipeline.Sample("rest", "Rest", 196801, 999912, 36),
)


def test_tables_are_identical_across_runs(tmp_path: Path) -> None:
    outputs = []
    for run in ("one", "two"):
        datasets = synthetic_datasets()
        cells = pipeline.run_all(datasets, SAMPLES, draws=99)
        paths = pipeline.write_tables(tmp_path / run, cells, SAMPLES, datasets)
        outputs.append({p.name: p.read_bytes() for p in paths})
    assert outputs[0].keys() == outputs[1].keys()
    assert set(outputs[0]) == {
        "metrics.csv",
        "summary.md",
        "tables_first_m36.md",
        "tables_rest_m36.md",
    }
    for name in outputs[0]:
        assert outputs[0][name] == outputs[1][name], name


def test_cells_cover_every_rule_and_the_right_months() -> None:
    datasets = synthetic_datasets()
    cells = pipeline.run_all(datasets, SAMPLES[:1], draws=99)
    assert len(cells) == 2 * (1 + 9)  # two datasets; mv-is plus nine rules
    first = cells[0]
    assert (first.rule, first.months, first.first_month, first.last_month) == (
        "mv-is",
        120 - 36,
        196601,
        197212,
    )
    ew = next(c for c in cells if c.rule == "ew")
    assert ew.jk_z == 0.0 and ew.jk_p == 1.0 and ew.turnover_rel == 1.0
    assert ew.hac_p == 1.0 and ew.boot == 1.0
    bl, vw = (next(c for c in cells if c.rule == r) for r in ("bl", "vw"))
    assert np.isclose(bl.sharpe, vw.sharpe) and np.isclose(bl.ceq, vw.ceq)
    assert np.allclose(bl.boot_p, vw.boot_p)  # the same blocks are resampled for every rule
    for c in cells:
        if c.rule == "mv-is":
            continue
        assert c.sample == "first_m36" and c.window == 36
        assert len(c.boot_p) == len(robust.BLOCKS)
        assert all(0.0 < p <= 1.0 for p in (c.jk_p, c.hac_p, c.ceq_p, *c.boot_p))


def test_net_sharpe_is_below_gross_when_a_rule_trades() -> None:
    cells = pipeline.run_all(synthetic_datasets(), SAMPLES[:1], draws=99)
    for c in cells:
        if c.rule in ("vw", "bl"):
            assert np.isclose(c.sharpe_net, c.sharpe)  # one asset, never traded
        elif c.rule != "mv-is":
            assert c.sharpe_net < c.sharpe
            # eq. (17) is negative exactly when the net Sharpe ratio beats that of ew
            ew = next(x for x in cells if x.dataset == c.dataset and x.rule == "ew")
            if c.rule != "ew":
                assert (c.return_loss < 0) == (c.sharpe_net > ew.sharpe_net)


def test_sample_tags_and_summary_counts() -> None:
    assert pipeline.Sample("paper", "P", 1, 2).tag == "paper"
    assert pipeline.Sample("paper", "P", 1, 2, 60).tag == "paper_m60"
    datasets = synthetic_datasets()
    cells = pipeline.run_all(datasets, SAMPLES[:1], draws=99)
    text = pipeline.summary_md(cells, SAMPLES[:1])
    assert "12 comparisons" in text and "0.6 would be significant" in text
    rows = pipeline.headline_rows([c for c in cells if c.rule not in ("ew", "mv-is")])
    assert len(rows) == 2 + len(robust.BLOCKS) + 1
    assert all(0 <= int(r[1]) + int(r[2]) <= 12 for r in rows)


def test_noise_of_zero_size_leaves_every_sharpe_ratio_alone() -> None:
    months = np.array([196307 + i for i in range(6)] + [196401 + i for i in range(12)])
    rng = np.random.default_rng(1)
    r = rng.normal(0.005, 0.04, (18, 3))
    d = data.Dataset("mkt_smb_hml", "X", months, ("a", "b", "c"), r, 2)
    rows = sensitivity.spreads({"mkt_smb_hml": d}, draws=3, scales_bp=(0.0, 50.0), window=8)
    still = [s for s in rows if s.scale_bp == 0.0]
    moved = [s for s in rows if s.scale_bp == 50.0]
    assert len(still) == len(moved) == len(sensitivity.RULES)
    assert all(s.minimum == s.maximum == s.base for s in still)
    assert all(s.minimum < s.maximum for s in moved)
    assert "paper_inside_min_max" in sensitivity.spreads_csv(rows)


# --- against the real files ----------------------------------------------------------------------

DATA_DIR = Path(__file__).resolve().parents[1] / "data"
needs_data = pytest.mark.needs_data  # skipped, with a notice, by conftest.py when data/ is empty


@pytest.fixture(scope="module")
def paper_cells() -> dict[tuple[str, str], pipeline.Cell]:
    datasets = data.build_datasets(DATA_DIR)
    sample = pipeline.Sample("paper", "Paper", paper.SAMPLE_START, paper.SAMPLE_END)
    cells = pipeline.run_all(datasets, (sample,), rules=("ew", "min", "min-c", "vw"), draws=99)
    return {(c.dataset, c.rule): c for c in cells}


@needs_data
def test_datasets_have_the_paper_dimensions() -> None:
    datasets = data.build_datasets(DATA_DIR)
    assert {k: len(d.assets) for k, d in datasets.items()} == {
        "industry": 11,
        "mkt_smb_hml": 3,
        "ff1": 21,
        "ff4": 24,
    }
    for d in datasets.values():
        w = d.window(paper.SAMPLE_START, paper.SAMPLE_END)
        assert w.returns.shape[0] == 497  # July 1963 to November 2004
        assert not np.isnan(w.returns).any()
        assert d.assets[d.market_index] == "MKT"


@needs_data
def test_equal_weight_column_matches_the_paper(
    paper_cells: dict[tuple[str, str], pipeline.Cell],
) -> None:
    # Tolerances were chosen after seeing the results on the 202609 vintage of the library.
    # Three datasets are within 0.001 of Table 3 and MKT/SMB/HML is 0.011 above it; each
    # tolerance is the observed gap rounded up. PAPER.md lists this under "Choices made
    # after seeing results".
    tol_sharpe = {"industry": 0.002, "mkt_smb_hml": 0.015, "ff1": 0.002, "ff4": 0.002}
    for i, key in enumerate(paper.DATASETS):
        cell = paper_cells[(key, "ew")]
        assert cell.months == 377
        published = paper.SHARPE["ew"][i], paper.CEQ["ew"][i], paper.TURNOVER["ew"][i]
        assert all(v is not None for v in published)
        assert abs(cell.sharpe - float(published[0] or 0)) < tol_sharpe[key]
        assert abs(cell.ceq - float(published[1] or 0)) < 0.0005
        assert abs(cell.turnover - float(published[2] or 0)) < 0.001


@needs_data
def test_constrained_minimum_variance_is_close_to_the_paper(
    paper_cells: dict[tuple[str, str], pipeline.Cell],
) -> None:
    for i, key in enumerate(paper.DATASETS):
        published = paper.SHARPE["min-c"][i]
        assert published is not None
        # 0.01 was chosen after seeing the largest gap, 0.0069 on FF-4-factor.
        assert abs(paper_cells[(key, "min-c")].sharpe - published) < 0.01
