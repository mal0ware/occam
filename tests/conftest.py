"""Make it obvious whether the tests against the real files ran.

Three tests need the French library files in ``data/``. Without them they are skipped, and a
green run then says nothing about the comparison with the paper. The header and the summary
line below say which case this is. Set ``OCCAM_REQUIRE_DATA=1`` to make the run fail
instead; CI does this whenever its download step succeeds.
"""

from __future__ import annotations

import os
from pathlib import Path

import pytest

from occam import data

DATA_DIR = Path(__file__).resolve().parents[1] / "data"
REQUIRE = os.environ.get("OCCAM_REQUIRE_DATA", "") not in ("", "0")
MARK = "needs_data"


def have_data() -> bool:
    names = (data.FACTORS_FILE, data.MOMENTUM_FILE, data.INDUSTRY_FILE, data.SIZE_BM_FILE)
    return all((DATA_DIR / name).exists() for name in names)


def pytest_configure(config: pytest.Config) -> None:
    config.addinivalue_line("markers", f"{MARK}: needs the French library files in data/")


def pytest_report_header(config: pytest.Config) -> str:
    if have_data():
        return "occam: data/ is present, the tests against the paper will run"
    return "occam: data/ is EMPTY, the tests against the paper will be skipped"


def pytest_collection_modifyitems(config: pytest.Config, items: list[pytest.Item]) -> None:
    if have_data():
        return
    if REQUIRE:
        pytest.exit("OCCAM_REQUIRE_DATA is set but data/ is empty", returncode=1)
    skip = pytest.mark.skip(reason="data/ is empty: run scripts/fetch_data.py first")
    for item in items:
        if item.get_closest_marker(MARK) is not None:
            item.add_marker(skip)


def pytest_terminal_summary(terminalreporter: pytest.TerminalReporter) -> None:
    if have_data():
        return
    terminalreporter.write_sep("=", "occam: tests against the paper did NOT run", yellow=True)
    terminalreporter.write_line(
        "data/ is empty, so every test marked needs_data was skipped. "
        "Run `uv run python scripts/fetch_data.py` and test again."
    )
