"""Download the Kenneth R. French Data Library files this project uses.

Files go to ``data/`` (never committed). No key or account is needed. A manifest
with the SHA-256 of every file is written next to them so a results table can be
traced to the exact vintage it was built from.

Usage:  uv run python scripts/fetch_data.py [--force]
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import urllib.request
from pathlib import Path

BASE_URL = "https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/ftp/"
FILES = (
    "F-F_Research_Data_Factors_CSV.zip",
    "F-F_Momentum_Factor_CSV.zip",
    "10_Industry_Portfolios_CSV.zip",
    "25_Portfolios_5x5_CSV.zip",
)
USER_AGENT = "occam-fetch/0.1 (python-urllib)"
DATA_DIR = Path(__file__).resolve().parents[1] / "data"


def download(name: str, dest: Path) -> None:
    """Fetch one file from the library into ``dest``."""
    request = urllib.request.Request(BASE_URL + name, headers={"User-Agent": USER_AGENT})
    with urllib.request.urlopen(request, timeout=120) as response:
        payload = response.read()
    if not payload.startswith(b"PK"):
        raise RuntimeError(f"{name}: the server did not return a zip archive")
    dest.write_bytes(payload)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--force", action="store_true", help="download even if the file exists")
    args = parser.parse_args()

    DATA_DIR.mkdir(exist_ok=True)
    manifest: dict[str, dict[str, str | int]] = {}
    for name in FILES:
        dest = DATA_DIR / name
        if args.force or not dest.exists():
            print(f"downloading {name}")
            download(name, dest)
        else:
            print(f"keeping existing {name}")
        payload = dest.read_bytes()
        manifest[name] = {
            "sha256": hashlib.sha256(payload).hexdigest(),
            "bytes": len(payload),
            "url": BASE_URL + name,
        }
    (DATA_DIR / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(f"wrote {DATA_DIR / 'manifest.json'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
