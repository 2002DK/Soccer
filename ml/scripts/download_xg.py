"""Download Premier League match xG from Understat.

Understat has EPL data from 2014-15. One JSON file per season is saved to
ml/data/raw/understat_EPL_<start year>.json. Existing files are skipped unless
--force is passed, except the current season, which is always re-downloaded.
"""

import argparse
import json
from pathlib import Path

import requests

from download_data import CURRENT_SEASON, RAW_DIR

URL = "https://understat.com/getLeagueData/EPL/{year}"
FIRST_SEASON = 2014  # 2014-15, Understat's first EPL season


def download_season(year: int, force: bool = False) -> Path:
    dest = RAW_DIR / f"understat_EPL_{year}.json"
    if dest.exists() and not force and year != CURRENT_SEASON:
        print(f"skip  {year} (already downloaded)")
        return dest

    resp = requests.get(
        URL.format(year=year),
        headers={
            # The endpoint only answers requests that look like the site's own XHR calls
            "X-Requested-With": "XMLHttpRequest",
            "Referer": f"https://understat.com/league/EPL/{year}",
        },
        timeout=30,
    )
    resp.raise_for_status()
    matches = resp.json()["dates"]
    dest.write_text(json.dumps(matches), encoding="utf-8")
    print(f"saved {year} ({len(matches)} matches)")
    return dest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--first", type=int, default=FIRST_SEASON, help="first season start year")
    parser.add_argument("--last", type=int, default=CURRENT_SEASON, help="last season start year")
    parser.add_argument("--force", action="store_true", help="re-download existing files")
    args = parser.parse_args()

    RAW_DIR.mkdir(parents=True, exist_ok=True)
    for year in range(args.first, args.last + 1):
        download_season(year, force=args.force)


if __name__ == "__main__":
    main()
