"""Download a league's match xG from Understat.

Understat has data from 2014-15. One JSON file per season is saved to
ml/data/<league>/raw/understat_<slug>_<start year>.json. Existing files are
skipped unless --force is passed, except the current season, which is always
re-downloaded. Leagues Understat doesn't cover are skipped.
"""

import argparse
import json
from pathlib import Path

import requests

from leagues import League, add_league_argument

URL = "https://understat.com/getLeagueData/{slug}/{year}"


def download_season(league: League, year: int, force: bool = False) -> Path:
    slug = league.understat_code
    dest = league.raw_dir / f"understat_{slug}_{year}.json"
    if dest.exists() and not force and year != league.current_season():
        print(f"skip  {year} (already downloaded)")
        return dest

    resp = requests.get(
        URL.format(slug=slug, year=year),
        headers={
            # The endpoint only answers requests that look like the site's own XHR calls
            "X-Requested-With": "XMLHttpRequest",
            "Referer": f"https://understat.com/league/{slug}/{year}",
        },
        timeout=30,
    )
    resp.raise_for_status()
    matches = resp.json()["dates"]
    dest.write_text(json.dumps(matches), encoding="utf-8")
    print(f"saved {year} ({len(matches)} matches)")
    return dest


def run(league: League, first: int | None = None, last: int | None = None, force: bool = False) -> None:
    if not league.understat_code:
        print(f"skip  Understat has no {league.name} data")
        return
    league.raw_dir.mkdir(parents=True, exist_ok=True)
    for year in range(first or league.understat_first_season, (last or league.current_season()) + 1):
        download_season(league, year, force=force)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    add_league_argument(parser)
    parser.add_argument("--first", type=int, help="first season start year (default: 2014)")
    parser.add_argument("--last", type=int, help="last season start year (default: the current season)")
    parser.add_argument("--force", action="store_true", help="re-download existing files")
    args = parser.parse_args()
    run(args.league, args.first, args.last, args.force)


if __name__ == "__main__":
    main()
