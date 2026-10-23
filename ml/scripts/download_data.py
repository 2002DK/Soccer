"""Download a league's match results from football-data.co.uk.

One CSV per season is saved to ml/data/<league>/raw/<division>_<season>.csv,
e.g. E0_2425.csv for the Premier League's 2024-25. Existing files are skipped
unless --force is passed, except the current season, which is always
re-downloaded because it is still being played.
"""

import argparse
from pathlib import Path

import requests

from leagues import League, add_league_argument

BASE_URL = "https://www.football-data.co.uk/mmz4281/{code}/{division}.csv"


def season_code(start_year: int) -> str:
    """2024 -> '2425'."""
    return f"{start_year % 100:02d}{(start_year + 1) % 100:02d}"


def download_season(league: League, start_year: int, force: bool = False) -> Path | None:
    code = season_code(start_year)
    dest = league.raw_dir / f"{league.football_data_code}_{code}.csv"
    is_current = start_year == league.current_season()
    if dest.exists() and not force and not is_current:
        print(f"skip  {code} (already downloaded)")
        return dest

    resp = requests.get(BASE_URL.format(code=code, division=league.football_data_code), timeout=30)
    if resp.status_code == 404 and is_current:
        # Early August: the new season's file appears once its first matches are played
        print(f"skip  {code} (not published yet)")
        return None
    resp.raise_for_status()
    dest.write_bytes(resp.content)
    print(f"saved {code} ({len(resp.content) / 1024:.0f} KB)")
    return dest


def run(league: League, first: int | None = None, last: int | None = None, force: bool = False) -> None:
    league.raw_dir.mkdir(parents=True, exist_ok=True)
    for year in range(first or league.first_season, (last or league.current_season()) + 1):
        download_season(league, year, force=force)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    add_league_argument(parser)
    parser.add_argument("--first", type=int, help="first season start year (default: the league's first)")
    parser.add_argument("--last", type=int, help="last season start year (default: the current season)")
    parser.add_argument("--force", action="store_true", help="re-download existing files")
    args = parser.parse_args()
    run(args.league, args.first, args.last, args.force)


if __name__ == "__main__":
    main()
