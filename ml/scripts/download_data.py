"""Download Premier League match results from football-data.co.uk.

One CSV per season is saved to ml/data/raw/E0_<season>.csv, e.g. E0_2425.csv
for 2024-25. Existing files are skipped unless --force is passed, except the
current season, which is always re-downloaded because it is still being played.
"""

import argparse
from datetime import date
from pathlib import Path

import requests

BASE_URL = "https://www.football-data.co.uk/mmz4281/{code}/E0.csv"
RAW_DIR = Path(__file__).resolve().parents[1] / "data" / "raw"

FIRST_SEASON = 2005  # 2005-06


def current_season(today: date | None = None) -> int:
    """Start year of the latest season: from August, the season starting that year."""
    today = today or date.today()
    return today.year if today.month >= 8 else today.year - 1


CURRENT_SEASON = current_season()


def season_code(start_year: int) -> str:
    """2024 -> '2425'."""
    return f"{start_year % 100:02d}{(start_year + 1) % 100:02d}"


def download_season(start_year: int, force: bool = False) -> Path | None:
    code = season_code(start_year)
    dest = RAW_DIR / f"E0_{code}.csv"
    if dest.exists() and not force and start_year != CURRENT_SEASON:
        print(f"skip  {code} (already downloaded)")
        return dest

    resp = requests.get(BASE_URL.format(code=code), timeout=30)
    if resp.status_code == 404 and start_year == CURRENT_SEASON:
        # Early August: the new season's file appears once its first matches are played
        print(f"skip  {code} (not published yet)")
        return None
    resp.raise_for_status()
    dest.write_bytes(resp.content)
    print(f"saved {code} ({len(resp.content) / 1024:.0f} KB)")
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
