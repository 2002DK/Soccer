"""League registry: where each competition's data comes from and where its files live.

Every pipeline script takes a --league code from here. Files are kept per league:
- ml/data/<code>/raw/        downloads (git-ignored)
- ml/data/<code>/processed/  matches.csv, features.csv
- ml/models/<code>/          model.pkl, model_meta.json, metrics.json
"""

import argparse
from dataclasses import dataclass, field
from datetime import date
from pathlib import Path

ML_DIR = Path(__file__).resolve().parents[1]


@dataclass(frozen=True, eq=False)
class League:
    code: str  # our id, used in paths and URLs
    name: str
    football_data_code: str  # football-data.co.uk division, e.g. "E0"
    first_season: int  # start year of the first season downloaded
    test_from: int  # start year of the first held-out test season
    understat_code: str | None = None  # Understat league slug; None if Understat lacks the league
    understat_first_season: int = 2014  # Understat starts in 2014-15
    # Understat team name -> football-data.co.uk team name (names not listed are identical)
    understat_names: dict[str, str] = field(default_factory=dict)

    def current_season(self, today: date | None = None) -> int:
        """Start year of the latest season: from August, the season starting that year."""
        today = today or date.today()
        return today.year if today.month >= 8 else today.year - 1

    def season_label(self, start_year: int) -> str:
        """2024 -> '2024-25'."""
        return f"{start_year}-{(start_year + 1) % 100:02d}"

    @property
    def raw_dir(self) -> Path:
        return ML_DIR / "data" / self.code / "raw"

    @property
    def processed_dir(self) -> Path:
        return ML_DIR / "data" / self.code / "processed"

    @property
    def matches_path(self) -> Path:
        return self.processed_dir / "matches.csv"

    @property
    def features_path(self) -> Path:
        return self.processed_dir / "features.csv"

    @property
    def model_dir(self) -> Path:
        return ML_DIR / "models" / self.code

    @property
    def model_path(self) -> Path:
        return self.model_dir / "model.pkl"

    @property
    def meta_path(self) -> Path:
        return self.model_dir / "model_meta.json"

    @property
    def metrics_path(self) -> Path:
        return self.model_dir / "metrics.json"


LEAGUES = {
    league.code: league
    for league in [
        League(
            code="epl",
            name="Premier League",
            football_data_code="E0",
            first_season=2005,
            test_from=2024,
            understat_code="EPL",
            understat_names={
                "Manchester City": "Man City",
                "Manchester United": "Man United",
                "Newcastle United": "Newcastle",
                "Nottingham Forest": "Nott'm Forest",
                "Queens Park Rangers": "QPR",
                "West Bromwich Albion": "West Brom",
                "Wolverhampton Wanderers": "Wolves",
            },
        ),
    ]
}
DEFAULT_LEAGUE = "epl"


def get_league(code: str) -> League:
    try:
        return LEAGUES[code]
    except KeyError:
        raise ValueError(f"unknown league {code!r}; available: {', '.join(LEAGUES)}") from None


def _league_arg(code: str) -> League:
    try:
        return get_league(code)
    except ValueError as e:
        raise argparse.ArgumentTypeError(str(e)) from None  # argparse shows this message as-is


def add_league_argument(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--league", type=_league_arg, default=get_league(DEFAULT_LEAGUE),
        help=f"league code ({', '.join(LEAGUES)}; default {DEFAULT_LEAGUE})",
    )
