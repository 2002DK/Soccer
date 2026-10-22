"""Predict a Premier League match with the saved model.

    python ml/scripts/predict.py Arsenal Chelsea

The FastAPI backend imports Predictor; the command line is for quick checks.
Team names are football-data.co.uk's (e.g. "Man City", "Nott'm Forest").
"""

import argparse
from datetime import date as Date, timedelta
from pathlib import Path

import joblib
import pandas as pd

from build_features import MATCHES_PATH, build_features

# Defined here, not imported from save_model, so serving needs no training libraries (e.g. XGBoost)
MODEL_PATH = Path(__file__).resolve().parents[1] / "models" / "model.pkl"


class Predictor:
    def __init__(self, model_path=MODEL_PATH, matches_path=MATCHES_PATH):
        bundle = joblib.load(model_path)
        self.model = bundle["model"]
        self.features = bundle["features"]
        self.labels = bundle["labels"]

        self.matches = pd.read_csv(matches_path, parse_dates=["date"])
        self.last_played = self.matches["date"].max().date()
        self.season = self.matches["season"].max()
        current = self.matches[self.matches["season"] == self.season]
        self.teams = sorted(set(current["home_team"]) | set(current["away_team"]))

    def default_date(self) -> Date:
        return self.last_played + timedelta(days=1)

    def feature_row(self, home_team: str, away_team: str, on: Date | None = None) -> pd.Series:
        """Features for home_team v away_team, as of `on` (default: the day after the last played match)."""
        for team in (home_team, away_team):
            if team not in self.teams:
                raise ValueError(f"{team!r} is not a {self.season} Premier League team")
        if home_team == away_team:
            raise ValueError("home and away team must be different")

        on = on or self.default_date()
        if on <= self.last_played:
            raise ValueError(f"date must be after the last played match ({self.last_played})")

        fixture = pd.DataFrame([{"season": self.season, "date": on, "home_team": home_team, "away_team": away_team}])
        return build_features(self.matches, fixture).iloc[-1]

    def predict(self, home_team: str, away_team: str, on: Date | None = None) -> dict[str, float]:
        """Probability of each outcome: {"H": ..., "D": ..., "A": ...}."""
        row = self.feature_row(home_team, away_team, on)
        X = row[self.features].to_frame().T.astype(float)
        proba = self.model.predict_proba(X)[0]
        return dict(zip(self.labels, proba.round(4).tolist()))


def main() -> None:
    parser = argparse.ArgumentParser(description="Predict a Premier League match.")
    parser.add_argument("home_team")
    parser.add_argument("away_team")
    parser.add_argument("--date", type=Date.fromisoformat, help="match date, YYYY-MM-DD")
    args = parser.parse_args()

    proba = Predictor().predict(args.home_team, args.away_team, args.date)
    print(f"{args.home_team} v {args.away_team}")
    for label, name in [("H", "Home win"), ("D", "Draw"), ("A", "Away win")]:
        print(f"  {name:<9} {proba[label]:.0%}")


if __name__ == "__main__":
    main()
