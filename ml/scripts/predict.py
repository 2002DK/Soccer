"""Predict a match with a league's saved model.

    python ml/scripts/predict.py Arsenal Chelsea
    python ml/scripts/predict.py --league epl Arsenal Chelsea

The FastAPI backend imports Predictor; the command line is for quick checks.
Team names are football-data.co.uk's (e.g. "Man City", "Nott'm Forest").

Imports nothing from the training code, so serving needs no training
libraries (e.g. XGBoost).
"""

import argparse
from datetime import date as Date, timedelta

import joblib
import pandas as pd

import numpy as np

from build_features import build_league_features
from leagues import League, add_league_argument


class Predictor:
    def __init__(self, league: League):
        self.league = league
        self.bundle = joblib.load(league.model_path)
        self.features = self.bundle["features"]
        self.labels = self.bundle["labels"]

        self.matches = pd.read_csv(league.matches_path, parse_dates=["date"])
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
                raise ValueError(f"{team!r} is not a {self.season} {self.league.name} team")
        if home_team == away_team:
            raise ValueError("home and away team must be different")

        on = on or self.default_date()
        if on <= self.last_played:
            raise ValueError(f"date must be after the last played match ({self.last_played})")

        fixture = pd.DataFrame([{"season": self.season, "date": on, "home_team": home_team, "away_team": away_team}])
        return build_league_features(self.league, self.matches, fixture).iloc[-1]

    def combine(self, X: pd.DataFrame, dc_probs: np.ndarray | None) -> np.ndarray:
        """V2 probabilities from a feature row and Dixon-Coles probabilities:
        Logistic Regression and/or Dixon-Coles, blended, then the optional calibration step."""
        b = self.bundle
        proba = np.zeros(3)
        if b["lr"] is not None:
            proba += b["lr_weight"] * b["lr"].predict_proba(X)[0]
        if dc_probs is not None:
            proba += (1 - b["lr_weight"]) * dc_probs
        if b["calibrator"] is not None:
            proba = b["calibrator"].predict_proba(np.log(np.clip(proba, 1e-6, 1))[None, :])[0]
        return proba

    def predict(self, home_team: str, away_team: str, on: Date | None = None) -> dict[str, float]:
        """Probability of each outcome: {"H": ..., "D": ..., "A": ...}."""
        row = self.feature_row(home_team, away_team, on)
        X = row[self.features].to_frame().T.astype(float)
        b = self.bundle
        if b.get("version") == 2:
            dc_probs = b["dc"].outcome_probs(home_team, away_team) if b["dc"] is not None else None
            proba = self.combine(X, dc_probs)
        else:
            proba = b["model"].predict_proba(X)[0]
        return dict(zip(self.labels, proba.round(4).tolist()))

    def scorelines(self, home_team: str, away_team: str) -> dict | None:
        """Expected goals, likely scorelines, over 2.5 and both teams to score (V2 models with Dixon-Coles)."""
        dc = self.bundle.get("dc")
        return dc.summary(home_team, away_team) if dc is not None else None

    def explain(self, home_team: str, away_team: str, on: Date | None = None) -> dict:
        """Probabilities plus the drivers behind them (explain.py)."""
        from explain import explain
        return explain(self, home_team, away_team, on)


def main() -> None:
    parser = argparse.ArgumentParser(description="Predict a match.")
    add_league_argument(parser)
    parser.add_argument("home_team")
    parser.add_argument("away_team")
    parser.add_argument("--date", type=Date.fromisoformat, help="match date, YYYY-MM-DD")
    args = parser.parse_args()

    proba = Predictor(args.league).predict(args.home_team, args.away_team, args.date)
    print(f"{args.home_team} v {args.away_team}")
    for label, name in [("H", "Home win"), ("D", "Draw"), ("A", "Away win")]:
        print(f"  {name:<9} {proba[label]:.0%}")


if __name__ == "__main__":
    main()
