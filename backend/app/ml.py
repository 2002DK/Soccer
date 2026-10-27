"""Bridge to the ML code in ml/scripts, which the API reuses rather than copies."""

import json
import sys
from functools import cache
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "ml" / "scripts"))

import pandas as pd  # noqa: E402

from leagues import DEFAULT_LEAGUE, LEAGUES, get_league  # noqa: E402
from predict import Predictor  # noqa: E402


def league_codes() -> list[str]:
    """Leagues whose model has been built (so a fresh registry entry without data is left out)."""
    return [code for code, league in LEAGUES.items() if league.model_path.exists()]


@cache
def get_ratings(league_code: str = DEFAULT_LEAGUE) -> dict:
    """Current Elo ratings, best first, with each team's rating after every match this season."""
    league = get_league(league_code)
    table = pd.read_csv(league.processed_dir / "ratings.csv")
    history = pd.read_csv(league.processed_dir / "elo.csv")
    season = history["season"].max()
    current = history[history["season"] == season]

    trends: dict[str, list] = {}
    for r in current.itertuples(index=False):
        for team, pre, post in ((r.home_team, r.home_elo_pre, r.home_elo_post), (r.away_team, r.away_elo_pre, r.away_elo_post)):
            if team not in trends:
                trends[team] = [{"date": None, "elo": pre}]  # season start, after the summer reset
            trends[team].append({"date": r.date, "elo": post})

    return {
        "season": season,
        "ratings": [
            {
                "rank": int(t.rank), "team": t.team, "elo": float(t.elo),
                "season_change": round(float(t.elo) - trends[t.team][0]["elo"], 1) if t.team in trends else 0.0,
                "trend": trends.get(t.team, []),
            }
            for t in table.itertuples(index=False)
        ],
    }


@cache
def get_predictor(league_code: str = DEFAULT_LEAGUE) -> Predictor:
    """Load a league's model and match data once per process."""
    return Predictor(get_league(league_code))


@cache
def get_upcoming(league_code: str = DEFAULT_LEAGUE) -> dict:
    """Stored predictions for the next 7 days' fixtures, written by the daily job (fixtures.py)."""
    path = get_league(league_code).processed_dir / "upcoming.json"
    if not path.exists():
        return {"fixtures": []}
    return json.loads(path.read_text())


@cache
def get_accuracy(league_code: str = DEFAULT_LEAGUE) -> dict:
    """Live accuracy of predictions frozen before kick-off, written by the daily job (tracker.py)."""
    path = get_league(league_code).processed_dir / "accuracy.json"
    if not path.exists():
        return {"frozen": 0, "scored": 0, "matches": []}
    return json.loads(path.read_text())


@cache
def get_model_info(league_code: str = DEFAULT_LEAGUE) -> dict:
    """The production model and every model's test scores, V1 and V2, in one list.

    V2 entries use feature_set "v2"; their calibrated variants and calibration
    tables are under "v2" for the detailed view.
    """
    league = get_league(league_code)
    meta = json.loads(league.meta_path.read_text())
    metrics = json.loads(league.metrics_path.read_text())
    comparison = list(metrics["results"])
    v2 = metrics.get("v2")
    if v2:
        baseline = comparison[0]["baseline"]  # same test matches as V1
        comparison += [
            {"feature_set": "v2", "baseline": baseline, **{k: v for k, v in r.items() if k != "calibration"}}
            for r in v2["results"] if not r["calibrated"]
        ]
    return {"production": meta, "comparison": comparison, "v2": v2}
