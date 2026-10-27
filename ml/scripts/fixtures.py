"""Upcoming fixtures and their stored predictions.

Reads the current season's Understat file (downloaded by download_xg.py),
which lists unplayed matches with kick-off times in UTC, and writes:
- ml/data/<league>/processed/fixtures.csv: every unplayed match this season
- ml/data/<league>/processed/upcoming.json: predictions, drivers and likely
  scorelines for matches kicking off in the next WINDOW_DAYS days

upcoming.json is only rewritten when a prediction or the fixture list changes
(not just its timestamp), so the daily job commits only when something is new.
"""

import argparse
import json
from datetime import datetime, timedelta, timezone

import pandas as pd

from leagues import League, add_league_argument
from predict import Predictor

WINDOW_DAYS = 7
OUTCOME_KEYS = {"H": "home_win", "D": "draw", "A": "away_win"}


def upcoming_path(league: League):
    return league.processed_dir / "upcoming.json"


def fixtures_path(league: League):
    return league.processed_dir / "fixtures.csv"


def read_fixtures(league: League) -> pd.DataFrame:
    """Unplayed matches this season, from Understat, with our team names."""
    if not league.understat_code:
        raise SystemExit(f"no fixture source for {league.name} yet")
    year = league.current_season()
    path = league.raw_dir / f"understat_{league.understat_code}_{year}.json"
    names = league.understat_names
    rows = [
        {
            "id": m["id"],
            "kickoff": pd.Timestamp(m["datetime"], tz="UTC"),
            "season": league.season_label(year),
            "home_team": names.get(m["h"]["title"], m["h"]["title"]),
            "away_team": names.get(m["a"]["title"], m["a"]["title"]),
        }
        for m in json.loads(path.read_text(encoding="utf-8"))
        if not m["isResult"]
    ]
    fixtures = pd.DataFrame(rows, columns=["id", "kickoff", "season", "home_team", "away_team"])

    # Drop matches our results already include (Understat can lag behind football-data.co.uk)
    played = pd.read_csv(league.matches_path, usecols=["season", "home_team", "away_team"])
    key = ["season", "home_team", "away_team"]
    fixtures = fixtures.merge(played.drop_duplicates(), on=key, how="left", indicator=True)
    fixtures = fixtures[fixtures["_merge"] == "left_only"].drop(columns="_merge")
    return fixtures.sort_values(["kickoff", "home_team"]).reset_index(drop=True)


def predict_upcoming(league: League, fixtures: pd.DataFrame, now: datetime) -> dict:
    """Predictions for fixtures kicking off after `now` and within WINDOW_DAYS."""
    p = Predictor(league)
    soon = fixtures[(fixtures["kickoff"] > now) & (fixtures["kickoff"] <= now + timedelta(days=WINDOW_DAYS))]
    unknown = sorted((set(soon["home_team"]) | set(soon["away_team"])) - set(p.teams))
    if unknown:
        raise SystemExit(f"fixture teams not in this season's results (missing name mapping?): {unknown}")

    out = []
    for f in soon.itertuples(index=False):
        result = p.explain(f.home_team, f.away_team, on=f.kickoff.date())
        proba = result["probabilities"]
        out.append({
            "id": f.id,
            "kickoff": f.kickoff.strftime("%Y-%m-%dT%H:%M:%SZ"),
            "home_team": f.home_team,
            "away_team": f.away_team,
            "probabilities": {OUTCOME_KEYS[k]: v for k, v in proba.items()},
            "prediction": result["predicted"],
            "why": result["why"],
            "elo": result["elo"],  # pre-match ratings, frozen with the prediction for the Elo-only baseline
            "scorelines": p.scorelines(f.home_team, f.away_team),
        })

    meta = json.loads(league.meta_path.read_text())
    return {
        "league": league.code,
        "season": p.season,
        "data_through": p.last_played.isoformat(),
        "model": meta.get("description", meta["model"]),
        "window_days": WINDOW_DAYS,
        "fixtures": out,
    }


def write_if_changed(path, payload: dict, now: datetime) -> bool:
    """Write payload plus a generated_at time, unless only the time would change."""
    if path.exists():
        old = json.loads(path.read_text())
        old.pop("generated_at", None)
        if old == payload:
            return False
    path.write_text(json.dumps({"generated_at": now.strftime("%Y-%m-%dT%H:%M:%SZ"), **payload}, indent=2))
    return True


def run(league: League, now: datetime | None = None) -> None:
    now = now or datetime.now(timezone.utc)
    fixtures = read_fixtures(league)
    fixtures.to_csv(fixtures_path(league), index=False, date_format="%Y-%m-%dT%H:%M:%SZ")

    payload = predict_upcoming(league, fixtures, now)
    changed = write_if_changed(upcoming_path(league), payload, now)
    print(
        f"{len(fixtures)} unplayed fixtures; {len(payload['fixtures'])} in the next {WINDOW_DAYS} days "
        f"({'updated' if changed else 'unchanged'}: {upcoming_path(league)})"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    add_league_argument(parser)
    parser.add_argument("--now", type=lambda s: datetime.fromisoformat(s).replace(tzinfo=timezone.utc),
                        help="pretend it is this UTC time, e.g. 2026-10-30T06:00 (for testing)")
    args = parser.parse_args()
    run(args.league, args.now)


if __name__ == "__main__":
    main()
