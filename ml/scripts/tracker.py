"""Live accuracy tracker: freeze predictions before kick-off, then score them.

Freezing: each run appends every fixture kicking off within FREEZE_HOURS that
isn't logged yet to ml/data/<league>/processed/predictions.csv, copied from
that run's upcoming.json (so the logged prediction is the one the site showed,
made from all results known at that time). The log is append-only: rows are
never changed or removed, and the daily workflow refuses to commit a change
that deletes a line. Git history shows when each row was added, which proves
it was made before the match.

FREEZE_HOURS is longer than a day because GitHub can start scheduled runs
hours late: a match is frozen by the last run that is sure to come before it.

Scoring: frozen predictions are matched to results in matches.csv and scored
(accuracy, log loss, Brier) next to two baselines frozen alongside them:
always picking the home team, and an Elo-only model. Results go to
ml/data/<league>/processed/accuracy.json.
"""

import argparse
import csv
import json
from datetime import datetime, timedelta, timezone

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression

from fixtures import upcoming_path, write_if_changed
from leagues import League, add_league_argument

FREEZE_HOURS = 36
LABELS = ["H", "D", "A"]
LOG_COLUMNS = [
    "fixture_id", "kickoff", "home_team", "away_team", "frozen_at", "data_through", "model",
    "p_home", "p_draw", "p_away", "predicted",
    "elo_p_home", "elo_p_draw", "elo_p_away",
    "xg_home", "xg_away", "top_scoreline",
]


def log_path(league: League):
    return league.processed_dir / "predictions.csv"


def accuracy_path(league: League):
    return league.processed_dir / "accuracy.json"


def read_log(league: League) -> pd.DataFrame:
    path = log_path(league)
    if not path.exists():
        return pd.DataFrame(columns=LOG_COLUMNS)
    return pd.read_csv(path, dtype={"fixture_id": str})


def elo_only_model(league: League) -> LogisticRegression:
    """Home/draw/away from the Elo rating gap alone, fitted on every played match."""
    f = pd.read_csv(league.features_path, usecols=["season", "result", "diff_elo"]).dropna()
    f = f[f["season"] > league.season_label(league.first_season)]
    return LogisticRegression(max_iter=1000).fit(f[["diff_elo"]], f["result"].map(LABELS.index))


def freeze(league: League, now: datetime) -> int:
    """Append not-yet-logged fixtures kicking off within FREEZE_HOURS. Returns how many were added."""
    upcoming = json.loads(upcoming_path(league).read_text())
    logged = set(read_log(league)["fixture_id"].astype(str))
    cutoff = now + timedelta(hours=FREEZE_HOURS)
    due = [
        f for f in upcoming["fixtures"]
        if str(f["id"]) not in logged
        and now < datetime.strptime(f["kickoff"], "%Y-%m-%dT%H:%M:%SZ").replace(tzinfo=timezone.utc) <= cutoff
    ]
    if not due:
        return 0

    elo_model = elo_only_model(league)
    rows = []
    for f in due:
        p = f["probabilities"]
        sc = f.get("scorelines") or {}
        top = sc.get("scorelines", [{}])[0]
        rows.append({
            "fixture_id": f["id"], "kickoff": f["kickoff"], "home_team": f["home_team"], "away_team": f["away_team"],
            "frozen_at": now.strftime("%Y-%m-%dT%H:%M:%SZ"), "data_through": upcoming["data_through"],
            "model": upcoming["model"],
            "p_home": p["home_win"], "p_draw": p["draw"], "p_away": p["away_win"], "predicted": f["prediction"],
            **dict(zip(["elo_p_home", "elo_p_draw", "elo_p_away"], _elo_probs(elo_model, f))),
            "xg_home": sc.get("expected_goals", {}).get("home"), "xg_away": sc.get("expected_goals", {}).get("away"),
            "top_scoreline": f"{top['home']}-{top['away']}" if top else "",
        })

    path = log_path(league)
    new_file = not path.exists()
    with path.open("a", newline="", encoding="utf-8") as fh:  # append only
        writer = csv.DictWriter(fh, fieldnames=LOG_COLUMNS)
        if new_file:
            writer.writeheader()
        writer.writerows(rows)
    return len(rows)


def _elo_probs(model: LogisticRegression, fixture: dict) -> list[float]:
    """Elo-only probabilities from the fixture's pre-match ratings."""
    gap = fixture["elo"]["home"] - fixture["elo"]["away"]
    return [round(float(x), 4) for x in model.predict_proba(pd.DataFrame({"diff_elo": [gap]}))[0]]


def score(league: League) -> dict:
    """Join frozen predictions to results and score them."""
    log = read_log(league)
    matches = pd.read_csv(league.matches_path, parse_dates=["date"])
    matches = matches[["date", "home_team", "away_team", "home_goals", "away_goals", "result"]]

    log["kickoff_date"] = pd.to_datetime(log["kickoff"]).dt.tz_localize(None).dt.normalize()
    # The same pairing happens once per season at each ground: the result is the one dated within a day of kick-off
    pairs = log[["fixture_id", "home_team", "away_team", "kickoff_date"]].merge(matches, on=["home_team", "away_team"])
    found = pairs[(pairs["date"] - pairs["kickoff_date"]).abs() <= pd.Timedelta(days=1)].drop_duplicates("fixture_id")
    joined = log.merge(found[["fixture_id", "date", "home_goals", "away_goals", "result"]], on="fixture_id", how="left")
    done = joined.dropna(subset=["result"]).sort_values("kickoff")

    def metrics(df: pd.DataFrame, cols: list[str]) -> dict:
        if df.empty:
            return {"accuracy": None, "log_loss": None, "brier": None}
        p = df[cols].to_numpy(float)
        y = df["result"].map(LABELS.index).to_numpy()
        onehot = np.eye(3)[y]
        return {
            "accuracy": round(float(np.mean(p.argmax(1) == y)), 4),
            "log_loss": round(float(-np.mean(np.log(np.clip(p[np.arange(len(y)), y], 1e-6, 1)))), 4),
            "brier": round(float(np.mean(np.sum((p - onehot) ** 2, axis=1))), 4),
        }

    model_cols, elo_cols = ["p_home", "p_draw", "p_away"], ["elo_p_home", "elo_p_draw", "elo_p_away"]
    started = log["frozen_at"].min() if len(log) else None
    # Matches played since tracking started that were never frozen (e.g. a run that didn't happen in time)
    missed = 0
    if started:
        since = matches[matches["date"] >= pd.Timestamp(started[:10])]
        key = set(zip(log["home_team"], log["away_team"], log["kickoff_date"].dt.date))
        missed = sum((h, a, d.date()) not in key for h, a, d in zip(since["home_team"], since["away_team"], since["date"]))

    weeks = done["kickoff_date"].dt.to_period("W-MON").dt.start_time
    by_week = []
    for week, g in done.groupby(weeks):
        by_week.append({"week_from": week.strftime("%Y-%m-%d"), "matches": len(g), **metrics(g, model_cols)})

    return {
        "league": league.code,
        "tracking_since": started,
        "frozen": int(len(log)),
        "scored": int(len(done)),
        "awaiting_result": int(len(log) - len(done)),
        "missed": int(missed),
        "model": metrics(done, model_cols),
        "baselines": {
            "elo_only": metrics(done, elo_cols),
            "always_home": {"accuracy": round(float(np.mean(done["result"] == "H")), 4) if len(done) else None},
        },
        "by_week": by_week,
        "matches": [
            {
                "fixture_id": r.fixture_id, "kickoff": r.kickoff, "home_team": r.home_team, "away_team": r.away_team,
                "frozen_at": r.frozen_at,
                "probabilities": {"home_win": r.p_home, "draw": r.p_draw, "away_win": r.p_away},
                "predicted": r.predicted, "top_scoreline": r.top_scoreline,
                "result": None if pd.isna(r.result) else r.result,
                "score": None if pd.isna(r.home_goals) else f"{int(r.home_goals)}-{int(r.away_goals)}",
                "correct": None if pd.isna(r.result) else bool(r.predicted == r.result),
            }
            for r in joined.sort_values("kickoff", ascending=False).itertuples(index=False)
        ],
    }


def run(league: League, now: datetime | None = None) -> None:
    now = now or datetime.now(timezone.utc)
    added = freeze(league, now)
    summary = score(league)
    changed = write_if_changed(accuracy_path(league), summary, now)
    m = summary["model"]
    print(
        f"froze {added} new prediction(s); {summary['frozen']} logged, {summary['scored']} scored, "
        f"{summary['missed']} missed; accuracy {m['accuracy']} ({'updated' if changed else 'unchanged'})"
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
