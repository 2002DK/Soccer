"""Tracker: predictions are frozen once, before kick-off, never rewritten, and scored against results."""

import json
import shutil
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

import leagues  # noqa: E402
import tracker  # noqa: E402

REAL_ML_DIR = leagues.ML_DIR


@pytest.fixture
def league(tmp_path, monkeypatch):
    """The Premier League, with its processed files copied into a temporary folder."""
    monkeypatch.setattr(leagues, "ML_DIR", tmp_path)
    epl = leagues.get_league("epl")
    epl.processed_dir.mkdir(parents=True)
    for name in ["matches.csv", "features.csv", "upcoming.json"]:
        shutil.copy(REAL_ML_DIR / "data" / "epl" / "processed" / name, epl.processed_dir / name)
    return epl


def kickoffs(league):
    fixtures = json.loads((league.processed_dir / "upcoming.json").read_text())["fixtures"]
    return sorted(datetime.strptime(f["kickoff"], "%Y-%m-%dT%H:%M:%SZ").replace(tzinfo=timezone.utc) for f in fixtures)


def test_freezes_once_before_kickoff_and_only_appends(league):
    times = kickoffs(league)
    if not times:
        pytest.skip("no upcoming fixtures")
    first = times[0] - timedelta(hours=tracker.FREEZE_HOURS - 1)  # first fixture just inside the window

    added = tracker.freeze(league, first)
    before = tracker.log_path(league).read_text()
    assert added >= 1
    assert tracker.freeze(league, first) == 0  # a second run adds nothing

    later = first + timedelta(days=1)
    tracker.freeze(league, later)
    after = tracker.log_path(league).read_text()
    assert after.startswith(before)  # earlier rows untouched

    log = tracker.read_log(league)
    assert not log["fixture_id"].duplicated().any()
    frozen = pd.to_datetime(log["frozen_at"])
    kickoff = pd.to_datetime(log["kickoff"])
    assert (frozen < kickoff).all()
    assert np.allclose(log[["p_home", "p_draw", "p_away"]].sum(axis=1), 1, atol=1e-3)


def test_scores_against_results(league):
    times = kickoffs(league)
    if not times:
        pytest.skip("no upcoming fixtures")
    tracker.freeze(league, times[0] - timedelta(hours=1))
    log = tracker.read_log(league)

    # Pretend every frozen match ended in a home win
    matches = pd.read_csv(league.matches_path)
    fake = pd.DataFrame({
        "season": matches["season"].max(),
        "date": pd.to_datetime(log["kickoff"]).dt.strftime("%Y-%m-%d"),
        "home_team": log["home_team"], "away_team": log["away_team"],
        "home_goals": 1, "away_goals": 0, "result": "H",
    })
    pd.concat([matches, fake], ignore_index=True).to_csv(league.matches_path, index=False)

    summary = tracker.score(league)
    assert summary["scored"] == len(log)
    assert summary["model"]["accuracy"] == pytest.approx(np.mean(log["predicted"] == "H"), abs=1e-4)
    assert summary["model"]["log_loss"] == pytest.approx(-np.mean(np.log(log["p_home"])), abs=1e-4)
    assert summary["baselines"]["always_home"]["accuracy"] == 1.0
    assert len(summary["matches"]) == len(log)
