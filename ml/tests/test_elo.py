import sys
from pathlib import Path

import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from elo import START, EloParams, compute_elo, expected_home, margin_multiplier  # noqa: E402


def match(season, day, home, away, hg, ag):
    return {"season": season, "date": pd.Timestamp(f"2020-01-{day:02d}"), "home_team": home,
            "away_team": away, "home_goals": hg, "away_goals": ag}


def test_expected_score_is_even_without_gap_or_home_advantage():
    assert expected_home(1500, 1500, 0) == pytest.approx(0.5)
    assert expected_home(1600, 1500, 0) == pytest.approx(1 - expected_home(1500, 1600, 0))
    assert expected_home(1500, 1500, 50) > 0.5


def test_margin_multiplier_grows_with_goal_difference():
    assert margin_multiplier(0) == margin_multiplier(1) == margin_multiplier(-1) == 1
    assert margin_multiplier(2) == 1.5
    assert margin_multiplier(4) > margin_multiplier(3) > margin_multiplier(2)


def test_ratings_are_zero_sum_and_move_the_right_way():
    games = pd.DataFrame([match("A", 1, "X", "Y", 2, 0), match("A", 2, "Y", "Z", 1, 1)])
    elo, ratings = compute_elo(games, EloParams(k=20, home_advantage=0, carry=1))
    assert sum(ratings.values()) == pytest.approx(3 * START)
    assert ratings["X"] > START > ratings["Y"]
    # Pre-match ratings of a match equal the post-match ratings of each team's previous match
    assert elo.loc[1, "home_elo_pre"] == elo.loc[0, "away_elo_post"]


def test_new_season_regresses_to_mean_and_promoted_team_inherits_relegated_rating():
    games = pd.DataFrame([
        match("A", 1, "X", "Y", 3, 0),
        match("B", 2, "X", "Z", 0, 0),  # Y relegated, Z promoted
    ])
    elo, _ = compute_elo(games, EloParams(k=20, home_advantage=0, carry=0.5))
    x_end, y_end = elo.loc[0, "home_elo_post"], elo.loc[0, "away_elo_post"]
    z_start = elo.loc[1, "away_elo_pre"]
    mean = (x_end + y_end) / 2
    assert elo.loc[1, "home_elo_pre"] == pytest.approx(mean + 0.5 * (x_end - mean))
    assert z_start == pytest.approx(mean + 0.5 * (y_end - mean))
