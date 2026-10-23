"""Elo team strength ratings for a league.

Every team starts at 1500. After each match both teams' ratings move by

    K * margin * (actual - expected)

where actual is 1 / 0.5 / 0 for a win / draw / loss, expected comes from the
rating gap plus a home advantage, and margin grows with the goal difference.
At each new season ratings move part of the way back to the league mean, and
promoted teams start at the average end rating of the teams they replaced.

K, the home advantage and that season carry-over are tuned on the seasons
before the league's test seasons, by how well the expected score predicts
the actual score. The test seasons then score an Elo-only model: a
multinomial logistic regression mapping the rating gap to home/draw/away
probabilities.

Writes:
- ml/data/<league>/processed/elo.csv: every match with both teams' ratings
  before and after it
- ml/data/<league>/processed/ratings.csv: current ratings, best first
- ml/models/<league>/elo.json: tuned parameters and test scores
"""

import argparse
import itertools
import json
from dataclasses import asdict, dataclass

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, log_loss

from leagues import League, add_league_argument

START = 1500.0
LABELS = ["H", "D", "A"]

# Candidate parameters tried when tuning
K_GRID = [10, 15, 20, 25, 30, 40]
HOME_GRID = [0, 25, 50, 75, 100]
CARRY_GRID = [1.0, 2 / 3, 1 / 2]  # share of a team's distance from the mean kept into the next season


@dataclass(frozen=True)
class EloParams:
    k: float = 20
    home_advantage: float = 50
    carry: float = 2 / 3


def expected_home(home_elo, away_elo, home_advantage: float):
    """Expected score for the home side (1 = certain win, 0.5 = even, 0 = certain loss)."""
    return 1 / (1 + 10 ** (-(home_elo + home_advantage - away_elo) / 400))


def margin_multiplier(goal_diff: int) -> float:
    """Bigger wins move ratings more (the World Football Elo scale)."""
    goal_diff = abs(goal_diff)
    if goal_diff <= 1:
        return 1.0
    if goal_diff == 2:
        return 1.5
    return (11 + goal_diff) / 8


def compute_elo(matches: pd.DataFrame, params: EloParams) -> tuple[pd.DataFrame, dict[str, float]]:
    """Ratings before and after every match, and each team's latest rating.

    `matches` must be sorted by date and hold played matches only.
    """
    ratings: dict[str, float] = {}
    rows = []
    season = None
    for m in matches.itertuples(index=False):
        if m.season != season:
            ratings = _new_season(ratings, matches.loc[matches["season"] == m.season], params)
            season = m.season

        home, away = ratings[m.home_team], ratings[m.away_team]
        exp = expected_home(home, away, params.home_advantage)
        actual = 1.0 if m.home_goals > m.away_goals else 0.5 if m.home_goals == m.away_goals else 0.0
        change = params.k * margin_multiplier(m.home_goals - m.away_goals) * (actual - exp)
        ratings[m.home_team] = home + change
        ratings[m.away_team] = away - change
        rows.append((home, away, home + change, away - change, exp, actual))

    elo = pd.DataFrame(
        rows, index=matches.index,
        columns=["home_elo_pre", "away_elo_pre", "home_elo_post", "away_elo_post", "home_expected", "home_actual"],
    )
    return elo, ratings


def _new_season(ratings: dict[str, float], games: pd.DataFrame, params: EloParams) -> dict[str, float]:
    teams = set(games["home_team"]) | set(games["away_team"])
    if not ratings:
        return {t: START for t in teams}

    # Promoted teams take over the average rating of the teams that went down
    leaving = [r for t, r in ratings.items() if t not in teams]
    promoted_rating = float(np.mean(leaving)) if leaving else START
    carried = {t: ratings.get(t, promoted_rating) for t in teams}

    mean = float(np.mean(list(carried.values())))
    return {t: mean + params.carry * (r - mean) for t, r in carried.items()}


def outcome_index(matches: pd.DataFrame) -> np.ndarray:
    return np.select(
        [matches["home_goals"] > matches["away_goals"], matches["home_goals"] == matches["away_goals"]], [0, 1], 2
    )


def tune(matches: pd.DataFrame, league: League) -> tuple[EloParams, float]:
    """Parameters with the lowest squared error of expected vs actual score on the pre-test seasons.

    The first season is skipped: everyone starts level, so its errors say little about the parameters.
    """
    first = league.season_label(league.first_season)
    test_from = league.season_label(league.test_from)
    scored = (matches["season"] > first) & (matches["season"] < test_from)

    best = None
    for k, home, carry in itertools.product(K_GRID, HOME_GRID, CARRY_GRID):
        params = EloParams(k=k, home_advantage=home, carry=carry)
        elo, _ = compute_elo(matches, params)
        err = float(np.mean((elo.loc[scored, "home_expected"] - elo.loc[scored, "home_actual"]) ** 2))
        if best is None or err < best[1]:
            best = (params, err)
    return best


def evaluate(matches: pd.DataFrame, elo: pd.DataFrame, league: League, params: EloParams) -> dict:
    """Score an Elo-only home/draw/away model on the test seasons."""
    gap = (elo["home_elo_pre"] + params.home_advantage - elo["away_elo_pre"]).to_frame("gap")
    y = outcome_index(matches)
    first = league.season_label(league.first_season)
    test_from = league.season_label(league.test_from)
    train = ((matches["season"] > first) & (matches["season"] < test_from)).to_numpy()
    test = (matches["season"] >= test_from).to_numpy()

    model = LogisticRegression(max_iter=1000).fit(gap[train], y[train])
    proba = model.predict_proba(gap[test])
    prior = np.tile(np.bincount(y[train], minlength=3) / train.sum(), (test.sum(), 1))
    return {
        "test_seasons": f"{test_from} onwards",
        "n_test": int(test.sum()),
        "accuracy": accuracy_score(y[test], proba.argmax(axis=1)),
        "log_loss": log_loss(y[test], proba, labels=range(3)),
        "baseline": {
            "always_home_accuracy": float(np.mean(y[test] == 0)),
            "class_prior_log_loss": log_loss(y[test], prior, labels=range(3)),
        },
    }


def run(league: League) -> None:
    matches = pd.read_csv(league.matches_path, parse_dates=["date"])
    matches = matches.dropna(subset=["home_goals"]).sort_values(["date", "home_team"]).reset_index(drop=True)

    params, tune_error = tune(matches, league)
    elo, ratings = compute_elo(matches, params)
    scores = evaluate(matches, elo, league, params)

    history = pd.concat([matches[["season", "date", "home_team", "away_team"]], elo.round(1)], axis=1)
    history.drop(columns=["home_expected", "home_actual"]).to_csv(
        league.processed_dir / "elo.csv", index=False, date_format="%Y-%m-%d"
    )

    current = matches.loc[matches["season"] == matches["season"].max()]
    teams = set(current["home_team"]) | set(current["away_team"])
    table = (
        pd.Series({t: ratings[t] for t in teams}, name="elo").round(1)
        .sort_values(ascending=False).rename_axis("team").reset_index()
    )
    table.insert(0, "rank", range(1, len(table) + 1))
    table.to_csv(league.processed_dir / "ratings.csv", index=False)

    league.model_dir.mkdir(parents=True, exist_ok=True)
    (league.model_dir / "elo.json").write_text(json.dumps({
        "params": asdict(params),
        "tune_squared_error": tune_error,
        "elo_only_model": scores,
        "ratings_after": matches["date"].max().strftime("%Y-%m-%d"),
    }, indent=2))

    print(f"tuned: K={params.k}, home advantage={params.home_advantage}, carry={params.carry:.2f}")
    print(
        f"Elo-only on {scores['test_seasons']}: accuracy {scores['accuracy']:.3f}, log loss {scores['log_loss']:.4f} "
        f"(always home {scores['baseline']['always_home_accuracy']:.3f}, "
        f"class prior {scores['baseline']['class_prior_log_loss']:.4f})"
    )
    print(table.head(6).to_string(index=False))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    add_league_argument(parser)
    run(parser.parse_args().league)


if __name__ == "__main__":
    main()
