"""The "Why" drivers must point the way their inputs say they should.

For each driver, the expected direction for (home win - away win) comes from
a plain summary of its inputs relative to the training average, e.g. a higher
home Elo than away Elo should raise the home side's chance.
"""

import sys
from itertools import permutations
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from explain import _training_means, explain  # noqa: E402
from leagues import get_league  # noqa: E402
from predict import Predictor  # noqa: E402

N_MATCHES = 20
# Only judge drivers of 1 percentage point or more. Smaller effects can tip either way for second-order
# reasons, e.g. two attacking teams make a draw less likely, which nudges the favourite up even when
# their attack/defence gap is slightly smaller than average. The site hides drivers under 1 point.
MIN_EFFECT = 0.01


@pytest.fixture(scope="module")
def predictor():
    return Predictor(get_league("epl"))


def expected_sign(key: str, row: pd.Series, means: dict, dc, home: str, away: str) -> float:
    if key == "strength":
        return np.sign(row["diff_elo"] - means["diff_elo"])
    if key == "goals":
        lam, mu = dc.rates(home, away)
        return np.sign(np.log(lam / mu) - dc.home)  # attack/defence gap beyond home advantage
    if key == "league":
        return np.sign(row["league_home_rate"] - means["league_home_rate"])
    raise ValueError(key)


def test_drivers_point_the_expected_way(predictor):
    means = _training_means(predictor.bundle)
    rng = np.random.default_rng(0)
    pairs = list(permutations(predictor.teams, 2))
    picks = [pairs[i] for i in rng.choice(len(pairs), N_MATCHES, replace=False)]

    wrong, checked = [], 0
    for home, away in picks:
        row = predictor.feature_row(home, away)
        for d in explain(predictor, home, away, top=10)["why"]:
            net = d["effect"]["home_win"] - d["effect"]["away_win"]
            sign = expected_sign(d["driver"], row, means, predictor.bundle["dc"], home, away)
            if sign == 0 or abs(net) < MIN_EFFECT:
                continue
            checked += 1
            if np.sign(net) != sign:
                wrong.append(f"{home} v {away}: {d['label']} {d['points']:+.1f} pts ({d['detail']})")
    assert checked >= N_MATCHES
    assert not wrong, "\n".join(wrong)


def test_explanation_matches_prediction(predictor):
    r = explain(predictor, predictor.teams[0], predictor.teams[1])
    assert r["probabilities"] == predictor.predict(predictor.teams[0], predictor.teams[1])
    assert 1 <= len(r["why"]) <= 5
    points = [abs(d["points"]) for d in r["why"]]
    assert points == sorted(points, reverse=True)
