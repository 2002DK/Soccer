""""Why" explanations: which factors push a prediction, and by how much.

A driver's effect is the change in the predicted probabilities when that
group of inputs is set to a league-average value, everything else unchanged:

    effect = prediction - prediction with the group at league average

For the Logistic Regression part, "league average" is each feature's
training mean. For the Dixon-Coles part (attack and defence ratings), it is
an average team on both sides; home advantage stays. Effects are measured
one group at a time, so they don't add up exactly to the gap between this
prediction and an all-average one.
"""

from dataclasses import dataclass, replace

import numpy as np
import pandas as pd


OUTCOME_KEYS = {"H": "home_win", "D": "draw", "A": "away_win"}


@dataclass(frozen=True)
class Group:
    key: str
    label: str
    columns: tuple[str, ...]  # Logistic Regression features in the group (empty for Dixon-Coles)


GROUPS = [
    Group("strength", "Team strength (Elo)", ("home_elo", "away_elo", "diff_elo")),
    Group("goals", "Attack and defence ratings", ()),
    Group("league", "League home advantage lately", ("league_home_rate", "no_crowd")),
]


def _detail(key: str, row: pd.Series, home: str, away: str, dc_rates) -> str:
    """The numbers behind a driver, in words."""
    def num(col, fmt="{:.1f}"):
        v = row.get(col)
        return "n/a" if v is None or pd.isna(v) else fmt.format(v)

    if key == "strength":
        return f"{home} {num('home_elo', '{:.0f}')} v {away} {num('away_elo', '{:.0f}')}"
    if key == "league":
        rate = row.get("league_home_rate")
        crowd = " (no crowds)" if row.get("no_crowd") == 1 else ""
        return f"home teams won {rate:.0%} of the last 380 league matches{crowd}" if pd.notna(rate) else "n/a"
    if key == "goals":
        lam, mu = dc_rates
        return f"expected goals {home} {lam:.1f} v {away} {mu:.1f}"
    raise ValueError(key)


def explain(predictor, home_team: str, away_team: str, on=None, top: int = 5) -> dict:
    """Probabilities plus the drivers behind them, biggest effect on the most likely outcome first."""
    b = predictor.bundle
    if b.get("version") != 2:
        raise ValueError("explanations need a V2 model")

    row = predictor.feature_row(home_team, away_team, on)
    X = row[predictor.features].to_frame().T.astype(float)
    dc_probs = b["dc"].outcome_probs(home_team, away_team) if b["dc"] is not None else None
    base = predictor.combine(X, dc_probs)
    predicted = predictor.labels[int(np.argmax(base))]
    i = predictor.labels.index(predicted)

    means = _training_means(b)
    drivers = []
    for g in GROUPS:
        if g.columns:
            if b["lr"] is None:
                continue
            X_avg = X.copy()
            cols = [c for c in g.columns if c in X_avg.columns]
            X_avg[cols] = [means[c] for c in cols]
            without = predictor.combine(X_avg, dc_probs)
        else:
            if b["dc"] is None:
                continue
            average_teams = replace(b["dc"], attack={}, defence={}, new_team_attack=0.0, new_team_defence=0.0)
            without = predictor.combine(X, average_teams.outcome_probs(home_team, away_team))
        effect = base - without
        drivers.append({
            "driver": g.key,
            "label": g.label,
            "detail": _detail(g.key, row, home_team, away_team, b["dc"].rates(home_team, away_team) if b["dc"] else None),
            "effect": {OUTCOME_KEYS[l]: round(float(e), 4) for l, e in zip(predictor.labels, effect)},
            "points": round(float(effect[i] * 100), 1),  # percentage points on the most likely outcome
        })

    drivers.sort(key=lambda d: abs(d["points"]), reverse=True)
    return {
        "probabilities": dict(zip(predictor.labels, base.round(4).tolist())),
        "predicted": predicted,
        "why": drivers[:top],
    }


def _training_means(bundle) -> dict[str, float]:
    """Each Logistic Regression feature's training mean (after missing values were filled)."""
    scaler = bundle["lr"].named_steps["standardscaler"]
    return dict(zip(bundle["features"], scaler.mean_))
