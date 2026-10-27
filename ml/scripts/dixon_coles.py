"""Dixon-Coles scoreline model.

Each side's goals follow a Poisson distribution with rate

    home: exp(intercept + home + attack[home team] + defence[away team])
    away: exp(intercept +        attack[away team] + defence[home team])

where a higher attack means more goals scored and a higher defence means
more goals conceded. Matches are weighted by age (half-life in days), so
recent form counts more, and Dixon and Coles' rho term corrects the Poisson
model's known underestimate of low-scoring draws (0-0, 1-1).

Ratings are fitted as a Poisson regression on the last few years of matches.
Teams with no matches in that window (usually promoted sides) get a
below-average default, since promoted teams are usually among the weakest.
"""

from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from scipy.optimize import minimize_scalar
from scipy.stats import poisson
from sklearn.linear_model import PoissonRegressor

MAX_GOALS = 10  # score grid is 0..MAX_GOALS for each side
GRID_GOALS = 5  # the grid shown on the site: 0..5 goals each
WINDOW_DAYS = 4 * 365  # matches older than this are left out of a fit
NEW_TEAM_QUANTILE = 0.25  # unseen teams get this quantile's attack (and the mirror quantile's defence)


@dataclass
class DixonColes:
    intercept: float
    home: float
    rho: float
    attack: dict[str, float] = field(default_factory=dict)
    defence: dict[str, float] = field(default_factory=dict)
    new_team_attack: float = 0.0
    new_team_defence: float = 0.0

    def rates(self, home_team: str, away_team: str) -> tuple[float, float]:
        """Expected goals (home, away)."""
        att = lambda t: self.attack.get(t, self.new_team_attack)  # noqa: E731
        dfc = lambda t: self.defence.get(t, self.new_team_defence)  # noqa: E731
        lam = np.exp(self.intercept + self.home + att(home_team) + dfc(away_team))
        mu = np.exp(self.intercept + att(away_team) + dfc(home_team))
        return float(lam), float(mu)

    def score_matrix(self, home_team: str, away_team: str) -> np.ndarray:
        """P(home scores i, away scores j) for i, j in 0..MAX_GOALS."""
        lam, mu = self.rates(home_team, away_team)
        goals = np.arange(MAX_GOALS + 1)
        grid = np.outer(poisson.pmf(goals, lam), poisson.pmf(goals, mu))
        grid[:2, :2] *= _tau_grid(lam, mu, self.rho)
        return grid / grid.sum()

    def outcome_probs(self, home_team: str, away_team: str) -> np.ndarray:
        """[P(home win), P(draw), P(away win)]."""
        grid = self.score_matrix(home_team, away_team)
        return np.array([np.tril(grid, -1).sum(), np.trace(grid), np.triu(grid, 1).sum()])

    def summary(self, home_team: str, away_team: str, top: int = 3) -> dict:
        """Expected goals, most likely scorelines, over 2.5 goals and both teams to score."""
        grid = self.score_matrix(home_team, away_team)
        lam, mu = self.rates(home_team, away_team)
        order = np.argsort(grid, axis=None)[::-1][:top]
        i, j = np.indices(grid.shape)
        return {
            "expected_goals": {"home": round(lam, 2), "away": round(mu, 2)},
            "scorelines": [
                {"home": int(h), "away": int(a), "probability": round(float(grid[h, a]), 4)}
                for h, a in zip(*np.unravel_index(order, grid.shape))
            ],
            "over_2_5": round(float(grid[i + j > 2].sum()), 4),
            "both_teams_score": round(float(grid[1:, 1:].sum()), 4),
            # P(home i, away j) for 0..GRID_GOALS goals each; rows are home goals
            "grid": grid[:GRID_GOALS + 1, :GRID_GOALS + 1].round(4).tolist(),
        }


def _tau_grid(lam: float, mu: float, rho: float) -> np.ndarray:
    """Dixon-Coles adjustment for scores 0-0, 0-1, 1-0, 1-1 (rows: home goals, cols: away goals)."""
    return np.array([[1 - lam * mu * rho, 1 + lam * rho], [1 + mu * rho, 1 - rho]])


def _tau(hg, ag, lam, mu, rho):
    out = np.ones_like(lam)
    out = np.where((hg == 0) & (ag == 0), 1 - lam * mu * rho, out)
    out = np.where((hg == 0) & (ag == 1), 1 + lam * rho, out)
    out = np.where((hg == 1) & (ag == 0), 1 + mu * rho, out)
    out = np.where((hg == 1) & (ag == 1), 1 - rho, out)
    return out


def fit(played: pd.DataFrame, as_of: pd.Timestamp, half_life_days: float) -> DixonColes:
    """Fit on played matches dated before `as_of`, weighting each by its age."""
    window = played[(played["date"] < as_of) & (played["date"] >= as_of - pd.Timedelta(days=WINDOW_DAYS))]
    age = (as_of - window["date"]).dt.days.to_numpy()
    weight = 0.5 ** (age / half_life_days)

    teams = sorted(set(window["home_team"]) | set(window["away_team"]))
    idx = {t: i for i, t in enumerate(teams)}
    n, k = len(window), len(teams)

    # Two rows per match: home goals, then away goals. Columns: attack (k), defence (k), home
    X = np.zeros((2 * n, 2 * k + 1))
    rows = np.arange(n)
    home_i = window["home_team"].map(idx).to_numpy()
    away_i = window["away_team"].map(idx).to_numpy()
    X[rows, home_i] = 1
    X[rows, k + away_i] = 1
    X[rows, 2 * k] = 1
    X[n + rows, away_i] = 1
    X[n + rows, k + home_i] = 1
    y = np.concatenate([window["home_goals"].to_numpy(), window["away_goals"].to_numpy()]).astype(float)
    w = np.concatenate([weight, weight])

    glm = PoissonRegressor(alpha=1e-4, max_iter=500).fit(X, y, sample_weight=w)
    attack, defence, home = glm.coef_[:k], glm.coef_[k:2 * k], glm.coef_[2 * k]

    # rho: maximise the weighted likelihood of the low-score adjustment, holding the rates fixed
    lam = np.exp(glm.intercept_ + home + attack[home_i] + defence[away_i])
    mu = np.exp(glm.intercept_ + attack[away_i] + defence[home_i])
    hg, ag = window["home_goals"].to_numpy(), window["away_goals"].to_numpy()

    def neg_ll(rho):
        tau = _tau(hg, ag, lam, mu, rho)
        return np.inf if (tau <= 0).any() else -np.sum(weight * np.log(tau))

    rho = minimize_scalar(neg_ll, bounds=(-0.2, 0.2), method="bounded").x

    return DixonColes(
        intercept=float(glm.intercept_), home=float(home), rho=float(rho),
        attack=dict(zip(teams, attack.round(6).tolist())),
        defence=dict(zip(teams, defence.round(6).tolist())),
        new_team_attack=float(np.quantile(attack, NEW_TEAM_QUANTILE)),
        new_team_defence=float(np.quantile(defence, 1 - NEW_TEAM_QUANTILE)),
    )


def walk_forward(matches: pd.DataFrame, predict_mask: np.ndarray, half_life_days: float) -> np.ndarray:
    """Home/draw/away probabilities for the masked matches, refitting once per week on earlier matches only.

    Returns an (n_masked, 3) array in the order of the masked rows.
    """
    played = matches.dropna(subset=["home_goals"])
    targets = matches.loc[predict_mask]
    out = np.zeros((len(targets), 3))
    weeks = targets["date"].dt.to_period("W-MON").dt.start_time  # weeks start on Tuesday
    for week_start in sorted(weeks.unique()):
        in_week = (weeks == week_start).to_numpy()
        model = fit(played, pd.Timestamp(week_start), half_life_days)
        out[in_week] = [model.outcome_probs(h, a) for h, a in targets.loc[in_week, ["home_team", "away_team"]].itertuples(index=False)]
    return out
