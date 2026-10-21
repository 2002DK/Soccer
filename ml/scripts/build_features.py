"""Build model input features from the clean match table.

Reads ml/data/processed/matches.csv and writes ml/data/processed/features.csv
with one row per match: identifiers, the target (`result`: H / D / A) and the
features. Every feature uses only matches played before that match's date, so
nothing leaks from the result being predicted.

build_features() also accepts unplayed fixtures, which is how the API will get
features for a match it is asked to predict.

Feature groups (each for the home side, the away side and, where useful, the
home-minus-away difference):
- form_*: average over the team's last 5 league matches (any venue), carried
  over from the previous season only if the team was in the league then
- venue_*: last 5 home matches for the home side, last 5 away for the away side
- season_*: season-to-date points and goal difference per game, and league
  position before the matchday
- h2h_*: last 5 meetings between the two teams, from the home side's view
"""

from pathlib import Path

import numpy as np
import pandas as pd

DATA_DIR = Path(__file__).resolve().parents[1] / "data" / "processed"
MATCHES_PATH = DATA_DIR / "matches.csv"
FEATURES_PATH = DATA_DIR / "features.csv"

WINDOW = 5

# Per-team stats from that team's point of view: (home column, away column)
TEAM_STATS = {
    "gf": ("home_goals", "away_goals"),
    "ga": ("away_goals", "home_goals"),
    "sot_for": ("home_shots_on_target", "away_shots_on_target"),
    "sot_against": ("away_shots_on_target", "home_shots_on_target"),
    "xg_for": ("home_xg", "away_xg"),
    "xg_against": ("away_xg", "home_xg"),
}
FORM_STATS = ["pts", *TEAM_STATS]
VENUE_STATS = ["pts", "gf", "ga"]
DIFF_FEATURES = [
    "form_pts", "form_gf", "form_ga", "form_sot_for", "form_xg_for", "form_xg_against",
    "venue_pts", "season_ppg", "season_gd_pg", "season_position",
]


def to_long(matches: pd.DataFrame) -> pd.DataFrame:
    """One row per team per match."""
    sides = []
    for is_home, team_col, opp_col in [(True, "home_team", "away_team"), (False, "away_team", "home_team")]:
        side = matches[["match_id", "season", "date", team_col, opp_col]].rename(
            columns={team_col: "team", opp_col: "opponent"}
        )
        side["is_home"] = is_home
        for stat, (home_col, away_col) in TEAM_STATS.items():
            side[stat] = matches[home_col if is_home else away_col].astype("float64")
        sides.append(side)

    long = pd.concat(sides, ignore_index=True)
    long["pts"] = np.select(
        [long["gf"] > long["ga"], long["gf"] == long["ga"]], [3.0, 1.0], default=0.0
    )
    long.loc[long["gf"].isna(), "pts"] = np.nan  # unplayed fixture
    return long.sort_values(["team", "date", "match_id"]).reset_index(drop=True)


def add_stints(long: pd.DataFrame) -> pd.DataFrame:
    """Number each unbroken run of seasons a team spent in the league.

    Rolling form is computed within a stint, so a promoted team does not carry
    over form from a Premier League spell years ago.
    """
    seasons = sorted(long["season"].unique())
    season_idx = {s: i for i, s in enumerate(seasons)}

    team_seasons = long[["team", "season"]].drop_duplicates().copy()
    team_seasons["idx"] = team_seasons["season"].map(season_idx)
    team_seasons = team_seasons.sort_values(["team", "idx"])
    gap = team_seasons.groupby("team")["idx"].diff().ne(1)
    team_seasons["stint"] = gap.groupby(team_seasons["team"]).cumsum()
    # First season of a stint, excluding the first season in the data (no history to judge by)
    team_seasons["promoted"] = gap & (team_seasons["idx"] > 0)

    return long.merge(team_seasons[["team", "season", "stint", "promoted"]], on=["team", "season"])


def rolling_prior_mean(long: pd.DataFrame, keys: list[str], stats: list[str]) -> pd.DataFrame:
    """Mean of each stat over the previous WINDOW rows in each group, excluding the current row."""
    grouped = long.groupby(keys, sort=False)[stats]
    return grouped.transform(lambda s: s.shift(1).rolling(WINDOW, min_periods=1).mean())


def add_team_features(long: pd.DataFrame) -> pd.DataFrame:
    long = long.sort_values(["team", "date", "match_id"]).reset_index(drop=True)

    form = rolling_prior_mean(long, ["team", "stint"], FORM_STATS)
    long[[f"form_{s}" for s in FORM_STATS]] = form.to_numpy()

    venue = rolling_prior_mean(long, ["team", "stint", "is_home"], VENUE_STATS)
    long[[f"venue_{s}" for s in VENUE_STATS]] = venue.to_numpy()

    by_season = long.groupby(["team", "season"], sort=False)
    played = by_season.cumcount()
    prior_pts = by_season["pts"].transform(lambda s: s.shift(1).cumsum()).fillna(0)
    prior_gd = (long["gf"] - long["ga"]).groupby([long["team"], long["season"]]).transform(
        lambda s: s.shift(1).cumsum()
    ).fillna(0)
    long["season_played"] = played
    long["season_ppg"] = (prior_pts / played).where(played > 0)
    long["season_gd_pg"] = (prior_gd / played).where(played > 0).astype("float64")
    return long


def league_positions(matches: pd.DataFrame) -> pd.DataFrame:
    """Each team's league position at the start of every matchday it played on.

    Matches on the same date are treated as simultaneous, so a team's position
    never includes results from its own matchday.
    """
    rows = []
    for season, games in matches.groupby("season"):
        played = games.dropna(subset=["home_goals"])
        teams = sorted(set(games["home_team"]) | set(games["away_team"]))
        for date in sorted(games["date"].unique()):
            before = played[played["date"] < date]
            if before.empty:
                continue  # first matchday: everyone level, position is meaningless
            table = pd.DataFrame(index=teams, data={"pts": 0, "gd": 0, "gf": 0})
            for team_col, gf_col, ga_col in [
                ("home_team", "home_goals", "away_goals"),
                ("away_team", "away_goals", "home_goals"),
            ]:
                gf, ga = before[gf_col].astype(int), before[ga_col].astype(int)
                pts = np.select([gf > ga, gf == ga], [3, 1], default=0)
                agg = pd.DataFrame({"team": before[team_col], "pts": pts, "gd": gf - ga, "gf": gf}).groupby("team").sum()
                table = table.add(agg, fill_value=0)
            # Same tie-breakers as the Premier League: points, goal difference, goals scored
            table = table.sort_values(["pts", "gd", "gf"], ascending=False)
            for pos, team in enumerate(table.index, start=1):
                rows.append((season, date, team, pos))
    return pd.DataFrame(rows, columns=["season", "date", "team", "season_position"])


def h2h_features(matches: pd.DataFrame) -> pd.DataFrame:
    """Last WINDOW meetings between the two teams (any venue, any season), from the home side's view.

    h2h_score is +1 for a home-side win, 0 for a draw, -1 for a loss, so it can
    be flipped between the two teams' points of view.
    """
    m = matches[["match_id", "date", "home_team", "away_team", "home_goals", "away_goals"]].copy()
    m["team_a"] = np.minimum(m["home_team"], m["away_team"])
    a_is_home = m["home_team"] == m["team_a"]
    gd_home = (m["home_goals"] - m["away_goals"]).astype("float64")
    m["gd_a"] = gd_home.where(a_is_home, -gd_home)
    m["score_a"] = np.sign(m["gd_a"])
    m["pair"] = m["team_a"] + "|" + np.maximum(m["home_team"], m["away_team"])

    m = m.sort_values(["pair", "date", "match_id"])
    grouped = m.groupby("pair", sort=False)
    prior = grouped[["score_a", "gd_a"]].transform(lambda s: s.shift(1).rolling(WINDOW, min_periods=1).mean())
    sign = np.where(a_is_home.loc[m.index], 1, -1)

    return pd.DataFrame({
        "match_id": m["match_id"],
        "h2h_meetings": grouped.cumcount().clip(upper=WINDOW),
        "h2h_score": prior["score_a"] * sign,
        "h2h_gd": prior["gd_a"] * sign,
    })


def build_features(matches: pd.DataFrame, fixtures: pd.DataFrame | None = None) -> pd.DataFrame:
    """Feature rows for every match in `matches` and every unplayed match in `fixtures`.

    `fixtures` needs season, date, home_team and away_team, and must be dated
    after the last played match.
    """
    matches = matches.copy()
    matches["date"] = pd.to_datetime(matches["date"])
    if fixtures is not None:
        fixtures = fixtures.copy()
        fixtures["date"] = pd.to_datetime(fixtures["date"])
        matches = pd.concat([matches, fixtures], ignore_index=True)
    matches = matches.sort_values(["date", "home_team"]).reset_index(drop=True)
    matches["match_id"] = np.arange(len(matches))

    long = add_team_features(add_stints(to_long(matches)))
    long = long.merge(league_positions(matches), on=["season", "date", "team"], how="left")

    team_cols = [
        "promoted", "season_played", "season_ppg", "season_gd_pg", "season_position",
        *[f"form_{s}" for s in FORM_STATS], *[f"venue_{s}" for s in VENUE_STATS],
    ]
    out = matches[["match_id", "season", "date", "home_team", "away_team", "result"]]
    for prefix, is_home in [("home", True), ("away", False)]:
        side = long.loc[long["is_home"] == is_home, ["match_id", *team_cols]]
        out = out.merge(side.add_prefix(f"{prefix}_").rename(columns={f"{prefix}_match_id": "match_id"}), on="match_id")

    for feat in DIFF_FEATURES:
        out[f"diff_{feat}"] = out[f"home_{feat}"] - out[f"away_{feat}"]

    out = out.merge(h2h_features(matches), on="match_id")
    return out.sort_values("match_id").drop(columns="match_id").reset_index(drop=True)


def main() -> None:
    matches = pd.read_csv(MATCHES_PATH, parse_dates=["date"])
    features = build_features(matches)
    features.to_csv(FEATURES_PATH, index=False, date_format="%Y-%m-%d", float_format="%.4f")
    print(f"wrote {len(features)} rows x {features.shape[1]} columns to {FEATURES_PATH}")


if __name__ == "__main__":
    main()
