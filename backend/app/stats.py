"""Season-to-date stats for the stats preview."""

import numpy as np
import pandas as pd

FORM_GAMES = 5


def team_results(matches: pd.DataFrame, season: str) -> pd.DataFrame:
    """One row per team per played match in `season`, from that team's point of view."""
    games = matches[matches["season"] == season]
    sides = []
    for team_col, opp_col, prefix, opp_prefix, venue in [
        ("home_team", "away_team", "home", "away", "H"),
        ("away_team", "home_team", "away", "home", "A"),
    ]:
        sides.append(pd.DataFrame({
            "date": games["date"],
            "team": games[team_col],
            "opponent": games[opp_col],
            "venue": venue,
            "gf": games[f"{prefix}_goals"],
            "ga": games[f"{opp_prefix}_goals"],
            "xg_for": games[f"{prefix}_xg"],
            "xg_against": games[f"{opp_prefix}_xg"],
        }))
    results = pd.concat(sides, ignore_index=True).sort_values(["date", "team"])
    results["outcome"] = np.select([results["gf"] > results["ga"], results["gf"] == results["ga"]], ["W", "D"], "L")
    results["pts"] = results["outcome"].map({"W": 3, "D": 1, "L": 0})
    return results


def league_table(results: pd.DataFrame, teams: list[str]) -> pd.DataFrame:
    table = results.groupby("team").agg(
        played=("pts", "size"),
        won=("outcome", lambda s: (s == "W").sum()),
        drawn=("outcome", lambda s: (s == "D").sum()),
        lost=("outcome", lambda s: (s == "L").sum()),
        goals_for=("gf", "sum"),
        goals_against=("ga", "sum"),
        points=("pts", "sum"),
    ).reindex(teams, fill_value=0)
    table["goal_difference"] = table["goals_for"] - table["goals_against"]
    # Premier League tie-breakers: points, goal difference, goals scored, then alphabetical
    table = table.rename_axis("team").reset_index().sort_values(
        ["points", "goal_difference", "goals_for", "team"], ascending=[False, False, False, True]
    )
    table.insert(0, "position", range(1, len(table) + 1))
    return table.set_index("team")


def team_stats(matches: pd.DataFrame, season: str, teams: list[str], team: str) -> dict:
    results = team_results(matches, season)
    row = league_table(results, teams).loc[team]
    own = results[results["team"] == team]
    played = int(row["played"])

    def per_game(total: float) -> float | None:
        return round(total / played, 2) if played else None

    # xG is missing for a match only if Understat had no data for it; then report none rather than a partial sum
    has_xg = played > 0 and own["xg_for"].notna().all()
    recent = own.tail(FORM_GAMES)

    return {
        "team": team,
        "season": season,
        "position": int(row["position"]),
        "played": played,
        "won": int(row["won"]),
        "drawn": int(row["drawn"]),
        "lost": int(row["lost"]),
        "points": int(row["points"]),
        "goals_for": int(row["goals_for"]),
        "goals_against": int(row["goals_against"]),
        "goal_difference": int(row["goal_difference"]),
        "goals_for_per_game": per_game(row["goals_for"]),
        "goals_against_per_game": per_game(row["goals_against"]),
        "xg_for": round(float(own["xg_for"].sum()), 2) if has_xg else None,
        "xg_against": round(float(own["xg_against"].sum()), 2) if has_xg else None,
        "form": [
            {
                "date": r.date.strftime("%Y-%m-%d"),
                "opponent": r.opponent,
                "venue": r.venue,
                "goals_for": int(r.gf),
                "goals_against": int(r.ga),
                "outcome": r.outcome,
            }
            for r in recent.itertuples()
        ],
    }
