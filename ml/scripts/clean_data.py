"""Combine a league's raw season CSVs into one clean match table.

Reads ml/data/<league>/raw/<division>_*.csv and writes
ml/data/<league>/processed/matches.csv with one row per played match, sorted
by date. Betting-odds columns are dropped; only match facts and stats are kept.
xG is joined from the Understat files (understat_*.json in the same folder),
so it is empty before 2014-15 and for leagues Understat doesn't cover.
"""

import argparse
import json
from pathlib import Path

import pandas as pd

from leagues import League, add_league_argument

# football-data.co.uk column -> our column name
COLUMNS = {
    "Date": "date",
    "Time": "kickoff",
    "HomeTeam": "home_team",
    "AwayTeam": "away_team",
    "FTHG": "home_goals",
    "FTAG": "away_goals",
    "FTR": "result",  # H / D / A
    "HTHG": "home_goals_ht",
    "HTAG": "away_goals_ht",
    "HTR": "result_ht",
    "Referee": "referee",
    "HS": "home_shots",
    "AS": "away_shots",
    "HST": "home_shots_on_target",
    "AST": "away_shots_on_target",
    "HF": "home_fouls",
    "AF": "away_fouls",
    "HC": "home_corners",
    "AC": "away_corners",
    "HY": "home_yellows",
    "AY": "away_yellows",
    "HR": "home_reds",
    "AR": "away_reds",
}

INT_COLUMNS = [
    "home_goals", "away_goals", "home_goals_ht", "away_goals_ht",
    "home_shots", "away_shots", "home_shots_on_target", "away_shots_on_target",
    "home_fouls", "away_fouls", "home_corners", "away_corners",
    "home_yellows", "away_yellows", "home_reds", "away_reds",
]


def read_season(path: Path) -> pd.DataFrame:
    for encoding in ("utf-8-sig", "latin-1"):
        try:
            df = pd.read_csv(path, encoding=encoding, usecols=lambda c: c in COLUMNS)
            break
        except UnicodeDecodeError:
            continue
    df = df.rename(columns=COLUMNS).reindex(columns=list(COLUMNS.values()))

    code = path.stem.split("_")[1]  # '2425'
    df.insert(0, "season", f"20{code[:2]}-{code[2:]}")
    return df


def read_understat(league: League) -> pd.DataFrame:
    """Played matches' xG from Understat, keyed by season and the two teams."""
    names = league.understat_names
    rows = []
    for path in sorted(league.raw_dir.glob(f"understat_{league.understat_code}_*.json")):
        season = league.season_label(int(path.stem.rsplit("_", 1)[1]))
        for m in json.loads(path.read_text(encoding="utf-8")):
            if not m["isResult"]:
                continue
            rows.append({
                "season": season,
                "home_team": names.get(m["h"]["title"], m["h"]["title"]),
                "away_team": names.get(m["a"]["title"], m["a"]["title"]),
                "home_xg": float(m["xG"]["h"]),
                "away_xg": float(m["xG"]["a"]),
            })
    return pd.DataFrame(rows, columns=["season", "home_team", "away_team", "home_xg", "away_xg"])


def add_xg(matches: pd.DataFrame, xg: pd.DataFrame) -> pd.DataFrame:
    # Each pair meets once per venue per season, so season + teams is a unique key
    merged = matches.merge(xg, on=["season", "home_team", "away_team"], how="left", validate="one_to_one")
    covered = merged["season"].isin(xg["season"].unique())
    missing = merged[covered & merged["home_xg"].isna()]
    if len(missing):
        print(f"warning: {len(missing)} matches in Understat seasons have no xG, e.g.")
        print(missing[["season", "date", "home_team", "away_team"]].head().to_string(index=False))
    return merged


def clean(df: pd.DataFrame) -> pd.DataFrame:
    # Trailing blank lines in the source files come through as empty rows
    df = df.dropna(subset=["date", "home_team", "away_team", "result"]).copy()

    # Dates are dd/mm/yy in older seasons and dd/mm/yyyy in newer ones
    df["date"] = pd.to_datetime(df["date"], dayfirst=True, format="mixed")

    for col in ["home_team", "away_team", "referee", "result", "result_ht"]:
        df[col] = df[col].astype("string").str.strip()

    for col in INT_COLUMNS:
        df[col] = pd.to_numeric(df[col], errors="coerce").astype("Int64")

    df = df.drop_duplicates(subset=["date", "home_team", "away_team"])
    return df.sort_values(["date", "kickoff", "home_team"], na_position="first").reset_index(drop=True)


def run(league: League) -> None:
    files = sorted(league.raw_dir.glob(f"{league.football_data_code}_*.csv"))
    if not files:
        raise SystemExit(f"No raw files in {league.raw_dir}. Run download_data.py first.")

    matches = clean(pd.concat([read_season(f) for f in files], ignore_index=True))
    matches = add_xg(matches, read_understat(league))

    out = league.matches_path
    out.parent.mkdir(parents=True, exist_ok=True)
    matches.to_csv(out, index=False, date_format="%Y-%m-%d")
    print(f"wrote {len(matches)} matches from {len(files)} seasons to {out}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    add_league_argument(parser)
    run(parser.parse_args().league)


if __name__ == "__main__":
    main()
