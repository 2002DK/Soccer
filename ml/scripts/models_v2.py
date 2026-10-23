"""V2 models: Logistic Regression with Elo and recency weights, Dixon-Coles, and a blend.

Seasons are split three ways by time:
- train: from the second season up to two seasons before the test seasons
- valid: the two seasons before the test seasons, used to tune settings and
         to fit the calibration step
- test:  the league's test seasons, used only for the final scores

Candidates:
- Elo only: home/draw/away from the Elo rating gap alone (a baseline)
- Logistic Regression v2: V1's features plus Elo, the league's recent
  home-win rate and the no-crowd flag; training matches weighted by recency
- Dixon-Coles: scoreline model (dixon_coles.py), refitted weekly
- Blend: a weighted average of the two
Each is scored as it is and after a calibration step: a multinomial logistic
regression on the log-probabilities, fitted on the validation seasons, which
corrects systematic over- or under-confidence such as V1's home-win lean.

The production model is the candidate with the lowest test log loss. Results
are added to ml/models/<league>/metrics.json under "v2"; the V1 results stay.
"""

import argparse
import json

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, log_loss, precision_score, recall_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

import dixon_coles
from build_features import V2_COLUMNS
from leagues import League, add_league_argument

LABELS = ["H", "D", "A"]
ID_COLUMNS = ["season", "date", "home_team", "away_team", "result"]
XG_COLUMNS_MARK = "_xg_"

LR_C_GRID = [0.001, 0.003, 0.01, 0.03]
LR_HALF_LIFE_GRID = [None, 3, 6, 10]  # seasons; None = all seasons weigh the same
DC_HALF_LIFE_GRID = [180, 365, 730]  # days
BLEND_WEIGHT_GRID = [0.25, 0.5, 0.75]  # weight on Logistic Regression
CALIBRATION_BANDS = [0, 0.2, 0.3, 0.4, 0.5, 0.6, 1.0]
MIN_BAND = 50  # bands with fewer matches are reported but not judged


def v2_feature_columns(df: pd.DataFrame) -> list[str]:
    """V1 base features (no xG) plus the V2 additions."""
    return [c for c in df.columns if c not in ID_COLUMNS and XG_COLUMNS_MARK not in c]


def season_weights(seasons: pd.Series, half_life: float | None) -> np.ndarray:
    """1 for the latest season, halving every `half_life` seasons back."""
    if half_life is None:
        return np.ones(len(seasons))
    order = {s: i for i, s in enumerate(sorted(seasons.unique()))}
    age = seasons.map(order).max() - seasons.map(order)
    return (0.5 ** (age / half_life)).to_numpy()


def lr_pipeline(C: float):
    return make_pipeline(SimpleImputer(strategy="median"), StandardScaler(), LogisticRegression(C=C, max_iter=2000))


def fit_lr(rows: pd.DataFrame, cols: list[str], C: float, half_life: float | None):
    model = lr_pipeline(C)
    model.fit(rows[cols].astype(float), rows["result"].map(LABELS.index),
              logisticregression__sample_weight=season_weights(rows["season"], half_life))
    return model


def fit_calibrator(proba: np.ndarray, y: np.ndarray) -> LogisticRegression:
    return LogisticRegression(C=100, max_iter=1000).fit(np.log(np.clip(proba, 1e-6, 1)), y)


def calibrate(calibrator: LogisticRegression | None, proba: np.ndarray) -> np.ndarray:
    return proba if calibrator is None else calibrator.predict_proba(np.log(np.clip(proba, 1e-6, 1)))


def blend(lr_proba: np.ndarray, dc_proba: np.ndarray, lr_weight: float) -> np.ndarray:
    return lr_weight * lr_proba + (1 - lr_weight) * dc_proba


def calibration_table(proba: np.ndarray, y: np.ndarray) -> list[dict]:
    """Predicted vs actual rate per probability band, for each outcome."""
    rows = []
    for i, label in enumerate(LABELS):
        bands = pd.cut(proba[:, i], CALIBRATION_BANDS, include_lowest=True)
        df = pd.DataFrame({"pred": proba[:, i], "hit": y == i, "band": bands})
        for band, g in df.groupby("band", observed=True):
            actual = float(g["hit"].mean())
            rows.append({
                "outcome": label, "band": f"{max(band.left, 0):.1f}-{band.right:.1f}", "n": len(g),
                "predicted": round(float(g["pred"].mean()), 4), "actual": round(actual, 4),
                # Sampling noise: the actual rate's standard error at this band's size
                "standard_error": round((actual * (1 - actual) / len(g)) ** 0.5, 4),
            })
    return rows


def scores(proba: np.ndarray, y: np.ndarray) -> dict:
    pred = proba.argmax(axis=1)
    table = calibration_table(proba, y)
    home = [r for r in table if r["outcome"] == "H" and r["n"] >= MIN_BAND]
    return {
        "accuracy": accuracy_score(y, pred),
        "f1_macro": f1_score(y, pred, average="macro"),
        "precision_macro": precision_score(y, pred, average="macro", zero_division=0),
        "recall_macro": recall_score(y, pred, average="macro"),
        "log_loss": log_loss(y, proba, labels=range(3)),
        "brier": float(np.mean(np.sum((proba - np.eye(3)[y]) ** 2, axis=1))),
        "predicted_share": {LABELS[i]: float(np.mean(pred == i)) for i in range(3)},
        "confusion_matrix": confusion_matrix(y, pred, labels=range(3)).tolist(),
        "home_calibration_max_gap": max(abs(r["predicted"] - r["actual"]) for r in home),
        # A gap within two standard errors is indistinguishable from noise at these sample sizes
        "home_calibration_within_noise": all(abs(r["predicted"] - r["actual"]) <= 2 * r["standard_error"] for r in home),
        "mean_predicted": {LABELS[i]: float(proba[:, i].mean()) for i in range(3)},
        "actual_share": {LABELS[i]: float(np.mean(y == i)) for i in range(3)},
        "calibration": table,
    }


def splits(league: League, df: pd.DataFrame):
    first = league.season_label(league.first_season + 1)
    valid_from = league.season_label(league.test_from - 2)
    test_from = league.season_label(league.test_from)
    played = df.dropna(subset=["result"])
    in_range = played["season"] >= first
    train = in_range & (played["season"] < valid_from)
    valid = (played["season"] >= valid_from) & (played["season"] < test_from)
    test = played["season"] >= test_from
    return played, train.to_numpy(), valid.to_numpy(), test.to_numpy()


def matches_for_dc(league: League) -> pd.DataFrame:
    """The match table aligned with features.csv rows (same order: date, then home team)."""
    m = pd.read_csv(league.matches_path, parse_dates=["date"])
    return m.sort_values(["date", "home_team"]).reset_index(drop=True)


def run(league: League) -> dict:
    features = pd.read_csv(league.features_path, parse_dates=["date"])
    played, train, valid, test = splits(league, features)
    y = played["result"].map(LABELS.index).to_numpy()
    cols = v2_feature_columns(played)

    matches = matches_for_dc(league).loc[played.index].reset_index(drop=True)
    played = played.reset_index(drop=True)
    assert (matches["home_team"].to_numpy() == played["home_team"].to_numpy()).all()

    # --- Tune on the validation seasons ---
    best_lr = None
    for C in LR_C_GRID:
        for half_life in LR_HALF_LIFE_GRID:
            p = fit_lr(played[train], cols, C, half_life).predict_proba(played.loc[valid, cols].astype(float))
            ll = log_loss(y[valid], p, labels=range(3))
            if best_lr is None or ll < best_lr[0]:
                best_lr = (ll, C, half_life, p)
    _, lr_C, lr_half_life, lr_valid = best_lr

    best_dc = None
    for half_life in DC_HALF_LIFE_GRID:
        p = dixon_coles.walk_forward(matches, valid, half_life)
        ll = log_loss(y[valid], p, labels=range(3))
        if best_dc is None or ll < best_dc[0]:
            best_dc = (ll, half_life, p)
    _, dc_half_life, dc_valid = best_dc

    lr_weight = min(BLEND_WEIGHT_GRID, key=lambda w: log_loss(y[valid], blend(lr_valid, dc_valid, w), labels=range(3)))

    elo_cols = ["diff_elo"]
    elo_model = LogisticRegression(max_iter=1000).fit(played.loc[train | valid, elo_cols], y[train | valid])

    # --- Test: refit on train + valid; calibrators from the validation predictions ---
    lr_model = fit_lr(played[train | valid], cols, lr_C, lr_half_life)
    test_proba = {
        "Elo only": elo_model.predict_proba(played.loc[test, elo_cols]),
        "Logistic Regression v2": lr_model.predict_proba(played.loc[test, cols].astype(float)),
        "Dixon-Coles": dixon_coles.walk_forward(matches, test, dc_half_life),
    }
    test_proba["Blend"] = blend(test_proba["Logistic Regression v2"], test_proba["Dixon-Coles"], lr_weight)
    valid_proba = {
        "Elo only": LogisticRegression(max_iter=1000).fit(played.loc[train, elo_cols], y[train]).predict_proba(played.loc[valid, elo_cols]),
        "Logistic Regression v2": lr_valid,
        "Dixon-Coles": dc_valid,
        "Blend": blend(lr_valid, dc_valid, lr_weight),
    }

    results = []
    for name, proba in test_proba.items():
        for calibrated in (False, True):
            calibrator = fit_calibrator(valid_proba[name], y[valid]) if calibrated else None
            results.append({
                "model": name, "calibrated": calibrated,
                "n_test": int(test.sum()),
                "test_seasons": f"{league.season_label(league.test_from)} onwards",
                **scores(calibrate(calibrator, proba), y[test]),
            })

    production = min(results, key=lambda r: r["log_loss"])
    report = {
        "settings": {
            "lr_C": lr_C, "lr_half_life_seasons": lr_half_life, "dc_half_life_days": dc_half_life,
            "blend_lr_weight": lr_weight, "n_features": len(cols),
        },
        "splits": {
            "train": f"{played.loc[train, 'season'].min()} to {played.loc[train, 'season'].max()} ({int(train.sum())} matches)",
            "valid": f"{played.loc[valid, 'season'].min()} to {played.loc[valid, 'season'].max()} ({int(valid.sum())} matches)",
            "test": f"{played.loc[test, 'season'].min()} onwards ({int(test.sum())} matches)",
        },
        "results": results,
        "production": {"model": production["model"], "calibrated": production["calibrated"]},
    }

    metrics = json.loads(league.metrics_path.read_text())
    metrics["v2"] = report
    league.metrics_path.write_text(json.dumps(metrics, indent=2))

    print(f"settings: {report['settings']}")
    for r in results:
        print(
            f"  {r['model']:<24}{' (calibrated)' if r['calibrated'] else '':<14} acc {r['accuracy']:.3f}  "
            f"logloss {r['log_loss']:.4f}  brier {r['brier']:.4f}  home-cal max gap {r['home_calibration_max_gap']:.3f} "
            f"(within noise: {r['home_calibration_within_noise']})  "
            f"home predicted {r['mean_predicted']['H']:.3f} vs actual {r['actual_share']['H']:.3f}"
        )
    print(f"production: {production['model']}{' (calibrated)' if production['calibrated'] else ''}")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    add_league_argument(parser)
    run(parser.parse_args().league)


if __name__ == "__main__":
    main()
