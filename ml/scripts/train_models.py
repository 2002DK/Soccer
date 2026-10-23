"""Train and compare Logistic Regression, Random Forest and XGBoost.

Reads ml/data/<league>/processed/features.csv and writes
ml/models/<league>/metrics.json with each model's test-set scores, for the
model comparison page.

Matches are split by time, never at random: models are tuned on earlier
seasons with time-ordered cross-validation and scored on the league's
test_from season onwards, which they never see during training.

Each model is trained on up to two feature sets:
- base: all seasons except the first (a warm-up season with no form history), no xG
- xg:   from when Understat xG starts (2014-15), base features plus xG;
        only for leagues Understat covers
"""

import argparse
import json
import time

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score, confusion_matrix, f1_score, log_loss, precision_score, recall_score,
)
from sklearn.model_selection import GridSearchCV, TimeSeriesSplit
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier

from leagues import League, add_league_argument

ID_COLUMNS = ["season", "date", "home_team", "away_team", "result"]

LABELS = ["H", "D", "A"]  # class index = position in this list
CV = TimeSeriesSplit(n_splits=5)

MODELS = {
    "Logistic Regression": (
        make_pipeline(SimpleImputer(strategy="median"), StandardScaler(), LogisticRegression(max_iter=2000)),
        {"logisticregression__C": [0.0003, 0.001, 0.003, 0.01, 0.1]},
    ),
    "Random Forest": (
        make_pipeline(SimpleImputer(strategy="median"), RandomForestClassifier(n_estimators=500, random_state=42, n_jobs=-1)),
        {"randomforestclassifier__max_depth": [4, 6, 10, None], "randomforestclassifier__min_samples_leaf": [20, 50]},
    ),
    "XGBoost": (
        # XGBoost handles missing values natively, so no imputer
        XGBClassifier(subsample=0.8, colsample_bytree=0.8, eval_metric="mlogloss", random_state=42, n_jobs=-1),
        {"n_estimators": [100, 200, 400], "max_depth": [2, 3], "learning_rate": [0.01, 0.03]},
    ),
}


def feature_sets(league: League) -> dict[str, str]:
    """Feature set -> first season it trains on."""
    sets = {"base": league.season_label(league.first_season + 1)}
    if league.understat_code:
        sets["xg"] = league.season_label(league.understat_first_season)
    return sets


def feature_columns(df: pd.DataFrame, feature_set: str) -> list[str]:
    cols = [c for c in df.columns if c not in ID_COLUMNS]
    if feature_set == "base":
        cols = [c for c in cols if "_xg_" not in c]
    return cols


def split(df: pd.DataFrame, league: League, feature_set: str):
    rows = df[df["season"] >= feature_sets(league)[feature_set]].sort_values("date")
    cols = feature_columns(df, feature_set)
    X = rows[cols].astype(float)
    y = rows["result"].map(LABELS.index)
    is_test = rows["season"] >= league.season_label(league.test_from)
    return X[~is_test], y[~is_test], X[is_test], y[is_test], cols


def evaluate(y_true, proba) -> dict:
    pred = proba.argmax(axis=1)
    return {
        "accuracy": accuracy_score(y_true, pred),
        "f1_macro": f1_score(y_true, pred, average="macro"),
        "precision_macro": precision_score(y_true, pred, average="macro", zero_division=0),
        "recall_macro": recall_score(y_true, pred, average="macro"),
        "log_loss": log_loss(y_true, proba, labels=range(len(LABELS))),
        "predicted_share": {LABELS[i]: float(np.mean(pred == i)) for i in range(len(LABELS))},
        "confusion_matrix": confusion_matrix(y_true, pred, labels=range(len(LABELS))).tolist(),
    }


def run(league: League) -> None:
    df = pd.read_csv(league.features_path, parse_dates=["date"])
    df = df.dropna(subset=["result"])
    sets = feature_sets(league)
    test_from = league.season_label(league.test_from)

    results = []
    for feature_set in sets:
        X_train, y_train, X_test, y_test, cols = split(df, league, feature_set)
        print(f"\n[{feature_set}] {len(cols)} features, {len(X_train)} train / {len(X_test)} test matches")

        # Reference points every model should beat
        home_share = float(np.mean(y_test == LABELS.index("H")))
        prior = np.tile(np.bincount(y_train, minlength=3) / len(y_train), (len(y_test), 1))
        baseline = {
            "always_home_accuracy": home_share,
            "class_prior_log_loss": log_loss(y_test, prior, labels=range(len(LABELS))),
        }
        print(f"  baseline: always-home accuracy {home_share:.3f}, prior log loss {baseline['class_prior_log_loss']:.4f}")

        for name, (estimator, grid) in MODELS.items():
            start = time.perf_counter()
            search = GridSearchCV(estimator, grid, cv=CV, scoring="neg_log_loss", n_jobs=-1)
            search.fit(X_train, y_train)
            metrics = evaluate(y_test, search.predict_proba(X_test))
            print(
                f"  {name:<20} acc {metrics['accuracy']:.3f}  f1 {metrics['f1_macro']:.3f}  "
                f"logloss {metrics['log_loss']:.4f}  ({time.perf_counter() - start:.0f}s)  {search.best_params_}"
            )
            results.append({
                "model": name,
                "feature_set": feature_set,
                "train_seasons": f"{sets[feature_set]} to season before {test_from}",
                "test_seasons": f"{test_from} onwards",
                "n_train": len(X_train),
                "n_test": len(X_test),
                "best_params": search.best_params_,
                "cv_log_loss": -search.best_score_,
                **metrics,
                "baseline": baseline,
            })

    league.model_dir.mkdir(parents=True, exist_ok=True)
    league.metrics_path.write_text(json.dumps({"labels": LABELS, "results": results}, indent=2))
    print(f"\nwrote {league.metrics_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    add_league_argument(parser)
    run(parser.parse_args().league)


if __name__ == "__main__":
    main()
