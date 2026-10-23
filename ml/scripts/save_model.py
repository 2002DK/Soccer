"""Refit a league's chosen model on every season and save it for the API.

The chosen model and feature set come from the league's model comparison
(ml/models/<league>/metrics.json), reusing the hyperparameters found there.
The model is refit on all seasons, test seasons included, since the test
scores have already been recorded.

Writes to ml/models/<league>/:
- model.pkl: the fitted pipeline, its feature columns and class labels
- model_meta.json: what was trained, on what, and its test scores
"""

import argparse
import json
from datetime import datetime, timezone

import joblib
import pandas as pd
from sklearn.base import clone

from leagues import League, add_league_argument
from train_models import LABELS, MODELS, feature_columns, feature_sets

MODEL_NAME = "Logistic Regression"
FEATURE_SET = "base"


def run(league: League) -> None:
    results = json.loads(league.metrics_path.read_text())["results"]
    chosen = next(r for r in results if r["model"] == MODEL_NAME and r["feature_set"] == FEATURE_SET)

    df = pd.read_csv(league.features_path, parse_dates=["date"]).dropna(subset=["result"])
    df = df[df["season"] >= feature_sets(league)[FEATURE_SET]].sort_values("date")
    cols = feature_columns(df, FEATURE_SET)

    model = clone(MODELS[MODEL_NAME][0]).set_params(**chosen["best_params"])
    model.fit(df[cols].astype(float), df["result"].map(LABELS.index))

    league.model_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump({"model": model, "features": cols, "labels": LABELS}, league.model_path)

    meta = {
        "model": MODEL_NAME,
        "feature_set": FEATURE_SET,
        "params": chosen["best_params"],
        "n_features": len(cols),
        "trained_on": {
            "matches": len(df),
            "first_season": df["season"].min(),
            "last_match": df["date"].max().strftime("%Y-%m-%d"),
        },
        "test_scores": {
            k: chosen[k] for k in ["test_seasons", "n_test", "accuracy", "f1_macro", "precision_macro", "recall_macro", "log_loss"]
        },
        "created_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }
    league.meta_path.write_text(json.dumps(meta, indent=2))
    print(f"saved {MODEL_NAME} ({FEATURE_SET}) trained on {len(df)} matches to {league.model_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    add_league_argument(parser)
    run(parser.parse_args().league)


if __name__ == "__main__":
    main()
