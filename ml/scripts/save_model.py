"""Refit the chosen model on every season and save it for the API.

The chosen model and feature set come from Phase 3's comparison
(ml/models/metrics.json), reusing the hyperparameters found there. The model
is refit on all seasons, test seasons included, since the test scores have
already been recorded.

Writes:
- ml/models/model.pkl: the fitted pipeline, its feature columns and class labels
- ml/models/model_meta.json: what was trained, on what, and its test scores
"""

import json
from datetime import datetime, timezone

import joblib
import pandas as pd
from sklearn.base import clone

from predict import MODEL_PATH
from train_models import FEATURE_SETS, FEATURES_PATH, LABELS, METRICS_PATH, MODELS, ML_DIR, feature_columns

MODEL_NAME = "Logistic Regression"
FEATURE_SET = "base"

META_PATH = ML_DIR / "models" / "model_meta.json"


def main() -> None:
    results = json.loads(METRICS_PATH.read_text())["results"]
    chosen = next(r for r in results if r["model"] == MODEL_NAME and r["feature_set"] == FEATURE_SET)

    df = pd.read_csv(FEATURES_PATH, parse_dates=["date"]).dropna(subset=["result"])
    df = df[df["season"] >= FEATURE_SETS[FEATURE_SET]].sort_values("date")
    cols = feature_columns(df, FEATURE_SET)

    model = clone(MODELS[MODEL_NAME][0]).set_params(**chosen["best_params"])
    model.fit(df[cols].astype(float), df["result"].map(LABELS.index))

    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump({"model": model, "features": cols, "labels": LABELS}, MODEL_PATH)

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
    META_PATH.write_text(json.dumps(meta, indent=2))
    print(f"saved {MODEL_NAME} ({FEATURE_SET}) trained on {len(df)} matches to {MODEL_PATH}")


if __name__ == "__main__":
    main()
