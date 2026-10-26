"""Refit a league's chosen model on every season and save it for the API.

The chosen model comes from the league's model comparison
(ml/models/<league>/metrics.json): the V2 production model when the V2
comparison has run (models_v2.py), otherwise V1's Logistic Regression. It
reuses the settings found there and is refit on all seasons, test seasons
included, since the test scores have already been recorded.

Writes to ml/models/<league>/:
- model.pkl: the fitted model(s), feature columns and class labels
- model_meta.json: what was trained, on what, and its test scores
"""

import argparse
import json
from datetime import datetime, timezone

import joblib
import numpy as np
import pandas as pd
from sklearn.base import clone

import dixon_coles
import models_v2
from leagues import League, add_league_argument
from train_models import LABELS, MODELS, feature_columns, feature_sets

MODEL_NAME = "Logistic Regression"
FEATURE_SET = "base"

# V2 candidate -> weight on Logistic Regression in the final probabilities (the rest is Dixon-Coles)
V2_LR_WEIGHT = {"Logistic Regression v2": 1.0, "Dixon-Coles": 0.0}
V2_DESCRIPTIONS = {
    "Logistic Regression v2": "Logistic Regression on Elo ratings, head-to-head and league home advantage",
    "Dixon-Coles": "Dixon-Coles scoreline model",
    "Blend": "Blend of an Elo-based Logistic Regression and the Dixon-Coles scoreline model",
}


def run(league: League) -> None:
    metrics = json.loads(league.metrics_path.read_text())
    if "v2" in metrics:
        run_v2(league, metrics["v2"])
    else:
        run_v1(league, metrics["results"])


def run_v2(league: League, report: dict) -> None:
    """Refit the V2 production model: Logistic Regression and/or Dixon-Coles, blended, optionally calibrated."""
    settings, production = report["settings"], report["production"]
    name = production["model"]
    if name == "Elo only":
        raise SystemExit("Elo only is a baseline, not a production model")
    lr_weight = V2_LR_WEIGHT.get(name, settings["blend_lr_weight"])

    features = pd.read_csv(league.features_path, parse_dates=["date"])
    played = features.dropna(subset=["result"])
    played = played[played["season"] >= league.season_label(league.first_season + 1)].reset_index(drop=True)
    cols = models_v2.v2_feature_columns(played)
    matches = models_v2.matches_for_dc(league)
    as_of = matches["date"].max() + pd.Timedelta(days=1)

    lr = models_v2.fit_lr(played, cols, settings["lr_C"], settings["lr_half_life_seasons"]) if lr_weight > 0 else None
    dc = dixon_coles.fit(matches.dropna(subset=["home_goals"]), as_of, settings["dc_half_life_days"]) if lr_weight < 1 else None

    calibrator = None
    if production["calibrated"]:
        # Fit on the last full season, predicted by models that only saw earlier seasons
        last_full = sorted(played["season"].unique())[-2]
        before, during = played["season"] < last_full, (played["season"] == last_full).to_numpy()
        y = played.loc[during, "result"].map(LABELS.index).to_numpy()
        proba = np.zeros((during.sum(), 3))
        if lr is not None:
            lr_before = models_v2.fit_lr(played[before], cols, settings["lr_C"], settings["lr_half_life_seasons"])
            proba += lr_weight * lr_before.predict_proba(played.loc[during, cols].astype(float))
        if dc is not None:
            m = matches.loc[matches["season"] >= league.season_label(league.first_season + 1)].reset_index(drop=True)
            proba += (1 - lr_weight) * dixon_coles.walk_forward(m, (m["season"] == last_full).to_numpy(), settings["dc_half_life_days"])
        calibrator = models_v2.fit_calibrator(proba, y)

    league.model_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump({
        "version": 2, "model_name": name, "lr": lr, "dc": dc, "lr_weight": lr_weight,
        "calibrator": calibrator, "features": cols, "labels": LABELS,
    }, league.model_path)

    chosen = next(r for r in report["results"] if r["model"] == name and r["calibrated"] == production["calibrated"])
    meta = {
        "model": name,
        "description": V2_DESCRIPTIONS[name] + (", calibrated" if production["calibrated"] else ""),
        "feature_set": "v2",
        "params": {k: settings[k] for k in settings if k != "n_features"},
        "n_features": len(cols),
        "trained_on": {
            "matches": len(played),
            "first_season": played["season"].min(),
            "last_match": played["date"].max().strftime("%Y-%m-%d"),
        },
        "test_scores": {
            k: chosen[k] for k in ["test_seasons", "n_test", "accuracy", "f1_macro", "precision_macro", "recall_macro", "log_loss", "brier"]
        },
        "created_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }
    league.meta_path.write_text(json.dumps(meta, indent=2))
    print(f"saved {meta['description']} trained on {len(played)} matches to {league.model_path}")


def run_v1(league: League, results: list[dict]) -> None:
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
