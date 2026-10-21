"""Bridge to the ML code in ml/scripts, which the API reuses rather than copies."""

import json
import sys
from functools import cache
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
ML_DIR = REPO_ROOT / "ml"
sys.path.insert(0, str(ML_DIR / "scripts"))

from predict import Predictor  # noqa: E402


@cache
def get_predictor() -> Predictor:
    """Load the model and match data once per process."""
    return Predictor()


@cache
def get_model_info() -> dict:
    meta = json.loads((ML_DIR / "models" / "model_meta.json").read_text())
    metrics = json.loads((ML_DIR / "models" / "metrics.json").read_text())
    return {"production": meta, "comparison": metrics["results"]}
