"""Bridge to the ML code in ml/scripts, which the API reuses rather than copies."""

import json
import sys
from functools import cache
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "ml" / "scripts"))

from leagues import DEFAULT_LEAGUE, get_league  # noqa: E402
from predict import Predictor  # noqa: E402


@cache
def get_predictor(league_code: str = DEFAULT_LEAGUE) -> Predictor:
    """Load a league's model and match data once per process."""
    return Predictor(get_league(league_code))


@cache
def get_model_info(league_code: str = DEFAULT_LEAGUE) -> dict:
    league = get_league(league_code)
    meta = json.loads(league.meta_path.read_text())
    metrics = json.loads(league.metrics_path.read_text())
    return {"production": meta, "comparison": metrics["results"]}
