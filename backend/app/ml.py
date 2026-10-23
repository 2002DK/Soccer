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
    """The production model and every model's test scores, V1 and V2, in one list.

    V2 entries use feature_set "v2"; their calibrated variants and calibration
    tables are under "v2" for the detailed view.
    """
    league = get_league(league_code)
    meta = json.loads(league.meta_path.read_text())
    metrics = json.loads(league.metrics_path.read_text())
    comparison = list(metrics["results"])
    v2 = metrics.get("v2")
    if v2:
        baseline = comparison[0]["baseline"]  # same test matches as V1
        comparison += [
            {"feature_set": "v2", "baseline": baseline, **{k: v for k, v in r.items() if k != "calibration"}}
            for r in v2["results"] if not r["calibrated"]
        ]
    return {"production": meta, "comparison": comparison, "v2": v2}
