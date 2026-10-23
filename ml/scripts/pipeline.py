"""Run a league's data and model pipeline in one command.

    python ml/scripts/pipeline.py --league epl
    python ml/scripts/pipeline.py --league epl --steps download,clean

Steps, in order:
- download: results from football-data.co.uk, xG from Understat
- clean:    combine seasons into matches.csv
- features: build features.csv
- compare:  tune and score every candidate model (slow; not run by default)
- fit:      refit the production model on all seasons and save it

compare needs a full training environment (XGBoost); the other steps also
run in the scheduled refresh job.
"""

import argparse

import build_features
import clean_data
import download_data
import download_xg
from leagues import add_league_argument

STEPS = ["download", "clean", "features", "compare", "fit"]
DEFAULT_STEPS = ["download", "clean", "features", "fit"]


def run_step(step: str, league) -> None:
    if step == "download":
        download_data.run(league)
        download_xg.run(league)
    elif step == "clean":
        clean_data.run(league)
    elif step == "features":
        build_features.run(league)
    elif step == "compare":
        import train_models  # imported only when needed: pulls in XGBoost
        train_models.run(league)
    elif step == "fit":
        import save_model
        save_model.run(league)


def parse_steps(value: str) -> list[str]:
    steps = [s.strip() for s in value.split(",") if s.strip()]
    unknown = [s for s in steps if s not in STEPS]
    if unknown:
        raise argparse.ArgumentTypeError(f"unknown step(s) {unknown}; choose from {STEPS}")
    return sorted(steps, key=STEPS.index)  # always run in pipeline order


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    add_league_argument(parser)
    parser.add_argument("--steps", type=parse_steps, default=DEFAULT_STEPS,
                        help=f"comma-separated steps (default: {','.join(DEFAULT_STEPS)})")
    args = parser.parse_args()

    for step in args.steps:
        print(f"\n== {args.league.name}: {step} ==")
        run_step(step, args.league)


if __name__ == "__main__":
    main()
