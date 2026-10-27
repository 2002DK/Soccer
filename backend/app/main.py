"""Football Predictor API.

Run locally from the backend folder:

    uvicorn app.main:app --reload

Interactive docs are served at /docs. Every endpoint is league-scoped under
/leagues/{league}; the V1 routes (/teams, /predict, ...) still answer for the
Premier League.
"""

import os
from datetime import date as Date
from functools import lru_cache
from typing import Literal

from fastapi import APIRouter, FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from .ml import (
    DEFAULT_LEAGUE, get_accuracy, get_league, get_model_info, get_predictor, get_ratings, get_upcoming, league_codes,
)
from .stats import team_stats

DEFAULT_ORIGINS = "http://localhost:3000,http://localhost:5173,https://2002dk.github.io"

app = FastAPI(
    title="Football Predictor API",
    description="Predicts match outcomes (home win, draw, away win), scorelines and the reasons behind them.",
    version="2.0.0",
)
app.add_middleware(
    CORSMiddleware,
    allow_origins=os.environ.get("ALLOWED_ORIGINS", DEFAULT_ORIGINS).split(","),
    allow_methods=["GET", "POST"],
    allow_headers=["Content-Type"],
)


class League(BaseModel):
    code: str
    name: str
    season: str
    data_through: Date


class TeamsResponse(BaseModel):
    season: str
    teams: list[str]
    data_through: Date = Field(description="date of the last played match in the data")


class PredictRequest(BaseModel):
    home_team: str
    away_team: str
    date: Date | None = Field(None, description="match date; defaults to the day after the last played match")


class Probabilities(BaseModel):
    home_win: float
    draw: float
    away_win: float


class Driver(BaseModel):
    driver: str = Field(description="driver id, e.g. 'strength'")
    label: str
    detail: str = Field(description="the numbers behind the driver, in words")
    effect: Probabilities = Field(description="change in each probability caused by this driver")
    points: float = Field(description="percentage points added to (or taken from) the most likely outcome")


class PredictResponse(BaseModel):
    home_team: str
    away_team: str
    date: Date
    probabilities: Probabilities
    prediction: Literal["H", "D", "A"]
    model: str
    why: list[Driver] = Field(description="what pushes this prediction, biggest effect first")
    scorelines: dict | None = Field(None, description="expected goals, likely scores, over 2.5 and both teams to score")


def predictor_for(league: str):
    if league not in league_codes():
        raise HTTPException(404, f"unknown league {league!r}; available: {', '.join(league_codes())}")
    return get_predictor(league)


@app.get("/health")
def health() -> dict:
    return {"status": "ok"}


@app.get("/leagues", response_model=list[League])
def leagues() -> list[League]:
    out = []
    for code in league_codes():
        p = get_predictor(code)
        out.append(League(code=code, name=get_league(code).name, season=p.season, data_through=p.last_played))
    return out


router = APIRouter(prefix="/leagues/{league}")


@router.get("/teams", response_model=TeamsResponse)
def teams(league: str) -> TeamsResponse:
    p = predictor_for(league)
    return TeamsResponse(season=p.season, teams=p.teams, data_through=p.last_played)


@router.get("/teams/{team}/stats")
def stats(league: str, team: str) -> dict:
    p = predictor_for(league)
    if team not in p.teams:
        raise HTTPException(404, f"{team!r} is not a {p.season} {p.league.name} team")
    out = team_stats(p.matches, p.season, p.teams, team)
    rating = next((r for r in get_ratings(league)["ratings"] if r["team"] == team), None)
    out["elo"] = rating and {"rating": rating["elo"], "rank": rating["rank"], "season_change": rating["season_change"]}
    return out


@lru_cache(maxsize=1024)
def cached_prediction(league: str, home_team: str, away_team: str, on: Date | None) -> dict:
    """Probabilities, drivers and scorelines. Building features takes ~1.5s and the
    data only changes on redeploy, so repeat requests are cached."""
    p = get_predictor(league)
    if p.bundle.get("version") == 2:
        return {**p.explain(home_team, away_team, on), "scorelines": p.scorelines(home_team, away_team)}
    return {"probabilities": p.predict(home_team, away_team, on), "why": [], "scorelines": None}  # V1 models


@router.post("/predict", response_model=PredictResponse)
def predict(league: str, req: PredictRequest) -> PredictResponse:
    p = predictor_for(league)
    try:
        result = cached_prediction(league, req.home_team, req.away_team, req.date)
    except ValueError as e:
        raise HTTPException(400, str(e))

    proba = result["probabilities"]
    production = get_model_info(league)["production"]
    return PredictResponse(
        home_team=req.home_team,
        away_team=req.away_team,
        date=req.date or p.default_date(),
        probabilities=Probabilities(home_win=proba["H"], draw=proba["D"], away_win=proba["A"]),
        prediction=max(proba, key=proba.get),
        model=production.get("description", production["model"]),
        why=result["why"],
        scorelines=result["scorelines"],
    )


@router.get("/fixtures")
def fixtures(league: str) -> dict:
    """Predictions for matches in the next 7 days, worked out by the daily job (kick-off times in UTC)."""
    predictor_for(league)
    return get_upcoming(league)


@router.get("/ratings")
def ratings(league: str) -> dict:
    """Current Elo ratings, best first, with each team's rating trend this season."""
    predictor_for(league)
    return get_ratings(league)


@router.get("/accuracy")
def accuracy(league: str) -> dict:
    """How predictions frozen before kick-off have done: scores, baselines, by week and per match."""
    predictor_for(league)
    return get_accuracy(league)


@router.get("/models")
def models(league: str) -> dict:
    """The production model's details and every model's test scores from training."""
    predictor_for(league)
    return get_model_info(league)


app.include_router(router)


# V1 routes, kept working for the Premier League
@app.get("/teams", response_model=TeamsResponse, include_in_schema=False)
def teams_v1() -> TeamsResponse:
    return teams(DEFAULT_LEAGUE)


@app.get("/teams/{team}/stats", include_in_schema=False)
def stats_v1(team: str) -> dict:
    return stats(DEFAULT_LEAGUE, team)


@app.post("/predict", response_model=PredictResponse, include_in_schema=False)
def predict_v1(req: PredictRequest) -> PredictResponse:
    return predict(DEFAULT_LEAGUE, req)


@app.get("/fixtures", include_in_schema=False)
def fixtures_v1() -> dict:
    return fixtures(DEFAULT_LEAGUE)


@app.get("/accuracy", include_in_schema=False)
def accuracy_v1() -> dict:
    return accuracy(DEFAULT_LEAGUE)


@app.get("/models", include_in_schema=False)
def models_v1() -> dict:
    return models(DEFAULT_LEAGUE)
