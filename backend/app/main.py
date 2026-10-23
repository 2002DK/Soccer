"""Premier League Predictor API.

Run locally from the backend folder:

    uvicorn app.main:app --reload

Interactive docs are served at /docs.
"""

import os
from datetime import date as Date
from functools import lru_cache
from typing import Literal

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from .ml import get_model_info, get_predictor
from .stats import team_stats

DEFAULT_ORIGINS = "http://localhost:3000,http://localhost:5173,https://2002dk.github.io"

app = FastAPI(
    title="Premier League Predictor API",
    description="Predicts Premier League match outcomes (home win, draw, away win).",
    version="1.0.0",
)
app.add_middleware(
    CORSMiddleware,
    allow_origins=os.environ.get("ALLOWED_ORIGINS", DEFAULT_ORIGINS).split(","),
    allow_methods=["GET", "POST"],
    allow_headers=["Content-Type"],
)


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


class PredictResponse(BaseModel):
    home_team: str
    away_team: str
    date: Date
    probabilities: Probabilities
    prediction: Literal["H", "D", "A"]
    model: str


@app.get("/health")
def health() -> dict:
    return {"status": "ok"}


@app.get("/teams", response_model=TeamsResponse)
def teams() -> TeamsResponse:
    p = get_predictor()
    return TeamsResponse(season=p.season, teams=p.teams, data_through=p.last_played)


@app.get("/teams/{team}/stats")
def stats(team: str) -> dict:
    p = get_predictor()
    if team not in p.teams:
        raise HTTPException(404, f"{team!r} is not a {p.season} Premier League team")
    return team_stats(p.matches, p.season, p.teams, team)


@lru_cache(maxsize=1024)
def cached_prediction(home_team: str, away_team: str, on: Date | None) -> dict[str, float]:
    # Building features takes ~1.5s and the data only changes on redeploy, so repeat requests are cached
    return get_predictor().predict(home_team, away_team, on)


@app.post("/predict", response_model=PredictResponse)
def predict(req: PredictRequest) -> PredictResponse:
    p = get_predictor()
    try:
        proba = cached_prediction(req.home_team, req.away_team, req.date)
    except ValueError as e:
        raise HTTPException(400, str(e))

    production = get_model_info()["production"]
    return PredictResponse(
        home_team=req.home_team,
        away_team=req.away_team,
        date=req.date or p.default_date(),
        probabilities=Probabilities(home_win=proba["H"], draw=proba["D"], away_win=proba["A"]),
        prediction=max(proba, key=proba.get),
        model=production.get("description", production["model"]),
    )


@app.get("/models")
def models() -> dict:
    """The production model's details and every model's test scores from training."""
    return get_model_info()
