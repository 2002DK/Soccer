from urllib.parse import quote

import pytest
from fastapi.testclient import TestClient

from app.main import app

client = TestClient(app)


@pytest.fixture(scope="module")
def teams() -> list[str]:
    return client.get("/teams").json()["teams"]


def test_health():
    assert client.get("/health").json() == {"status": "ok"}


def test_teams(teams):
    body = client.get("/teams").json()
    assert len(teams) == 20
    assert teams == sorted(teams)
    assert body["season"] and body["data_through"]


def test_stats_positions_form_a_league_table(teams):
    positions = []
    for team in teams:
        r = client.get(f"/teams/{quote(team)}/stats")
        assert r.status_code == 200, team
        s = r.json()
        assert s["won"] + s["drawn"] + s["lost"] == s["played"]
        assert s["points"] == 3 * s["won"] + s["drawn"]
        assert len(s["form"]) == min(5, s["played"])
        positions.append(s["position"])
    assert sorted(positions) == list(range(1, 21))


def test_stats_unknown_team():
    assert client.get("/teams/Barcelona/stats").status_code == 404


def test_predict(teams):
    r = client.post("/predict", json={"home_team": teams[0], "away_team": teams[1]})
    assert r.status_code == 200
    body = r.json()
    p = body["probabilities"]
    assert sum(p.values()) == pytest.approx(1, abs=1e-3)
    assert body["prediction"] == {"home_win": "H", "draw": "D", "away_win": "A"}[max(p, key=p.get)]
    # Drivers: biggest effect on the most likely outcome first
    points = [abs(d["points"]) for d in body["why"]]
    assert points == sorted(points, reverse=True)
    assert all(d["label"] and d["detail"] for d in body["why"])


def test_predict_swapping_venue_changes_prediction(teams):
    home = client.post("/predict", json={"home_team": teams[0], "away_team": teams[1]}).json()["probabilities"]
    away = client.post("/predict", json={"home_team": teams[1], "away_team": teams[0]}).json()["probabilities"]
    assert home != away


@pytest.mark.parametrize("payload", [
    {"home_team": "Barcelona", "away_team": "Arsenal"},
    {"home_team": "Arsenal", "away_team": "Arsenal"},
    {"home_team": "Arsenal", "away_team": "Chelsea", "date": "2000-01-01"},
])
def test_predict_rejects_bad_input(payload):
    r = client.post("/predict", json=payload)
    assert r.status_code == 400
    assert r.json()["detail"]


def test_models():
    body = client.get("/models").json()
    production = body["production"]
    # The production model appears in the comparison list, so the Models page can show its scores
    assert any(r["model"] == production["model"] and r["feature_set"] == production["feature_set"] for r in body["comparison"])
    assert {"Logistic Regression", "Random Forest", "XGBoost"} <= {r["model"] for r in body["comparison"]}
    if body["v2"]:
        assert body["v2"]["production"]["model"] == production["model"]
        assert all(r["calibration"] for r in body["v2"]["results"])


def test_fixtures(teams):
    body = client.get("/fixtures").json()
    for f in body["fixtures"]:
        assert f["home_team"] in teams and f["away_team"] in teams
        assert sum(f["probabilities"].values()) == pytest.approx(1, abs=1e-3)
        assert f["kickoff"].endswith("Z")
        assert f["scorelines"]["scorelines"]
    kickoffs = [f["kickoff"] for f in body["fixtures"]]
    assert kickoffs == sorted(kickoffs)
