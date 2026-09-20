"""Automated integration tests for FastAPI backend serving endpoints."""

import pytest
from fastapi.testclient import TestClient

from mdk_trading_oracle.api.app import app


@pytest.fixture(scope="module")
def client():
    return TestClient(app)


def test_health_endpoint(client):
    response = client.get("/api/v1/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "healthy"
    assert "engine" in data


def test_metadata_instruments(client):
    response = client.get("/api/v1/meta/instruments")
    assert response.status_code == 200
    instruments = response.json()
    assert len(instruments) > 0
    symbols = [i["symbol"] for i in instruments]
    assert "XU030" in symbols
    assert "THYAO" in symbols


def test_metadata_brokers(client):
    response = client.get("/api/v1/meta/brokers")
    assert response.status_code == 200
    brokers = response.json()
    assert len(brokers) > 0
    # Primary target MLB must be at the top
    assert brokers[0]["broker_id"] == "MLB"


def test_market_summary(client):
    response = client.get("/api/v1/market/summary?symbol=THYAO&broker_id=MLB")
    assert response.status_code == 200
    data = response.json()
    assert data["symbol"] == "THYAO"
    assert data["broker_id"] == "MLB"
    assert "bias_badge" in data
    assert "broker_net_flow_tl" in data


def test_tertip_portfolio(client):
    response = client.get("/api/v1/tertip/portfolio?broker_id=MLB")
    assert response.status_code == 200
    data = response.json()
    assert data["broker_id"] == "MLB"
    assert "positions" in data
    assert "total_market_value_tl" in data


def test_tertip_lots(client):
    response = client.get("/api/v1/tertip/lots?broker_id=MLB&limit=10")
    assert response.status_code == 200
    lots = response.json()
    assert isinstance(lots, list)


def test_tertip_history(client):
    response = client.get("/api/v1/tertip/history?broker_id=MLB&limit=10")
    assert response.status_code == 200
    points = response.json()
    assert isinstance(points, list)


def test_event_study_scan(client):
    response = client.get("/api/v1/event-study/scan?symbol=THYAO&limit=10")
    assert response.status_code == 200
    items = response.json()
    assert isinstance(items, list)


def test_time_window_analyze(client):
    response = client.get("/api/v1/time-window/analyze?symbol=THYAO&broker_id=MLB")
    assert response.status_code == 200
    data = response.json()
    assert data["symbol"] == "THYAO"
    assert "windows" in data
    assert len(data["windows"]) > 0


def test_signals_all(client):
    response = client.get("/api/v1/signals/all")
    assert response.status_code == 200
    data = response.json()
    assert "forecast_date" in data
    assert "sector_allocations" in data
    assert "stock_reactions" in data


def test_candlesticks_1d(client):
    response = client.get("/api/v1/market/candles?symbol=THYAO&interval=1d&limit=10")
    assert response.status_code == 200
    candles = response.json()
    assert len(candles) > 0
    assert "open" in candles[0]
    assert "close" in candles[0]


def test_candlesticks_xu030(client):
    response = client.get("/api/v1/market/candles?symbol=XU030&interval=1d&limit=10")
    assert response.status_code == 200
    candles = response.json()
    assert len(candles) > 0
    assert "open" in candles[0]


def test_metadata_date_range(client):
    response = client.get("/api/v1/meta/date-range")
    assert response.status_code == 200
    data = response.json()
    assert "min_date" in data
    assert "max_date" in data
    assert "latest_date" in data
    assert data["latest_date"] >= "2026-09-16"
