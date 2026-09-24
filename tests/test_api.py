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
    # Primary target MLB must be at the top, followed by BIG5 and KAMU bundles
    assert brokers[0]["broker_id"] == "MLB"
    assert brokers[1]["broker_id"] == "BIG5"
    assert brokers[1]["category"] == "Institutional Bundle"
    assert brokers[2]["broker_id"] == "KAMU"
    assert brokers[2]["category"] == "Institutional Bundle"


def test_big5_all_endpoints(client):
    # 1. Market Summary
    res = client.get("/api/v1/market/summary?symbol=THYAO&broker_id=BIG5&trade_date=2026-09-16")
    assert res.status_code == 200
    data = res.json()
    assert data["broker_id"] == "BIG5"
    assert data["fifo_avg_cost"] > 0
    assert data["broker_buy_turnover_tl"] > 0

    # 2. Daily Candlesticks with BIG5 flow
    res = client.get("/api/v1/market/candles?symbol=THYAO&interval=1d&broker_id=BIG5&limit=10")
    assert res.status_code == 200
    candles = res.json()
    assert len(candles) > 0
    assert "bofa_net_flow_tl" in candles[0]

    # 3. Tertip Portfolio
    res = client.get("/api/v1/tertip/portfolio?broker_id=BIG5&trade_date=2026-09-16")
    assert res.status_code == 200
    port = res.json()
    assert port["broker_id"] == "BIG5"
    assert len(port["positions"]) > 0

    # 4. Open Lots
    res = client.get("/api/v1/tertip/lots?broker_id=BIG5&symbol=THYAO&limit=10")
    assert res.status_code == 200
    assert isinstance(res.json(), list)

    # 5. Daily History
    res = client.get("/api/v1/tertip/history?broker_id=BIG5&symbol=THYAO&limit=10")
    assert res.status_code == 200
    assert isinstance(res.json(), list)

    # 6. Time Window Terminal
    res = client.get("/api/v1/time-window/analyze?symbol=THYAO&broker_id=BIG5&trade_date=2026-09-16")
    assert res.status_code == 200
    tw = res.json()
    assert tw["broker_id"] == "BIG5"
    assert len(tw["windows"]) == 5

    # 7. Event Study Scanner
    res = client.get("/api/v1/event-study/scan?broker_id=BIG5&symbol=THYAO&limit=10")
    assert res.status_code == 200
    assert isinstance(res.json(), list)


def test_kamu_all_endpoints(client):
    # 1. Market Summary
    res = client.get("/api/v1/market/summary?symbol=THYAO&broker_id=KAMU&trade_date=2026-09-16")
    assert res.status_code == 200
    data = res.json()
    assert data["broker_id"] == "KAMU"
    assert data["fifo_avg_cost"] > 0
    assert data["broker_buy_turnover_tl"] > 0

    # 2. Daily Candlesticks with KAMU flow
    res = client.get("/api/v1/market/candles?symbol=THYAO&interval=1d&broker_id=KAMU&limit=10")
    assert res.status_code == 200
    candles = res.json()
    assert len(candles) > 0
    assert "bofa_net_flow_tl" in candles[0]

    # 3. Tertip Portfolio
    res = client.get("/api/v1/tertip/portfolio?broker_id=KAMU&trade_date=2026-09-16")
    assert res.status_code == 200
    port = res.json()
    assert port["broker_id"] == "KAMU"
    assert len(port["positions"]) > 0

    # 4. Open Lots
    res = client.get("/api/v1/tertip/lots?broker_id=KAMU&symbol=THYAO&limit=10")
    assert res.status_code == 200
    assert isinstance(res.json(), list)

    # 5. Daily History
    res = client.get("/api/v1/tertip/history?broker_id=KAMU&symbol=THYAO&limit=10")
    assert res.status_code == 200
    assert isinstance(res.json(), list)

    # 6. Time Window Terminal
    res = client.get("/api/v1/time-window/analyze?symbol=THYAO&broker_id=KAMU&trade_date=2026-09-16")
    assert res.status_code == 200
    tw = res.json()
    assert tw["broker_id"] == "KAMU"
    assert len(tw["windows"]) == 5

    # 7. Event Study Scanner
    res = client.get("/api/v1/event-study/scan?broker_id=KAMU&symbol=THYAO&limit=10")
    assert res.status_code == 200
    assert isinstance(res.json(), list)


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
    assert "turnover_tl" in candles[0]


def test_market_summary_xu030(client):
    response = client.get("/api/v1/market/summary?symbol=XU030&broker_id=MLB")
    assert response.status_code == 200
    summary = response.json()
    assert summary["symbol"] == "XU030"
    assert "close_price" in summary
    assert "total_turnover_tl" in summary
    assert "broker_net_flow_tl" in summary
    assert "bias_badge" in summary


def test_time_window_xu030(client):
    response = client.get("/api/v1/time-window/analyze?symbol=XU030&broker_id=MLB")
    assert response.status_code == 200
    tw = response.json()
    assert tw["symbol"] == "XU030"
    assert "windows" in tw


def test_metadata_date_range(client):
    response = client.get("/api/v1/meta/date-range")
    assert response.status_code == 200
    data = response.json()
    assert "min_date" in data
    assert "max_date" in data
    assert "latest_date" in data
    assert data["latest_date"] >= "2026-09-16"


def test_market_shock_days(client):
    """Test /api/v1/market/shock-days endpoint returns shock days with correct schema and filters."""
    # 1. Default threshold (3%)
    response = client.get("/api/v1/market/shock-days")
    assert response.status_code == 200
    shocks = response.json()
    assert len(shocks) > 0
    first = shocks[0]
    assert "trade_date" in first
    assert "time" in first
    assert "close_price" in first
    assert "daily_return_pct" in first
    assert "shock_type" in first
    assert first["shock_type"] in ("POSITIVE_SHOCK", "NEGATIVE_SHOCK")
    assert "is_positive_shock" in first
    assert "is_negative_shock" in first
    assert "shock_magnitude_pct" in first
    assert abs(first["daily_return_pct"]) >= 0.03

    # 2. Dynamic threshold override (4%)
    res_4pct = client.get("/api/v1/market/shock-days?threshold_pct=0.04")
    assert res_4pct.status_code == 200
    shocks_4pct = res_4pct.json()
    assert len(shocks_4pct) <= len(shocks)
    for s in shocks_4pct:
        assert s["shock_magnitude_pct"] >= 0.04

