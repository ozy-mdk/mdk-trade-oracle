"""Integration tests for Tertip Horizons and Timeseries API endpoints."""

from fastapi.testclient import TestClient

from mdk_trading_oracle.api.app import app

client = TestClient(app)


def test_get_tertip_horizons():
    response = client.get("/api/v1/tertip/horizons?symbol=THYAO&broker_id=MLB&trade_date=2026-09-16")
    assert response.status_code == 200
    data = response.json()
    assert data["symbol"] == "THYAO"
    assert data["broker_id"] == "MLB"
    assert "diagnostic" in data
    assert "diagnostic_badge" in data["diagnostic"]
    assert "horizons" in data
    assert len(data["horizons"]) == 7  # 1D, 1W, 2W, 1M, 3M, 6M, 12M

    codes = [h["code"] for h in data["horizons"]]
    assert codes == ["1D", "1W", "2W", "1M", "3M", "6M", "12M"]


def test_get_tertip_timeseries():
    response = client.get("/api/v1/tertip/timeseries?symbol=THYAO&broker_id=MLB&limit_days=30")
    assert response.status_code == 200
    data = response.json()
    assert isinstance(data, list)
    assert len(data) > 0
    pt = data[-1]
    assert "time" in pt
    assert "trade_date" in pt
    assert "close_price" in pt
    assert "fifo_avg_cost" in pt
    assert "open_quantity" in pt
    assert "ewma_qty_5d" in pt
    assert "ewma_qty_21d" in pt
    assert "ewma_qty_63d" in pt
    assert "ewma_qty_126d" in pt
    assert "ewma_qty_252d" in pt


def test_get_tertip_horizons_big5():
    response = client.get("/api/v1/tertip/horizons?symbol=THYAO&broker_id=BIG5&trade_date=2026-09-16")
    assert response.status_code == 200
    data = response.json()
    assert data["symbol"] == "THYAO"
    assert data["broker_id"] == "BIG5"
    assert "diagnostic" in data
    assert "diagnostic_badge" in data["diagnostic"]
    assert "horizons" in data
    assert len(data["horizons"]) == 7
    codes = [h["code"] for h in data["horizons"]]
    assert codes == ["1D", "1W", "2W", "1M", "3M", "6M", "12M"]
    # Check that aggregated inventory and flow are positive/meaningful
    assert data["open_stock_quantity"] > 0
    assert data["fifo_avg_cost"] > 0


def test_get_tertip_timeseries_big5():
    response = client.get("/api/v1/tertip/timeseries?symbol=THYAO&broker_id=BIG5&limit_days=30")
    assert response.status_code == 200
    data = response.json()
    assert isinstance(data, list)
    assert len(data) > 0
    pt = data[-1]
    assert "time" in pt
    assert "trade_date" in pt
    assert "close_price" in pt
    assert "fifo_avg_cost" in pt
    assert "open_quantity" in pt
    assert "ewma_qty_5d" in pt
    assert "ewma_qty_21d" in pt
    assert "ewma_qty_63d" in pt
    assert "ewma_qty_126d" in pt
    assert "ewma_qty_252d" in pt


def test_get_tertip_horizons_kamu():
    response = client.get("/api/v1/tertip/horizons?symbol=THYAO&broker_id=KAMU&trade_date=2026-09-16")
    assert response.status_code == 200
    data = response.json()
    assert data["symbol"] == "THYAO"
    assert data["broker_id"] == "KAMU"
    assert "diagnostic" in data
    assert "diagnostic_badge" in data["diagnostic"]
    assert "horizons" in data
    assert len(data["horizons"]) == 7
    codes = [h["code"] for h in data["horizons"]]
    assert codes == ["1D", "1W", "2W", "1M", "3M", "6M", "12M"]
    assert data["open_stock_quantity"] > 0
    assert data["fifo_avg_cost"] > 0


def test_get_tertip_timeseries_kamu():
    response = client.get("/api/v1/tertip/timeseries?symbol=THYAO&broker_id=KAMU&limit_days=30")
    assert response.status_code == 200
    data = response.json()
    assert isinstance(data, list)
    assert len(data) > 0
    pt = data[-1]
    assert "time" in pt
    assert "trade_date" in pt
    assert "close_price" in pt
    assert "fifo_avg_cost" in pt
    assert "open_quantity" in pt
    assert "ewma_qty_5d" in pt
    assert "ewma_qty_21d" in pt
    assert "ewma_qty_63d" in pt
    assert "ewma_qty_126d" in pt
    assert "ewma_qty_252d" in pt

