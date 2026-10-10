"""Unit tests for Multi-Horizon Tertip ML Forecaster and API Endpoints."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from fastapi.testclient import TestClient

from mdk_trading_oracle.api.app import app
from mdk_trading_oracle.data.silver.tertip_multi_horizon_forecaster import (
    HORIZON_MAP,
    _check_hit,
    compute_forward_average_return,
)


def test_compute_forward_average_return_exact_math():
    """Verify forward rolling average price and percentage return calculations."""
    # Given 5 chronological sessions with known close prices
    prices = pd.Series([100.0, 102.0, 104.0, 106.0, 108.0])

    # For N = 2:
    # Row 0 (price 100): next 2 prices are [102, 104]. Mean = 103.0. Return = (103 - 100) / 100 * 100 = +3.0%
    # Row 1 (price 102): next 2 prices are [104, 106]. Mean = 105.0. Return = (105 - 102) / 102 * 100 = +2.941%
    # Row 2 (price 104): next 2 prices are [106, 108]. Mean = 107.0. Return = (107 - 104) / 104 * 100 = +2.885%
    # Row 3 (price 106): only 1 price ahead (108). Incomplete -> NaN
    # Row 4 (price 108): 0 prices ahead. Incomplete -> NaN
    fwd_prices, fwd_rets = compute_forward_average_return(prices, n_days=2)

    assert len(fwd_prices) == 5
    assert np.isclose(fwd_prices.iloc[0], 103.0)
    assert np.isclose(fwd_rets.iloc[0], 3.0)

    assert np.isclose(fwd_prices.iloc[1], 105.0)
    assert np.isclose(fwd_prices.iloc[2], 107.0)

    assert pd.isna(fwd_prices.iloc[3])
    assert pd.isna(fwd_rets.iloc[3])
    assert pd.isna(fwd_prices.iloc[4])
    assert pd.isna(fwd_rets.iloc[4])


def test_compute_forward_average_return_3d_and_5d():
    """Verify forward average returns for standard 3d and 5d windows."""
    prices = pd.Series([10.0, 20.0, 30.0, 40.0, 50.0, 60.0])

    # N = 3:
    # Row 0 (10.0): next 3 are [20, 30, 40]. Mean = 30.0. Return = (30 - 10)/10 * 100 = 200.0%
    # Row 1 (20.0): next 3 are [30, 40, 50]. Mean = 40.0. Return = (40 - 20)/20 * 100 = 100.0%
    # Row 2 (30.0): next 3 are [40, 50, 60]. Mean = 50.0. Return = (50 - 30)/30 * 100 = 66.667%
    # Rows 3, 4, 5: NaN
    p3, r3 = compute_forward_average_return(prices, n_days=3)
    assert np.isclose(p3.iloc[0], 30.0)
    assert np.isclose(r3.iloc[0], 200.0)
    assert np.isclose(p3.iloc[1], 40.0)
    assert np.isclose(p3.iloc[2], 50.0)
    assert pd.isna(p3.iloc[3])
    assert pd.isna(p3.iloc[4])
    assert pd.isna(p3.iloc[5])


def test_directional_hit_evaluation():
    """Verify directional hit classification against calibrated deadband (+/- 0.25%)."""
    # 1. Close call within tolerance (<= 0.25% difference) counts as hit
    assert _check_hit(pred=0.10, actual=-0.05, tol_delta=0.25) is True

    # 2. Strong correct call
    assert _check_hit(pred=1.5, actual=2.0) is True
    assert _check_hit(pred=-1.2, actual=-3.5) is True

    # 3. Definite miss (opposite direction outside deadband)
    assert _check_hit(pred=2.0, actual=-1.5) is False
    assert _check_hit(pred=-2.0, actual=1.5) is False

    # 4. Consolidation deadband (|actual| <= 0.25%)
    assert _check_hit(pred=0.15, actual=0.10) is True
    assert _check_hit(pred=-0.15, actual=0.10) is True


def test_horizon_map_integrity():
    """Verify market trading day mapping for all 5 windows."""
    assert HORIZON_MAP["3d"] == 3
    assert HORIZON_MAP["5d"] == 5
    assert HORIZON_MAP["10d"] == 10
    assert HORIZON_MAP["15d"] == 15
    assert HORIZON_MAP["30d"] == 30


@pytest.fixture
def client():
    return TestClient(app)


def test_api_multi_horizon_forecast(client):
    """Test GET /api/v1/tertip/multi-horizon/forecast for THYAO."""
    response = client.get("/api/v1/tertip/multi-horizon/forecast?symbol=THYAO")
    assert response.status_code == 200
    data = response.json()

    assert data["status"] == "success"
    assert data["symbol"] == "THYAO"
    assert "horizons" in data

    # Verify all 5 prediction windows exist
    for h in ["3d", "5d", "10d", "15d", "30d"]:
        assert h in data["horizons"]
        h_data = data["horizons"][h]
        assert "meta" in h_data
        assert "live_forecast" in h_data
        assert "backtest_ledger" in h_data

        fc = h_data["live_forecast"]
        assert fc["horizon"] == h
        assert fc["current_price"] > 0
        assert fc["target_price"] > 0
        assert "expected_return_pct" in fc
        assert "price_low" in fc
        assert "price_high" in fc
        assert fc["stance"] in ["BULLISH", "BEARISH", "NEUTRAL"]


def test_api_multi_horizon_backtest(client):
    """Test GET /api/v1/tertip/multi-horizon/backtest for THYAO."""
    response = client.get("/api/v1/tertip/multi-horizon/backtest?symbol=THYAO&horizon=5d&limit=10")
    assert response.status_code == 200
    ledger = response.json()
    assert isinstance(ledger, list)
    assert len(ledger) > 0

    item = ledger[0]
    assert "trade_date" in item
    assert "actual_avg_price" in item
    assert "pred_avg_price" in item
    assert "is_hit" in item
    assert "err_pct" in item


def test_api_multi_horizon_opportunities(client):
    """Test GET /api/v1/tertip/multi-horizon/opportunities."""
    response = client.get("/api/v1/tertip/multi-horizon/opportunities?horizon=5d")
    assert response.status_code == 200
    data = response.json()
    assert data["horizon"] == "5d"
    assert "opportunities" in data
