"""Unit and Integration Tests for Week Start Predictive Forecaster & Endpoints."""

from fastapi.testclient import TestClient

from mdk_trading_oracle.api.app import app
from mdk_trading_oracle.core.db import PostgresManager
from mdk_trading_oracle.data.silver.week_start_forecaster import (
    WEEK_START_FEATURES,
    extract_week_start_time_series,
    predict_live_week_start,
    run_weekly_walk_forward_arena,
)

client = TestClient(app)


def test_week_start_feature_columns():
    """Verify that all 22 lean features are defined and zero-leakage."""
    assert len(WEEK_START_FEATURES) == 22
    assert "feat_fri_w5_mlb_share" in WEEK_START_FEATURES
    assert "feat_wtd_mlb_net_share_turnover" in WEEK_START_FEATURES
    assert "feat_weekend_carry_cost_bps" in WEEK_START_FEATURES
    assert "feat_tertip_weekend_inventory_cost_spread_pct" in WEEK_START_FEATURES


def test_week_start_data_extraction():
    """Verify that weekly transition data extracts valid rows."""
    db = PostgresManager()
    df = extract_week_start_time_series(db, "THYAO")
    assert not df.empty
    assert len(df) > 100
    for col in WEEK_START_FEATURES:
        assert col in df.columns, f"Missing feature column: {col}"
    assert "target_return_pct" in df.columns
    assert "as_of_friday_date" in df.columns
    assert "target_monday_date" in df.columns


def test_week_start_walk_forward_arena():
    """Verify that weekly walk-forward arena calculates accurate hit rates and loss."""
    db = PostgresManager()
    df = extract_week_start_time_series(db, "THYAO")
    _, summary, ledger = run_weekly_walk_forward_arena(df, n_weeks=5, lookback_weeks=26)

    assert "lightgbm_dir_hits" in summary
    assert "bayesianridge_dir_hits" in summary
    assert len(ledger) == 5
    assert "trade_date" in ledger[0]
    assert "actual_return_pct" in ledger[0]


def test_week_start_live_prediction():
    """Verify that live week start inference produces valid price targets and bounds."""
    db = PostgresManager()
    pred = predict_live_week_start(db, "THYAO", champion_model="BayesianRidge", lookback_weeks=52, crowned_horizon="12m")
    assert pred is not None
    assert pred["symbol"] == "THYAO"
    assert pred["target_price"] > 0
    assert pred["price_low"] <= pred["target_price"] <= pred["price_high"]
    assert pred["stance"] in ("BULLISH", "BEARISH", "CONSOLIDATION")
    assert pred["weekend_carry_cost_bps"] > 0


def test_api_week_start_forecast():
    """Test GET /api/v1/week-start/forecast endpoint."""
    res = client.get("/api/v1/week-start/forecast?symbol=THYAO")
    assert res.status_code == 200
    data = res.json()
    assert data["symbol"] == "THYAO"
    assert "target_price" in data
    assert "expected_return_pct" in data
    assert "backtest_ledger" in data
    assert len(data["backtest_ledger"]) > 0


def test_api_week_start_opportunities():
    """Test GET /api/v1/week-start/opportunities endpoint."""
    res = client.get("/api/v1/week-start/opportunities")
    assert res.status_code == 200
    data = res.json()
    assert data["total_constituents"] >= 25
    assert len(data["top_longs"]) > 0
    assert len(data["opportunities"]) >= 25
    assert "avg_weekend_carry_bps" in data


def test_api_week_start_backtest():
    """Test GET /api/v1/week-start/backtest endpoint."""
    res = client.get("/api/v1/week-start/backtest?symbol=THYAO&limit=10")
    assert res.status_code == 200
    data = res.json()
    assert len(data) <= 10
    assert "ml_is_hit" in data[0]
    assert "weekend_carry_cost_bps" in data[0]
