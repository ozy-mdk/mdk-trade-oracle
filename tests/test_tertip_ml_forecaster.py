"""Unit tests for Tertip 3-Pillar ML Forecaster & Tournament Arena."""

import numpy as np
import pandas as pd
import pytest

from mdk_trading_oracle.core.db import PostgresManager
from mdk_trading_oracle.data.silver.tertip_ml_forecaster import (
    get_tertip_ml_forecast,
    run_30d_walk_forward_arena,
)


def test_run_30d_walk_forward_arena_synthetic():
    """Test walk-forward engine logic on synthetic DataFrame."""
    dates = pd.date_range("2026-01-01", periods=65)
    np.random.seed(42)

    df = pd.DataFrame({
        "trade_date": dates,
        "close_price": np.cumsum(np.random.randn(65)) + 100.0,
        "daily_return_pct": np.random.randn(65) * 0.02,
        "total_turnover_tl": np.random.uniform(1e8, 5e8, 65),
        "fifo_avg_cost": np.ones(65) * 98.0,
        "feat_cost_spread_pct": np.random.randn(65) * 2.0,
        "feat_mlb_flow_share": np.random.randn(65) * 0.1,
        "feat_big5_flow_share": np.random.randn(65) * 0.15,
        "feat_kamu_flow_share": np.random.randn(65) * 0.05,
        "feat_mlb_w5_share": np.random.randn(65) * 0.03,
        "feat_ret_1d_pct": np.random.randn(65) * 1.5,
        "mlb_flow": np.random.randn(65) * 1e7,
        "big5_flow": np.random.randn(65) * 2e7,
        "kamu_flow": np.random.randn(65) * 5e6,
    })

    ledger, summary, model = run_30d_walk_forward_arena(df, n_sessions=10)

    assert len(ledger) == 10
    assert "champion" in summary
    assert "ml_hit_rate_pct" in summary
    assert "prophet_hit_rate_pct" in summary
    assert "ml_mae_pct" in summary
    assert summary["total_sessions"] == 10
    assert model is not None

    # Check ledger columns
    row = ledger[0]
    assert "date" in row
    assert "actual_price" in row
    assert "ml_pred_price" in row
    assert "prophet_pred_price" in row
    assert "mlb_action" in row
    assert "big5_action" in row
    assert "kamu_action" in row
    assert "winner" in row
    assert row["winner"] in ("CHALLENGER", "BASE")


@pytest.mark.integration
def test_tertip_ml_forecaster_live_database():
    """Verify live database extraction and ML forecasting on ASELS."""
    db = PostgresManager()
    data = get_tertip_ml_forecast(db, "ASELS")

    assert "symbol" in data
    assert data["symbol"] == "ASELS"
    assert "target_price" in data
    assert data["target_price"] > 0
    assert "expected_return_pct" in data
    assert "stance" in data
    assert "tournament_summary" in data
    assert len(data["pillar_matrix"]) == 3
    assert len(data["walk_forward_ledger"]) == 30

    pillars = {p["pillar"] for p in data["pillar_matrix"]}
    assert pillars == {"MLB", "BIG5", "KAMU"}

    # Verify first and last rows of reality ledger
    last_row = data["walk_forward_ledger"][-1]
    assert "date" in last_row
    assert "actual_price" in last_row
    assert "mlb_action" in last_row
    assert "big5_action" in last_row
    assert "kamu_action" in last_row
