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

    data_dict = {
        "trade_date": dates,
        "close_price": np.cumsum(np.random.randn(65)) + 100.0,
        "daily_return_pct": np.random.randn(65) * 0.02,
        "total_turnover_tl": np.random.uniform(1e8, 5e8, 65),
        "fifo_avg_cost": np.ones(65) * 98.0,
        "feat_cost_spread_pct": np.random.randn(65) * 2.0,
        "feat_ret_today_pct": np.random.randn(65) * 1.5,
        "feat_ret_yesterday_pct": np.random.randn(65) * 1.5,
        "feat_mlb_w5_share": np.random.randn(65) * 0.03,
        "feat_prophet_ret_today_pct": np.random.randn(65) * 1.5,
        "bist30_return_pct": np.random.randn(65) * 1.5,
        "feat_bist30_ret_today_pct": np.random.randn(65) * 1.5,
        "feat_mlb_tertip_3m_ratio": np.random.randn(65) * 5.0,
        "feat_big5_tertip_3m_ratio": np.random.randn(65) * 4.0,
        "feat_kamu_tertip_3m_ratio": np.random.randn(65) * 3.0,
        "mlb_flow": np.random.randn(65) * 1e7,
        "big5_flow": np.random.randn(65) * 2e7,
        "kamu_flow": np.random.randn(65) * 5e6,
        "mlb_buy_tl": np.random.uniform(1e7, 3e7, 65),
        "mlb_sell_tl": np.random.uniform(1e7, 3e7, 65),
        "mlb_daily_pnl_tl": np.random.randn(65) * 5e5,
        "big5_buy_tl": np.random.uniform(2e7, 5e7, 65),
        "big5_sell_tl": np.random.uniform(2e7, 5e7, 65),
        "big5_daily_pnl_tl": np.random.randn(65) * 1e6,
        "kamu_buy_tl": np.random.uniform(1e7, 4e7, 65),
        "kamu_sell_tl": np.random.uniform(1e7, 4e7, 65),
        "kamu_daily_pnl_tl": np.random.randn(65) * 8e5,
        "feat_days_since_pos_shock": np.random.uniform(0, 63, 65),
        "feat_days_since_neg_shock": np.random.uniform(0, 63, 65),
        "feat_total_inst_net_share_today": np.random.randn(65) * 5.0,
    }

    # Add 9 today's execution share features
    for p in ["mlb", "big5", "kamu"]:
        data_dict[f"feat_{p}_buy_share_today"] = np.random.randn(65) * 5.0
        data_dict[f"feat_{p}_sell_share_today"] = np.random.randn(65) * 5.0
        data_dict[f"feat_{p}_pnl_share_today"] = np.random.randn(65) * 1.0

    df = pd.DataFrame(data_dict)

    ledger, summary, model = run_30d_walk_forward_arena(df, n_sessions=10)

    assert len(ledger) == 10
    assert "champion" in summary
    assert "ml_champion_type" in summary
    assert summary["ml_champion_type"] in ("XGBoost", "Ridge", "LightGBM", "Huber", "BayesianRidge", "GaussianProcess", "TFT")
    assert "champion_dir_hits" in summary
    assert "champion_dir_hit_rate_pct" in summary
    assert "ml_dir_hits" in summary
    assert "prophet_dir_hits" in summary
    assert "ml_error_wins" in summary
    assert "prophet_error_wins" in summary
    assert "ridge_hit_rate_pct" in summary
    assert "xgboost_hit_rate_pct" in summary
    assert "lightgbm_hit_rate_pct" in summary
    assert "huber_hit_rate_pct" in summary
    assert "bayesian_ridge_hit_rate_pct" in summary
    assert "ml_hit_rate_pct" in summary
    assert "prophet_hit_rate_pct" in summary
    assert "ml_mae_pct" in summary
    assert "champion_30d_penalty_loss" in summary
    assert "ml_penalty_loss_30d" in summary
    assert "prophet_penalty_loss_30d" in summary
    assert "ridge_penalty_loss_30d" in summary
    assert "xgboost_penalty_loss_30d" in summary
    assert "lightgbm_penalty_loss_30d" in summary
    assert "huber_penalty_loss_30d" in summary
    assert "bayesian_ridge_penalty_loss_30d" in summary
    assert summary["champion_30d_penalty_loss"] >= 0.0
    assert summary["total_sessions"] == 10
    assert model is not None

    # Check ledger columns
    row = ledger[0]
    assert "date" in row
    assert "actual_price" in row
    assert "bist30_ret_pct" in row
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
    assert "bist30_trend" in data
    assert "today_pct" in data["bist30_trend"]
    assert "yesterday_pct" in data["bist30_trend"]
    assert "day_before_pct" in data["bist30_trend"]
    assert "tournament_summary" in data
    assert len(data["pillar_matrix"]) == 3
    assert len(data["walk_forward_ledger"]) >= 30

    # Verify 39 active features including 3-pillar cost spreads, tertip shares, intensities, and shock defense
    assert "active_features" in data
    assert data["active_features_count"] == 39
    assert "feat_mlb_cost_spread_pct" in data["active_features"]
    assert "feat_big5_cost_spread_pct" in data["active_features"]
    assert "feat_kamu_cost_spread_pct" in data["active_features"]
    assert "feat_total_inst_net_share_today" in data["active_features"]
    assert "feat_bofa_aggression_ratio" in data["active_features"]
    assert "feat_tertip_squeeze_delta" in data["active_features"]
    assert "feat_volatility_pinch_5d_20d" in data["active_features"]
    assert "feat_tertip_inventory_zscore" in data["active_features"]
    assert "feat_days_since_pos_shock" in data["active_features"]
    assert "feat_days_since_neg_shock" in data["active_features"]
    assert "feat_kamu_post_shock_defense" in data["active_features"]
    assert "feat_ret_3d_cum_pct" in data["active_features"]
    assert "feat_ret_acceleration_pct" in data["active_features"]
    assert "feat_intraday_range_pct" in data["active_features"]
    assert "feat_mlb_turnover_intensity" in data["active_features"]
    assert "feat_big5_turnover_intensity" in data["active_features"]
    assert "feat_kamu_turnover_intensity" in data["active_features"]
    assert "feat_bollinger_bandwidth_pct" in data["active_features"]
    assert "feat_mlb_tertip_share" in data["active_features"]
    assert "feat_big5_tertip_share" in data["active_features"]
    assert "feat_kamu_tertip_share" in data["active_features"]
    assert "feat_mlb_profitable_tertip_share" in data["active_features"]
    assert "feat_big5_profitable_tertip_share" in data["active_features"]
    assert "feat_kamu_profitable_tertip_share" in data["active_features"]

    pillars = {p["pillar"] for p in data["pillar_matrix"]}
    assert pillars == {"MLB", "BIG5", "KAMU"}

    # Verify first and last rows of reality ledger
    last_row = data["walk_forward_ledger"][-1]
    assert "date" in last_row
    assert "actual_price" in last_row
    assert "bist30_ret_pct" in last_row
    assert "mlb_action" in last_row
    assert "big5_action" in last_row
    assert "kamu_action" in last_row
    assert "mlb_buy_tl" in last_row
    assert "mlb_sell_tl" in last_row
    assert "mlb_pnl_tl" in last_row

    # Verify pillar_execution (Today vs Yesterday)
    assert "pillar_execution" in data
    pexec = data["pillar_execution"]
    assert "today_date" in pexec
    assert "yesterday_date" in pexec
    assert "mlb" in pexec and "today" in pexec["mlb"] and "yesterday" in pexec["mlb"]
    assert "big5" in pexec and "today" in pexec["big5"] and "yesterday" in pexec["big5"]
    assert "kamu" in pexec and "today" in pexec["kamu"] and "yesterday" in pexec["kamu"]
