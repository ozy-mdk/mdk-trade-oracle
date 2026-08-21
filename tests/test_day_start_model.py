"""Tests for Model 1: Day-Start Institutional Forecaster."""

import pytest

from mdk_trading_oracle.core.db import DuckDBManager
from mdk_trading_oracle.data.bronze import initialize_bronze_schema
from mdk_trading_oracle.data.gold import initialize_gold_schema
from mdk_trading_oracle.data.silver import SilverTransformer, initialize_silver_schema
from mdk_trading_oracle.models.base import ForecastDirection
from mdk_trading_oracle.models.day_start import (
    DayStartBayesianModel,
    DayStartFeatureExtractor,
    DayStartForecaster,
    DayStartLightGBMModel,
    DayStartModelArena,
    DayStartNaivePersistenceModel,
    DayStartPyMCModel,
    DayStartRollingMeanModel,
)


@pytest.fixture
def populated_test_db():
    """Create in-memory DuckDB populated with synthetic multi-day trading data."""
    db = DuckDBManager(in_memory=True)
    conn = db.get_connection()

    initialize_bronze_schema(db)
    initialize_silver_schema(db)
    initialize_gold_schema(db)

    # Insert trades across 4 distinct days (Day 1 to Day 4) in Turkish Time (TRT)
    conn.execute("""
        INSERT INTO bronze_raw_trades (trade_id, timestamp, symbol, price, volume, buyer_broker_id, seller_broker_id, raw_source)
        VALUES 
            -- Day 1 (Monday)
            ('t1', '2026-03-02 10:15:00', 'THYAO', 300.0, 1000.0, 'MLB', 'ISY', 'test'),
            ('t2', '2026-03-02 17:30:00', 'THYAO', 305.0, 2000.0, 'MLB', 'GAR', 'test'),
            -- Day 2 (Tuesday)
            ('t3', '2026-03-03 10:15:00', 'THYAO', 306.0, 1500.0, 'MLB', 'ISY', 'test'),
            ('t4', '2026-03-03 17:30:00', 'AKBNK', 60.0, 5000.0, 'GAR', 'MLB', 'test'),
            -- Day 3 (Wednesday)
            ('t5', '2026-03-04 10:15:00', 'AKBNK', 61.0, 4000.0, 'MLB', 'YKR', 'test'),
            ('t6', '2026-03-04 17:30:00', 'THYAO', 310.0, 3000.0, 'MLB', 'AKB', 'test'),
            -- Day 4 (Thursday)
            ('t7', '2026-03-05 10:15:00', 'THYAO', 312.0, 2500.0, 'MLB', 'ISY', 'test'),
            ('t8', '2026-03-05 17:30:00', 'AKBNK', 62.0, 6000.0, 'MLB', 'GAR', 'test');
    """)

    silver = SilverTransformer(db)
    silver.run_all()
    return db


def test_day_start_feature_extraction(populated_test_db):
    """Test DayStartFeatureExtractor computes all 7 Feature Clusters with no data leakage."""
    extractor = DayStartFeatureExtractor(populated_test_db, target_broker_id="MLB")
    df = extractor.extract_features()

    assert df.height > 0
    # Verify key cluster features exist
    expected_cols = [
        "trade_date", "is_monday", "is_friday",
        "feat_bofa_w4_net_flow_tl", "feat_bofa_prev_day_net_flow_tl",
        "feat_bofa_vs_top5_w4_flow_delta_tl", "feat_institutional_hegemony_share",
        "feat_bofa_cum_net_flow_5d_tl", "target_open_net_flow_tl"
    ]
    for col in expected_cols:
        assert col in df.columns, f"Missing feature column: {col}"


def test_day_start_candidate_models(populated_test_db):
    """Test candidate models (Baselines, Bayesian, LightGBM) fit and predict."""
    extractor = DayStartFeatureExtractor(populated_test_db, target_broker_id="MLB")
    df_pl = extractor.extract_features()
    df_pd = df_pl.to_pandas()

    X = df_pd.drop(columns=["target_open_net_flow_tl", "target_open_direction"], errors="ignore")
    y = df_pd["target_open_net_flow_tl"]

    # 1. Baseline Persistence
    m_base0 = DayStartNaivePersistenceModel()
    m_base0.fit(X, y)
    res0 = m_base0.predict(X.iloc[[0]])
    assert res0.predicted_direction in ForecastDirection.all_valid()

    # 2. Baseline Rolling Mean
    m_base1 = DayStartRollingMeanModel()
    m_base1.fit(X, y)
    res1 = m_base1.predict(X.iloc[[0]])
    assert res1.predicted_net_flow_tl is not None

    # 3. Bayesian Model with Credible Intervals
    m_bayes = DayStartBayesianModel()
    m_bayes.fit(X, y)
    res_bayes = m_bayes.predict(X.iloc[[0]])
    assert res_bayes.predicted_flow_lower_90 < res_bayes.predicted_flow_upper_90
    assert 0.0 <= res_bayes.direction_confidence <= 1.0

    # 4. LightGBM Model
    m_lgb = DayStartLightGBMModel()
    m_lgb.fit(X, y)
    res_lgb = m_lgb.predict(X.iloc[[0]])
    assert res_lgb.predicted_net_flow_tl is not None

    # 5. PyMC Full Bayesian Model (MAP)
    m_pymc = DayStartPyMCModel(use_map=True)
    m_pymc.fit(X, y)
    res_pymc = m_pymc.predict(X.iloc[[0]])
    assert res_pymc.predicted_flow_lower_90 < res_pymc.predicted_flow_upper_90
    assert 0.0 <= res_pymc.direction_confidence <= 1.0

    # 6. Walk-Forward Expanding Window Evaluation
    wf_metrics = m_bayes.walk_forward_evaluate(X, y, min_train_samples=2)
    assert "hit_rate_pct" in wf_metrics
    assert "picp_90_pct" in wf_metrics
    assert len(wf_metrics["oos_predictions"]) > 0


def test_day_start_model_arena(populated_test_db):
    """Test DayStartModelArena tournament and champion selection."""
    extractor = DayStartFeatureExtractor(populated_test_db, target_broker_id="MLB")
    df_pl = extractor.extract_features()
    df_pd = df_pl.to_pandas()

    X = df_pd.drop(columns=["target_open_net_flow_tl", "target_open_direction"], errors="ignore")
    y = df_pd["target_open_net_flow_tl"]

    arena = DayStartModelArena()
    scoreboard_df, champion_model = arena.run_tournament(X, y, min_train_samples=2)

    assert len(scoreboard_df) == 5
    assert champion_model is not None
    assert champion_model.model_name in ["day_start_bayesian_ridge", "day_start_pymc", "day_start_lightgbm", "day_start_baseline_persistence", "day_start_baseline_rolling_mean"]


def test_day_start_next_day_feature_extraction(populated_test_db):
    """Test DayStartFeatureExtractor computes next-day features for tomorrow morning."""
    extractor = DayStartFeatureExtractor(populated_test_db, target_broker_id="MLB")
    df_next = extractor.extract_next_day_features()

    assert df_next.height == 1
    assert "trade_date" in df_next.columns
    assert "feat_bofa_w4_net_flow_tl" in df_next.columns
    assert "feat_bofa_cum_net_flow_5d_tl" in df_next.columns
    # Trade date for next day should be after the last bronze trade date (2026-03-05 -> 2026-03-06 Friday)
    assert str(df_next["trade_date"][0])[:10] == "2026-03-06"


def test_day_start_forecaster_auto_orchestration(populated_test_db):
    """Test DayStartForecaster in 'auto' mode end-to-end live next-day forecast and backtesting."""
    forecaster = DayStartForecaster(populated_test_db, model_type="auto")
    
    # 1. Live Next-Day Forecast
    next_res = forecaster.forecast_next_day()
    assert next_res is not None
    assert str(next_res.forecast_date)[:10] == "2026-03-06"
    assert next_res.predicted_direction in ForecastDirection.all_valid()
    assert forecaster.champion_name is not None

    # 2. Historical Backtest Track Record
    backtest_res = forecaster.backtest_all_history()
    assert len(backtest_res) > 0

    # 3. Default train_and_forecast_all (returns next day)
    forecasts = forecaster.train_and_forecast_all(include_history=False, include_next_day=True)
    assert len(forecasts) == 1
    assert str(forecasts[0].forecast_date)[:10] == "2026-03-06"

    saved_count = forecaster.save_forecasts_to_gold(forecasts, replace_active=True)
    assert saved_count == 1  # Strictly 1 active upcoming forecast!

    # 4. Performance Ledger Reconciliation
    saved_perf = forecaster.reconcile_and_update_performance_ledger(backtest_res)
    assert saved_perf >= 1

    # 5. Point-in-Time Historical Backfill (zero lookahead)
    backfilled_count = forecaster.backfill_historical_performance(target_dates=["2026-03-04", "2026-03-05"])
    assert backfilled_count >= 1

    # 6. Persist historical backtests to dedicated table
    saved_backtests = forecaster.save_backtests_to_gold(backtest_res)
    assert saved_backtests >= 1

    conn = populated_test_db.get_connection()
    gold_rows = conn.execute("""
        SELECT forecast_date, predicted_open_net_flow_tl, predicted_direction, direction_confidence, predicted_playbook, model_name
        FROM gold_bofa_day_start_forecasts;
    """).fetchall()

    # Verify strictly 1 active forecast row in forecast table
    assert len(gold_rows) == 1
    assert str(gold_rows[0][0])[:10] == "2026-03-06"
    assert gold_rows[0][1] is not None
    assert gold_rows[0][2] is not None

    # Verify performance ledger table has evaluated records with actuals and errors
    perf_row = conn.execute("""
        SELECT trade_date, predicted_open_net_flow_tl, actual_open_net_flow_tl, error_open_net_flow_tl, absolute_error_tl, is_direction_hit, is_inside_90_ci, model_name
        FROM gold_bofa_day_start_performance
        ORDER BY trade_date DESC
        LIMIT 1;
    """).fetchone()

    assert perf_row is not None
    assert perf_row[0] is not None
    assert perf_row[1] is not None
    assert perf_row[2] is not None
    assert perf_row[4] >= 0.0  # absolute_error_tl >= 0
    assert perf_row[5] in [True, False]
    assert perf_row[6] in [True, False]

    backtest_row = conn.execute("""
        SELECT trade_date, predicted_open_net_flow_tl, actual_open_net_flow_tl, is_direction_hit, is_inside_90_ci, model_name
        FROM gold_bofa_day_start_backtests
        ORDER BY trade_date DESC
        LIMIT 1;
    """).fetchone()

    assert backtest_row is not None
    assert backtest_row[0] is not None
    assert backtest_row[1] is not None
    assert backtest_row[3] in [True, False]



