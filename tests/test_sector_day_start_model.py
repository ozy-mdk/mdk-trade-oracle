"""Unit and integration tests for Model 2: Sector Day-Start Forecaster & Arena."""

import pandas as pd
import pytest

from mdk_trading_oracle.core.db import DuckDBManager
from mdk_trading_oracle.data.bronze import initialize_bronze_schema
from mdk_trading_oracle.data.gold import initialize_gold_schema
from mdk_trading_oracle.data.silver import SilverTransformer, initialize_silver_schema
from mdk_trading_oracle.models.base import ForecastDirection, ForecastResult
from mdk_trading_oracle.models.sector_day_start import (
    SectorDayStartBayesianModel,
    SectorDayStartFeatureExtractor,
    SectorDayStartForecaster,
    SectorDayStartLightGBMModel,
    SectorDayStartModelArena,
    SectorDayStartNaivePersistenceModel,
    SectorDayStartPyMCModel,
    SectorDayStartRollingMeanModel,
    SectorDayStartXGBoostModel,
)


@pytest.fixture
def db_conn():
    """Provides a read-only DuckDB connection manager for testing."""
    return DuckDBManager(read_only=True)


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


def test_sector_day_start_feature_extraction(db_conn):
    """Verify that SectorDayStartFeatureExtractor extracts valid lagged sector features."""
    extractor = SectorDayStartFeatureExtractor(db_conn, target_broker_id="MLB")
    tracked_sectors = extractor.get_tracked_sectors(min_session_count=5)
    assert len(tracked_sectors) > 0
    assert "Banking" in tracked_sectors or "Transportation" in tracked_sectors

    df = extractor.extract_features(sector="Banking")
    assert df.height > 0
    assert "feat_sector_bofa_w4_net_flow_tl" in df.columns
    assert "feat_sector_bofa_vs_top5_w4_delta_tl" in df.columns
    assert "feat_sector_bofa_cum_net_flow_5d_tl" in df.columns
    assert "target_sector_open_net_flow_tl" in df.columns
    assert "target_sector_open_direction" in df.columns


def test_sector_day_start_candidate_models(db_conn):
    """Verify fitting and forecasting across all 6 candidate sector models."""
    extractor = SectorDayStartFeatureExtractor(db_conn, target_broker_id="MLB")
    df_pl = extractor.extract_features(sector="Banking")
    df = df_pl.to_pandas()

    X = df.drop(columns=["target_sector_open_net_flow_tl", "target_sector_open_direction"], errors="ignore")
    y = df["target_sector_open_net_flow_tl"]

    models = [
        SectorDayStartNaivePersistenceModel(),
        SectorDayStartRollingMeanModel(),
        SectorDayStartLightGBMModel(),
        SectorDayStartXGBoostModel(),
        SectorDayStartBayesianModel(),
        SectorDayStartPyMCModel(use_map=True),
    ]

    for model in models:
        model.fit(X, y)
        assert model.is_fitted

        res: ForecastResult = model.predict(X.iloc[[0]])
        assert isinstance(res, ForecastResult)
        assert res.predicted_direction in ForecastDirection.all_valid()
        assert res.predicted_flow_lower_90 <= res.predicted_flow_upper_90
        assert 0.0 <= res.direction_confidence <= 1.0


def test_sector_day_start_model_arena(db_conn):
    """Verify that SectorDayStartModelArena runs walk-forward validation and crowns a champion."""
    extractor = SectorDayStartFeatureExtractor(db_conn, target_broker_id="MLB")
    df_pl = extractor.extract_features(sector="Banking")
    df = df_pl.to_pandas()

    X = df.drop(columns=["target_sector_open_net_flow_tl", "target_sector_open_direction"], errors="ignore")
    y = df["target_sector_open_net_flow_tl"]

    arena = SectorDayStartModelArena(include_pymc=False)
    scoreboard, champion = arena.run_tournament(X, y, min_train_samples=5)

    assert isinstance(scoreboard, pd.DataFrame)
    assert len(scoreboard) == 5
    assert "hit_rate_pct" in scoreboard.columns
    assert "picp_90_pct" in scoreboard.columns
    assert champion is not None
    assert champion.model_name in [model.model_name for model in arena.candidates.values()]


def test_sector_day_start_next_day_feature_extraction(db_conn):
    """Verify next-day sector feature extraction."""
    extractor = SectorDayStartFeatureExtractor(db_conn, target_broker_id="MLB")
    df_next = extractor.extract_next_day_features(sectors=["Banking", "Transportation"])
    assert df_next.height == 2
    assert "trade_date" in df_next.columns
    assert "sector" in df_next.columns
    assert "feat_sector_bofa_w4_net_flow_tl" in df_next.columns
    assert "feat_sector_bofa_cum_net_flow_5d_tl" in df_next.columns


def test_sector_day_start_forecaster_orchestration(populated_test_db):
    """Verify SectorDayStartForecaster end-to-end live next-day forecast and backtesting."""
    forecaster = SectorDayStartForecaster(populated_test_db, model_type="auto")

    # 1. Live Next-Day Forecast across sectors
    live_forecasts = forecaster.forecast_next_day(sectors=["Banking", "Transportation"])
    assert len(live_forecasts) == 2

    for f in live_forecasts:
        assert isinstance(f, ForecastResult)
        assert f.top_predicted_buy_sector in ["Banking", "Transportation"]
        assert f.predicted_direction in ForecastDirection.all_valid()

    # 2. Historical Backtest Track Record
    backtest_forecasts = forecaster.backtest_all_history(sectors=["Banking", "Transportation"])
    assert len(backtest_forecasts) > 0

    # 3. Default train_and_forecast_all
    forecasts = forecaster.train_and_forecast_all(
        sectors=["Banking", "Transportation"], include_history=False, include_next_day=True
    )
    assert len(forecasts) == 2

    saved_forecasts = forecaster.save_forecasts_to_gold(forecasts, replace_active=True)
    assert saved_forecasts == 2  # Strictly 2 active sector forecasts for tomorrow!

    # 4. Sector Performance Ledger Reconciliation
    saved_perf = forecaster.reconcile_and_update_performance_ledger(
        backtest_forecasts, sectors=["Banking", "Transportation"]
    )
    assert saved_perf >= 2

    # 5. Point-in-Time Historical Sector Backfill
    backfilled_count = forecaster.backfill_historical_performance(
        target_dates=["2026-03-04", "2026-03-05"], sectors=["Banking", "Transportation"]
    )
    assert backfilled_count >= 2

    # 6. Persist historical sector backtests
    saved_backtests = forecaster.save_backtests_to_gold(backtest_forecasts, sectors=["Banking", "Transportation"])
    assert saved_backtests >= 2

    conn = populated_test_db.get_connection()
    gold_rows = conn.execute("""
        SELECT forecast_date, sector, predicted_open_net_flow_tl, predicted_direction, direction_confidence, model_name
        FROM gold_bofa_sector_day_start_forecasts;
    """).fetchall()

    # Verify strictly 2 active forecast rows in forecast table
    assert len(gold_rows) == 2
    assert str(gold_rows[0][0])[:10] == "2026-03-06"
    assert gold_rows[0][1] in ["Banking", "Transportation"]

    # Verify sector performance ledger table has evaluated records with actuals and errors
    perf_row = conn.execute("""
        SELECT trade_date, sector, predicted_open_net_flow_tl, actual_open_net_flow_tl, error_open_net_flow_tl, absolute_error_tl, is_direction_hit, is_inside_90_ci, model_name
        FROM gold_bofa_sector_day_start_performance
        ORDER BY trade_date DESC
        LIMIT 1;
    """).fetchone()

    assert perf_row is not None
    assert perf_row[0] is not None
    assert perf_row[1] in ["Banking", "Transportation"]
    assert perf_row[2] is not None
    assert perf_row[3] is not None
    assert perf_row[5] >= 0.0  # absolute_error_tl >= 0
    assert perf_row[6] in [True, False]
    assert perf_row[7] in [True, False]

    backtest_row = conn.execute("""
        SELECT trade_date, sector, predicted_open_net_flow_tl, actual_open_net_flow_tl, is_direction_hit, is_inside_90_ci, model_name
        FROM gold_bofa_sector_day_start_backtests
        ORDER BY trade_date DESC
        LIMIT 1;
    """).fetchone()

    assert backtest_row is not None
    assert backtest_row[1] in ["Banking", "Transportation"]
    assert backtest_row[4] in [True, False]
