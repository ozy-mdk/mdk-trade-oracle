"""Tests for Model 1: Day-Start Institutional Forecaster."""

import pandas as pd
import pytest

from mdk_trading_oracle.core.db import DuckDBManager
from mdk_trading_oracle.data.bronze import initialize_bronze_schema
from mdk_trading_oracle.data.gold import initialize_gold_schema
from mdk_trading_oracle.data.silver import SilverTransformer, initialize_silver_schema
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

    # Insert trades across 4 distinct days (Day 1 to Day 4)
    conn.execute("""
        INSERT INTO bronze_raw_trades (trade_id, timestamp, symbol, price, volume, buyer_broker_id, seller_broker_id, raw_source)
        VALUES 
            -- Day 1 (Monday)
            ('t1', '2026-03-02 10:00:00', 'THYAO', 300.0, 1000.0, 'MLB', 'ISY', 'test'),
            ('t2', '2026-03-02 17:30:00', 'THYAO', 305.0, 2000.0, 'MLB', 'GAR', 'test'),
            -- Day 2 (Tuesday)
            ('t3', '2026-03-03 10:00:00', 'THYAO', 306.0, 1500.0, 'MLB', 'ISY', 'test'),
            ('t4', '2026-03-03 17:30:00', 'AKBNK', 60.0, 5000.0, 'GAR', 'MLB', 'test'),
            -- Day 3 (Wednesday)
            ('t5', '2026-03-04 10:00:00', 'AKBNK', 61.0, 4000.0, 'MLB', 'YKR', 'test'),
            ('t6', '2026-03-04 17:30:00', 'THYAO', 310.0, 3000.0, 'MLB', 'AKB', 'test'),
            -- Day 4 (Thursday)
            ('t7', '2026-03-05 10:00:00', 'THYAO', 312.0, 2500.0, 'MLB', 'ISY', 'test'),
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
    assert res0.predicted_direction in ["ACCUMULATE", "DISTRIBUTE", "NEUTRAL", "STRONG_ACCUMULATE", "STRONG_DISTRIBUTE"]

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


def test_day_start_model_arena_rejects_constant_target():
    """Fail loudly when a broken window configuration produces a constant target."""
    X = pd.DataFrame({"feat_bofa_w4_net_flow_tl": [1.0, 2.0, 3.0, 4.0]})
    y = pd.Series([0.0, 0.0, 0.0, 0.0], name="target_open_net_flow_tl")

    with pytest.raises(ValueError, match="at least two distinct values"):
        DayStartModelArena().run_tournament(X, y, min_train_samples=2)


def test_day_start_forecaster_auto_orchestration(populated_test_db):
    """Test DayStartForecaster in 'auto' mode end-to-end training and DuckDB Gold table persistence."""
    forecaster = DayStartForecaster(populated_test_db, model_type="auto")
    forecasts = forecaster.train_and_forecast_all()
    assert len(forecasts) > 0
    assert forecaster.champion_name is not None

    saved_count = forecaster.save_forecasts_to_gold(forecasts)
    assert saved_count == len(forecasts)

    conn = populated_test_db.get_connection()
    gold_row = conn.execute("""
        SELECT forecast_date, predicted_open_net_flow_tl, predicted_direction, direction_confidence, predicted_playbook, model_name
        FROM gold_bofa_day_start_forecasts
        LIMIT 1;
    """).fetchone()

    assert gold_row is not None
    assert gold_row[1] is not None
    assert gold_row[2] is not None
    assert gold_row[3] is not None
    assert gold_row[4] is not None
    assert gold_row[5] is not None
