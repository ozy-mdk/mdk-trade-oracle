"""Unit and integration tests for Model 2: Sector Day-Start Forecaster & Arena."""

import pandas as pd
import pytest

from mdk_trading_oracle.core.db import DuckDBManager
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
)


@pytest.fixture
def db_conn():
    """Provides a read-only DuckDB connection manager for testing."""
    return DuckDBManager(read_only=True)


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
    """Verify fitting and forecasting across all 5 candidate sector models."""
    extractor = SectorDayStartFeatureExtractor(db_conn, target_broker_id="MLB")
    df_pl = extractor.extract_features(sector="Banking")
    df = df_pl.to_pandas()

    X = df.drop(columns=["target_sector_open_net_flow_tl", "target_sector_open_direction"], errors="ignore")
    y = df["target_sector_open_net_flow_tl"]

    models = [
        SectorDayStartNaivePersistenceModel(),
        SectorDayStartRollingMeanModel(),
        SectorDayStartLightGBMModel(),
        SectorDayStartBayesianModel(),
        SectorDayStartPyMCModel(use_map=True),
    ]

    for model in models:
        model.fit(X, y)
        assert model.is_fitted

        res: ForecastResult = model.predict(X.iloc[[0]])
        assert isinstance(res, ForecastResult)
        assert res.predicted_direction in [
            ForecastDirection.STRONG_ACCUMULATE,
            ForecastDirection.ACCUMULATE,
            ForecastDirection.NEUTRAL,
            ForecastDirection.DISTRIBUTE,
            ForecastDirection.STRONG_DISTRIBUTE,
        ]
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
    assert len(scoreboard) == 4
    assert "hit_rate_pct" in scoreboard.columns
    assert "picp_90_pct" in scoreboard.columns
    assert champion is not None
    assert champion.model_name in [model.model_name for model in arena.candidates.values()]


def test_sector_day_start_forecaster_orchestration(db_conn):
    """Verify SectorDayStartForecaster end-to-end training and forecast generation."""
    forecaster = SectorDayStartForecaster(db_conn, model_type="auto")
    forecasts = forecaster.train_and_forecast_all(sectors=["Banking", "Transportation"])
    assert len(forecasts) > 0

    for f in forecasts:
        assert isinstance(f, ForecastResult)
        assert f.top_predicted_buy_sector in ["Banking", "Transportation"]
