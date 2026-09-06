"""Unit tests for Model 3 (Stock Intraday Reaction Forecaster) point-in-time backfill engine."""

from datetime import date
from unittest.mock import MagicMock, patch

import polars as pl

from mdk_trading_oracle.models.stock_reaction.forecaster import StockReactionForecaster
from mdk_trading_oracle.models.stock_reaction.models import StockReactionForecastResult
from mdk_trading_oracle.models.stock_reaction.orchestrator import StockReactionOrchestrator


def test_stock_reaction_backfill_historical_performance_mocked():
    """Verify backfill_historical_performance executes zero-lookahead forecasts and reconciles against Silver."""
    mock_db = MagicMock()
    mock_conn = MagicMock()
    mock_db.get_connection.return_value = mock_conn

    # Silver trade dates available in database
    mock_conn.execute.return_value.fetchall.side_effect = [
        # 1. all_silver_dates
        [
            (date(2026, 3, 2),),
            (date(2026, 3, 3),),
            (date(2026, 3, 4),),
            (date(2026, 3, 5),),
            (date(2026, 3, 6),),
            (date(2026, 3, 9),),
        ],
        # 2. SHOW TABLES
        [("gold_bofa_stock_reaction_w2_performance",)],
        # 3. existing_perf_dates
        [(date(2026, 3, 2),), (date(2026, 3, 3),)],
    ]

    forecaster = StockReactionForecaster(symbol="AKBNK", window="w2", db=mock_db)

    # Mock forecast_next_window and extract_features
    sample_result = StockReactionForecastResult(
        forecast_date=date(2026, 3, 4),
        symbol="AKBNK",
        window_name="first_reaction",
        predicted_return_pct=1.25,
        predicted_return_lower_90=0.10,
        predicted_return_upper_90=2.40,
        predicted_direction="RALLY",
        direction_confidence=0.85,
        predicted_playbook="MOMENTUM_EXPANSION",
        bofa_w1_direction="BUY",
        bofa_w1_net_flow_tl=15_000_000.0,
        bofa_w1_volume_share=0.22,
        model_name="StockReactionLightGBMModel",
        model_version="1.0.0",
    )

    with patch.object(forecaster, "forecast_next_window", return_value=sample_result) as mock_fc, \
         patch.object(forecaster.extractor, "extract_features") as mock_ef:
        # Mock actual return in Silver for target date
        mock_ef.return_value = pl.DataFrame({
            "trade_date": [date(2026, 3, 4)],
            "target_w2_return_pct": [1.10],
        })

        # Run backfill for specific date
        count = forecaster.backfill_historical_performance(target_dates=["2026-03-04"])

        assert count == 1
        # Verify forecast_next_window was called with strict zero-lookahead as_of_date=2026-03-03
        mock_fc.assert_called_once_with(
            forecast_date=date(2026, 3, 4),
            as_of_date=date(2026, 3, 3),
            replace_active=False,
        )


def test_stock_reaction_orchestrator_backfill_historical_performance():
    """Verify orchestrator coordinates backfilling across multiple symbols and windows."""
    mock_db = MagicMock()
    orchestrator = StockReactionOrchestrator(
        db=mock_db,
        symbols=["AKBNK", "GARAN"],
        windows=["w2"],
    )

    with patch.object(StockReactionForecaster, "backfill_historical_performance", return_value=3) as mock_bf:
        res = orchestrator.backfill_historical_performance(
            all_missing=True,
            lookback_days=30,
        )

        assert res["total_backfilled"] == 6  # 2 symbols * 1 window * 3 rows
        assert res["symbols_count"] == 2
        assert res["windows_count"] == 1
        assert res["results"]["AKBNK"]["w2"] == 3
        assert res["results"]["GARAN"]["w2"] == 3
        assert mock_bf.call_count == 2
