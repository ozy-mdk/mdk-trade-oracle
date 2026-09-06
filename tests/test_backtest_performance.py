"""Unit tests for backtest performance calculation, reporting, and visualization suite."""

import numpy as np
import pandas as pd
import pytest

from mdk_trading_oracle.backtest.metrics import BacktestMetricsCalculator
from mdk_trading_oracle.backtest.report import BacktestReportGenerator
from mdk_trading_oracle.backtest.types import BacktestSummary, TargetUnit
from mdk_trading_oracle.backtest.visualizer import BacktestVisualizer


@pytest.fixture
def sample_macro_backtest_df() -> pd.DataFrame:
    """Create a realistic sample DataFrame matching Model 1: Macro Day-Start backtest output."""
    np.random.seed(42)
    n = 20
    dates = pd.date_range("2026-03-01", periods=n, freq="B").date

    actual = np.random.normal(loc=15e6, scale=40e6, size=n)
    noise = np.random.normal(loc=0, scale=15e6, size=n)
    predicted = actual + noise  # Positive correlation

    lower_90 = predicted - 35e6
    upper_90 = predicted + 35e6

    is_hit = (actual * predicted > 0) | ((actual == 0) & (predicted == 0))
    is_inside_ci = (actual >= lower_90) & (actual <= upper_90)

    directions = []
    for p in predicted:
        if p >= 30e6:
            directions.append("STRONG_BUY")
        elif p >= 10e6:
            directions.append("BUY")
        elif p >= -10e6:
            directions.append("NEUTRAL")
        elif p >= -30e6:
            directions.append("SELL")
        else:
            directions.append("STRONG_SELL")

    playbooks = np.random.choice(["SQUEEZE_LONG", "MOMENTUM_EXPANSION", "DEFENSE_SUPPORT"], size=n)
    days = [d.weekday() for d in dates]

    return pd.DataFrame(
        {
            "trade_date": dates,
            "day_of_week": days,
            "is_monday": [d == 0 for d in days],
            "actual_open_net_flow_tl": actual,
            "predicted_open_net_flow_tl": predicted,
            "predicted_open_flow_lower_90": lower_90,
            "predicted_open_flow_upper_90": upper_90,
            "predicted_direction": directions,
            "is_direction_hit": is_hit,
            "is_inside_90_ci": is_inside_ci,
            "predicted_playbook": playbooks,
        }
    )


@pytest.fixture
def sample_stock_backtest_df() -> pd.DataFrame:
    """Create a realistic sample DataFrame matching Model 3: Stock Intraday Reaction."""
    np.random.seed(123)
    symbols = ["THYAO", "GARAN", "AKBNK", "EREGL"]
    windows = ["w2", "w3", "w5"]
    rows = []

    for sym in symbols:
        for w in windows:
            for d in range(10):
                act = float(np.random.normal(0.2, 1.5))
                pred = act + float(np.random.normal(0, 0.8))
                low = pred - 1.8
                high = pred + 1.8
                is_hit = (act * pred > 0)
                is_ci = (act >= low) and (act <= high)
                rows.append(
                    {
                        "trade_date": f"2026-03-{d+2:02d}",
                        "symbol": sym,
                        "window_name": w,
                        "actual_return_pct": act,
                        "predicted_return_pct": pred,
                        "predicted_return_lower_90": low,
                        "predicted_return_upper_90": high,
                        "predicted_direction": "RALLY" if pred > 0 else "DECLINE",
                        "is_direction_hit": is_hit,
                        "is_inside_90_ci": is_ci,
                    }
                )

    return pd.DataFrame(rows)


def test_metrics_calculator_macro_summary(sample_macro_backtest_df: pd.DataFrame):
    """Test full metrics calculation for monetary macro flow targets."""
    summary = BacktestMetricsCalculator.calculate_summary(
        df=sample_macro_backtest_df,
        model_name="Macro Day-Start",
        target_name="Opening Net Flow (TL)",
        actual_col="actual_open_net_flow_tl",
        predicted_col="predicted_open_net_flow_tl",
        lower_90_col="predicted_open_flow_lower_90",
        upper_90_col="predicted_open_flow_upper_90",
        direction_col="predicted_direction",
        is_hit_col="is_direction_hit",
        is_inside_ci_col="is_inside_90_ci",
        target_unit=TargetUnit.TL,
        slice_columns=["day_of_week", "predicted_playbook"],
    )

    assert isinstance(summary, BacktestSummary)
    assert summary.total_samples == 20
    assert summary.target_unit == TargetUnit.TL

    # Regression asserts
    assert summary.regression.mae > 0
    assert summary.regression.rmse >= summary.regression.mae
    assert -1.0 <= summary.regression.pearson_r <= 1.0

    # Directional asserts
    assert 0.0 <= summary.directional.hit_rate_pct <= 100.0
    assert summary.directional.hits + summary.directional.misses == 20
    assert len(summary.directional.conviction_hit_rates) > 0

    # Probabilistic asserts
    assert 0.0 <= summary.probabilistic.picp_90_pct <= 100.0
    assert summary.probabilistic.mpiw > 0

    # Slice asserts
    assert "day_of_week" in summary.slices
    assert "predicted_playbook" in summary.slices
    assert len(summary.slices["predicted_playbook"]) > 0


def test_metrics_calculator_stock_return_summary(sample_stock_backtest_df: pd.DataFrame):
    """Test metrics calculation for percentage return targets."""
    summary = BacktestMetricsCalculator.calculate_summary(
        df=sample_stock_backtest_df,
        model_name="Stock Reaction",
        target_name="Return (%)",
        actual_col="actual_return_pct",
        predicted_col="predicted_return_pct",
        lower_90_col="predicted_return_lower_90",
        upper_90_col="predicted_return_upper_90",
        direction_col="predicted_direction",
        is_hit_col="is_direction_hit",
        is_inside_ci_col="is_inside_90_ci",
        target_unit=TargetUnit.PERCENTAGE,
        slice_columns=["window_name", "symbol"],
    )

    assert summary.target_unit == TargetUnit.PERCENTAGE
    assert summary.total_samples == 120
    assert summary.directional.hit_rate_pct > 0
    assert "window_name" in summary.slices
    assert len(summary.slices["window_name"]) == 3  # w2, w3, w5
    assert "symbol" in summary.slices
    assert len(summary.slices["symbol"]) == 4


def test_metrics_calculator_empty_df_raises():
    """Test that empty DataFrame raises ValueError."""
    with pytest.raises(ValueError, match="Cannot calculate backtest metrics on empty DataFrame"):
        BacktestMetricsCalculator.calculate_summary(
            df=pd.DataFrame(),
            model_name="Test Empty",
        )


def test_visualizer_figures_generation(sample_macro_backtest_df: pd.DataFrame, sample_stock_backtest_df: pd.DataFrame):
    """Verify all Plotly figure builders execute without error and return valid figures."""
    summary = BacktestMetricsCalculator.calculate_summary(
        df=sample_macro_backtest_df,
        model_name="Macro Test",
        target_unit=TargetUnit.TL,
        slice_columns=["predicted_playbook"],
    )

    # 1. Track Record
    fig_track = BacktestVisualizer.plot_track_record(
        df=sample_macro_backtest_df,
        summary=summary,
        unit=TargetUnit.TL,
    )
    assert fig_track is not None
    assert len(fig_track.data) >= 3

    # 2. Parity & Residuals
    fig_parity = BacktestVisualizer.plot_parity_and_residuals(
        df=sample_macro_backtest_df,
        summary=summary,
        unit=TargetUnit.TL,
    )
    assert fig_parity is not None
    assert len(fig_parity.data) >= 2

    # 3. Cumulative Performance
    fig_cum = BacktestVisualizer.plot_cumulative_performance(
        df=sample_macro_backtest_df,
        summary=summary,
        unit=TargetUnit.TL,
    )
    assert fig_cum is not None
    assert len(fig_cum.data) == 2

    # 4. Conviction & Calibration
    fig_calib = BacktestVisualizer.plot_conviction_and_calibration(summary=summary)
    assert fig_calib is not None

    # 5. Slice Leaderboard
    fig_lead = BacktestVisualizer.plot_slice_leaderboard(
        slice_metrics=summary.slices["predicted_playbook"],
        unit=TargetUnit.TL,
    )
    assert fig_lead is not None

    # 6. Stock Window Heatmap Matrix
    fig_matrix = BacktestVisualizer.plot_stock_window_matrix(
        df=sample_stock_backtest_df,
        symbol_col="symbol",
        window_col="window_name",
        is_hit_col="is_direction_hit",
    )
    assert fig_matrix is not None
    assert len(fig_matrix.data) == 1

    # 7. Executive HTML Scorecard
    card_html = BacktestVisualizer.format_executive_scorecard_html(summary)
    assert "Macro Test" in card_html
    assert "Direction Hit Rate" in card_html


def test_report_generator(sample_macro_backtest_df: pd.DataFrame, sample_stock_backtest_df: pd.DataFrame):
    """Test markdown summary report generation and multi-model comparison table."""
    summary_macro = BacktestMetricsCalculator.calculate_summary(
        df=sample_macro_backtest_df,
        model_name="Model 1: Day Start",
        target_unit=TargetUnit.TL,
        slice_columns=["predicted_playbook"],
    )

    summary_stock = BacktestMetricsCalculator.calculate_summary(
        df=sample_stock_backtest_df,
        model_name="Model 3: Stock Reaction",
        actual_col="actual_return_pct",
        predicted_col="predicted_return_pct",
        target_unit=TargetUnit.PERCENTAGE,
        slice_columns=["window_name"],
    )

    md = BacktestReportGenerator.generate_markdown_summary(summary_macro)
    assert "## Backtest Performance Summary: Model 1: Day Start" in md
    assert "Sign Hit Rate %" in md
    assert "90% CI Coverage (PICP)" in md

    df_comp = BacktestReportGenerator.compare_models([summary_macro, summary_stock])
    assert len(df_comp) == 2
    assert "Hit Rate (%)" in df_comp.columns
    assert "PICP 90% (%)" in df_comp.columns


def test_loader_duckdb_integration():
    """Test BacktestLoader against real DuckDB Gold backtest tables."""
    from mdk_trading_oracle.backtest.loader import BacktestLoader

    loader = BacktestLoader()

    # Test Model 1 Macro
    df_m1 = loader.load_day_start()
    assert not df_m1.empty
    summary_m1 = loader.summarize_day_start()
    assert summary_m1.total_samples > 0
    assert summary_m1.target_unit == TargetUnit.TL

    # Test Model 2 Sector
    df_m2 = loader.load_sector_day_start(sectors=["Banking", "Transportation"])
    assert not df_m2.empty
    assert set(df_m2["sector"].unique()).issubset({"Banking", "Transportation"})
    summary_m2 = loader.summarize_sector_day_start()
    assert summary_m2.total_samples > 0
    assert "sector" in summary_m2.slices

    # Test Model 3 Stock Reaction
    df_m3 = loader.load_stock_reaction(windows=["w2"])
    assert not df_m3.empty
    summary_m3 = loader.summarize_stock_reaction(windows=["w2"])
    assert summary_m3.total_samples > 0
    assert summary_m3.target_unit == TargetUnit.PERCENTAGE

