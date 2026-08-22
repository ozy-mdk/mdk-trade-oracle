"""Unit tests for FeatureSelector, Feature Configuration, and Model Ablation."""

import pandas as pd
import polars as pl

from mdk_trading_oracle.core.db import DuckDBManager
from mdk_trading_oracle.models.day_start.forecaster import DayStartForecaster
from mdk_trading_oracle.models.features_config import FeatureSelector
from mdk_trading_oracle.models.sector_day_start.forecaster import SectorDayStartForecaster


def test_feature_selector_defaults():
    """Verify that FeatureSelector loads full canonical features by default."""
    selector_macro = FeatureSelector(model_name="day_start")
    all_macro_feats = selector_macro.get_all_features()
    active_macro_feats = selector_macro.get_active_features()

    assert len(all_macro_feats) == 39
    assert len(active_macro_feats) == 39
    assert "feat_bofa_w4_net_flow_tl" in active_macro_feats
    assert "feat_macro_interest_rate" in active_macro_feats
    assert "feat_bist30_volatility_20d" in active_macro_feats

    selector_sector = FeatureSelector(model_name="sector_day_start")
    all_sector_feats = selector_sector.get_all_features()
    active_sector_feats = selector_sector.get_active_features()

    assert len(all_sector_feats) == 27
    assert len(active_sector_feats) == 27
    assert "feat_sector_bofa_w4_net_flow_tl" in active_sector_feats
    assert "feat_sector_beta_x_bist30_momentum" in active_sector_feats


def test_feature_selector_exclude_individual_features():
    """Verify that individual features can be excluded while keeping the rest of the cluster active."""
    excluded = ["feat_macro_rate_shock_decay", "feat_bofa_holding_flow_prev_day"]
    selector = FeatureSelector(
        model_name="day_start",
        exclude_features=excluded,
    )
    active = selector.get_active_features()

    assert len(active) == 39 - 2
    for ef in excluded:
        assert ef not in active

    # Verify other features in macro_rates and sector_flows are still active
    assert "feat_macro_interest_rate" in active
    assert "feat_bofa_banking_flow_prev_day" in active


def test_feature_selector_disable_cluster():
    """Verify that disabling an entire cluster removes all its child features."""
    selector = FeatureSelector(
        model_name="day_start",
        disabled_clusters=["macro_rates", "calendar_dynamics"],
    )
    active = selector.get_active_features()
    active_clusters = selector.get_active_clusters()

    assert "macro_rates" not in active_clusters
    assert "calendar_dynamics" not in active_clusters

    # 4 macro rate features + 3 calendar features = 7 features removed
    assert len(active) == 39 - 7
    assert "feat_macro_interest_rate" not in active
    assert "day_of_week" not in active
    assert "feat_bofa_w4_net_flow_tl" in active


def test_feature_selector_enable_specific_clusters():
    """Verify that specifying enabled_clusters exclusively activates only those clusters."""
    selector = FeatureSelector(
        model_name="day_start",
        enabled_clusters=["closing_momentum", "competitor_deltas"],
    )
    active = selector.get_active_features()
    active_clusters = selector.get_active_clusters()

    assert set(active_clusters) == {"closing_momentum", "competitor_deltas"}
    # 3 closing momentum + 5 competitor deltas = 8 features
    assert len(active) == 8
    assert "feat_bofa_w4_net_flow_tl" in active
    assert "feat_bofa_vs_top5_w4_flow_delta_tl" in active
    assert "feat_macro_interest_rate" not in active


def test_feature_selector_include_override():
    """Verify that include_features force-includes a feature even if its cluster is disabled."""
    selector = FeatureSelector(
        model_name="day_start",
        disabled_clusters=["macro_rates"],
        include_features=["feat_macro_interest_rate"],
    )
    active = selector.get_active_features()

    assert "feat_macro_interest_rate" in active
    assert "feat_macro_rate_shock_decay" not in active
    assert "feat_macro_rate_spread_vs_30d_mean" not in active


def test_feature_selector_filter_dataframe():
    """Verify that filter_dataframe drops inactive feature columns in both polars and pandas."""
    selector = FeatureSelector(
        model_name="day_start",
        exclude_features=["feat_macro_rate_shock_decay"],
        disabled_clusters=["calendar_dynamics"],
    )

    data = {
        "trade_date": ["2026-03-09", "2026-03-10"],
        "day_of_week": [1, 2],
        "is_monday": [True, False],
        "feat_bofa_w4_net_flow_tl": [10e6, -5e6],
        "feat_macro_interest_rate": [45.0, 45.0],
        "feat_macro_rate_shock_decay": [0.1, 0.05],
        "target_open_net_flow_tl": [15e6, -2e6],
    }

    # Test Polars
    df_pl = pl.DataFrame(data)
    df_filtered_pl = selector.filter_dataframe(df_pl)
    assert "feat_macro_rate_shock_decay" not in df_filtered_pl.columns
    assert "day_of_week" not in df_filtered_pl.columns
    assert "is_monday" not in df_filtered_pl.columns
    assert "feat_bofa_w4_net_flow_tl" in df_filtered_pl.columns
    assert "feat_macro_interest_rate" in df_filtered_pl.columns
    assert "trade_date" in df_filtered_pl.columns
    assert "target_open_net_flow_tl" in df_filtered_pl.columns

    # Test Pandas
    df_pd = pd.DataFrame(data)
    df_filtered_pd = selector.filter_dataframe(df_pd)
    assert "feat_macro_rate_shock_decay" not in df_filtered_pd.columns
    assert "day_of_week" not in df_filtered_pd.columns
    assert "feat_bofa_w4_net_flow_tl" in df_filtered_pd.columns


def test_day_start_forecaster_feature_selection(tmp_path):
    """Test DayStartForecaster end-to-end integration with custom feature exclusions."""
    db = DuckDBManager(read_only=True)
    forecaster = DayStartForecaster(
        db=db,
        model_type="bayesian",
        exclude_features=["feat_macro_rate_shock_decay", "feat_bofa_holding_flow_prev_day"],
        disabled_clusters=["calendar_dynamics"],
    )

    assert len(forecaster.feature_selector.active_features) == 39 - 1 - 1 - 3
    assert "feat_macro_rate_shock_decay" not in forecaster.feature_selector.active_features
    assert "day_of_week" not in forecaster.feature_selector.active_features

    # Verify live forecast generation respects filtered feature matrix
    res = forecaster.forecast_next_day()
    assert res is not None
    assert res.predicted_direction in ["BUY", "SELL", "STRONG_BUY", "STRONG_SELL", "WEAK_BUY", "WEAK_SELL", "NEUTRAL"]


def test_sector_day_start_forecaster_feature_selection():
    """Test SectorDayStartForecaster end-to-end integration with disabled clusters."""
    db = DuckDBManager(read_only=True)
    forecaster = SectorDayStartForecaster(
        db=db,
        model_type="bayesian",
        disabled_clusters=["benchmark_relative_alpha"],
    )

    assert "benchmark_relative_alpha" not in forecaster.feature_selector.get_active_clusters()
    # 27 total - 5 benchmark relative alpha = 22 features
    assert len(forecaster.feature_selector.active_features) == 22

    res_list = forecaster.forecast_next_day(sector="Banking")
    assert len(res_list) == 1
    assert res_list[0].top_predicted_buy_sector == "Banking"


def test_day_start_ablation_study():
    """Test automated Leave-One-Cluster-Out ablation study execution."""
    db = DuckDBManager(read_only=True)
    forecaster = DayStartForecaster(db=db, eval_window_days=5)
    ablation_df = forecaster.run_ablation_study()

    assert not ablation_df.empty
    assert "Experiment" in ablation_df.columns
    assert "Hit_Rate_Pct" in ablation_df.columns
    assert "RMSE_Million_TL" in ablation_df.columns
    assert len(ablation_df) >= 2  # Baseline + at least 1 cluster removed

