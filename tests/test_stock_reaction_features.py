"""Unit tests for Model 3 (Stock Intraday Reaction Forecaster) feature extraction and configuration."""

import pytest

from mdk_trading_oracle.models.features_config import FeatureSelector
from mdk_trading_oracle.models.stock_reaction.features import (
    DEFAULT_TRACKED_BROKERS,
    StockReactionFeatureExtractor,
)


def test_stock_reaction_feature_columns_count():
    """Assert exactly 49 feature columns across 8 semantic microstructure clusters."""
    extractor = StockReactionFeatureExtractor(symbol="AKBNK")
    feat_cols = extractor.get_feature_columns()

    assert len(feat_cols) == 49, f"Expected 49 features, got {len(feat_cols)}: {feat_cols}"
    assert len(StockReactionFeatureExtractor.CLUSTER_NAMES) == 8

    # Cluster 1: BofA W1 Execution Signal
    assert "feat_bofa_w1_buy_vol" in feat_cols
    assert "feat_bofa_w1_net_flow_tl" in feat_cols
    assert "feat_bofa_w1_vol_share" in feat_cols
    assert "feat_bofa_w1_direction_sign" in feat_cols
    assert "feat_bofa_w1_direction_strength" in feat_cols
    assert "feat_bofa_w1_market_vwap" in feat_cols

    # Cluster 2: Multi-broker W1 alignment
    assert "feat_comp_w1_net_flow_tl" in feat_cols
    assert "feat_iym_w1_net_flow_tl" in feat_cols
    assert "feat_tra_w1_net_flow_tl" in feat_cols
    assert "feat_w1_bofa_comp_alignment" in feat_cols
    assert "feat_w1_bofa_tra_contra_signal" in feat_cols

    # Cluster 3: T-1 stock momentum
    assert "feat_stock_ret_t1_1d" in feat_cols
    assert "feat_stock_ret_t1_5d" in feat_cols
    assert "feat_stock_dist_sma20_t1" in feat_cols
    assert "feat_stock_vol_20d_t1" in feat_cols

    # Cluster 4: FIFO inventory
    assert "feat_bofa_t1_open_qty" in feat_cols
    assert "feat_bofa_t1_cost_spread_pct" in feat_cols
    assert "feat_bofa_t1_unrealized_pnl_tl" in feat_cols
    assert "feat_tra_t1_open_qty" in feat_cols
    assert "feat_dom5_t1_open_qty" in feat_cols

    # Cluster 5: Multi-day accumulation
    assert "feat_bofa_accum_5d_t1_tl" in feat_cols
    assert "feat_bofa_accum_20d_t1_tl" in feat_cols
    assert "feat_bofa_flow_zscore_t1" in feat_cols

    # Cluster 6: Sector breadth
    assert "feat_sector_ret_t1" in feat_cols
    assert "feat_peer_spread_t1" in feat_cols

    # Cluster 7: Macro carry & shock dynamics
    assert "feat_macro_repo_rate_t1" in feat_cols
    assert "feat_macro_rate_delta_t1" in feat_cols
    assert "feat_macro_carry_t1" in feat_cols
    assert "feat_macro_rate_shock_decay_t1" in feat_cols

    # Cluster 8: Calendar
    assert "feat_day_of_week" in feat_cols
    assert "feat_is_monday" in feat_cols
    assert "feat_is_friday" in feat_cols
    assert "feat_day_of_month" in feat_cols


def test_stock_reaction_target_mapping():
    """Assert window target columns correctly resolve for all aliases."""
    extractor = StockReactionFeatureExtractor(symbol="AKBNK")
    assert extractor.get_target_column("w2") == "target_w2_return_pct"
    assert extractor.get_target_column("w3") == "target_w3_return_pct"
    assert extractor.get_target_column("w5") == "target_w5_return_pct"
    assert extractor.get_target_column("first_reaction") == "target_w2_return_pct"
    assert extractor.get_target_column("midday_followup") == "target_w3_return_pct"
    assert extractor.get_target_column("closing_session") == "target_w5_return_pct"

    with pytest.raises(ValueError):
        extractor.get_target_column("invalid_window")


def test_stock_reaction_feature_selector():
    """Verify FeatureSelector loads stock_reaction cluster config from features.yaml."""
    selector = FeatureSelector(model_name="stock_reaction")
    all_feats = selector.get_all_features()
    active_feats = selector.get_active_features()

    assert len(all_feats) == 49
    assert len(active_feats) == 49
    assert "feat_bofa_w1_net_flow_tl" in active_feats
    assert "feat_bofa_w1_direction_strength" in active_feats
    assert "feat_w1_bofa_tra_contra_signal" in active_feats
    assert "feat_bofa_t1_open_qty" in active_feats


def test_stock_reaction_tracked_brokers_universe():
    """Verify default 7-broker universe contains BofA, 5 domestic banks, and Tera."""
    assert "MLB" in DEFAULT_TRACKED_BROKERS
    assert "TRA" in DEFAULT_TRACKED_BROKERS
    for dom in ["IYM", "YKR", "AKM", "GRM", "ZRY"]:
        assert dom in DEFAULT_TRACKED_BROKERS
