"""Unit and Integration Tests for MDK Trading Oracle Explainability Engine."""

import numpy as np
import pandas as pd
import pytest
from lightgbm import LGBMRegressor
from sklearn.linear_model import BayesianRidge

from mdk_trading_oracle.explainability import (
    FeatureAuditor,
    ModelExplainer,
    format_markdown_card,
    plot_cluster_donut,
    plot_waterfall,
)


@pytest.fixture
def synthetic_data():
    """Generate synthetic microstructure feature dataset."""
    np.random.seed(42)
    n = 60
    feats = [
        "feat_bofa_w4_net_flow_tl",
        "feat_bofa_w4_turnover_tl",
        "feat_top5_domestic_w4_net_flow_tl",
        "feat_bofa_cost_basis_spread_20d_pct",
        "feat_macro_interest_rate",
        "feat_noise_random",
    ]
    data = {
        "feat_bofa_w4_net_flow_tl": np.random.normal(20e6, 50e6, n),
        "feat_bofa_w4_turnover_tl": np.random.uniform(50e6, 200e6, n),
        "feat_top5_domestic_w4_net_flow_tl": np.random.normal(-10e6, 40e6, n),
        "feat_bofa_cost_basis_spread_20d_pct": np.random.normal(0.01, 0.03, n),
        "feat_macro_interest_rate": np.full(n, 45.0) + np.random.normal(0, 0.5, n),
        "feat_noise_random": np.random.normal(0, 1.0, n),
    }
    # Create target with strong dependence on w4 flow and cost spread
    target = (
        0.5 * data["feat_bofa_w4_net_flow_tl"]
        + 0.3 * data["feat_top5_domestic_w4_net_flow_tl"]
        + 10e6 * data["feat_bofa_cost_basis_spread_20d_pct"]
        + np.random.normal(0, 5e6, n)
    )
    df = pd.DataFrame(data)
    cluster_map = {
        "feat_bofa_w4_net_flow_tl": "closing_momentum",
        "feat_bofa_w4_turnover_tl": "closing_momentum",
        "feat_top5_domestic_w4_net_flow_tl": "competitor_deltas",
        "feat_bofa_cost_basis_spread_20d_pct": "cost_basis_pnl",
        "feat_macro_interest_rate": "macro_rates",
        "feat_noise_random": "noise",
    }
    return df, pd.Series(target, name="target_open_net_flow_tl"), feats, cluster_map


def test_tree_shap_local_additive_attribution(synthetic_data):
    """Verify that TreeSHAP satisfies the additive efficiency axiom: base_value + sum(attrs) == y_hat."""
    X, y, feats, cluster_map = synthetic_data
    model = LGBMRegressor(n_estimators=20, max_depth=3, random_state=42, verbose=-1)
    model.fit(X[feats], y)

    explainer = ModelExplainer(
        model=model,
        feature_names=feats,
        cluster_map=cluster_map,
        background_data=X[feats],
        model_name="TestLGBM",
        unit="TL",
    )

    test_row = X.iloc[[0]]
    local_exp = explainer.explain_instance(test_row)

    assert local_exp.model_name == "TestLGBM"
    assert local_exp.unit == "TL"
    assert len(local_exp.feature_attributions) == len(feats)

    # Additive efficiency verification
    attr_sum = sum(a.attribution for a in local_exp.feature_attributions)
    reconstructed = local_exp.base_value + attr_sum
    np.testing.assert_allclose(reconstructed, local_exp.predicted_value, rtol=1e-4)

    # Verify cluster aggregation
    cluster_attr_sum = sum(c.total_attribution for c in local_exp.cluster_attributions)
    np.testing.assert_allclose(cluster_attr_sum, attr_sum, rtol=1e-4)

    # Verify percentage shares sum to 100%
    total_share = sum(c.percentage_share for c in local_exp.cluster_attributions)
    np.testing.assert_allclose(total_share, 100.0, rtol=1e-3)


def test_bayesian_ridge_analytical_attribution(synthetic_data):
    """Verify analytical linear attribution for Bayesian Ridge models."""
    X, y, feats, cluster_map = synthetic_data
    model = BayesianRidge()
    model.fit(X[feats], y)

    explainer = ModelExplainer(
        model=model,
        feature_names=feats,
        cluster_map=cluster_map,
        background_data=X[feats],
        model_name="TestBayesianRidge",
        unit="TL",
    )

    test_row = X.iloc[[5]]
    local_exp = explainer.explain_instance(test_row)

    attr_sum = sum(a.attribution for a in local_exp.feature_attributions)
    reconstructed = local_exp.base_value + attr_sum
    np.testing.assert_allclose(reconstructed, local_exp.predicted_value, rtol=1e-4)


def test_global_explanation_rankings(synthetic_data):
    """Verify global feature ranking and cluster importance calculations."""
    X, y, feats, cluster_map = synthetic_data
    model = LGBMRegressor(n_estimators=20, max_depth=3, random_state=42, verbose=-1)
    model.fit(X[feats], y)

    explainer = ModelExplainer(
        model=model,
        feature_names=feats,
        cluster_map=cluster_map,
        background_data=X[feats],
        model_name="TestLGBM",
        unit="TL",
    )

    global_exp = explainer.explain_global(X[feats])
    assert not global_exp.feature_importance_df.empty
    assert not global_exp.cluster_importance_df.empty
    assert len(global_exp.top_features) > 0

    # Ensure relative percentage shares sum to ~100%
    feat_pct_sum = global_exp.feature_importance_df["relative_importance_pct"].sum()
    np.testing.assert_allclose(feat_pct_sum, 100.0, rtol=1e-2)

    cluster_pct_sum = global_exp.cluster_importance_df["relative_importance_pct"].sum()
    np.testing.assert_allclose(cluster_pct_sum, 100.0, rtol=1e-2)


def test_feature_auditor_collinearity_and_pruning(synthetic_data):
    """Verify FeatureAuditor identifies redundant collinear features and zero-alpha noise."""
    X, y, feats, cluster_map = synthetic_data

    # Artificially inject an almost identical copy of w4 net flow
    X_with_collinear = X.copy()
    X_with_collinear["feat_bofa_w4_clone"] = X_with_collinear["feat_bofa_w4_net_flow_tl"] * 0.999 + 100.0

    all_feats = feats + ["feat_bofa_w4_clone"]
    extended_cluster_map = {**cluster_map, "feat_bofa_w4_clone": "closing_momentum"}

    model = LGBMRegressor(n_estimators=20, max_depth=3, random_state=42, verbose=-1)
    model.fit(X_with_collinear[all_feats], y)

    auditor = FeatureAuditor(
        feature_names=all_feats,
        cluster_map=extended_cluster_map,
        collinearity_threshold=0.90,
    )

    report = auditor.audit(model, X_with_collinear[all_feats], y, model_name="day_start_test")

    assert report.total_features == len(all_feats)
    assert report.evaluated_sessions == len(X_with_collinear)

    # Must detect the collinear clone
    collinear_found = any(
        (p["feature_a"] == "feat_bofa_w4_net_flow_tl" and p["feature_b"] == "feat_bofa_w4_clone")
        or (p["feature_a"] == "feat_bofa_w4_clone" and p["feature_b"] == "feat_bofa_w4_net_flow_tl")
        for p in report.collinear_pairs
    )
    assert collinear_found

    # Must have prune candidates and generated yaml
    assert len(report.prune_candidates) > 0
    assert "exclude_features" in report.recommended_features_yaml


def test_visualizers(synthetic_data):
    """Verify Plotly waterfall and donut charts and markdown formatting."""
    X, y, feats, cluster_map = synthetic_data
    model = LGBMRegressor(n_estimators=10, max_depth=2, random_state=42, verbose=-1)
    model.fit(X[feats], y)

    explainer = ModelExplainer(
        model=model,
        feature_names=feats,
        cluster_map=cluster_map,
        background_data=X[feats],
        model_name="TestLGBM",
        unit="TL",
    )

    local_exp = explainer.explain_instance(X.iloc[[0]])
    global_exp = explainer.explain_global(X[feats])

    # Plotly figures
    fig_waterfall = plot_waterfall(local_exp)
    assert fig_waterfall is not None
    assert len(fig_waterfall.data) > 0

    fig_donut = plot_cluster_donut(global_exp)
    assert fig_donut is not None
    assert len(fig_donut.data) > 0

    # Markdown card
    md_card = format_markdown_card(local_exp)
    assert "Forecast Attribution Breakdown" in md_card
    assert "Top Catalysts" in md_card
    assert "Top Headwinds" in md_card
    assert "Microstructure Cluster Rollup" in md_card
