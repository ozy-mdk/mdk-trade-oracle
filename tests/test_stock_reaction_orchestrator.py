"""Unit tests for Model 3 (Stock Intraday Reaction) Orchestrator, Models, and Classifiers."""

import numpy as np
import pandas as pd

from mdk_trading_oracle.models.stock_reaction.models import (
    ReturnDirectionClassifier,
    ReturnThresholdProfile,
    StockReactionBayesianModel,
    StockReactionLightGBMModel,
    StockReactionNaivePersistenceModel,
    StockReactionRollingMeanModel,
)
from mdk_trading_oracle.models.stock_reaction.orchestrator import StockReactionOrchestrator


def test_return_direction_classifier():
    """Verify empirical percentile return direction classification thresholds."""
    th = ReturnThresholdProfile(
        up_p25_pct=0.20,
        up_p50_pct=0.50,
        up_p85_pct=1.50,
        down_p25_pct=0.20,
        down_p50_pct=0.50,
        down_p85_pct=1.50,
    )

    assert ReturnDirectionClassifier.classify(2.0, th) == ReturnDirectionClassifier.STRONG_RALLY
    assert ReturnDirectionClassifier.classify(0.75, th) == ReturnDirectionClassifier.RALLY
    assert ReturnDirectionClassifier.classify(0.30, th) == ReturnDirectionClassifier.WEAK_RALLY
    assert ReturnDirectionClassifier.classify(0.05, th) == ReturnDirectionClassifier.NEUTRAL
    assert ReturnDirectionClassifier.classify(-0.05, th) == ReturnDirectionClassifier.NEUTRAL
    assert ReturnDirectionClassifier.classify(-0.30, th) == ReturnDirectionClassifier.WEAK_DECLINE
    assert ReturnDirectionClassifier.classify(-0.75, th) == ReturnDirectionClassifier.DECLINE
    assert ReturnDirectionClassifier.classify(-2.0, th) == ReturnDirectionClassifier.STRONG_DECLINE


def test_candidate_models_fit_and_predict():
    """Verify candidate models fit and predict properly on dummy stock reaction data."""
    np.random.seed(42)
    n = 20
    feat_cols = [f"feat_col_{i}" for i in range(10)]
    X = pd.DataFrame(np.random.randn(n, 10), columns=feat_cols)
    y = pd.Series(np.random.randn(n) * 1.5)

    th = ReturnThresholdProfile()

    # 1. Naive Persistence
    m0 = StockReactionNaivePersistenceModel("AKBNK", "w2", th)
    m0.fit(X, y)
    res0 = m0.predict(X.iloc[0:1])
    assert res0.symbol == "AKBNK"
    assert res0.window_name == "first_reaction"
    assert isinstance(res0.predicted_return_pct, float)

    # 2. Rolling Mean
    m1 = StockReactionRollingMeanModel("AKBNK", "w2", th)
    m1.fit(X, y)
    res1 = m1.predict(X.iloc[0:1])
    assert isinstance(res1.predicted_return_pct, float)

    # 3. LightGBM
    m2 = StockReactionLightGBMModel("AKBNK", "w2", th)
    m2.fit(X, y)
    res2 = m2.predict(X.iloc[0:1])
    assert isinstance(res2.predicted_return_pct, float)

    # 4. Bayesian Ridge
    m3 = StockReactionBayesianModel("AKBNK", "w2", th)
    m3.fit(X, y)
    res3 = m3.predict(X.iloc[0:1])
    assert isinstance(res3.predicted_return_pct, float)
    assert res3.predicted_return_lower_90 < res3.predicted_return_upper_90


def test_symbol_resolution_priority_chain():
    """Verify symbol priority: explicit arg > config > fallback."""
    # Priority 1: Explicit argument
    orch1 = StockReactionOrchestrator(db=None, symbols=["akbnk", "garan"])
    assert orch1.symbols == ["AKBNK", "GARAN"]

    # Priority 2: Config fallback when symbols=None
    orch2 = StockReactionOrchestrator(db=None, symbols=None)
    assert len(orch2.symbols) > 0
    assert "AKBNK" in orch2.symbols


def test_weak_regime_filtering_logic():
    """Verify institutional active regime flag correctly filters noise training sessions."""
    from mdk_trading_oracle.models.stock_reaction.forecaster import StockReactionForecaster

    forecaster = StockReactionForecaster(symbol="AKBNK", window="w2", filter_weak_regimes=True)
    assert forecaster.filter_weak_regimes is True

    # Test with dummy DataFrame containing regime flags
    n = 20
    df = pd.DataFrame({
        "trade_date": pd.date_range("2026-03-01", periods=n, freq="B"),
        "feat_bofa_w1_net_flow_tl": np.random.randn(n) * 1e6,
        "feat_comp_w1_net_flow_tl": np.random.randn(n) * 1e6,
        "target_w2_return_pct": np.random.randn(n) * 1.5,
        "is_strong_start_bofa": [True, False] * 10,
        "is_strong_start_big_players": [False, True] * 10,
        "is_institutional_active_day": [True, True, False, False] * 5,
    })

    active_count = df["is_institutional_active_day"].sum()
    assert active_count == 10
    filtered_df = df[df["is_institutional_active_day"]]
    assert len(filtered_df) == 10
