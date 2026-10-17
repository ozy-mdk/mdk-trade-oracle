"""Regression tests: corporate actions (splits) must not leak into ML targets, features or realized returns."""

import numpy as np
import pandas as pd
import pytest

from mdk_trading_oracle.data.silver.tertip_ml_forecaster import adjusted_price
from mdk_trading_oracle.data.silver.tertip_multi_horizon_forecaster import (
    compute_forward_terminal_return,
    engineer_multi_horizon_features,
)


def _split_frame(n: int = 80, split_at: int = 50) -> pd.DataFrame:
    """Smooth +0.5%/day price path with a 2:1 split at `split_at` (raw price halves, adjusted is continuous)."""
    adj = 100.0 * 1.005 ** np.arange(n)
    factor = np.where(np.arange(n) < split_at, 2.0, 1.0)  # back-adjusted: latest bar equals raw
    raw = adj * factor
    return pd.DataFrame({
        "trade_date": pd.bdate_range("2026-01-01", periods=n),
        "open_price": raw, "high_price": raw * 1.01, "low_price": raw * 0.99, "close_price": raw,
        "adj_open_price": adj, "adj_high_price": adj * 1.01, "adj_low_price": adj * 0.99, "adj_close_price": adj,
        "daily_return_pct": np.zeros(n), "total_turnover_tl": np.full(n, 1e8),
        "mlb_flow": np.zeros(n), "big5_flow": np.zeros(n),
    })


def test_adjusted_price_falls_back_to_raw():
    df = pd.DataFrame({"close_price": [10.0, 11.0], "adj_close_price": [5.0, np.nan]})
    assert adjusted_price(df).tolist() == [5.0, 11.0]
    assert adjusted_price(df.drop(columns="adj_close_price")).tolist() == [10.0, 11.0]


def test_forward_targets_ignore_split():
    df = _split_frame()
    _, raw_ret = compute_forward_terminal_return(df["close_price"], 5)
    price, ret = compute_forward_terminal_return(df["close_price"], 5, adjusted_price(df))
    window = slice(45, 50)  # targets spanning the split
    assert raw_ret.iloc[window].min() < -40.0  # the bug: split shows up as a ~-49% move
    assert ret.dropna().to_numpy() == pytest.approx((1.005 ** 5 - 1) * 100.0)
    # Terminal price restated in session-t share terms
    assert price.iloc[45] == pytest.approx(df["close_price"].iloc[45] * 1.005 ** 5)


def test_trend_and_atr_features_ignore_split():
    feats, _ = engineer_multi_horizon_features(_split_frame())
    after = feats.iloc[52:60]
    assert after["feat_close_ret_5d"].to_numpy() == pytest.approx((1.005 ** 5 - 1) * 100.0)
    assert after["feat_price_vs_sma20_pct"].abs().max() < 10.0
    assert feats["atr_14d_pct"].max() < 3.0
