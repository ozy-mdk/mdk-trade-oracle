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


def test_institutional_positions_are_side_aware_and_carried_forward():
    from mdk_trading_oracle.data.silver.tertip_ml_forecaster import institutional_positions

    pos = pd.DataFrame({
        "trade_date": ["2026-01-01", "2026-01-01", "2026-01-03"],
        "broker_id": ["YKR", "GRM", "YKR"],
        "position_side": ["LONG", "SHORT", "LONG"],
        "open_stock_quantity": [100.0, 80.0, 120.0],
        "open_fifo_cost_tl": [1000.0, 880.0, 1240.0],  # YKR long @10 (then @10.33), GRM short @11
    })
    dates = pd.Series(pd.to_datetime(["2026-01-01", "2026-01-02", "2026-01-03"]))
    out = institutional_positions(pos, dates, pd.Series([10.0, 12.0, 12.0]))
    # BIG5 net = long 100 - short 80 = +20 (not 180); GRM's short is carried to 01-02 and 01-03 without new rows
    assert out["big5_net_qty"].tolist() == [20.0, 20.0, 40.0]
    assert out["big5_gross_qty"].tolist() == [180.0, 180.0, 200.0]
    # 01-02 @12: YKR long +200, GRM short 880 - 960 = -80 -> +120; only YKR is in profit
    assert out["big5_unrealized_pnl_tl_mtm"].iloc[1] == pytest.approx(120.0)
    assert out["big5_profitable_qty"].iloc[1] == 100.0
    assert out["mlb_net_qty"].tolist() == [0.0, 0.0, 0.0]
