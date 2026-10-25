"""Event-calendar and market-regime features for the per-horizon production policy.

Feature sets (tested end-to-end with the honest re-crowning harness, docs/RESEARCH_HANDOFF_2026-10.md §0.8):
    production       the existing feature modes unchanged
    events           + TCMB MPC decision distance / last change, ex-dividend distance / yield, BIST 30 rebalancing
                     distance and index entry / exit (1D, 3D)
    regime_no_sma50  + BIST 30 downtrend flag, trend strength and mean-reversion features x downtrend flag, without
                     `feat_price_vs_sma50_pct` (5D, 10D)

Everything is point-in-time at the close of row t. Look-ahead only to dates that are public in advance: the MPC
calendar (published a year ahead), ex-dividend dates within 20 sessions and BIST 30 entries / exits within 10 sessions
of the quarterly effective date.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from mdk_trading_oracle.core.db import PostgresManager

EVENT_COLS = [
    "feat_evt_mpc_sessions_to_next", "feat_evt_mpc_sessions_since_last", "feat_evt_mpc_next_session",
    "feat_evt_mpc_last_change_bps", "feat_evt_div_sessions_to_next", "feat_evt_div_yield_next_pct",
    "feat_evt_rebal_sessions_to_next", "feat_evt_rebal_entering", "feat_evt_rebal_leaving",
]
MEAN_REVERSION = [
    "feat_price_vs_sma20_pct", "feat_price_vs_sma50_pct", "feat_sma20_vs_sma50_pct", "feat_close_ret_20d",
    "feat_big5_cost_spread_pct", "feat_mlb_cost_spread_pct", "feat_big5_ytd_pnl_share_20d_to", "feat_mlb_tertip_share",
]
REGIME_COLS = ["feat_regime_down", "feat_regime_trend"]
DROPPED_REGIME = ["feat_price_vs_sma50_pct", "feat_price_vs_sma50_pct_x_down"]
FEATURE_SETS = ("production", "events", "regime_no_sma50")

# Per-horizon production policy: feature set, lookback excluded from the candidate pool, crowned with the policy pool
# (True) or by the plain production crowning (False: 30D keeps its own family-first crowning).
HORIZON_POLICY: dict[str, tuple[str, int | None, bool]] = {
    "1d": ("events", None, True),
    "3d": ("events", None, True),
    "5d": ("regime_no_sma50", None, True),
    "10d": ("regime_no_sma50", None, True),
    "15d": ("production", 63, True),
    "30d": ("production", None, False),
}
KEY_SEP = "|"


def policy_key(feature_set: str, base_key: str) -> str:
    """Candidate key stored in gold: production keys unchanged, other feature sets prefixed (`events|Ridge/252/...`)."""
    return base_key if feature_set == "production" else f"{feature_set}{KEY_SEP}{base_key}"


def split_key(key: str) -> tuple[str, str]:
    return tuple(key.split(KEY_SEP, 1)) if KEY_SEP in key else ("production", key)  # type: ignore[return-value]


def load_event_tables(db: PostgresManager) -> dict[str, Any]:
    q = db.query_df
    return {
        "mpc": q("SELECT meeting_date::text AS d FROM bronze_tcmb_mpc_meetings WHERE meeting_type = 'scheduled' ORDER BY 1")["d"].tolist(),
        "rates": q("SELECT trade_date::text AS d, rate_change FROM silver_daily_macro_rates ORDER BY 1"),
        "div": q("SELECT symbol, ex_date::text AS d, gross_dividend_tl FROM bronze_dividends ORDER BY 1, 2"),
        "rebal": q("SELECT effective_date::text AS d, COALESCE(in_symbols, '') AS ins, COALESCE(out_symbols, '') AS outs "
                   "FROM bronze_bist30_changes ORDER BY 1"),
        "bench": q("SELECT trade_date, close_price FROM silver_daily_benchmark_index ORDER BY trade_date"),
    }


def add_event_features(df: pd.DataFrame, sym: str, ev: dict[str, Any]) -> pd.DataFrame:
    df = df.copy()
    dates = pd.to_datetime(df["trade_date"]).dt.strftime("%Y-%m-%d").tolist()
    n = len(dates)

    def sessions_until(target: str, i: int) -> float:
        j = next((k for k in range(i + 1, n) if dates[k] >= target), None)
        if j is not None:
            return float(j - i)
        return float(i - n + 1 + len(pd.bdate_range(dates[-1], target)) - 1) if target > dates[-1] else 0.0

    mpc = ev["mpc"]
    rc = dict(zip(ev["rates"]["d"], ev["rates"]["rate_change"].fillna(0.0)))
    div = ev["div"][ev["div"]["symbol"] == sym]
    rebal = ev["rebal"]
    out = {c: np.zeros(n) for c in EVENT_COLS}
    close = df["close_price"].to_numpy(float)
    for i, d in enumerate(dates):
        nxt = next((m for m in mpc if m > d), None)
        prv = next((m for m in reversed(mpc) if m <= d), None)
        out["feat_evt_mpc_sessions_to_next"][i] = min(30.0, sessions_until(nxt, i)) if nxt else 30.0
        out["feat_evt_mpc_next_session"][i] = 1.0 if nxt and out["feat_evt_mpc_sessions_to_next"][i] == 1.0 else 0.0
        if prv:
            k = next((j for j in range(i, -1, -1) if dates[j] <= prv), None)
            out["feat_evt_mpc_sessions_since_last"][i] = min(30.0, float(i - k)) if k is not None else 30.0
            after = next((x for x in dates if x > prv), None)  # the rate change is visible the session after
            out["feat_evt_mpc_last_change_bps"][i] = rc.get(after, 0.0) * 100.0 if after and after <= d else 0.0
        else:
            out["feat_evt_mpc_sessions_since_last"][i] = 30.0
        nd = div[div["d"] > d]
        if len(nd):
            k = sessions_until(nd["d"].iloc[0], i)
            out["feat_evt_div_sessions_to_next"][i] = min(20.0, k)
            out["feat_evt_div_yield_next_pct"][i] = nd["gross_dividend_tl"].iloc[0] / close[i] * 100.0 if k <= 20 and close[i] > 0 else 0.0
        else:
            out["feat_evt_div_sessions_to_next"][i] = 20.0
        nr = rebal[rebal["d"] > d]
        if len(nr):
            k = sessions_until(nr["d"].iloc[0], i)
            out["feat_evt_rebal_sessions_to_next"][i] = min(30.0, k)
            if k <= 10:
                out["feat_evt_rebal_entering"][i] = 1.0 if sym in [x.strip() for x in nr["ins"].iloc[0].split(",")] else 0.0
                out["feat_evt_rebal_leaving"][i] = 1.0 if sym in [x.strip() for x in nr["outs"].iloc[0].split(",")] else 0.0
        else:
            out["feat_evt_rebal_sessions_to_next"][i] = 30.0
    for c, v in out.items():
        df[c] = v
    return df


def add_regime_features(df: pd.DataFrame, ev: dict[str, Any]) -> pd.DataFrame:
    """BIST 30 downtrend flag (below SMA50 and 20-session return < 0), trend strength, mean-reversion x downtrend;
    `feat_price_vs_sma50_pct` removed. Index statistics on the benchmark's own sessions, carried to the stock's dates."""
    df = df.copy()
    b = ev["bench"].copy()
    b["trade_date"] = pd.to_datetime(b["trade_date"])
    px = b.set_index("trade_date")["close_price"].astype(float)
    vs_sma50 = (px / px.rolling(50, min_periods=25).mean() - 1.0) * 100.0
    ret20 = (px / px.shift(20) - 1.0) * 100.0
    m = pd.DataFrame({"vs": vs_sma50, "r20": ret20}).sort_index().ffill()
    m = m.reindex(pd.to_datetime(df["trade_date"]), method="ffill").reset_index(drop=True)
    down = ((m["vs"] < 0) & (m["r20"] < 0)).astype(float).to_numpy()
    df["feat_regime_down"] = down
    df["feat_regime_trend"] = m["vs"].clip(-15, 15).fillna(0.0).to_numpy()
    for c in MEAN_REVERSION:
        if c in df.columns:
            df[f"{c}_x_down"] = (df[c].fillna(0.0).to_numpy() * down).astype(float)
    return df.drop(columns=[c for c in DROPPED_REGIME if c in df.columns])


def apply_feature_set(df: pd.DataFrame, sym: str, feature_set: str, ev: dict[str, Any] | None) -> tuple[pd.DataFrame, list[str], list[str]]:
    """(frame, extra 1D columns, dropped columns) for a feature set; swing modes pick up new feat_ columns automatically."""
    if feature_set == "production":
        return df, [], []
    if feature_set == "events":
        return add_event_features(df, sym, ev), list(EVENT_COLS), []
    if feature_set == "regime_no_sma50":
        out = add_regime_features(df, ev)
        extra = REGIME_COLS + [f"{c}_x_down" for c in MEAN_REVERSION if f"{c}_x_down" in out.columns and f"{c}_x_down" not in DROPPED_REGIME]
        return out, extra, list(DROPPED_REGIME)
    raise ValueError(f"unknown feature set {feature_set}")
