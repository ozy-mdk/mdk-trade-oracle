#!/usr/bin/env python
"""Generate point-in-time microstructure context for all BIST 30 stocks.

Features:
- Execution Urgency & Aggression (BofA VWAP distance vs Market VWAP)
- Closing Session / Window 5 (16:00-18:15 TRT) Net Flow & Dominance
- Closing Auction Markup vs Day VWAP
- Retail vs Smart Money Flow Divergence (BofA MLB vs Tera TRA)
- Banking Desk Basket Concordance & Net Flow Share (for banks)
- Market-wide BofA Breadth & Net Flow Share (for non-banks)

All features are strictly known as of session T Close (18:15 TRT).
Outputs: ~/data/mdk_oracle/experiments/tournament_variants/microstructure_context.parquet
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from mdk_trading_oracle.core.db import PostgresManager

OUT = Path.home() / "data" / "mdk_oracle" / "experiments" / "tournament_variants"
OUT.mkdir(parents=True, exist_ok=True)
CACHE_FILE = OUT / "microstructure_context.parquet"

BANKS = ["AKBNK", "GARAN", "ISCTR", "YKBNK", "VAKBN", "HALKB"]


def build_microstructure_context() -> pd.DataFrame:
    db = PostgresManager(read_only=True)
    print("Querying silver tables...")

    stocks = db.query_df("""
        SELECT symbol, trade_date::text AS d, close_price, market_vwap, bofa_buy_vwap, bofa_sell_vwap, 
               bofa_net_flow_tl, total_turnover_tl, daily_return_pct, index_name
        FROM silver_daily_stock_summary 
        WHERE index_name = 'BIST30' OR symbol = ANY(%s)
        ORDER BY symbol, trade_date
    """, params=[BANKS])
    stocks["trade_date"] = pd.to_datetime(stocks["d"])
    stocks = stocks.sort_values(["symbol", "trade_date"]).reset_index(drop=True)

    tra = db.query_df("""
        SELECT symbol, trade_date::text AS d, net_flow_tl AS tra_net_flow_tl
        FROM silver_daily_broker_summary
        WHERE broker_id = 'TRA'
    """)
    tra["trade_date"] = pd.to_datetime(tra["d"])

    w5 = db.query_df("""
        SELECT symbol, trade_date::text AS d, net_flow_tl AS mlb_w5_net_flow_tl, total_turnover_tl AS w5_turnover_tl
        FROM silver_intraday_broker_window_summary
        WHERE broker_id = 'MLB' AND window_order = 5
    """)
    w5["trade_date"] = pd.to_datetime(w5["d"])

    print("Merging context frames...")
    df = stocks.merge(tra[["symbol", "trade_date", "tra_net_flow_tl"]], on=["symbol", "trade_date"], how="left")
    df = df.merge(w5[["symbol", "trade_date", "mlb_w5_net_flow_tl", "w5_turnover_tl"]], on=["symbol", "trade_date"], how="left")
    df["tra_net_flow_tl"] = df["tra_net_flow_tl"].fillna(0.0)
    df["mlb_w5_net_flow_tl"] = df["mlb_w5_net_flow_tl"].fillna(0.0)
    df["w5_turnover_tl"] = df["w5_turnover_tl"].fillna(0.0)

    # Rolling median turnover
    df["med_20d_to"] = df.groupby("symbol")["total_turnover_tl"].transform(lambda x: x.rolling(20, min_periods=5).median())

    # 1. BofA Execution Urgency
    net = df["bofa_net_flow_tl"].fillna(0.0)
    mkt = df["market_vwap"].replace(0, np.nan)
    urgency = np.where(
        net > 0, (df["bofa_buy_vwap"] - mkt) / mkt * 100.0,
        np.where(net < 0, -(mkt - df["bofa_sell_vwap"]) / mkt * 100.0, 0.0)
    )
    df["feat_mlb_urgency"] = np.clip(np.nan_to_num(urgency, nan=0.0), -10.0, 10.0)

    intensity = (net / (df["med_20d_to"] + 1e5) * 100.0).clip(-100.0, 100.0)
    df["feat_mlb_urgent_intensity"] = np.where(np.sign(df["feat_mlb_urgency"]) == np.sign(intensity), intensity, 0.0)

    # 2. Closing Auction Markup & Window 5 late flows
    df["feat_close_vs_vwap_pct"] = ((df["close_price"] - mkt) / mkt * 100.0).clip(-10.0, 10.0).fillna(0.0)
    df["feat_mlb_w5_vs_day_ratio"] = (df["mlb_w5_net_flow_tl"] / (net.abs() + 1e5)).clip(-5.0, 5.0).fillna(0.0)
    df["feat_mlb_w5_divergence"] = (np.sign(df["mlb_w5_net_flow_tl"]) * np.sign(df["daily_return_pct"])).fillna(0.0)

    # 3. Retail Divergence (BofA MLB vs Tera TRA)
    df["feat_smart_vs_retail_divergence"] = ((net - df["tra_net_flow_tl"]) / (df["med_20d_to"] + 1e5) * 100.0).clip(-50.0, 50.0).fillna(0.0)

    # 4. Bank Basket vs Market-wide Breadth
    bank_sub = df[df["symbol"].isin(BANKS)]
    bank_daily = bank_sub.groupby("trade_date").agg(
        bank_mlb_flow=("bofa_net_flow_tl", "sum"),
        bank_turnover=("total_turnover_tl", "sum"),
        bank_buyers=("bofa_net_flow_tl", lambda x: (x > 0).sum())
    ).reset_index()
    bank_daily["bank_mlb_share"] = (bank_daily["bank_mlb_flow"] / bank_daily["bank_turnover"] * 100.0).clip(-50.0, 50.0)
    bank_daily["bank_concordance"] = bank_daily["bank_buyers"] / float(len(BANKS))

    bist30_sub = df[df["index_name"] == "BIST30"]
    bist_daily = bist30_sub.groupby("trade_date").agg(
        bist_mlb_flow=("bofa_net_flow_tl", "sum"),
        bist_turnover=("total_turnover_tl", "sum"),
        bist_buyers=("bofa_net_flow_tl", lambda x: (x > 0).sum()),
        bist_total=("symbol", "count")
    ).reset_index()
    bist_daily["bist_mlb_share"] = (bist_daily["bist_mlb_flow"] / bist_daily["bist_turnover"] * 100.0).clip(-50.0, 50.0)
    bist_daily["bist_breadth"] = bist_daily["bist_buyers"] / bist_daily["bist_total"].replace(0, 1)

    df = df.merge(bank_daily[["trade_date", "bank_mlb_share", "bank_concordance"]], on="trade_date", how="left")
    df = df.merge(bist_daily[["trade_date", "bist_mlb_share", "bist_breadth"]], on="trade_date", how="left")

    df["feat_group_mlb_share"] = np.where(df["symbol"].isin(BANKS), df["bank_mlb_share"], df["bist_mlb_share"])
    df["feat_group_mlb_share"] = np.clip(np.nan_to_num(df["feat_group_mlb_share"], nan=0.0), -50.0, 50.0)

    df["feat_group_concordance"] = np.where(df["symbol"].isin(BANKS), df["bank_concordance"], df["bist_breadth"])
    df["feat_group_concordance"] = np.nan_to_num(df["feat_group_concordance"], nan=0.5)

    keep_cols = [
        "symbol", "d", "feat_mlb_urgency", "feat_mlb_urgent_intensity", "feat_close_vs_vwap_pct",
        "feat_mlb_w5_vs_day_ratio", "feat_mlb_w5_divergence",
        "feat_smart_vs_retail_divergence", "feat_group_mlb_share", "feat_group_concordance"
    ]
    out_df = df[keep_cols].rename(columns={"d": "trade_date"})
    out_df.to_parquet(CACHE_FILE)
    print(f"Saved microstructure context to {CACHE_FILE}, shape: {out_df.shape}")
    return out_df


if __name__ == "__main__":
    build_microstructure_context()
