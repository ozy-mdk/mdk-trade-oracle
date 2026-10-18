#!/usr/bin/env python
"""Step 3.3 — extend the 11:30 session signal with new information, one family at a time (isolated, read-only).

Base score (fixed, from intraday_flow_experiment.py): z(yesterday MLB intensity) + z(morning move).
Each candidate is added with equal weight (z-score, trailing 252 sessions per stock, nothing fitted). Keep/drop is
decided on DISCOVERY (2023-2024); the HOLDOUT (2025-2026) is read afterwards. Target: last print before 11:30 ->
official closing auction. Trader metric: sign accuracy on called sessions; also strict edge vs base rate.

Candidates (all known by 11:30 on day d):
    urgency_yday        MLB execution urgency on d-1: buying above / selling below the market VWAP (signed, %)
    urgent_intensity    yesterday MLB intensity kept only when MLB was urgent (otherwise 0)
    group_mlb_yday      MLB net flow on d-1 across the 7 banks (banks) or across BIST30 (ASELS, TTKOM), in % of the
                        group's 20d median turnover
    market_mlb_yday     MLB net flow on d-1 across BIST30 for every stock
    peer_morning        mean morning move (open -> 11:30) of the other focus banks (banks only; 0 for non-banks)

Usage:
    .venv/bin/python scripts/experiments/session_signal_extensions.py
"""

from __future__ import annotations

import sys
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

sys.path.insert(0, str(Path(__file__).parent))
from intraday_flow_experiment import DB, DISCOVERY_END, EVAL_START, OUT_DIR, engineer  # noqa: E402

BANKS = ["AKBNK", "GARAN", "HALKB", "ISCTR", "TSKB", "VAKBN", "YKBNK"]
WINDOW = 252


def daily_context() -> pd.DataFrame:
    """Per (symbol, d) yesterday-known context, keyed to the NEXT trading session of that symbol."""
    from mdk_trading_oracle.core.db import PostgresManager

    db = PostgresManager(read_only=True)
    d = db.query_df(
        "SELECT symbol, trade_date, index_name, market_vwap, bofa_buy_vwap, bofa_sell_vwap, bofa_net_flow_tl, total_turnover_tl "
        "FROM silver_daily_stock_summary WHERE index_name = 'BIST30' OR symbol = ANY(%s)",
        (BANKS,),
    )
    d["trade_date"] = pd.to_datetime(d["trade_date"])
    d = d.sort_values(["symbol", "trade_date"])
    d["med_tl"] = d.groupby("symbol")["total_turnover_tl"].transform(lambda x: x.rolling(20, min_periods=5).median())
    net = d["bofa_net_flow_tl"].fillna(0.0)
    mkt = d["market_vwap"].replace(0, np.nan)
    d["urgency"] = np.where(net > 0, (d["bofa_buy_vwap"] - mkt) / mkt, np.where(net < 0, -(mkt - d["bofa_sell_vwap"]) / mkt, 0.0)) * 100
    d["intensity"] = net / d["med_tl"] * 100

    def group_flow(mask: pd.Series) -> pd.Series:
        g = d[mask].groupby("trade_date")
        return g["bofa_net_flow_tl"].sum() / g["med_tl"].sum() * 100

    bank_flow = group_flow(d["symbol"].isin(BANKS))
    bist_flow = group_flow(d["index_name"] == "BIST30")
    d["bank_mlb"] = d["trade_date"].map(bank_flow)
    d["market_mlb"] = d["trade_date"].map(bist_flow)
    d["group_mlb"] = np.where(d["symbol"].isin(BANKS), d["bank_mlb"], d["market_mlb"])
    d["urgent_intensity"] = np.where(np.sign(d["urgency"]) == np.sign(d["intensity"]), d["intensity"], 0.0)
    # yesterday's values aligned to today's row
    out = d[["symbol", "trade_date"]].copy()
    for src, dst in (("urgency", "urgency_yday"), ("urgent_intensity", "urgent_intensity"), ("group_mlb", "group_mlb_yday"), ("market_mlb", "market_mlb_yday")):
        out[dst] = d.groupby("symbol")[src].shift(1)
    return out


def zscore(c: pd.DataFrame, f: str) -> pd.Series:
    g = c.groupby("symbol")[f]
    mu = g.transform(lambda x: x.shift(1).rolling(WINDOW, min_periods=126).mean())
    sd = g.transform(lambda x: x.shift(1).rolling(WINDOW, min_periods=126).std())
    return ((c[f] - mu) / sd).clip(-5, 5)


def calls(c: pd.DataFrame, score: str, q: float) -> pd.Series:
    g = c.groupby("symbol")[score]
    hi = g.transform(lambda x: x.shift(1).rolling(WINDOW, min_periods=126).quantile(q))
    lo = g.transform(lambda x: x.shift(1).rolling(WINDOW, min_periods=126).quantile(1 - q))
    return pd.Series(np.where(c[score] > hi, 1, np.where(c[score] < lo, -1, 0)), index=c.index)


def stats(c: pd.DataFrame, call: pd.Series) -> dict[str, float]:
    k = call != 0
    y = c["rest_1130"]
    base = np.where(call[k] > 0, (y > DB).mean(), (y < -DB).mean())
    hit = call[k] * y[k] > DB
    sign_ok = np.sign(call[k]) == np.sign(y[k])
    per_stock = pd.Series(sign_ok.to_numpy(), index=c.loc[k, "symbol"].to_numpy()).groupby(level=0).mean()
    pnl = call[k] * y[k]
    return {"coverage_pct": k.mean() * 100, "sign_acc_pct": sign_ok.mean() * 100, "edge_pp": (hit.mean() - base.mean()) * 100,
            "mean_ret_pct": pnl.mean(), "t_stat": pnl.mean() / pnl.std(ddof=1) * np.sqrt(len(pnl)),
            "stocks_sign_acc_gt_50": int((per_stock > 0.5).sum()), "worst_stock_acc_pct": per_stock.min() * 100}


def main() -> None:
    pd.set_option("display.width", 250)
    panel, _ = engineer(pd.read_parquet(OUT_DIR / "tick_panel.parquet"))
    c = panel.dropna(subset=["rest_1130"]).merge(daily_context(), on=["symbol", "trade_date"], how="left")
    c = c.sort_values(["symbol", "trade_date"]).reset_index(drop=True)
    banks_in_panel = [s for s in c["symbol"].unique() if s in BANKS]
    mr = c.pivot_table(index="trade_date", columns="symbol", values="morning_ret")
    peer = {s: mr[[b for b in banks_in_panel if b != s]].mean(axis=1) for s in banks_in_panel}
    c["peer_morning"] = [peer[s].get(d, np.nan) if s in peer else 0.0 for s, d in zip(c["symbol"], c["trade_date"])]

    for f in ("mlb_net_yday_intensity", "morning_ret", "urgency_yday", "urgent_intensity", "group_mlb_yday", "market_mlb_yday", "peer_morning"):
        c[f"z_{f}"] = zscore(c, f) if f != "peer_morning" else np.where(c["symbol"].isin(banks_in_panel), zscore(c, f), 0.0)
    c["period"] = np.where(c["trade_date"] < EVAL_START, None, np.where(c["trade_date"] <= DISCOVERY_END, "discovery", "holdout"))

    print("=== standalone IC vs rest-of-session (mean over stocks; banks only for peer_morning) ===")
    rows = []
    for f in ("urgency_yday", "urgent_intensity", "group_mlb_yday", "market_mlb_yday", "peer_morning"):
        r = {"feature": f}
        for per, g in c.dropna(subset=["period"]).groupby("period"):
            g = g[g["symbol"].isin(banks_in_panel)] if f == "peer_morning" else g
            ics = g.groupby("symbol").apply(lambda x: x[f].corr(x["rest_1130"], method="spearman"))
            r[f"ic_{per}"], r[f"pos_{per}"] = ics.mean(), int((ics > 0).sum())
        rows.append(r)
    print(pd.DataFrame(rows).round(3).to_string(index=False))

    variants = {
        "base": ["z_mlb_net_yday_intensity", "z_morning_ret"],
        "+urgency": ["z_mlb_net_yday_intensity", "z_morning_ret", "z_urgency_yday"],
        "+urgent_intensity": ["z_mlb_net_yday_intensity", "z_morning_ret", "z_urgent_intensity"],
        "+group_mlb": ["z_mlb_net_yday_intensity", "z_morning_ret", "z_group_mlb_yday"],
        "+market_mlb": ["z_mlb_net_yday_intensity", "z_morning_ret", "z_market_mlb_yday"],
        "+peer_morning": ["z_mlb_net_yday_intensity", "z_morning_ret", "z_peer_morning"],
    }
    rows = []
    for name, cols in variants.items():
        c["score"] = c[cols].sum(axis=1, min_count=len(cols))
        for q in (0.8, 0.9):
            call = calls(c, "score", q)
            for per in ("discovery", "holdout"):
                m = c["period"] == per
                rows.append({"variant": name, "q": q, "period": per, **stats(c[m], call[m])})
    res = pd.DataFrame(rows)
    res.to_csv(OUT_DIR / "extensions.csv", index=False)
    for per in ("discovery", "holdout"):
        print(f"\n=== {per.upper()} — combined score variants (call beyond trailing q / 1-q) ===")
        print(res[res["period"] == per].drop(columns="period").round(2).to_string(index=False))


if __name__ == "__main__":
    main()
