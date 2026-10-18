#!/usr/bin/env python
"""Event study around MPC rate decisions and BIST30 rebalances (isolated experiment, read-only; data already in DB).

Events since 2023:
    MPC       rate changes in bronze_central_bank_rates (1-week repo). rate_date is the effective date; the decision is
              announced on the previous session (14:00), which is day 0. Holds are not in our data (changes only).
    REBAL     BIST30 effective dates from bronze_bist30_changes (day 0 = effective date).
Per event, days -5..+5: BIST30 close-to-close return, mean focus-stock return, and index-level net flow of KAMU / BIG5
/ MLB across BIST30 constituents (% of BIST30 turnover). Insider question: does pre-event positioning (-5..-1)
predict the post-event move (0..+5)? With ~19 MPC events the power is low; signs and counts are reported, not t-stats.

Usage:
    .venv/bin/python scripts/experiments/event_study.py
"""

from __future__ import annotations

import warnings
from pathlib import Path

warnings.filterwarnings("ignore")

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

FOCUS = ["ASELS", "GARAN", "AKBNK", "YKBNK", "ISCTR", "TTKOM"]
OUT = Path.home() / "data" / "mdk_oracle" / "experiments" / "event_study"
WIN = range(-5, 6)


def daily_frame() -> pd.DataFrame:
    from mdk_trading_oracle.core.db import PostgresManager

    db = PostgresManager(read_only=True)
    ix = pd.read_parquet(Path.home() / "data" / "mdk_oracle" / "experiments" / "kamu_defense" / "index_flows.parquet")
    st = db.query_df("SELECT symbol, trade_date, adj_close_price FROM silver_daily_stock_summary WHERE symbol = ANY(%s)", (FOCUS,))
    st["trade_date"] = pd.to_datetime(st["trade_date"])
    st = st.sort_values(["symbol", "trade_date"])
    st["ret"] = st.groupby("symbol")["adj_close_price"].pct_change() * 100
    focus = st.groupby("trade_date")["ret"].mean().rename("focus_ret")
    d = ix[["trade_date", "bist30_ret", "kamu_day_share", "big5_day_share", "mlb_day_share"]].merge(focus, left_on="trade_date", right_index=True, how="left")
    return d.sort_values("trade_date").reset_index(drop=True)


def events(d: pd.DataFrame) -> pd.DataFrame:
    from mdk_trading_oracle.core.db import PostgresManager

    db = PostgresManager(read_only=True)
    r = db.query_df("SELECT rate_date, rate_change FROM bronze_central_bank_rates WHERE is_rate_change_day AND NOT is_forward_filled AND rate_date >= '2023-01-01'")
    b = db.query_df("SELECT effective_date FROM bronze_bist30_changes WHERE effective_date >= '2023-01-01'")
    dates = pd.Series(d["trade_date"])
    rows = []
    for _, e in r.iterrows():
        eff = pd.Timestamp(e["rate_date"])
        prior = dates[dates < eff]
        if len(prior):
            rows.append({"type": "MPC hike" if e["rate_change"] > 0 else "MPC cut", "day0": prior.iloc[-1], "size_pp": e["rate_change"]})
    for _, e in b.iterrows():
        eff = pd.Timestamp(e["effective_date"])
        on = dates[dates >= eff]
        if len(on):
            rows.append({"type": "BIST30 rebalance", "day0": on.iloc[0], "size_pp": np.nan})
    ev = pd.DataFrame(rows)
    ev["period"] = np.where(ev["day0"] <= pd.Timestamp("2024-12-31"), "discovery", "holdout")
    return ev


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    pd.set_option("display.width", 250)
    d = daily_frame()
    ev = events(d)
    pos = {t: i for i, t in enumerate(d["trade_date"])}
    rows = []
    for _, e in ev.iterrows():
        i0 = pos[e["day0"]]
        for k in WIN:
            if 0 <= i0 + k < len(d):
                r = d.iloc[i0 + k]
                rows.append({"type": e["type"], "day0": e["day0"], "period": e["period"], "k": k, **{c: r[c] for c in ["bist30_ret", "focus_ret", "kamu_day_share", "big5_day_share", "mlb_day_share"]}})
    p = pd.DataFrame(rows)
    p.to_parquet(OUT / "event_panel.parquet")
    print("events:", ev.groupby(["type", "period"]).size().to_dict())
    for typ, g in p.groupby("type"):
        t = g.groupby("k")[["bist30_ret", "focus_ret", "kamu_day_share", "big5_day_share", "mlb_day_share"]].mean()
        t["bist30_cum"] = t["bist30_ret"].cumsum()
        print(f"\n=== {typ} ({g['day0'].nunique()} events): mean by day relative to event (returns %, flows % of BIST30 turnover) ===")
        print(t.round(3).to_string())

    # insider question: pre-event flow (-5..-1) vs post-event move (0..+5), per event
    pre = p[p["k"].between(-5, -1)].groupby(["type", "day0", "period"])[["kamu_day_share", "mlb_day_share", "big5_day_share"]].sum()
    post = p[p["k"].between(0, 5)].groupby(["type", "day0", "period"])[["bist30_ret", "focus_ret"]].sum()
    x = pre.join(post).reset_index()
    print("\n=== pre-event net flow (-5..-1, summed) vs post-event return (0..+5, summed), per event ===")
    rows = []
    for (typ), g in x.groupby("type"):
        for f in ("kamu_day_share", "mlb_day_share", "big5_day_share"):
            same = (np.sign(g[f]) == np.sign(g["bist30_ret"])).mean() * 100
            rows.append({"type": typ, "pre_flow": f, "events": len(g), "spearman_vs_post_bist30": g[f].corr(g["bist30_ret"], method="spearman"),
                         "pct_events_flow_sign_eq_post_move": same})
    print(pd.DataFrame(rows).round(2).to_string(index=False))
    print("\nper MPC event:")
    m = x[x["type"].str.startswith("MPC")].sort_values("day0")
    print(m[["type", "day0", "kamu_day_share", "mlb_day_share", "big5_day_share", "bist30_ret", "focus_ret"]].round(2).to_string(index=False))


if __name__ == "__main__":
    main()
