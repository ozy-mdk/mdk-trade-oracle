#!/usr/bin/env python
"""Variant test: live input = move from the PREVIOUS CLOSE to 11:30 (one number for the trader) instead of open -> 11:30.

Same candidates, tournament, costs and walk-forward code as production (`data/gold/intraday_session.py`); only the
panel's live input changes: morning_ret := price_1130 (split-adjusted) / previous adjusted close - 1. Target unchanged
(11:30 -> close). Nothing is written to the database; outputs go to ~/data/mdk_oracle/experiments/intraday_variant/.
The production variant (open -> 11:30) is read from the gold SIMULATED ledger / candidate results for comparison.

Usage:
    .venv/bin/python scripts/experiments/intraday_variant_close1130.py            # simulate + report (~15 min)
    .venv/bin/python scripts/experiments/intraday_variant_close1130.py --report-only
"""

from __future__ import annotations

import argparse
import os
import warnings
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

warnings.filterwarnings("ignore")

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

OUT = Path.home() / "data" / "mdk_oracle" / "experiments" / "intraday_variant"
FOCUS = ["ASELS", "GARAN", "AKBNK", "YKBNK", "ISCTR", "TTKOM"]
COST = 0.14


def variant_panel(db, sym: str) -> pd.DataFrame:
    from mdk_trading_oracle.models import intraday_session as m

    p = m.build_panel(db, sym)
    s = db.query_df("SELECT trade_date, close_price AS raw_close, adj_close_price FROM silver_daily_stock_summary WHERE symbol = %s", (sym,))
    s["trade_date"] = pd.to_datetime(s["trade_date"])
    s = s.sort_values("trade_date")
    s["factor"] = s["adj_close_price"] / s["raw_close"]
    s["prev_adj_close"] = s["adj_close_price"].shift(1)
    p = p.merge(s[["trade_date", "factor", "prev_adj_close"]], on="trade_date", how="left")
    p["morning_ret"] = (p["price_1130"] * p["factor"] / p["prev_adj_close"] - 1) * 100  # previous close -> 11:30
    p["usable"] = p["usable"] & p["morning_ret"].notna()
    return p


def simulate(args: tuple[str, str]) -> tuple[pd.DataFrame, list[dict]]:
    os.environ["OMP_NUM_THREADS"] = "1"
    sym, start = args
    from mdk_trading_oracle.core.db import PostgresManager
    from mdk_trading_oracle.data.gold import intraday_session as g
    from mdk_trading_oracle.models import intraday_session as m

    panel = variant_panel(PostgresManager(read_only=True), sym)
    rule = m.rule_state(panel)
    results, ledger, cache, champ, month = [], [], {}, None, None
    empty = pd.DataFrame(columns=["candidate", "session_date", "forecast_pct", "actual_pct", "sigma_pct", "traded"])
    for i in panel.index[panel["trade_date"] >= start]:
        row = panel.loc[i]
        day = row["trade_date"]
        if month != (day.year, day.month):
            month = (day.year, day.month)
            champ = m.crown(pd.DataFrame(results) if results else empty, pd.Timestamp(day.year, day.month, 1))
        grids = m.candidate_grids(panel, i, rule, cache=cache, refit_every=g.SIM_REFIT_EVERY)
        sig = {"symbol": sym, "session_date": day, "source": "SIMULATED", "candidate": champ.name,
               "grid_forecast_pct": list(grids[champ.name]) if champ.name in grids else None}
        gr = g.grade(sig, row)
        if gr:
            ledger.append(gr)
        if bool(row["usable"]):
            results.extend(g._result_rows(sym, row, grids))
    return pd.DataFrame(results), ledger


def summarize(ledger: pd.DataFrame, results: pd.DataFrame, label: str) -> pd.DataFrame:
    rows = []
    ledger = ledger.assign(session_date=pd.to_datetime(ledger["session_date"]))
    results = results.assign(session_date=pd.to_datetime(results["session_date"]))
    rule = results[(results["candidate"] == "RULE") & (results["forecast_pct"] != 0)]
    for grp, syms in (("focus 6", FOCUS), ("other 24", None)):
        for per, lo, hi in (("2023-24", "2023-01-01", "2024-12-31"), ("2025-26", "2025-01-01", "2030-01-01")):
            def sel(d):
                m_ = d["session_date"].between(lo, hi)
                return d[m_ & (d["symbol"].isin(FOCUS) if syms else ~d["symbol"].isin(FOCUS))]
            t = sel(ledger)
            t = t[t["call"] != 0]
            r = sel(rule)
            rows.append({"variant": label, "group": grp, "period": per,
                         "tournament_trades": len(t), "tournament_hit": t["hit"].mean() * 100, "tournament_gain": (t["call"] * t["actual_pct"] - COST).mean(),
                         "rule_trades": len(r), "rule_hit": (np.sign(r["forecast_pct"]) == np.sign(r["actual_pct"])).mean() * 100,
                         "rule_gain": (np.sign(r["forecast_pct"]) * r["actual_pct"] - COST).mean()})
    return pd.DataFrame(rows)


def main() -> None:
    from mdk_trading_oracle.core.db import PostgresManager
    from mdk_trading_oracle.data.gold.intraday_session import SIM_START, universe

    ap = argparse.ArgumentParser()
    ap.add_argument("--report-only", action="store_true")
    a = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    db = PostgresManager(read_only=True)
    if not a.report_only:
        syms = universe(db)
        with ProcessPoolExecutor(max_workers=8) as ex:
            out = list(ex.map(simulate, [(s, SIM_START) for s in syms]))
        pd.concat([o[0] for o in out], ignore_index=True).to_parquet(OUT / "candidate_results.parquet")
        pd.DataFrame([r for o in out for r in o[1]]).to_parquet(OUT / "ledger.parquet")
    v1 = summarize(pd.read_parquet(OUT / "ledger.parquet"), pd.read_parquet(OUT / "candidate_results.parquet"), "prev close -> 11:30 (1 input)")
    led0 = db.query_df("SELECT symbol, session_date, call, actual_pct, hit FROM gold_intraday_session_ledger WHERE source = 'SIMULATED'")
    res0 = db.query_df("SELECT symbol, session_date, candidate, forecast_pct, actual_pct FROM gold_intraday_candidate_results WHERE candidate = 'RULE'")
    v0 = summarize(led0, res0, "open -> 11:30 (2 inputs, current)")
    pd.set_option("display.width", 250)
    rep = pd.concat([v0, v1]).set_index(["group", "period", "variant"]).sort_index().round(2)
    print(rep.to_string())
    rep.to_csv(OUT / "comparison.csv")


if __name__ == "__main__":
    main()
