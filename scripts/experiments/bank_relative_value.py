#!/usr/bin/env python
"""Step 3.5 — relative value within the banks from end-of-day MLB flow (isolated experiment, read-only).

All inputs are known at the close of T (end-of-day data delivery); the target is session T+1.
Hypothesis: BofA reallocates within the sector (buys one bank, sells another). Removing the common sector move,
the stock-specific pressure should persist into the next session.

    universe   7 banks (AKBNK, GARAN, HALKB, ISCTR, TSKB, VAKBN, YKBNK); focus-4 subset reported separately
    feature    MLB net intensity at T (net flow / trailing 20d median turnover), z-scored on each bank's trailing 252
               sessions, then demeaned across banks on T
    baseline   yesterday's relative return (short-term reversal competitor), placebo noise
    target     relative T+1 return = bank return - mean bank return; close->close (adjusted) and open->close
    metrics    daily cross-sectional Spearman IC (mean, t over days); daily pair long top-flow / short bottom-flow
               bank: hit = positive spread, mean spread, t, rolling-60 share of windows with hit >= 60%;
               selective pair: trade only when the top-bottom flow gap is beyond its trailing 80th percentile
Discovery 2023-2024 decides; holdout 2025-2026 read afterwards.

Usage:
    .venv/bin/python scripts/experiments/bank_relative_value.py
"""

from __future__ import annotations

import warnings
from pathlib import Path

warnings.filterwarnings("ignore")

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

BANKS = ["AKBNK", "GARAN", "HALKB", "ISCTR", "TSKB", "VAKBN", "YKBNK"]
FOCUS_BANKS = ["AKBNK", "GARAN", "ISCTR", "YKBNK"]
OUT = Path.home() / "data" / "mdk_oracle" / "experiments" / "bank_relative_value"
EVAL_START, DISCOVERY_END = "2023-01-01", "2024-12-31"
WINDOW = 252


def load() -> pd.DataFrame:
    from mdk_trading_oracle.core.db import PostgresManager

    d = PostgresManager(read_only=True).query_df(
        "SELECT symbol, trade_date, open_price, close_price, adj_close_price, total_turnover_tl, bofa_net_flow_tl "
        "FROM silver_daily_stock_summary WHERE symbol = ANY(%s) ORDER BY symbol, trade_date",
        (BANKS,),
    )
    d["trade_date"] = pd.to_datetime(d["trade_date"])
    g = d.groupby("symbol")
    med = g["total_turnover_tl"].transform(lambda x: x.rolling(20, min_periods=5).median())
    d["mlb_int"] = d["bofa_net_flow_tl"].fillna(0) / med * 100
    mu = g["mlb_int"].transform(lambda x: x.shift(1).rolling(WINDOW, min_periods=126).mean())
    sd = g["mlb_int"].transform(lambda x: x.shift(1).rolling(WINDOW, min_periods=126).std())
    d["z_flow"] = ((d["mlb_int"] - mu) / sd).clip(-5, 5)
    d["ret_cc"] = (d["adj_close_price"] / g["adj_close_price"].shift(1) - 1) * 100
    d["ret_oc"] = (d["close_price"] / d["open_price"] - 1) * 100
    d["placebo"] = np.random.default_rng(9).normal(size=len(d))
    # next-session targets aligned to row T
    for t in ("ret_cc", "ret_oc"):
        d[f"next_{t}"] = d.groupby("symbol")[t].shift(-1)
    return d


def cross_section(d: pd.DataFrame, universe: list[str]) -> pd.DataFrame:
    x = d[d["symbol"].isin(universe)].copy()
    n = x.groupby("trade_date")["symbol"].transform("count")
    x = x[n == len(universe)]  # complete cross-sections only
    for c in ("z_flow", "ret_cc", "placebo"):
        x[f"rel_{c}"] = x[c] - x.groupby("trade_date")[c].transform("mean")
    for t in ("next_ret_cc", "next_ret_oc"):
        x[f"rel_{t}"] = x[t] - x.groupby("trade_date")[t].transform("mean")
    x["period"] = np.where(x["trade_date"] < EVAL_START, None, np.where(x["trade_date"] <= DISCOVERY_END, "discovery", "holdout"))
    return x[x["period"].notna()]


def daily_ic(x: pd.DataFrame, f: str, t: str) -> pd.Series:
    return x.dropna(subset=[f, t]).groupby("trade_date").apply(lambda g: g[f].corr(g[t], method="spearman") if g[f].nunique() > 1 else np.nan).dropna()


def pairs(x: pd.DataFrame, f: str, t: str, sign: float = 1.0) -> pd.DataFrame:
    """Daily long top-f / short bottom-f bank (sign=-1 flips for reversal-type features)."""
    rows = []
    for day, g in x.dropna(subset=[f, t]).groupby("trade_date"):
        s = g.sort_values(f)
        lo, hi = s.iloc[0], s.iloc[-1]
        long_, short_ = (hi, lo) if sign > 0 else (lo, hi)
        rows.append({"trade_date": day, "period": g["period"].iloc[0], "gap": hi[f] - lo[f], "spread": long_[t] - short_[t]})
    p = pd.DataFrame(rows).sort_values("trade_date")
    p["gap_hi"] = p["gap"].shift(1).rolling(WINDOW, min_periods=126).quantile(0.8)
    return p


def pair_stats(p: pd.DataFrame) -> dict[str, float]:
    s = p["spread"]
    hits = (s > 0).astype(float)
    roll = hits.rolling(60, min_periods=60).mean().dropna() if len(hits) >= 60 else pd.Series(dtype=float)
    return {"days": len(s), "pair_hit_pct": hits.mean() * 100, "mean_spread_pct": s.mean(), "t_stat": s.mean() / s.std(ddof=1) * np.sqrt(len(s)),
            "windows_ge_60_pct": (roll >= 0.6).mean() * 100 if len(roll) else np.nan}


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    pd.set_option("display.width", 250)
    d = load()
    rows, prow = [], []
    for uname, universe in (("7 banks", BANKS), ("focus 4", FOCUS_BANKS)):
        x = cross_section(d, universe)
        for f, sign in (("rel_z_flow", 1.0), ("rel_ret_cc", -1.0), ("rel_placebo", 1.0)):
            for t in ("rel_next_ret_cc", "rel_next_ret_oc"):
                for per, g in x.groupby("period"):
                    ic = daily_ic(g, f, t)
                    rows.append({"universe": uname, "feature": f, "target": t.replace("rel_next_", ""), "period": per, "mean_ic": ic.mean(),
                                 "t_ic": ic.mean() / ic.std(ddof=1) * np.sqrt(len(ic)), "days": len(ic)})
                    if f != "rel_placebo":
                        pp = pairs(g, f, t, sign)
                        prow.append({"universe": uname, "feature": f, "target": t.replace("rel_next_", ""), "period": per, "mode": "every day", **pair_stats(pp)})
                        sel = pp[pp["gap"] > pp["gap_hi"]]
                        if len(sel) > 20:
                            prow.append({"universe": uname, "feature": f, "target": t.replace("rel_next_", ""), "period": per, "mode": "wide gap (top 20%)", **pair_stats(sel)})
    ic_tab, pair_tab = pd.DataFrame(rows), pd.DataFrame(prow)
    ic_tab.to_csv(OUT / "ic.csv", index=False)
    pair_tab.to_csv(OUT / "pairs.csv", index=False)
    print("=== daily cross-sectional IC vs next-session relative return (rel_ret_cc is the reversal baseline) ===")
    print(ic_tab.pivot_table(index=["universe", "feature", "target"], columns="period", values=["mean_ic", "t_ic"]).round(3).to_string())
    print("\n=== pair trade: long top / short bottom bank (reversal baseline: long yesterday's loser) ===")
    print(pair_tab.round(2).to_string(index=False))


if __name__ == "__main__":
    main()
