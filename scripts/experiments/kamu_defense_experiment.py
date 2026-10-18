#!/usr/bin/env python
"""KAMU (state-owned brokers) index-defense hypothesis, with BIG5 and MLB for comparison (isolated, read-only).

User hypothesis (2026-10-18): KAMU desks (ZRY, VKY, HLY, TRA) defend BIST30 levels, especially around negative index
shocks (politically sensitive), and may trade with an information advantage. Previous tests were unconditional and
per stock and used the stock's OWN +/-3% move as "shock"; this tests the index-level, conditional version.

K1 (descriptive gate): across that day's BIST30 constituents, aggregate net flow of KAMU / BIG5 / MLB (whole day and
    closing session W5, 16:00-18:15) in % of BIST30 turnover, bucketed by the BIST30 close-to-close return.
    Defense = KAMU net buying rising as the index falls, concentrated in the close.

K2-K5 (predictive; all features known at the close of T, index level, trailing-252 z-scores; decided on discovery):
    K2  strong vs weak KAMU defense on down days (BIST30 <= -1% / <= -2%) -> next session
    K3  KAMU closing-session (W5) buying on down days -> next gap / next open->close
    K4  KAMU absorbing MLB selling (KAMU z > 0 and MLB z < 0) on down days -> next session
    K5  BIG5 and MLB - BIG5 at index level, all days -> next session
Targets (graded separately): BIST30 next close->close, next gap (open T+1 / close T), next open->close; focus stocks
next open->close and next 11:30->close.

Usage:
    .venv/bin/python scripts/experiments/kamu_defense_experiment.py --k1
    .venv/bin/python scripts/experiments/kamu_defense_experiment.py --k2
"""

from __future__ import annotations

import argparse
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

OUT = Path.home() / "data" / "mdk_oracle" / "experiments" / "kamu_defense"
EVAL_START, DISCOVERY_END = "2023-01-01", "2024-12-31"
BUCKETS = [-100, -3, -2, -1, 0, 1, 2, 3, 100]


def index_flows() -> pd.DataFrame:
    """Per trading day: BIST30 return and group net flows across that day's BIST30 constituents."""
    from mdk_trading_oracle.core.db import PostgresManager
    from mdk_trading_oracle.data.silver.tertip_analytics import BIG_FIVE_BROKERS, KAMU_BROKERS

    db = PostgresManager(read_only=True)
    groups = {"mlb": ["MLB"], "big5": list(BIG_FIVE_BROKERS), "kamu": list(KAMU_BROKERS)}
    case = " ".join(f"WHEN b.broker_id = ANY(ARRAY{ids}) THEN '{g}'" for g, ids in groups.items())
    members = "SELECT trade_date, symbol, total_turnover_tl FROM silver_daily_stock_summary WHERE index_name = 'BIST30'"
    day = db.query_df(f"""
        WITH m AS ({members})
        SELECT b.trade_date, CASE {case} END AS grp, SUM(b.net_flow_tl) AS net
        FROM silver_daily_broker_summary b JOIN m USING (trade_date, symbol)
        WHERE b.broker_id = ANY(%s) GROUP BY 1, 2""", (sum(groups.values(), []),))
    w5 = db.query_df(f"""
        WITH m AS ({members})
        SELECT b.trade_date, CASE {case} END AS grp, SUM(b.net_flow_tl) AS net
        FROM silver_intraday_broker_window_summary b JOIN m USING (trade_date, symbol)
        WHERE b.broker_id = ANY(%s) AND b.window_name = 'closing_session' GROUP BY 1, 2""", (sum(groups.values(), []),))
    tt = db.query_df(f"SELECT trade_date, SUM(total_turnover_tl) AS bist30_tl FROM ({members}) m GROUP BY 1")
    idx = db.query_df(
        "SELECT trade_date, daily_return_pct * 100 AS bist30_ret, intraday_return_pct * 100 AS bist30_intraday, is_forward_filled "
        "FROM silver_daily_benchmark_index WHERE index_code = 'XU030' OR index_code IS NOT NULL"
    )
    d = idx[~idx["is_forward_filled"].fillna(False)].drop(columns="is_forward_filled").drop_duplicates("trade_date")
    d = d.merge(tt, on="trade_date")
    for name, src in (("day", day), ("w5", w5)):
        p = src.pivot_table(index="trade_date", columns="grp", values="net", aggfunc="sum").add_suffix(f"_{name}_net").reset_index()
        d = d.merge(p, on="trade_date", how="left")
    d["trade_date"] = pd.to_datetime(d["trade_date"])
    d = d.sort_values("trade_date").reset_index(drop=True)
    for g in ("mlb", "big5", "kamu"):
        for w in ("day", "w5"):
            d[f"{g}_{w}_share"] = d[f"{g}_{w}_net"] / d["bist30_tl"] * 100
    d["period"] = np.where(d["trade_date"] < EVAL_START, None, np.where(d["trade_date"] <= DISCOVERY_END, "discovery", "holdout"))
    return d


def k1(d: pd.DataFrame) -> None:
    pd.set_option("display.width", 250)
    v = d[d["period"].notna()].copy()
    v["bucket"] = pd.cut(v["bist30_ret"], BUCKETS, labels=["<=-3", "-3..-2", "-2..-1", "-1..0", "0..1", "1..2", "2..3", ">=3"])
    cols = ["kamu_day_share", "kamu_w5_share", "big5_day_share", "big5_w5_share", "mlb_day_share", "mlb_w5_share"]
    for per, g in v.groupby("period"):
        t = g.groupby("bucket")[cols].mean()
        t.insert(0, "days", g.groupby("bucket").size())
        print(f"\n=== K1 {per}: mean net flow across BIST30 constituents, % of BIST30 turnover, by BIST30 close-to-close return ===")
        print(t.round(3).to_string())
        rho = {c: g[c].corr(g["bist30_ret"], method="spearman") for c in cols}
        print("same-day Spearman with BIST30 return: " + ", ".join(f"{c} {r:+.2f}" for c, r in rho.items()))
        down = g[g["bist30_ret"] <= -2]
        if len(down) > 5:
            print(f"on {len(down)} days <= -2%: share of days KAMU net buyer {(down['kamu_day_share'] > 0).mean() * 100:.0f}% "
                  f"(all days {(g['kamu_day_share'] > 0).mean() * 100:.0f}%), KAMU W5 net buyer {(down['kamu_w5_share'] > 0).mean() * 100:.0f}% "
                  f"(all days {(g['kamu_w5_share'] > 0).mean() * 100:.0f}%), KAMU vs MLB opposite sign {(np.sign(down['kamu_day_share']) != np.sign(down['mlb_day_share'])).mean() * 100:.0f}%")


def with_targets(d: pd.DataFrame) -> pd.DataFrame:
    """Next-session targets for BIST30 and the focus stocks, aligned to row T; trailing z-scores of index flows."""
    import sys

    sys.path.insert(0, str(Path(__file__).parent))
    from intraday_flow_experiment import OUT_DIR, engineer

    from mdk_trading_oracle.core.db import PostgresManager

    db = PostgresManager(read_only=True)
    ix = db.query_df("SELECT trade_date, open_price, close_price FROM silver_daily_benchmark_index WHERE index_code = 'XU030' AND NOT COALESCE(is_forward_filled, FALSE)")
    ix["trade_date"] = pd.to_datetime(ix["trade_date"])
    ix = ix.sort_values("trade_date")
    ix["next_cc"] = (ix["close_price"].shift(-1) / ix["close_price"] - 1) * 100
    ix["next_gap"] = (ix["open_price"].shift(-1) / ix["close_price"] - 1) * 100
    ix["next_oc"] = (ix["close_price"].shift(-1) / ix["open_price"].shift(-1) - 1) * 100
    d = d.merge(ix[["trade_date", "next_cc", "next_gap", "next_oc"]], on="trade_date", how="left")
    for c in ("kamu_day_share", "kamu_w5_share", "big5_day_share", "mlb_day_share"):
        mu = d[c].shift(1).rolling(252, min_periods=126).mean()
        sd = d[c].shift(1).rolling(252, min_periods=126).std()
        d[f"z_{c}"] = ((d[c] - mu) / sd).clip(-5, 5)
    d["z_mlb_minus_big5"] = d["z_mlb_day_share"] - d["z_big5_day_share"]
    # focus stocks: next-session open->close and 11:30->close, averaged across the six (equal weight)
    p, _ = engineer(pd.read_parquet(OUT_DIR / "tick_panel.parquet"))
    p = p.sort_values(["symbol", "trade_date"])
    p["oc"] = (p["close_price"] / p["open_price"] - 1) * 100
    stock = p.groupby("trade_date")[["oc", "rest_1130"]].mean()
    nxt = stock.shift(-1).rename(columns={"oc": "focus_next_oc", "rest_1130": "focus_next_1130"})  # next trading row
    return d.merge(nxt, left_on="trade_date", right_index=True, how="left")


def cond_table(d: pd.DataFrame, mask: pd.Series, split: pd.Series, label: str, targets: list[str]) -> list[dict]:
    rows = []
    for per in ("discovery", "holdout"):
        m = mask & (d["period"] == per)
        for side, sm in (("strong", m & split), ("weak", m & ~split)):
            g = d[sm]
            r = {"test": label, "period": per, "group": side, "days": len(g)}
            for t in targets:
                y = g[t].dropna()
                r[f"{t}_mean"] = y.mean()
                r[f"{t}_up_pct"] = (y > 0).mean() * 100 if len(y) else np.nan
                r[f"{t}_t"] = y.mean() / y.std(ddof=1) * np.sqrt(len(y)) if len(y) > 2 else np.nan
            rows.append(r)
    return rows


def k2_k5(d: pd.DataFrame) -> None:
    pd.set_option("display.width", 300)
    pd.set_option("display.max_columns", 40)
    d = with_targets(d)
    d = d[d["period"].notna()]
    T = ["next_cc", "next_gap", "next_oc", "focus_next_oc", "focus_next_1130"]
    rows = []
    for thr in (-1.0, -2.0):
        down = d["bist30_ret"] <= thr
        rows += cond_table(d, down, d["z_kamu_day_share"] > 1.0, f"K2 KAMU day z>1 | BIST30<={thr:+.0f}%", T)
        rows += cond_table(d, down, d["z_kamu_w5_share"] > 1.0, f"K3 KAMU W5 z>1 | BIST30<={thr:+.0f}%", T)
        rows += cond_table(d, down, (d["z_kamu_day_share"] > 0) & (d["z_mlb_day_share"] < 0), f"K4 KAMU buys vs MLB sells | BIST30<={thr:+.0f}%", T)
    res = pd.DataFrame(rows)
    res.to_csv(OUT / "k2_k4_conditional.csv", index=False)
    show = ["test", "period", "group", "days"] + [f"{t}_{k}" for t in T for k in ("mean", "up_pct")]
    print("=== K2-K4: next-session outcomes after down days, strong vs weak KAMU defense (mean %, % of sessions up) ===")
    print(res[show].round(2).to_string(index=False))

    print("\n=== K5 + all-days ICs of index-level flows vs next session (Spearman over days) ===")
    rows = []
    for f in ("z_kamu_day_share", "z_kamu_w5_share", "z_big5_day_share", "z_mlb_day_share", "z_mlb_minus_big5"):
        r = {"feature": f}
        for per, g in d.groupby("period"):
            for t in T:
                r[f"{t}_{per[:4]}"] = g[f].corr(g[t], method="spearman")
        rows.append(r)
    print(pd.DataFrame(rows).set_index("feature").round(3).to_string())


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--k1", action="store_true")
    ap.add_argument("--k2", action="store_true", help="K2-K5 predictive tests")
    a = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    d = index_flows()
    d.to_parquet(OUT / "index_flows.parquet")
    if a.k1:
        k1(d)
    if a.k2:
        k2_k5(d)


if __name__ == "__main__":
    main()
