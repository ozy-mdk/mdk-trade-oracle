#!/usr/bin/env python
"""Step 3.2 — does early-session institutional flow predict the rest of the session? (isolated, read-only)

Decision time 11:30 TRT (end of W2). Everything used as a feature comes from trades with time < 11:30:00, the same
half-open cut-off silver uses for W1+W2. Entry price = last trade before 11:30; exit = the official closing-auction
close (silver close_price, verified = last tick on 99.9% of days). Same-day prices, so no split adjustment is needed.
A W1 variant uses 10:30 as decision time. Half-day sessions (last trade before 17:00) are excluded.

Targets (graded only against themselves, never mixed with T+1 numbers):
    rest_1130   close / p(11:30) - 1   (%)    main
    rest_1030   close / p(10:30) - 1   (%)    W1 variant
Features at 11:30 (shares of early-session market turnover, except intensity):
    mlb / big5 / kamu / inst net early share, MLB gross buy early share, MLB net early intensity
    (vs trailing 20-session median daily turnover), morning return p(11:30)/open, opening gap open/prev close,
    yesterday's full-day MLB intensity; placebo iid noise.
Protocol: discovery 2023-2024, holdout 2025-2026; ICs with per-stock sign checks; selective rule (trailing 80/20
quantile of a feature) vs the base rate of that move; pooled Ridge on z-scored early features with selective calls.

Usage:
    .venv/bin/python scripts/experiments/intraday_flow_experiment.py               # build tick panel + report
    .venv/bin/python scripts/experiments/intraday_flow_experiment.py --report-only
"""

from __future__ import annotations

import argparse
import glob
import os
import sys
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import polars as pl  # noqa: E402

FOCUS = ["ASELS", "GARAN", "AKBNK", "YKBNK", "ISCTR", "TTKOM"]
OUT_DIR = Path.home() / "data" / "mdk_oracle" / "experiments" / "intraday_flow"
ARCHIVE = str(Path.home() / "data" / "mdk_oracle" / "00_raw_data" / "mdk_parquet_part_0*" / "*.parquet")
EVAL_START, DISCOVERY_END = "2023-01-01", "2024-12-31"
DB = 0.25
N_BOOT, BLOCK = 1000, 10


def tick_frame() -> pl.LazyFrame:
    from mdk_trading_oracle.core.db import PostgresManager

    cols = ["timestamp", "symbol", "price", "volume", "buyer_broker_id", "seller_broker_id"]
    arch = pl.scan_parquet(sorted(glob.glob(ARCHIVE))).filter(pl.col("symbol").is_in(FOCUS)).select(cols)
    arch_max = arch.select(pl.col("timestamp").max()).collect().item()
    db = PostgresManager(read_only=True)
    recent = db.query_df(
        "SELECT timestamp, symbol, price, volume, buyer_broker_id, seller_broker_id FROM bronze_raw_trades "
        "WHERE symbol = ANY(%s) AND timestamp > %s",
        (FOCUS, arch_max),
    )
    recent_pl = pl.from_pandas(recent).with_columns(pl.col("timestamp").cast(pl.Datetime("us")), pl.col("price").cast(pl.Float64), pl.col("volume").cast(pl.Float64))
    return pl.concat([arch, recent_pl.lazy().select(cols)], how="vertical_relaxed")


def build_panel() -> pd.DataFrame:
    from mdk_trading_oracle.core.db import PostgresManager
    from mdk_trading_oracle.data.silver.tertip_analytics import BIG_FIVE_BROKERS, KAMU_BROKERS

    t = tick_frame().with_columns(
        pl.col("timestamp").dt.date().alias("trade_date"),
        pl.col("timestamp").dt.time().alias("t"),
        (pl.col("price") * pl.col("volume")).alias("tl"),
    )
    groups = {"mlb": ["MLB"], "big5": list(BIG_FIVE_BROKERS), "kamu": list(KAMU_BROKERS)}
    early = pl.col("t") < pl.time(11, 30)
    w1 = pl.col("t") < pl.time(10, 30)
    aggs = [
        pl.col("price").sort_by("timestamp").first().alias("open_tick"),
        pl.col("timestamp").max().dt.time().alias("last_time"),
        pl.col("price").filter(w1).sort_by(pl.col("timestamp").filter(w1)).last().alias("p_1030"),
        pl.col("price").filter(early).sort_by(pl.col("timestamp").filter(early)).last().alias("p_1130"),
        pl.col("tl").filter(early).sum().alias("early_tl"),
        pl.col("tl").filter(w1).sum().alias("w1_tl"),
        pl.col("tl").sum().alias("day_tl"),
        # realistic entry: VWAP just after the 11:30 decision
        (pl.col("tl").filter(pl.col("t").is_between(pl.time(11, 30), pl.time(11, 35), closed="left")).sum()
         / pl.col("volume").filter(pl.col("t").is_between(pl.time(11, 30), pl.time(11, 35), closed="left")).sum()).alias("vwap_1130_1135"),
        (pl.col("tl").filter(pl.col("t").is_between(pl.time(11, 30), pl.time(11, 45), closed="left")).sum()
         / pl.col("volume").filter(pl.col("t").is_between(pl.time(11, 30), pl.time(11, 45), closed="left")).sum()).alias("vwap_1130_1145"),
    ]
    for g, ids in groups.items():
        buy, sell = pl.col("buyer_broker_id").is_in(ids), pl.col("seller_broker_id").is_in(ids)
        aggs += [
            pl.col("tl").filter(early & buy).sum().alias(f"{g}_buy_early_tl"),
            pl.col("tl").filter(early & sell).sum().alias(f"{g}_sell_early_tl"),
            pl.col("tl").filter(w1 & buy).sum().alias(f"{g}_buy_w1_tl"),
            pl.col("tl").filter(w1 & sell).sum().alias(f"{g}_sell_w1_tl"),
            pl.col("tl").filter(buy).sum().alias(f"{g}_buy_day_tl"),
            pl.col("tl").filter(sell).sum().alias(f"{g}_sell_day_tl"),
        ]
    day = t.group_by(["symbol", "trade_date"]).agg(aggs).collect().to_pandas()
    day["trade_date"] = pd.to_datetime(day["trade_date"])

    db = PostgresManager(read_only=True)
    s = db.query_df(
        "SELECT symbol, trade_date, open_price, close_price, adj_close_price FROM silver_daily_stock_summary WHERE symbol = ANY(%s)", (FOCUS,)
    )
    s["trade_date"] = pd.to_datetime(s["trade_date"])
    s = s.sort_values(["symbol", "trade_date"])
    s["prev_adj_close"] = s.groupby("symbol")["adj_close_price"].shift(1)
    s["adj_factor"] = s["adj_close_price"] / s["close_price"]
    panel = s.merge(day, on=["symbol", "trade_date"], how="inner")
    panel.to_parquet(OUT_DIR / "tick_panel.parquet")
    return panel


def engineer(p: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    p = p.sort_values(["symbol", "trade_date"]).copy()
    p = p[p["last_time"].astype(str) >= "17:00:00"]  # drop half-day sessions
    et, w1t = p["early_tl"].replace(0, np.nan), p["w1_tl"].replace(0, np.nan)
    for g in ("mlb", "big5", "kamu"):
        p[f"{g}_net_early_share"] = (p[f"{g}_buy_early_tl"] - p[f"{g}_sell_early_tl"]) / et * 100
        p[f"{g}_net_w1_share"] = (p[f"{g}_buy_w1_tl"] - p[f"{g}_sell_w1_tl"]) / w1t * 100
    p["inst_net_early_share"] = p["mlb_net_early_share"] + p["big5_net_early_share"] + p["kamu_net_early_share"]
    p["mlb_buy_early_share"] = p["mlb_buy_early_tl"] / et * 100
    med_tl = p.groupby("symbol")["day_tl"].transform(lambda x: x.shift(1).rolling(20, min_periods=5).median())
    p["mlb_net_early_intensity"] = (p["mlb_buy_early_tl"] - p["mlb_sell_early_tl"]) / med_tl * 100
    p["mlb_net_yday_intensity"] = p.groupby("symbol").apply(lambda g: ((g["mlb_buy_day_tl"] - g["mlb_sell_day_tl"]) / med_tl.loc[g.index]).shift(1) * 100).reset_index(level=0, drop=True)
    # same-day price path (raw prices within the session; prev close restated to today's share basis)
    p["morning_ret"] = (p["p_1130"] / p["open_price"] - 1) * 100
    p["gap_ret"] = (p["open_price"] * p["adj_factor"] / p["prev_adj_close"] - 1) * 100
    p["placebo_noise"] = np.random.default_rng(5).normal(size=len(p))
    p["rest_1130"] = (p["close_price"] / p["p_1130"] - 1) * 100
    p["rest_1030"] = (p["close_price"] / p["p_1030"] - 1) * 100
    # pre-2023 sessions stay in the frame as warm-up for trailing statistics but are never graded (period = NaN,
    # which every groupby("period") drops)
    p["period"] = np.where(p["trade_date"] < EVAL_START, None, np.where(p["trade_date"] <= DISCOVERY_END, "discovery", "holdout"))
    feats = ["mlb_net_early_share", "big5_net_early_share", "kamu_net_early_share", "inst_net_early_share", "mlb_buy_early_share",
             "mlb_net_early_intensity", "mlb_net_yday_intensity", "morning_ret", "gap_ret", "mlb_net_w1_share", "placebo_noise"]
    return p, feats


def block_ci(v: np.ndarray, rng: np.random.Generator) -> tuple[float, float]:
    n = len(v)
    if n < 2 * BLOCK:
        return (np.nan, np.nan)
    starts = rng.integers(0, n - BLOCK + 1, size=(N_BOOT, int(np.ceil(n / BLOCK))))
    idx = (starts[:, :, None] + np.arange(BLOCK)).reshape(N_BOOT, -1)[:, :n]
    return tuple(np.percentile(v[idx].mean(axis=1), [2.5, 97.5]))


def ic_table(p: pd.DataFrame, feats: list[str], target: str) -> pd.DataFrame:
    rows = []
    for f in feats:
        r = {"feature": f}
        for period, g in p.groupby("period"):
            per = g.groupby("symbol").apply(lambda x: x[f].corr(x[target], method="spearman"))
            r[f"ic_{period}"] = per.mean()
            r[f"pos_stocks_{period}"] = int((per > 0).sum())
        r["same_sign_disc_hold"] = np.sign(r["ic_discovery"]) == np.sign(r["ic_holdout"])
        rows.append(r)
    return pd.DataFrame(rows).set_index("feature")


def selective(p: pd.DataFrame, score: str, target: str, q: float, rng: np.random.Generator, sign: float = 1.0) -> pd.DataFrame:
    """Call in the direction sign*score when the score is beyond its trailing-252 q / 1-q quantile (per stock)."""
    out = []
    p = p.sort_values(["symbol", "trade_date"])
    hi = p.groupby("symbol")[score].transform(lambda x: x.shift(1).rolling(252, min_periods=126).quantile(q))
    lo = p.groupby("symbol")[score].transform(lambda x: x.shift(1).rolling(252, min_periods=126).quantile(1 - q))
    call = sign * np.where(p[score] > hi, 1, np.where(p[score] < lo, -1, 0))
    p = p.assign(call=call)
    for period, g in p.groupby("period"):
        base_up = g.groupby("symbol")[target].transform(lambda v: (v > DB).mean())
        base_dn = g.groupby("symbol")[target].transform(lambda v: (v < -DB).mean())
        k = g["call"] != 0
        hit = (g.loc[k, "call"] * g.loc[k, target] > DB).astype(float)
        base = np.where(g.loc[k, "call"] > 0, base_up[k], base_dn[k])
        edge = hit.to_numpy() - base
        per_stock = pd.Series(edge, index=g.loc[k, "symbol"].to_numpy()).groupby(level=0).mean()
        lo_ci, hi_ci = block_ci(edge * 100, rng)
        out.append({"score": score, "target": target, "q": q, "period": period, "coverage_pct": k.mean() * 100, "n_calls": int(k.sum()),
                    "hit_pct": hit.mean() * 100, "base_pct": base.mean() * 100, "edge_pp": edge.mean() * 100, "edge_ci_lo": lo_ci, "edge_ci_hi": hi_ci,
                    "stocks_pos": int((per_stock > 0).sum()), "avg_ret_in_call_dir_pct": float((g.loc[k, "call"] * g.loc[k, target]).mean())})
    return pd.DataFrame(out)


def pooled_ridge(p: pd.DataFrame, feats: list[str], target: str) -> pd.DataFrame:
    """Pooled Ridge on per-stock trailing z-scores; refit monthly on rows strictly before the month (same-day target,
    so a row's label is realized at its own close)."""
    from sklearn.linear_model import Ridge

    p = p.sort_values(["symbol", "trade_date"]).copy()
    zc = []
    for f in feats:
        mu = p.groupby("symbol")[f].transform(lambda x: x.shift(1).rolling(252, min_periods=126).mean())
        sd = p.groupby("symbol")[f].transform(lambda x: x.shift(1).rolling(252, min_periods=126).std())
        p[f"z_{f}"] = ((p[f] - mu) / sd).clip(-5, 5)
        zc.append(f"z_{f}")
    p = p.dropna(subset=zc + [target])
    p["month"] = p["trade_date"].dt.to_period("M")
    preds = []
    for m in sorted(p["month"].unique()):
        tr = p[(p["trade_date"] < m.start_time) & (p["trade_date"] >= m.start_time - pd.DateOffset(years=2))]
        te = p[p["month"] == m]
        if len(tr) < 600 or te.empty:
            continue
        assert tr["trade_date"].max() < te["trade_date"].min()
        y = tr[target] - tr.groupby("symbol")[target].transform("mean")
        model = Ridge(alpha=10.0).fit(tr[zc], y)
        preds.append(te.assign(ridge_score=model.predict(te[zc])))
    return pd.concat(preds)


def combo_scores(p: pd.DataFrame) -> pd.DataFrame:
    """Session signal: z(yesterday MLB intensity) + z(morning move), trailing-252 z-scores per stock (known at 11:30)."""
    c = p.sort_values(["symbol", "trade_date"]).copy()
    for f in ("mlb_net_yday_intensity", "morning_ret"):
        mu = c.groupby("symbol")[f].transform(lambda x: x.shift(1).rolling(252, min_periods=126).mean())
        sd = c.groupby("symbol")[f].transform(lambda x: x.shift(1).rolling(252, min_periods=126).std())
        c[f"z_{f}"] = ((c[f] - mu) / sd).clip(-5, 5)
    c["combo_score"] = c["z_mlb_net_yday_intensity"] + c["z_morning_ret"]
    c["agree_score"] = np.where(np.sign(c["z_mlb_net_yday_intensity"]) == np.sign(c["z_morning_ret"]), c["combo_score"], 0.0)
    return c


def report(p: pd.DataFrame, feats: list[str]) -> None:
    pd.set_option("display.width", 250)
    rng = np.random.default_rng(3)
    # a 10:30 decision may only use what is known at 10:30 (W1 flow, the opening gap, yesterday)
    known_at = {"rest_1130": feats, "rest_1030": ["mlb_net_w1_share", "gap_ret", "mlb_net_yday_intensity", "placebo_noise"]}
    for target, tf in known_at.items():
        g = p.dropna(subset=[target])
        print(f"\n=== {target}: sessions {g['trade_date'].min().date()}..{g['trade_date'].max().date()}, n={len(g)}, "
              f"mean |move| {g[target].abs().mean():.2f}%, P(>|0.25|) {(g[target].abs() > DB).mean() * 100:.1f}% ===")
        print(ic_table(g, tf, target).round(3).to_string())

    sel = []
    for f in feats:
        if f == "mlb_net_w1_share":
            sel += [selective(p.dropna(subset=["rest_1030"]), f, "rest_1030", q, rng) for q in (0.8, 0.9)]
        else:
            sel += [selective(p.dropna(subset=["rest_1130"]), f, "rest_1130", q, rng) for q in (0.8, 0.9)]
    sel = pd.concat(sel, ignore_index=True)
    print("\n=== selective rule: call in the feature's direction beyond its trailing 80/20 or 90/10 quantile (no model) ===")
    print(sel.round(2).to_string(index=False))

    flow = ["mlb_net_early_share", "inst_net_early_share", "mlb_buy_early_share", "mlb_net_early_intensity", "morning_ret", "gap_ret"]
    pr = pooled_ridge(p, flow, "rest_1130")
    rs = pd.concat([selective(pr, "ridge_score", "rest_1130", q, rng) for q in (0.5, 0.8, 0.9)], ignore_index=True)
    rs.loc[rs["q"] == 0.5, "note"] = "q=0.5: call every day by sign vs trailing median"
    print("\n=== pooled Ridge (monthly refit, 2y window) on " + ", ".join(flow) + " ===")
    print(rs.round(2).to_string(index=False))
    sel.to_csv(OUT_DIR / "selective_rules.csv", index=False)
    rs.to_csv(OUT_DIR / "pooled_ridge.csv", index=False)

    # Combination chosen on DISCOVERY results only (edge CI excluding 0 in 2023-2024): yesterday's MLB intensity and
    # the morning move. Equal weights on trailing z-scores, nothing fitted; the holdout confirms or refutes.
    c = combo_scores(p.dropna(subset=["rest_1130"]))
    combo = pd.concat([selective(c, s, "rest_1130", q, rng) for s in ("combo_score", "agree_score") for q in (0.8, 0.9)], ignore_index=True)
    print("\n=== discovery-chosen combination: z(yesterday MLB intensity) + z(morning move), equal weights ===")
    print(combo.round(2).to_string(index=False))
    combo.to_csv(OUT_DIR / "combo.csv", index=False)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--report-only", action="store_true")
    args = ap.parse_args()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    panel = pd.read_parquet(OUT_DIR / "tick_panel.parquet") if args.report_only else build_panel()
    p, feats = engineer(panel)
    report(p, feats)
    print(f"\nwrote {OUT_DIR}", file=sys.stderr)


if __name__ == "__main__":
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    main()
