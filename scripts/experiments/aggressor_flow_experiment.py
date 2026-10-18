#!/usr/bin/env python
"""Step 3.4 — aggressor-side order flow from the ticks (isolated experiment, read-only).

Tick rule per (symbol, day) in tick order over the continuous session [10:00, 18:00): a print above the previous
price is buyer-initiated (+1), below is seller-initiated (-1), unchanged inherits the last sign. Auction prints
(< 10:00 opening auction, >= 18:00 closing auction / trading at last price) stay unclassified.
MLB trades split into aggressive (MLB initiated: MLB buyer on +1, MLB seller on -1) and passive (MLB's resting order
filled: MLB buyer on -1, MLB seller on +1).

Hypotheses (decided on discovery 2023-2024, holdout 2025-2026 read afterwards):
    H1  morning market order-flow imbalance (OFI) predicts the rest of the session
    H2  MLB aggressive morning flow predicts continuation; passive flow does not / reverts
    H3  yesterday's MLB signal is carried by its aggressive part (also checked vs next close-to-close)
Validation first: same-window correlations of OFI / MLB aggressive / MLB passive with the morning return.

Usage:
    .venv/bin/python scripts/experiments/aggressor_flow_experiment.py              # scan ticks + report
    .venv/bin/python scripts/experiments/aggressor_flow_experiment.py --report-only
"""

from __future__ import annotations

import argparse
import glob
import sys
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import polars as pl  # noqa: E402

sys.path.insert(0, str(Path(__file__).parent))
from intraday_flow_experiment import ARCHIVE, DB, FOCUS, OUT_DIR, engineer  # noqa: E402

WINDOW = 252


def scan_symbol(sym: str, recent: pd.DataFrame) -> pd.DataFrame:
    cols = ["timestamp", "price", "volume", "buyer_broker_id", "seller_broker_id"]
    t = pl.scan_parquet(sorted(glob.glob(ARCHIVE))).filter(pl.col("symbol") == sym).select(cols).collect()
    r = recent[recent["symbol"] == sym][cols]
    if len(r):
        t = pl.concat([t, pl.from_pandas(r).with_columns(pl.col("timestamp").cast(pl.Datetime("us")), pl.col("price").cast(pl.Float64), pl.col("volume").cast(pl.Float64))], how="vertical_relaxed")
    t = t.with_row_index("seq").sort(["timestamp", "seq"])  # stable: keeps feed order within the same second
    t = t.with_columns(
        pl.col("timestamp").dt.date().alias("trade_date"),
        pl.col("timestamp").dt.time().alias("t"),
        (pl.col("price") * pl.col("volume")).alias("tl"),
    )
    cont = (pl.col("t") >= pl.time(10, 0)) & (pl.col("t") < pl.time(18, 0))
    t = t.with_columns(cont.alias("cont"))
    # tick sign within each day on continuous prints only
    c = t.filter(pl.col("cont")).with_columns(
        (pl.col("price") - pl.col("price").shift(1).over("trade_date")).sign().alias("d")
    ).with_columns(pl.when(pl.col("d") == 0).then(None).otherwise(pl.col("d")).alias("d"))
    c = c.with_columns(pl.col("d").forward_fill().over("trade_date").fill_null(0).alias("side"))
    mlb_b, mlb_s = pl.col("buyer_broker_id") == "MLB", pl.col("seller_broker_id") == "MLB"
    buy_i, sell_i = pl.col("side") == 1, pl.col("side") == -1
    early, late = pl.col("t") < pl.time(11, 30), pl.col("t") >= pl.time(11, 30)

    def block(mask: pl.Expr, tag: str) -> list[pl.Expr]:
        return [
            pl.col("tl").filter(mask).sum().alias(f"cont_tl_{tag}"),
            pl.col("tl").filter(mask & buy_i).sum().alias(f"buyinit_tl_{tag}"),
            pl.col("tl").filter(mask & sell_i).sum().alias(f"sellinit_tl_{tag}"),
            pl.col("tl").filter(mask & mlb_b & buy_i).sum().alias(f"mlb_aggr_buy_{tag}"),
            pl.col("tl").filter(mask & mlb_s & sell_i).sum().alias(f"mlb_aggr_sell_{tag}"),
            pl.col("tl").filter(mask & mlb_b & sell_i).sum().alias(f"mlb_pass_buy_{tag}"),
            pl.col("tl").filter(mask & mlb_s & buy_i).sum().alias(f"mlb_pass_sell_{tag}"),
        ]

    agg = c.group_by("trade_date").agg(block(early, "early") + block(late, "late") + block(pl.lit(True), "day")).sort("trade_date")
    out = agg.to_pandas()
    out["symbol"] = sym
    out["trade_date"] = pd.to_datetime(out["trade_date"])
    return out


def build() -> pd.DataFrame:
    from mdk_trading_oracle.core.db import PostgresManager

    arch_max = pl.scan_parquet(sorted(glob.glob(ARCHIVE))).select(pl.col("timestamp").max()).collect().item()
    recent = PostgresManager(read_only=True).query_df(
        "SELECT timestamp, symbol, price, volume, buyer_broker_id, seller_broker_id FROM bronze_raw_trades "
        "WHERE symbol = ANY(%s) AND timestamp > %s ORDER BY timestamp",
        (FOCUS, arch_max),
    )
    frames = []
    for s in FOCUS:
        frames.append(scan_symbol(s, recent))
        print(f"scanned {s}", file=sys.stderr)
    a = pd.concat(frames, ignore_index=True)
    a.to_parquet(OUT_DIR / "aggressor_panel.parquet")
    return a


def features(a: pd.DataFrame) -> pd.DataFrame:
    p, _ = engineer(pd.read_parquet(OUT_DIR / "tick_panel.parquet"))
    p = p.merge(a, on=["symbol", "trade_date"], how="inner").sort_values(["symbol", "trade_date"]).reset_index(drop=True)
    for tag in ("early", "late", "day"):
        tot = p[f"cont_tl_{tag}"].replace(0, np.nan)
        p[f"ofi_{tag}"] = (p[f"buyinit_tl_{tag}"] - p[f"sellinit_tl_{tag}"]) / tot * 100
        p[f"mlb_aggr_{tag}"] = (p[f"mlb_aggr_buy_{tag}"] - p[f"mlb_aggr_sell_{tag}"]) / tot * 100
        p[f"mlb_pass_{tag}"] = (p[f"mlb_pass_buy_{tag}"] - p[f"mlb_pass_sell_{tag}"]) / tot * 100
        p[f"mlb_aggr_share_{tag}"] = (p[f"mlb_aggr_buy_{tag}"] + p[f"mlb_aggr_sell_{tag}"]) / (
            p[f"mlb_aggr_buy_{tag}"] + p[f"mlb_aggr_sell_{tag}"] + p[f"mlb_pass_buy_{tag}"] + p[f"mlb_pass_sell_{tag}"]).replace(0, np.nan) * 100
    med = p.groupby("symbol")["day_tl"].transform(lambda x: x.shift(1).rolling(20, min_periods=5).median())
    for kind in ("aggr", "pass"):
        net_day = p[f"mlb_{kind}_buy_day"] - p[f"mlb_{kind}_sell_day"]
        p[f"mlb_{kind}_yday_intensity"] = (net_day / med * 100).groupby(p["symbol"]).shift(1)
    p["ofi_yday"] = p.groupby("symbol")["ofi_day"].shift(1)
    p["late_ret"] = (p["close_price"] / p["p_1130"] - 1) * 100  # = rest_1130; same-window check for late flows
    p["next_cc"] = (p.groupby("symbol")["adj_close_price"].shift(-1) / p["adj_close_price"] - 1) * 100
    return p


def ic(p: pd.DataFrame, f: str, target: str) -> dict[str, float]:
    r = {}
    for per, g in p.dropna(subset=["period"]).groupby("period"):
        ics = g.groupby("symbol").apply(lambda x: x[f].corr(x[target], method="spearman"))
        r[f"ic_{per}"], r[f"pos_{per}"] = ics.mean(), int((ics > 0).sum())
    return r


def zscore(p: pd.DataFrame, f: str) -> pd.Series:
    g = p.groupby("symbol")[f]
    mu = g.transform(lambda x: x.shift(1).rolling(WINDOW, min_periods=126).mean())
    sd = g.transform(lambda x: x.shift(1).rolling(WINDOW, min_periods=126).std())
    return ((p[f] - mu) / sd).clip(-5, 5)


def selective_stats(p: pd.DataFrame, score: pd.Series, target: str, q: float = 0.8) -> list[dict[str, float]]:
    s = pd.Series(score, index=p.index)
    g = s.groupby(p["symbol"])
    hi = g.transform(lambda x: x.shift(1).rolling(WINDOW, min_periods=126).quantile(q))
    lo = g.transform(lambda x: x.shift(1).rolling(WINDOW, min_periods=126).quantile(1 - q))
    call = pd.Series(np.where(s > hi, 1, np.where(s < lo, -1, 0)), index=p.index)
    rows = []
    for per in ("discovery", "holdout"):
        m = (p["period"] == per) & p[target].notna()
        k = m & (call != 0)
        y = p.loc[k, target]
        sign_ok = np.sign(call[k]) == np.sign(y)
        base = np.where(call[k] > 0, (p.loc[m, target] > DB).mean(), (p.loc[m, target] < -DB).mean())
        per_stock = pd.Series(sign_ok.to_numpy(), index=p.loc[k, "symbol"].to_numpy()).groupby(level=0).mean()
        pnl = call[k] * y
        # rolling 60-session windows: share with >= 60% sign accuracy (>= 5 calls)
        ok = pd.Series(np.where(call != 0, (np.sign(call) == np.sign(p[target])).astype(float), np.nan), index=p.index)[m]
        win = []
        for _, gs in ok.groupby(p.loc[m, "symbol"]):
            hits, n = gs.fillna(0).rolling(60).sum(), gs.notna().astype(int).rolling(60).sum()
            a = (hits / n)[n >= 5].dropna()
            win.append(a)
        win = pd.concat(win) if win else pd.Series(dtype=float)
        rows.append({"period": per, "coverage_pct": k.sum() / m.sum() * 100, "sign_acc_pct": sign_ok.mean() * 100,
                     "edge_pp": ((call[k] * y > DB).mean() - base.mean()) * 100, "mean_ret_pct": pnl.mean(),
                     "t_stat": pnl.mean() / pnl.std(ddof=1) * np.sqrt(len(pnl)), "stocks_acc_gt_50": int((per_stock > 0.5).sum()),
                     "windows_ge_60_pct": (win >= 0.6).mean() * 100 if len(win) else np.nan})
    return rows


def report(p: pd.DataFrame) -> None:
    pd.set_option("display.width", 250)
    v = p[p["period"].notna()]
    print("=== validation: same-window Spearman with the window's own return (all stocks, all periods) ===")
    rows = []
    for tag, ret in (("early", "morning_ret"), ("late", "late_ret")):
        for f in (f"ofi_{tag}", f"mlb_aggr_{tag}", f"mlb_pass_{tag}"):
            rows.append({"window": tag, "feature": f, "rho_same_window": v.groupby("symbol").apply(lambda x: x[f].corr(x[ret], method="spearman")).mean()})
    print(pd.DataFrame(rows).round(3).to_string(index=False))
    share = v.groupby("symbol")["mlb_aggr_share_day"].median()
    print("median share of MLB volume executed aggressively:", ", ".join(f"{s} {x:.0f}%" for s, x in share.items()))

    print("\n=== H1-H3 standalone ICs ===")
    tests = [("H1", "ofi_early", "rest_1130"), ("H2", "mlb_aggr_early", "rest_1130"), ("H2", "mlb_pass_early", "rest_1130"),
             ("ref", "mlb_net_early_share", "rest_1130"), ("H3", "mlb_aggr_yday_intensity", "rest_1130"), ("H3", "mlb_pass_yday_intensity", "rest_1130"),
             ("ref", "mlb_net_yday_intensity", "rest_1130"), ("H3", "ofi_yday", "rest_1130"),
             ("H3-T+1", "mlb_aggr_day", "next_cc"), ("H3-T+1", "mlb_pass_day", "next_cc"), ("H3-T+1", "ofi_day", "next_cc")]
    print(pd.DataFrame([{"hyp": h, "feature": f, "target": t, **ic(p, f, t)} for h, f, t in tests]).round(3).to_string(index=False))

    print("\n=== combined session score variants (target rest_1130, call beyond trailing 80/20) ===")
    z = {f: zscore(p, f) for f in ("mlb_net_yday_intensity", "morning_ret", "mlb_aggr_yday_intensity", "mlb_pass_yday_intensity", "ofi_early", "mlb_aggr_early", "mlb_pass_early")}
    variants = {
        "base: yday MLB + morning": z["mlb_net_yday_intensity"] + z["morning_ret"],
        "yday MLB aggr + morning": z["mlb_aggr_yday_intensity"] + z["morning_ret"],
        "yday MLB aggr - pass + morning": z["mlb_aggr_yday_intensity"] - z["mlb_pass_yday_intensity"] + z["morning_ret"],
        "base + OFI early": z["mlb_net_yday_intensity"] + z["morning_ret"] + z["ofi_early"],
        "base + MLB aggr early": z["mlb_net_yday_intensity"] + z["morning_ret"] + z["mlb_aggr_early"],
        "base - MLB pass early": z["mlb_net_yday_intensity"] + z["morning_ret"] - z["mlb_pass_early"],
    }
    rows = []
    for name, sc in variants.items():
        for r in selective_stats(p, sc, "rest_1130"):
            rows.append({"variant": name, **r})
    res = pd.DataFrame(rows)
    res.to_csv(OUT_DIR / "aggressor_variants.csv", index=False)
    for per in ("discovery", "holdout"):
        print(f"\n--- {per} ---")
        print(res[res["period"] == per].drop(columns="period").round(2).to_string(index=False))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--report-only", action="store_true")
    args = ap.parse_args()
    a = pd.read_parquet(OUT_DIR / "aggressor_panel.parquet") if args.report_only else build()
    report(features(a))


if __name__ == "__main__":
    main()
