#!/usr/bin/env python
"""Feature diagnostics for the T+1 model on the focus stocks (isolated experiment, read-only on the database).

For every model feature (all feature modes) plus candidate intraday-window flow shares, measures:
    information  Spearman IC vs four targets of row d:
                     cc    adj close d   -> close d+1   (current T+1 target)
                     on    adj close d   -> open d+1    (overnight gap)
                     id    adj open d+1  -> close d+1   (next-session intraday)
                     same  adj close d-1 -> close d     (same-day price impact, not tradable; shows what flow "is")
    stability    IC per (stock, half-year); sign consistency and t-stat across those cells
    honesty      discovery period (2023-2024) vs holdout (2025-2026): does the discovery sign persist?
    quality      share of zeros / clip-bound values, lag-1 autocorrelation, level drift (2026 vs 2023 median, in IQRs)
    redundancy   |Spearman| between features, clusters at |rho| >= 0.8

Outputs (~/data/mdk_oracle/experiments/feature_diagnostics/): ic_cells.parquet, feature_table.csv, redundancy.csv

Usage:
    .venv/bin/python scripts/experiments/feature_diagnostics.py
    .venv/bin/python scripts/experiments/feature_diagnostics.py --report-only
"""

from __future__ import annotations

import argparse
import os
import warnings
from multiprocessing import Pool
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "1")
warnings.filterwarnings("ignore")

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

FOCUS = ["ASELS", "GARAN", "AKBNK", "YKBNK", "ISCTR", "TTKOM"]
OUT_DIR = Path.home() / "data" / "mdk_oracle" / "experiments" / "feature_diagnostics"
START = "2023-01-01"  # FIFO ledgers start flat in Jan 2022; skip the warm-up year
DISCOVERY_END = "2024-12-31"
TARGETS = ["cc", "on", "id", "same"]
WINDOWS = ["day_start", "first_reaction", "midday_followup", "afternoon_reaction", "closing_session"]
WIN_SHORT = {"day_start": "w1", "first_reaction": "w2", "midday_followup": "w3", "afternoon_reaction": "w4", "closing_session": "w5"}


def model_features() -> list[str]:
    from mdk_trading_oracle.data.silver.tertip_ml_forecaster import FEATURE_COLS, YTD_FEATURE_COLS

    return list(dict.fromkeys(list(FEATURE_COLS) + list(YTD_FEATURE_COLS)))


def window_flow_features(db, sym: str) -> pd.DataFrame:
    """Net flow per window and group in % of the day's total turnover (knowable at the close of d)."""
    from mdk_trading_oracle.data.silver.tertip_analytics import BIG_FIVE_BROKERS, KAMU_BROKERS

    w = db.query_df(
        "SELECT trade_date, broker_id, window_name, net_flow_tl FROM silver_intraday_broker_window_summary "
        "WHERE symbol = %s AND broker_id = ANY(%s)",
        (sym, ["MLB", *BIG_FIVE_BROKERS, *KAMU_BROKERS]),
    )
    grp = np.where(w["broker_id"] == "MLB", "mlb", np.where(w["broker_id"].isin(BIG_FIVE_BROKERS), "big5", "kamu"))
    w = w.assign(grp=grp, win=w["window_name"].map(WIN_SHORT))
    p = w.pivot_table(index="trade_date", columns=["grp", "win"], values="net_flow_tl", aggfunc="sum").fillna(0.0)
    p.columns = [f"cand_{g}_{wn}_flow_tl" for g, wn in p.columns]
    return p.reset_index()


def run_symbol(sym: str) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    from scipy.stats import spearmanr

    from mdk_trading_oracle.core.db import PostgresManager
    from mdk_trading_oracle.data.silver.tertip_ml_forecaster import adjusted_price, extract_3pillar_time_series

    db = PostgresManager(read_only=True)
    df = extract_3pillar_time_series(db, sym, lookback_days=5000).sort_values("trade_date").reset_index(drop=True)
    df["trade_date"] = pd.to_datetime(df["trade_date"])
    wf = window_flow_features(db, sym)
    wf["trade_date"] = pd.to_datetime(wf["trade_date"])
    df = df.merge(wf, on="trade_date", how="left")
    tt = df["total_turnover_tl"].replace(0, np.nan)
    cand = []
    for c in [c for c in df.columns if c.startswith("cand_") and c.endswith("_flow_tl")]:
        name = c.replace("_flow_tl", "_share")
        df[name] = (df[c].fillna(0.0) / tt * 100.0).clip(-100, 100)
        cand.append(name)
    for g in ["mlb", "big5", "kamu"]:  # early (W1+W2) vs late (W4+W5) flow
        df[f"cand_{g}_early_share"] = df[f"cand_{g}_w1_share"] + df[f"cand_{g}_w2_share"]
        df[f"cand_{g}_late_share"] = df[f"cand_{g}_w4_share"] + df[f"cand_{g}_w5_share"]
        cand += [f"cand_{g}_early_share", f"cand_{g}_late_share"]

    c, o = adjusted_price(df, "close"), adjusted_price(df, "open")
    # placebos: a persistent series with no information about future returns must score ~0 if the IC test is sound
    rng = np.random.default_rng(abs(hash(sym)) % 2**32)
    df["placebo_log_price"] = np.log(c)
    df["placebo_random_walk"] = np.cumsum(rng.normal(size=len(df)))
    df["placebo_iid_noise"] = rng.normal(size=len(df))
    cand += ["placebo_log_price", "placebo_random_walk", "placebo_iid_noise"]
    df["y_cc"] = (c.shift(-1) / c - 1) * 100
    df["y_on"] = (o.shift(-1) / c - 1) * 100
    df["y_id"] = (c.shift(-1) / o.shift(-1) - 1) * 100
    df["y_same"] = (c / c.shift(1) - 1) * 100

    feats = model_features() + cand
    feats = [f for f in feats if f in df.columns]
    d = df[df["trade_date"] >= START].copy()
    d["half"] = d["trade_date"].dt.year.astype(str) + "H" + np.where(d["trade_date"].dt.month <= 6, "1", "2")
    d["period"] = np.where(d["trade_date"] <= DISCOVERY_END, "discovery", "holdout")

    cells = []
    for f in feats:
        x = d[f]
        if x.nunique() < 3:
            continue
        for t in TARGETS:
            y = d[f"y_{t}"]
            for grp_col in ("half", "period"):
                for key, idx in d.groupby(grp_col).groups.items():
                    xs, ys = x.loc[idx], y.loc[idx]
                    ok = xs.notna() & ys.notna()
                    if ok.sum() < 40 or xs[ok].nunique() < 3:
                        continue
                    cells.append({"symbol": sym, "feature": f, "target": t, "cell_type": grp_col, "cell": key,
                                  "ic": spearmanr(xs[ok], ys[ok])[0], "n": int(ok.sum())})

    q = []
    for f in feats:
        x = d[f].astype(float)
        lo, hi = x.quantile(0.01), x.quantile(0.99)
        mode_share = x.round(6).value_counts(normalize=True).iloc[0] if x.notna().any() else np.nan
        iqr = (x.quantile(0.75) - x.quantile(0.25)) or np.nan
        m23 = x[d["trade_date"].dt.year == 2023].median()
        m26 = x[d["trade_date"].dt.year == 2026].median()
        q.append({"symbol": sym, "feature": f, "zero_share": float((x == 0).mean()), "mode_share": float(mode_share),
                  "at_extreme_share": float(((x <= x.min()) | (x >= x.max())).mean()) if x.max() > x.min() else 1.0,
                  "autocorr1": float(x.autocorr(1)), "drift_iqr": float(abs(m26 - m23) / iqr) if iqr == iqr else np.nan,
                  "p01": lo, "p99": hi})
    corr = d[feats].rank().corr().abs()
    corr["symbol"] = sym
    return pd.DataFrame(cells), pd.DataFrame(q), corr


def build() -> None:
    with Pool(6) as pool:
        res = pool.map(run_symbol, FOCUS)
    pd.concat([r[0] for r in res], ignore_index=True).to_parquet(OUT_DIR / "ic_cells.parquet")
    pd.concat([r[1] for r in res], ignore_index=True).to_parquet(OUT_DIR / "quality.parquet")
    corr = sum(r[2].drop(columns="symbol") for r in res) / len(res)
    corr.to_parquet(OUT_DIR / "corr.parquet")


def report() -> None:
    pd.set_option("display.width", 250)
    pd.set_option("display.max_rows", 200)
    cells = pd.read_parquet(OUT_DIR / "ic_cells.parquet")
    quality = pd.read_parquet(OUT_DIR / "quality.parquet").groupby("feature").mean(numeric_only=True)
    corr = pd.read_parquet(OUT_DIR / "corr.parquet")

    half = cells[cells["cell_type"] == "half"]
    per = cells[cells["cell_type"] == "period"].pivot_table(index=["feature", "target", "symbol"], columns="cell", values="ic").reset_index()

    rows = []
    for (f, t), g in half.groupby(["feature", "target"]):
        mean_ic = g["ic"].mean()
        tstat = mean_ic / (g["ic"].std(ddof=1) / np.sqrt(len(g))) if len(g) > 2 and g["ic"].std() > 0 else np.nan
        p = per[(per["feature"] == f) & (per["target"] == t)]
        disc, hold = p["discovery"].mean(), p["holdout"].mean()
        stocks_same_sign_hold = float((np.sign(p["holdout"]) == np.sign(p["discovery"])).mean()) if len(p) else np.nan
        rows.append({"feature": f, "target": t, "ic": mean_ic, "t_cells": tstat, "sign_consist": float((np.sign(g["ic"]) == np.sign(mean_ic)).mean()),
                     "ic_disc": disc, "ic_hold": hold, "hold_same_sign_stocks": stocks_same_sign_hold})
    tab = pd.DataFrame(rows)
    wide = tab.pivot_table(index="feature", columns="target", values=["ic", "t_cells", "sign_consist", "ic_disc", "ic_hold", "hold_same_sign_stocks"])
    wide.columns = [f"{a}_{b}" for a, b in wide.columns]
    wide = wide.join(quality[["zero_share", "mode_share", "autocorr1", "drift_iqr"]])
    # redundancy: strongest partner
    partner, prho = [], []
    for f in wide.index:
        if f in corr.index:
            s = corr.loc[f].drop(f, errors="ignore")
            partner.append(s.idxmax())
            prho.append(s.max())
        else:
            partner.append(None)
            prho.append(np.nan)
    wide["max_rho"] = prho
    wide["max_rho_with"] = partner
    wide["in_model"] = ~wide.index.str.startswith("cand_")
    wide.to_csv(OUT_DIR / "feature_table.csv")

    cols = ["in_model", "ic_same", "ic_cc", "t_cells_cc", "sign_consist_cc", "ic_disc_cc", "ic_hold_cc", "hold_same_sign_stocks_cc",
            "ic_on", "t_cells_on", "ic_id", "t_cells_id", "zero_share", "autocorr1", "drift_iqr", "max_rho", "max_rho_with"]
    print("=== all features, sorted by |t| of next-day close-to-close IC (cells = 6 stocks x 8 half-years) ===")
    print(wide.reindex(wide["t_cells_cc"].abs().sort_values(ascending=False).index)[cols].round(3).to_string())

    for t, label in (("on", "overnight gap"), ("id", "next-session open->close")):
        print(f"\n=== top 15 by |t| for {label} ===")
        print(wide.reindex(wide[f"t_cells_{t}"].abs().sort_values(ascending=False).index)[[f"ic_{t}", f"t_cells_{t}", f"sign_consist_{t}", f"ic_disc_{t}", f"ic_hold_{t}", f"hold_same_sign_stocks_{t}"]].head(15).round(3).to_string())

    model = wide[wide["in_model"]]
    print("\n=== redundancy clusters among model features (|rho| >= 0.8) ===")
    seen = set()
    for f in model.index:
        if f in seen or f not in corr.index:
            continue
        members = [g for g in corr.columns if g in model.index and corr.loc[f, g] >= 0.8]
        if len(members) > 1:
            seen.update(members)
            print(" ~ ", ", ".join(members))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--report-only", action="store_true")
    args = ap.parse_args()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    if not args.report_only:
        build()
    report()


if __name__ == "__main__":
    main()
