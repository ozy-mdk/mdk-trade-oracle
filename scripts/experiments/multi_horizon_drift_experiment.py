#!/usr/bin/env python
"""Drift treatment (a) and non-overlapping evaluation (c) for the 3D-30D swing models (isolated, read-only on the DB).

Question: is there anything beyond drift? Production demeans training targets (zero drift), so its models call UP
only ~40-45% of the time while the focus stocks rose in 55-60% of N-day windows, and "always UP" beats every crowned
champion. Fixed, never-selected configurations only (no tournament), each fitted point-in-time at every origin
(models/horizon_forecast.py):
    k0        - production: model on demeaned target                       pred = m
    k05       - half the trailing drift restored                            pred = m + 0.5 * mean(y_train)
    k1        - raw target (full trailing drift)                            pred = m + mean(y_train)
    drift     - drift only, no features                                     pred = mean(y_train)
    always_up / always_down - constant UP / DOWN calls. A forecast must beat BOTH in the same period (user rule
                2026-10-25): the edge is measured against the better of the two constants
All predictions go through production's ATR calibration and horizon bound, so directions are what the hub would show.

(c) N-day labels overlap: consecutive origins share N-1 sessions of outcome. Every metric is also computed on
non-overlapping origins (every N-th session, averaged over the N phase offsets) with an iid bootstrap over those dates
(stocks pooled per date), including the paired edge versus the better constant call (max of always-UP / always-DOWN,
taken per bootstrap sample, so the CI accounts for not knowing in advance which constant wins).

Discovery 2024 decides; holdout 2025-01 .. latest only confirms.

Outputs (under ~/data/mdk_oracle/experiments/multi_horizon_drift/): fits.parquet, summary.csv
Requires the feature frames of multi_horizon_ledger_rebuild.py (~/data/mdk_oracle/experiments/multi_horizon_ledger/frames).

Usage:
    .venv/bin/python scripts/experiments/multi_horizon_drift_experiment.py
    .venv/bin/python scripts/experiments/multi_horizon_drift_experiment.py --report-only
"""

from __future__ import annotations

import os

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[_v] = "1"

import argparse  # noqa: E402
import sys  # noqa: E402
import time  # noqa: E402
import warnings  # noqa: E402
from multiprocessing import Pool  # noqa: E402
from pathlib import Path  # noqa: E402

warnings.filterwarnings("ignore")

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

FOCUS = ["ASELS", "GARAN", "AKBNK", "YKBNK", "ISCTR", "TTKOM"]
FRAMES_DIR = Path.home() / "data" / "mdk_oracle" / "experiments" / "multi_horizon_ledger" / "frames"
OUT_DIR = Path.home() / "data" / "mdk_oracle" / "experiments" / "multi_horizon_drift"
HORIZONS = ["3d", "5d", "10d", "15d", "30d"]
HORIZON_DAYS = {"3d": 3, "5d": 5, "10d": 10, "15d": 15, "30d": 30}
FIXED_CONFIGS = [("Ridge", 252), ("BayesianRidge", 252), ("LightGBM", 252), ("Ridge", 504), ("LightGBM", 504)]
FEATURE_MODE = "FIFO_CORE"
FIRST_ORIGIN = "2024-01-01"
DISCOVERY_END = "2025-01-01"
KAPPAS = {"k0": 0.0, "k05": 0.5, "k1": 1.0}
N_BOOT = 2000


def deadband(h: str) -> float:
    from mdk_trading_oracle.data.silver.tertip_multi_horizon_forecaster import get_horizon_deadband

    return get_horizon_deadband(HORIZON_DAYS[h])


# ----------------------------------------------------------------------------------------------------------------------
# Build


def feature_sets(df: pd.DataFrame) -> dict[str, list[str]]:
    from mdk_trading_oracle.data.silver.tertip_multi_horizon_forecaster import filter_multi_horizon_features

    all_cols = [c for c in df.columns if c.startswith("feat_")]
    return {"full": filter_multi_horizon_features(all_cols, FEATURE_MODE)}


def run_task(args: tuple[str, str]) -> pd.DataFrame:
    sym, h = args
    from mdk_trading_oracle.models.horizon_forecast import fit_horizon, horizon_labels

    df = pd.read_parquet(FRAMES_DIR / f"{sym}.parquet").sort_values("trade_date").reset_index(drop=True)
    n = HORIZON_DAYS[h]
    db = deadband(h)
    y = horizon_labels(df, n).to_numpy(float)
    atr = df["atr_14d_pct"].to_numpy(float)
    dates = df["trade_date"].astype(str).str[:10].to_numpy()
    origins = [t for t in range(len(df) - n) if dates[t] >= FIRST_ORIGIN]
    rows = []
    for fs_name, cols in feature_sets(df).items():
        X = np.ascontiguousarray(df[cols].to_numpy(float))
        for model, lb in FIXED_CONFIGS:
            for t in origins:
                fit = fit_horizon(X, y, t, n, lb, model, db)
                if fit is None:
                    continue
                rows.append({
                    "symbol": sym, "horizon": h, "feature_set": fs_name, "config": f"{model}/{lb}",
                    "trade_date": dates[t], "target_date_end": dates[t + n], "actual_return_pct": y[t],
                    "atr_pct": atr[t], "demeaned_pred": fit.demeaned_pred, "train_mean": fit.train_mean, "n_train": fit.n_train,
                })
    return pd.DataFrame(rows)


def build(symbols: list[str], workers: int) -> pd.DataFrame:
    t0 = time.time()
    tasks = [(s, h) for h in HORIZONS[::-1] for s in symbols]
    with Pool(workers) as pool:
        fits = pd.concat(pool.map(run_task, tasks), ignore_index=True)
    fits.to_parquet(OUT_DIR / "fits.parquet")
    print(f"wrote {len(fits)} fits in {time.time() - t0:.0f}s", file=sys.stderr)
    return fits


# ----------------------------------------------------------------------------------------------------------------------
# (b) Lean MLB flow family: the only family with a consistent next-day IC (handoff §0.3 / §4.7). Price-level proxies
# (cost spreads, tertip shares, YTD PnL) are excluded. Each feature is z-scored per stock over its trailing 252
# sessions (row t uses rows <= t), so the six stocks can share one model.

LEAN_MLB = [
    "feat_mlb_turnover_intensity", "feat_bofa_aggression_ratio", "feat_mlb_ewma_flow_share_5d", "feat_mlb_buy_share_today",
    "feat_mlb_sell_share_today", "feat_mlb_w5_share", "feat_mlb_flow_3d_share", "feat_mlb_flow_5d_share",
    "feat_mlb_flow_10d_share", "feat_mlb_flow_divergence_3d_10d",
]
LEAN_LOOKBACKS = [252, 504]
Z_WINDOW = 252


def lean_frame(sym: str) -> pd.DataFrame:
    df = pd.read_parquet(FRAMES_DIR / f"{sym}.parquet").sort_values("trade_date").reset_index(drop=True)
    for c in LEAN_MLB:
        x = df[c].astype(float)
        mu = x.rolling(Z_WINDOW, min_periods=63).mean()
        sd = x.rolling(Z_WINDOW, min_periods=63).std().replace(0, np.nan)
        df[f"z_{c}"] = ((x - mu) / sd).clip(-4.0, 4.0).fillna(0.0)
    df["trade_date"] = df["trade_date"].astype(str).str[:10]
    return df


def run_lean_stock(args: tuple[str, str]) -> pd.DataFrame:
    """Per-stock lean MLB Ridge (production Ridge, point-in-time fit at every origin)."""
    sym, h = args
    from mdk_trading_oracle.models.horizon_forecast import fit_horizon, horizon_labels

    df = lean_frame(sym)
    n, db = HORIZON_DAYS[h], deadband(h)
    y = horizon_labels(df, n).to_numpy(float)
    X = np.ascontiguousarray(df[[f"z_{c}" for c in LEAN_MLB]].to_numpy(float))
    dates = df["trade_date"].to_numpy()
    rows = []
    for lb in LEAN_LOOKBACKS:
        for t in range(len(df) - n):
            if dates[t] < FIRST_ORIGIN:
                continue
            fit = fit_horizon(X, y, t, n, lb, "Ridge", db)
            if fit is None:
                continue
            rows.append({
                "symbol": sym, "horizon": h, "feature_set": "lean_mlb", "config": f"Ridge/{lb}", "trade_date": dates[t],
                "target_date_end": dates[t + n], "actual_return_pct": y[t], "atr_pct": df["atr_14d_pct"].iloc[t],
                "demeaned_pred": fit.demeaned_pred, "train_mean": fit.train_mean, "n_train": fit.n_train,
            })
    return pd.DataFrame(rows)


def run_lean_pooled(args: tuple[str, list[str]]) -> pd.DataFrame:
    """One lean MLB Ridge for all focus stocks, refitted at every origin date.

    Training rows at origin D: every stock's rows whose label is realized by the close of D (target_date_end <= D)
    within the last `lookback` sessions; labels demeaned per stock (each stock keeps its own trailing drift).
    """
    h, symbols = args
    from sklearn.linear_model import Ridge

    from mdk_trading_oracle.models.horizon_forecast import horizon_labels

    n = HORIZON_DAYS[h]
    zc = [f"z_{c}" for c in LEAN_MLB]
    parts = []
    for s in symbols:
        df = lean_frame(s)
        lab = horizon_labels(df, n)
        parts.append(pd.DataFrame({
            "symbol": s, "trade_date": df["trade_date"], "target_date_end": df["trade_date"].shift(-n),
            "y": lab, "atr_pct": df["atr_14d_pct"], **{c: df[c] for c in zc},
        }))
    panel = pd.concat(parts, ignore_index=True)
    grid = sorted(panel["trade_date"].unique())
    pos = {d: i for i, d in enumerate(grid)}
    panel["pos"] = panel["trade_date"].map(pos)
    X_all = panel[zc].to_numpy(float)
    y_all = panel["y"].to_numpy(float)
    rows = []
    for D in grid:
        if D < FIRST_ORIGIN:
            continue
        today = panel.index[(panel["trade_date"] == D) & panel["target_date_end"].notna()]
        if len(today) == 0:
            continue
        realized = (panel["target_date_end"] <= D).to_numpy() & np.isfinite(y_all)
        for lb in LEAN_LOOKBACKS:
            m = realized & (panel["pos"].to_numpy() > pos[D] - n - lb)
            if m.sum() < 6 * 63:
                continue
            assert panel.loc[m, "target_date_end"].max() <= D
            tr = panel.loc[m, ["symbol"]].assign(y=y_all[m])
            stock_mean = tr.groupby("symbol")["y"].mean()
            y_dm = y_all[m] - tr["symbol"].map(stock_mean).to_numpy()
            model = Ridge(alpha=2.0).fit(X_all[m], y_dm)
            pred = model.predict(X_all[today])
            for i, p in zip(today, pred):
                s = panel.at[i, "symbol"]
                rows.append({
                    "symbol": s, "horizon": h, "feature_set": "lean_mlb_pooled", "config": f"Ridge/{lb}", "trade_date": D,
                    "target_date_end": panel.at[i, "target_date_end"], "actual_return_pct": y_all[i], "atr_pct": panel.at[i, "atr_pct"],
                    "demeaned_pred": float(p), "train_mean": float(stock_mean.get(s, 0.0)), "n_train": int(m.sum()),
                })
    return pd.DataFrame(rows)


def build_lean(symbols: list[str], workers: int) -> pd.DataFrame:
    t0 = time.time()
    with Pool(workers) as pool:
        stock = pool.map(run_lean_stock, [(s, h) for h in HORIZONS for s in symbols])
        pooled = pool.map(run_lean_pooled, [(h, symbols) for h in HORIZONS])
    fits = pd.concat(stock + pooled, ignore_index=True)
    fits.to_parquet(OUT_DIR / "fits_lean.parquet")
    print(f"wrote {len(fits)} lean fits in {time.time() - t0:.0f}s", file=sys.stderr)
    return fits


# ----------------------------------------------------------------------------------------------------------------------
# Predictions per variant


def variants(fits: pd.DataFrame) -> pd.DataFrame:
    """Long frame: one row per (symbol, horizon, feature_set, config, variant, session) with the calibrated prediction."""
    from mdk_trading_oracle.models.horizon_forecast import calibrate_horizon

    out = []
    for h, g in fits.groupby("horizon"):
        n, db = HORIZON_DAYS[h], deadband(h)
        cal = np.vectorize(lambda raw, atr: calibrate_horizon(raw, atr, n, db))
        for name, k in KAPPAS.items():
            out.append(g.assign(variant=name, pred=cal(g["demeaned_pred"] + k * g["train_mean"], g["atr_pct"])))
        # drift-only and the constant calls do not depend on the model; take one config's rows
        base = g[(g["config"] == g["config"].iloc[0]) & (g["feature_set"] == g["feature_set"].iloc[0])]
        out.append(base.assign(variant="drift", config="-", feature_set="-", pred=cal(base["train_mean"], base["atr_pct"])))
        out.append(base.assign(variant="always_up", config="-", feature_set="-", pred=2.0 * db))
        out.append(base.assign(variant="always_down", config="-", feature_set="-", pred=-2.0 * db))
    v = pd.concat(out, ignore_index=True)
    v["period"] = np.where(v["trade_date"] < DISCOVERY_END, "discovery_2024", "holdout_2025_26")
    return v


# ----------------------------------------------------------------------------------------------------------------------
# Metrics


def hits(pred: np.ndarray, actual: np.ndarray, db: float) -> dict[str, np.ndarray]:
    near = np.abs(pred - actual) <= 0.25
    flat = np.abs(actual) <= db
    hub = near | np.where(flat, (np.abs(pred) <= db) | (pred * actual >= 0.0), np.where(actual > db, pred > 0.0, pred < 0.0))
    strict = ((pred > db) & (actual > db)) | ((pred < -db) & (actual < -db))
    return {"hub": hub, "sign": np.sign(pred) == np.sign(actual), "strict": strict}


def score(g: pd.DataFrame, up: pd.DataFrame, down: pd.DataFrame, h: str, rng: np.random.Generator) -> dict[str, float]:
    """All-sessions and non-overlapping metrics of one track `g`, with the paired edges vs always-UP / always-DOWN."""
    n, db = HORIZON_DAYS[h], deadband(h)
    m = g.merge(up[["symbol", "trade_date", "pred"]].rename(columns={"pred": "up_pred"}), on=["symbol", "trade_date"])
    m = m.merge(down[["symbol", "trade_date", "pred"]].rename(columns={"pred": "down_pred"}), on=["symbol", "trade_date"])
    a = m["actual_return_pct"].to_numpy()
    hm = hits(m["pred"].to_numpy(), a, db)
    hu, hd = hits(m["up_pred"].to_numpy(), a, db), hits(m["down_pred"].to_numpy(), a, db)
    m = m.assign(hub=hm["hub"], sign=hm["sign"], strict=hm["strict"], up_hub=hu["hub"], down_hub=hd["hub"])
    rec = {
        "n": len(m), "hub_hit_pct": m["hub"].mean() * 100, "sign_acc_pct": m["sign"].mean() * 100,
        "strict_pct": m["strict"].mean() * 100, "up_call_pct": (m["pred"] > db).mean() * 100,
        "mae_pct": np.abs(m["pred"] - m["actual_return_pct"]).mean(),
    }
    # non-overlapping: every n-th session of the common date grid, averaged over phase offsets
    grid = {d: i for i, d in enumerate(sorted(m["trade_date"].unique()))}
    m["phase"] = m["trade_date"].map(grid) % n
    per_date = m.groupby(["phase", "trade_date"]).agg(
        hub=("hub", "mean"), up=("up_hub", "mean"), down=("down_hub", "mean"), cnt=("hub", "size")
    )
    acc: dict[str, list[float]] = {k: [] for k in ("est", "lo", "hi", "e_up", "e_down", "e_best", "eb_lo", "eb_hi", "eu_lo", "ed_lo", "n")}
    for _, p in per_date.groupby(level="phase"):
        x, u, d, w = p["hub"].to_numpy(), p["up"].to_numpy(), p["down"].to_numpy(), p["cnt"].to_numpy(float)
        idx = rng.integers(0, len(x), size=(N_BOOT, len(x)))
        bw = w[idx]
        rate = (x[idx] * bw).sum(1) / bw.sum(1) * 100
        rate_u = (u[idx] * bw).sum(1) / bw.sum(1) * 100
        rate_d = (d[idx] * bw).sum(1) / bw.sum(1) * 100
        xr, ur, dr = (np.average(v, weights=w) * 100 for v in (x, u, d))
        acc["est"].append(xr)
        acc["lo"].append(np.percentile(rate, 2.5))
        acc["hi"].append(np.percentile(rate, 97.5))
        acc["e_up"].append(xr - ur)
        acc["e_down"].append(xr - dr)
        acc["e_best"].append(xr - max(ur, dr))
        best_b = rate - np.maximum(rate_u, rate_d)
        acc["eb_lo"].append(np.percentile(best_b, 2.5))
        acc["eb_hi"].append(np.percentile(best_b, 97.5))
        acc["eu_lo"].append(np.percentile(rate - rate_u, 2.5))
        acc["ed_lo"].append(np.percentile(rate - rate_d, 2.5))
        acc["n"].append(len(x))
    mean = {k: float(np.mean(v)) for k, v in acc.items()}
    rec.update({
        "nonovl_dates": mean["n"], "nonovl_hub_pct": mean["est"], "nonovl_ci_lo": mean["lo"], "nonovl_ci_hi": mean["hi"],
        "edge_vs_up_pp": mean["e_up"], "edge_up_ci_lo": mean["eu_lo"],
        "edge_vs_down_pp": mean["e_down"], "edge_down_ci_lo": mean["ed_lo"],
        "edge_vs_best_pp": mean["e_best"], "edge_best_ci_lo": mean["eb_lo"], "edge_best_ci_hi": mean["eb_hi"],
        "beats_both": bool(mean["eu_lo"] > 0 and mean["ed_lo"] > 0),
    })
    return rec


def summarize(v: pd.DataFrame) -> pd.DataFrame:
    rng = np.random.default_rng(11)
    rows = []
    for (period, h), g in v.groupby(["period", "horizon"]):
        up, down = g[g["variant"] == "always_up"], g[g["variant"] == "always_down"]
        for (fs, cfg, var), t in g.groupby(["feature_set", "config", "variant"]):
            rows.append({"period": period, "horizon": h, "feature_set": fs, "config": cfg, "variant": var, **score(t, up, down, h, rng)})
    s = pd.DataFrame(rows)
    s["horizon"] = pd.Categorical(s["horizon"], HORIZONS, ordered=True)
    return s.sort_values(["period", "horizon", "feature_set", "variant", "config"]).reset_index(drop=True)


# ----------------------------------------------------------------------------------------------------------------------
# Report


def pivot(s: pd.DataFrame, value: str, by: list[str]) -> pd.DataFrame:
    return s.pivot_table(index=by, columns="horizon", values=value, aggfunc="mean", observed=True).round(1)


def ic_table(fits: pd.DataFrame) -> pd.DataFrame:
    """Information beyond drift: Spearman IC of the demeaned forecast vs the demeaned outcome (actual - trailing mean).

    Pooled over stocks, non-overlapping origins (every N-th session, averaged over phase offsets); CI from an iid
    bootstrap over those dates (all stocks of a date resampled together).
    """
    rng = np.random.default_rng(5)
    rows = []
    f = fits.assign(period=np.where(fits["trade_date"] < DISCOVERY_END, "discovery_2024", "holdout_2025_26"))
    f["excess"] = f["actual_return_pct"] - f["train_mean"]
    for (period, h, fs, cfg), g in f.groupby(["period", "horizon", "feature_set", "config"]):
        n = HORIZON_DAYS[h]
        grid = {d: i for i, d in enumerate(sorted(g["trade_date"].unique()))}
        g = g.assign(phase=g["trade_date"].map(grid) % n)
        ics, los, his = [], [], []
        for _, p in g.groupby("phase"):
            dates = p["trade_date"].unique()
            by_date = {d: x for d, x in p.groupby("trade_date")}
            ic = p["demeaned_pred"].rank().corr(p["excess"].rank())
            boot = []
            for _ in range(300):
                s = pd.concat([by_date[d] for d in rng.choice(dates, size=len(dates))])
                boot.append(s["demeaned_pred"].rank().corr(s["excess"].rank()))
            ics.append(ic)
            los.append(np.nanpercentile(boot, 2.5))
            his.append(np.nanpercentile(boot, 97.5))
        rows.append({"period": period, "horizon": h, "feature_set": fs, "config": cfg, "ic": np.mean(ics), "ic_lo": np.mean(los), "ic_hi": np.mean(his)})
    s = pd.DataFrame(rows)
    s["horizon"] = pd.Categorical(s["horizon"], HORIZONS, ordered=True)
    s["cell"] = s.apply(lambda r: f"{r.ic:+.3f} [{r.ic_lo:+.2f},{r.ic_hi:+.2f}]", axis=1)
    return s


def report(fits: pd.DataFrame) -> None:
    pd.set_option("display.width", 250)
    pd.set_option("display.max_columns", 40)
    v = variants(fits)
    s = summarize(v)
    s.to_csv(OUT_DIR / "summary.csv", index=False)
    order = ["always_up", "always_down", "drift", "k0", "k05", "k1"]
    for period in ("discovery_2024", "holdout_2025_26"):
        p = s[s["period"] == period].copy()
        p["variant"] = pd.Categorical(p["variant"], order, ordered=True)
        print(f"\n################ {period} ################")
        print("\n-- hub hit %, all sessions (model variants averaged over the fixed configs) --")
        print(pivot(p, "hub_hit_pct", ["feature_set", "variant"]).to_string())
        print("\n-- share of UP calls % --")
        print(pivot(p, "up_call_pct", ["feature_set", "variant"]).to_string())
        print("\n-- non-overlapping hub hit % [95% CI] --")
        nv = p.groupby(["feature_set", "variant", "horizon"], observed=True)[["nonovl_hub_pct", "nonovl_ci_lo", "nonovl_ci_hi", "nonovl_dates"]].mean()
        print(nv.round(1).unstack("horizon").to_string())
        print("\n-- non-overlapping edge vs the better constant call (max of always-UP / always-DOWN), pp [95% CI]; * = beats both --")
        e = p[~p["variant"].isin(["always_up", "always_down"])]
        e = e.assign(cell=e.apply(
            lambda r: f"{r.edge_vs_best_pp:+.1f} [{r.edge_best_ci_lo:+.1f},{r.edge_best_ci_hi:+.1f}]{'*' if r.beats_both else ''}", axis=1
        ))
        print(e.pivot_table(index=["feature_set", "variant", "config"], columns="horizon", values="cell", aggfunc="first", observed=True).to_string())
        n_beat = int(e["beats_both"].sum())
        print(f"\nconfigs x horizons beating BOTH constant calls (both paired CIs above 0): {n_beat} of {len(e)}")
    ic = ic_table(fits)
    ic.to_csv(OUT_DIR / "ic_beyond_drift.csv", index=False)
    print("\n################ information beyond drift: rank IC of demeaned forecast vs demeaned outcome [95% CI] ################")
    print(ic.pivot_table(index=["period", "feature_set", "config"], columns="horizon", values="cell", aggfunc="first", observed=True).to_string())
    print(f"\nsummary -> {OUT_DIR / 'summary.csv'}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--symbols", default=",".join(FOCUS))
    ap.add_argument("--workers", type=int, default=16)
    ap.add_argument("--part", choices=["a", "b"], default="a", help="a: full features; b: add the lean MLB models")
    ap.add_argument("--report-only", action="store_true")
    args = ap.parse_args()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    symbols = [s.strip().upper() for s in args.symbols.split(",") if s.strip()]
    if args.report_only or args.part == "b":
        fits = pd.read_parquet(OUT_DIR / "fits.parquet")
    else:
        fits = build(symbols, args.workers)
    if args.part == "b":
        lean = pd.read_parquet(OUT_DIR / "fits_lean.parquet") if args.report_only else build_lean(symbols, args.workers)
        fits = pd.concat([fits, lean], ignore_index=True)
    report(fits)


if __name__ == "__main__":
    main()
