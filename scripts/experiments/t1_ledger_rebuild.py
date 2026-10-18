#!/usr/bin/env python
"""Clean vs leaky T+1 walk-forward ledger for the focus stocks (isolated experiment, read-only on the database).

For every session T since --start, the forecast graded for T is made at origin T-1:
    clean - model fitted at origin T-1 (labels realized by the T-1 close), predicting from row T-1
    leaky - production's reconstructed ledger row: model fitted at origin T (its training set holds row T-1 with
            label r(T-1 -> T), the graded outcome), predicting from row T-1
Configurations: the crowned config (config/tertip_crowned_models.yaml) and a fixed, never-selected Ridge/252/FIFO_CORE.
Each forecast is graded both as the ATR-calibrated model output (what the ledger has always stored) and as the
volatility-sized forecast that is published (direction identical to the model output, never inside the deadband).

Outputs (under ~/data/mdk_oracle/experiments/t1_ledger/):
    ledger.parquet   one row per (symbol, config, mode, session)
    summary.csv      hit rates, MAE, random baseline, bootstrap CIs

Usage:
    .venv/bin/python scripts/experiments/t1_ledger_rebuild.py                    # build + report
    .venv/bin/python scripts/experiments/t1_ledger_rebuild.py --report-only      # report from ledger.parquet
"""

from __future__ import annotations

import argparse
import os
import sys
import warnings
from multiprocessing import Pool
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "1")
warnings.filterwarnings("ignore")

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import yaml  # noqa: E402

FOCUS = ["ASELS", "GARAN", "AKBNK", "YKBNK", "ISCTR", "TTKOM"]
OUT_DIR = Path.home() / "data" / "mdk_oracle" / "experiments" / "t1_ledger"
CROWNED_PATH = Path("config/tertip_crowned_models.yaml")
SELECTION_WINDOW_SESSIONS = 60  # crowning window of config/tertip_crowned_models.yaml (eval_window 60d)
BIG_MOVE_PCT = 2.0
N_PERM = 300
N_BOOT = 1000
BLOCK = 10


def configs_for(sym: str):
    from mdk_trading_oracle.models.t1_forecast import T1Config

    cfg = yaml.safe_load(open(CROWNED_PATH))
    selected_on = str(cfg.get("_metadata", {}).get("calibration_date", "")) or None
    return [
        T1Config.from_crowned(cfg["symbols"][sym], selected_on=selected_on),
        T1Config("Ridge", 252, "FIFO_CORE", label="fixed_ridge"),
    ]


def run_symbol(args: tuple[str, str]) -> pd.DataFrame:
    sym, start = args
    from mdk_trading_oracle.core.db import PostgresManager
    from mdk_trading_oracle.data.silver.tertip_ml_forecaster import (
        DEADBAND_PCT,
        _check_hit,
        extract_3pillar_time_series,
        get_feature_cols_for_mode,
    )
    from mdk_trading_oracle.models.stochastic_risk import fit_volatility_state, load_adjusted_log_returns, size_forecast
    from mdk_trading_oracle.models.t1_forecast import (
        direction,
        fit_t1_model,
        predict_t1_row,
        prepare_feature_frame,
        strict_hit,
        t1_labels,
    )

    db = PostgresManager(read_only=True)
    raw = extract_3pillar_time_series(db, sym, lookback_days=5000)
    log_ret = load_adjusted_log_returns(db, sym)
    cfgs = configs_for(sym)
    all_cols = sorted({c for cfg in cfgs for c in get_feature_cols_for_mode(cfg.feature_mode)})
    df = prepare_feature_frame(raw, all_cols)
    labels = t1_labels(df)
    dates = df["trade_date"].astype(str).str[:10].tolist()
    graded_idx = [i for i in range(1, len(df)) if dates[i] >= start]

    vol_states: dict[int, object] = {}

    def vol_state(origin_idx: int):
        if origin_idx not in vol_states:
            vol_states[origin_idx] = fit_volatility_state(log_ret.loc[: dates[origin_idx]], dates[origin_idx])
        return vol_states[origin_idx]

    rows = []
    for cfg in cfgs:
        f_cols = get_feature_cols_for_mode(cfg.feature_mode)
        sel_window_start = None
        if cfg.selected_on:
            before = [d for d in dates if d < cfg.selected_on]
            sel_window_start = before[-SELECTION_WINDOW_SESSIONS] if len(before) >= SELECTION_WINDOW_SESSIONS else before[0]
        fits: dict[int, object] = {}

        def fit_at(origin_idx: int):
            if origin_idx not in fits:
                fits[origin_idx] = fit_t1_model(df, origin_idx, cfg, f_cols, labels)
            return fits[origin_idx]

        for t in graded_idx:
            origin = t - 1
            actual = float(labels.iloc[origin])
            if not np.isfinite(actual):
                continue
            if cfg.selected_on is None:
                status = "NEVER_SELECTED"
            elif dates[t] >= cfg.selected_on:
                status = "POST_SELECTION"
            elif dates[t] >= sel_window_start:
                status = "IN_SELECTION_WINDOW"
            else:
                status = "PRE_SELECTION"
            for mode, fit_origin in (("clean", origin), ("leaky", t)):
                fit = fit_at(fit_origin)
                if fit is None:
                    continue
                if fit_origin == origin:
                    assert fit.train_label_end_date <= dates[origin] < dates[t]
                pred = predict_t1_row(fit, df, origin, f_cols)
                if pred is None:
                    continue
                model_ret = pred["model_return_pct"]
                sized = size_forecast(model_ret, vol_state(origin), 1, seed_key=f"{dates[origin]}:1d")
                published = sized["expected_return_pct"] if sized else model_ret
                rec = {
                    "symbol": sym, "config": cfg.label, "model_spec": cfg.spec, "mode": mode,
                    "trade_date": dates[t], "origin_date": dates[origin], "train_label_end_date": fit.train_label_end_date,
                    "selection_status": status, "actual_return_pct": actual,
                    "raw_return_pct": pred["raw_return_pct"], "n_train": fit.n_train,
                }
                for kind, p in (("model", model_ret), ("published", published)):
                    rec[f"{kind}_pred_pct"] = p
                    rec[f"{kind}_direction"] = direction(p, DEADBAND_PCT)
                    rec[f"{kind}_hub_hit"] = bool(_check_hit(p, actual))
                    rec[f"{kind}_strict"] = strict_hit(p, actual, DEADBAND_PCT)
                    rec[f"{kind}_called"] = abs(p) > DEADBAND_PCT
                    rec[f"{kind}_abs_err"] = abs(p - actual)
                rows.append(rec)
    return pd.DataFrame(rows)


# ----------------------------------------------------------------------------------------------------------------------
# Reporting


def metrics(pred: np.ndarray, actual: np.ndarray, deadband: float = 0.25) -> dict[str, float]:
    from mdk_trading_oracle.data.silver.tertip_ml_forecaster import _check_hit

    called = np.abs(pred) > deadband
    strict = ((pred > deadband) & (actual > deadband)) | ((pred < -deadband) & (actual < -deadband))
    hub = np.array([_check_hit(p, a) for p, a in zip(pred, actual)])
    sign_ok = np.sign(pred) == np.sign(actual)
    big = np.abs(actual) >= BIG_MOVE_PCT
    err = np.abs(pred - actual)
    return {
        "hub_hit_pct": hub.mean() * 100,
        "strict_pct": strict.mean() * 100,
        "strict_on_calls_pct": strict[called].mean() * 100 if called.any() else np.nan,
        "big_move_sign_pct": sign_ok[big].mean() * 100 if big.any() else np.nan,
        "mae_pct": err.mean(),
        "hit_mae_pct": err[hub].mean() if hub.any() else np.nan,
        "miss_mae_pct": err[~hub].mean() if (~hub).any() else np.nan,
    }


def block_bootstrap_ci(values: np.ndarray, rng: np.random.Generator) -> tuple[float, float]:
    n = len(values)
    if n < BLOCK * 2:
        return (np.nan, np.nan)
    n_blocks = int(np.ceil(n / BLOCK))
    starts = rng.integers(0, n - BLOCK + 1, size=(N_BOOT, n_blocks))
    idx = (starts[:, :, None] + np.arange(BLOCK)).reshape(N_BOOT, -1)[:, :n]
    means = values[idx].mean(axis=1)
    return tuple(np.percentile(means, [2.5, 97.5]))


def summarize(d: pd.DataFrame, kind: str) -> pd.DataFrame:
    rng = np.random.default_rng(7)
    out = []
    for (sym, cfg, mode), g in d.groupby(["symbol", "config", "mode"]):
        pred, actual = g[f"{kind}_pred_pct"].to_numpy(), g["actual_return_pct"].to_numpy()
        m = metrics(pred, actual)
        perm = [metrics(rng.permutation(pred), actual) for _ in range(N_PERM)]
        strict = g[f"{kind}_strict"].to_numpy(dtype=float) * 100
        lo, hi = block_bootstrap_ci(strict, rng)
        out.append({
            "symbol": sym, "config": cfg, "mode": mode, "n": len(g), "spec": g["model_spec"].iloc[0], **m,
            "strict_ci_lo": lo, "strict_ci_hi": hi,
            "random_strict_pct": np.mean([p["strict_pct"] for p in perm]),
            "random_hub_hit_pct": np.mean([p["hub_hit_pct"] for p in perm]),
            "random_mae_pct": np.mean([p["mae_pct"] for p in perm]),
        })
    return pd.DataFrame(out)


def side_by_side(s: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    w = s.pivot_table(index=["config", "symbol"], columns="mode", values=cols)
    w.columns = [f"{c}_{m}" for c, m in w.columns]
    order = [f"{c}_{m}" for c in cols for m in ("leaky", "clean")]
    return w[[c for c in order if c in w.columns]]


def pooled(d: pd.DataFrame, kind: str) -> pd.DataFrame:
    d = d.assign(symbol="ALL6")
    return summarize(d, kind)


def report(d: pd.DataFrame) -> None:
    pd.set_option("display.width", 250)
    pd.set_option("display.max_columns", 40)
    frames = []
    for kind in ("model", "published"):
        s = pd.concat([summarize(d, kind), pooled(d, kind)], ignore_index=True).assign(forecast=kind)
        frames.append(s)
        print(f"\n=== {kind.upper()} forecast ({'ATR-calibrated model output, as stored in the ledger' if kind == 'model' else 'volatility-sized, as published'}) ===")
        print(f"sessions {d['trade_date'].min()} .. {d['trade_date'].max()}")
        print(side_by_side(s, ["hub_hit_pct", "strict_pct", "big_move_sign_pct", "mae_pct", "hit_mae_pct", "miss_mae_pct"]).round(2).to_string())
        clean = s[s["mode"] == "clean"].set_index(["config", "symbol"])
        print("\nclean vs random (strict % with 95% block-bootstrap CI; random = predictions shuffled across sessions)")
        print(clean[["n", "strict_pct", "strict_ci_lo", "strict_ci_hi", "random_strict_pct", "hub_hit_pct", "random_hub_hit_pct", "mae_pct", "random_mae_pct"]].round(1).to_string())
    pd.concat(frames, ignore_index=True).to_csv(OUT_DIR / "summary.csv", index=False)

    crowned = d[d["config"] == "crowned"]
    print("\n=== crowned config by selection status (model forecast, ALL6 pooled) ===")
    by = crowned.groupby(["selection_status", "mode"]).apply(
        lambda g: pd.Series({**metrics(g["model_pred_pct"].to_numpy(), g["actual_return_pct"].to_numpy()), "n": len(g)})
    )
    print(by[["n", "hub_hit_pct", "strict_pct", "big_move_sign_pct", "mae_pct"]].round(2).to_string())

    last = sorted(d["trade_date"].unique())[-30:]
    print("\n=== last 30 sessions (hub window), crowned, model forecast ===")
    w = crowned[crowned["trade_date"].isin(last)]
    print(side_by_side(summarize(pd.concat([w, w.assign(symbol="ALL6")]), "model"), ["hub_hit_pct", "strict_pct", "big_move_sign_pct", "mae_pct"]).round(1).to_string())


def compare_with_production(d: pd.DataFrame) -> None:
    """Agreement of this experiment's leaky/clean directions with the production ledger rows written nightly."""
    from mdk_trading_oracle.core.db import PostgresManager

    db = PostgresManager(read_only=True)
    prod = db.query_df(
        "SELECT symbol, trade_date::text AS trade_date, ml_pred_return_pct, ml_direction, calculated_at::date::text AS calc "
        "FROM gold_tertip_walk_forward_backtests WHERE symbol IN (%s)" % ",".join(f"'{s}'" for s in FOCUS)
    )
    m = d[d["config"] == "crowned"].merge(prod, on=["symbol", "trade_date"])
    if m.empty:
        return
    m["agree"] = m["model_direction"] == m["ml_direction"]
    print("\n=== direction agreement with the production ledger (crowned) — by production row origin ===")
    m["prod_origin"] = np.where(m["calc"] >= "2026-10-13", "nightly append (>=10-13)", "tournament/export (<10-13)")
    print(m.pivot_table(index="prod_origin", columns="mode", values="agree", aggfunc="mean").mul(100).round(1).to_string())


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--symbols", default=",".join(FOCUS))
    ap.add_argument("--start", default="2024-01-01")
    ap.add_argument("--report-only", action="store_true")
    args = ap.parse_args()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    path = OUT_DIR / "ledger.parquet"
    if not args.report_only:
        syms = [s.strip().upper() for s in args.symbols.split(",") if s.strip()]
        with Pool(min(6, len(syms))) as pool:
            d = pd.concat(pool.map(run_symbol, [(s, args.start) for s in syms]), ignore_index=True)
        d.to_parquet(path)
        print(f"wrote {len(d)} rows -> {path}", file=sys.stderr)
    d = pd.read_parquet(path)
    report(d)
    compare_with_production(d)


if __name__ == "__main__":
    main()
