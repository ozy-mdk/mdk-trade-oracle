#!/usr/bin/env python
"""Experiment: new tournament candidates vs the current five, per stock, same protocol.

Candidates (per-stock models, same features / lookbacks / sample weights / calibration as production):
    Current : Ridge, XGBoost, LightGBM, BayesianRidge, Huber
    New     : SVR (epsilon = deadband), ARD, ExtraTrees, CatBoost, LGBM_Mono (monotone institutional flow)

Protocols
    Multi-horizon (3D-30D): production `evaluate_horizon_walk_forward` over 180 sessions with the crowned
        per-stock lookback / feature mode; new models are injected through `_build_model`.
    T+1: exact replica of `run_30d_walk_forward_arena` (targets, weights, clipping, calibration, hit rule) over
        120 sessions with the crowned per-stock lookback / feature mode.

Honest selection test (mirrors daily re-crowning, zero look-ahead):
    For each evaluation session s in the latter half, the champion is the pool member with the lowest production
    composite loss over sessions whose outcomes were already realized at s (T+1: trailing 30 sessions;
    multi-horizon: trailing 90 realized sessions). Its prediction at s is scored. Pools: CURRENT, NEW, ALL.

Usage:
    .venv/bin/python scripts/experiments/model_candidate_arena.py [--symbols AKBNK,GARAN] [--workers 14]
"""

from __future__ import annotations

import argparse
import os

os.environ.setdefault("OMP_NUM_THREADS", "1")

import math  # noqa: E402
import time  # noqa: E402
import warnings  # noqa: E402
from multiprocessing import Pool  # noqa: E402
from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import yaml  # noqa: E402

warnings.filterwarnings("ignore")

CURRENT = ["Ridge", "XGBoost", "LightGBM", "BayesianRidge", "Huber"]
NEW = ["SVR", "ARD", "ExtraTrees", "CatBoost", "LGBM_Mono"]
POOLS = {"CURRENT": CURRENT, "NEW": ["Ridge", "LightGBM"] + NEW, "ALL": CURRENT + NEW}
MONOTONE_POSITIVE = {"feat_total_inst_net_share_today", "feat_bofa_aggression_ratio"}

MH_EVAL_SESSIONS = 180
T1_EVAL_SESSIONS = 120
OUT_DIR = Path(os.environ.get("ARENA_OUT", Path.home() / "data" / "mdk_oracle" / "experiments" / "model_candidate_arena"))


# ── Candidate wrappers (fit(X, y, sample_weight) / predict) ───────────────────


class _Scaled:
    """StandardScaler + estimator that forwards sample weights when supported."""

    def __init__(self, est, weighted_lsq: bool = False):
        from sklearn.preprocessing import StandardScaler

        self.scaler = StandardScaler()
        self.est = est
        self.weighted_lsq = weighted_lsq
        self.offset = 0.0

    def fit(self, X, y, sample_weight=None):
        X = self.scaler.fit_transform(np.asarray(X, dtype=np.float64))
        X = np.nan_to_num(X)
        y = np.asarray(y, dtype=np.float64)
        if sample_weight is None:
            sample_weight = np.ones(len(y))
        if self.weighted_lsq:
            # Weighted least squares via sqrt-weight row scaling (estimator lacks sample_weight)
            w = np.asarray(sample_weight, dtype=np.float64)
            xm = np.average(X, axis=0, weights=w)
            self.offset = float(np.average(y, weights=w))
            sw = np.sqrt(w)[:, None]
            self.xm = xm
            self.est.fit((X - xm) * sw, (y - self.offset) * sw[:, 0])
        else:
            self.est.fit(X, y, sample_weight=sample_weight)
        return self

    def predict(self, X):
        X = np.nan_to_num(self.scaler.transform(np.asarray(X, dtype=np.float64)))
        if self.weighted_lsq:
            return self.est.predict(X - self.xm) + self.offset
        return self.est.predict(X)


def build_new_model(name: str, deadband: float, feature_cols: list[str]):
    if name == "SVR":
        from sklearn.svm import SVR

        return _Scaled(SVR(kernel="rbf", C=1.0, epsilon=deadband, gamma="scale"))
    if name == "ARD":
        from sklearn.linear_model import ARDRegression

        return _Scaled(ARDRegression(fit_intercept=False, max_iter=300), weighted_lsq=True)
    if name == "ExtraTrees":
        from sklearn.ensemble import ExtraTreesRegressor

        return ExtraTreesRegressor(n_estimators=150, min_samples_leaf=5, max_features=0.5, n_jobs=1, random_state=42)
    if name == "CatBoost":
        from catboost import CatBoostRegressor

        return CatBoostRegressor(
            iterations=150, depth=4, learning_rate=0.05, l2_leaf_reg=5.0, random_seed=42, verbose=0, thread_count=1,
            allow_writing_files=False,
        )
    if name == "LGBM_Mono":
        import lightgbm as lgb

        return lgb.LGBMRegressor(
            n_estimators=50, max_depth=3, learning_rate=0.05, num_leaves=11, subsample=0.8, colsample_bytree=0.8,
            random_state=42, n_jobs=1, verbosity=-1, max_bin=64,
            monotone_constraints=[1 if c in MONOTONE_POSITIVE else 0 for c in feature_cols],
        )
    raise ValueError(name)


def build_t1_current(name: str):
    """Exact replicas of the inline T+1 arena candidates."""
    import lightgbm as lgb
    from sklearn.linear_model import BayesianRidge, HuberRegressor, Ridge
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler
    from xgboost import XGBRegressor

    if name == "Ridge":
        return Ridge(alpha=2.0, random_state=42), "sample_weight"
    if name == "XGBoost":
        return XGBRegressor(n_estimators=60, max_depth=3, learning_rate=0.05, subsample=0.8, colsample_bytree=0.8,
                            random_state=42, n_jobs=1), "sample_weight"
    if name == "LightGBM":
        return lgb.LGBMRegressor(n_estimators=60, max_depth=3, learning_rate=0.05, num_leaves=11, subsample=0.8,
                                 colsample_bytree=0.8, random_state=42, n_jobs=1, verbose=-1), "sample_weight"
    if name == "Huber":
        return Pipeline([("scaler", StandardScaler()), ("regressor", HuberRegressor(epsilon=1.35, alpha=1.0, max_iter=300))]), "regressor__sample_weight"
    if name == "BayesianRidge":
        return Pipeline([("scaler", StandardScaler()), ("regressor", BayesianRidge(max_iter=300))]), "regressor__sample_weight"
    raise ValueError(name)


# ── Per-symbol worker ─────────────────────────────────────────────────────────


def _run_t1(df: pd.DataFrame, f_cols: list[str], train_lb: int) -> list[dict]:
    from mdk_trading_oracle.data.silver.tertip_ml_forecaster import DEADBAND_PCT, _check_hit, calibrate_expected_return

    for col in f_cols:
        if col not in df.columns:
            df[col] = 0.0
    rows = []
    for idx in range(len(df) - T1_EVAL_SESSIONS, len(df)):
        train = df.iloc[:idx]
        if len(train) < 63:
            continue
        n_tr = min(len(train), train_lb)
        test_row, prev_row = df.iloc[idx], df.iloc[idx - 1]
        actual_price, prev_price = float(test_row["close_price"]), float(prev_row["close_price"])
        actual_ret = (actual_price - prev_price) / prev_price * 100.0
        prev_atr = float(prev_row.get("atr_14d_pct", 2.2))
        tr = train.iloc[-n_tr:].dropna(subset=f_cols)
        y = ((tr["close_price"].shift(-1) - tr["close_price"]) / tr["close_price"] * 100.0).iloc[:-1]
        X = tr[f_cols].iloc[:-1]
        X_prev = pd.DataFrame([prev_row[f_cols]])
        w = np.where(y.abs() < 1.0, 0.50, np.where(y.abs() < 2.0, 3.50, 6.00))
        for name in CURRENT + NEW:
            try:
                if name in CURRENT:
                    model, wkey = build_t1_current(name)
                    model.fit(X, y, **{wkey: w})
                else:
                    model = build_new_model(name, DEADBAND_PCT, f_cols)
                    model.fit(X.to_numpy(dtype=np.float64), y.to_numpy(), sample_weight=w)
                raw = float(model.predict(X_prev.to_numpy(dtype=np.float64) if name in NEW else X_prev)[0])
                if not math.isfinite(raw):
                    raw = 0.0
            except Exception:
                raw = 0.0
            pred = calibrate_expected_return(max(-10.0, min(10.0, raw)), prev_atr)
            pred_price = prev_price * (1.0 + pred / 100.0)
            rows.append({
                "horizon": "1d", "n_days": 1, "deadband": DEADBAND_PCT, "eval_idx": idx,
                "trade_date": str(test_row["trade_date"])[:10], "model": name, "pred": pred, "actual": actual_ret,
                "hit": bool(_check_hit(pred, actual_ret)), "err": abs(pred_price - actual_price) / actual_price * 100.0,
            })
    return rows


def _run_mh(df: pd.DataFrame, h_key: str, cfg: dict) -> list[dict]:
    from mdk_trading_oracle.data.silver import tertip_multi_horizon_forecaster as tmh

    n_days = tmh.HORIZON_MAP[h_key]
    deadband = tmh.get_horizon_deadband(n_days)
    lb = int(cfg.get("training_lookback_sessions", 126))
    f_mode = cfg.get("feature_mode", "FIFO_CORE")
    _, f_cols = tmh.engineer_multi_horizon_features(df.copy(), feature_mode=f_mode)
    original = tmh._build_model
    rows = []
    for name in CURRENT + NEW:
        if name in NEW:
            tmh._build_model = lambda _n, seed=42, _name=name: build_new_model(_name, deadband, f_cols)
        try:
            res = tmh.evaluate_horizon_walk_forward(df.copy(), h_key, name, lb, MH_EVAL_SESSIONS, f_mode)
        finally:
            tmh._build_model = original
        for i, r in enumerate(res.get("ledger", [])):
            rows.append({
                "horizon": h_key, "n_days": n_days, "deadband": deadband, "eval_idx": i,
                "trade_date": r["trade_date"], "model": name, "pred": r["pred_return_pct"],
                "actual": r["actual_return_pct"], "hit": bool(r["is_hit"]), "err": r["err_pct"],
            })
    return rows


def run_symbol(sym: str) -> pd.DataFrame:
    from mdk_trading_oracle.core.db import PostgresManager
    from mdk_trading_oracle.data.silver.tertip_ml_forecaster import (
        extract_3pillar_time_series,
        get_feature_cols_for_mode,
    )

    t0 = time.time()
    db = PostgresManager(read_only=True)
    t1_cfg = yaml.safe_load(open("config/tertip_crowned_models.yaml"))["symbols"].get(sym, {})
    mh_cfg = yaml.safe_load(open("config/tertip_multi_horizon_crowned_models.yaml"))["symbols"].get(sym, {})
    rows: list[dict] = []

    df_mh = extract_3pillar_time_series(db, sym, lookback_days=550)
    for h_key in ["3d", "5d", "10d", "15d", "30d"]:
        if h_key in mh_cfg and len(df_mh) >= 120:
            rows += _run_mh(df_mh, h_key, mh_cfg[h_key])

    if t1_cfg:
        lb = int(t1_cfg.get("training_lookback_sessions", 252))
        df_t1 = extract_3pillar_time_series(db, sym, lookback_days=int((lb + T1_EVAL_SESSIONS + 60) * 1.5))
        if len(df_t1) >= T1_EVAL_SESSIONS + 63:
            rows += _run_t1(df_t1, get_feature_cols_for_mode(t1_cfg.get("feature_mode", "FIFO_CORE")), lb)

    out = pd.DataFrame(rows)
    out["symbol"] = sym
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out.to_parquet(OUT_DIR / f"{sym}.parquet", index=False)
    print(f"  {sym}: {len(out)} predictions in {time.time() - t0:.0f}s", flush=True)
    return out


# ── Analysis ──────────────────────────────────────────────────────────────────


def composite_loss(g: pd.DataFrame) -> float:
    """Production 3-criteria loss with linear recency weights."""
    n = len(g)
    if n == 0:
        return float("inf")
    w = np.linspace(0.5, 1.5, n) if n > 1 else np.ones(1)
    w = w / w.mean()
    hit = g["hit"].to_numpy(dtype=float)
    err = g["err"].to_numpy(dtype=float)
    whr = float(np.sum(hit * w) / np.sum(w) * 100.0)
    mae = float(np.average(err, weights=w))
    hmae = float(np.average(err[hit == 1], weights=w[hit == 1])) if hit.any() else mae
    mmae = float(np.average(err[hit == 0], weights=w[hit == 0])) if (hit == 0).any() else mae
    return (100.0 - whr) + hmae + 2.5 * mmae


def strict_dir(df: pd.DataFrame) -> pd.Series:
    """Sign accuracy on sessions that moved beyond the deadband."""
    moved = df["actual"].abs() > df["deadband"]
    return (np.sign(df["pred"]) == np.sign(df["actual"]))[moved]


def unconditional_table(res: pd.DataFrame) -> pd.DataFrame:
    out = []
    for (h, m), g in res.groupby(["horizon", "model"]):
        per_sym = g.groupby("symbol")
        out.append({
            "horizon": h, "model": m, "hit_pct": g["hit"].mean() * 100, "strict_dir_pct": strict_dir(g).mean() * 100,
            "mae_pct": g["err"].mean(), "loss": per_sym.apply(composite_loss).mean(),
            "corr": g[["pred", "actual"]].corr().iloc[0, 1],
        })
    return pd.DataFrame(out)


def selection_table(res: pd.DataFrame) -> pd.DataFrame:
    """Rolling zero-look-ahead re-crowning within each (symbol, horizon); score champions on the latter half."""
    out = []
    for (sym, h), g in res.groupby(["symbol", "horizon"]):
        n_days = int(g["n_days"].iloc[0])
        window = 30 if h == "1d" else 90
        piv = {m: gm.set_index("eval_idx").sort_index() for m, gm in g.groupby("model")}
        idxs = sorted(g["eval_idx"].unique())
        eval_idxs = idxs[len(idxs) // 2:]
        for pool, members in POOLS.items():
            members = [m for m in members if m in piv]
            for s in eval_idxs:
                realized_end = s - n_days  # outcomes known at s
                hist = [i for i in idxs if realized_end - window < i <= realized_end]
                if len(hist) < 10:
                    continue
                champ = min(members, key=lambda m: composite_loss(piv[m].loc[hist]))
                row = piv[champ].loc[s]
                out.append({"symbol": sym, "horizon": h, "pool": pool, "eval_idx": s, "champion": champ,
                            "hit": row["hit"], "err": row["err"], "pred": row["pred"], "actual": row["actual"],
                            "deadband": row["deadband"]})
    return pd.DataFrame(out)


HORIZON_ORDER = ["1d", "3d", "5d", "10d", "15d", "30d"]


def report(res: pd.DataFrame) -> str:
    lines = []
    unc = unconditional_table(res)
    lines.append("## Per-model performance (all stocks, every evaluation session, no selection)\n")
    for metric, label in [("hit_pct", "Hit rate % (production rule)"), ("strict_dir_pct", "Strict direction % (moves beyond deadband)"),
                          ("loss", "Composite loss (mean per stock, lower = better)"), ("corr", "Corr(pred, actual)")]:
        t = unc.pivot(index="model", columns="horizon", values=metric).reindex(columns=HORIZON_ORDER)
        t = t.reindex(CURRENT + NEW)
        lines.append(f"### {label}\n\n{t.round(3 if metric == 'corr' else 1).to_string()}\n")

    sel = selection_table(res)
    lines.append("## Honest re-crowning test (champion picked only from already-realized sessions)\n")
    rows = []
    for (h, pool), g in sel.groupby(["horizon", "pool"]):
        rows.append({"horizon": h, "pool": pool, "hit_pct": g["hit"].mean() * 100,
                     "strict_dir_pct": strict_dir(g).mean() * 100, "mae_pct": g["err"].mean(), "n": len(g)})
    t = pd.DataFrame(rows)
    for metric in ["hit_pct", "strict_dir_pct", "mae_pct"]:
        lines.append(f"### {metric}\n\n{t.pivot(index='pool', columns='horizon', values=metric).reindex(columns=HORIZON_ORDER).round(2).to_string()}\n")
    share = sel[sel.pool == "ALL"].groupby(["horizon", "champion"]).size().unstack(0).fillna(0)
    share = (share / share.sum() * 100).reindex(columns=HORIZON_ORDER).round(1)
    lines.append(f"### Champion share in the ALL pool (% of sessions)\n\n{share.to_string()}\n")
    return "\n".join(lines)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--symbols", default="")
    ap.add_argument("--workers", type=int, default=14)
    ap.add_argument("--report-only", action="store_true")
    args = ap.parse_args()

    if args.report_only:
        res = pd.concat([pd.read_parquet(p) for p in sorted(OUT_DIR.glob("*.parquet"))], ignore_index=True)
    else:
        if args.symbols:
            symbols = [s.strip().upper() for s in args.symbols.split(",") if s.strip()]
        else:
            cfg = yaml.safe_load(open("config/tertip_multi_horizon_crowned_models.yaml"))["symbols"]
            symbols = sorted(cfg.keys())
        print(f"Running {len(symbols)} symbols with {args.workers} workers -> {OUT_DIR}", flush=True)
        with Pool(processes=min(args.workers, len(symbols))) as pool:
            res = pd.concat(pool.map(run_symbol, symbols, chunksize=1), ignore_index=True)

    md = report(res)
    (OUT_DIR / "REPORT.md").write_text(md, encoding="utf-8")
    print(md)


if __name__ == "__main__":
    main()
