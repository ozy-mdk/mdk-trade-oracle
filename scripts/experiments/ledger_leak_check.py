#!/usr/bin/env python
"""Quantify the look-ahead leak in the gold T+1 walk-forward ledger.

`update_gold_tertip_forecasts` scores session T by predicting from row T-1 with a model whose training set includes
row T-1 and its label r(T-1 -> T), i.e. the outcome being graded. This script re-scores the last 30 sessions per
stock with the crowned configuration twice:
    leaky - production training window (row T-1 and its label included)
    clean - row T-1 excluded (what was knowable at the T-1 close)

Usage:
    .venv/bin/python scripts/experiments/ledger_leak_check.py ~/data/mdk_oracle/experiments/ledger_leak_check.parquet
"""

from __future__ import annotations

import os
import sys
import warnings
from multiprocessing import Pool

os.environ.setdefault("OMP_NUM_THREADS", "1")
warnings.filterwarnings("ignore")

import lightgbm as lgb  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import yaml  # noqa: E402
from sklearn.linear_model import BayesianRidge, HuberRegressor, Ridge  # noqa: E402
from sklearn.pipeline import Pipeline  # noqa: E402
from sklearn.preprocessing import StandardScaler  # noqa: E402
from xgboost import XGBRegressor  # noqa: E402

N_SESSIONS = 30


def build(name: str):
    """Exact candidates used by `data/gold/tertip_forecasts.py`; returns (model, sample-weight fit key)."""
    if name == "XGBoost":
        model = XGBRegressor(n_estimators=45, max_depth=2, learning_rate=0.03, subsample=0.8, colsample_bytree=0.8, random_state=42, n_jobs=1)
        return model, "sample_weight"
    if name == "LightGBM":
        model = lgb.LGBMRegressor(n_estimators=45, max_depth=3, learning_rate=0.03, num_leaves=7, subsample=0.8, colsample_bytree=0.8, random_state=42, n_jobs=1, verbose=-1)
        return model, "sample_weight"
    if name == "BayesianRidge":
        return Pipeline([("scaler", StandardScaler()), ("regressor", BayesianRidge(max_iter=300))]), "regressor__sample_weight"
    if name == "Huber":
        return Pipeline([("scaler", StandardScaler()), ("regressor", HuberRegressor(epsilon=1.35, alpha=1.0, max_iter=300))]), "regressor__sample_weight"
    return Ridge(alpha=2.0, random_state=42), "sample_weight"


def run(sym: str) -> pd.DataFrame:
    from mdk_trading_oracle.core.db import PostgresManager
    from mdk_trading_oracle.data.silver.tertip_ml_forecaster import (
        _check_hit,
        adjusted_price,
        calibrate_expected_return,
        extract_3pillar_time_series,
        get_feature_cols_for_mode,
    )

    cfg = yaml.safe_load(open("config/tertip_crowned_models.yaml"))["symbols"][sym]
    lb = int(cfg["training_lookback_sessions"])
    f_cols = get_feature_cols_for_mode(cfg.get("feature_mode", "FIFO_CORE"))
    df = extract_3pillar_time_series(PostgresManager(read_only=True), sym, lookback_days=int((lb + 90) * 1.5))
    for c in f_cols:
        if c not in df.columns:
            df[c] = 0.0
    df = df.reset_index(drop=True)
    nxt = adjusted_price(df, "close").pct_change().shift(-1) * 100.0  # label of row t = return t -> t+1

    rows = []
    for T in range(len(df) - N_SESSIONS, len(df)):  # realized session T, ledger scored at the T close
        tr = df.iloc[: T + 1].iloc[-lb:].dropna(subset=f_cols)  # production: df.iloc[-train_lb:]
        y = nxt.loc[tr.index]
        actual = float(nxt.iloc[T - 1])
        for mode, X_idx in [("leaky", tr.index[:-1]), ("clean", tr.index[:-2])]:
            yy = y.loc[X_idx]
            w = np.where(yy.abs() >= 2.0, 6.0, np.where(yy.abs() >= 1.0, 3.5, 0.5))
            model, key = build(cfg["model"])
            model.fit(tr.loc[X_idx, f_cols], yy, **{key: w})
            raw = max(-10.0, min(10.0, float(model.predict(df.loc[[T - 1], f_cols])[0])))
            pred = calibrate_expected_return(raw, float(df["atr_14d_pct"].iloc[T - 1]))
            rows.append({
                "symbol": sym, "model": cfg["model"], "mode": mode, "date": str(df["trade_date"].iloc[T])[:10],
                "lenient": bool(_check_hit(pred, actual)),
                "strict": (pred > 0.25 and actual > 0.25) or (pred < -0.25 and actual < -0.25),
                "called": abs(pred) > 0.25, "big": abs(actual) >= 2.0, "sign_ok": bool(np.sign(pred) == np.sign(actual)),
            })
    return pd.DataFrame(rows)


def main() -> None:
    out = sys.argv[1] if len(sys.argv) > 1 else os.path.expanduser("~/data/mdk_oracle/experiments/ledger_leak_check.parquet")
    syms = sorted(yaml.safe_load(open("config/tertip_crowned_models.yaml"))["symbols"])
    with Pool(8) as pool:
        d = pd.concat(pool.map(run, syms), ignore_index=True)
    d.to_parquet(out)
    summary = d.groupby("mode").apply(lambda g: pd.Series({
        "lenient_hit_pct": g["lenient"].mean() * 100,
        "strict_on_calls_pct": g.loc[g["called"], "strict"].mean() * 100,
        "big_move_sign_pct": g.loc[g["big"], "sign_ok"].mean() * 100,
    }))
    print(summary.round(1).to_string())
    print(d.groupby(["model", "mode"])["lenient"].mean().unstack().mul(100).round(1).to_string())


if __name__ == "__main__":
    main()
