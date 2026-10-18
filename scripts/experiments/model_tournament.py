#!/usr/bin/env python
"""Model tournament: feature sets x model families x training windows, daily walk-forward (isolated, read-only).

Every candidate forecasts a % change for every session. For session D it is refit on sessions strictly before D
(their targets are realized by the close of D-1), so no future data enters any forecast.

Forecast types (separate tournaments):
    preopen   target = % change from D's opening auction price to D's closing auction price; features known at the
              close of D-1 (yesterday's flows)
    s1130     target = % change from the last print before 11:30 on D to D's close; same features + D's morning move
              (open -> 11:30), which the trader sees live
Feature sets (yesterday's values, row = session D):
    LEAN_MLB   MLB buy share, MLB net intensity, total institutional net share, MLB late (W4+W5) net share
    LEAN_INST  LEAN_MLB + KAMU / BIG5 net intensity + KAMU / BIG5 closing-session shares
    OLD_39     the 39 production FIFO_CORE features
Models (production settings): Ridge, BayesianRidge, Huber, XGBoost, LightGBM; baselines: ZERO (always 0%),
MEAN (the stock's mean move over the training window).
Windows: 63, 126, 252, 504 sessions.

Crowning (per stock and forecast type): best direction hit rate (MAE tie-break) over the VALIDATION window = the 60
sessions before the last 60. The champion is then shown on the TEST window = the last 60 sessions. For contrast the
report also shows the champion crowned on the test window itself (in-sample; this is what inflated the old hub).

Usage:
    .venv/bin/python scripts/experiments/model_tournament.py            # run (~10 min, 6 processes) + report
    .venv/bin/python scripts/experiments/model_tournament.py --report-only
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

sys.path.insert(0, str(Path(__file__).parent))

FOCUS = ["ASELS", "GARAN", "AKBNK", "YKBNK", "ISCTR", "TTKOM"]
OUT = Path.home() / "data" / "mdk_oracle" / "experiments" / "model_tournament"
WINDOWS = (63, 126, 252, 504)
MODELS = ("Ridge", "BayesianRidge", "Huber", "XGBoost", "LightGBM", "ZERO", "MEAN")
EVAL_SESSIONS, TEST_SESSIONS = 120, 60
MISSING_RAW_DAYS = pd.to_datetime(["2026-09-25", "2026-10-01", "2026-10-02", "2026-10-05"])
LEAN_MLB = ["feat_mlb_buy_share_today", "feat_mlb_turnover_intensity", "feat_total_inst_net_share_today", "mlb_late_share"]
LEAN_INST = LEAN_MLB + ["feat_kamu_turnover_intensity", "feat_big5_turnover_intensity", "feat_kamu_w5_share", "feat_big5_w5_share"]


def make_model(name: str):
    import lightgbm as lgb
    from sklearn.linear_model import BayesianRidge, HuberRegressor, Ridge
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler
    from xgboost import XGBRegressor

    if name == "XGBoost":
        return XGBRegressor(n_estimators=45, max_depth=2, learning_rate=0.03, subsample=0.8, colsample_bytree=0.8, random_state=42, n_jobs=1), "sample_weight"
    if name == "LightGBM":
        return lgb.LGBMRegressor(n_estimators=45, max_depth=3, learning_rate=0.03, num_leaves=7, subsample=0.8, colsample_bytree=0.8,
                                 random_state=42, n_jobs=1, verbose=-1), "sample_weight"
    if name == "BayesianRidge":
        return Pipeline([("sc", StandardScaler()), ("m", BayesianRidge(max_iter=300))]), "m__sample_weight"
    if name == "Huber":
        return Pipeline([("sc", StandardScaler()), ("m", HuberRegressor(epsilon=1.35, alpha=1.0, max_iter=300))]), "m__sample_weight"
    return Pipeline([("sc", StandardScaler()), ("m", Ridge(alpha=2.0))]), "m__sample_weight"


def weights(y: np.ndarray) -> np.ndarray:
    a = np.abs(y)
    return np.where(a >= 2.0, 6.0, np.where(a >= 1.0, 3.5, 0.5))  # production T+1 weighting


def stock_frame(sym: str) -> pd.DataFrame:
    """Row = session D: yesterday's features (shifted one session), D's morning move and both targets."""
    from feature_diagnostics import window_flow_features
    from intraday_flow_experiment import OUT_DIR, engineer

    from mdk_trading_oracle.core.db import PostgresManager
    from mdk_trading_oracle.data.silver.tertip_ml_forecaster import FEATURE_COLS, extract_3pillar_time_series

    db = PostgresManager(read_only=True)
    df = extract_3pillar_time_series(db, sym, lookback_days=5000).sort_values("trade_date").reset_index(drop=True)
    df["trade_date"] = pd.to_datetime(df["trade_date"])
    wf = window_flow_features(db, sym)
    wf["trade_date"] = pd.to_datetime(wf["trade_date"])
    df = df.merge(wf, on="trade_date", how="left")
    tt = df["total_turnover_tl"].replace(0, np.nan)
    df["mlb_late_share"] = ((df["cand_mlb_w4_flow_tl"].fillna(0) + df["cand_mlb_w5_flow_tl"].fillna(0)) / tt * 100).clip(-100, 100)
    feats = sorted(set(FEATURE_COLS) | set(LEAN_INST))
    y = df[["trade_date"]].copy()
    for f in feats:
        y[f] = df[f].shift(1)  # known at the previous close
    y["prev_date"] = df["trade_date"].shift(1)
    y["preopen"] = (df["close_price"] / df["open_price"] - 1) * 100
    p, _ = engineer(pd.read_parquet(OUT_DIR / "tick_panel.parquet"))
    p = p[p["symbol"] == sym][["trade_date", "morning_ret", "rest_1130"]].rename(columns={"rest_1130": "s1130"})
    y = y.merge(p, on="trade_date", how="left")
    y["gap"] = [pd.notna(a) and bool(((MISSING_RAW_DAYS > a) & (MISSING_RAW_DAYS < b)).any()) for a, b in zip(y["prev_date"], y["trade_date"])]
    y["symbol"] = sym
    return y.dropna(subset=["prev_date"]).reset_index(drop=True)


def run_symbol(sym: str) -> pd.DataFrame:
    from mdk_trading_oracle.data.silver.tertip_ml_forecaster import FEATURE_COLS

    d = stock_frame(sym)
    sets = {"LEAN_MLB": LEAN_MLB, "LEAN_INST": LEAN_INST, "OLD_39": list(FEATURE_COLS)}
    eval_idx = list(range(len(d) - EVAL_SESSIONS, len(d)))
    out = []
    for kind in ("preopen", "s1130"):
        for sname, cols in sets.items():
            cols = cols + (["morning_ret"] if kind == "s1130" else [])
            for i in eval_idx:
                day = d.loc[i, "trade_date"]
                x_today = d.loc[[i], cols]
                actual = d.loc[i, kind]
                for w in WINDOWS:
                    hist = d.iloc[max(0, i - w):i]  # sessions strictly before `day`
                    hist = hist.dropna(subset=cols + [kind])
                    assert (hist["trade_date"] < day).all()
                    for m in MODELS:
                        if len(hist) < min(40, w - 5) or x_today.isna().any(axis=None):
                            pred = np.nan
                        elif m == "ZERO":
                            pred = 0.0
                        elif m == "MEAN":
                            pred = float(hist[kind].mean())
                        else:
                            model, key = make_model(m)
                            yv = hist[kind].to_numpy()
                            model.fit(hist[cols], yv, **{key: weights(yv)})
                            pred = float(np.clip(model.predict(x_today)[0], -10, 10))
                        out.append({"symbol": sym, "kind": kind, "features": sname, "model": m, "window": w, "trade_date": day,
                                    "forecast_pct": pred, "actual_pct": actual, "gap": bool(d.loc[i, "gap"])})
        print(f"{sym} {kind} done", file=sys.stderr)
    return pd.DataFrame(out)


def score(g: pd.DataFrame) -> pd.Series:
    g = g[~g["gap"]].dropna(subset=["forecast_pct", "actual_pct"])
    hit = np.sign(g["forecast_pct"]) == np.sign(g["actual_pct"])
    return pd.Series({"days": len(g), "hit_rate_pct": hit.mean() * 100, "mae_pp": (g["forecast_pct"] - g["actual_pct"]).abs().mean()})


def report() -> None:
    pd.set_option("display.width", 250)
    pd.set_option("display.max_rows", 200)
    P = pd.read_parquet(OUT / "predictions.parquet")
    P["candidate"] = P["features"] + " | " + P["model"] + " | " + P["window"].astype(str)
    P.loc[P["model"].isin(["ZERO", "MEAN"]), "features"] = "-"
    dates = sorted(P["trade_date"].unique())
    test_days = dates[-TEST_SESSIONS:]
    P["phase"] = np.where(P["trade_date"].isin(test_days), "test", "validation")
    S = P.groupby(["kind", "symbol", "candidate", "phase"]).apply(score).unstack("phase")
    S.columns = [f"{a}_{b}" for a, b in S.columns]
    S = S.reset_index()
    rows = []
    for (kind, sym), g in S.groupby(["kind", "symbol"]):
        g = g[g["days_validation"] >= 40]
        champ = g.sort_values(["hit_rate_pct_validation", "mae_pp_validation"], ascending=[False, True]).iloc[0]
        insample = g.sort_values(["hit_rate_pct_test", "mae_pp_test"], ascending=[False, True]).iloc[0]
        zero = g[g["candidate"].str.contains("ZERO \\| 252")].iloc[0]
        rows.append({"kind": kind, "symbol": sym, "champion (crowned on validation)": champ["candidate"],
                     "val_hit": champ["hit_rate_pct_validation"], "TEST_hit": champ["hit_rate_pct_test"], "TEST_mae": champ["mae_pp_test"],
                     "zero_mae": zero["mae_pp_test"], "crowned_on_test_itself": insample["candidate"], "its_test_hit (in-sample)": insample["hit_rate_pct_test"],
                     "median_candidate_test_hit": g["hit_rate_pct_test"].median()})
    C = pd.DataFrame(rows)
    C.to_csv(OUT / "champions.csv", index=False)
    S.to_csv(OUT / "scores.csv", index=False)
    for kind in ("preopen", "s1130"):
        print(f"\n=== {kind}: champion per stock (crowned on the 60 sessions before; shown on the last 60) ===")
        print(C[C["kind"] == kind].drop(columns="kind").round(1).to_string(index=False))
    # champion forecasts on the test window, for the page
    sel = []
    for _, c in C.iterrows():
        m = P[(P["kind"] == c["kind"]) & (P["symbol"] == c["symbol"]) & (P["candidate"] == c["champion (crowned on validation)"]) & (P["phase"] == "test")]
        sel.append(m)
    pd.concat(sel).to_parquet(OUT / "champion_test_forecasts.parquet")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--report-only", action="store_true")
    a = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    if not a.report_only:
        with Pool(6) as pool:
            P = pd.concat(pool.map(run_symbol, FOCUS), ignore_index=True)
        P.to_parquet(OUT / "predictions.parquet")
    report()


if __name__ == "__main__":
    main()
