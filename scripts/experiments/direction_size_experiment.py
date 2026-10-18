#!/usr/bin/env python
"""Experiment: calibrated direction probabilities + volatility-based sizing vs current regressor forecasts.

Per stock and horizon (T+1, 3D, 5D, 10D, 15D, 30D), with the crowned lookback / feature mode and zero look-ahead:

  OLD  : crowned production regressor + `calibrate_expected_return` (split-adjusted targets, as in production).
  NEW  : 3-class direction classifiers (down / flat / up vs the horizon deadband), probability-calibrated with
         time-ordered sigmoid calibration inside each training window:
            LR   - standardized multinomial logistic regression
            LGBM - shallow LightGBM classifier
            AVG  - mean of the two calibrated probability vectors
         Edge = P(up) - P(down). A call is made when |edge| clears a per-stock threshold set from the stock's own
         *past* out-of-sample edges so that a target call-rate floor is met (50% ... 100%); otherwise NEUTRAL.
         Size = typical move in the called direction from the GJR-GARCH-t filtered historical simulation, scaled by
         confidence (0.75x - 1.25x of the conditional median), capped at the conditional 90th percentile;
         T+1 sizes above 5% are compressed smoothly towards a ~7% ceiling.

Raw per-session outputs are saved so the call-rate sweep and sizing rules are analysed offline (`--report-only`).

Usage:
    .venv/bin/python scripts/experiments/direction_size_experiment.py [--symbols AKBNK] [--workers 12]
    .venv/bin/python scripts/experiments/direction_size_experiment.py --report-only
"""

from __future__ import annotations

import argparse
import os

os.environ.setdefault("OMP_NUM_THREADS", "1")

import time  # noqa: E402
import warnings  # noqa: E402
import zlib  # noqa: E402
from multiprocessing import Pool  # noqa: E402
from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import yaml  # noqa: E402

warnings.filterwarnings("ignore")

HORIZONS = {"1d": 1, "3d": 3, "5d": 5, "10d": 10, "15d": 15, "30d": 30}
T1_EVAL, MH_EVAL = 120, 180
FLOORS = [0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
BASE_MODELS = {"LR": ("LR", True, False), "LGBM": ("LGBM", True, False), "LRB": ("LR", False, True), "LGBMB": ("LGBM", False, True)}
# Variants analysed offline: calibrated (LR, LGBM, AVG), class-balanced (LRB, LGBMB, AVGB) and prior de-biased (*_DB)
# REG: the crowned regressor's own (calibrated) prediction is the edge; gating and sizing are applied on top of it
# REG_Z: regressor signal-to-noise (prediction / horizon sigma); AGREE: REG_Z only where the de-biased classifier
# ensemble points the same way (disagreement -> zero edge, i.e. NEUTRAL first)
CLASSIFIERS = ["REG", "REG_Z", "AGREE", "LR", "LGBM", "AVG", "LRB", "LGBMB", "AVGB", "LR_DB", "LGBM_DB", "AVG_DB"]
MIN_EDGE_HISTORY = 20
T1_SOFT_START, T1_CEILING = 5.0, 7.0
N_PATHS = 2000
OUT_DIR = Path(os.environ.get("DS_OUT", Path.home() / "data" / "mdk_oracle" / "experiments" / "direction_size"))


# ── Classifiers ───────────────────────────────────────────────────────────────


def _make_classifier(name: str, balanced: bool = False):
    cw = "balanced" if balanced else None
    if name == "LR":
        from sklearn.linear_model import LogisticRegression
        from sklearn.pipeline import Pipeline
        from sklearn.preprocessing import StandardScaler

        return Pipeline([("scaler", StandardScaler()), ("clf", LogisticRegression(C=0.5, max_iter=500, class_weight=cw))])
    import lightgbm as lgb

    return lgb.LGBMClassifier(
        n_estimators=60, max_depth=3, num_leaves=7, learning_rate=0.05, subsample=0.8, subsample_freq=1,
        colsample_bytree=0.8, min_child_samples=10, random_state=42, n_jobs=1, verbose=-1, class_weight=cw,
    )


def fit_predict_proba(
    name: str, X: np.ndarray, y: np.ndarray, x_new: np.ndarray, calibrate: bool = True, balanced: bool = False
) -> np.ndarray:
    """[P(down), P(flat), P(up)]: time-ordered sigmoid calibration (falls back to raw) or class-balanced raw model."""
    from sklearn.calibration import CalibratedClassifierCV
    from sklearn.model_selection import TimeSeriesSplit

    out = np.zeros(3)
    classes = np.unique(y)
    if len(classes) < 2:
        out[int(classes[0])] = 1.0
        return out
    try:
        if not calibrate:
            raise ValueError("uncalibrated")
        model = CalibratedClassifierCV(_make_classifier(name), method="sigmoid", cv=TimeSeriesSplit(n_splits=3))
        model.fit(X, y)
    except Exception:
        model = _make_classifier(name, balanced=balanced)
        model.fit(X, y)
    proba = model.predict_proba(x_new.reshape(1, -1))[0]
    for cls, p in zip(model.classes_, proba):
        out[int(cls)] = p
    return out / out.sum()


def label_classes(ret: np.ndarray, deadband: float) -> np.ndarray:
    return np.where(ret > deadband, 2, np.where(ret < -deadband, 0, 1))


# ── Volatility state per stock (zero-leakage: parameters fitted before the evaluation windows) ──


def volatility_paths_factory(adj_close: pd.Series, first_eval_idx: int):
    from mdk_trading_oracle.models import stochastic_risk as sr

    log_ret = (np.log(adj_close.astype(float)).diff() * 100.0).to_numpy()
    pre = log_ret[1 : first_eval_idx + 1]
    pre = pre[np.isfinite(pre)][-sr.GARCH_FIT_LOOKBACK:]
    mu = pre.mean()
    params = sr.fit_garch_params(pre)
    init = float(np.var(pre[:60]))
    s2_pre = sr.filter_conditional_variance(pre - mu, params, init)
    z = (pre - mu) / np.sqrt(np.maximum(s2_pre[:-1], 1e-12))
    pool = (z - z.mean()) / z.std()
    s2_all = sr.filter_conditional_variance(np.nan_to_num(log_ret[1:]) - mu, params, init)

    def directional_stats(i: int, n: int, rng: np.random.Generator) -> dict:
        """Zero-drift terminal return distribution from origin i (info through i) over n sessions."""
        state = sr.VolatilityState(params=params, sigma2_next=float(s2_all[i]), std_resid=pool, n_obs=len(pre), as_of_date="")
        term = (np.exp(sr.simulate_log_paths(state, n, N_PATHS, rng)[:, -1] / 100.0) - 1.0) * 100.0
        up, dn = term[term > 0], -term[term < 0]
        return {
            "med_up": float(np.median(up)) if len(up) else 0.0,
            "med_dn": float(np.median(dn)) if len(dn) else 0.0,
            "q90_up": float(np.quantile(up, 0.9)) if len(up) else 0.0,
            "q90_dn": float(np.quantile(dn, 0.9)) if len(dn) else 0.0,
            "sigma_h": float(np.std(term)),
        }

    return directional_stats, params.method


# ── Per-symbol worker ─────────────────────────────────────────────────────────


def run_symbol(sym: str) -> pd.DataFrame:
    from mdk_trading_oracle.core.db import PostgresManager
    from mdk_trading_oracle.data.silver import tertip_multi_horizon_forecaster as tmh
    from mdk_trading_oracle.data.silver.tertip_ml_forecaster import (
        DEADBAND_PCT,
        _check_hit,
        adjusted_price,
        calibrate_expected_return,
        extract_3pillar_time_series,
        get_feature_cols_for_mode,
    )

    sys_path_arena = Path(__file__).resolve().parent
    import sys

    sys.path.insert(0, str(sys_path_arena))
    import model_candidate_arena as arena

    t0 = time.time()
    db = PostgresManager(read_only=True)
    t1_cfg = yaml.safe_load(open("config/tertip_crowned_models.yaml"))["symbols"].get(sym, {})
    mh_cfg = yaml.safe_load(open("config/tertip_multi_horizon_crowned_models.yaml"))["symbols"].get(sym, {})
    rng = np.random.default_rng(zlib.crc32(sym.encode("utf-8")))
    rows: list[dict] = []

    # ---------- T+1 ----------
    if t1_cfg:
        lb = int(t1_cfg.get("training_lookback_sessions", 252))
        f_cols = get_feature_cols_for_mode(t1_cfg.get("feature_mode", "FIFO_CORE"))
        df = extract_3pillar_time_series(db, sym, lookback_days=int((lb + T1_EVAL + 60) * 1.5))
        if len(df) >= T1_EVAL + 63:
            for c in f_cols:
                if c not in df.columns:
                    df[c] = 0.0
            adj = adjusted_price(df, "close").reset_index(drop=True)
            df = df.reset_index(drop=True)
            next_ret = (adj.shift(-1) / adj - 1.0) * 100.0
            X_all = np.nan_to_num(df[f_cols].to_numpy(dtype=np.float64))
            stats_fn, _ = volatility_paths_factory(adj, len(df) - T1_EVAL - 1)
            for idx in range(len(df) - T1_EVAL, len(df)):
                o = idx - 1  # origin session (features known at its close)
                start = max(0, o - lb)
                tr_idx = np.arange(start, o)  # labels r(i -> i+1) realized by o
                tr_idx = tr_idx[np.isfinite(next_ret.iloc[tr_idx].to_numpy())]
                if len(tr_idx) < 63:
                    continue
                y_reg = next_ret.iloc[tr_idx].to_numpy()
                actual = float(next_ret.iloc[o])
                # OLD: crowned regressor replica (adjusted targets, production weights / clipping / calibration)
                try:
                    model, wkey = arena.build_t1_current(t1_cfg["model"])
                    w = np.where(np.abs(y_reg) < 1.0, 0.50, np.where(np.abs(y_reg) < 2.0, 3.50, 6.00))
                    model.fit(pd.DataFrame(X_all[tr_idx], columns=f_cols), y_reg, **{wkey: w})
                    raw = float(model.predict(pd.DataFrame(X_all[o : o + 1], columns=f_cols))[0])
                except Exception:
                    raw = 0.0
                old_pred = calibrate_expected_return(max(-10.0, min(10.0, raw)), float(df["atr_14d_pct"].iloc[o]))
                rec = {"symbol": sym, "horizon": "1d", "n_days": 1, "deadband": DEADBAND_PCT,
                       "trade_date": str(df["trade_date"].iloc[o])[:10], "actual": actual, "old_pred": old_pred,
                       "old_lenient_hit": bool(_check_hit(old_pred, actual))}
                y_cls = label_classes(y_reg, DEADBAND_PCT)
                probs = {k: fit_predict_proba(b, X_all[tr_idx], y_cls, X_all[o], cal, bal) for k, (b, cal, bal) in BASE_MODELS.items()}
                probs["AVG"] = (probs["LR"] + probs["LGBM"]) / 2.0
                probs["AVGB"] = (probs["LRB"] + probs["LGBMB"]) / 2.0
                rec["prior_up"], rec["prior_dn"] = float((y_cls == 2).mean()), float((y_cls == 0).mean())
                for name, p in probs.items():
                    rec[f"{name}_p_dn"], rec[f"{name}_p_flat"], rec[f"{name}_p_up"] = p
                rec.update(stats_fn(o, 1, rng))
                rows.append(rec)

    # ---------- Multi-horizon ----------
    if mh_cfg:
        df0 = extract_3pillar_time_series(db, sym, lookback_days=550)
        for h_key in ["3d", "5d", "10d", "15d", "30d"]:
            cfg = mh_cfg.get(h_key)
            if not cfg or len(df0) < 120:
                continue
            n = HORIZONS[h_key]
            deadband = tmh.get_horizon_deadband(n)
            lb = int(cfg.get("training_lookback_sessions", 126))
            f_mode = cfg.get("feature_mode", "FIFO_CORE")
            old = tmh.evaluate_horizon_walk_forward(df0.copy(), h_key, cfg["model"], lb, MH_EVAL, f_mode)
            old_by_date = {r["trade_date"]: r for r in old.get("ledger", [])}
            if not old_by_date:
                continue
            df, f_cols = tmh.engineer_multi_horizon_features(df0.copy(), feature_mode=f_mode)
            df = df.reset_index(drop=True)
            adj = adjusted_price(df, "close")
            fwd = (adj.shift(-n) / adj - 1.0) * 100.0
            X_all = np.nan_to_num(df[f_cols].to_numpy(dtype=np.float64))
            dates = [str(d)[:10] for d in df["trade_date"]]
            first = min(dates.index(d) for d in old_by_date)
            stats_fn, _ = volatility_paths_factory(adj, first)
            for t, d in enumerate(dates):
                if d not in old_by_date:
                    continue
                train_end = t - n
                start = max(0, train_end - lb + 1)
                tr_idx = np.arange(start, train_end + 1)
                tr_idx = tr_idx[np.isfinite(fwd.iloc[tr_idx].to_numpy())]
                if len(tr_idx) < 40:
                    continue
                o = old_by_date[d]
                rec = {"symbol": sym, "horizon": h_key, "n_days": n, "deadband": deadband, "trade_date": d,
                       "actual": float(fwd.iloc[t]), "old_pred": float(o["pred_return_pct"]),
                       "old_lenient_hit": bool(o["is_hit"])}
                y_cls = label_classes(fwd.iloc[tr_idx].to_numpy(), deadband)
                probs = {k: fit_predict_proba(b, X_all[tr_idx], y_cls, X_all[t], cal, bal) for k, (b, cal, bal) in BASE_MODELS.items()}
                probs["AVG"] = (probs["LR"] + probs["LGBM"]) / 2.0
                probs["AVGB"] = (probs["LRB"] + probs["LGBMB"]) / 2.0
                rec["prior_up"], rec["prior_dn"] = float((y_cls == 2).mean()), float((y_cls == 0).mean())
                for name, p in probs.items():
                    rec[f"{name}_p_dn"], rec[f"{name}_p_flat"], rec[f"{name}_p_up"] = p
                rec.update(stats_fn(t, n, rng))
                rows.append(rec)

    out = pd.DataFrame(rows)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out.to_parquet(OUT_DIR / f"{sym}.parquet", index=False)
    print(f"  {sym}: {len(out)} sessions in {time.time() - t0:.0f}s", flush=True)
    return out


# ── Offline analysis: call-rate floors, sizing, scorecards ────────────────────


def t1_compress(x: np.ndarray) -> np.ndarray:
    """Smoothly compress T+1 sizes above 5% towards a 7% ceiling."""
    width = T1_CEILING - T1_SOFT_START
    return np.where(x <= T1_SOFT_START, x, T1_SOFT_START + width * np.tanh((x - T1_SOFT_START) / width))


def build_new_forecasts(g: pd.DataFrame, clf: str, floor: float) -> pd.DataFrame:
    """Zero-look-ahead call thresholds and sizes for one (symbol, horizon) series, in date order."""
    g = g.sort_values("trade_date").copy()
    base = clf[:-3] if clf.endswith("_DB") else clf
    if clf == "REG":
        edge = g["old_pred"].to_numpy()
    elif clf in ("REG_Z", "AGREE"):
        edge = (g["old_pred"] / g["sigma_h"].replace(0, np.nan)).fillna(0.0).to_numpy()
        if clf == "AGREE":
            cls_edge = (g["AVG_p_up"] - g["AVG_p_dn"] - (g["prior_up"] - g["prior_dn"])).to_numpy()
            edge = np.where(np.sign(edge) == np.sign(cls_edge), edge, 0.0)
    else:
        edge = (g[f"{base}_p_up"] - g[f"{base}_p_dn"]).to_numpy()
    if clf.endswith("_DB"):  # judge the edge against the training window's own up/down base rate
        edge = edge - (g["prior_up"] - g["prior_dn"]).to_numpy()
    abs_edge = np.abs(edge)
    called = np.zeros(len(g), dtype=bool)
    conf = np.full(len(g), 0.5)
    for k in range(len(g)):
        hist = abs_edge[:k]
        if len(hist) < MIN_EDGE_HISTORY or floor >= 1.0:
            called[k] = abs_edge[k] > 0 if floor >= 1.0 else True
            conf[k] = 0.5 if len(hist) < MIN_EDGE_HISTORY else float((hist < abs_edge[k]).mean())
            continue
        tau = np.quantile(hist, 1.0 - floor)
        called[k] = abs_edge[k] >= tau
        conf[k] = float((hist < abs_edge[k]).mean())
    up = edge > 0
    med = np.where(up, g["med_up"], g["med_dn"])
    q90 = np.where(up, g["q90_up"], g["q90_dn"])
    size = np.minimum((0.75 + 0.5 * conf) * med, q90)
    if g["n_days"].iloc[0] == 1:
        size = t1_compress(size)
    g["new_pred"] = np.where(called, np.where(up, size, -size), 0.0)
    g["new_called"] = called
    g["p_dir"] = np.nan if clf in ("REG", "REG_Z", "AGREE") else np.where(up, g[f"{base}_p_up"], g[f"{base}_p_dn"])
    return g


def scorecard(d: pd.DataFrame, pred_col: str, called: pd.Series) -> dict:
    from mdk_trading_oracle.data.silver.tertip_ml_forecaster import _check_hit

    a, p, db, n = d["actual"], d[pred_col], d["deadband"], d["n_days"]
    strict = np.where(p > 0, a > db, a < -db)
    big1 = a.abs() >= 1.0 * np.sqrt(n)
    big2 = a.abs() >= 2.0 * np.sqrt(n)
    capture = lambda mask: float(((np.sign(p) == np.sign(a)) & called)[mask].mean() * 100) if mask.any() else float("nan")  # noqa: E731
    correct = called & (np.sign(p) == np.sign(a))
    ratio = (a.abs() / p.abs().replace(0, np.nan))[correct]
    tol = np.where(n == 1, 0.20, 0.25)
    lenient = [bool(_check_hit(pp, aa, tol_delta=tt, deadband=dd)) for pp, aa, tt, dd in zip(p, a, tol, db)]
    return {
        "strict_precision": float(strict[called].mean() * 100) if called.any() else float("nan"),
        "call_rate": float(called.mean() * 100),
        "big1_capture": capture(big1),
        "big2_capture": capture(big2),
        "size_ratio_on_hits": float(ratio.median()) if len(ratio) else float("nan"),
        "median_abs_pred": float(p[called].abs().median()) if called.any() else 0.0,
        "share_above_5pct": float((p.abs() > 5.0).mean() * 100),
        "legacy_lenient_hit": float(np.mean(lenient) * 100),
    }


def report(res: pd.DataFrame) -> str:
    H = list(HORIZONS)
    lines = ["# Direction + size experiment\n"]
    old_rows = []
    for h in H:
        d = res[res.horizon == h]
        if d.empty:
            continue
        old_called = d["old_pred"].abs() > d["deadband"]
        old_rows.append({"horizon": h, **scorecard(d, "old_pred", old_called)})
    old = pd.DataFrame(old_rows).set_index("horizon")
    lines.append(f"## OLD (crowned regressor)\n\n{old.round(2).to_string()}\n")

    sweep = []
    built = {}
    for clf in CLASSIFIERS:
        for floor in FLOORS:
            parts = [build_new_forecasts(g, clf, floor) for _, g in res.groupby(["symbol", "horizon"])]
            nd = pd.concat(parts)
            built[(clf, floor)] = nd
            for h in H:
                d = nd[nd.horizon == h]
                if d.empty:
                    continue
                sweep.append({"clf": clf, "floor": floor, "horizon": h, **scorecard(d, "new_pred", d["new_called"])})
    sw = pd.DataFrame(sweep)
    for metric in ["strict_precision", "call_rate", "big1_capture", "legacy_lenient_hit"]:
        t = sw.pivot_table(index=["clf", "floor"], columns="horizon", values=metric).reindex(columns=H)
        lines.append(f"## NEW {metric}\n\n{t.round(1).to_string()}\n")
    for clf in ["REG", "REG_Z", "AGREE", "AVG_DB"]:
        t = sw[(sw.clf == clf) & (sw.floor == 0.7)].set_index("horizon").reindex(H)
        lines.append(f"## NEW {clf} @ 70% floor - full scorecard\n\n{t.drop(columns=['clf', 'floor']).round(2).to_string()}\n")

    # Probability reliability (AVG, all called + uncalled sessions)
    nd = built[("AVG", 1.0)]
    nd = nd.assign(hit=np.where(nd["new_pred"] > 0, nd["actual"] > nd["deadband"], nd["actual"] < -nd["deadband"]))
    nd["bin"] = pd.cut(nd["p_dir"], [0, 0.35, 0.4, 0.45, 0.5, 0.55, 0.6, 0.7, 1.0])
    rel = nd.groupby("bin", observed=True).agg(model_p=("p_dir", lambda x: x.mean() * 100), realized=("hit", lambda x: x.mean() * 100), n=("hit", "size"))
    lines.append(f"## Probability reliability (AVG): P(called direction) vs realized\n\n{rel.round(1).to_string()}\n")
    return "\n".join(lines)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--symbols", default="")
    ap.add_argument("--workers", type=int, default=12)
    ap.add_argument("--report-only", action="store_true")
    args = ap.parse_args()
    if args.report_only:
        res = pd.concat([pd.read_parquet(p) for p in sorted(OUT_DIR.glob("*.parquet"))], ignore_index=True)
    else:
        if args.symbols:
            symbols = [s.strip().upper() for s in args.symbols.split(",") if s.strip()]
        else:
            symbols = sorted(yaml.safe_load(open("config/tertip_crowned_models.yaml"))["symbols"])
        print(f"Running {len(symbols)} symbols with {args.workers} workers -> {OUT_DIR}", flush=True)
        with Pool(processes=min(args.workers, len(symbols))) as pool:
            res = pd.concat(pool.map(run_symbol, symbols, chunksize=1), ignore_index=True)
    md = report(res)
    (OUT_DIR / "REPORT.md").write_text(md, encoding="utf-8")
    print(md)


if __name__ == "__main__":
    main()
