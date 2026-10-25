#!/usr/bin/env python
"""Feature importance of the crowned models, market-regime features and pruning (isolated, read-only on the DB).

Stocks: focus six + THYAO. Horizons: 1D, 3D, 5D, 10D, 15D, 30D.

Part A — crowned models (October 2026 champions from gold_tertip_multi_horizon_crowns):
    native importance   standardized |coef| (linear) or gain share (trees) of the champion fitted at the latest close
    OOS permutation     champion refitted point-in-time at each of the last 90 graded origins; one feature shuffled
                        across those test rows (20 repeats); drop in hit rate / rise in MAE vs unshuffled
Part B — feature x horizon information: rank IC of every feature with the forward N-day return, pooled over the
    stocks, on the DISCOVERY window (origins whose outcome is realized before the evaluation window starts) and on the
    EVALUATION window (last 90 graded origins). Pruning decisions use discovery only.
Part C — variants, fixed configs (no selection), scored on the evaluation window against always-UP / always-DOWN:
    base        production FIFO_CORE features
    base_mkt    base + market-regime features
    pruned      base minus features without consistent discovery IC
    pruned_mkt  pruned + market-regime features
    horizon     per-horizon set chosen on discovery (top features by |IC| incl. market features)
    mkt_only    market-regime + relative-strength features only

Every fit is point-in-time: swing via models/horizon_forecast.fit_horizon (production models, weights, demeaning,
ATR calibration), 1D via models/t1_forecast.fit_t1_model.

Outputs: ~/data/mdk_oracle/experiments/feature_importance/*.parquet|csv
Usage:   .venv/bin/python scripts/experiments/feature_importance_experiment.py [--part A|B|C|all] [--report-only]
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

STOCKS = ["ASELS", "GARAN", "AKBNK", "YKBNK", "ISCTR", "TTKOM", "THYAO"]
HORIZONS = ["1d", "3d", "5d", "10d", "15d", "30d"]
HDAYS = {"1d": 1, "3d": 3, "5d": 5, "10d": 10, "15d": 15, "30d": 30}
OUT = Path.home() / "data" / "mdk_oracle" / "experiments" / "feature_importance"
EVAL_SESSIONS = 90
DISCOVERY_START = "2025-01-01"
N_PERM = 20
FIXED_CONFIGS = [("Ridge", 252), ("BayesianRidge", 252), ("LightGBM", 252)]
MKT_PREFIX = "mkt_"


def deadband(h: str) -> float:
    if h == "1d":
        return 0.25
    from mdk_trading_oracle.data.silver.tertip_multi_horizon_forecaster import get_horizon_deadband

    return get_horizon_deadband(HDAYS[h])


def hub_hit(pred: np.ndarray, actual: np.ndarray, h: str) -> np.ndarray:
    db, tol = deadband(h), (0.20 if h == "1d" else 0.25)
    near = np.abs(pred - actual) <= tol
    flat = np.abs(actual) <= db
    return near | np.where(flat, (np.abs(pred) <= db) | (pred * actual >= 0.0), np.where(actual > db, pred > 0.0, pred < 0.0))


# ----------------------------------------------------------------------------------------------------------------------
# Data: production features + market-regime features (all backward-looking, row t uses data <= close of t)


def market_frame(db) -> pd.DataFrame:
    """Market-regime features on the stock trading-date grid."""
    idx = db.query_df("SELECT trade_date, close_price FROM silver_daily_benchmark_index ORDER BY trade_date")
    idx["trade_date"] = pd.to_datetime(idx["trade_date"])
    px = idx.set_index("trade_date")["close_price"].astype(float)
    m = pd.DataFrame(index=px.index)
    for n in (5, 10, 20, 60):
        m[f"{MKT_PREFIX}bist30_ret_{n}d"] = (px / px.shift(n) - 1.0) * 100.0
    for n in (20, 50):
        m[f"{MKT_PREFIX}bist30_vs_sma{n}"] = (px / px.rolling(n, min_periods=n // 2).mean() - 1.0) * 100.0
    m[f"{MKT_PREFIX}bist30_drawdown_60d"] = (px / px.rolling(60, min_periods=20).max() - 1.0) * 100.0
    r = np.log(px).diff()
    m[f"{MKT_PREFIX}bist30_vol_20d"] = r.rolling(20, min_periods=10).std() * np.sqrt(252) * 100.0
    m[f"{MKT_PREFIX}bist30_mom_spread_5_20"] = m[f"{MKT_PREFIX}bist30_ret_5d"] - m[f"{MKT_PREFIX}bist30_ret_20d"] / 4.0

    st = db.query_df(
        """
        SELECT trade_date, symbol, COALESCE(adj_close_price, close_price) AS px, total_turnover_tl, bofa_net_flow_tl
        FROM silver_daily_stock_summary WHERE index_name = 'BIST30' ORDER BY trade_date
        """
    )
    st["trade_date"] = pd.to_datetime(st["trade_date"])
    wide = st.pivot_table(index="trade_date", columns="symbol", values="px")
    above = (wide > wide.rolling(20, min_periods=10).mean()).where(wide.notna())
    breadth = pd.DataFrame(index=wide.index)
    breadth[f"{MKT_PREFIX}breadth_above_sma20"] = above.mean(axis=1) * 100.0
    adv = (wide.pct_change() > 0).where(wide.notna())
    breadth[f"{MKT_PREFIX}breadth_adv_5d"] = adv.mean(axis=1).rolling(5, min_periods=3).mean() * 100.0
    flow = st.groupby("trade_date")[["bofa_net_flow_tl", "total_turnover_tl"]].sum()
    for n in (1, 5, 10):
        breadth[f"{MKT_PREFIX}mlb_flow_share_{n}d"] = (
            flow["bofa_net_flow_tl"].rolling(n, min_periods=1).sum() / flow["total_turnover_tl"].rolling(n, min_periods=1).sum() * 100.0
        )
    rates = db.query_df("SELECT trade_date, interest_rate FROM silver_daily_macro_rates ORDER BY trade_date")
    rates["trade_date"] = pd.to_datetime(rates["trade_date"])
    rr = rates.set_index("trade_date")["interest_rate"].astype(float)
    rate = pd.DataFrame({f"{MKT_PREFIX}repo_change_60d": rr - rr.shift(60)})
    out = breadth.join(m, how="left").join(rate, how="left")
    return out.sort_index().ffill()


def stock_frame(db, sym: str, mkt: pd.DataFrame) -> pd.DataFrame:
    from mdk_trading_oracle.data.silver.tertip_ml_forecaster import adjusted_price, extract_3pillar_time_series
    from mdk_trading_oracle.data.silver.tertip_multi_horizon_forecaster import engineer_multi_horizon_features

    df = extract_3pillar_time_series(db, sym, lookback_days=5000)
    df, _ = engineer_multi_horizon_features(df, feature_mode="FIFO_PLUS_YTD")
    df = df.sort_values("trade_date").reset_index(drop=True)
    td = pd.to_datetime(df["trade_date"])
    m = mkt.reindex(td, method="ffill").reset_index(drop=True)
    for c in m.columns:
        df[c] = m[c].to_numpy()
    px = adjusted_price(df, "close")
    for n in (10, 20):
        df[f"{MKT_PREFIX}rel_strength_{n}d"] = (px / px.shift(n) - 1.0) * 100.0 - df[f"{MKT_PREFIX}bist30_ret_{n}d"]
    df["trade_date"] = td.dt.strftime("%Y-%m-%d")
    return df


def base_features(df: pd.DataFrame) -> list[str]:
    from mdk_trading_oracle.data.silver.tertip_multi_horizon_forecaster import filter_multi_horizon_features

    return filter_multi_horizon_features([c for c in df.columns if c.startswith("feat_")], "FIFO_CORE")


def mkt_features(df: pd.DataFrame) -> list[str]:
    return [c for c in df.columns if c.startswith(MKT_PREFIX)]


def mode_features(df: pd.DataFrame, mode: str, horizon: str) -> list[str]:
    """Feature list a production candidate of `mode` sees for `horizon` (1D uses the T+1 feature suite)."""
    if horizon == "1d":
        from mdk_trading_oracle.data.silver.tertip_ml_forecaster import get_feature_cols_for_mode

        return get_feature_cols_for_mode(mode)
    from mdk_trading_oracle.data.silver.tertip_multi_horizon_forecaster import filter_multi_horizon_features

    return filter_multi_horizon_features([c for c in df.columns if c.startswith("feat_")], mode)


def labels(df: pd.DataFrame, h: str) -> np.ndarray:
    from mdk_trading_oracle.models.horizon_forecast import horizon_labels

    return horizon_labels(df, HDAYS[h]).to_numpy(float)


def windows(df: pd.DataFrame, h: str) -> tuple[list[int], list[int]]:
    """Evaluation = last 90 graded origins; discovery = origins >= DISCOVERY_START realized before evaluation starts."""
    n = HDAYS[h]
    dates = df["trade_date"].tolist()
    last = len(df) - n - 1
    ev = list(range(last - EVAL_SESSIONS + 1, last + 1))
    ev_start = dates[ev[0]]
    disc = [t for t in range(len(df) - n) if dates[t] >= DISCOVERY_START and t + n < len(df) and dates[t + n] < ev_start]
    return disc, ev


# ----------------------------------------------------------------------------------------------------------------------
# Point-in-time fits


def fit_predict(df: pd.DataFrame, y: np.ndarray, t: int, h: str, model: str, lookback: int, cols: list[str]):
    """Returns (model object, scaler-free X row, raw prediction, calibrated prediction) at origin t, or None."""
    from mdk_trading_oracle.models.horizon_forecast import calibrate_horizon, fit_horizon

    if h == "1d":
        from mdk_trading_oracle.models.t1_forecast import T1Config, fit_t1_model, predict_t1_row

        frame = df if all(c in df.columns for c in cols) else df.assign(**{c: 0.0 for c in cols if c not in df.columns})
        fit = fit_t1_model(frame, t, T1Config(model, lookback, "custom"), cols, pd.Series(y))
        if fit is None:
            return None
        p = predict_t1_row(fit, frame, t, cols)
        return None if p is None else (fit.model, p["raw_return_pct"], p["model_return_pct"])
    X = np.ascontiguousarray(np.nan_to_num(df[cols].to_numpy(float)))
    fit = fit_horizon(X, y, t, HDAYS[h], lookback, model, deadband(h))
    if fit is None:
        return None
    raw = fit.demeaned_pred
    return None, raw, calibrate_horizon(raw, float(df["atr_14d_pct"].iloc[t]), HDAYS[h], deadband(h))


def walk(df: pd.DataFrame, h: str, model: str, lookback: int, cols: list[str], origins: list[int]) -> pd.DataFrame:
    y = labels(df, h)
    rows = []
    for t in origins:
        r = fit_predict(df, y, t, h, model, lookback, cols)
        if r is None or not np.isfinite(y[t]):
            continue
        rows.append({"t": t, "trade_date": df["trade_date"].iloc[t], "pred": r[2], "raw": r[1], "actual": y[t]})
    return pd.DataFrame(rows)


# ----------------------------------------------------------------------------------------------------------------------
# Part A: crowned-model importance


def native_importance(df: pd.DataFrame, h: str, model: str, lookback: int, cols: list[str]) -> pd.Series:
    """Importance of the champion as fitted at the latest close (what the live forecast relies on)."""
    from mdk_trading_oracle.data.silver.tertip_multi_horizon_forecaster import (
        _build_model,
        _compute_sample_weights,
        _fit_model,
    )
    from mdk_trading_oracle.models.t1_forecast import build_t1_model, t1_sample_weights

    y = labels(df, h)
    n = HDAYS[h]
    t = len(df) - 1
    end = t - n
    lo = max(0, end - lookback + 1)
    X = pd.DataFrame(np.nan_to_num(df[cols].to_numpy(float)[lo : end + 1]), columns=cols)
    yt = y[lo : end + 1]
    ok = np.isfinite(yt)
    X, yt = X[ok], yt[ok]
    if h == "1d":
        m, key = build_t1_model(model)
        m.fit(X, yt, **{key: t1_sample_weights(yt)})
    else:
        m = _fit_model(_build_model(model), X.to_numpy(), yt - yt.mean(), _compute_sample_weights(yt, deadband=deadband(h)))
    est = m.named_steps["regressor"] if hasattr(m, "named_steps") else m
    if hasattr(est, "coef_"):
        coef = np.ravel(est.coef_)
        scale = 1.0 if hasattr(m, "named_steps") else X.std().replace(0, 1.0).to_numpy()
        imp = np.abs(coef * scale)
    elif hasattr(est, "feature_importances_"):
        imp = np.asarray(est.feature_importances_, dtype=float)
        if model == "LightGBM":
            imp = np.asarray(est.booster_.feature_importance(importance_type="gain"), dtype=float)
    else:
        imp = np.zeros(len(cols))
    s = pd.Series(imp, index=cols)
    return s / s.sum() * 100.0 if s.sum() > 0 else s


def permutation_importance(df: pd.DataFrame, h: str, model: str, lookback: int, cols: list[str], origins: list[int], rng) -> pd.DataFrame:
    """OOS: refit at each origin, then shuffle one feature across the test rows; drop in hit rate / rise in MAE."""
    from mdk_trading_oracle.data.silver.tertip_multi_horizon_forecaster import (
        _build_model,
        _compute_sample_weights,
        _fit_model,
    )
    from mdk_trading_oracle.models.horizon_forecast import calibrate_horizon
    from mdk_trading_oracle.models.t1_forecast import build_t1_model, t1_sample_weights

    y = labels(df, h)
    n = HDAYS[h]
    Xall = np.nan_to_num(df[cols].to_numpy(float))
    atr = df["atr_14d_pct"].to_numpy(float)
    models, rows, ts = [], [], []
    for t in origins:
        end = t - n
        lo = max(0, end - lookback + 1)
        yt = y[lo : end + 1]
        ok = np.isfinite(yt)
        if ok.sum() < 30 or not np.isfinite(y[t]):
            continue
        Xt = Xall[lo : end + 1][ok]
        yt = yt[ok]
        if h == "1d":
            m, key = build_t1_model(model)
            m.fit(pd.DataFrame(Xt, columns=cols), yt, **{key: t1_sample_weights(yt)})
        else:
            m = _fit_model(_build_model(model), Xt, yt - yt.mean(), _compute_sample_weights(yt, deadband=deadband(h)))
        models.append(m)
        rows.append(Xall[t])
        ts.append(t)
    if not models:
        return pd.DataFrame()
    Xtest = np.vstack(rows)
    actual = y[ts]

    def score(Xs: np.ndarray) -> tuple[float, float]:
        preds = []
        for k, m in enumerate(models):
            x = Xs[k : k + 1]
            raw = float(m.predict(pd.DataFrame(x, columns=cols) if h == "1d" else x)[0])
            if h == "1d":
                from mdk_trading_oracle.data.silver.tertip_ml_forecaster import calibrate_expected_return

                preds.append(calibrate_expected_return(float(np.clip(raw, -10, 10)), atr[ts[k]] if np.isfinite(atr[ts[k]]) else 2.5, deadband=0.25))
            else:
                preds.append(calibrate_horizon(raw, atr[ts[k]], n, deadband(h)))
        p = np.array(preds)
        return hub_hit(p, actual, h).mean() * 100.0, float(np.mean(np.abs(p - actual)))

    base_hit, base_mae = score(Xtest)
    out = []
    for j, c in enumerate(cols):
        hits, maes = [], []
        for _ in range(N_PERM):
            Xp = Xtest.copy()
            Xp[:, j] = rng.permutation(Xp[:, j])
            a, b = score(Xp)
            hits.append(a)
            maes.append(b)
        out.append({"feature": c, "perm_hit_drop_pp": base_hit - np.mean(hits), "perm_mae_rise": np.mean(maes) - base_mae, "base_hit": base_hit})
    return pd.DataFrame(out)


def task_a(args) -> dict:
    sym, h, crown = args
    df = pd.read_parquet(OUT / "frames" / f"{sym}.parquet")
    cols = mode_features(df, crown["feature_mode"], h)
    cols = [c for c in cols if c in df.columns] if h != "1d" else cols
    for c in cols:
        if c not in df.columns:
            df[c] = 0.0
    _, ev = windows(df, h)
    rng = np.random.default_rng(1)
    nat = native_importance(df, h, crown["model"], int(crown["lookback"]), cols)
    perm = permutation_importance(df, h, crown["model"], int(crown["lookback"]), cols, ev, rng)
    return {"symbol": sym, "horizon": h, "crown": crown, "native": nat, "perm": perm}


def task_a2(args) -> pd.DataFrame:
    """Honest permutation importance: the last 90 graded origins, each with the champion in force at that origin
    (monthly crowns), features shuffled within each champion's own segment; drops weighted by segment length."""
    sym, h, crowns = args
    df = pd.read_parquet(OUT / "frames" / f"{sym}.parquet")
    _, ev = windows(df, h)
    dates = df["trade_date"].tolist()
    crowns = sorted(crowns, key=lambda c: c["crown_date"])
    seg: dict[int, list[int]] = {}
    for t in ev:
        k = max((i for i, c in enumerate(crowns) if c["crown_date"] <= dates[t]), default=None)
        if k is not None:
            seg.setdefault(k, []).append(t)
    rng = np.random.default_rng(2)
    parts = []
    for k, origins in seg.items():
        c = crowns[k]
        cols = mode_features(df, c["feature_mode"], h)
        for col in cols:
            if col not in df.columns:
                df[col] = 0.0
        perm = permutation_importance(df, h, c["model"], int(c["lookback"]), cols, origins, rng)
        if not perm.empty:
            parts.append(perm.assign(weight=len(origins), crown_date=c["crown_date"], config_key=c["config_key"]))
    if not parts:
        return pd.DataFrame()
    p = pd.concat(parts, ignore_index=True)
    total = sum(len(v) for v in seg.values())
    honest_hit = float((p.drop_duplicates("crown_date")[["base_hit", "weight"]].prod(axis=1)).sum() / total)
    g = p.assign(w_drop=p["perm_hit_drop_pp"] * p["weight"] / total).groupby("feature")["w_drop"].sum()
    return g.rename("honest_perm_drop_pp").reset_index().assign(symbol=sym, horizon=h, honest_hit=honest_hit,
                                                               n_champions=len(seg))


# ----------------------------------------------------------------------------------------------------------------------
# Part B: IC map


def task_b(sym: str) -> pd.DataFrame:
    df = pd.read_parquet(OUT / "frames" / f"{sym}.parquet")
    feats = sorted(set(base_features(df)) | set(mkt_features(df)) | set(mode_features(df, "FIFO_PLUS_YTD", "1d")) & set(df.columns))
    rows = []
    for h in HORIZONS:
        y = labels(df, h)
        disc, ev = windows(df, h)
        for name, idx in (("discovery", disc), ("evaluation", ev)):
            sub = df.iloc[idx]
            yy = pd.Series(y[idx], index=sub.index)
            for c in feats:
                x = sub[c].astype(float)
                ok = x.notna() & yy.notna()
                ic = x[ok].rank().corr(yy[ok].rank()) if ok.sum() > 20 and x[ok].nunique() > 2 else np.nan
                rows.append({"symbol": sym, "horizon": h, "window": name, "feature": c, "ic": ic, "n": int(ok.sum())})
    return pd.DataFrame(rows)


# ----------------------------------------------------------------------------------------------------------------------
# Part C: variants


def variant_sets(df: pd.DataFrame, ic: pd.DataFrame, h: str) -> dict[str, list[str]]:
    base = base_features(df) if h != "1d" else [c for c in mode_features(df, "FIFO_CORE", "1d") if c in df.columns]
    mkt = mkt_features(df)
    d = ic[(ic["window"] == "discovery") & (ic["horizon"] == h)]
    g = d.groupby("feature")["ic"]
    mean_ic = g.mean()
    same_sign = g.apply(lambda s: max((s > 0).mean(), (s < 0).mean()))
    informative = set(mean_ic[(mean_ic.abs() >= 0.03) & (same_sign >= 5 / 7)].index)
    pruned = [c for c in base if c in informative] or base[:5]
    ranked = mean_ic.reindex(base + mkt).abs().sort_values(ascending=False)
    consistent = [c for c in ranked.index if same_sign.get(c, 0) >= 5 / 7]
    horizon_set = consistent[:12] or pruned
    return {
        "base": base,
        "base_mkt": base + mkt,
        "pruned": pruned,
        "pruned_mkt": pruned + [c for c in mkt if c not in pruned],
        "horizon": horizon_set,
        "mkt_only": mkt,
    }


def task_c(args) -> pd.DataFrame:
    sym, h = args
    df = pd.read_parquet(OUT / "frames" / f"{sym}.parquet")
    ic = pd.read_parquet(OUT / "ic.parquet")
    sets = variant_sets(df, ic, h)
    disc, ev = windows(df, h)
    out = []
    for vname, cols in sets.items():
        for model, lb in FIXED_CONFIGS:
            for wname, origins in (("discovery", disc[-120:]), ("evaluation", ev)):
                w = walk(df, h, model, lb, cols, origins)
                if w.empty:
                    continue
                out.append(w.assign(symbol=sym, horizon=h, variant=vname, config=f"{model}/{lb}", window=wname, n_features=len(cols)))
    return pd.concat(out, ignore_index=True) if out else pd.DataFrame()


# ----------------------------------------------------------------------------------------------------------------------
# Part D: honest monthly re-crowning on the pruned feature sets (25 candidates: 5 models x 5 lookbacks)


def task_d(sym: str) -> pd.DataFrame:
    """Candidate ledger on pruned features for every horizon over the trailing 300 origins, then monthly crowning with
    the production rule (realized outcomes only) and the champion-in-force track."""
    from mdk_trading_oracle.models.horizon_crowning import (
        FAMILIES,
        LOOKBACKS,
        HorizonConfig,
        crown,
        crown_single,
        month_starts,
    )

    df = pd.read_parquet(OUT / "frames" / f"{sym}.parquet")
    ic = pd.read_parquet(OUT / "ic.parquet")
    configs = [HorizonConfig(fam, m, lbl, lb, "PRUNED") for fam, models in FAMILIES.items() for m in models for lbl, lb in LOOKBACKS]
    dates = df["trade_date"].tolist()
    start = max(0, len(df) - 300)
    rows = []
    for h in HORIZONS:
        cols = variant_sets(df, ic, h)["pruned"]
        y = labels(df, h)
        last = len(df) - HDAYS[h] - 1
        for c in configs:
            w = walk(df, h, c.model, c.lookback, cols, list(range(start, last + 1)))
            for r in w.itertuples(index=False):
                rows.append({"horizon": h, "config_key": c.key, "trade_date": r.trade_date,
                             "target_date_end": dates[r.t + HDAYS[h]], "pred": r.pred, "actual": r.actual,
                             "is_hit": bool(hub_hit(np.array([r.pred]), np.array([r.actual]), h)[0]),
                             "err_pct": abs(r.pred - r.actual)})
        del y
    led = pd.DataFrame(rows)
    months = [m for m in month_starts(pd.Series(dates), dates[0]) if m >= dates[len(df) - 150]]
    out = []
    for i, m in enumerate(months):
        end = months[i + 1] if i + 1 < len(months) else "9999-12-31"
        champs = crown(led[led["horizon"] != "1d"], m, configs=configs) or {}
        c1 = crown_single(led, m, "1d", configs=configs)
        if c1:
            champs["1d"] = c1
        for h, ch in champs.items():
            sel = led[(led["horizon"] == h) & (led["config_key"] == ch["config"].key) & (led["trade_date"] >= m) & (led["trade_date"] < end)]
            out.append(sel.assign(symbol=sym, crown_date=m))
    return pd.concat(out, ignore_index=True) if out else pd.DataFrame()


# ----------------------------------------------------------------------------------------------------------------------
# Orchestration


def build_frames() -> None:
    from mdk_trading_oracle.core.db import PostgresManager

    (OUT / "frames").mkdir(parents=True, exist_ok=True)
    db = PostgresManager(read_only=True)
    mkt = market_frame(db)
    for sym in STOCKS:
        stock_frame(db, sym, mkt).to_parquet(OUT / "frames" / f"{sym}.parquet")
    print(f"frames for {len(STOCKS)} stocks, {len(mkt.columns)} market features", file=sys.stderr)


def current_crowns() -> dict[tuple[str, str], dict]:
    from mdk_trading_oracle.core.db import PostgresManager

    db = PostgresManager(read_only=True)
    c = db.query_df(
        "SELECT symbol, horizon, crown_date::text AS crown_date, model, lookback, feature_mode, config_key FROM gold_tertip_multi_horizon_crowns "
        "WHERE symbol IN (%s) ORDER BY crown_date" % ",".join(f"'{s}'" for s in STOCKS)
    )
    return {(r.symbol, r.horizon): r._asdict() for r in c.itertuples(index=False)}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--part", default="all", choices=["A", "A2", "B", "C", "D", "all"])
    ap.add_argument("--workers", type=int, default=16)
    ap.add_argument("--report-only", action="store_true", help="(reserved) outputs are read by the report notebook/markdown")
    args = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    if args.part == "all" or not (OUT / "frames" / f"{STOCKS[-1]}.parquet").exists():
        build_frames()
    with Pool(args.workers) as pool:
        if args.part in ("all", "A"):
            crowns = current_crowns()
            res = pool.map(task_a, [(s, h, crowns[(s, h)]) for s in STOCKS for h in HORIZONS if (s, h) in crowns])
            nat = pd.concat([r["native"].rename("native_pct").rename_axis("feature").reset_index().assign(symbol=r["symbol"], horizon=r["horizon"]) for r in res])
            perm = pd.concat([r["perm"].assign(symbol=r["symbol"], horizon=r["horizon"]) for r in res if not r["perm"].empty])
            imp = nat.merge(perm, on=["symbol", "horizon", "feature"], how="left")
            cr = pd.DataFrame([{**r["crown"], "symbol": r["symbol"], "horizon": r["horizon"]} for r in res])
            imp.to_parquet(OUT / "importance.parquet")
            cr.to_csv(OUT / "crowns.csv", index=False)
            print(f"A done {time.time() - t0:.0f}s", file=sys.stderr)
        if args.part in ("all", "A", "A2"):
            from mdk_trading_oracle.core.db import PostgresManager

            db = PostgresManager(read_only=True)
            allc = db.query_df(
                "SELECT symbol, horizon, crown_date::text AS crown_date, model, lookback, feature_mode, config_key "
                "FROM gold_tertip_multi_horizon_crowns WHERE symbol IN (%s)" % ",".join(f"'{s}'" for s in STOCKS)
            )
            jobs = [(s, h, g.to_dict("records")) for (s, h), g in allc.groupby(["symbol", "horizon"])]
            res = [r for r in pool.map(task_a2, jobs) if not r.empty]
            pd.concat(res, ignore_index=True).to_parquet(OUT / "importance_honest.parquet")
            print(f"A2 done {time.time() - t0:.0f}s", file=sys.stderr)
        if args.part in ("all", "B"):
            pd.concat(pool.map(task_b, STOCKS), ignore_index=True).to_parquet(OUT / "ic.parquet")
            print(f"B done {time.time() - t0:.0f}s", file=sys.stderr)
        if args.part in ("all", "D"):
            parts = pool.map(task_d, STOCKS)
            pd.concat([p for p in parts if not p.empty], ignore_index=True).to_parquet(OUT / "recrowned_pruned.parquet")
            print(f"D done {time.time() - t0:.0f}s", file=sys.stderr)
        if args.part in ("all", "C"):
            parts = pool.map(task_c, [(s, h) for h in HORIZONS[::-1] for s in STOCKS])
            pd.concat([p for p in parts if not p.empty], ignore_index=True).to_parquet(OUT / "variants.parquet")
            print(f"C done {time.time() - t0:.0f}s", file=sys.stderr)


if __name__ == "__main__":
    main()
