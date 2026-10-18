#!/usr/bin/env python
"""Hit-rate experiments (per-stock models throughout), zero look-ahead, split-corrected data.

Part A - RECROWN: re-run the production tournament grid (5 models x 5 lookbacks x 3 feature modes) on corrected data.
    Champions are crowned on the first half of the evaluation window (only outcomes realized before the second half)
    and scored on the second half, against the currently crowned configuration on the same sessions.
    Split-affected stocks plus a control group of unaffected stocks (to separate "data fix" from selection noise).

Part B - RELATIVE: predict each stock's move relative to BIST 30 (beta = 1) with its crowned configuration.
    Adjusted prices are divided by the index level, so targets, momentum / trend / ATR features and realized returns
    become market-relative. Scored on (i) relative direction (the flow model's own skill) and (ii) absolute direction
    when the relative forecast is used as the absolute forecast (index forecast = 0).

Usage:
    .venv/bin/python scripts/experiments/hit_rate_experiments.py --part recrown [--workers 14]
    .venv/bin/python scripts/experiments/hit_rate_experiments.py --part relative [--workers 14]
    .venv/bin/python scripts/experiments/hit_rate_experiments.py --part report
"""

from __future__ import annotations

import argparse
import os

os.environ.setdefault("OMP_NUM_THREADS", "1")

import sys  # noqa: E402
import time  # noqa: E402
import warnings  # noqa: E402
from multiprocessing import Pool  # noqa: E402
from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import yaml  # noqa: E402

warnings.filterwarnings("ignore")
sys.path.insert(0, str(Path(__file__).resolve().parent))

SPLIT_STOCKS = ["FROTO", "AEFES", "EREGL", "PGSUS", "ISCTR", "BIMAS", "SASA", "TUPRS", "ASELS", "PETKM"]
CONTROL_STOCKS = ["AKBNK", "GARAN", "THYAO", "KCHOL", "TCELL"]
MODELS = ["Ridge", "XGBoost", "LightGBM", "BayesianRidge", "Huber"]
LOOKBACKS = [63, 126, 252, 378, 504]
MODES = ["FIFO_CORE", "YTD_SUBSTITUTION", "FIFO_PLUS_YTD"]
MH_KEYS = ["3d", "5d", "10d", "15d", "30d"]
T1_EVAL, MH_EVAL = 120, 180
OUT_DIR = Path(os.environ.get("HR_OUT", Path.home() / "data" / "mdk_oracle" / "experiments" / "hit_rate"))


def _crowned(sym: str) -> tuple[dict, dict]:
    t1 = yaml.safe_load(open("config/tertip_crowned_models.yaml"))["symbols"].get(sym, {})
    mh = yaml.safe_load(open("config/tertip_multi_horizon_crowned_models.yaml"))["symbols"].get(sym, {})
    return t1, mh


def t1_walk_forward(df: pd.DataFrame, model_name: str, lb: int, f_cols: list[str], n_eval: int = T1_EVAL) -> list[dict]:
    """Production T+1 arena replica on split-adjusted (or relative) closes."""
    import model_candidate_arena as arena

    from mdk_trading_oracle.data.silver.tertip_ml_forecaster import (
        _check_hit,
        adjusted_price,
        calibrate_expected_return,
    )

    df = df.reset_index(drop=True)
    for c in f_cols:
        if c not in df.columns:
            df[c] = 0.0
    adj = adjusted_price(df, "close")
    nxt = (adj.shift(-1) / adj - 1.0) * 100.0
    X_all = df[f_cols]
    rows = []
    for idx in range(len(df) - n_eval, len(df)):
        o = idx - 1
        start = max(0, o - lb)
        tr = np.arange(start, o)
        tr = tr[np.isfinite(nxt.iloc[tr].to_numpy())]
        tr = tr[~X_all.iloc[tr].isna().any(axis=1).to_numpy()]
        if len(tr) < 63:
            continue
        y = nxt.iloc[tr].to_numpy()
        w = np.where(np.abs(y) < 1.0, 0.50, np.where(np.abs(y) < 2.0, 3.50, 6.00))
        try:
            model, wkey = arena.build_t1_current(model_name)
            model.fit(X_all.iloc[tr], y, **{wkey: w})
            raw = float(model.predict(X_all.iloc[o : o + 1])[0])
        except Exception:
            raw = 0.0
        actual = float(nxt.iloc[o])
        pred = calibrate_expected_return(max(-10.0, min(10.0, raw)), float(df["atr_14d_pct"].iloc[o]))
        rows.append({"trade_date": str(df["trade_date"].iloc[o])[:10], "pred": pred, "actual": actual,
                     "hit": bool(_check_hit(pred, actual)), "err": abs((1 + pred / 100) / (1 + actual / 100) - 1) * 100})
    return rows


def mh_walk_forward(df: pd.DataFrame, h_key: str, model_name: str, lb: int, mode: str) -> list[dict]:
    from mdk_trading_oracle.data.silver.tertip_multi_horizon_forecaster import evaluate_horizon_walk_forward

    res = evaluate_horizon_walk_forward(df.copy(), h_key, model_name, lb, MH_EVAL, mode)
    return [{"trade_date": r["trade_date"], "pred": r["pred_return_pct"], "actual": r["actual_return_pct"],
             "hit": bool(r["is_hit"]), "err": r["err_pct"]} for r in res.get("ledger", [])]


# ── Part A ────────────────────────────────────────────────────────────────────


def run_recrown(sym: str) -> pd.DataFrame:
    from mdk_trading_oracle.core.db import PostgresManager
    from mdk_trading_oracle.data.silver.tertip_ml_forecaster import (
        extract_3pillar_time_series,
        get_feature_cols_for_mode,
    )

    t0 = time.time()
    db = PostgresManager(read_only=True)
    t1_cfg, mh_cfg = _crowned(sym)
    out = []
    df_mh = extract_3pillar_time_series(db, sym, lookback_days=550)
    df_t1 = extract_3pillar_time_series(db, sym, lookback_days=int((max(LOOKBACKS) + T1_EVAL + 60) * 1.5))
    for model in MODELS:
        for lb in LOOKBACKS:
            for mode in MODES:
                tag = {"symbol": sym, "model": model, "lookback": lb, "mode": mode}
                for r in t1_walk_forward(df_t1, model, lb, get_feature_cols_for_mode(mode)):
                    out.append({**tag, "horizon": "1d", **r})
                for h in MH_KEYS:
                    for r in mh_walk_forward(df_mh, h, model, lb, mode):
                        out.append({**tag, "horizon": h, **r})
    res = pd.DataFrame(out)
    res["is_current"] = False
    if t1_cfg:
        m = (res.horizon == "1d") & (res.model == t1_cfg["model"]) & (res.lookback == int(t1_cfg.get("training_lookback_sessions", 252))) & (res["mode"] == t1_cfg.get("feature_mode", "FIFO_CORE"))
        res.loc[m, "is_current"] = True
    for h, c in mh_cfg.items():
        m = (res.horizon == h) & (res.model == c["model"]) & (res.lookback == int(c.get("training_lookback_sessions", 126))) & (res["mode"] == c.get("feature_mode", "FIFO_CORE"))
        res.loc[m, "is_current"] = True
    (OUT_DIR / "recrown").mkdir(parents=True, exist_ok=True)
    res.to_parquet(OUT_DIR / "recrown" / f"{sym}.parquet", index=False)
    print(f"  {sym}: {len(res)} rows in {time.time() - t0:.0f}s", flush=True)
    return res


# ── Part B ────────────────────────────────────────────────────────────────────


def _relative_frame(df: pd.DataFrame, index_close: pd.Series) -> pd.DataFrame:
    rel = df.copy()
    idx = rel["trade_date"].astype(str).str[:10].map(index_close).astype(float).ffill()
    scale = 10000.0 / idx
    for col in ["adj_open_price", "adj_high_price", "adj_low_price", "adj_close_price"]:
        rel[col] = rel[col].astype(float) * scale
    return rel


def run_relative(sym: str) -> pd.DataFrame:
    from mdk_trading_oracle.core.db import PostgresManager
    from mdk_trading_oracle.data.silver.tertip_ml_forecaster import (
        adjusted_price,
        extract_3pillar_time_series,
        get_feature_cols_for_mode,
    )
    from mdk_trading_oracle.data.silver.tertip_multi_horizon_forecaster import HORIZON_MAP

    t0 = time.time()
    db = PostgresManager(read_only=True)
    t1_cfg, mh_cfg = _crowned(sym)
    max_stock_date = db.execute("SELECT MAX(trade_date) FROM silver_daily_stock_summary").fetchone()[0]
    ix = db.execute(
        "SELECT trade_date, close_price FROM silver_daily_benchmark_index WHERE index_code = 'XU030' AND trade_date <= %s",
        (max_stock_date,),
    ).fetchall()
    index_close = pd.Series({str(d)[:10]: float(c) for d, c in ix})
    out = []

    def _absolute_returns(df: pd.DataFrame, n: int) -> dict:
        adj = adjusted_price(df.reset_index(drop=True), "close")
        fwd = (adj.shift(-n) / adj - 1.0) * 100.0
        return dict(zip(df["trade_date"].astype(str).str[:10], fwd))

    if t1_cfg:
        lb = int(t1_cfg.get("training_lookback_sessions", 252))
        df = extract_3pillar_time_series(db, sym, lookback_days=int((lb + T1_EVAL + 60) * 1.5))
        abs_ret = _absolute_returns(df, 1)
        for r in t1_walk_forward(_relative_frame(df, index_close), t1_cfg["model"], lb, get_feature_cols_for_mode(t1_cfg.get("feature_mode", "FIFO_CORE"))):
            out.append({"symbol": sym, "horizon": "1d", **r, "actual_abs": abs_ret.get(r["trade_date"])})
    if mh_cfg:
        df = extract_3pillar_time_series(db, sym, lookback_days=550)
        rel = _relative_frame(df, index_close)
        for h, c in mh_cfg.items():
            abs_ret = _absolute_returns(df, HORIZON_MAP[h])
            for r in mh_walk_forward(rel, h, c["model"], int(c.get("training_lookback_sessions", 126)), c.get("feature_mode", "FIFO_CORE")):
                out.append({"symbol": sym, "horizon": h, **r, "actual_abs": abs_ret.get(r["trade_date"])})
    res = pd.DataFrame(out)
    (OUT_DIR / "relative").mkdir(parents=True, exist_ok=True)
    res.to_parquet(OUT_DIR / "relative" / f"{sym}.parquet", index=False)
    print(f"  {sym}: {len(res)} rows in {time.time() - t0:.0f}s", flush=True)
    return res


# ── Report ────────────────────────────────────────────────────────────────────

H = ["1d", "3d", "5d", "10d", "15d", "30d"]
N = {"1d": 1, "3d": 3, "5d": 5, "10d": 10, "15d": 15, "30d": 30}


def _deadband(h: str) -> float:
    from mdk_trading_oracle.data.silver.tertip_multi_horizon_forecaster import get_horizon_deadband

    return 0.25 if h == "1d" else get_horizon_deadband(N[h])


def strict(pred: pd.Series, actual: pd.Series, db: float) -> float:
    called = pred.abs() > db
    ok = np.where(pred > 0, actual > db, actual < -db)
    return float(ok[called.to_numpy()].mean() * 100) if called.any() else float("nan")


def report_recrown() -> str:
    import model_candidate_arena as arena

    files = sorted((OUT_DIR / "recrown").glob("*.parquet"))
    if not files:
        return ""
    res = pd.concat([pd.read_parquet(p) for p in files], ignore_index=True)
    rows = []
    for (sym, h), g in res.groupby(["symbol", "horizon"]):
        dates = sorted(g["trade_date"].unique())
        split = dates[len(dates) // 2]
        crown_end = dates[max(0, len(dates) // 2 - N[h])]  # outcomes realized before the second half starts
        first = g[g["trade_date"] < crown_end]
        second = g[g["trade_date"] >= split]
        losses = first.groupby(["model", "lookback", "mode"]).apply(arena.composite_loss)
        champ = losses.idxmin()
        new = second.set_index(["model", "lookback", "mode"]).loc[champ]
        cur = second[second["is_current"]]
        db = _deadband(h)
        rows.append({
            "symbol": sym, "horizon": h, "group": "split" if sym in SPLIT_STOCKS else "control",
            "new_config": f"{champ[0]}/{champ[1]}/{champ[2]}", "n": len(new),
            "cur_strict": strict(cur["pred"], cur["actual"], db), "new_strict": strict(new["pred"], new["actual"], db),
            "cur_lenient": cur["hit"].mean() * 100, "new_lenient": new["hit"].mean() * 100,
        })
    t = pd.DataFrame(rows)
    t.to_csv(OUT_DIR / "recrown_summary.csv", index=False)
    lines = ["## Part A - Re-crowning on corrected data (second-half, out-of-sample for the new crown)\n"]
    for grp in ["split", "control"]:
        s = t[t.group == grp].groupby("horizon")[["cur_strict", "new_strict", "cur_lenient", "new_lenient"]].mean().reindex(H)
        lines.append(f"### {grp} stocks (mean across stocks)\n\n{s.round(1).to_string()}\n")
    per = t[t.group == "split"].pivot(index="symbol", columns="horizon", values=["cur_strict", "new_strict"])
    lines.append(f"### Split stocks per horizon (strict %)\n\n{per.round(0).to_string()}\n")
    return "\n".join(lines)


def report_relative() -> str:
    files = sorted((OUT_DIR / "relative").glob("*.parquet"))
    base_dir = Path.home() / "data" / "mdk_oracle" / "experiments" / "direction_size"
    if not files:
        return ""
    rel = pd.concat([pd.read_parquet(p) for p in files], ignore_index=True)
    cur = pd.concat([pd.read_parquet(p) for p in base_dir.glob("*.parquet")], ignore_index=True)
    cur = cur[["symbol", "horizon", "trade_date", "old_pred", "actual"]].rename(columns={"actual": "cur_actual_abs"})
    m = rel.merge(cur, on=["symbol", "horizon", "trade_date"], how="inner")
    rows = []
    for h in H:
        g = m[m.horizon == h].dropna(subset=["actual_abs"])
        db = _deadband(h)
        rows.append({
            "horizon": h, "n": len(g),
            "current_model_abs": strict(g["old_pred"], g["actual_abs"], db),
            "current_model_on_relative": strict(g["old_pred"], g["actual"], db),
            "relative_model_on_relative": strict(g["pred"], g["actual"], db),
            "relative_model_as_abs": strict(g["pred"], g["actual_abs"], db),
            "random_relative": float(((g["actual"] > db) | (g["actual"] < -db)).mean() * 50),
            "random_abs": float(((g["actual_abs"] > db) | (g["actual_abs"] < -db)).mean() * 50),
        })
    t = pd.DataFrame(rows).set_index("horizon")
    return f"## Part B - Market-relative target (strict direction %, same sessions)\n\n{t.round(1).to_string()}\n"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--part", choices=["recrown", "relative", "report"], required=True)
    ap.add_argument("--symbols", default="")
    ap.add_argument("--workers", type=int, default=14)
    args = ap.parse_args()
    if args.part != "report":
        if args.symbols:
            symbols = [s.strip().upper() for s in args.symbols.split(",") if s.strip()]
        elif args.part == "recrown":
            symbols = SPLIT_STOCKS + CONTROL_STOCKS
        else:
            symbols = sorted(yaml.safe_load(open("config/tertip_crowned_models.yaml"))["symbols"])
        fn = run_recrown if args.part == "recrown" else run_relative
        print(f"{args.part}: {len(symbols)} symbols, {args.workers} workers -> {OUT_DIR}", flush=True)
        with Pool(processes=min(args.workers, len(symbols))) as pool:
            pool.map(fn, symbols, chunksize=1)
    md = "\n".join(x for x in [report_recrown(), report_relative()] if x)
    (OUT_DIR / "REPORT.md").write_text(md, encoding="utf-8")
    print(md)


if __name__ == "__main__":
    main()
