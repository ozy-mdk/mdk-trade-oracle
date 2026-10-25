#!/usr/bin/env python
"""Honest 3D-30D track records for the focus stocks (isolated experiment, read-only on the database).

Per-session forecasts come from production's `evaluate_horizon_walk_forward` (trains on rows i <= t - n only; see
tests/test_multi_horizon_leakage.py), for all 75 tournament candidates x 5 horizons, every session since --start.
On that prediction grid:
    crowned   - the configs in config/tertip_multi_horizon_crowned_models.yaml, split into the crowning window
                (last 90 sessions realized before the crowning date) and the sessions before it
    hub_180   - the crowned configs on the hub's 180 displayed sessions (should reproduce gold)
    monthly   - production's tournament rule re-run on the first trading day of every month using only outcomes
                realized before that day; each champion graded on that month's origins only (models/horizon_crowning.py)
    fixed     - Ridge / 252 / FIFO_CORE, never selected
    avg_cfg   - mean over all 75 candidates (what selection is choosing among)
    hindsight - best candidate per stock/horizon over the scored period (upper bound of selection luck)
Baselines: random (monthly-crowning predictions shuffled across sessions) and always-UP.

Outputs (under ~/data/mdk_oracle/experiments/multi_horizon_ledger/):
    frames/<SYM>.parquet   feature frames used
    grid.parquet           one row per (symbol, horizon, config, session)
    summary.csv            all tables of the report

Usage:
    .venv/bin/python scripts/experiments/multi_horizon_ledger_rebuild.py                  # build + report
    .venv/bin/python scripts/experiments/multi_horizon_ledger_rebuild.py --report-only    # report from grid.parquet
"""

from __future__ import annotations

import os

# BLAS / OpenMP threads must be pinned before numpy loads (18 OpenBLAS threads x workers slows fits 100x+)
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
import yaml  # noqa: E402

FOCUS = ["ASELS", "GARAN", "AKBNK", "YKBNK", "ISCTR", "TTKOM"]
OUT_DIR = Path.home() / "data" / "mdk_oracle" / "experiments" / "multi_horizon_ledger"
CROWNED_PATH = Path("config/tertip_multi_horizon_crowned_models.yaml")
CROWNED_AT = "2026-10-10"  # _metadata.calibrated_at of the crowned yaml
GRID_START = "2023-07-01"  # first graded origin in the grid (crowning windows need ~90 realized rows before 2024-01)
SCORE_START = "2024-01-01"  # first month of the simulated monthly crowning track
HUB_SESSIONS = 180
FIXED_KEY = "Ridge/252/FIFO_CORE"
N_PERM = 200
N_BOOT = 1000
HORIZON_DAYS = {"3d": 3, "5d": 5, "10d": 10, "15d": 15, "30d": 30}


def deadband(h: str) -> float:
    from mdk_trading_oracle.data.silver.tertip_multi_horizon_forecaster import get_horizon_deadband

    return get_horizon_deadband(HORIZON_DAYS[h])


# ----------------------------------------------------------------------------------------------------------------------
# Build


def build_frame(sym: str) -> str:
    from mdk_trading_oracle.core.db import PostgresManager
    from mdk_trading_oracle.data.silver.tertip_ml_forecaster import extract_3pillar_time_series
    from mdk_trading_oracle.data.silver.tertip_multi_horizon_forecaster import engineer_multi_horizon_features

    db = PostgresManager(read_only=True)
    df = extract_3pillar_time_series(db, sym, lookback_days=5000)
    df, _ = engineer_multi_horizon_features(df, feature_mode="FIFO_PLUS_YTD")
    path = OUT_DIR / "frames" / f"{sym}.parquet"
    df.to_parquet(path)
    return str(path)


def run_task(args: tuple[str, str, str]) -> pd.DataFrame:
    sym, horizon, model = args
    from mdk_trading_oracle.data.silver.tertip_multi_horizon_forecaster import evaluate_horizon_walk_forward
    from mdk_trading_oracle.models.horizon_crowning import candidate_configs

    df = pd.read_parquet(OUT_DIR / "frames" / f"{sym}.parquet")
    n_eval = int((df["trade_date"].astype(str).str[:10] >= GRID_START).sum())
    rows = []
    for c in candidate_configs():
        if c.model != model:
            continue
        res = evaluate_horizon_walk_forward(
            df.copy(), horizon, c.model, train_lookback_sessions=c.lookback, n_eval_sessions=n_eval, feature_mode=c.feature_mode
        )
        for r in res["ledger"]:
            rows.append({
                "symbol": sym, "horizon": horizon, "config_key": c.key, "family": c.family, "model": c.model,
                "lookback": c.lookback, "feature_mode": c.feature_mode,
                "trade_date": r["trade_date"], "target_date_end": str(r["target_date_end"])[:10],
                "current_price": r["current_price"], "actual_return_pct": r["actual_return_pct"],
                "pred_return_pct": r["pred_return_pct"], "is_hit": bool(r["is_hit"]), "err_pct": r["err_pct"],
                "actual_training_sessions": r["actual_training_sessions"],
            })
    return pd.DataFrame(rows)


def build(symbols: list[str], workers: int) -> pd.DataFrame:
    from mdk_trading_oracle.models.horizon_crowning import FAMILIES, HORIZONS

    (OUT_DIR / "frames").mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    with Pool(min(workers, len(symbols))) as pool:
        pool.map(build_frame, symbols)
    print(f"frames built in {time.time() - t0:.0f}s", file=sys.stderr)
    # heavy models first for load balance
    order = ["Huber", "XGBoost", "LightGBM", "BayesianRidge", "Ridge"]
    models = [m for m in order if any(m in v for v in FAMILIES.values())]
    tasks = [(s, h, m) for m in models for s in symbols for h in HORIZONS]
    parts = []
    with Pool(workers) as pool:
        for i, part in enumerate(pool.imap_unordered(run_task, tasks), 1):
            parts.append(part)
            if i % 10 == 0:
                print(f"  {i}/{len(tasks)} tasks, {time.time() - t0:.0f}s", file=sys.stderr)
    grid = pd.concat(parts, ignore_index=True)
    grid.to_parquet(OUT_DIR / "grid.parquet")
    print(f"wrote {len(grid)} rows in {time.time() - t0:.0f}s", file=sys.stderr)
    return grid


# ----------------------------------------------------------------------------------------------------------------------
# Tracks


def crowned_configs() -> dict[str, dict[str, str]]:
    cfg = yaml.safe_load(open(CROWNED_PATH))["symbols"]
    out = {}
    for sym, hz in cfg.items():
        out[sym] = {
            h: f"{v['model']}/{int(v.get('training_lookback_sessions', v.get('lookback', 126)))}/{v.get('feature_mode', 'FIFO_CORE')}"
            for h, v in hz.items()
        }
    return out


def tracks(grid: pd.DataFrame) -> pd.DataFrame:
    """All graded tracks in one frame with a `track` column (one row per symbol, horizon, session)."""
    from mdk_trading_oracle.models.horizon_crowning import CROWN_EVAL_SESSIONS, month_starts, simulate_crowning

    crowned = crowned_configs()
    out = []
    for sym, g in grid.groupby("symbol"):
        # current crowns, labelled by selection status
        for h, key in crowned[sym].items():
            c = g[(g["horizon"] == h) & (g["config_key"] == key)].sort_values("trade_date")
            realized_before = c[c["target_date_end"] < CROWNED_AT]
            window = set(realized_before["trade_date"].iloc[-CROWN_EVAL_SESSIONS:])
            status = np.where(
                c["trade_date"].isin(window), "IN_CROWNING_WINDOW",
                np.where(c["target_date_end"] >= CROWNED_AT, "POST_CROWNING", "PRE_CROWNING"),
            )
            c = c.assign(selection_status=status)
            out.append(c.assign(track="crowned"))
            out.append(c.iloc[-HUB_SESSIONS:].assign(track="hub_180"))
        out.append(g[g["config_key"] == FIXED_KEY].assign(track="fixed"))
        dates = g["trade_date"].drop_duplicates()
        sim = simulate_crowning(g, month_starts(dates, SCORE_START))
        out.append(sim.assign(track="monthly"))
    return pd.concat(out, ignore_index=True)


# ----------------------------------------------------------------------------------------------------------------------
# Metrics


def hub_hit(pred: np.ndarray, actual: np.ndarray, db: float, tol: float = 0.25) -> np.ndarray:
    """Vectorised production `_check_hit` (the hub's hit rule)."""
    near = np.abs(pred - actual) <= tol
    flat = np.abs(actual) <= db
    flat_ok = (np.abs(pred) <= db) | (pred * actual >= 0.0)
    trend_ok = np.where(actual > db, pred > 0.0, pred < 0.0)
    return near | np.where(flat, flat_ok, trend_ok)


def metrics(pred: np.ndarray, actual: np.ndarray, h: str) -> dict[str, float]:
    db = deadband(h)
    called = np.abs(pred) > db
    strict = ((pred > db) & (actual > db)) | ((pred < -db) & (actual < -db))
    big = np.abs(actual) >= 2.0 * np.sqrt(HORIZON_DAYS[h])
    sign_ok = np.sign(pred) == np.sign(actual)
    return {
        "n": len(pred),
        "hub_hit_pct": hub_hit(pred, actual, db).mean() * 100,
        "strict_pct": strict.mean() * 100,
        "strict_on_calls_pct": strict[called].mean() * 100 if called.any() else np.nan,
        "call_rate_pct": called.mean() * 100,
        "big_move_sign_pct": sign_ok[big].mean() * 100 if big.any() else np.nan,
        "mae_pct": np.abs(pred - actual).mean(),
        "up_share_pct": (pred > db).mean() * 100,
    }


def date_block_ci(d: pd.DataFrame, h: str, rng: np.random.Generator) -> tuple[float, float]:
    """95% block-bootstrap CI of the pooled hub hit rate; blocks of consecutive dates (overlapping labels)."""
    db = deadband(h)
    per_date = d.assign(hit=hub_hit(d["pred_return_pct"].to_numpy(), d["actual_return_pct"].to_numpy(), db)).groupby("trade_date")["hit"].agg(["sum", "count"])
    s, c = per_date["sum"].to_numpy(float), per_date["count"].to_numpy(float)
    n = len(s)
    block = max(10, 2 * HORIZON_DAYS[h])
    if n < 2 * block:
        return (np.nan, np.nan)
    nb = int(np.ceil(n / block))
    starts = rng.integers(0, n - block + 1, size=(N_BOOT, nb))
    idx = (starts[:, :, None] + np.arange(block)).reshape(N_BOOT, -1)[:, :n]
    rates = s[idx].sum(1) / c[idx].sum(1) * 100
    return tuple(np.percentile(rates, [2.5, 97.5]))


def baselines(d: pd.DataFrame, h: str, rng: np.random.Generator) -> dict[str, float]:
    db = deadband(h)
    actual = d["actual_return_pct"].to_numpy()
    perm = []
    for _ in range(N_PERM):
        p = np.concatenate([rng.permutation(g["pred_return_pct"].to_numpy()) for _, g in d.groupby("symbol")])
        a = np.concatenate([g["actual_return_pct"].to_numpy() for _, g in d.groupby("symbol")])
        perm.append(hub_hit(p, a, db).mean() * 100)
    up = np.full_like(actual, 2.0 * db)
    return {"random_hub_hit_pct": float(np.mean(perm)), "always_up_hub_hit_pct": hub_hit(up, actual, db).mean() * 100}


def table(t: pd.DataFrame, group_cols: list[str], with_ci: bool = False) -> pd.DataFrame:
    rng = np.random.default_rng(7)
    rows = []
    for keys, g in t.groupby(group_cols, sort=False):
        keys = keys if isinstance(keys, tuple) else (keys,)
        h = dict(zip(group_cols, keys))["horizon"]
        rec = {**dict(zip(group_cols, keys)), **metrics(g["pred_return_pct"].to_numpy(), g["actual_return_pct"].to_numpy(), h)}
        if with_ci:
            rec["ci_lo"], rec["ci_hi"] = date_block_ci(g, h, rng)
            rec.update(baselines(g, h, rng))
        rows.append(rec)
    return pd.DataFrame(rows)


# ----------------------------------------------------------------------------------------------------------------------
# Report


def reproduce_gold(t: pd.DataFrame) -> None:
    from mdk_trading_oracle.core.db import PostgresManager

    db = PostgresManager(read_only=True)
    gold = db.query_df(
        "SELECT symbol, horizon, trade_date::text AS trade_date, pred_return_pct AS gold_pred, is_hit AS gold_hit "
        "FROM gold_tertip_multi_horizon_backtests WHERE NOT is_pending AND symbol IN (%s)" % ",".join(f"'{s}'" for s in FOCUS)
    )
    m = t[t["track"] == "hub_180"].merge(gold, on=["symbol", "horizon", "trade_date"])
    if m.empty:
        print("\n(no gold rows to compare)")
        return
    m["same_hit"] = m["is_hit"] == m["gold_hit"]
    m["same_dir"] = np.sign(m["pred_return_pct"]) == np.sign(m["gold_pred"])
    print("\n=== reproduction of the gold ledger (crowned config, hub sessions) ===")
    print(m.groupby("horizon")[["same_hit", "same_dir"]].mean().mul(100).round(1).assign(n=m.groupby("horizon").size()).to_string())


def main_report(grid: pd.DataFrame) -> None:
    from mdk_trading_oracle.models.horizon_crowning import HORIZONS

    pd.set_option("display.width", 250)
    pd.set_option("display.max_columns", 40)
    t = tracks(grid)
    reproduce_gold(t)
    scored = t[t["trade_date"] >= SCORE_START]
    last_dates = {h: sorted(grid.loc[grid["horizon"] == h, "trade_date"].unique())[-HUB_SESSIONS:] for h in HORIZONS}
    frames = []

    def show(title: str, tt: pd.DataFrame, cols: list[str], ci: bool = False) -> pd.DataFrame:
        s = table(tt, cols, with_ci=ci)
        s["horizon"] = pd.Categorical(s["horizon"], HORIZONS, ordered=True)
        s = s.sort_values([c for c in cols if c != "horizon"] + ["horizon"])
        print(f"\n=== {title} ===")
        print(s.round(1).to_string(index=False))
        frames.append(s.assign(table=title))
        return s

    show("Hub display: crowned config on the 180 displayed sessions (ALL6 pooled)", t[t["track"] == "hub_180"], ["horizon"])
    show(
        "Current crowns by selection status (ALL6 pooled, sessions since 2024)",
        scored[scored["track"] == "crowned"], ["selection_status", "horizon"],
    )
    show(f"Simulated monthly crowning, {SCORE_START[:4]}-onward (ALL6 pooled)", scored[scored["track"] == "monthly"], ["horizon"], ci=True)
    show(f"Fixed {FIXED_KEY}, never selected (ALL6 pooled)", scored[scored["track"] == "fixed"], ["horizon"], ci=True)

    g = grid[grid["trade_date"] >= SCORE_START]
    cfg_rates = g.assign(hit=g["is_hit"].astype(float)).groupby(["symbol", "horizon", "config_key"])["hit"].mean().mul(100)
    agg = cfg_rates.groupby("horizon").agg(avg_cfg_hit_pct="mean")
    agg["hindsight_best_hit_pct"] = cfg_rates.groupby(["symbol", "horizon"]).max().groupby("horizon").mean()
    print("\n=== Selection pool: mean over all 75 candidates vs best in hindsight (hub hit %, mean over stocks) ===")
    print(agg.reindex(HORIZONS).round(1).to_string())

    mon = scored[scored["track"] == "monthly"].copy()
    mon["period"] = mon["trade_date"].str[:4]
    show("Simulated monthly crowning by year (ALL6 pooled)", mon, ["period", "horizon"])
    hub_like = pd.concat([mon[(mon["horizon"] == h) & mon["trade_date"].isin(last_dates[h])] for h in HORIZONS])
    show("Simulated monthly crowning on the hub's 180 sessions (what an honest hub would show)", hub_like, ["horizon"], ci=True)
    show("Simulated monthly crowning per stock", mon, ["symbol", "horizon"])

    fam = mon.drop_duplicates(["symbol", "crown_date"]).groupby("crowned_family").size()
    print("\ncrowned family over all monthly crownings:", fam.to_dict())
    pd.concat(frames, ignore_index=True).to_csv(OUT_DIR / "summary.csv", index=False)
    print(f"\nsummary -> {OUT_DIR / 'summary.csv'}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--symbols", default=",".join(FOCUS))
    ap.add_argument("--workers", type=int, default=16)
    ap.add_argument("--report-only", action="store_true")
    args = ap.parse_args()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    if args.report_only:
        grid = pd.read_parquet(OUT_DIR / "grid.parquet")
    else:
        grid = build([s.strip().upper() for s in args.symbols.split(",") if s.strip()], args.workers)
    main_report(grid)


if __name__ == "__main__":
    main()
