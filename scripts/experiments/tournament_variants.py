#!/usr/bin/env python
"""Improvement candidates for the honest re-crowned tournament, tested end-to-end (isolated, read-only on the DB).

Each variant runs the production pipeline unchanged except for its one idea: every one of the 75 candidates
(5 models x 5 lookbacks x 3 feature modes) is walked forward point-in-time over the trailing 300 origins for all six
horizons (swing: production `evaluate_horizon_walk_forward`; 1D: production `forecast_t1`), champions are crowned on
each month's first trading day from realized outcomes only (`models/horizon_crowning.py`), and each of the last 90
graded sessions is scored with the champion in force that day. Production itself is read from
gold_tertip_multi_horizon_candidates (same code, same data), so every variant is compared on identical sessions.

Variants
    production          gold candidates, all 75 configs
    no_3m               production pool without the 63-session lookback (15 configs fewer)
    no_sma50            feature feat_price_vs_sma50_pct removed
    regime              regime-conditional mean reversion: BIST 30 downtrend flag (below SMA50 and 20D return < 0),
                        trend strength (BIST 30 vs SMA50), and mean-reversion features x downtrend flag
    regime_no_sma50     regime + no_sma50
    combo               regime + no_sma50, crowned without the 63-session lookback
    ens5                production pool; each month the forecasts of the 5 best candidates (same ranking) are averaged
    ens5_regime         ens5 on the regime_no_sma50 pool
    events              production features + event calendar: TCMB MPC decision distance / last change,
                        ex-dividend distance and yield, BIST 30 rebalancing distance and index entry / exit
    hit150              production pool; each month the candidate with the best plain hit rate over the last 150
                        realized sessions (MAE tie-break, no recency buckets, no family stage)

Stocks: focus six + THYAO (--symbols to scale). Frames: feature_importance_experiment.py (production features +
market features, data through the latest silver session).

Outputs: ~/data/mdk_oracle/experiments/tournament_variants/{candidates_<variant>_<SYM>.parquet, tracks.parquet}
Usage:   .venv/bin/python scripts/experiments/tournament_variants.py [--symbols ...] [--variants ...] [--report-only]
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
FRAMES = Path.home() / "data" / "mdk_oracle" / "experiments" / "feature_importance" / "frames"
OUT = Path.home() / "data" / "mdk_oracle" / "experiments" / "tournament_variants"
HISTORY = 300
DISPLAY = 90

MEAN_REVERSION = [
    "feat_price_vs_sma20_pct", "feat_price_vs_sma50_pct", "feat_sma20_vs_sma50_pct", "feat_close_ret_20d",
    "feat_big5_cost_spread_pct", "feat_mlb_cost_spread_pct", "feat_big5_ytd_pnl_share_20d_to", "feat_mlb_tertip_share",
]
REGIME_COLS = ["feat_regime_down", "feat_regime_trend"]

# variant -> (feature transform, configs to keep)
COMPUTED = ["no_sma50", "regime", "regime_no_sma50", "events"]
DERIVED = {"no_3m": ("production", 63), "combo": ("regime_no_sma50", 63)}
ENSEMBLE = {"ens5": "production", "ens5_regime": "regime_no_sma50"}
# Per-horizon policies: horizon -> (candidate ledger source, excluded lookback or None). Crowned jointly, as in production.
POLICIES = {
    "policy_a": {"1d": ("events", None), "3d": ("events", None), "5d": ("regime_no_sma50", None),
                 "10d": ("regime_no_sma50", None), "15d": ("production", 63), "30d": ("production", None)},
    # policy_a for 1D-15D; 30D keeps production's own crowning (crowned separately, so its family is unaffected)
    "policy_c": {"1d": ("events", None), "3d": ("events", None), "5d": ("regime_no_sma50", None),
                 "10d": ("regime_no_sma50", None), "15d": ("production", 63), "30d": ("production_track", None)},
    "policy_b": {"1d": ("events", None), "3d": ("events", None), "5d": ("production", 63),
                 "10d": ("regime_no_sma50", None), "15d": ("production", 63), "30d": ("production", None)},
}
CROWN_WINDOW = {"hit150": ("production", 150)}


def regime(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    down = ((df["mkt_bist30_vs_sma50"] < 0) & (df["mkt_bist30_ret_20d"] < 0)).astype(float)
    df["feat_regime_down"] = down
    df["feat_regime_trend"] = df["mkt_bist30_vs_sma50"].clip(-15, 15).fillna(0.0)
    for c in MEAN_REVERSION:
        if c in df.columns:
            df[f"{c}_x_down"] = (df[c].fillna(0.0) * down).astype(float)
    return df


_EVENTS: dict | None = None


def _event_tables() -> dict:
    global _EVENTS
    if _EVENTS is None:
        from mdk_trading_oracle.core.db import PostgresManager

        db = PostgresManager(read_only=True)
        _EVENTS = {
            "mpc": db.query_df("SELECT meeting_date::text AS d FROM bronze_tcmb_mpc_meetings WHERE meeting_type = 'scheduled' ORDER BY 1")["d"].tolist(),
            "rates": db.query_df("SELECT trade_date::text AS d, rate_change FROM silver_daily_macro_rates ORDER BY 1"),
            "div": db.query_df("SELECT symbol, ex_date::text AS d, gross_dividend_tl FROM bronze_dividends ORDER BY 1, 2"),
            "rebal": db.query_df("SELECT effective_date::text AS d, COALESCE(in_symbols, '') AS ins, COALESCE(out_symbols, '') AS outs FROM bronze_bist30_changes ORDER BY 1"),
        }
    return _EVENTS


EVENT_COLS = [
    "feat_evt_mpc_sessions_to_next", "feat_evt_mpc_sessions_since_last", "feat_evt_mpc_next_session",
    "feat_evt_mpc_last_change_bps", "feat_evt_div_sessions_to_next", "feat_evt_div_yield_next_pct",
    "feat_evt_rebal_sessions_to_next", "feat_evt_rebal_entering", "feat_evt_rebal_leaving",
]


def add_event_features(df: pd.DataFrame, sym: str) -> pd.DataFrame:
    """Event-calendar features at the close of each row t (only dates known in advance are looked ahead to)."""
    ev = _event_tables()
    df = df.copy()
    dates = pd.to_datetime(df["trade_date"]).dt.strftime("%Y-%m-%d").tolist()
    pos = {d: i for i, d in enumerate(dates)}
    n = len(dates)

    def sessions_until(target: str, i: int) -> float:
        """Trading sessions from row i to the first session on/after `target` (business days beyond the data)."""
        j = next((k for k in range(i + 1, n) if dates[k] >= target), None)
        if j is not None:
            return float(j - i)
        return float(i - n + 1 + len(pd.bdate_range(dates[-1], target)) - 1) if target > dates[-1] else 0.0

    mpc = ev["mpc"]
    rc = dict(zip(ev["rates"]["d"], ev["rates"]["rate_change"].fillna(0.0)))
    div = ev["div"][ev["div"]["symbol"] == sym]
    rebal = ev["rebal"]
    out = {c: np.zeros(n) for c in EVENT_COLS}
    close = df["close_price"].to_numpy(float)
    for i, d in enumerate(dates):
        nxt = next((m for m in mpc if m > d), None)
        prv = next((m for m in reversed(mpc) if m <= d), None)
        out["feat_evt_mpc_sessions_to_next"][i] = min(30.0, sessions_until(nxt, i)) if nxt else 30.0
        out["feat_evt_mpc_next_session"][i] = 1.0 if nxt and out["feat_evt_mpc_sessions_to_next"][i] == 1.0 else 0.0
        if prv:
            k = next((j for j in range(i, -1, -1) if dates[j] <= prv), None)
            out["feat_evt_mpc_sessions_since_last"][i] = min(30.0, float(i - k)) if k is not None else 30.0
            after = next((x for x in dates if x > prv), None)  # rate change shows on the session after the decision
            out["feat_evt_mpc_last_change_bps"][i] = rc.get(after, 0.0) * 100.0 if after and after <= d else 0.0
        else:
            out["feat_evt_mpc_sessions_since_last"][i] = 30.0
        nd = div[div["d"] > d]
        if len(nd):
            k = sessions_until(nd["d"].iloc[0], i)
            out["feat_evt_div_sessions_to_next"][i] = min(20.0, k)
            out["feat_evt_div_yield_next_pct"][i] = nd["gross_dividend_tl"].iloc[0] / close[i] * 100.0 if k <= 20 and close[i] > 0 else 0.0
        else:
            out["feat_evt_div_sessions_to_next"][i] = 20.0
        nr = rebal[rebal["d"] > d]
        if len(nr):
            k = sessions_until(nr["d"].iloc[0], i)
            out["feat_evt_rebal_sessions_to_next"][i] = min(30.0, k)
            if k <= 10:
                out["feat_evt_rebal_entering"][i] = 1.0 if sym in [x.strip() for x in nr["ins"].iloc[0].split(",")] else 0.0
                out["feat_evt_rebal_leaving"][i] = 1.0 if sym in [x.strip() for x in nr["outs"].iloc[0].split(",")] else 0.0
        else:
            out["feat_evt_rebal_sessions_to_next"][i] = 30.0
    for c, v in out.items():
        df[c] = v
    del pos
    return df


def transform(df: pd.DataFrame, variant: str, sym: str | None = None) -> tuple[pd.DataFrame, list[str], list[str]]:
    """Returns (frame, extra 1D columns, dropped columns) for a computed variant."""
    extra, drop = [], []
    if variant == "events":
        return add_event_features(df, sym), list(EVENT_COLS), []
    if variant in ("regime", "regime_no_sma50"):
        df = regime(df)
        extra = REGIME_COLS + [f"{c}_x_down" for c in MEAN_REVERSION if f"{c}_x_down" in df.columns]
    if variant in ("no_sma50", "regime_no_sma50"):
        drop = ["feat_price_vs_sma50_pct", "feat_price_vs_sma50_pct_x_down"]
        df = df.drop(columns=[c for c in drop if c in df.columns])
        extra = [c for c in extra if c not in drop]
    return df, extra, drop


# ----------------------------------------------------------------------------------------------------------------------
# Candidate ledgers


def candidates(args: tuple) -> str:
    """Production candidate generation (all 75 configs, trailing 300 origins) on a variant frame.

    args = (variant, symbol) for all six horizons, or (variant, symbol, horizon) for one horizon (written to a part
    file; `merge_parts` combines them) so work can be split per stock *and* horizon for load balance."""
    variant, sym = args[0], args[1]
    only = args[2] if len(args) > 2 else None
    from mdk_trading_oracle.data.silver.tertip_ml_forecaster import _check_hit, get_feature_cols_for_mode
    from mdk_trading_oracle.data.silver.tertip_multi_horizon_forecaster import evaluate_horizon_walk_forward
    from mdk_trading_oracle.models.horizon_crowning import candidate_configs
    from mdk_trading_oracle.models.t1_forecast import T1Config, forecast_t1, prepare_feature_frame, t1_labels

    path = OUT / (f"candidates_{variant}_{sym}.parquet" if only is None else f"part_{variant}_{sym}_{only}.parquet")
    if path.exists() or (OUT / f"candidates_{variant}_{sym}.parquet").exists():
        return str(path)
    base = pd.read_parquet(FRAMES / f"{sym}.parquet").sort_values("trade_date").reset_index(drop=True)
    df, extra, drop = transform(base, variant, sym)
    mkt = [c for c in df.columns if c.startswith("mkt_")]
    df_swing = df.drop(columns=mkt)  # production features (+ variant feat_ columns) only
    dates = df["trade_date"].tolist()
    n_total = len(df)
    start = max(0, n_total - HISTORY)
    rows = []
    configs = candidate_configs()
    # swing
    for h in [x for x in HORIZONS[1:] if only in (None, x)]:
        n = HDAYS[h]
        n_eval = (n_total - n - 1) - start + 1
        for c in configs:
            res = evaluate_horizon_walk_forward(df_swing.copy(), h, c.model, c.lookback, n_eval, c.feature_mode)
            for r in res["ledger"]:
                rows.append({"horizon": h, "config_key": c.key, "trade_date": r["trade_date"],
                             "target_date_end": str(r["target_date_end"])[:10], "pred": r["pred_return_pct"],
                             "actual": r["actual_return_pct"], "is_hit": bool(r["is_hit"]), "err_pct": r["err_pct"]})
    # 1D
    for c in configs if only in (None, "1d") else []:
        cols = [x for x in get_feature_cols_for_mode(c.feature_mode) if x not in drop] + extra
        frame = prepare_feature_frame(df, cols)
        lab = t1_labels(frame)
        close = frame["close_price"].to_numpy(float)
        cfg = T1Config(c.model, c.lookback, c.feature_mode, label=c.key)
        for t in range(start, n_total - 1):
            a = float(lab.iloc[t])
            if not np.isfinite(a):
                continue
            fc = forecast_t1(frame, t, cfg, cols, lab)
            if fc is None:
                continue
            p = float(fc["model_return_pct"])
            ap, pp = close[t] * (1 + a / 100), close[t] * (1 + p / 100)
            rows.append({"horizon": "1d", "config_key": c.key, "trade_date": dates[t], "target_date_end": dates[t + 1],
                         "pred": p, "actual": a, "is_hit": bool(_check_hit(p, a)), "err_pct": abs(pp - ap) / ap * 100})
    pd.DataFrame(rows).to_parquet(path)
    return str(path)


def merge_parts(variant: str, sym: str) -> bool:
    """Combine per-horizon part files into the per-stock candidate ledger once all six exist."""
    parts = [OUT / f"part_{variant}_{sym}_{h}.parquet" for h in HORIZONS]
    if not all(p.exists() for p in parts):
        return False
    pd.concat([pd.read_parquet(p) for p in parts], ignore_index=True).to_parquet(OUT / f"candidates_{variant}_{sym}.parquet")
    for p in parts:
        p.unlink()
    return True


def production_candidates(sym: str) -> pd.DataFrame:
    from mdk_trading_oracle.core.db import PostgresManager

    db = PostgresManager(read_only=True)
    return db.query_df(
        """
        SELECT horizon, config_key, trade_date::text AS trade_date, target_date_end::text AS target_date_end,
               pred_return_pct AS pred, actual_return_pct AS actual, is_hit, err_pct
        FROM gold_tertip_multi_horizon_candidates WHERE symbol = %s AND config_key NOT LIKE '%%|%%'
        """,
        params=[sym],
    )


# ----------------------------------------------------------------------------------------------------------------------
# Honest re-crowned track


def recrowned_track(led: pd.DataFrame, dates: list[str], exclude_lookback: int | None = None, eval_sessions: int = 90) -> pd.DataFrame:
    from mdk_trading_oracle.models.horizon_crowning import candidate_configs, crown, crown_single, month_starts

    configs = [c for c in candidate_configs() if c.lookback != exclude_lookback]
    keep = {c.key for c in configs}
    led = led[led["config_key"].isin(keep)]
    months = [m for m in month_starts(pd.Series(dates), dates[0]) if m >= dates[max(0, len(dates) - 150)]]
    parts = []
    for i, m in enumerate(months):
        end = months[i + 1] if i + 1 < len(months) else "9999-12-31"
        champs = crown(led[led["horizon"] != "1d"], m, configs=configs, eval_sessions=eval_sessions) or {}
        c1 = crown_single(led, m, "1d", configs=configs, eval_sessions=eval_sessions)
        if c1:
            champs["1d"] = c1
        for h, ch in champs.items():
            sel = led[(led["horizon"] == h) & (led["config_key"] == ch["config"].key) & (led["trade_date"] >= m) & (led["trade_date"] < end)]
            parts.append(sel.assign(crown_date=m))
    tr = pd.concat(parts, ignore_index=True) if parts else led.iloc[:0]
    return pd.concat([g.sort_values("trade_date").iloc[-DISPLAY:] for _, g in tr.groupby("horizon")], ignore_index=True)


def ensemble_track(led: pd.DataFrame, dates: list[str], k: int = 5) -> pd.DataFrame:
    """Each month and horizon: average the forecasts of the k best candidates by the production ranking (option-1
    recency score, last-30 hit rate, recency MAE) over outcomes realized before the month; regrade with the hub rule."""
    from mdk_trading_oracle.models.horizon_crowning import month_starts, score_window

    months = [m for m in month_starts(pd.Series(dates), dates[0]) if m >= dates[max(0, len(dates) - 150)]]
    parts = []
    for i, m in enumerate(months):
        end = months[i + 1] if i + 1 < len(months) else "9999-12-31"
        sc = score_window(led, m)
        for h, g in sc.groupby("horizon"):
            g = g[g["n_window"] >= 60]
            if g.empty:
                continue
            top = g.sort_values(["opt1_score", "r_0_30", "mae_pct"], ascending=[False, False, True]).head(k)["config_key"]
            sel = led[(led["horizon"] == h) & led["config_key"].isin(top) & (led["trade_date"] >= m) & (led["trade_date"] < end)]
            if sel.empty:
                continue
            e = sel.groupby("trade_date").agg(pred=("pred", "mean"), actual=("actual", "first"), target_date_end=("target_date_end", "first")).reset_index()
            e["is_hit"] = hub_hit(e["pred"].to_numpy(float), e["actual"].to_numpy(float), h)
            parts.append(e.assign(horizon=h, config_key=f"ens{k}", crown_date=m))
    tr = pd.concat(parts, ignore_index=True)
    return pd.concat([g.sort_values("trade_date").iloc[-DISPLAY:] for _, g in tr.groupby("horizon")], ignore_index=True)


def plain_hit_track(led: pd.DataFrame, dates: list[str], window: int = 150) -> pd.DataFrame:
    """Each month and horizon: the candidate with the best hit rate over its last `window` realized rows."""
    from mdk_trading_oracle.models.horizon_crowning import month_starts

    months = [m for m in month_starts(pd.Series(dates), dates[0]) if m >= dates[max(0, len(dates) - 150)]]
    parts = []
    for i, m in enumerate(months):
        end = months[i + 1] if i + 1 < len(months) else "9999-12-31"
        known = led[led["target_date_end"] < m]
        for h, g in known.groupby("horizon"):
            sc = (g.sort_values("trade_date").groupby("config_key").tail(window)
                  .groupby("config_key").agg(hit=("is_hit", "mean"), mae=("err_pct", "mean"), n=("is_hit", "size")))
            sc = sc[sc["n"] >= 90]
            if sc.empty:
                continue
            best = sc.sort_values(["hit", "mae"], ascending=[False, True]).index[0]
            sel = led[(led["horizon"] == h) & (led["config_key"] == best) & (led["trade_date"] >= m) & (led["trade_date"] < end)]
            parts.append(sel.assign(crown_date=m))
    tr = pd.concat(parts, ignore_index=True)
    return pd.concat([g.sort_values("trade_date").iloc[-DISPLAY:] for _, g in tr.groupby("horizon")], ignore_index=True)


def crown_available(led: pd.DataFrame, crown_date: str) -> dict:
    """Production crowning rule (family-first over 3D-30D, option-1 recency score, same tie-break; 1D on its own)
    restricted to the candidates present at each horizon (per-horizon exclusions allowed)."""
    from mdk_trading_oracle.models.horizon_crowning import FAMILIES, HORIZON_WEIGHTS, candidate_configs, score_window

    sc = score_window(led, crown_date)
    if sc.empty:
        return {}
    sc = sc[sc["n_window"] >= 60]
    fam_of = {c.key: c.family for c in candidate_configs()}
    order = {c.key: i for i, c in enumerate(candidate_configs())}
    sc = sc.assign(family=sc["config_key"].map(fam_of), order=sc["config_key"].map(order))

    def best(g: pd.DataFrame):
        g = g.sort_values(["opt1_score", "r_0_30", "mae_pct", "order"], ascending=[False, False, True, True])
        return g.iloc[0] if len(g) else None

    out = {}
    one = best(sc[sc["horizon"] == "1d"])
    if one is not None:
        out["1d"] = one["config_key"]
    swing = sc[sc["horizon"] != "1d"]
    if swing["horizon"].nunique() == 5:
        fam_score = {}
        for fam in FAMILIES:
            tot = 0.0
            for h, w in HORIZON_WEIGHTS.items():
                b = best(swing[(swing["horizon"] == h) & (swing["family"] == fam)])
                tot += w * (b["opt1_score"] if b is not None else 0.0)
            fam_score[fam] = tot
        fam = max(fam_score, key=fam_score.get)
        for h in HORIZON_WEIGHTS:
            b = best(swing[(swing["horizon"] == h) & (swing["family"] == fam)])
            if b is not None:
                out[h] = b["config_key"]
    return out


def policy_track(leds: dict, dates: list[str], policy: dict) -> pd.DataFrame:
    """Combine per-horizon candidate sources (and per-horizon lookback exclusions), crown jointly like production."""
    from mdk_trading_oracle.models.horizon_crowning import month_starts

    keep_prod = [h for h, (src, _) in policy.items() if src == "production_track"]
    policy = {h: (("production", None) if src == "production_track" else (src, excl)) for h, (src, excl) in policy.items()}
    parts = []
    for h, (src, excl) in policy.items():
        g = leds[src]
        g = g[g["horizon"] == h]
        if excl is not None:
            g = g[~g["config_key"].str.split("/").str[1].eq(str(excl))]
        parts.append(g)
    led = pd.concat(parts, ignore_index=True)
    months = [m for m in month_starts(pd.Series(dates), dates[0]) if m >= dates[max(0, len(dates) - 150)]]
    out = []
    for i, m in enumerate(months):
        end = months[i + 1] if i + 1 < len(months) else "9999-12-31"
        for h, key in crown_available(led, m).items():
            out.append(led[(led["horizon"] == h) & (led["config_key"] == key) & (led["trade_date"] >= m) & (led["trade_date"] < end)].assign(crown_date=m))
    tr = pd.concat(out, ignore_index=True)
    if keep_prod:
        prod = recrowned_track(leds["production"], dates)
        tr = pd.concat([tr[~tr["horizon"].isin(keep_prod)], prod[prod["horizon"].isin(keep_prod)]], ignore_index=True)
    return pd.concat([g.sort_values("trade_date").iloc[-DISPLAY:] for _, g in tr.groupby("horizon")], ignore_index=True)


STAGED = {"staged3": False, "staged9": True, "staged_c": "constrained", "staged_f": "forced"}
# stage-2 forced: the horizon's validated feature set only (no per-stock choice)
STAGE2_FORCED = {"1d": ("events",), "3d": ("events",), "5d": ("regime_no_sma50",), "10d": ("regime_no_sma50",),
                 "15d": ("production",), "30d": ("production",)}
# stage-2 feature sets allowed per horizon (constrained: production or the feature set validated for the horizon)
STAGE2_ALLOWED = {"1d": ("production", "events"), "3d": ("production", "events"), "5d": ("production", "regime_no_sma50"),
                  "10d": ("production", "regime_no_sma50"), "15d": ("production",), "30d": ("production",)}  # stage 2 over feature sets only, or feature set x feature mode
STAGE2_SETS = ("production", "events", "regime_no_sma50")


def staged_track(leds: dict, dates: list[str], with_modes: bool) -> pd.DataFrame:
    """Staged monthly tournament (production `crown_staged`): stage 1 model + window on the production pool, stage 2
    feature set x feature mode (or feature set only, keeping stage 1's mode, when `with_modes` is False)."""
    from mdk_trading_oracle.models.horizon_crowning import crown, crown_single, crown_staged, month_starts, score_window

    keyed = [leds["production"]]
    for fs in STAGE2_SETS[1:]:
        g = leds[fs].copy()
        g["config_key"] = fs + "|" + g["config_key"]
        keyed.append(g)
    allc = pd.concat(keyed, ignore_index=True)
    months = [m for m in month_starts(pd.Series(dates), dates[0]) if m >= dates[max(0, len(dates) - 150)]]
    out = []
    for i, m in enumerate(months):
        end = months[i + 1] if i + 1 < len(months) else "9999-12-31"
        if with_modes == "constrained":
            picks = crown_staged(allc, m, STAGE2_SETS, allowed=STAGE2_ALLOWED)
        elif with_modes == "forced":
            picks = crown_staged(allc, m, STAGE2_SETS, allowed=STAGE2_FORCED)
        elif with_modes:
            picks = crown_staged(allc, m, STAGE2_SETS)
        else:  # stage 2 over feature sets only, keeping stage 1's feature mode
            prod = leds["production"]
            st1 = {h: v["config"].key for h, v in (crown(prod[prod["horizon"] != "1d"], m) or {}).items()}
            one = crown_single(prod, m, "1d")
            if one:
                st1["1d"] = one["config"].key
            picks = {}
            for h, key in st1.items():
                sub = allc[(allc["horizon"] == h) & allc["config_key"].str.split("|").str[-1].eq(key)]
                sc = score_window(sub, m)
                sc = sc[sc["n_window"] >= 60]
                picks[h] = sc.sort_values(["opt1_score", "r_0_30", "mae_pct"], ascending=[False, False, True])["config_key"].iloc[0] if len(sc) else key
        for h, key in picks.items():
            out.append(allc[(allc["horizon"] == h) & (allc["config_key"] == key) & (allc["trade_date"] >= m) & (allc["trade_date"] < end)].assign(crown_date=m))
    tr = pd.concat(out, ignore_index=True)
    return pd.concat([g.sort_values("trade_date").iloc[-DISPLAY:] for _, g in tr.groupby("horizon")], ignore_index=True)


def tracks_for(sym: str, variants: list[str]) -> pd.DataFrame:
    dates = pd.read_parquet(FRAMES / f"{sym}.parquet")["trade_date"].sort_values().tolist()
    leds = {"production": production_candidates(sym)}
    pol_src = {src for x in variants if x in POLICIES for src, _ in POLICIES[x].values()} - {"production_track"}
    if any(x in STAGED for x in variants):
        pol_src |= set(STAGE2_SETS[1:])
    for v in set(variants) | {ENSEMBLE[x] for x in variants if x in ENSEMBLE} | {DERIVED[x][0] for x in variants if x in DERIVED} | pol_src:
        if v in COMPUTED and (OUT / f"candidates_{v}_{sym}.parquet").exists():
            leds[v] = pd.read_parquet(OUT / f"candidates_{v}_{sym}.parquet")
    out = []
    for v in ["production"] + variants:
        if v in STAGED:
            if any(fs not in leds for fs in STAGE2_SETS[1:]):
                continue
            tr = staged_track(leds, dates, STAGED[v])
        elif v in POLICIES:
            if any(src not in leds and src != "production_track" for src, _ in POLICIES[v].values()):
                continue
            tr = policy_track(leds, dates, POLICIES[v])
        elif v in ENSEMBLE:
            if ENSEMBLE[v] not in leds:
                continue
            tr = ensemble_track(leds[ENSEMBLE[v]], dates)
        elif v in CROWN_WINDOW:
            src, w = CROWN_WINDOW[v]
            tr = plain_hit_track(leds[src], dates, window=w)
        elif v in DERIVED:
            src, lb = DERIVED[v]
            if src not in leds:
                continue
            tr = recrowned_track(leds[src], dates, exclude_lookback=lb)
        elif v in leds:
            tr = recrowned_track(leds[v], dates)
        else:
            continue
        out.append(tr.assign(variant=v, symbol=sym))
    return pd.concat(out, ignore_index=True)


# ----------------------------------------------------------------------------------------------------------------------
# Report


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


def report(tr: pd.DataFrame) -> pd.DataFrame:
    """Per variant x horizon: hit rate on the production sessions, edge vs the better constant, stocks beating both."""
    prod_sessions = tr[tr["variant"] == "production"][["symbol", "horizon", "trade_date"]]
    tr = tr.merge(prod_sessions, on=["symbol", "horizon", "trade_date"])
    rows = []
    for (v, h), g in tr.groupby(["variant", "horizon"]):
        a = g["actual"].to_numpy(float)
        db = deadband(h)
        g = g.assign(up=hub_hit(np.full(len(g), 2 * db), a, h), down=hub_hit(np.full(len(g), -2 * db), a, h))
        per = g.groupby("symbol").agg(hit=("is_hit", "mean"), up=("up", "mean"), down=("down", "mean"))
        rows.append({
            "variant": v, "horizon": h, "sessions": len(g), "hit_pct": g["is_hit"].mean() * 100,
            "always_up_pct": g["up"].mean() * 100, "always_down_pct": g["down"].mean() * 100,
            "stocks_beat_both": int((per["hit"] > per[["up", "down"]].max(axis=1)).sum()), "stocks": len(per),
            "up_calls_pct": (g["pred"] > db).mean() * 100,
        })
    r = pd.DataFrame(rows)
    r["edge_vs_better_pp"] = r["hit_pct"] - r[["always_up_pct", "always_down_pct"]].max(axis=1)
    prod = r[r["variant"] == "production"].set_index("horizon")["hit_pct"]
    r["vs_production_pp"] = r["hit_pct"] - r["horizon"].map(prod)
    return r


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--symbols", default=",".join(STOCKS))
    ap.add_argument("--variants", default="no_3m,no_sma50,regime,regime_no_sma50,combo,ens5,ens5_regime,hit150")
    ap.add_argument("--workers", type=int, default=16)
    ap.add_argument("--report-only", action="store_true")
    args = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    syms = [s.strip().upper() for s in args.symbols.split(",") if s.strip()]
    variants = [v.strip() for v in args.variants.split(",") if v.strip()]
    t0 = time.time()
    if not args.report_only:
        need = sorted({v for v in variants if v in COMPUTED} | {DERIVED[v][0] for v in variants if v in DERIVED and DERIVED[v][0] in COMPUTED})
        tasks = [(v, s) for v in need for s in syms]
        with Pool(min(args.workers, len(tasks) or 1)) as pool:
            for i, _ in enumerate(pool.imap_unordered(candidates, tasks), 1):
                print(f"  candidates {i}/{len(tasks)} ({time.time() - t0:.0f}s)", file=sys.stderr, flush=True)
    tr = pd.concat([tracks_for(s, variants) for s in syms], ignore_index=True)
    tr.to_parquet(OUT / f"tracks_{len(syms)}.parquet")
    r = report(tr)
    r.to_csv(OUT / f"summary_{len(syms)}.csv", index=False)
    pd.set_option("display.width", 250)
    order = ["production"] + variants
    for col in ("hit_pct", "vs_production_pp", "edge_vs_better_pp", "stocks_beat_both", "up_calls_pct"):
        t = r.pivot(index="variant", columns="horizon", values=col).reindex(order)[HORIZONS]
        print(f"\n== {col} ==\n{t.round(1).to_string()}")
    base = r[r["variant"] == "production"].set_index("horizon")
    print("\nconstants (production sessions): UP", base["always_up_pct"].reindex(HORIZONS).round(1).to_dict(),
          "DOWN", base["always_down_pct"].reindex(HORIZONS).round(1).to_dict())
    print(f"done in {time.time() - t0:.0f}s", file=sys.stderr)


if __name__ == "__main__":
    main()
