"""Gold Layer: honest monthly re-crowning for the 1D (next-day) and 3D-30D swing forecasters.

Three tables:
    gold_tertip_multi_horizon_candidates  every tournament candidate's point-in-time walk-forward forecast per graded
                                          session (production `evaluate_horizon_walk_forward`), kept for the trailing
                                          CANDIDATE_HISTORY_SESSIONS origins and extended incrementally each night
    gold_tertip_multi_horizon_crowns      one champion per (symbol, horizon, month): chosen on the first trading day
                                          of the month by the production tournament rule using only outcomes realized
                                          before that day (models/horizon_crowning.py)
    gold_tertip_multi_horizon_backtests   (existing) the displayed track: each session graded with the forecast of
                                          the champion in force for that session's month, never a later champion

So the displayed hit rate never contains the sessions a champion was selected on (docs/RESEARCH_HANDOFF_2026-10.md
§0.8). The live forecast comes from the champion crowned for the current month.

1D uses the same tables with horizon '1d': candidates are the production next-day model (models/t1_forecast.py,
training labels realized by the origin close) over the same 75 configurations; its displayed track is written to
gold_tertip_walk_forward_backtests by data/gold/tertip_forecasts.py.

Per-horizon policy (data/silver/event_regime_features.HORIZON_POLICY, tested 2026-10-25, handoff §0.8): 1D / 3D use
the `events` feature set, 5D / 10D `regime_no_sma50`, 15D the production features without 63-session lookbacks, 30D
the production features with its own production crowning. Candidate keys of non-production sets are prefixed
(`events|Ridge/252/FIFO_CORE`). Production candidates are kept for every swing horizon (30D's family-first crowning).
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
import polars as pl

from mdk_trading_oracle.core.db import PostgresManager
from mdk_trading_oracle.core.logger import get_logger
from mdk_trading_oracle.data.silver.event_regime_features import (
    HORIZON_POLICY,
    KEY_SEP,
    apply_feature_set,
    load_event_tables,
    policy_key,
    split_key,
)
from mdk_trading_oracle.models.horizon_crowning import (
    CROWN_EVAL_SESSIONS,
    HORIZONS,
    candidate_configs,
    crown,
    crown_policy,
    month_starts,
)

logger = get_logger("mdk_oracle.gold.tertip_multi_horizon_crowning")

# Origins kept per stock: the 90 displayed sessions of the longest horizon reach back ~120 sessions; the crowning of
# their oldest month needs CROWN_EVAL_SESSIONS realized rows before it (90 + horizon), plus a margin
CANDIDATE_HISTORY_SESSIONS = 300
# Months crowned: every month start within the last CROWN_HISTORY_SESSIONS sessions (covers the displayed window)
CROWN_HISTORY_SESSIONS = 150
HORIZON_DAYS = {"1d": 1, "3d": 3, "5d": 5, "10d": 10, "15d": 15, "30d": 30}
T1_HORIZON = "1d"


def ensure_crowning_tables(db: PostgresManager) -> None:
    db.execute("""
    CREATE TABLE IF NOT EXISTS gold_tertip_multi_horizon_candidates (
        symbol VARCHAR(16) NOT NULL,
        horizon VARCHAR(8) NOT NULL,
        config_key VARCHAR(64) NOT NULL,
        trade_date DATE NOT NULL,
        target_date_end DATE NOT NULL,
        current_price DOUBLE PRECISION,
        actual_return_pct DOUBLE PRECISION,
        pred_return_pct DOUBLE PRECISION,
        is_hit BOOLEAN,
        err_pct DOUBLE PRECISION,
        actual_training_sessions INTEGER,
        data_sufficiency_status VARCHAR(32),
        PRIMARY KEY (symbol, horizon, config_key, trade_date)
    );

    CREATE TABLE IF NOT EXISTS gold_tertip_multi_horizon_crowns (
        symbol VARCHAR(16) NOT NULL,
        horizon VARCHAR(8) NOT NULL,
        crown_date DATE NOT NULL,
        config_key VARCHAR(64) NOT NULL,
        model VARCHAR(32) NOT NULL,
        lookback_label VARCHAR(8),
        lookback INTEGER NOT NULL,
        feature_mode VARCHAR(32) NOT NULL,
        family VARCHAR(32),
        opt1_score DOUBLE PRECISION,
        window_label_end_date DATE,
        calculated_at TIMESTAMPTZ DEFAULT CURRENT_TIMESTAMP,
        PRIMARY KEY (symbol, horizon, crown_date)
    );

    ALTER TABLE gold_tertip_multi_horizon_forecasts ADD COLUMN IF NOT EXISTS crowned_on DATE;
    """)


# ----------------------------------------------------------------------------------------------------------------------
# Candidate ledger


def _dates(df: pd.DataFrame) -> list[str]:
    return [str(d)[:10] for d in df["trade_date"].tolist()]


def _pool_filter(feature_set: str) -> str:
    """SQL condition selecting one pool's candidate keys."""
    return f"config_key NOT LIKE '%%{KEY_SEP}%%'" if feature_set == "production" else f"config_key LIKE '{feature_set}{KEY_SEP}%%'"


def swing_pools() -> list[tuple[str, list[str]]]:
    """(feature set, horizons) for which candidates are generated: production for every swing horizon, plus each
    policy feature set for the horizons that use it."""
    pools = [("production", list(HORIZONS))]
    for fs in sorted({HORIZON_POLICY[h][0] for h in HORIZONS} - {"production"}):
        pools.append((fs, [h for h in HORIZONS if HORIZON_POLICY[h][0] == fs]))
    return pools


def update_candidate_ledger(db: PostgresManager, symbol: str, df: pd.DataFrame, ev: dict | None = None) -> int:
    """Extend every candidate's walk-forward ledger to the latest realized origin; returns rows written.

    `df` is the extract_3pillar_time_series frame (sorted, 0..n-1). Rows are point-in-time: the forecast at origin t
    trains on rows i <= t - N only (tests/test_multi_horizon_leakage.py), so re-running never changes earlier rows.
    """
    from mdk_trading_oracle.data.silver.tertip_multi_horizon_forecaster import (
        engineer_multi_horizon_features,
        evaluate_horizon_walk_forward,
    )

    sym = symbol.upper()
    df = df.sort_values("trade_date").reset_index(drop=True)
    df, _ = engineer_multi_horizon_features(df, feature_mode="FIFO_PLUS_YTD")  # all columns once; modes filter later
    ev = ev or load_event_tables(db)
    dates = _dates(df)
    n_total = len(df)
    start_idx = max(0, n_total - CANDIDATE_HISTORY_SESSIONS)
    stored = db.query_df(
        """
        SELECT horizon, config_key, MAX(trade_date)::text AS last_date
        FROM gold_tertip_multi_horizon_candidates WHERE symbol = %s GROUP BY 1, 2
        """,
        params=[sym],
    )
    last_by = {(r.horizon, r.config_key): r.last_date for r in stored.itertuples(index=False)}
    configs = candidate_configs()
    written = 0
    for fs, horizons in swing_pools():
        frame = apply_feature_set(df, sym, fs, ev)[0]
        for h in horizons:
            n_days = HORIZON_DAYS[h]
            last_idx = n_total - n_days - 1  # last origin whose N-day outcome is realized
            lasts = [last_by.get((h, policy_key(fs, c.key))) for c in configs]
            first_idx = start_idx
            if all(lasts):
                first_idx = max(start_idx, dates.index(min(lasts)) + 1 if min(lasts) in dates else start_idx)
            n_eval = last_idx - first_idx + 1
            if n_eval <= 0:
                continue
            rows: list[dict[str, Any]] = []
            for c in configs:
                res = evaluate_horizon_walk_forward(
                    frame.copy(), h, c.model, train_lookback_sessions=c.lookback, n_eval_sessions=n_eval, feature_mode=c.feature_mode
                )
                for r in res["ledger"]:
                    rows.append({
                        "symbol": sym, "horizon": h, "config_key": policy_key(fs, c.key), "trade_date": r["trade_date"],
                        "target_date_end": str(r["target_date_end"])[:10], "current_price": r["current_price"],
                        "actual_return_pct": r["actual_return_pct"], "pred_return_pct": r["pred_return_pct"],
                        "is_hit": bool(r["is_hit"]), "err_pct": r["err_pct"],
                        "actual_training_sessions": int(r["actual_training_sessions"]),
                        "data_sufficiency_status": r["data_sufficiency_status"],
                    })
            if not rows:
                continue
            db.execute(
                f"DELETE FROM gold_tertip_multi_horizon_candidates WHERE symbol = %s AND horizon = %s AND trade_date >= %s AND {_pool_filter(fs)}",
                params=[sym, h, dates[first_idx]],
            )
            written += db.copy_df_to_table(pl.DataFrame(rows), "gold_tertip_multi_horizon_candidates")
    db.execute(
        "DELETE FROM gold_tertip_multi_horizon_candidates WHERE symbol = %s AND horizon <> %s AND trade_date < %s",
        params=[sym, T1_HORIZON, dates[start_idx]],
    )
    return written


def update_t1_candidate_ledger(db: PostgresManager, symbol: str, df: pd.DataFrame, ev: dict | None = None) -> int:
    """Extend every candidate's point-in-time next-day ledger (horizon '1d') to the latest realized session.

    Row t: model fitted at the close of t on labels realized by then (`forecast_t1`), predicting r(t -> t+1);
    `trade_date` is the origin t, `target_date_end` the graded session t+1. Uses the 1D policy feature set.
    """
    from mdk_trading_oracle.data.silver.tertip_ml_forecaster import _check_hit, get_feature_cols_for_mode
    from mdk_trading_oracle.models.t1_forecast import T1Config, forecast_t1, prepare_feature_frame, t1_labels

    sym = symbol.upper()
    fs = HORIZON_POLICY[T1_HORIZON][0]
    configs = candidate_configs()
    src, extra, drop = apply_feature_set(df.sort_values("trade_date").reset_index(drop=True), sym, fs, ev or load_event_tables(db))
    mode_cols = {m: [c for c in get_feature_cols_for_mode(m) if c not in drop] + extra for m in {c.feature_mode for c in configs}}
    frame = prepare_feature_frame(src, sorted({c for cols in mode_cols.values() for c in cols}))
    labels = t1_labels(frame)
    dates = _dates(frame)
    close = frame["close_price"].to_numpy(dtype=float)
    n_total = len(frame)
    start_idx = max(0, n_total - CANDIDATE_HISTORY_SESSIONS)
    last_idx = n_total - 2  # last origin whose next-session outcome is realized
    stored = db.query_df(
        "SELECT config_key, MAX(trade_date)::text AS last_date FROM gold_tertip_multi_horizon_candidates "
        "WHERE symbol = %s AND horizon = %s GROUP BY 1",
        params=[sym, T1_HORIZON],
    )
    last_by = dict(zip(stored["config_key"], stored["last_date"]))
    lasts = [last_by.get(policy_key(fs, c.key)) for c in configs]
    first_idx = start_idx
    if all(lasts) and min(lasts) in dates:
        first_idx = max(start_idx, dates.index(min(lasts)) + 1)
    if last_idx < first_idx:
        return 0
    rows: list[dict[str, Any]] = []
    for c in configs:
        cfg = T1Config(c.model, c.lookback, c.feature_mode, label=c.key)
        f_cols = mode_cols[c.feature_mode]
        for t in range(first_idx, last_idx + 1):
            actual = float(labels.iloc[t])
            if not np.isfinite(actual):
                continue
            fc = forecast_t1(frame, t, cfg, f_cols, labels)
            if fc is None:
                continue
            pred = float(fc["model_return_pct"])
            price = float(close[t])
            actual_px, pred_px = price * (1.0 + actual / 100.0), price * (1.0 + pred / 100.0)
            rows.append({
                "symbol": sym, "horizon": T1_HORIZON, "config_key": policy_key(fs, c.key), "trade_date": dates[t],
                "target_date_end": dates[t + 1], "current_price": round(price, 4),
                "actual_return_pct": round(actual, 4), "pred_return_pct": round(pred, 4),
                "is_hit": bool(_check_hit(pred, actual)), "err_pct": round(abs(pred_px - actual_px) / actual_px * 100.0, 4),
                "actual_training_sessions": int(fc["n_train"]), "data_sufficiency_status": fc["data_sufficiency_status"],
            })
    if not rows:
        return 0
    db.execute(
        f"DELETE FROM gold_tertip_multi_horizon_candidates WHERE symbol = %s AND horizon = %s AND trade_date >= %s AND {_pool_filter(fs)}",
        params=[sym, T1_HORIZON, dates[first_idx]],
    )
    db.execute(
        "DELETE FROM gold_tertip_multi_horizon_candidates WHERE symbol = %s AND horizon = %s AND trade_date < %s",
        params=[sym, T1_HORIZON, dates[start_idx]],
    )
    return db.copy_df_to_table(pl.DataFrame(rows), "gold_tertip_multi_horizon_candidates")


def load_candidates(db: PostgresManager, symbol: str) -> pd.DataFrame:
    d = db.query_df(
        """
        SELECT horizon, config_key, trade_date::text AS trade_date, target_date_end::text AS target_date_end,
               current_price, actual_return_pct, pred_return_pct, is_hit, err_pct, actual_training_sessions,
               data_sufficiency_status
        FROM gold_tertip_multi_horizon_candidates WHERE symbol = %s
        ORDER BY horizon, config_key, trade_date
        """,
        params=[symbol.upper()],
    )
    return d


# ----------------------------------------------------------------------------------------------------------------------
# Monthly crowns


def policy_ledger(candidates: pd.DataFrame) -> pd.DataFrame:
    """Each horizon's candidate pool under HORIZON_POLICY (feature set + excluded lookback)."""
    parts = []
    for h, (fs, excl, _) in HORIZON_POLICY.items():
        g = candidates[candidates["horizon"] == h]
        keys = g["config_key"].map(lambda k: split_key(k)[0])
        g = g[keys == fs]
        if excl is not None:
            g = g[g["config_key"].map(lambda k: split_key(k)[1].split("/")[1]) != str(excl)]
        parts.append(g)
    return pd.concat(parts, ignore_index=True)


def update_crowns(db: PostgresManager, symbol: str, candidates: pd.DataFrame, trade_dates: list[str]) -> int:
    """Crown every month start of the recent window that has no complete set of crowns yet; returns crowns written.

    1D-15D: `crown_policy` over the per-horizon policy pools; 30D: production family-first crowning over the
    production candidates. Crowns on month start M see only rows with target_date_end < M (asserted).
    """
    sym = symbol.upper()
    first = trade_dates[max(0, len(trade_dates) - CROWN_HISTORY_SESSIONS)]
    crowned = db.query_df(
        "SELECT horizon, crown_date::text AS d FROM gold_tertip_multi_horizon_crowns WHERE symbol = %s", params=[sym]
    )
    have = set(zip(crowned["horizon"], crowned["d"]))
    pol = policy_ledger(candidates)
    prod_swing = candidates[(candidates["horizon"] != T1_HORIZON) & ~candidates["config_key"].str.contains(KEY_SEP, regex=False)]
    written = 0
    # first trading day of each whole month (computed on all dates, so a window start mid-month is not a crown date)
    for m in [d for d in month_starts(pd.Series(trade_dates), trade_dates[0]) if d >= first]:
        missing = [h for h in HORIZON_POLICY if (h, m) not in have]
        if not missing:
            continue
        champs = crown_policy(pol, m, eval_sessions=CROWN_EVAL_SESSIONS)
        prod = crown(prod_swing, m, eval_sessions=CROWN_EVAL_SESSIONS) or {}
        for h in missing:
            use_policy = HORIZON_POLICY[h][2]
            key = champs.get(h) if use_policy else (prod[h]["config"].key if h in prod else None)
            if key is None:
                logger.warning(f"{sym}: not enough realized candidate history to crown {h} on {m}")
                continue
            written += _store_crown(db, sym, h, m, key)
    return written


def _store_crown(db: PostgresManager, sym: str, horizon: str, crown_date: str, key: str) -> int:
    fs, base = split_key(key)
    c = {x.key: x for x in candidate_configs()}[base]
    db.execute(
        """
        INSERT INTO gold_tertip_multi_horizon_crowns (
            symbol, horizon, crown_date, config_key, model, lookback_label, lookback, feature_mode, family
        ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)
        ON CONFLICT (symbol, horizon, crown_date) DO NOTHING;
        """,
        params=[sym, horizon, crown_date, key, c.model, c.lookback_label, c.lookback, c.feature_mode, c.family],
    )
    return 1


def load_crowns(db: PostgresManager, symbol: str) -> pd.DataFrame:
    return db.query_df(
        """
        SELECT horizon, crown_date::text AS crown_date, config_key, model, lookback_label, lookback, feature_mode, family
        FROM gold_tertip_multi_horizon_crowns WHERE symbol = %s ORDER BY horizon, crown_date
        """,
        params=[symbol.upper()],
    )


def champion_track(candidates: pd.DataFrame, crowns: pd.DataFrame, horizon: str) -> pd.DataFrame:
    """Each graded session with the forecast of the champion crowned for its month (crown_date <= trade_date < next)."""
    cr = crowns[crowns["horizon"] == horizon].sort_values("crown_date")
    if cr.empty:
        return candidates.iloc[:0].assign(crown_date=None, model=None, feature_mode=None, lookback=None)
    cand = candidates[candidates["horizon"] == horizon]
    bounds = list(cr["crown_date"]) + ["9999-12-31"]
    parts = []
    for (_, c), end in zip(cr.iterrows(), bounds[1:]):
        sel = cand[(cand["config_key"] == c["config_key"]) & (cand["trade_date"] >= c["crown_date"]) & (cand["trade_date"] < end)]
        parts.append(sel.assign(crown_date=c["crown_date"], model=c["model"], feature_mode=c["feature_mode"], lookback=int(c["lookback"])))
    return pd.concat(parts, ignore_index=True).sort_values("trade_date") if parts else cand.iloc[:0]


def constant_call_hit_rates(actual_pct: np.ndarray, n_days: int) -> tuple[float, float]:
    """Hit rates of always-UP and always-DOWN under the hub rule on the same sessions (the bar to beat)."""
    from mdk_trading_oracle.data.silver.tertip_ml_forecaster import _check_hit as t1_check_hit
    from mdk_trading_oracle.data.silver.tertip_multi_horizon_forecaster import _check_hit, get_horizon_deadband

    if len(actual_pct) == 0:
        return 0.0, 0.0
    db = get_horizon_deadband(n_days)
    if n_days == 1:  # same rule as the graded 1D rows (0.20 pp tolerance)
        def hit(p: float, a: float) -> bool:
            return bool(t1_check_hit(p, a))
    else:
        def hit(p: float, a: float) -> bool:
            return bool(_check_hit(p, a, tol_delta=0.25, deadband=db))
    up = np.mean([hit(2.0 * db, float(a)) for a in actual_pct]) * 100.0
    down = np.mean([hit(-2.0 * db, float(a)) for a in actual_pct]) * 100.0
    return round(float(up), 1), round(float(down), 1)


# ----------------------------------------------------------------------------------------------------------------------
# Payload (same shape as get_multi_horizon_forecaster_payload)


def build_recrowned_payload(
    db: PostgresManager,
    symbol: str,
    df: pd.DataFrame,
    n_eval_sessions: int = 90,
    vol_state: Any = None,
    refresh: bool = True,
) -> dict[str, Any]:
    """Live forecasts from the current month's champions and the honest re-crowned 90-session track per horizon."""
    from mdk_trading_oracle.data.silver.tertip_multi_horizon_forecaster import (
        HORIZON_MAP,
        MIN_TRAIN_SESSIONS,
        _ledger_metrics,
        engineer_multi_horizon_features,
        generate_live_horizon_forecast,
    )

    sym = symbol.upper()
    df = df.sort_values("trade_date").reset_index(drop=True)
    ensure_crowning_tables(db)
    if refresh:
        update_candidate_ledger(db, sym, df)
    candidates = load_candidates(db, sym)
    if refresh:
        update_crowns(db, sym, candidates, _dates(df))
    crowns = load_crowns(db, sym)
    feat, _ = engineer_multi_horizon_features(df.copy(), feature_mode="FIFO_PLUS_YTD")
    ev = load_event_tables(db)
    all_dates = _dates(df)
    next_date = dict(zip(all_dates[:-1], all_dates[1:]))
    as_of = all_dates[-1]

    horizons: dict[str, Any] = {}
    for h in HORIZONS:
        n_days = HORIZON_MAP[h]
        cr = crowns[(crowns["horizon"] == h) & (crowns["crown_date"] <= as_of)]
        if cr.empty:
            continue
        cur = cr.iloc[-1]
        track = champion_track(candidates, crowns, h).iloc[-n_eval_sessions:]
        ledger = [
            {
                "trade_date": r.trade_date,
                "target_date_start": next_date.get(r.trade_date, r.trade_date),
                "target_date_end": r.target_date_end,
                "current_price": float(r.current_price),
                "actual_avg_price": round(float(r.current_price) * (1.0 + float(r.actual_return_pct) / 100.0), 2),
                "actual_return_pct": float(r.actual_return_pct),
                "pred_avg_price": round(float(r.current_price) * (1.0 + float(r.pred_return_pct) / 100.0), 2),
                "pred_return_pct": float(r.pred_return_pct),
                "pred_direction": _direction(float(r.pred_return_pct), n_days),
                "actual_direction": _direction(float(r.actual_return_pct), n_days),
                "err_pct": float(r.err_pct),
                "is_hit": bool(r.is_hit),
                "is_pending": False,
                "champion_model": r.model,
                "feature_mode": r.feature_mode,
                "crown_date": r.crown_date,
                "actual_training_sessions": int(r.actual_training_sessions),
                "target_lookback_sessions": int(r.lookback),
                "actual_training_months": round(int(r.actual_training_sessions) / 21.0, 1),
                "data_sufficiency_status": r.data_sufficiency_status or "FULL",
            }
            for r in track.itertuples(index=False)
        ]
        up, down = constant_call_hit_rates(track["actual_return_pct"].to_numpy(float), n_days)
        metrics = _ledger_metrics(ledger)
        live_frame = apply_feature_set(feat, sym, split_key(str(cur["config_key"]))[0], ev)[0]
        live = generate_live_horizon_forecast(
            live_frame.copy(), h, champion_model=str(cur["model"]), train_lookback_sessions=int(cur["lookback"]),
            feature_mode=str(cur["feature_mode"]), vol_state=vol_state,
        )
        live["feature_set"] = split_key(str(cur["config_key"]))[0]
        live["crowned_on"] = cur["crown_date"]
        horizons[h] = {
            "meta": {
                "horizon": h,
                "horizon_days": n_days,
                "crowned_model": str(cur["model"]),
                "crowned_horizon": str(cur["lookback_label"]),
                "crowned_on": cur["crown_date"],
                "training_lookback_sessions": int(cur["lookback"]),
                "feature_mode": str(cur["feature_mode"]),
                "feature_set": split_key(str(cur["config_key"]))[0],
                "total_evals": len(ledger),
                "eval_sessions_requested": n_eval_sessions,
                "target_lookback_sessions": int(cur["lookback"]),
                "target_lookback_months": int(cur["lookback"]) // 21,
                "min_required_train_sessions": MIN_TRAIN_SESSIONS,
                "always_up_hit_rate_pct": up,
                "always_down_hit_rate_pct": down,
                **metrics,
            },
            "live_forecast": live,
            "backtest_ledger": ledger,
        }
    return {
        "status": "success" if horizons else "error",
        "message": None if horizons else f"No crowned champions for {sym}",
        "symbol": sym,
        "as_of_date": as_of,
        "current_price": round(float(df["close_price"].iloc[-1]), 2),
        "horizons": horizons,
    }


def _direction(ret_pct: float, n_days: int) -> str:
    from mdk_trading_oracle.data.silver.tertip_multi_horizon_forecaster import get_horizon_deadband

    db = get_horizon_deadband(n_days)
    return "BUY" if ret_pct > db else ("SELL" if ret_pct < -db else "NEUTRAL")
