"""Gold Layer: Tertip Daily (T+1) Forecasts & Opportunity Hub Synchronizer.

Honest by construction (docs/RESEARCH_HANDOFF_2026-10.md §0.8, §4.5):
    - every candidate's next-day forecast is point-in-time (models/t1_forecast.py: training labels realized by the
      origin close) and kept in gold_tertip_multi_horizon_candidates (horizon '1d')
    - one champion per stock and month, crowned on the month's first trading day from realized outcomes only
      (gold_tertip_multi_horizon_crowns)
    - gold_tertip_walk_forward_backtests holds the last 90 sessions, each graded with the forecast the champion in
      force made at the previous close; nothing is re-created with a model that saw the graded outcome
    - the live forecast comes from the current month's champion
"""

from __future__ import annotations

import json
import logging
from datetime import datetime
from typing import Any

import pandas as pd
from prophet import Prophet

from mdk_trading_oracle.core.db import PostgresManager
from mdk_trading_oracle.data.gold.tertip_multi_horizon import ensure_multi_horizon_tables, upsert_horizon
from mdk_trading_oracle.data.gold.tertip_multi_horizon_crowning import (
    T1_HORIZON,
    _dates,
    champion_track,
    ensure_crowning_tables,
    load_candidates,
    load_crowns,
    update_crowns,
    update_t1_candidate_ledger,
)
from mdk_trading_oracle.data.silver.tertip_ml_forecaster import (
    DEADBAND_PCT,
    _check_hit,
    adjusted_price,
    attach_prophet_rolling_features,
    calibrate_expected_return,
    clear_forecast_cache,
    extract_3pillar_time_series,
    get_feature_cols_for_mode,
)
from mdk_trading_oracle.models.stochastic_risk import (
    build_forecast_distribution,
    get_volatility_state,
    legacy_band,
    size_forecast,
)
from mdk_trading_oracle.models.t1_forecast import T1Config, forecast_t1, prepare_feature_frame, t1_labels

logger = logging.getLogger("mdk_oracle.data.gold.tertip_forecasts")
DISPLAY_SESSIONS = 90  # app-wide evaluation window


def update_gold_tertip_forecasts(db: PostgresManager, symbols: list[str] | None = None) -> dict[str, Any]:
    """Publish live T+1 forecasts of the current month's champions and their honest re-crowned 90-session track."""
    start_time = datetime.now()
    ensure_crowning_tables(db)
    ensure_multi_horizon_tables(db)

    # 1. Ensure tables exist
    db.execute("""
    CREATE TABLE IF NOT EXISTS gold_tertip_daily_forecasts (
        symbol VARCHAR(16) NOT NULL,
        as_of_date DATE NOT NULL,
        current_price DOUBLE PRECISION,
        target_price DOUBLE PRECISION,
        expected_return_pct DOUBLE PRECISION,
        price_low DOUBLE PRECISION,
        price_high DOUBLE PRECISION,
        stance VARCHAR(32),
        conviction VARCHAR(32),
        playbook VARCHAR(64),
        ml_champion_type VARCHAR(32),
        champion_dir_hits INTEGER,
        champion_dir_hit_rate_pct DOUBLE PRECISION,
        champion_mae_pct DOUBLE PRECISION,
        prophet_target_price DOUBLE PRECISION,
        prophet_expected_return_pct DOUBLE PRECISION,
        training_lookback_sessions INTEGER DEFAULT 252,
        crowned_horizon VARCHAR(16) DEFAULT '12m',
        actual_training_sessions INTEGER DEFAULT 252,
        data_sufficiency_status VARCHAR(32) DEFAULT 'FULL',
        feature_mode VARCHAR(32) DEFAULT 'FIFO_CORE',
        calculated_at TIMESTAMPTZ DEFAULT CURRENT_TIMESTAMP,
        PRIMARY KEY (symbol, as_of_date)
    );

    ALTER TABLE gold_tertip_daily_forecasts ADD COLUMN IF NOT EXISTS feature_mode VARCHAR(32) DEFAULT 'FIFO_CORE';
    ALTER TABLE gold_tertip_daily_forecasts ADD COLUMN IF NOT EXISTS risk_profile JSONB;
    ALTER TABLE gold_tertip_daily_forecasts ADD COLUMN IF NOT EXISTS crowned_on DATE;

    CREATE TABLE IF NOT EXISTS gold_tertip_walk_forward_backtests (
        symbol VARCHAR(16) NOT NULL,
        trade_date DATE NOT NULL,
        actual_price DOUBLE PRECISION,
        actual_return_pct DOUBLE PRECISION,
        bist30_ret_pct DOUBLE PRECISION,
        ml_pred_price DOUBLE PRECISION,
        ml_pred_return_pct DOUBLE PRECISION,
        ml_direction VARCHAR(16),
        ml_err_pct DOUBLE PRECISION,
        ml_is_hit BOOLEAN,
        prophet_pred_price DOUBLE PRECISION,
        prophet_pred_return_pct DOUBLE PRECISION,
        prophet_direction VARCHAR(16),
        prophet_err_pct DOUBLE PRECISION,
        prophet_is_hit BOOLEAN,
        winner VARCHAR(32),
        is_shock_day BOOLEAN,
        shock_type VARCHAR(32),
        days_since_pos_shock INTEGER,
        days_since_neg_shock INTEGER,
        mlb_action VARCHAR(32),
        mlb_flow_tl DOUBLE PRECISION,
        mlb_buy_tl DOUBLE PRECISION,
        mlb_sell_tl DOUBLE PRECISION,
        mlb_pnl_tl DOUBLE PRECISION,
        big5_action VARCHAR(32),
        big5_flow_tl DOUBLE PRECISION,
        big5_buy_tl DOUBLE PRECISION,
        big5_sell_tl DOUBLE PRECISION,
        big5_pnl_tl DOUBLE PRECISION,
        kamu_action VARCHAR(32),
        kamu_flow_tl DOUBLE PRECISION,
        kamu_buy_tl DOUBLE PRECISION,
        kamu_sell_tl DOUBLE PRECISION,
        kamu_pnl_tl DOUBLE PRECISION,
        ml_champion_type VARCHAR(32),
        training_lookback_sessions INTEGER DEFAULT 252,
        actual_training_sessions INTEGER DEFAULT 252,
        data_sufficiency_status VARCHAR(32) DEFAULT 'FULL',
        calculated_at TIMESTAMPTZ DEFAULT CURRENT_TIMESTAMP,
        PRIMARY KEY (symbol, trade_date)
    );
    """)

    upsert_forecast_sql = """
    INSERT INTO gold_tertip_daily_forecasts (
        symbol, as_of_date, current_price, target_price, expected_return_pct,
        price_low, price_high, stance, conviction, playbook,
        ml_champion_type, champion_dir_hits, champion_dir_hit_rate_pct, champion_mae_pct,
        prophet_target_price, prophet_expected_return_pct,
        training_lookback_sessions, crowned_horizon, actual_training_sessions, data_sufficiency_status,
        feature_mode, risk_profile, crowned_on, calculated_at
    ) VALUES (
        %(symbol)s, %(as_of_date)s, %(current_price)s, %(target_price)s, %(expected_return_pct)s,
        %(price_low)s, %(price_high)s, %(stance)s, %(conviction)s, %(playbook)s,
        %(ml_champion_type)s, %(champion_dir_hits)s, %(champion_dir_hit_rate_pct)s, %(champion_mae_pct)s,
        %(prophet_target_price)s, %(prophet_expected_return_pct)s,
        %(training_lookback_sessions)s, %(crowned_horizon)s, %(actual_training_sessions)s, %(data_sufficiency_status)s,
        %(feature_mode)s, %(risk_profile)s::jsonb, %(crowned_on)s, CURRENT_TIMESTAMP
    ) ON CONFLICT (symbol, as_of_date) DO UPDATE SET
        current_price = EXCLUDED.current_price,
        target_price = EXCLUDED.target_price,
        expected_return_pct = EXCLUDED.expected_return_pct,
        price_low = EXCLUDED.price_low,
        price_high = EXCLUDED.price_high,
        stance = EXCLUDED.stance,
        conviction = EXCLUDED.conviction,
        playbook = EXCLUDED.playbook,
        ml_champion_type = EXCLUDED.ml_champion_type,
        champion_dir_hits = EXCLUDED.champion_dir_hits,
        champion_dir_hit_rate_pct = EXCLUDED.champion_dir_hit_rate_pct,
        champion_mae_pct = EXCLUDED.champion_mae_pct,
        prophet_target_price = EXCLUDED.prophet_target_price,
        prophet_expected_return_pct = EXCLUDED.prophet_expected_return_pct,
        training_lookback_sessions = EXCLUDED.training_lookback_sessions,
        crowned_horizon = EXCLUDED.crowned_horizon,
        actual_training_sessions = EXCLUDED.actual_training_sessions,
        data_sufficiency_status = EXCLUDED.data_sufficiency_status,
        feature_mode = EXCLUDED.feature_mode,
        risk_profile = EXCLUDED.risk_profile,
        crowned_on = EXCLUDED.crowned_on,
        calculated_at = CURRENT_TIMESTAMP;
    """

    upsert_backtest_sql = """
    INSERT INTO gold_tertip_walk_forward_backtests (
        symbol, trade_date, actual_price, actual_return_pct, bist30_ret_pct,
        ml_pred_price, ml_pred_return_pct, ml_direction, ml_err_pct, ml_is_hit,
        prophet_pred_price, prophet_pred_return_pct, prophet_direction, prophet_err_pct, prophet_is_hit,
        winner, is_shock_day, shock_type, days_since_pos_shock, days_since_neg_shock,
        mlb_action, mlb_flow_tl, mlb_buy_tl, mlb_sell_tl, mlb_pnl_tl,
        big5_action, big5_flow_tl, big5_buy_tl, big5_sell_tl, big5_pnl_tl,
        kamu_action, kamu_flow_tl, kamu_buy_tl, kamu_sell_tl, kamu_pnl_tl,
        ml_champion_type, training_lookback_sessions, actual_training_sessions, data_sufficiency_status, calculated_at
    ) VALUES (
        %(symbol)s, %(trade_date)s, %(actual_price)s, %(actual_return_pct)s, %(bist30_ret_pct)s,
        %(ml_pred_price)s, %(ml_pred_return_pct)s, %(ml_direction)s, %(ml_err_pct)s, %(ml_is_hit)s,
        %(prophet_pred_price)s, %(prophet_pred_return_pct)s, %(prophet_direction)s, %(prophet_err_pct)s, %(prophet_is_hit)s,
        %(winner)s, %(is_shock_day)s, %(shock_type)s, %(days_since_pos_shock)s, %(days_since_neg_shock)s,
        %(mlb_action)s, %(mlb_flow_tl)s, %(mlb_buy_tl)s, %(mlb_sell_tl)s, %(mlb_pnl_tl)s,
        %(big5_action)s, %(big5_flow_tl)s, %(big5_buy_tl)s, %(big5_sell_tl)s, %(big5_pnl_tl)s,
        %(kamu_action)s, %(kamu_flow_tl)s, %(kamu_buy_tl)s, %(kamu_sell_tl)s, %(kamu_pnl_tl)s,
        %(ml_champion_type)s, %(training_lookback_sessions)s, %(actual_training_sessions)s, %(data_sufficiency_status)s, CURRENT_TIMESTAMP
    ) ON CONFLICT (symbol, trade_date) DO UPDATE SET
        actual_price = EXCLUDED.actual_price,
        actual_return_pct = EXCLUDED.actual_return_pct,
        bist30_ret_pct = EXCLUDED.bist30_ret_pct,
        ml_pred_price = EXCLUDED.ml_pred_price,
        ml_pred_return_pct = EXCLUDED.ml_pred_return_pct,
        ml_direction = EXCLUDED.ml_direction,
        ml_err_pct = EXCLUDED.ml_err_pct,
        ml_is_hit = EXCLUDED.ml_is_hit,
        prophet_pred_price = EXCLUDED.prophet_pred_price,
        prophet_pred_return_pct = EXCLUDED.prophet_pred_return_pct,
        prophet_direction = EXCLUDED.prophet_direction,
        prophet_err_pct = EXCLUDED.prophet_err_pct,
        prophet_is_hit = EXCLUDED.prophet_is_hit,
        winner = EXCLUDED.winner,
        is_shock_day = EXCLUDED.is_shock_day,
        shock_type = EXCLUDED.shock_type,
        days_since_pos_shock = EXCLUDED.days_since_pos_shock,
        days_since_neg_shock = EXCLUDED.days_since_neg_shock,
        mlb_action = EXCLUDED.mlb_action,
        mlb_flow_tl = EXCLUDED.mlb_flow_tl,
        mlb_buy_tl = EXCLUDED.mlb_buy_tl,
        mlb_sell_tl = EXCLUDED.mlb_sell_tl,
        mlb_pnl_tl = EXCLUDED.mlb_pnl_tl,
        big5_action = EXCLUDED.big5_action,
        big5_flow_tl = EXCLUDED.big5_flow_tl,
        big5_buy_tl = EXCLUDED.big5_buy_tl,
        big5_sell_tl = EXCLUDED.big5_sell_tl,
        big5_pnl_tl = EXCLUDED.big5_pnl_tl,
        kamu_action = EXCLUDED.kamu_action,
        kamu_flow_tl = EXCLUDED.kamu_flow_tl,
        kamu_buy_tl = EXCLUDED.kamu_buy_tl,
        kamu_sell_tl = EXCLUDED.kamu_sell_tl,
        kamu_pnl_tl = EXCLUDED.kamu_pnl_tl,
        ml_champion_type = EXCLUDED.ml_champion_type,
        training_lookback_sessions = EXCLUDED.training_lookback_sessions,
        actual_training_sessions = EXCLUDED.actual_training_sessions,
        data_sufficiency_status = EXCLUDED.data_sufficiency_status,
        calculated_at = CURRENT_TIMESTAMP;
    """

    if symbols:
        target = [x.upper() for x in symbols]
    else:
        target = db.query_df("SELECT symbol FROM bronze_bist30_membership WHERE is_active = TRUE ORDER BY symbol")["symbol"].tolist()

    success_count = 0
    latest_trade_date = None
    for sym in target:
        try:
            res = _update_symbol(db, sym, upsert_forecast_sql, upsert_backtest_sql)
        except Exception as e:
            logger.error(f"T+1 gold update failed for {sym}: {e}", exc_info=True)
            continue
        if res:
            success_count += 1
            latest_trade_date = res

    # Purge exited constituents for active periods (e.g. DSTKF as of 2026-10-01)
    db.execute("""
        DELETE FROM gold_tertip_daily_forecasts
        WHERE as_of_date >= '2026-10-01'
          AND symbol NOT IN (SELECT symbol FROM bronze_bist30_membership WHERE is_active = TRUE);
    """)

    # Invalidate in-memory cache so UI serves fresh data immediately
    clear_forecast_cache()
    elapsed = (datetime.now() - start_time).total_seconds()
    logger.info(f"Updated live Tertip Gold forecasts for {success_count}/{len(target)} equities in {elapsed:.1f}s (Latest Date: {latest_trade_date})")

    return {
        "status": "success",
        "equities_updated": success_count,
        "as_of_date": latest_trade_date,
        "elapsed_sec": elapsed,
    }


def _stance(ret_pct: float) -> tuple[str, str]:
    if ret_pct >= 2.0:
        return "STRONG_BUY", "STRONG BUY ACCUMULATION"
    if ret_pct > DEADBAND_PCT:
        return "BUY", "BUY ABSORPTION REBOUND"
    if ret_pct <= -2.0:
        return "STRONG_SELL", "STRONG SELL PRESSURE"
    if ret_pct < -DEADBAND_PCT:
        return "SELL", "DISTRIBUTION FADE"
    return "NEUTRAL", "NEUTRAL CONSOLIDATION"


def _direction(ret_pct: float) -> str:
    return "UP" if ret_pct > DEADBAND_PCT else ("DOWN" if ret_pct < -DEADBAND_PCT else "FLAT")


def _prophet_next(df: pd.DataFrame, lookback: int) -> float:
    """Prophet one-session-ahead return from the trailing `lookback` adjusted closes (univariate baseline)."""
    px = adjusted_price(df, "close").iloc[-lookback:]
    m = Prophet(daily_seasonality=False, weekly_seasonality=False, yearly_seasonality=False, changepoint_prior_scale=0.05)
    m.fit(pd.DataFrame({"ds": pd.to_datetime(df["trade_date"].iloc[-lookback:]), "y": px.to_numpy()}))
    yhat = float(m.predict(m.make_future_dataframe(periods=1)).iloc[-1]["yhat"])
    last = float(px.iloc[-1])
    return (yhat - last) / last * 100.0 if last > 0 else 0.0


def _update_symbol(db: PostgresManager, sym: str, upsert_forecast_sql: str, upsert_backtest_sql: str) -> str | None:
    df = extract_3pillar_time_series(db, sym, lookback_days=550)
    if df.empty or len(df) < 120:
        logger.warning(f"Insufficient history for {sym}: {len(df)} sessions. Skipping T+1.")
        return None
    df = df.sort_values("trade_date").reset_index(drop=True)
    # Prophet's one-step forecast for each recent session from data through the previous close (baseline column)
    df = attach_prophet_rolling_features(df, sym, n_history_needed=DISPLAY_SESSIONS + 10)

    update_t1_candidate_ledger(db, sym, df)
    candidates = load_candidates(db, sym)
    update_crowns(db, sym, candidates, _dates(df))
    crowns = load_crowns(db, sym)
    dates = _dates(df)
    latest_date = dates[-1]
    cr = crowns[(crowns["horizon"] == T1_HORIZON) & (crowns["crown_date"] <= latest_date)]
    if cr.empty:
        logger.warning(f"{sym}: no 1D champion crowned yet")
        return None
    cur = cr.iloc[-1]

    # 1. Honest track: last 90 graded sessions, each with the champion in force at its origin
    track = champion_track(candidates, crowns, T1_HORIZON).iloc[-DISPLAY_SESSIONS:]
    row_of = {d: i for i, d in enumerate(dates)}
    db.execute("DELETE FROM gold_tertip_walk_forward_backtests WHERE symbol = %s", (sym,))
    for r in track.itertuples(index=False):
        g = row_of.get(r.target_date_end)
        if g is None:
            continue
        graded = df.iloc[g]
        prev_close = float(r.current_price)
        actual_ret, pred_ret = float(r.actual_return_pct), float(r.pred_return_pct)
        actual_eq = prev_close * (1.0 + actual_ret / 100.0)
        p_ret = float(graded.get("feat_prophet_ret_today_pct", 0.0) or 0.0)
        p_price = prev_close * (1.0 + p_ret / 100.0)
        bist30_ret = float(graded.get("feat_bist30_ret_today_pct", 0.0) or 0.0)
        ml_hit = bool(r.is_hit)

        def _f(col: str) -> float:
            v = graded.get(col, 0.0)
            return round(float(v or 0.0), 1)

        db.execute(upsert_backtest_sql, {
            "symbol": sym,
            "trade_date": r.target_date_end,
            "actual_price": round(float(graded["close_price"]), 2),
            "actual_return_pct": round(actual_ret, 2),
            "bist30_ret_pct": round(bist30_ret, 2),
            "ml_pred_price": round(prev_close * (1.0 + pred_ret / 100.0), 2),
            "ml_pred_return_pct": round(pred_ret, 2),
            "ml_direction": _direction(pred_ret),
            "ml_err_pct": round(float(r.err_pct), 2),
            "ml_is_hit": ml_hit,
            "prophet_pred_price": round(p_price, 2),
            "prophet_pred_return_pct": round(p_ret, 2),
            "prophet_direction": _direction(p_ret),
            "prophet_err_pct": round(abs(p_price - actual_eq) / actual_eq * 100.0, 2),
            "prophet_is_hit": bool(_check_hit(p_ret, actual_ret)),
            "winner": "TERTIP_ML_CHALLENGER" if ml_hit else "PROPHET_BASE",
            "is_shock_day": abs(bist30_ret) >= 3.0,
            "shock_type": "POSITIVE" if bist30_ret >= 3.0 else ("NEGATIVE" if bist30_ret <= -3.0 else "NONE"),
            "days_since_pos_shock": int(graded.get("feat_days_since_pos_shock", 7) or 7),
            "days_since_neg_shock": int(graded.get("feat_days_since_neg_shock", 7) or 7),
            "mlb_action": "BUY" if _f("mlb_flow") > 0 else "SELL",
            "mlb_flow_tl": _f("mlb_flow"), "mlb_buy_tl": _f("mlb_buy_tl"), "mlb_sell_tl": _f("mlb_sell_tl"),
            "mlb_pnl_tl": _f("mlb_daily_pnl_tl"),
            "big5_action": "BUY" if _f("big5_flow") > 0 else "SELL",
            "big5_flow_tl": _f("big5_flow"), "big5_buy_tl": _f("big5_buy_tl"), "big5_sell_tl": _f("big5_sell_tl"),
            "big5_pnl_tl": _f("big5_daily_pnl_tl"),
            "kamu_action": "BUY" if _f("kamu_flow") > 0 else "SELL",
            "kamu_flow_tl": _f("kamu_flow"), "kamu_buy_tl": _f("kamu_buy_tl"), "kamu_sell_tl": _f("kamu_sell_tl"),
            "kamu_pnl_tl": _f("kamu_daily_pnl_tl"),
            "ml_champion_type": r.model,
            "training_lookback_sessions": int(r.lookback),
            "actual_training_sessions": int(r.actual_training_sessions),
            "data_sufficiency_status": r.data_sufficiency_status or "FULL",
        })

    # 2. Live T+1 forecast of the current month's champion, fitted at the latest close
    cfg = T1Config(str(cur["model"]), int(cur["lookback"]), str(cur["feature_mode"]), label=str(cur["config_key"]))
    f_cols = get_feature_cols_for_mode(cfg.feature_mode)
    frame = prepare_feature_frame(df, f_cols)
    fc = forecast_t1(frame, len(frame) - 1, cfg, f_cols, t1_labels(frame))
    if fc is None:
        logger.warning(f"{sym}: champion {cfg.spec} cannot be fitted at {latest_date}")
        return None
    latest_price = float(df["close_price"].iloc[-1])
    model_ret = float(fc["model_return_pct"])
    vol_state = get_volatility_state(db, sym, latest_date)
    sized = size_forecast(model_ret, vol_state, 1, seed_key=f"{latest_date}:1d")
    pred_ret = sized["expected_return_pct"] if sized else model_ret
    target_price = latest_price * (1.0 + pred_ret / 100.0)
    risk_profile = build_forecast_distribution(
        latest_price, model_ret, vol_state, 1, DEADBAND_PCT, seed_key=f"{latest_date}:1d", target_return_pct=pred_ret,
    )
    if risk_profile:
        risk_profile["model_expected_return_pct"] = round(model_ret, 2)
        risk_profile["sizing_confidence"] = sized["sizing_confidence"] if sized else None
        price_low, price_high = risk_profile["price_low"], risk_profile["price_high"]
    else:
        vol_20d = float(df["daily_return_pct"].iloc[-20:].std() * 100.0) if len(df) >= 20 else 2.5
        price_low, price_high = legacy_band(target_price, vol_20d)
    stance, playbook = _stance(pred_ret)

    atr = float(df["atr_14d_pct"].iloc[-1]) if "atr_14d_pct" in df.columns else 2.5
    try:
        prophet_ret = calibrate_expected_return(_prophet_next(df, min(len(df), int(cur["lookback"]))), atr, deadband=DEADBAND_PCT)
    except Exception as e:
        logger.warning(f"{sym}: Prophet baseline failed: {e}")
        prophet_ret = 0.0

    hits = int(track["is_hit"].sum())
    total = len(track)
    db.execute(upsert_forecast_sql, {
        "symbol": sym,
        "as_of_date": latest_date,
        "current_price": round(latest_price, 2),
        "target_price": round(target_price, 2),
        "expected_return_pct": round(pred_ret, 2),
        "price_low": round(price_low, 2),
        "price_high": round(price_high, 2),
        "stance": stance,
        "conviction": stance,
        "playbook": playbook,
        "ml_champion_type": cfg.model,
        "champion_dir_hits": hits,
        "champion_dir_hit_rate_pct": round(hits / total * 100.0, 1) if total else None,
        "champion_mae_pct": round(float(track["err_pct"].mean()), 2) if total else None,
        "prophet_target_price": round(latest_price * (1.0 + prophet_ret / 100.0), 2),
        "prophet_expected_return_pct": round(prophet_ret, 2),
        "training_lookback_sessions": cfg.lookback,
        "crowned_horizon": str(cur["lookback_label"]),
        "actual_training_sessions": int(fc["n_train"]),
        "data_sufficiency_status": fc["data_sufficiency_status"],
        "feature_mode": cfg.feature_mode,
        "risk_profile": json.dumps(risk_profile) if risk_profile else None,
        "crowned_on": str(cur["crown_date"]),
    })

    # 3. Same forecast and track as horizon '1d' of the multi-horizon tab (one presentation for all horizons)
    def _dir(ret: float) -> str:
        return "BUY" if ret > DEADBAND_PCT else ("SELL" if ret < -DEADBAND_PCT else "NEUTRAL")

    ledger = [
        {
            "trade_date": r.trade_date,
            "target_date_start": r.target_date_end,
            "target_date_end": r.target_date_end,
            "current_price": round(float(r.current_price), 2),
            "actual_avg_price": round(float(r.current_price) * (1.0 + float(r.actual_return_pct) / 100.0), 2),
            "actual_return_pct": round(float(r.actual_return_pct), 2),
            "pred_avg_price": round(float(r.current_price) * (1.0 + float(r.pred_return_pct) / 100.0), 2),
            "pred_return_pct": round(float(r.pred_return_pct), 2),
            "pred_direction": _dir(float(r.pred_return_pct)),
            "actual_direction": _dir(float(r.actual_return_pct)),
            "err_pct": round(float(r.err_pct), 2),
            "is_hit": bool(r.is_hit),
            "is_pending": False,
            "champion_model": r.model,
            "feature_mode": r.feature_mode,
            "actual_training_sessions": int(r.actual_training_sessions),
            "data_sufficiency_status": r.data_sufficiency_status or "FULL",
        }
        for r in track.itertuples(index=False)
    ]
    swing_stance = "BULLISH" if pred_ret > DEADBAND_PCT else ("BEARISH" if pred_ret < -DEADBAND_PCT else "NEUTRAL")
    upsert_horizon(db, sym, latest_date, round(latest_price, 2), T1_HORIZON, {
        "meta": {
            "crowned_model": cfg.model,
            "crowned_horizon": str(cur["lookback_label"]),
            "crowned_on": str(cur["crown_date"]),
            "hits": hits,
            "total_evals": total,
            "hit_rate_pct": round(hits / total * 100.0, 1) if total else 0.0,
            "mae_pct": round(float(track["err_pct"].mean()), 2) if total else 0.0,
            "training_lookback_sessions": cfg.lookback,
            "feature_mode": cfg.feature_mode,
        },
        "live_forecast": {
            "target_price": round(target_price, 2),
            "expected_return_pct": round(pred_ret, 2),
            "price_low": round(price_low, 2),
            "price_high": round(price_high, 2),
            "stance": swing_stance,
            "conviction": stance,
            "playbook": playbook.replace(" ", "_"),
            "actual_training_sessions": int(fc["n_train"]),
            "data_sufficiency_status": fc["data_sufficiency_status"],
            "risk_profile": risk_profile,
        },
        "backtest_ledger": ledger,
    })
    return latest_date
