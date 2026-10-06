"""Gold Layer: Tertip Daily Forecasts & Opportunity Hub Synchronizer.

Updates live T+1 forecasts and synchronizes the session-by-session walk-forward
backtest ledger in PostgreSQL whenever new Silver data arrives.
"""

from __future__ import annotations

import logging
from datetime import datetime
from pathlib import Path
from typing import Any

import lightgbm as lgb
import pandas as pd
import yaml
from sklearn.linear_model import BayesianRidge, HuberRegressor, Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from xgboost import XGBRegressor

from mdk_trading_oracle.core.db import PostgresManager
from mdk_trading_oracle.data.silver.tertip_ml_forecaster import (
    DEADBAND_PCT,
    FEATURE_COLS,
    _check_hit,
    attach_prophet_rolling_features,
    clear_forecast_cache,
    extract_3pillar_time_series,
)

logger = logging.getLogger("mdk_oracle.data.gold.tertip_forecasts")
CONFIG_PATH = Path("config/tertip_crowned_models.yaml")


def update_gold_tertip_forecasts(db: PostgresManager) -> dict[str, Any]:
    """Update live T+1 forecasts and append completed sessions to walk-forward ledger."""
    start_time = datetime.now()

    if not CONFIG_PATH.exists():
        logger.warning(f"Crowned models config not found at {CONFIG_PATH}. Skipping Tertip forecast update.")
        return {"status": "skipped", "reason": "missing_config"}

    with open(CONFIG_PATH, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    symbols_map = cfg.get("symbols", {})
    if not symbols_map:
        return {"status": "skipped", "reason": "empty_symbols"}

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
        calculated_at TIMESTAMPTZ DEFAULT CURRENT_TIMESTAMP,
        PRIMARY KEY (symbol, as_of_date)
    );

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
        training_lookback_sessions, crowned_horizon, calculated_at
    ) VALUES (
        %(symbol)s, %(as_of_date)s, %(current_price)s, %(target_price)s, %(expected_return_pct)s,
        %(price_low)s, %(price_high)s, %(stance)s, %(conviction)s, %(playbook)s,
        %(ml_champion_type)s, %(champion_dir_hits)s, %(champion_dir_hit_rate_pct)s, %(champion_mae_pct)s,
        %(prophet_target_price)s, %(prophet_expected_return_pct)s,
        %(training_lookback_sessions)s, %(crowned_horizon)s, CURRENT_TIMESTAMP
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
        ml_champion_type, training_lookback_sessions, calculated_at
    ) VALUES (
        %(symbol)s, %(trade_date)s, %(actual_price)s, %(actual_return_pct)s, %(bist30_ret_pct)s,
        %(ml_pred_price)s, %(ml_pred_return_pct)s, %(ml_direction)s, %(ml_err_pct)s, %(ml_is_hit)s,
        %(prophet_pred_price)s, %(prophet_pred_return_pct)s, %(prophet_direction)s, %(prophet_err_pct)s, %(prophet_is_hit)s,
        %(winner)s, %(is_shock_day)s, %(shock_type)s, %(days_since_pos_shock)s, %(days_since_neg_shock)s,
        %(mlb_action)s, %(mlb_flow_tl)s, %(mlb_buy_tl)s, %(mlb_sell_tl)s, %(mlb_pnl_tl)s,
        %(big5_action)s, %(big5_flow_tl)s, %(big5_buy_tl)s, %(big5_sell_tl)s, %(big5_pnl_tl)s,
        %(kamu_action)s, %(kamu_flow_tl)s, %(kamu_buy_tl)s, %(kamu_sell_tl)s, %(kamu_pnl_tl)s,
        %(ml_champion_type)s, %(training_lookback_sessions)s, CURRENT_TIMESTAMP
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
        calculated_at = CURRENT_TIMESTAMP;
    """

    success_count = 0
    symbols = sorted(list(symbols_map.keys()))
    latest_trade_date = None

    for sym in symbols:
        meta = symbols_map[sym]
        model_name = meta.get("model", "XGBoost")
        h_name = meta.get("horizon", "3m")
        train_lb = int(meta.get("training_lookback_sessions", 63))

        # Extract recent data
        df = extract_3pillar_time_series(db, sym, lookback_days=int((train_lb + 60) * 1.5))
        if df.empty or len(df) < 15:
            continue

        df = attach_prophet_rolling_features(df, sym, n_history_needed=50)
        latest_row = df.iloc[-1]
        latest_trade_date = str(latest_row["trade_date"]).split(" ")[0]
        latest_price = float(latest_row["close_price"])

        # Target: 1-session forward return
        tr_full = df.iloc[-train_lb:].dropna(subset=FEATURE_COLS).copy()
        y_tr_full = (tr_full["close_price"].shift(-1) - tr_full["close_price"]) / tr_full["close_price"] * 100.0
        X_tr_full = tr_full[FEATURE_COLS].iloc[:-1]
        y_tr_full = y_tr_full.iloc[:-1]

        if len(X_tr_full) < 10:
            continue

        # Fit Champion Model
        if model_name == "XGBoost":
            model = XGBRegressor(n_estimators=45, max_depth=2, learning_rate=0.03, subsample=0.8, colsample_bytree=0.8, random_state=42, n_jobs=1)
        elif model_name == "LightGBM":
            model = lgb.LGBMRegressor(n_estimators=45, max_depth=3, learning_rate=0.03, num_leaves=7, subsample=0.8, colsample_bytree=0.8, random_state=42, n_jobs=1, verbose=-1)
        elif model_name == "BayesianRidge":
            model = Pipeline([("scaler", StandardScaler()), ("regressor", BayesianRidge(max_iter=300))])
        elif model_name == "Huber":
            model = Pipeline([("scaler", StandardScaler()), ("regressor", HuberRegressor(epsilon=1.35, alpha=1.0, max_iter=300))])
        else:
            model = Ridge(alpha=2.0, random_state=42)

        model.fit(X_tr_full, y_tr_full)
        pred_ret = float(model.predict(pd.DataFrame([latest_row[FEATURE_COLS]]))[0])
        pred_ret = max(-10.0, min(10.0, pred_ret))
        target_price = latest_price * (1.0 + pred_ret / 100.0)

        # Range bounds using 20d volatility
        vol_20d = float(df["daily_return_pct"].iloc[-20:].std() * 100.0) if len(df) >= 20 else 2.5
        price_low = target_price * (1.0 - (1.645 * vol_20d / 100.0))
        price_high = target_price * (1.0 + (1.645 * vol_20d / 100.0))

        # Stance & Playbook
        if pred_ret >= 2.0:
            stance = "STRONG_BUY"
            playbook = "STRONG BUY ACCUMULATION"
        elif pred_ret > DEADBAND_PCT:
            stance = "BUY"
            playbook = "BUY ABSORPTION REBOUND"
        elif pred_ret <= -2.0:
            stance = "STRONG_SELL"
            playbook = "STRONG SELL PRESSURE"
        elif pred_ret < -DEADBAND_PCT:
            stance = "SELL"
            playbook = "DISTRIBUTION FADE"
        else:
            stance = "NEUTRAL"
            playbook = "NEUTRAL CONSOLIDATION"

        prophet_ret = float(latest_row.get("feat_prophet_ret_today_pct", 0.0))
        prophet_target_price = latest_price * (1.0 + prophet_ret / 100.0)

        # Upsert live forecast into gold_tertip_daily_forecasts
        db.execute(upsert_forecast_sql, {
            "symbol": sym,
            "as_of_date": latest_trade_date,
            "current_price": round(latest_price, 2),
            "target_price": round(target_price, 2),
            "expected_return_pct": round(pred_ret, 2),
            "price_low": round(price_low, 2),
            "price_high": round(price_high, 2),
            "stance": stance,
            "conviction": stance,
            "playbook": playbook,
            "ml_champion_type": model_name,
            "champion_dir_hits": meta.get("recent_30d_hits"),
            "champion_dir_hit_rate_pct": meta.get("recent_30d_hit_rate_pct"),
            "champion_mae_pct": meta.get("recent_30d_mae_pct"),
            "prophet_target_price": round(prophet_target_price, 2),
            "prophet_expected_return_pct": round(prophet_ret, 2),
            "training_lookback_sessions": train_lb,
            "crowned_horizon": h_name,
        })

        # Check if the latest completed session T has realized ground truth to append to walk_forward_backtests
        if len(df) >= 2:
            prev_session_row = df.iloc[-2]
            p_close = float(prev_session_row["close_price"])
            realized_ret = (latest_price - p_close) / p_close * 100.0

            # Predict on session T-1 for session T
            x_prev = prev_session_row[FEATURE_COLS]
            pred_prev_ret = float(model.predict(pd.DataFrame([x_prev]))[0])
            pred_prev_price = p_close * (1.0 + pred_prev_ret / 100.0)
            err_pct = abs(pred_prev_price - latest_price) / latest_price * 100.0
            is_hit = _check_hit(pred_prev_ret, realized_ret)
            direction = "UP" if pred_prev_ret > DEADBAND_PCT else ("DOWN" if pred_prev_ret < -DEADBAND_PCT else "FLAT")

            p_prophet_ret = float(latest_row.get("feat_prophet_ret_today_pct", 0.0))
            p_prophet_price = p_close * (1.0 + p_prophet_ret / 100.0)
            p_prophet_err = abs(p_prophet_price - latest_price) / latest_price * 100.0
            p_prophet_hit = _check_hit(p_prophet_ret, realized_ret)
            p_prophet_dir = "UP" if p_prophet_ret > DEADBAND_PCT else ("DOWN" if p_prophet_ret < -DEADBAND_PCT else "FLAT")

            bist30_ret = float(latest_row.get("feat_bist30_ret_today_pct", 0.0))
            is_shock = abs(bist30_ret) >= 3.0
            shock_type = "POSITIVE" if bist30_ret >= 3.0 else ("NEGATIVE" if bist30_ret <= -3.0 else "NONE")

            db.execute(upsert_backtest_sql, {
                "symbol": sym,
                "trade_date": latest_trade_date,
                "actual_price": round(latest_price, 2),
                "actual_return_pct": round(realized_ret, 2),
                "bist30_ret_pct": round(bist30_ret, 2),
                "ml_pred_price": round(pred_prev_price, 2),
                "ml_pred_return_pct": round(pred_prev_ret, 2),
                "ml_direction": direction,
                "ml_err_pct": round(err_pct, 2),
                "ml_is_hit": is_hit,
                "prophet_pred_price": round(p_prophet_price, 2),
                "prophet_pred_return_pct": round(p_prophet_ret, 2),
                "prophet_direction": p_prophet_dir,
                "prophet_err_pct": round(p_prophet_err, 2),
                "prophet_is_hit": p_prophet_hit,
                "winner": "TERTIP_ML_CHALLENGER" if is_hit else "PROPHET_BASE",
                "is_shock_day": is_shock,
                "shock_type": shock_type,
                "days_since_pos_shock": int(latest_row.get("feat_days_since_pos_shock", 63)),
                "days_since_neg_shock": int(latest_row.get("feat_days_since_neg_shock", 63)),
                "mlb_action": "BUY" if float(latest_row.get("mlb_flow", 0.0) or 0.0) > 0 else "SELL",
                "mlb_flow_tl": round(float(latest_row.get("mlb_flow", 0.0) or 0.0), 1),
                "mlb_buy_tl": round(float(latest_row.get("mlb_buy_tl", 0.0) or 0.0), 1),
                "mlb_sell_tl": round(float(latest_row.get("mlb_sell_tl", 0.0) or 0.0), 1),
                "mlb_pnl_tl": round(float(latest_row.get("mlb_daily_pnl_tl", 0.0) or 0.0), 1),
                "big5_action": "BUY" if float(latest_row.get("big5_flow", 0.0) or 0.0) > 0 else "SELL",
                "big5_flow_tl": round(float(latest_row.get("big5_flow", 0.0) or 0.0), 1),
                "big5_buy_tl": round(float(latest_row.get("big5_buy_tl", 0.0) or 0.0), 1),
                "big5_sell_tl": round(float(latest_row.get("big5_sell_tl", 0.0) or 0.0), 1),
                "big5_pnl_tl": round(float(latest_row.get("big5_daily_pnl_tl", 0.0) or 0.0), 1),
                "kamu_action": "BUY" if float(latest_row.get("kamu_flow", 0.0) or 0.0) > 0 else "SELL",
                "kamu_flow_tl": round(float(latest_row.get("kamu_flow", 0.0) or 0.0), 1),
                "kamu_buy_tl": round(float(latest_row.get("kamu_buy_tl", 0.0) or 0.0), 1),
                "kamu_sell_tl": round(float(latest_row.get("kamu_sell_tl", 0.0) or 0.0), 1),
                "kamu_pnl_tl": round(float(latest_row.get("kamu_daily_pnl_tl", 0.0) or 0.0), 1),
                "ml_champion_type": model_name,
                "training_lookback_sessions": train_lb,
            })

        success_count += 1

    # Invalidate in-memory cache so UI serves fresh data immediately
    clear_forecast_cache()
    elapsed = (datetime.now() - start_time).total_seconds()
    logger.info(f"Updated live Tertip Gold forecasts for {success_count}/{len(symbols)} equities in {elapsed:.1f}s (Latest Date: {latest_trade_date})")

    return {
        "status": "success",
        "equities_updated": success_count,
        "as_of_date": latest_trade_date,
        "elapsed_sec": elapsed,
    }
