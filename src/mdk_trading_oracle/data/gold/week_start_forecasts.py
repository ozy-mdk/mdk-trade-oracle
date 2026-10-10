"""Gold Layer: Week Start (Monday Day Close) Forecasts & Walk-Forward Backtest Tables.

Persists the crowned model tournament results, live upcoming Monday price forecast cards,
and multi-week walk-forward reality ledgers into PostgreSQL:
    - gold_week_start_daily_forecasts
    - gold_week_start_walk_forward_backtests
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

from mdk_trading_oracle.core.db import PostgresManager
from mdk_trading_oracle.core.logger import get_logger
from mdk_trading_oracle.data.silver.week_start_forecaster import (
    WEEKLY_HORIZONS,
    extract_week_start_time_series,
    predict_live_week_start,
    run_weekly_walk_forward_arena,
)

logger = get_logger("mdk_oracle.gold.week_start_forecasts")

CONFIG_PATH = Path("config/week_start_crowned_models.yaml")


def initialize_week_start_tables(db: PostgresManager) -> None:
    """Ensure PostgreSQL tables exist for week start forecasts and backtests."""
    db.execute("""
    CREATE TABLE IF NOT EXISTS gold_week_start_daily_forecasts (
        symbol VARCHAR(16) NOT NULL,
        as_of_date DATE NOT NULL,
        target_date DATE NOT NULL,
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
        weekend_carry_cost_bps DOUBLE PRECISION,
        fri_w5_mlb_share_pct DOUBLE PRECISION,
        wtd_mlb_net_flow_tl DOUBLE PRECISION,
        training_lookback_weeks INTEGER DEFAULT 52,
        crowned_horizon VARCHAR(16) DEFAULT '12m',
        actual_training_weeks INTEGER DEFAULT 52,
        actual_training_months DOUBLE PRECISION,
        data_sufficiency_status VARCHAR(32) DEFAULT 'FULL',
        calculated_at TIMESTAMPTZ DEFAULT CURRENT_TIMESTAMP,
        PRIMARY KEY (symbol, as_of_date)
    );

    CREATE TABLE IF NOT EXISTS gold_week_start_walk_forward_backtests (
        symbol VARCHAR(16) NOT NULL,
        trade_date DATE NOT NULL,
        prior_date DATE NOT NULL,
        actual_price DOUBLE PRECISION,
        actual_return_pct DOUBLE PRECISION,
        bist30_ret_pct DOUBLE PRECISION,
        ml_pred_price DOUBLE PRECISION,
        ml_pred_return_pct DOUBLE PRECISION,
        ml_direction VARCHAR(16),
        ml_err_pct DOUBLE PRECISION,
        ml_is_hit BOOLEAN,
        weekend_carry_cost_bps DOUBLE PRECISION,
        fri_w5_mlb_share_pct DOUBLE PRECISION,
        wtd_mlb_net_flow_tl DOUBLE PRECISION,
        ml_champion_type VARCHAR(32),
        training_lookback_weeks INTEGER DEFAULT 52,
        crowned_horizon VARCHAR(16) DEFAULT '12m',
        actual_training_weeks INTEGER DEFAULT 52,
        actual_training_months DOUBLE PRECISION,
        data_sufficiency_status VARCHAR(32) DEFAULT 'FULL',
        calculated_at TIMESTAMPTZ DEFAULT CURRENT_TIMESTAMP,
        PRIMARY KEY (symbol, trade_date)
    );
    """)


def sync_week_start_forecasts_from_yaml(
    db: PostgresManager,
    config_path: Path = CONFIG_PATH,
) -> dict[str, Any]:
    """Read crowned models YAML, compute live forecast & backtest ledger, and upsert to Gold tables."""
    if not config_path.exists():
        logger.warning("Crowned models config not found at %s. Skipping week start sync.", config_path)
        return {"status": "skipped", "reason": "missing_config"}

    with open(config_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    symbols_map = cfg.get("symbols", {})
    if not symbols_map:
        return {"status": "skipped", "reason": "empty_symbols"}

    initialize_week_start_tables(db)

    upsert_forecast_sql = """
    INSERT INTO gold_week_start_daily_forecasts (
        symbol, as_of_date, target_date, current_price, target_price, expected_return_pct,
        price_low, price_high, stance, conviction, playbook,
        ml_champion_type, champion_dir_hits, champion_dir_hit_rate_pct, champion_mae_pct,
        weekend_carry_cost_bps, fri_w5_mlb_share_pct, wtd_mlb_net_flow_tl,
        training_lookback_weeks, crowned_horizon, actual_training_weeks, actual_training_months, data_sufficiency_status, calculated_at
    ) VALUES (
        %(symbol)s, %(as_of_date)s, %(target_date)s, %(current_price)s, %(target_price)s, %(expected_return_pct)s,
        %(price_low)s, %(price_high)s, %(stance)s, %(conviction)s, %(playbook)s,
        %(ml_champion_type)s, %(champion_dir_hits)s, %(champion_dir_hit_rate_pct)s, %(champion_mae_pct)s,
        %(weekend_carry_cost_bps)s, %(fri_w5_mlb_share_pct)s, %(wtd_mlb_net_flow_tl)s,
        %(training_lookback_weeks)s, %(crowned_horizon)s, %(actual_training_weeks)s, %(actual_training_months)s, %(data_sufficiency_status)s, CURRENT_TIMESTAMP
    ) ON CONFLICT (symbol, as_of_date) DO UPDATE SET
        target_date = EXCLUDED.target_date,
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
        weekend_carry_cost_bps = EXCLUDED.weekend_carry_cost_bps,
        fri_w5_mlb_share_pct = EXCLUDED.fri_w5_mlb_share_pct,
        wtd_mlb_net_flow_tl = EXCLUDED.wtd_mlb_net_flow_tl,
        training_lookback_weeks = EXCLUDED.training_lookback_weeks,
        crowned_horizon = EXCLUDED.crowned_horizon,
        actual_training_weeks = EXCLUDED.actual_training_weeks,
        actual_training_months = EXCLUDED.actual_training_months,
        data_sufficiency_status = EXCLUDED.data_sufficiency_status,
        calculated_at = CURRENT_TIMESTAMP;
    """

    upsert_backtest_sql = """
    INSERT INTO gold_week_start_walk_forward_backtests (
        symbol, trade_date, prior_date, actual_price, actual_return_pct, bist30_ret_pct,
        ml_pred_price, ml_pred_return_pct, ml_direction, ml_err_pct, ml_is_hit,
        weekend_carry_cost_bps, fri_w5_mlb_share_pct, wtd_mlb_net_flow_tl,
        ml_champion_type, training_lookback_weeks, crowned_horizon,
        actual_training_weeks, actual_training_months, data_sufficiency_status, calculated_at
    ) VALUES (
        %(symbol)s, %(trade_date)s, %(prior_date)s, %(actual_price)s, %(actual_return_pct)s, %(bist30_ret_pct)s,
        %(ml_pred_price)s, %(ml_pred_return_pct)s, %(ml_direction)s, %(ml_err_pct)s, %(ml_is_hit)s,
        %(weekend_carry_cost_bps)s, %(fri_w5_mlb_share_pct)s, %(wtd_mlb_net_flow_tl)s,
        %(ml_champion_type)s, %(training_lookback_weeks)s, %(crowned_horizon)s,
        %(actual_training_weeks)s, %(actual_training_months)s, %(data_sufficiency_status)s, CURRENT_TIMESTAMP
    ) ON CONFLICT (symbol, trade_date) DO UPDATE SET
        prior_date = EXCLUDED.prior_date,
        actual_price = EXCLUDED.actual_price,
        actual_return_pct = EXCLUDED.actual_return_pct,
        bist30_ret_pct = EXCLUDED.bist30_ret_pct,
        ml_pred_price = EXCLUDED.ml_pred_price,
        ml_pred_return_pct = EXCLUDED.ml_pred_return_pct,
        ml_direction = EXCLUDED.ml_direction,
        ml_err_pct = EXCLUDED.ml_err_pct,
        ml_is_hit = EXCLUDED.ml_is_hit,
        weekend_carry_cost_bps = EXCLUDED.weekend_carry_cost_bps,
        fri_w5_mlb_share_pct = EXCLUDED.fri_w5_mlb_share_pct,
        wtd_mlb_net_flow_tl = EXCLUDED.wtd_mlb_net_flow_tl,
        ml_champion_type = EXCLUDED.ml_champion_type,
        training_lookback_weeks = EXCLUDED.training_lookback_weeks,
        crowned_horizon = EXCLUDED.crowned_horizon,
        actual_training_weeks = EXCLUDED.actual_training_weeks,
        actual_training_months = EXCLUDED.actual_training_months,
        data_sufficiency_status = EXCLUDED.data_sufficiency_status,
        calculated_at = CURRENT_TIMESTAMP;
    """

    synced_symbols: list[str] = []

    for sym, model_cfg in symbols_map.items():
        champion_model = model_cfg.get("model", "LightGBM")
        crowned_horizon = model_cfg.get("horizon", "12m")
        lb_weeks = model_cfg.get("training_lookback_weeks", WEEKLY_HORIZONS.get(crowned_horizon, 52))

        # 1. Generate live forecast
        fc = predict_live_week_start(
            db,
            sym,
            champion_model=champion_model,
            lookback_weeks=lb_weeks,
            crowned_horizon=crowned_horizon,
        )
        if not fc:
            logger.warning("Could not compute live week start forecast for %s", sym)
            continue

        fc["champion_dir_hits"] = model_cfg.get("recent_20w_hits")
        fc["champion_dir_hit_rate_pct"] = model_cfg.get("recent_20w_hit_rate_pct")
        fc["champion_mae_pct"] = model_cfg.get("recent_20w_mae_pct")

        db.execute(upsert_forecast_sql, fc)

        # 2. Compute backtest ledger
        df = extract_week_start_time_series(db, sym)
        if not df.empty and len(df) >= 13:
            _, _, ledger_records = run_weekly_walk_forward_arena(df, n_weeks=20, lookback_weeks=lb_weeks)
            prefix = champion_model.lower()
            for rec in ledger_records:
                pred_ret = rec.get(f"{prefix}_pred_return_pct", 0.0)
                pred_price = rec.get(f"{prefix}_pred_price", 0.0)
                is_hit = rec.get(f"{prefix}_is_hit", False)
                err = rec.get(f"{prefix}_err_pct", 0.0)

                direction = "BUY" if pred_ret > 0.25 else ("SELL" if pred_ret < -0.25 else "NEUTRAL")

                row = {
                    "symbol": sym,
                    "trade_date": rec["trade_date"],
                    "prior_date": rec["prior_date"],
                    "actual_price": rec["actual_price"],
                    "actual_return_pct": rec["actual_return_pct"],
                    "bist30_ret_pct": rec["bist30_ret_pct"],
                    "ml_pred_price": pred_price,
                    "ml_pred_return_pct": pred_ret,
                    "ml_direction": direction,
                    "ml_err_pct": err,
                    "ml_is_hit": is_hit,
                    "weekend_carry_cost_bps": rec["weekend_carry_cost_bps"],
                    "fri_w5_mlb_share_pct": rec["fri_w5_mlb_share_pct"],
                    "wtd_mlb_net_flow_tl": rec["wtd_mlb_net_flow_tl"],
                    "ml_champion_type": champion_model,
                    "training_lookback_weeks": lb_weeks,
                    "crowned_horizon": crowned_horizon,
                    "actual_training_weeks": rec.get("actual_training_weeks", lb_weeks),
                    "actual_training_months": rec.get("actual_training_months", round(lb_weeks / 4.33, 1)),
                    "data_sufficiency_status": rec.get("data_sufficiency_status", "FULL"),
                }
                db.execute(upsert_backtest_sql, row)

        synced_symbols.append(sym)

    return {"status": "success", "synced_count": len(synced_symbols), "symbols": synced_symbols}
