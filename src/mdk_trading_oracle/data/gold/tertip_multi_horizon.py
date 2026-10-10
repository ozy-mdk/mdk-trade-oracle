"""Gold Layer: Multi-Horizon Tertip Forecasts and Audited Backtests.

Persists multi-horizon forecasts (3d, 5d, 10d, 15d, 30d) and zero-lookahead walk-forward
backtest ledgers into PostgreSQL TimescaleDB:
    - gold_tertip_multi_horizon_forecasts
    - gold_tertip_multi_horizon_backtests
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

from mdk_trading_oracle.core.db import PostgresManager
from mdk_trading_oracle.core.logger import get_logger
from mdk_trading_oracle.data.silver.tertip_multi_horizon_forecaster import (
    DEFAULT_CHAMPIONS,
    HORIZON_MAP,
    get_multi_horizon_forecaster_payload,
)

logger = get_logger("mdk_oracle.gold.tertip_multi_horizon")

CONFIG_PATH = Path.cwd() / "config" / "tertip_multi_horizon_crowned_models.yaml"


def ensure_multi_horizon_tables(db: PostgresManager) -> None:
    """Ensure multi-horizon Gold hypertable/tables exist in PostgreSQL."""
    db.execute("""
    CREATE TABLE IF NOT EXISTS gold_tertip_multi_horizon_forecasts (
        symbol VARCHAR(16) NOT NULL,
        as_of_date DATE NOT NULL,
        horizon VARCHAR(8) NOT NULL,
        horizon_days INTEGER NOT NULL,
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
        champion_total_evals INTEGER,
        champion_dir_hit_rate_pct DOUBLE PRECISION,
        champion_mae_pct DOUBLE PRECISION,
        training_lookback_sessions INTEGER DEFAULT 126,
        actual_training_sessions INTEGER DEFAULT 126,
        data_sufficiency_status VARCHAR(32) DEFAULT 'FULL',
        crowned_horizon VARCHAR(16) DEFAULT '6m',
        calculated_at TIMESTAMPTZ DEFAULT CURRENT_TIMESTAMP,
        PRIMARY KEY (symbol, as_of_date, horizon)
    );

    CREATE TABLE IF NOT EXISTS gold_tertip_multi_horizon_backtests (
        symbol VARCHAR(16) NOT NULL,
        horizon VARCHAR(8) NOT NULL,
        trade_date DATE NOT NULL,
        horizon_days INTEGER NOT NULL,
        current_price DOUBLE PRECISION,
        target_date_start DATE,
        target_date_end DATE,
        actual_avg_price DOUBLE PRECISION,
        actual_return_pct DOUBLE PRECISION,
        pred_avg_price DOUBLE PRECISION,
        pred_return_pct DOUBLE PRECISION,
        pred_direction VARCHAR(16),
        actual_direction VARCHAR(16),
        err_pct DOUBLE PRECISION,
        is_hit BOOLEAN,
        is_pending BOOLEAN DEFAULT FALSE,
        champion_model VARCHAR(32),
        actual_training_sessions INTEGER,
        data_sufficiency_status VARCHAR(32) DEFAULT 'FULL',
        PRIMARY KEY (symbol, horizon, trade_date)
    );

    -- Ensure backwards compatibility if tables already exist
    ALTER TABLE gold_tertip_multi_horizon_forecasts
        ADD COLUMN IF NOT EXISTS actual_training_sessions INTEGER DEFAULT 126,
        ADD COLUMN IF NOT EXISTS data_sufficiency_status VARCHAR(32) DEFAULT 'FULL';

    ALTER TABLE gold_tertip_multi_horizon_backtests
        ADD COLUMN IF NOT EXISTS actual_training_sessions INTEGER,
        ADD COLUMN IF NOT EXISTS data_sufficiency_status VARCHAR(32) DEFAULT 'FULL';

    CREATE INDEX IF NOT EXISTS idx_gold_multi_fc_sym_date
        ON gold_tertip_multi_horizon_forecasts(symbol, as_of_date DESC);
    CREATE INDEX IF NOT EXISTS idx_gold_multi_bt_sym_hz_date
        ON gold_tertip_multi_horizon_backtests(symbol, horizon, trade_date DESC);
    """)


def load_crowned_multi_horizon_config() -> dict[str, Any]:
    """Load crowned multi-horizon model configuration or return defaults."""
    if not CONFIG_PATH.exists():
        return {}
    try:
        with open(CONFIG_PATH, "r", encoding="utf-8") as f:
            return yaml.safe_load(f) or {}
    except Exception as e:
        logger.warning(f"Failed to read {CONFIG_PATH}: {e}")
        return {}


def update_gold_tertip_multi_horizon_forecasts(
    db: PostgresManager,
    symbols: list[str] | None = None,
) -> dict[str, Any]:
    """Update live forecasts and backtest ledgers for all 5 horizons across target symbols."""
    ensure_multi_horizon_tables(db)

    crowned_cfg = load_crowned_multi_horizon_config()
    symbols_cfg = crowned_cfg.get("symbols", {})

    # Determine universe to update
    if symbols:
        target_symbols = [s.upper() for s in symbols]
    elif symbols_cfg:
        target_symbols = list(symbols_cfg.keys())
    else:
        # Default to BIST 30 symbols
        rows = db.query_pl(
            "SELECT DISTINCT symbol FROM bronze_bist30_membership ORDER BY symbol ASC"
        ).to_pandas()
        target_symbols = rows["symbol"].tolist() if not rows.empty else ["THYAO", "ASELS", "GARAN", "AKBNK", "ISCTR"]

    updated_count = 0
    errors = []

    for sym in target_symbols:
        try:
            sym_comp = symbols_cfg.get(sym, DEFAULT_CHAMPIONS)
            payload = get_multi_horizon_forecaster_payload(db, sym, selected_composition=sym_comp)
            if payload.get("status") != "success":
                continue

            as_of_date = payload["as_of_date"][:10]
            current_price = payload["current_price"]

            for h_key, h_data in payload.get("horizons", {}).items():
                meta = h_data.get("meta", {})
                fc = h_data.get("live_forecast", {})
                ledger = h_data.get("backtest_ledger", [])

                # 1. Upsert live forecast
                db.execute(
                    """
                    INSERT INTO gold_tertip_multi_horizon_forecasts (
                        symbol, as_of_date, horizon, horizon_days,
                        current_price, target_price, expected_return_pct,
                        price_low, price_high, stance, conviction, playbook,
                        ml_champion_type, champion_dir_hits, champion_total_evals,
                        champion_dir_hit_rate_pct, champion_mae_pct,
                        training_lookback_sessions, actual_training_sessions,
                        data_sufficiency_status, crowned_horizon, calculated_at
                    ) VALUES (
                        %s, %s, %s, %s,
                        %s, %s, %s,
                        %s, %s, %s, %s, %s,
                        %s, %s, %s,
                        %s, %s,
                        %s, %s, %s, %s, NOW()
                    )
                    ON CONFLICT (symbol, as_of_date, horizon) DO UPDATE SET
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
                        champion_total_evals = EXCLUDED.champion_total_evals,
                        champion_dir_hit_rate_pct = EXCLUDED.champion_dir_hit_rate_pct,
                        champion_mae_pct = EXCLUDED.champion_mae_pct,
                        training_lookback_sessions = EXCLUDED.training_lookback_sessions,
                        actual_training_sessions = EXCLUDED.actual_training_sessions,
                        data_sufficiency_status = EXCLUDED.data_sufficiency_status,
                        crowned_horizon = EXCLUDED.crowned_horizon,
                        calculated_at = NOW();
                    """,
                    params=[
                        sym,
                        as_of_date,
                        h_key,
                        HORIZON_MAP.get(h_key, 3),
                        current_price,
                        fc.get("target_price", current_price),
                        fc.get("expected_return_pct", 0.0),
                        fc.get("price_low", current_price),
                        fc.get("price_high", current_price),
                        fc.get("stance", "NEUTRAL"),
                        fc.get("conviction", "NEUTRAL"),
                        fc.get("playbook", "RANGE_BOUND_CONSOLIDATION"),
                        meta.get("crowned_model", "Ridge"),
                        meta.get("hits", 0),
                        meta.get("total_evals", 0),
                        meta.get("hit_rate_pct", 0.0),
                        meta.get("mae_pct", 0.0),
                        meta.get("training_lookback_sessions", 126),
                        fc.get("actual_training_sessions", meta.get("training_lookback_sessions", 126)),
                        fc.get("data_sufficiency_status", "FULL"),
                        meta.get("crowned_horizon", "6m"),
                    ],
                )

                # 2. Upsert backtest ledger
                for b in ledger:
                    t_date = str(b.get("trade_date"))[:10]
                    db.execute(
                        """
                        INSERT INTO gold_tertip_multi_horizon_backtests (
                            symbol, horizon, trade_date, horizon_days,
                            current_price, target_date_start, target_date_end,
                            actual_avg_price, actual_return_pct,
                            pred_avg_price, pred_return_pct,
                            pred_direction, actual_direction,
                            err_pct, is_hit, is_pending, champion_model,
                            actual_training_sessions, data_sufficiency_status
                        ) VALUES (
                            %s, %s, %s, %s,
                            %s, %s, %s,
                            %s, %s,
                            %s, %s,
                            %s, %s,
                            %s, %s, %s, %s,
                            %s, %s
                        )
                        ON CONFLICT (symbol, horizon, trade_date) DO UPDATE SET
                            current_price = EXCLUDED.current_price,
                            target_date_start = EXCLUDED.target_date_start,
                            target_date_end = EXCLUDED.target_date_end,
                            actual_avg_price = EXCLUDED.actual_avg_price,
                            actual_return_pct = EXCLUDED.actual_return_pct,
                            pred_avg_price = EXCLUDED.pred_avg_price,
                            pred_return_pct = EXCLUDED.pred_return_pct,
                            pred_direction = EXCLUDED.pred_direction,
                            actual_direction = EXCLUDED.actual_direction,
                            err_pct = EXCLUDED.err_pct,
                            is_hit = EXCLUDED.is_hit,
                            is_pending = EXCLUDED.is_pending,
                            champion_model = EXCLUDED.champion_model,
                            actual_training_sessions = EXCLUDED.actual_training_sessions,
                            data_sufficiency_status = EXCLUDED.data_sufficiency_status;
                        """,
                        params=[
                            sym,
                            h_key,
                            t_date,
                            HORIZON_MAP.get(h_key, 3),
                            b.get("current_price"),
                            str(b.get("target_date_start"))[:10],
                            str(b.get("target_date_end"))[:10],
                            b.get("actual_avg_price"),
                            b.get("actual_return_pct"),
                            b.get("pred_avg_price"),
                            b.get("pred_return_pct"),
                            b.get("pred_direction"),
                            b.get("actual_direction"),
                            b.get("err_pct"),
                            b.get("is_hit"),
                            b.get("is_pending", False),
                            meta.get("crowned_model", "Ridge"),
                            b.get("actual_training_sessions"),
                            b.get("data_sufficiency_status", "FULL"),
                        ],
                    )

            updated_count += 1
            logger.info(f"Updated multi-horizon forecasts & ledger for {sym} ({as_of_date}).")
        except Exception as e:
            logger.error(f"Error updating multi-horizon forecasts for {sym}: {e}", exc_info=True)
            errors.append({"symbol": sym, "error": str(e)})

    return {
        "status": "success",
        "updated_symbols": updated_count,
        "total_requested": len(target_symbols),
        "errors": errors,
    }
