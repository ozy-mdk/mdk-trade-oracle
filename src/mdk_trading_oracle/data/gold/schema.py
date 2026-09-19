"""Gold Layer schema definitions for PostgreSQL."""

from mdk_trading_oracle.core.db import PostgresManager
from mdk_trading_oracle.core.logger import get_logger

logger = get_logger("mdk_oracle.data.gold.schema")


def initialize_gold_schema(db: PostgresManager) -> None:
    """Initialize Gold layer feature tables, institutional signals, and model forecast tables."""

    # 1. Rolling Institutional Flow Signals & Multi-Day Accumulation
    db.execute("""
        CREATE TABLE IF NOT EXISTS gold_institutional_daily_signals (
            trade_date DATE,
            symbol VARCHAR,
            bofa_net_flow_tl DOUBLE PRECISION,
            bofa_volume_share DOUBLE PRECISION,
            bofa_flow_zscore_20d DOUBLE PRECISION,
            bofa_accum_5d_tl DOUBLE PRECISION,
            bofa_accum_20d_tl DOUBLE PRECISION,
            market_vwap DOUBLE PRECISION,
            close_price DOUBLE PRECISION,
            calculated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            PRIMARY KEY (trade_date, symbol)
        );
    """)

    # 2. Model 1 Output Table: Day-Start Macro Forecasts
    db.execute("""
        CREATE TABLE IF NOT EXISTS gold_bofa_day_start_forecasts (
            forecast_date DATE PRIMARY KEY,
            day_of_week INTEGER,
            is_monday BOOLEAN,
            predicted_open_net_flow_tl DOUBLE PRECISION,
            predicted_open_flow_lower_90 DOUBLE PRECISION,
            predicted_open_flow_upper_90 DOUBLE PRECISION,
            predicted_direction VARCHAR,
            direction_confidence DOUBLE PRECISION,
            predicted_playbook VARCHAR,
            top_predicted_buy_sector VARCHAR,
            top_predicted_sell_sector VARCHAR,
            model_name VARCHAR,
            model_version VARCHAR,
            generated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        );
    """)

    # 3. Model 2 Output Table: Day-Start Sector Allocations
    db.execute("""
        CREATE TABLE IF NOT EXISTS gold_bofa_sector_day_start_forecasts (
            forecast_date DATE,
            sector VARCHAR,
            day_of_week INTEGER,
            is_monday BOOLEAN,
            predicted_open_net_flow_tl DOUBLE PRECISION,
            predicted_open_flow_lower_90 DOUBLE PRECISION,
            predicted_open_flow_upper_90 DOUBLE PRECISION,
            predicted_direction VARCHAR,
            direction_confidence DOUBLE PRECISION,
            predicted_playbook VARCHAR,
            model_name VARCHAR,
            model_version VARCHAR,
            generated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            PRIMARY KEY (forecast_date, sector)
        );
    """)

    # 4. Model 1 Dedicated Historical Performance Tracking Ledger
    db.execute("""
        CREATE TABLE IF NOT EXISTS gold_bofa_day_start_performance (
            trade_date DATE PRIMARY KEY,
            day_of_week INTEGER,
            is_monday BOOLEAN,
            predicted_open_net_flow_tl DOUBLE PRECISION,
            predicted_open_flow_lower_90 DOUBLE PRECISION,
            predicted_open_flow_upper_90 DOUBLE PRECISION,
            actual_open_net_flow_tl DOUBLE PRECISION,
            error_open_net_flow_tl DOUBLE PRECISION,
            absolute_error_tl DOUBLE PRECISION,
            predicted_direction VARCHAR,
            actual_direction VARCHAR,
            is_direction_hit BOOLEAN,
            is_inside_90_ci BOOLEAN,
            direction_confidence DOUBLE PRECISION,
            predicted_playbook VARCHAR,
            top_predicted_buy_sector VARCHAR,
            top_predicted_sell_sector VARCHAR,
            model_name VARCHAR,
            model_version VARCHAR,
            forecast_generated_at TIMESTAMP,
            realized_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        );
    """)

    # 5. Model 2 Dedicated Historical Sector Performance Tracking Ledger
    db.execute("""
        CREATE TABLE IF NOT EXISTS gold_bofa_sector_day_start_performance (
            trade_date DATE,
            sector VARCHAR,
            day_of_week INTEGER,
            is_monday BOOLEAN,
            predicted_open_net_flow_tl DOUBLE PRECISION,
            predicted_open_flow_lower_90 DOUBLE PRECISION,
            predicted_open_flow_upper_90 DOUBLE PRECISION,
            actual_open_net_flow_tl DOUBLE PRECISION,
            error_open_net_flow_tl DOUBLE PRECISION,
            absolute_error_tl DOUBLE PRECISION,
            predicted_direction VARCHAR,
            actual_direction VARCHAR,
            is_direction_hit BOOLEAN,
            is_inside_90_ci BOOLEAN,
            direction_confidence DOUBLE PRECISION,
            predicted_playbook VARCHAR,
            model_name VARCHAR,
            model_version VARCHAR,
            forecast_generated_at TIMESTAMP,
            realized_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            PRIMARY KEY (trade_date, sector)
        );
    """)

    # 6. Model 1 Dedicated Historical Simulation Backtest Table
    db.execute("""
        CREATE TABLE IF NOT EXISTS gold_bofa_day_start_backtests (
            trade_date DATE PRIMARY KEY,
            day_of_week INTEGER,
            is_monday BOOLEAN,
            predicted_open_net_flow_tl DOUBLE PRECISION,
            predicted_open_flow_lower_90 DOUBLE PRECISION,
            predicted_open_flow_upper_90 DOUBLE PRECISION,
            actual_open_net_flow_tl DOUBLE PRECISION,
            error_open_net_flow_tl DOUBLE PRECISION,
            predicted_direction VARCHAR,
            actual_direction VARCHAR,
            is_direction_hit BOOLEAN,
            is_inside_90_ci BOOLEAN,
            direction_confidence DOUBLE PRECISION,
            predicted_playbook VARCHAR,
            top_predicted_buy_sector VARCHAR,
            top_predicted_sell_sector VARCHAR,
            model_name VARCHAR,
            model_version VARCHAR,
            calculated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        );
    """)

    # 7. Model 2 Dedicated Historical Sector Simulation Backtest Table
    db.execute("""
        CREATE TABLE IF NOT EXISTS gold_bofa_sector_day_start_backtests (
            trade_date DATE,
            sector VARCHAR,
            day_of_week INTEGER,
            is_monday BOOLEAN,
            predicted_open_net_flow_tl DOUBLE PRECISION,
            predicted_open_flow_lower_90 DOUBLE PRECISION,
            predicted_open_flow_upper_90 DOUBLE PRECISION,
            actual_open_net_flow_tl DOUBLE PRECISION,
            error_open_net_flow_tl DOUBLE PRECISION,
            predicted_direction VARCHAR,
            actual_direction VARCHAR,
            is_direction_hit BOOLEAN,
            is_inside_90_ci BOOLEAN,
            direction_confidence DOUBLE PRECISION,
            predicted_playbook VARCHAR,
            model_name VARCHAR,
            model_version VARCHAR,
            calculated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            PRIMARY KEY (trade_date, sector)
        );
    """)

    # ──────────────────────────────────────────────────────────────────────────────────────────────
    # MODEL 3: BIST30 Stock Intraday Reaction Forecaster
    # Three windows: W2 (first_reaction), W3 (midday_followup), W5 (closing_session)
    # ──────────────────────────────────────────────────────────────────────────────────────────────

    for _window in ["w2", "w3", "w5"]:
        _wname = {"w2": "first_reaction", "w3": "midday_followup", "w5": "closing_session"}[_window]

        # Live Upcoming Forecasts
        db.execute(f"""
            CREATE TABLE IF NOT EXISTS gold_bofa_stock_reaction_{_window}_forecasts (
                forecast_date DATE NOT NULL,
                symbol VARCHAR NOT NULL,
                window_name VARCHAR NOT NULL,
                predicted_return_pct DOUBLE PRECISION,
                predicted_return_lower_90 DOUBLE PRECISION,
                predicted_return_upper_90 DOUBLE PRECISION,
                predicted_direction VARCHAR,
                direction_confidence DOUBLE PRECISION,
                predicted_playbook VARCHAR,
                bofa_w1_direction VARCHAR,
                bofa_w1_net_flow_tl DOUBLE PRECISION,
                bofa_w1_volume_share DOUBLE PRECISION,
                model_name VARCHAR,
                model_version VARCHAR,
                generated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                PRIMARY KEY (forecast_date, symbol)
            );
        """)

        # Historical Performance Ledgers
        db.execute(f"""
            CREATE TABLE IF NOT EXISTS gold_bofa_stock_reaction_{_window}_performance (
                trade_date DATE NOT NULL,
                symbol VARCHAR NOT NULL,
                window_name VARCHAR NOT NULL,
                predicted_return_pct DOUBLE PRECISION,
                predicted_return_lower_90 DOUBLE PRECISION,
                predicted_return_upper_90 DOUBLE PRECISION,
                predicted_direction VARCHAR,
                direction_confidence DOUBLE PRECISION,
                predicted_playbook VARCHAR,
                actual_return_pct DOUBLE PRECISION,
                actual_direction VARCHAR,
                error_return_pct DOUBLE PRECISION,
                absolute_error_pct DOUBLE PRECISION,
                is_direction_hit BOOLEAN,
                is_inside_90_ci BOOLEAN,
                bofa_w1_direction VARCHAR,
                bofa_w1_net_flow_tl DOUBLE PRECISION,
                bofa_w1_volume_share DOUBLE PRECISION,
                model_name VARCHAR,
                model_version VARCHAR,
                forecast_generated_at TIMESTAMP,
                realized_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                PRIMARY KEY (trade_date, symbol)
            );
        """)

        # Simulation Backtests
        db.execute(f"""
            CREATE TABLE IF NOT EXISTS gold_bofa_stock_reaction_{_window}_backtests (
                trade_date DATE NOT NULL,
                symbol VARCHAR NOT NULL,
                window_name VARCHAR NOT NULL,
                predicted_return_pct DOUBLE PRECISION,
                predicted_return_lower_90 DOUBLE PRECISION,
                predicted_return_upper_90 DOUBLE PRECISION,
                predicted_direction VARCHAR,
                direction_confidence DOUBLE PRECISION,
                predicted_playbook VARCHAR,
                actual_return_pct DOUBLE PRECISION,
                actual_direction VARCHAR,
                error_return_pct DOUBLE PRECISION,
                absolute_error_pct DOUBLE PRECISION,
                is_direction_hit BOOLEAN,
                is_inside_90_ci BOOLEAN,
                training_start_date DATE,
                training_end_date DATE,
                training_samples INTEGER,
                model_name VARCHAR,
                model_version VARCHAR,
                calculated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                PRIMARY KEY (trade_date, symbol)
            );
        """)

    logger.info("Gold schemas initialized (Models 1, 2, 3 — including Stock Reaction tables).")
