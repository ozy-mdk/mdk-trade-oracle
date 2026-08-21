"""Gold Layer schema definitions in DuckDB."""

from mdk_trading_oracle.core.db import DuckDBManager
from mdk_trading_oracle.core.logger import get_logger

logger = get_logger("mdk_oracle.data.gold.schema")


def initialize_gold_schema(db: DuckDBManager) -> None:
    """Initialize Gold layer feature tables, institutional signals, and model forecast tables in DuckDB."""
    conn = db.get_connection()

    # 1. Rolling Institutional Flow Signals & Multi-Day Accumulation
    conn.execute("""
        CREATE TABLE IF NOT EXISTS gold_institutional_daily_signals (
            trade_date DATE,
            symbol VARCHAR,
            bofa_net_flow_tl DOUBLE,
            bofa_volume_share DOUBLE,
            bofa_flow_zscore_20d DOUBLE,
            bofa_accum_5d_tl DOUBLE,
            bofa_accum_20d_tl DOUBLE,
            market_vwap DOUBLE,
            close_price DOUBLE,
            calculated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            PRIMARY KEY (trade_date, symbol)
        );
    """)

    # 2. Model 1 Output Table: Day-Start Macro Forecasts
    conn.execute("""
        CREATE TABLE IF NOT EXISTS gold_bofa_day_start_forecasts (
            forecast_date DATE PRIMARY KEY,
            day_of_week INTEGER,
            is_monday BOOLEAN,
            predicted_open_net_flow_tl DOUBLE,
            predicted_open_flow_lower_90 DOUBLE,
            predicted_open_flow_upper_90 DOUBLE,
            predicted_direction VARCHAR,
            direction_confidence DOUBLE,
            predicted_playbook VARCHAR,
            top_predicted_buy_sector VARCHAR,
            top_predicted_sell_sector VARCHAR,
            model_name VARCHAR,
            model_version VARCHAR,
            generated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        );
    """)

    # 3. Model 2 Output Table: Day-Start Sector Allocations
    existing_tables = [r[0] for r in conn.execute("SHOW TABLES;").fetchall()]
    if "gold_bofa_sector_day_start_forecasts" in existing_tables:
        sector_cols = [r[1] for r in conn.execute("PRAGMA table_info('gold_bofa_sector_day_start_forecasts');").fetchall()]
        if "day_of_week" not in sector_cols:
            conn.execute("DROP TABLE gold_bofa_sector_day_start_forecasts;")

    conn.execute("""
        CREATE TABLE IF NOT EXISTS gold_bofa_sector_day_start_forecasts (
            forecast_date DATE,
            sector VARCHAR,
            day_of_week INTEGER,
            is_monday BOOLEAN,
            predicted_open_net_flow_tl DOUBLE,
            predicted_open_flow_lower_90 DOUBLE,
            predicted_open_flow_upper_90 DOUBLE,
            predicted_direction VARCHAR,
            direction_confidence DOUBLE,
            predicted_playbook VARCHAR,
            model_name VARCHAR,
            model_version VARCHAR,
            generated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            PRIMARY KEY (forecast_date, sector)
        );
    """)

    # 4. Model 1 Dedicated Historical Performance Tracking Ledger
    conn.execute("""
        CREATE TABLE IF NOT EXISTS gold_bofa_day_start_performance (
            trade_date DATE PRIMARY KEY,
            day_of_week INTEGER,
            is_monday BOOLEAN,
            predicted_open_net_flow_tl DOUBLE,
            predicted_open_flow_lower_90 DOUBLE,
            predicted_open_flow_upper_90 DOUBLE,
            actual_open_net_flow_tl DOUBLE,
            error_open_net_flow_tl DOUBLE,
            absolute_error_tl DOUBLE,
            predicted_direction VARCHAR,
            actual_direction VARCHAR,
            is_direction_hit BOOLEAN,
            is_inside_90_ci BOOLEAN,
            direction_confidence DOUBLE,
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
    conn.execute("""
        CREATE TABLE IF NOT EXISTS gold_bofa_sector_day_start_performance (
            trade_date DATE,
            sector VARCHAR,
            day_of_week INTEGER,
            is_monday BOOLEAN,
            predicted_open_net_flow_tl DOUBLE,
            predicted_open_flow_lower_90 DOUBLE,
            predicted_open_flow_upper_90 DOUBLE,
            actual_open_net_flow_tl DOUBLE,
            error_open_net_flow_tl DOUBLE,
            absolute_error_tl DOUBLE,
            predicted_direction VARCHAR,
            actual_direction VARCHAR,
            is_direction_hit BOOLEAN,
            is_inside_90_ci BOOLEAN,
            direction_confidence DOUBLE,
            predicted_playbook VARCHAR,
            model_name VARCHAR,
            model_version VARCHAR,
            forecast_generated_at TIMESTAMP,
            realized_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            PRIMARY KEY (trade_date, sector)
        );
    """)

    # 6. Model 1 Dedicated Historical Simulation Backtest Table
    conn.execute("""
        CREATE TABLE IF NOT EXISTS gold_bofa_day_start_backtests (
            trade_date DATE PRIMARY KEY,
            day_of_week INTEGER,
            is_monday BOOLEAN,
            predicted_open_net_flow_tl DOUBLE,
            predicted_open_flow_lower_90 DOUBLE,
            predicted_open_flow_upper_90 DOUBLE,
            actual_open_net_flow_tl DOUBLE,
            error_open_net_flow_tl DOUBLE,
            predicted_direction VARCHAR,
            actual_direction VARCHAR,
            is_direction_hit BOOLEAN,
            is_inside_90_ci BOOLEAN,
            direction_confidence DOUBLE,
            predicted_playbook VARCHAR,
            top_predicted_buy_sector VARCHAR,
            top_predicted_sell_sector VARCHAR,
            model_name VARCHAR,
            model_version VARCHAR,
            calculated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        );
    """)

    # 7. Model 2 Dedicated Historical Sector Simulation Backtest Table
    conn.execute("""
        CREATE TABLE IF NOT EXISTS gold_bofa_sector_day_start_backtests (
            trade_date DATE,
            sector VARCHAR,
            day_of_week INTEGER,
            is_monday BOOLEAN,
            predicted_open_net_flow_tl DOUBLE,
            predicted_open_flow_lower_90 DOUBLE,
            predicted_open_flow_upper_90 DOUBLE,
            actual_open_net_flow_tl DOUBLE,
            error_open_net_flow_tl DOUBLE,
            predicted_direction VARCHAR,
            actual_direction VARCHAR,
            is_direction_hit BOOLEAN,
            is_inside_90_ci BOOLEAN,
            direction_confidence DOUBLE,
            predicted_playbook VARCHAR,
            model_name VARCHAR,
            model_version VARCHAR,
            calculated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            PRIMARY KEY (trade_date, sector)
        );
    """)

    logger.info("DuckDB Gold schemas initialized.")

