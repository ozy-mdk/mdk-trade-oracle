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
        sector_cols = [
            r[1] for r in conn.execute("PRAGMA table_info('gold_bofa_sector_day_start_forecasts');").fetchall()
        ]
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

    # ──────────────────────────────────────────────────────────────────────────────────────────────
    # MODEL 3: BIST30 Stock Intraday Reaction Forecaster
    # Three windows: W2 (first_reaction), W3 (midday_followup), W5 (closing_session)
    # Three table categories per window: forecasts / performance / backtests
    # All tables use composite PK (trade_date, symbol) — all 30 BIST30 stocks share the same tables.
    # ──────────────────────────────────────────────────────────────────────────────────────────────

    for _window in ["w2", "w3", "w5"]:
        _wname = {"w2": "first_reaction", "w3": "midday_followup", "w5": "closing_session"}[_window]

        # 8/9/10. Live Upcoming Forecasts — strictly holds only the next tradeable window (replaced each run)
        conn.execute(f"""
            CREATE TABLE IF NOT EXISTS gold_bofa_stock_reaction_{_window}_forecasts (
                forecast_date DATE NOT NULL,      -- trade_date for which we are forecasting
                symbol VARCHAR NOT NULL,          -- BIST ticker (e.g. 'AKBNK')
                window_name VARCHAR NOT NULL,     -- '{_wname}'
                -- Continuous target: execution-aware return%
                predicted_return_pct DOUBLE,
                predicted_return_lower_90 DOUBLE,
                predicted_return_upper_90 DOUBLE,
                -- Direction conviction (driven by silver_stock_reaction_thresholds empirical quantiles)
                predicted_direction VARCHAR,      -- STRONG_RALLY/RALLY/WEAK_RALLY/NEUTRAL/WEAK_DECLINE/DECLINE/STRONG_DECLINE
                direction_confidence DOUBLE,
                -- Institutional playbook context
                predicted_playbook VARCHAR,       -- MOMENTUM_EXPANSION/LIQUIDITY_FADE/DEFENSE_SUPPORT/SQUEEZE_LONG/SECTOR_ROTATION/NEUTRAL_WAIT
                -- Feature cluster summary (top contributing signals)
                bofa_w1_direction VARCHAR,        -- BofA W1 execution direction (BUY/SELL/NEUTRAL)
                bofa_w1_net_flow_tl DOUBLE,       -- BofA W1 net flow TL
                bofa_w1_volume_share DOUBLE,      -- BofA W1 volume market share
                -- Metadata
                model_name VARCHAR,
                model_version VARCHAR,
                generated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                PRIMARY KEY (forecast_date, symbol)
            );
        """)

        # 11/12/13. Historical Performance Ledgers — permanent audited reconciliation against actuals
        conn.execute(f"""
            CREATE TABLE IF NOT EXISTS gold_bofa_stock_reaction_{_window}_performance (
                trade_date DATE NOT NULL,
                symbol VARCHAR NOT NULL,
                window_name VARCHAR NOT NULL,
                -- Predictions (at time of forecast)
                predicted_return_pct DOUBLE,
                predicted_return_lower_90 DOUBLE,
                predicted_return_upper_90 DOUBLE,
                predicted_direction VARCHAR,
                direction_confidence DOUBLE,
                predicted_playbook VARCHAR,
                -- Actuals (filled in by reconcile_and_update_performance_ledger)
                actual_return_pct DOUBLE,
                actual_direction VARCHAR,
                -- Error metrics
                error_return_pct DOUBLE,          -- predicted - actual
                absolute_error_pct DOUBLE,
                is_direction_hit BOOLEAN,
                is_inside_90_ci BOOLEAN,
                -- Metadata
                bofa_w1_net_flow_tl DOUBLE,
                bofa_w1_volume_share DOUBLE,
                model_name VARCHAR,
                model_version VARCHAR,
                forecast_generated_at TIMESTAMP,
                realized_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                PRIMARY KEY (trade_date, symbol)
            );
        """)

        # 14/15/16. Simulation Backtests — full historical walk-forward OOS simulation ledger
        conn.execute(f"""
            CREATE TABLE IF NOT EXISTS gold_bofa_stock_reaction_{_window}_backtests (
                trade_date DATE NOT NULL,
                symbol VARCHAR NOT NULL,
                window_name VARCHAR NOT NULL,
                -- OOS predictions
                predicted_return_pct DOUBLE,
                predicted_return_lower_90 DOUBLE,
                predicted_return_upper_90 DOUBLE,
                predicted_direction VARCHAR,
                direction_confidence DOUBLE,
                predicted_playbook VARCHAR,
                -- Actuals
                actual_return_pct DOUBLE,
                actual_direction VARCHAR,
                -- Error metrics
                error_return_pct DOUBLE,
                absolute_error_pct DOUBLE,
                is_direction_hit BOOLEAN,
                is_inside_90_ci BOOLEAN,
                -- Walk-forward metadata
                training_start_date DATE,
                training_end_date DATE,
                training_samples INTEGER,
                model_name VARCHAR,
                model_version VARCHAR,
                calculated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                PRIMARY KEY (trade_date, symbol)
            );
        """)

    logger.info("DuckDB Gold schemas initialized (Models 1, 2, 3 — including 9 Stock Reaction tables).")

