"""Silver Layer schema definitions for PostgreSQL and TimescaleDB."""

from mdk_trading_oracle.core.db import PostgresManager
from mdk_trading_oracle.core.logger import get_logger

logger = get_logger("mdk_oracle.data.silver.schema")


def initialize_silver_schema(db: PostgresManager) -> None:
    """Initialize Silver layer aggregation, candlestick, sector, broker overview, and window tables."""

    # 0. React Candlestick Chart Aggregates (1m, 5m, 1d) with BofA Net Flow
    # Attempt creating TimescaleDB continuous aggregates if hypertable exists; fallback to table
    try:
        db.execute("""
            CREATE MATERIALIZED VIEW IF NOT EXISTS silver_candles_5m
            WITH (timescaledb.continuous) AS
            SELECT
                symbol,
                time_bucket('5 minutes', timestamp) AS candle_time,
                first(price, timestamp)              AS open_price,
                max(price)                            AS high_price,
                min(price)                            AS low_price,
                last(price, timestamp)               AS close_price,
                sum(volume)                           AS total_volume,
                sum(price * volume)                   AS total_turnover_tl,
                count(*)                              AS trade_count,
                sum(CASE WHEN buyer_broker_id = 'MLB' THEN price * volume ELSE 0 END) -
                sum(CASE WHEN seller_broker_id = 'MLB' THEN price * volume ELSE 0 END) AS bofa_net_flow_tl
            FROM bronze_raw_trades
            GROUP BY symbol, time_bucket('5 minutes', timestamp);
        """)
        db.execute("""
            CREATE MATERIALIZED VIEW IF NOT EXISTS silver_candles_1m
            WITH (timescaledb.continuous) AS
            SELECT
                symbol,
                time_bucket('1 minute', timestamp)  AS candle_time,
                first(price, timestamp)              AS open_price,
                max(price)                            AS high_price,
                min(price)                            AS low_price,
                last(price, timestamp)               AS close_price,
                sum(volume)                           AS total_volume,
                sum(price * volume)                   AS total_turnover_tl,
                count(*)                              AS trade_count,
                sum(CASE WHEN buyer_broker_id = 'MLB' THEN price * volume ELSE 0 END) -
                sum(CASE WHEN seller_broker_id = 'MLB' THEN price * volume ELSE 0 END) AS bofa_net_flow_tl
            FROM bronze_raw_trades
            GROUP BY symbol, time_bucket('1 minute', timestamp);
        """)
    except Exception as e:
        logger.debug(f"Timescale continuous aggregates creation notice (fallback to tables): {e}")
        db.execute("""
            CREATE TABLE IF NOT EXISTS silver_candles_5m (
                symbol VARCHAR NOT NULL,
                candle_time TIMESTAMP NOT NULL,
                open_price DOUBLE PRECISION,
                high_price DOUBLE PRECISION,
                low_price DOUBLE PRECISION,
                close_price DOUBLE PRECISION,
                total_volume DOUBLE PRECISION,
                total_turnover_tl DOUBLE PRECISION,
                trade_count BIGINT,
                bofa_net_flow_tl DOUBLE PRECISION,
                PRIMARY KEY (symbol, candle_time)
            );
        """)
        db.execute("""
            CREATE TABLE IF NOT EXISTS silver_candles_1m (
                symbol VARCHAR NOT NULL,
                candle_time TIMESTAMP NOT NULL,
                open_price DOUBLE PRECISION,
                high_price DOUBLE PRECISION,
                low_price DOUBLE PRECISION,
                close_price DOUBLE PRECISION,
                total_volume DOUBLE PRECISION,
                total_turnover_tl DOUBLE PRECISION,
                trade_count BIGINT,
                bofa_net_flow_tl DOUBLE PRECISION,
                PRIMARY KEY (symbol, candle_time)
            );
        """)

    # 1. Silver Table: Daily Stock x Broker Summary
    db.execute("""
        CREATE TABLE IF NOT EXISTS silver_daily_broker_summary (
            trade_date DATE,
            symbol VARCHAR,
            symbol_name VARCHAR,
            sector VARCHAR,
            broker_id VARCHAR,
            broker_name VARCHAR,
            broker_category VARCHAR,
            is_primary_target BOOLEAN DEFAULT FALSE,
            buy_volume DOUBLE PRECISION,
            buy_turnover_tl DOUBLE PRECISION,
            buy_vwap DOUBLE PRECISION,
            buy_trade_count BIGINT,
            sell_volume DOUBLE PRECISION,
            sell_turnover_tl DOUBLE PRECISION,
            sell_vwap DOUBLE PRECISION,
            sell_trade_count BIGINT,
            total_volume DOUBLE PRECISION,
            total_turnover_tl DOUBLE PRECISION,
            total_vwap DOUBLE PRECISION,
            net_volume DOUBLE PRECISION,
            net_flow_tl DOUBLE PRECISION,
            broker_symbol_turnover_share DOUBLE PRECISION,
            calculated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            PRIMARY KEY (trade_date, symbol, broker_id)
        );
    """)

    # 2. Silver Table: Macro Broker Daily Overview (Market Share %, Rank, Top-5 Flag)
    db.execute("""
        CREATE TABLE IF NOT EXISTS silver_daily_broker_overview (
            trade_date DATE,
            day_of_week INTEGER,
            is_monday BOOLEAN,
            is_friday BOOLEAN,
            broker_id VARCHAR,
            broker_name VARCHAR,
            broker_category VARCHAR,
            is_primary_target BOOLEAN DEFAULT FALSE,
            total_buy_turnover_tl DOUBLE PRECISION,
            total_sell_turnover_tl DOUBLE PRECISION,
            net_flow_tl DOUBLE PRECISION,
            total_turnover_tl DOUBLE PRECISION,
            total_buy_volume DOUBLE PRECISION,
            total_sell_volume DOUBLE PRECISION,
            total_volume DOUBLE PRECISION,
            total_trades BIGINT,
            active_symbols_traded BIGINT,
            market_turnover_share DOUBLE PRECISION,
            market_turnover_rank INTEGER,
            market_net_flow_rank INTEGER,
            is_top_5_broker BOOLEAN DEFAULT FALSE,
            top_bought_symbol VARCHAR,
            top_sold_symbol VARCHAR,
            top_sector_name VARCHAR,
            top_sector_share DOUBLE PRECISION,
            calculated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            PRIMARY KEY (trade_date, broker_id)
        );
    """)

    # 3. Silver Table: Daily Stock Summary (OHLCV, CR5 concentration, top desks, BofA footprint, adjusted metrics)
    db.execute("""
        CREATE TABLE IF NOT EXISTS silver_daily_stock_summary (
            trade_date DATE,
            day_of_week INTEGER,
            is_monday BOOLEAN,
            is_friday BOOLEAN,
            symbol VARCHAR,
            canonical_symbol VARCHAR,
            symbol_name VARCHAR,
            sector VARCHAR,
            index_name VARCHAR,
            open_price DOUBLE PRECISION,
            high_price DOUBLE PRECISION,
            low_price DOUBLE PRECISION,
            close_price DOUBLE PRECISION,
            market_vwap DOUBLE PRECISION,
            daily_return_pct DOUBLE PRECISION,
            price_range_pct DOUBLE PRECISION,
            total_volume DOUBLE PRECISION,
            total_turnover_tl DOUBLE PRECISION,
            total_trades BIGINT,
            active_brokers_count BIGINT,
            quantity_factor DOUBLE PRECISION DEFAULT 1.0,
            has_unresolved_paid_action BOOLEAN DEFAULT FALSE,
            adj_open_price DOUBLE PRECISION,
            adj_high_price DOUBLE PRECISION,
            adj_low_price DOUBLE PRECISION,
            adj_close_price DOUBLE PRECISION,
            adj_market_vwap DOUBLE PRECISION,
            adj_total_volume DOUBLE PRECISION,
            adj_daily_return_pct DOUBLE PRECISION,
            top_buyer_broker_id VARCHAR,
            top_buyer_turnover_tl DOUBLE PRECISION,
            top_buyer_share DOUBLE PRECISION,
            top_seller_broker_id VARCHAR,
            top_seller_turnover_tl DOUBLE PRECISION,
            top_seller_share DOUBLE PRECISION,
            top_5_buyers_net_flow_tl DOUBLE PRECISION,
            top_5_sellers_net_flow_tl DOUBLE PRECISION,
            top_5_concentration_ratio DOUBLE PRECISION,
            top_5_domestic_net_flow_tl DOUBLE PRECISION,
            bofa_buy_turnover_tl DOUBLE PRECISION,
            bofa_sell_turnover_tl DOUBLE PRECISION,
            bofa_net_flow_tl DOUBLE PRECISION,
            bofa_stock_turnover_share DOUBLE PRECISION,
            bofa_buy_vwap DOUBLE PRECISION,
            bofa_sell_vwap DOUBLE PRECISION,
            bofa_total_vwap DOUBLE PRECISION,
            bofa_vwap_spread_pct DOUBLE PRECISION,
            adj_bofa_buy_vwap DOUBLE PRECISION,
            adj_bofa_sell_vwap DOUBLE PRECISION,
            adj_bofa_total_vwap DOUBLE PRECISION,
            bofa_rank_in_stock INTEGER,
            calculated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            PRIMARY KEY (trade_date, symbol)
        );
    """)

    # 4. Silver Table: Daily Sector Summary (Returns, Breadth, Institutional Inflow)
    db.execute("""
        CREATE TABLE IF NOT EXISTS silver_daily_sector_summary (
            trade_date DATE,
            sector VARCHAR,
            broker_id VARCHAR,
            broker_name VARCHAR,
            broker_category VARCHAR,
            is_primary_target BOOLEAN DEFAULT FALSE,
            buy_volume DOUBLE PRECISION,
            buy_turnover_tl DOUBLE PRECISION,
            sell_volume DOUBLE PRECISION,
            sell_turnover_tl DOUBLE PRECISION,
            total_volume DOUBLE PRECISION,
            total_turnover_tl DOUBLE PRECISION,
            net_volume DOUBLE PRECISION,
            net_flow_tl DOUBLE PRECISION,
            active_symbols_count BIGINT,
            trade_count BIGINT,
            sector_turnover_share DOUBLE PRECISION,
            calculated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            PRIMARY KEY (trade_date, sector, broker_id)
        );
    """)

    # 5. Silver Table: Intraday Broker Window Summary (Stock x Broker x Time Window)
    db.execute("""
        CREATE TABLE IF NOT EXISTS silver_intraday_broker_window_summary (
            trade_date DATE,
            symbol VARCHAR,
            sector VARCHAR,
            broker_id VARCHAR,
            broker_name VARCHAR,
            is_primary_target BOOLEAN DEFAULT FALSE,
            window_name VARCHAR,
            window_order INTEGER,
            window_start_time VARCHAR,
            window_end_time VARCHAR,
            buy_volume DOUBLE PRECISION,
            buy_turnover_tl DOUBLE PRECISION,
            buy_vwap DOUBLE PRECISION,
            sell_volume DOUBLE PRECISION,
            sell_turnover_tl DOUBLE PRECISION,
            sell_vwap DOUBLE PRECISION,
            total_volume DOUBLE PRECISION,
            total_turnover_tl DOUBLE PRECISION,
            net_volume DOUBLE PRECISION,
            net_flow_tl DOUBLE PRECISION,
            trade_count BIGINT,
            calculated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            PRIMARY KEY (trade_date, symbol, broker_id, window_name)
        );
    """)

    # 6. Silver Table: Intraday Sector Window Summary (Sector x Broker x Time Window)
    db.execute("""
        CREATE TABLE IF NOT EXISTS silver_intraday_sector_window_summary (
            trade_date DATE,
            sector VARCHAR,
            broker_id VARCHAR,
            broker_name VARCHAR,
            is_primary_target BOOLEAN DEFAULT FALSE,
            window_name VARCHAR,
            window_order INTEGER,
            buy_volume DOUBLE PRECISION,
            buy_turnover_tl DOUBLE PRECISION,
            sell_volume DOUBLE PRECISION,
            sell_turnover_tl DOUBLE PRECISION,
            total_volume DOUBLE PRECISION,
            total_turnover_tl DOUBLE PRECISION,
            net_volume DOUBLE PRECISION,
            net_flow_tl DOUBLE PRECISION,
            active_symbols_count BIGINT,
            trade_count BIGINT,
            calculated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            PRIMARY KEY (trade_date, sector, broker_id, window_name)
        );
    """)

    # 7. Backward compatibility table: silver_market_daily
    db.execute("""
        CREATE TABLE IF NOT EXISTS silver_market_daily (
            trade_date DATE,
            symbol VARCHAR,
            open_price DOUBLE PRECISION,
            high_price DOUBLE PRECISION,
            low_price DOUBLE PRECISION,
            close_price DOUBLE PRECISION,
            total_volume DOUBLE PRECISION,
            total_turnover_tl DOUBLE PRECISION,
            market_vwap DOUBLE PRECISION,
            total_trades BIGINT,
            active_brokers BIGINT,
            bofa_net_flow_tl DOUBLE PRECISION,
            bofa_volume_share DOUBLE PRECISION,
            calculated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            PRIMARY KEY (trade_date, symbol)
        );
    """)

    # 8. Silver Macro Table: Daily Policy Interest Rates & Decision Momentum
    db.execute("""
        CREATE TABLE IF NOT EXISTS silver_daily_macro_rates (
            trade_date DATE PRIMARY KEY,
            interest_rate DOUBLE PRECISION NOT NULL,
            rate_change DOUBLE PRECISION DEFAULT 0.0,
            is_rate_change_day BOOLEAN DEFAULT FALSE,
            days_since_last_rate_change INTEGER,
            days_since_last_hike INTEGER,
            days_since_last_cut INTEGER,
            last_rate_change_bps DOUBLE PRECISION DEFAULT 0.0,
            rate_change_decay_bps DOUBLE PRECISION DEFAULT 0.0,
            rolling_30d_rate_mean DOUBLE PRECISION,
            rate_spread_vs_30d_mean DOUBLE PRECISION,
            daily_carry_cost_bps DOUBLE PRECISION,
            is_forward_filled BOOLEAN DEFAULT FALSE,
            calculated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        );
    """)

    # 9. Silver Distribution Table: BofA Historical Flow Percentile Thresholds
    db.execute("""
        CREATE TABLE IF NOT EXISTS silver_bofa_historical_flow_thresholds (
            scope_type VARCHAR NOT NULL,       -- 'MACRO' or 'SECTOR'
            scope_name VARCHAR NOT NULL,       -- 'ALL' (for Macro) or Sector Name (e.g. 'Banking')
            broker_id VARCHAR NOT NULL,        -- 'MLB'
            window_name VARCHAR NOT NULL,      -- 'day_start'
            buy_p25_tl DOUBLE PRECISION NOT NULL,
            buy_p50_tl DOUBLE PRECISION NOT NULL,
            buy_p85_tl DOUBLE PRECISION NOT NULL,
            buy_count INTEGER NOT NULL,
            sell_p25_tl DOUBLE PRECISION NOT NULL,
            sell_p50_tl DOUBLE PRECISION NOT NULL,
            sell_p85_tl DOUBLE PRECISION NOT NULL,
            sell_count INTEGER NOT NULL,
            total_sessions INTEGER NOT NULL,
            calculated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            PRIMARY KEY (scope_type, scope_name, broker_id, window_name)
        );
    """)

    # 10. Silver Benchmark Table: Daily BIST 30 Benchmark Metrics & Rolling Indicators
    db.execute("""
        CREATE TABLE IF NOT EXISTS silver_daily_benchmark_index (
            trade_date DATE PRIMARY KEY,
            index_code VARCHAR DEFAULT 'XU030',
            open_price DOUBLE PRECISION,
            high_price DOUBLE PRECISION,
            low_price DOUBLE PRECISION,
            close_price DOUBLE PRECISION NOT NULL,
            volume DOUBLE PRECISION,
            daily_return_pct DOUBLE PRECISION,
            intraday_return_pct DOUBLE PRECISION,
            price_range_pct DOUBLE PRECISION,
            rolling_5d_return_pct DOUBLE PRECISION,
            rolling_20d_return_pct DOUBLE PRECISION,
            rolling_20d_volatility DOUBLE PRECISION,
            index_trend_vs_20d_sma DOUBLE PRECISION,
            is_forward_filled BOOLEAN DEFAULT FALSE,
            calculated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        );
    """)

    # 11. Silver Tertip Table: Daily Broker FIFO Matched Flow & Inventory Ledger
    db.execute("""
        CREATE TABLE IF NOT EXISTS silver_broker_fifo_daily (
            trade_date DATE,
            symbol VARCHAR,
            symbol_name VARCHAR,
            sector VARCHAR,
            broker_id VARCHAR,
            broker_name VARCHAR,
            buy_volume DOUBLE PRECISION,
            buy_turnover_tl DOUBLE PRECISION,
            buy_vwap DOUBLE PRECISION,
            sell_volume DOUBLE PRECISION,
            sell_turnover_tl DOUBLE PRECISION,
            sell_vwap DOUBLE PRECISION,
            matched_volume DOUBLE PRECISION,
            matched_buy_value_tl DOUBLE PRECISION,
            matched_sell_value_tl DOUBLE PRECISION,
            intraday_realized_pnl_tl DOUBLE PRECISION,
            residual_volume DOUBLE PRECISION,
            residual_value_tl DOUBLE PRECISION,
            residual_flow_unit_cost DOUBLE PRECISION,
            carry_fifo_realized_pnl_tl DOUBLE PRECISION,
            daily_realized_pnl_tl DOUBLE PRECISION,
            position_side VARCHAR,
            open_stock_quantity DOUBLE PRECISION,
            open_fifo_cost_tl DOUBLE PRECISION,
            fifo_avg_cost DOUBLE PRECISION,
            market_close_price DOUBLE PRECISION,
            market_value_tl DOUBLE PRECISION,
            unrealized_pnl_tl DOUBLE PRECISION,
            total_daily_pnl_tl DOUBLE PRECISION,
            cumulative_realized_pnl_tl DOUBLE PRECISION,
            calculated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            PRIMARY KEY (trade_date, symbol, broker_id)
        );
    """)

    # 12. Silver Tertip Table: Immutable FIFO Lot Entries
    db.execute("""
        CREATE TABLE IF NOT EXISTS silver_broker_fifo_lot_entries (
            lot_id VARCHAR PRIMARY KEY,
            broker_id VARCHAR NOT NULL,
            symbol VARCHAR NOT NULL,
            direction VARCHAR NOT NULL,
            open_date DATE NOT NULL,
            opened_quantity DOUBLE PRECISION NOT NULL,
            opened_value_tl DOUBLE PRECISION NOT NULL,
            opened_unit_cost DOUBLE PRECISION NOT NULL,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        );
    """)

    # 13. Silver Tertip Table: Currently Active Open Lots
    db.execute("""
        CREATE TABLE IF NOT EXISTS silver_broker_fifo_lots (
            lot_id VARCHAR PRIMARY KEY,
            broker_id VARCHAR NOT NULL,
            symbol VARCHAR NOT NULL,
            direction VARCHAR NOT NULL,
            open_date DATE NOT NULL,
            remaining_quantity DOUBLE PRECISION NOT NULL,
            remaining_value_tl DOUBLE PRECISION NOT NULL,
            unit_cost DOUBLE PRECISION NOT NULL,
            last_updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        );
    """)

    # 14. Silver Tertip Table: Audited FIFO Lot Realizations
    db.execute("""
        CREATE TABLE IF NOT EXISTS silver_broker_fifo_lot_realizations (
            realization_id VARCHAR PRIMARY KEY,
            lot_id VARCHAR NOT NULL,
            broker_id VARCHAR NOT NULL,
            symbol VARCHAR NOT NULL,
            close_date DATE NOT NULL,
            direction VARCHAR NOT NULL,
            quantity_closed DOUBLE PRECISION NOT NULL,
            entry_value_closed_tl DOUBLE PRECISION NOT NULL,
            closing_value_tl DOUBLE PRECISION NOT NULL,
            realized_pnl_tl DOUBLE PRECISION NOT NULL,
            remaining_quantity_after DOUBLE PRECISION NOT NULL,
            is_final BOOLEAN NOT NULL,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        );
    """)

    # 15. Silver Tertip Table: Lot Lifecycle Summary
    db.execute("""
        CREATE TABLE IF NOT EXISTS silver_broker_fifo_lot_lifecycle (
            lot_id VARCHAR PRIMARY KEY,
            broker_id VARCHAR NOT NULL,
            symbol VARCHAR NOT NULL,
            direction VARCHAR NOT NULL,
            open_date DATE NOT NULL,
            opened_quantity DOUBLE PRECISION NOT NULL,
            opened_value_tl DOUBLE PRECISION NOT NULL,
            opened_unit_cost DOUBLE PRECISION NOT NULL,
            status VARCHAR NOT NULL,
            closed_date DATE,
            total_quantity_closed DOUBLE PRECISION NOT NULL,
            total_entry_value_closed_tl DOUBLE PRECISION NOT NULL,
            total_closing_value_tl DOUBLE PRECISION NOT NULL,
            total_realized_pnl_tl DOUBLE PRECISION NOT NULL,
            remaining_quantity DOUBLE PRECISION NOT NULL,
            remaining_value_tl DOUBLE PRECISION NOT NULL,
            calculated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        );
    """)

    # 16. Silver Corporate Actions Table: Continuous Point-in-Time Share Adjustment Periods
    db.execute("""
        CREATE TABLE IF NOT EXISTS silver_corporate_action_adjustment_periods (
            source_symbol VARCHAR NOT NULL,
            effective_from DATE NOT NULL,
            effective_to DATE NOT NULL,
            canonical_symbol VARCHAR NOT NULL,
            quantity_factor DOUBLE PRECISION NOT NULL,
            has_unresolved_paid_action BOOLEAN NOT NULL,
            calculated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            PRIMARY KEY (source_symbol, effective_from)
        );
    """)

    # 17. Silver Stock Reaction Thresholds: Per-Stock Per-Window Return Percentile Distribution
    db.execute("""
        CREATE TABLE IF NOT EXISTS silver_stock_reaction_thresholds (
            symbol VARCHAR NOT NULL,
            window_name VARCHAR NOT NULL,
            up_p25_pct DOUBLE PRECISION NOT NULL,
            up_p50_pct DOUBLE PRECISION NOT NULL,
            up_p85_pct DOUBLE PRECISION NOT NULL,
            down_p25_pct DOUBLE PRECISION NOT NULL,
            down_p50_pct DOUBLE PRECISION NOT NULL,
            down_p85_pct DOUBLE PRECISION NOT NULL,
            up_session_count INTEGER NOT NULL,
            down_session_count INTEGER NOT NULL,
            total_sessions INTEGER NOT NULL,
            calculated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            PRIMARY KEY (symbol, window_name)
        );
    """)

    logger.info(
        "Silver schemas initialized for all core aggregation, candlestick, macro, benchmark, tertip FIFO, "
        "corporate action adjustment, and stock reaction threshold tables."
    )
