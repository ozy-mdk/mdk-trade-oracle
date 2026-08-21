"""Silver Layer schema definitions in DuckDB."""

from mdk_trading_oracle.core.db import DuckDBManager
from mdk_trading_oracle.core.logger import get_logger

logger = get_logger("mdk_oracle.data.silver.schema")


def initialize_silver_schema(db: DuckDBManager) -> None:
    """Initialize Silver layer aggregation, sector, broker overview, and intraday window tables in DuckDB."""
    conn = db.get_connection()

    # 1. Silver Table: Daily Stock x Broker Summary
    conn.execute("""
        CREATE TABLE IF NOT EXISTS silver_daily_broker_summary (
            trade_date DATE,
            symbol VARCHAR,
            symbol_name VARCHAR,
            sector VARCHAR,
            broker_id VARCHAR,
            broker_name VARCHAR,
            broker_category VARCHAR,
            is_primary_target BOOLEAN DEFAULT FALSE,
            buy_volume DOUBLE,
            buy_turnover_tl DOUBLE,
            buy_vwap DOUBLE,
            buy_trade_count BIGINT,
            sell_volume DOUBLE,
            sell_turnover_tl DOUBLE,
            sell_vwap DOUBLE,
            sell_trade_count BIGINT,
            total_volume DOUBLE,
            total_turnover_tl DOUBLE,
            total_vwap DOUBLE,
            net_volume DOUBLE,
            net_flow_tl DOUBLE,
            broker_symbol_turnover_share DOUBLE,
            calculated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            PRIMARY KEY (trade_date, symbol, broker_id)
        );
    """)

    # 2. Silver Table: Macro Broker Daily Overview (Market Share %, Rank, Top-5 Flag)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS silver_daily_broker_overview (
            trade_date DATE,
            day_of_week INTEGER,
            is_monday BOOLEAN,
            is_friday BOOLEAN,
            broker_id VARCHAR,
            broker_name VARCHAR,
            broker_category VARCHAR,
            is_primary_target BOOLEAN DEFAULT FALSE,
            total_buy_turnover_tl DOUBLE,
            total_sell_turnover_tl DOUBLE,
            net_flow_tl DOUBLE,
            total_turnover_tl DOUBLE,
            total_buy_volume DOUBLE,
            total_sell_volume DOUBLE,
            total_volume DOUBLE,
            total_trades BIGINT,
            active_symbols_traded BIGINT,
            market_turnover_share DOUBLE,
            market_turnover_rank INTEGER,
            market_net_flow_rank INTEGER,
            is_top_5_broker BOOLEAN DEFAULT FALSE,
            top_bought_symbol VARCHAR,
            top_sold_symbol VARCHAR,
            top_sector_name VARCHAR,
            top_sector_share DOUBLE,
            calculated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            PRIMARY KEY (trade_date, broker_id)
        );
    """)

    # 3. Silver Table: Daily Stock Summary (OHLCV, CR5 concentration, top desks, BofA footprint)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS silver_daily_stock_summary (
            trade_date DATE,
            day_of_week INTEGER,
            is_monday BOOLEAN,
            is_friday BOOLEAN,
            symbol VARCHAR,
            symbol_name VARCHAR,
            sector VARCHAR,
            index_name VARCHAR,
            open_price DOUBLE,
            high_price DOUBLE,
            low_price DOUBLE,
            close_price DOUBLE,
            market_vwap DOUBLE,
            daily_return_pct DOUBLE,
            price_range_pct DOUBLE,
            total_volume DOUBLE,
            total_turnover_tl DOUBLE,
            total_trades BIGINT,
            active_brokers_count BIGINT,
            top_buyer_broker_id VARCHAR,
            top_buyer_turnover_tl DOUBLE,
            top_buyer_share DOUBLE,
            top_seller_broker_id VARCHAR,
            top_seller_turnover_tl DOUBLE,
            top_seller_share DOUBLE,
            top_5_buyers_net_flow_tl DOUBLE,
            top_5_sellers_net_flow_tl DOUBLE,
            top_5_concentration_ratio DOUBLE,
            top_5_domestic_net_flow_tl DOUBLE,
            bofa_buy_turnover_tl DOUBLE,
            bofa_sell_turnover_tl DOUBLE,
            bofa_net_flow_tl DOUBLE,
            bofa_stock_turnover_share DOUBLE,
            bofa_buy_vwap DOUBLE,
            bofa_sell_vwap DOUBLE,
            bofa_total_vwap DOUBLE,
            bofa_vwap_spread_pct DOUBLE,
            bofa_rank_in_stock INTEGER,
            calculated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            PRIMARY KEY (trade_date, symbol)
        );
    """)

    # 4. Silver Table: Daily Sector Summary (Returns, Breadth, Institutional Inflow)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS silver_daily_sector_summary (
            trade_date DATE,
            sector VARCHAR,
            broker_id VARCHAR,
            broker_name VARCHAR,
            broker_category VARCHAR,
            is_primary_target BOOLEAN DEFAULT FALSE,
            buy_volume DOUBLE,
            buy_turnover_tl DOUBLE,
            sell_volume DOUBLE,
            sell_turnover_tl DOUBLE,
            total_volume DOUBLE,
            total_turnover_tl DOUBLE,
            net_volume DOUBLE,
            net_flow_tl DOUBLE,
            active_symbols_count BIGINT,
            trade_count BIGINT,
            sector_turnover_share DOUBLE,
            calculated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            PRIMARY KEY (trade_date, sector, broker_id)
        );
    """)

    # 5. Silver Table: Intraday Broker Window Summary (Stock x Broker x Time Window)
    conn.execute("""
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
            buy_volume DOUBLE,
            buy_turnover_tl DOUBLE,
            buy_vwap DOUBLE,
            sell_volume DOUBLE,
            sell_turnover_tl DOUBLE,
            sell_vwap DOUBLE,
            total_volume DOUBLE,
            total_turnover_tl DOUBLE,
            net_volume DOUBLE,
            net_flow_tl DOUBLE,
            trade_count BIGINT,
            calculated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            PRIMARY KEY (trade_date, symbol, broker_id, window_name)
        );
    """)

    # 6. Silver Table: Intraday Sector Window Summary (Sector x Broker x Time Window)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS silver_intraday_sector_window_summary (
            trade_date DATE,
            sector VARCHAR,
            broker_id VARCHAR,
            broker_name VARCHAR,
            is_primary_target BOOLEAN DEFAULT FALSE,
            window_name VARCHAR,
            window_order INTEGER,
            buy_volume DOUBLE,
            buy_turnover_tl DOUBLE,
            sell_volume DOUBLE,
            sell_turnover_tl DOUBLE,
            total_volume DOUBLE,
            total_turnover_tl DOUBLE,
            net_volume DOUBLE,
            net_flow_tl DOUBLE,
            active_symbols_count BIGINT,
            trade_count BIGINT,
            calculated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            PRIMARY KEY (trade_date, sector, broker_id, window_name)
        );
    """)

    # 7. Backward compatibility table: silver_market_daily
    conn.execute("""
        CREATE TABLE IF NOT EXISTS silver_market_daily (
            trade_date DATE,
            symbol VARCHAR,
            open_price DOUBLE,
            high_price DOUBLE,
            low_price DOUBLE,
            close_price DOUBLE,
            total_volume DOUBLE,
            total_turnover_tl DOUBLE,
            market_vwap DOUBLE,
            total_trades BIGINT,
            active_brokers BIGINT,
            bofa_net_flow_tl DOUBLE,
            bofa_volume_share DOUBLE,
            calculated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            PRIMARY KEY (trade_date, symbol)
        );
    """)

    # 8. Silver Macro Table: Daily Policy Interest Rates & Decision Momentum
    conn.execute("""
        CREATE TABLE IF NOT EXISTS silver_daily_macro_rates (
            trade_date DATE PRIMARY KEY,
            interest_rate DOUBLE NOT NULL,
            rate_change DOUBLE DEFAULT 0.0,
            is_rate_change_day BOOLEAN DEFAULT FALSE,
            days_since_last_rate_change INTEGER,
            days_since_last_hike INTEGER,
            days_since_last_cut INTEGER,
            last_rate_change_bps DOUBLE DEFAULT 0.0,
            rolling_30d_rate_mean DOUBLE,
            rate_spread_vs_30d_mean DOUBLE,
            daily_carry_cost_bps DOUBLE,
            is_forward_filled BOOLEAN DEFAULT FALSE,
            calculated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        );
    """)

    # 9. Silver Distribution Table: BofA Historical Flow Percentile Thresholds
    conn.execute("""
        CREATE TABLE IF NOT EXISTS silver_bofa_historical_flow_thresholds (
            scope_type VARCHAR NOT NULL,       -- 'MACRO' or 'SECTOR'
            scope_name VARCHAR NOT NULL,       -- 'ALL' (for Macro) or Sector Name (e.g. 'Banking')
            broker_id VARCHAR NOT NULL,        -- 'MLB'
            window_name VARCHAR NOT NULL,      -- 'day_start'
            buy_p25_tl DOUBLE NOT NULL,
            buy_p50_tl DOUBLE NOT NULL,
            buy_p85_tl DOUBLE NOT NULL,
            buy_count INTEGER NOT NULL,
            sell_p25_tl DOUBLE NOT NULL,
            sell_p50_tl DOUBLE NOT NULL,
            sell_p85_tl DOUBLE NOT NULL,
            sell_count INTEGER NOT NULL,
            total_sessions INTEGER NOT NULL,
            calculated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            PRIMARY KEY (scope_type, scope_name, broker_id, window_name)
        );
    """)

    logger.info("DuckDB Silver schemas initialized for all core aggregation, macro, and distribution tables.")
