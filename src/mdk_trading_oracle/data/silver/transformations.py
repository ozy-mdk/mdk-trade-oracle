"""Silver Layer Transformations: High-Performance Multi-Broker, Stock, Sector, and Intraday Aggregations."""

from typing import Any

from mdk_trading_oracle.core.config import get_settings
from mdk_trading_oracle.core.db import DuckDBManager
from mdk_trading_oracle.core.logger import get_logger
from mdk_trading_oracle.data.silver.schema import initialize_silver_schema

logger = get_logger("mdk_oracle.data.silver.transformations")


class SilverTransformer:
    """Transforms raw Bronze tick data into clean, highly-aggregated Silver tables in DuckDB."""

    def __init__(self, db: DuckDBManager):
        self.db = db
        self.settings = get_settings()

    def _build_intraday_window_case_sql(self) -> tuple[str, str, str, str]:
        """Build non-overlapping local-time CASE statements for intraday windows."""
        windows = self.settings.get_intraday_windows()
        name_branches = []
        order_branches = []
        start_branches = []
        end_branches = []

        for idx, w in enumerate(windows):
            name = w["name"]
            start_t = w["start_time"]
            end_t = w["end_time"]
            order = w.get("order", 1)
            end_operator = "<=" if idx == len(windows) - 1 else "<"
            condition = (
                f"CAST(timestamp AS TIME) >= TIME '{start_t}' "
                f"AND CAST(timestamp AS TIME) {end_operator} TIME '{end_t}'"
            )
            name_branches.append(f"WHEN {condition} THEN '{name}'")
            order_branches.append(f"WHEN {condition} THEN {order}")
            start_branches.append(f"WHEN {condition} THEN '{start_t}'")
            end_branches.append(f"WHEN {condition} THEN '{end_t}'")

        case_name = "CASE " + " ".join(name_branches) + " ELSE NULL END"
        case_order = "CASE " + " ".join(order_branches) + " ELSE NULL END"
        case_start = "CASE " + " ".join(start_branches) + " ELSE NULL END"
        case_end = "CASE " + " ".join(end_branches) + " ELSE NULL END"

        return case_name, case_order, case_start, case_end

    def transform_daily_broker_summary(self) -> dict[str, Any]:
        """Aggregate buy/sell volume, turnover, and buy/sell VWAP per (trade_date, symbol, broker_id)."""
        conn = self.db.get_connection()
        logger.info("Computing `silver_daily_broker_summary` from `bronze_raw_trades`...")

        query = """
            CREATE OR REPLACE TABLE silver_daily_broker_summary AS
            WITH buys AS (
                SELECT 
                    CAST(timestamp AS DATE) AS trade_date,
                    symbol,
                    buyer_broker_id AS broker_id,
                    SUM(volume) AS buy_volume,
                    SUM(price * volume) AS buy_turnover_tl,
                    SUM(price * volume) / NULLIF(SUM(volume), 0) AS buy_vwap,
                    COUNT(*) AS buy_trade_count
                FROM bronze_raw_trades
                WHERE buyer_broker_id IS NOT NULL AND buyer_broker_id != ''
                GROUP BY CAST(timestamp AS DATE), symbol, buyer_broker_id
            ),
            sells AS (
                SELECT 
                    CAST(timestamp AS DATE) AS trade_date,
                    symbol,
                    seller_broker_id AS broker_id,
                    SUM(volume) AS sell_volume,
                    SUM(price * volume) AS sell_turnover_tl,
                    SUM(price * volume) / NULLIF(SUM(volume), 0) AS sell_vwap,
                    COUNT(*) AS sell_trade_count
                FROM bronze_raw_trades
                WHERE seller_broker_id IS NOT NULL AND seller_broker_id != ''
                GROUP BY CAST(timestamp AS DATE), symbol, seller_broker_id
            ),
            combined AS (
                SELECT 
                    COALESCE(b.trade_date, s.trade_date) AS trade_date,
                    COALESCE(b.symbol, s.symbol) AS symbol,
                    COALESCE(b.broker_id, s.broker_id) AS broker_id,
                    COALESCE(b.buy_volume, 0.0) AS buy_volume,
                    COALESCE(b.buy_turnover_tl, 0.0) AS buy_turnover_tl,
                    b.buy_vwap,
                    COALESCE(b.buy_trade_count, 0) AS buy_trade_count,
                    COALESCE(s.sell_volume, 0.0) AS sell_volume,
                    COALESCE(s.sell_turnover_tl, 0.0) AS sell_turnover_tl,
                    s.sell_vwap,
                    COALESCE(s.sell_trade_count, 0) AS sell_trade_count,
                    COALESCE(b.buy_volume, 0.0) + COALESCE(s.sell_volume, 0.0) AS total_volume,
                    COALESCE(b.buy_turnover_tl, 0.0) + COALESCE(s.sell_turnover_tl, 0.0) AS total_turnover_tl,
                    (COALESCE(b.buy_turnover_tl, 0.0) + COALESCE(s.sell_turnover_tl, 0.0)) / 
                        NULLIF(COALESCE(b.buy_volume, 0.0) + COALESCE(s.sell_volume, 0.0), 0.0) AS total_vwap,
                    COALESCE(b.buy_volume, 0.0) - COALESCE(s.sell_volume, 0.0) AS net_volume,
                    COALESCE(b.buy_turnover_tl, 0.0) - COALESCE(s.sell_turnover_tl, 0.0) AS net_flow_tl
                FROM buys b
                FULL OUTER JOIN sells s
                    ON b.trade_date = s.trade_date 
                    AND b.symbol = s.symbol 
                    AND b.broker_id = s.broker_id
            )
            SELECT 
                c.trade_date,
                c.symbol,
                COALESCE(inst.name, c.symbol) AS symbol_name,
                COALESCE(inst.sector, 'Unknown') AS sector,
                c.broker_id,
                COALESCE(brk.broker_name, c.broker_id) AS broker_name,
                COALESCE(brk.category, 'unknown') AS broker_category,
                COALESCE(brk.is_primary_target, FALSE) AS is_primary_target,
                c.buy_volume,
                c.buy_turnover_tl,
                c.buy_vwap,
                c.buy_trade_count,
                c.sell_volume,
                c.sell_turnover_tl,
                c.sell_vwap,
                c.sell_trade_count,
                c.total_volume,
                c.total_turnover_tl,
                c.total_vwap,
                c.net_volume,
                c.net_flow_tl,
                c.total_turnover_tl / NULLIF(SUM(c.total_turnover_tl) OVER (PARTITION BY c.trade_date, c.symbol), 0.0) AS broker_symbol_turnover_share,
                CURRENT_TIMESTAMP AS calculated_at
            FROM combined c
            LEFT JOIN bronze_brokers brk ON c.broker_id = brk.broker_id
            LEFT JOIN bronze_instruments inst ON c.symbol = inst.symbol;
        """
        conn.execute(query)

        rows = conn.execute("SELECT COUNT(*) FROM silver_daily_broker_summary;").fetchone()[0]
        logger.info(f"Successfully populated `silver_daily_broker_summary`: {rows:,} rows.")
        return {"table": "silver_daily_broker_summary", "rows": rows, "status": "success"}

    def transform_daily_broker_overview(self) -> dict[str, Any]:
        """Aggregate whole-market daily broker footprint, market share %, rankings, and Top-5 flags."""
        conn = self.db.get_connection()
        logger.info("Computing `silver_daily_broker_overview` macro market shares...")

        query = """
            CREATE OR REPLACE TABLE silver_daily_broker_overview AS
            WITH broker_totals AS (
                SELECT 
                    trade_date,
                    broker_id,
                    broker_name,
                    broker_category,
                    is_primary_target,
                    SUM(buy_turnover_tl) AS total_buy_turnover_tl,
                    SUM(sell_turnover_tl) AS total_sell_turnover_tl,
                    SUM(net_flow_tl) AS net_flow_tl,
                    SUM(total_turnover_tl) AS total_turnover_tl,
                    SUM(buy_volume) AS total_buy_volume,
                    SUM(sell_volume) AS total_sell_volume,
                    SUM(total_volume) AS total_volume,
                    SUM(buy_trade_count + sell_trade_count) AS total_trades,
                    COUNT(DISTINCT symbol) AS active_symbols_traded,
                    ARG_MAX(symbol, buy_turnover_tl) AS top_bought_symbol,
                    ARG_MAX(symbol, sell_turnover_tl) AS top_sold_symbol
                FROM silver_daily_broker_summary
                GROUP BY trade_date, broker_id, broker_name, broker_category, is_primary_target
            ),
            sector_allocation AS (
                SELECT 
                    trade_date,
                    broker_id,
                    ARG_MAX(sector, sector_turnover) AS top_sector_name,
                    MAX(sector_turnover) / NULLIF(SUM(sector_turnover), 0.0) AS top_sector_share
                FROM (
                    SELECT 
                        trade_date,
                        broker_id,
                        sector,
                        SUM(total_turnover_tl) AS sector_turnover
                    FROM silver_daily_broker_summary
                    GROUP BY trade_date, broker_id, sector
                )
                GROUP BY trade_date, broker_id
            ),
            market_daily_total AS (
                SELECT 
                    trade_date,
                    SUM(total_turnover_tl) AS market_total_turnover_tl
                FROM broker_totals
                GROUP BY trade_date
            )
            SELECT 
                b.trade_date,
                EXTRACT(DOW FROM b.trade_date) AS day_of_week,
                (EXTRACT(DOW FROM b.trade_date) = 1) AS is_monday,
                (EXTRACT(DOW FROM b.trade_date) = 5) AS is_friday,
                b.broker_id,
                b.broker_name,
                b.broker_category,
                b.is_primary_target,
                b.total_buy_turnover_tl,
                b.total_sell_turnover_tl,
                b.net_flow_tl,
                b.total_turnover_tl,
                b.total_buy_volume,
                b.total_sell_volume,
                b.total_volume,
                b.total_trades,
                b.active_symbols_traded,
                b.total_turnover_tl / NULLIF(m.market_total_turnover_tl, 0.0) AS market_turnover_share,
                DENSE_RANK() OVER (PARTITION BY b.trade_date ORDER BY b.total_turnover_tl DESC) AS market_turnover_rank,
                DENSE_RANK() OVER (PARTITION BY b.trade_date ORDER BY b.net_flow_tl DESC) AS market_net_flow_rank,
                (DENSE_RANK() OVER (PARTITION BY b.trade_date ORDER BY b.total_turnover_tl DESC) <= 5) AS is_top_5_broker,
                b.top_bought_symbol,
                b.top_sold_symbol,
                sa.top_sector_name,
                sa.top_sector_share,
                CURRENT_TIMESTAMP AS calculated_at
            FROM broker_totals b
            JOIN market_daily_total m ON b.trade_date = m.trade_date
            LEFT JOIN sector_allocation sa ON b.trade_date = sa.trade_date AND b.broker_id = sa.broker_id;
        """
        conn.execute(query)

        rows = conn.execute("SELECT COUNT(*) FROM silver_daily_broker_overview;").fetchone()[0]
        logger.info(f"Successfully populated `silver_daily_broker_overview`: {rows:,} rows.")
        return {"table": "silver_daily_broker_overview", "rows": rows, "status": "success"}

    def transform_daily_stock_summary(self) -> dict[str, Any]:
        """Compute Daily Stock Summary with OHLCV, VWAPs, CR5 concentration, and BofA price levels."""
        conn = self.db.get_connection()
        logger.info("Computing `silver_daily_stock_summary` (and syncing `silver_market_daily`)...")

        query = """
            CREATE OR REPLACE TABLE silver_daily_stock_summary AS
            WITH daily_trades AS (
                SELECT 
                    CAST(timestamp AS DATE) AS trade_date,
                    symbol,
                    MIN(price) AS low_price,
                    MAX(price) AS high_price,
                    SUM(volume) AS total_volume,
                    SUM(price * volume) AS total_turnover_tl,
                    SUM(price * volume) / NULLIF(SUM(volume), 0) AS market_vwap,
                    COUNT(*) AS total_trades
                FROM bronze_raw_trades
                GROUP BY CAST(timestamp AS DATE), symbol
            ),
            first_last_prices AS (
                SELECT 
                    trade_date,
                    symbol,
                    ARG_MIN(price, timestamp) AS open_price,
                    ARG_MAX(price, timestamp) AS close_price
                FROM (
                    SELECT 
                        CAST(timestamp AS DATE) AS trade_date,
                        symbol,
                        price,
                        timestamp
                    FROM bronze_raw_trades
                )
                GROUP BY trade_date, symbol
            ),
            broker_stock_ranks AS (
                SELECT 
                    trade_date,
                    symbol,
                    broker_id,
                    buy_turnover_tl,
                    sell_turnover_tl,
                    total_turnover_tl,
                    net_flow_tl,
                    buy_vwap,
                    sell_vwap,
                    total_vwap,
                    is_primary_target,
                    broker_category,
                    ROW_NUMBER() OVER (PARTITION BY trade_date, symbol ORDER BY total_turnover_tl DESC) AS rank_in_stock
                FROM silver_daily_broker_summary
            ),
            stock_top_desks AS (
                SELECT 
                    trade_date,
                    symbol,
                    COUNT(DISTINCT broker_id) AS active_brokers_count,
                    ARG_MAX(broker_id, buy_turnover_tl) AS top_buyer_broker_id,
                    MAX(buy_turnover_tl) AS top_buyer_turnover_tl,
                    ARG_MAX(broker_id, sell_turnover_tl) AS top_seller_broker_id,
                    MAX(sell_turnover_tl) AS top_seller_turnover_tl,
                    SUM(CASE WHEN rank_in_stock <= 5 AND net_flow_tl > 0 THEN net_flow_tl ELSE 0.0 END) AS top_5_buyers_net_flow_tl,
                    SUM(CASE WHEN rank_in_stock <= 5 AND net_flow_tl < 0 THEN net_flow_tl ELSE 0.0 END) AS top_5_sellers_net_flow_tl,
                    SUM(CASE WHEN rank_in_stock <= 5 THEN total_turnover_tl ELSE 0.0 END) / 
                        NULLIF(SUM(total_turnover_tl), 0.0) AS top_5_concentration_ratio,
                    SUM(CASE WHEN broker_id IN ('IYM', 'YKR', 'AKM', 'GRM', 'ZRY') THEN net_flow_tl ELSE 0.0 END) AS top_5_domestic_net_flow_tl,
                    SUM(CASE WHEN broker_id = 'MLB' OR is_primary_target = TRUE THEN buy_turnover_tl ELSE 0.0 END) AS bofa_buy_turnover_tl,
                    SUM(CASE WHEN broker_id = 'MLB' OR is_primary_target = TRUE THEN sell_turnover_tl ELSE 0.0 END) AS bofa_sell_turnover_tl,
                    SUM(CASE WHEN broker_id = 'MLB' OR is_primary_target = TRUE THEN net_flow_tl ELSE 0.0 END) AS bofa_net_flow_tl,
                    SUM(CASE WHEN broker_id = 'MLB' OR is_primary_target = TRUE THEN total_turnover_tl ELSE 0.0 END) / 
                        NULLIF(SUM(total_turnover_tl), 0.0) AS bofa_stock_turnover_share,
                    MAX(CASE WHEN broker_id = 'MLB' OR is_primary_target = TRUE THEN buy_vwap ELSE NULL END) AS bofa_buy_vwap,
                    MAX(CASE WHEN broker_id = 'MLB' OR is_primary_target = TRUE THEN sell_vwap ELSE NULL END) AS bofa_sell_vwap,
                    MAX(CASE WHEN broker_id = 'MLB' OR is_primary_target = TRUE THEN total_vwap ELSE NULL END) AS bofa_total_vwap,
                    MIN(CASE WHEN broker_id = 'MLB' OR is_primary_target = TRUE THEN rank_in_stock ELSE NULL END) AS bofa_rank_in_stock
                FROM broker_stock_ranks
                GROUP BY trade_date, symbol
            )
            SELECT 
                d.trade_date,
                EXTRACT(DOW FROM d.trade_date) AS day_of_week,
                (EXTRACT(DOW FROM d.trade_date) = 1) AS is_monday,
                (EXTRACT(DOW FROM d.trade_date) = 5) AS is_friday,
                d.symbol,
                COALESCE(inst.name, d.symbol) AS symbol_name,
                COALESCE(inst.sector, 'Unknown') AS sector,
                COALESCE(inst.index_name, 'BIST100') AS index_name,
                f.open_price,
                d.high_price,
                d.low_price,
                f.close_price,
                d.market_vwap,
                (f.close_price - f.open_price) / NULLIF(f.open_price, 0.0) AS daily_return_pct,
                (d.high_price - d.low_price) / NULLIF(d.low_price, 0.0) AS price_range_pct,
                d.total_volume,
                d.total_turnover_tl,
                d.total_trades,
                COALESCE(std.active_brokers_count, 0) AS active_brokers_count,
                std.top_buyer_broker_id,
                std.top_buyer_turnover_tl,
                std.top_buyer_turnover_tl / NULLIF(d.total_turnover_tl, 0.0) AS top_buyer_share,
                std.top_seller_broker_id,
                std.top_seller_turnover_tl,
                std.top_seller_turnover_tl / NULLIF(d.total_turnover_tl, 0.0) AS top_seller_share,
                COALESCE(std.top_5_buyers_net_flow_tl, 0.0) AS top_5_buyers_net_flow_tl,
                COALESCE(std.top_5_sellers_net_flow_tl, 0.0) AS top_5_sellers_net_flow_tl,
                COALESCE(std.top_5_concentration_ratio, 0.0) AS top_5_concentration_ratio,
                COALESCE(std.top_5_domestic_net_flow_tl, 0.0) AS top_5_domestic_net_flow_tl,
                COALESCE(std.bofa_buy_turnover_tl, 0.0) AS bofa_buy_turnover_tl,
                COALESCE(std.bofa_sell_turnover_tl, 0.0) AS bofa_sell_turnover_tl,
                COALESCE(std.bofa_net_flow_tl, 0.0) AS bofa_net_flow_tl,
                COALESCE(std.bofa_stock_turnover_share, 0.0) AS bofa_stock_turnover_share,
                std.bofa_buy_vwap,
                std.bofa_sell_vwap,
                std.bofa_total_vwap,
                (std.bofa_buy_vwap - f.close_price) / NULLIF(f.close_price, 0.0) AS bofa_vwap_spread_pct,
                std.bofa_rank_in_stock,
                CURRENT_TIMESTAMP AS calculated_at
            FROM daily_trades d
            JOIN first_last_prices f ON d.trade_date = f.trade_date AND d.symbol = f.symbol
            LEFT JOIN bronze_instruments inst ON d.symbol = inst.symbol
            LEFT JOIN stock_top_desks std ON d.trade_date = std.trade_date AND d.symbol = std.symbol;
        """
        conn.execute(query)

        # Sync backward-compatible silver_market_daily
        conn.execute("""
            CREATE OR REPLACE TABLE silver_market_daily AS
            SELECT 
                trade_date,
                symbol,
                open_price,
                high_price,
                low_price,
                close_price,
                total_volume,
                total_turnover_tl,
                market_vwap,
                total_trades,
                active_brokers_count AS active_brokers,
                bofa_net_flow_tl,
                bofa_stock_turnover_share AS bofa_volume_share,
                calculated_at
            FROM silver_daily_stock_summary;
        """)

        rows = conn.execute("SELECT COUNT(*) FROM silver_daily_stock_summary;").fetchone()[0]
        logger.info(f"Successfully populated `silver_daily_stock_summary`: {rows:,} rows.")
        return {"table": "silver_daily_stock_summary", "rows": rows, "status": "success"}

    def transform_daily_sector_summary(self) -> dict[str, Any]:
        """Aggregate daily broker flows per sector (trade_date, sector, broker_id)."""
        conn = self.db.get_connection()
        logger.info("Computing `silver_daily_sector_summary`...")

        query = """
            CREATE OR REPLACE TABLE silver_daily_sector_summary AS
            WITH sector_totals AS (
                SELECT 
                    trade_date,
                    sector,
                    broker_id,
                    broker_name,
                    broker_category,
                    is_primary_target,
                    SUM(buy_volume) AS buy_volume,
                    SUM(buy_turnover_tl) AS buy_turnover_tl,
                    SUM(sell_volume) AS sell_volume,
                    SUM(sell_turnover_tl) AS sell_turnover_tl,
                    SUM(total_volume) AS total_volume,
                    SUM(total_turnover_tl) AS total_turnover_tl,
                    SUM(net_volume) AS net_volume,
                    SUM(net_flow_tl) AS net_flow_tl,
                    COUNT(DISTINCT symbol) AS active_symbols_count,
                    SUM(buy_trade_count + sell_trade_count) AS trade_count
                FROM silver_daily_broker_summary
                WHERE sector IS NOT NULL AND sector != ''
                GROUP BY trade_date, sector, broker_id, broker_name, broker_category, is_primary_target
            )
            SELECT 
                s.trade_date,
                s.sector,
                s.broker_id,
                s.broker_name,
                s.broker_category,
                s.is_primary_target,
                s.buy_volume,
                s.buy_turnover_tl,
                s.sell_volume,
                s.sell_turnover_tl,
                s.total_volume,
                s.total_turnover_tl,
                s.net_volume,
                s.net_flow_tl,
                s.active_symbols_count,
                s.trade_count,
                s.total_turnover_tl / NULLIF(SUM(s.total_turnover_tl) OVER (PARTITION BY s.trade_date, s.sector), 0.0) AS sector_turnover_share,
                CURRENT_TIMESTAMP AS calculated_at
            FROM sector_totals s;
        """
        conn.execute(query)

        rows = conn.execute("SELECT COUNT(*) FROM silver_daily_sector_summary;").fetchone()[0]
        logger.info(f"Successfully populated `silver_daily_sector_summary`: {rows:,} rows.")
        return {"table": "silver_daily_sector_summary", "rows": rows, "status": "success"}

    def transform_intraday_broker_windows(self) -> dict[str, Any]:
        """Aggregate parameterized intraday time windows (trade_date, symbol, broker_id, window_name)."""
        conn = self.db.get_connection()
        logger.info("Computing `silver_intraday_broker_window_summary` across 4 time windows...")

        _, case_order, _, _ = self._build_intraday_window_case_sql()
        windows = self.settings.get_intraday_windows()

        def metadata_case(field: str) -> str:
            branches = []
            for idx, window in enumerate(windows, start=1):
                value = str(window[field]).replace("'", "''")
                order = int(window.get("order", idx))
                branches.append(f"WHEN c.window_order = {order} THEN '{value}'")
            return "CASE " + " ".join(branches) + " END"

        window_name_case = metadata_case("name")
        window_start_case = metadata_case("start_time")
        window_end_case = metadata_case("end_time")

        query = f"""
            CREATE OR REPLACE TABLE silver_intraday_broker_window_summary AS
            WITH windowed_trades AS (
                SELECT 
                    CAST(timestamp AS DATE) AS trade_date,
                    symbol,
                    volume,
                    price * volume AS turnover_tl,
                    buyer_broker_id,
                    seller_broker_id,
                    {case_order} AS window_order
                FROM bronze_raw_trades
            ),
            broker_sides AS (
                SELECT 
                    trade_date,
                    symbol,
                    window_order,
                    UNNEST([buyer_broker_id, seller_broker_id]) AS broker_id,
                    UNNEST([volume, CAST(0 AS BIGINT)]) AS buy_volume,
                    UNNEST([turnover_tl, CAST(0 AS DOUBLE)]) AS buy_turnover_tl,
                    UNNEST([CAST(1 AS BIGINT), CAST(0 AS BIGINT)]) AS buy_trades,
                    UNNEST([CAST(0 AS BIGINT), volume]) AS sell_volume,
                    UNNEST([CAST(0 AS DOUBLE), turnover_tl]) AS sell_turnover_tl,
                    UNNEST([CAST(0 AS BIGINT), CAST(1 AS BIGINT)]) AS sell_trades
                FROM windowed_trades
                WHERE window_order IS NOT NULL
            ),
            combined AS (
                SELECT 
                    trade_date,
                    symbol,
                    broker_id,
                    window_order,
                    SUM(buy_volume) AS buy_volume,
                    SUM(buy_turnover_tl) AS buy_turnover_tl,
                    SUM(buy_turnover_tl) / NULLIF(SUM(buy_volume), 0.0) AS buy_vwap,
                    SUM(sell_volume) AS sell_volume,
                    SUM(sell_turnover_tl) AS sell_turnover_tl,
                    SUM(sell_turnover_tl) / NULLIF(SUM(sell_volume), 0.0) AS sell_vwap,
                    SUM(buy_volume) + SUM(sell_volume) AS total_volume,
                    SUM(buy_turnover_tl) + SUM(sell_turnover_tl) AS total_turnover_tl,
                    SUM(buy_volume) - SUM(sell_volume) AS net_volume,
                    SUM(buy_turnover_tl) - SUM(sell_turnover_tl) AS net_flow_tl,
                    SUM(buy_trades) + SUM(sell_trades) AS trade_count
                FROM broker_sides
                WHERE broker_id IS NOT NULL AND broker_id != ''
                GROUP BY trade_date, symbol, broker_id, window_order
            )
            SELECT 
                c.trade_date,
                c.symbol,
                COALESCE(inst.sector, 'Unknown') AS sector,
                c.broker_id,
                COALESCE(brk.broker_name, c.broker_id) AS broker_name,
                COALESCE(brk.is_primary_target, FALSE) AS is_primary_target,
                {window_name_case} AS window_name,
                c.window_order,
                {window_start_case} AS window_start_time,
                {window_end_case} AS window_end_time,
                c.buy_volume,
                c.buy_turnover_tl,
                c.buy_vwap,
                c.sell_volume,
                c.sell_turnover_tl,
                c.sell_vwap,
                c.total_volume,
                c.total_turnover_tl,
                c.net_volume,
                c.net_flow_tl,
                c.trade_count,
                CURRENT_TIMESTAMP AS calculated_at
            FROM combined c
            LEFT JOIN bronze_brokers brk ON c.broker_id = brk.broker_id
            LEFT JOIN bronze_instruments inst ON c.symbol = inst.symbol;
        """
        conn.execute(query)

        rows = conn.execute("SELECT COUNT(*) FROM silver_intraday_broker_window_summary;").fetchone()[0]
        logger.info(f"Successfully populated `silver_intraday_broker_window_summary`: {rows:,} rows.")
        return {"table": "silver_intraday_broker_window_summary", "rows": rows, "status": "success"}

    def transform_intraday_sector_windows(self) -> dict[str, Any]:
        """Aggregate intraday sector windows (trade_date, sector, broker_id, window_name)."""
        conn = self.db.get_connection()
        logger.info("Computing `silver_intraday_sector_window_summary`...")

        query = """
            CREATE OR REPLACE TABLE silver_intraday_sector_window_summary AS
            SELECT 
                trade_date,
                sector,
                broker_id,
                broker_name,
                is_primary_target,
                window_name,
                window_order,
                SUM(buy_volume) AS buy_volume,
                SUM(buy_turnover_tl) AS buy_turnover_tl,
                SUM(sell_volume) AS sell_volume,
                SUM(sell_turnover_tl) AS sell_turnover_tl,
                SUM(total_volume) AS total_volume,
                SUM(total_turnover_tl) AS total_turnover_tl,
                SUM(net_volume) AS net_volume,
                SUM(net_flow_tl) AS net_flow_tl,
                COUNT(DISTINCT symbol) AS active_symbols_count,
                SUM(trade_count) AS trade_count,
                CURRENT_TIMESTAMP AS calculated_at
            FROM silver_intraday_broker_window_summary
            WHERE sector IS NOT NULL AND sector != ''
            GROUP BY trade_date, sector, broker_id, broker_name, is_primary_target, window_name, window_order;
        """
        conn.execute(query)

        rows = conn.execute("SELECT COUNT(*) FROM silver_intraday_sector_window_summary;").fetchone()[0]
        logger.info(f"Successfully populated `silver_intraday_sector_window_summary`: {rows:,} rows.")
        return {"table": "silver_intraday_sector_window_summary", "rows": rows, "status": "success"}

    def run_all(self) -> dict[str, Any]:
        """Run full Silver transformation pipeline in dependency order."""
        initialize_silver_schema(self.db)
        res_broker = self.transform_daily_broker_summary()
        res_overview = self.transform_daily_broker_overview()
        res_stock = self.transform_daily_stock_summary()
        res_sector = self.transform_daily_sector_summary()
        res_win_broker = self.transform_intraday_broker_windows()
        res_win_sector = self.transform_intraday_sector_windows()

        return {
            "silver_daily_broker_summary": res_broker,
            "silver_daily_broker_overview": res_overview,
            "silver_daily_stock_summary": res_stock,
            "silver_daily_sector_summary": res_sector,
            "silver_intraday_broker_window_summary": res_win_broker,
            "silver_intraday_sector_window_summary": res_win_sector,
            "status": "success",
        }
