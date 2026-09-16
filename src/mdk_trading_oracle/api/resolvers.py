"""Database resolvers for GraphQL queries — 100% PostgreSQL powered."""

import logging
import math
import statistics
from typing import List, Optional

import yaml

from mdk_trading_oracle.api.types import (
    Broker,
    BrokerSummary,
    CandleWithBroker,
    DailyFifoRecord,
    EventStudyOccurrence,
    EventStudyResult,
    ForwardHorizonStat,
    ForwardReturnPoint,
    Instrument,
    TertipExecutiveSummary,
    TertipLot,
    TimeWindowAnalysisResult,
    TimeWindowAuctionDetail,
    TimeWindowCandle,
)
from mdk_trading_oracle.core.config import get_settings
from mdk_trading_oracle.data.postgres.candle_engine import MultiTimeframeCandleEngine
from mdk_trading_oracle.data.postgres.connection import PostgresConnectionManager

logger = logging.getLogger(__name__)


def clean_float(val: Optional[float], digits: int = 2) -> Optional[float]:
    """Ensure float is neither None, NaN, nor Inf, returning safely rounded float or None."""
    if val is None:
        return None
    try:
        f = float(val)
        if math.isnan(f) or math.isinf(f):
            return None
        return round(f, digits)
    except (ValueError, TypeError):
        return None


def get_instruments() -> List[Instrument]:
    """Return list of active instruments from PostgreSQL instruments table."""
    try:
        pm = PostgresConnectionManager()
        with pm.get_connection() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    "SELECT symbol, name, sector FROM instruments ORDER BY (symbol = 'XU030') DESC, symbol ASC;"
                )
                rows = cur.fetchall()
                if rows:
                    return [Instrument(symbol=r[0], name=r[1], sector=r[2]) for r in rows]
    except Exception as e:
        logger.warning(f"Failed to fetch instruments from PostgreSQL: {e}")

    settings = get_settings()
    inst_path = settings.config_dir / "instruments.yaml"
    if inst_path.exists():
        with open(inst_path, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f)
        return [
            Instrument(
                symbol=item["symbol"],
                name=item.get("name", item["symbol"]),
                sector=item.get("sector", "GENEL"),
            )
            for item in data.get("instruments", [])
        ]
    return [
        Instrument(symbol="XU030", name="BIST 30 Endeksi", sector="ENDEKS"),
        Instrument(symbol="THYAO", name="Türk Hava Yolları", sector="Transportation"),
        Instrument(symbol="AKBNK", name="Akbank", sector="Banking"),
        Instrument(symbol="ASELS", name="Aselsan", sector="Defense"),
    ]


def get_brokers() -> List[Broker]:
    """Return list of active brokerages from PostgreSQL brokers table."""
    try:
        pm = PostgresConnectionManager()
        with pm.get_connection() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    "SELECT broker_id, broker_name, category FROM brokers ORDER BY (broker_id = 'MLB') DESC, broker_id ASC;"
                )
                rows = cur.fetchall()
                if rows:
                    return [Broker(broker_id=r[0], broker_name=r[1], category=r[2]) for r in rows]
    except Exception as e:
        logger.warning(f"Failed to fetch brokers from PostgreSQL: {e}")

    settings = get_settings()
    brk_path = settings.config_dir / "brokers.yaml"
    if brk_path.exists():
        with open(brk_path, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f)
        brokers_raw = data.get("brokers", [])
        brokers_sorted = sorted(
            brokers_raw,
            key=lambda x: (
                0 if x.get("code") == "MLB" or x.get("broker_id") == "MLB" else 1,
                x.get("code", x.get("broker_id", "")),
            ),
        )
        return [
            Broker(
                broker_id=item.get("code", item.get("broker_id", "")),
                broker_name=item.get("name", item.get("broker_name", item.get("code", ""))),
                category=item.get("type", item.get("category", "INSTITUTIONAL")),
            )
            for item in brokers_sorted
            if item.get("code") or item.get("broker_id")
        ]
    return [
        Broker(broker_id="MLB", broker_name="Bank of America", category="Foreign Institutional"),
        Broker(broker_id="IYM", broker_name="İş Yatırım", category="Domestic Major Bank"),
        Broker(broker_id="YKR", broker_name="Yapı Kredi", category="Domestic Major Bank"),
    ]


def get_available_dates() -> List[str]:
    """Return distinct dates with available trade data from PostgreSQL."""
    pm = PostgresConnectionManager()
    try:
        with pm.get_connection() as conn:
            with conn.cursor() as cur:
                cur.execute("""
                    SELECT DISTINCT to_char(trade_date, 'YYYY-MM-DD') as d 
                    FROM broker_fifo_daily 
                    ORDER BY d DESC 
                    LIMIT 60;
                """)
                rows = cur.fetchall()
                dates = [r[0] for r in rows if r[0]]
                if dates:
                    return dates

                cur.execute("""
                    SELECT DISTINCT to_char(bucket_start, 'YYYY-MM-DD') as d 
                    FROM market_candles 
                    ORDER BY d DESC;
                """)
                rows = cur.fetchall()
                dates = [r[0] for r in rows if r[0]]
                if dates:
                    return dates
    except Exception as e:
        logger.warning(f"Failed to fetch dates from PostgreSQL: {e}")

    return ["2026-09-14", "2026-03-02"]


def get_candles(
    symbol: str,
    timeframe: str = "60m",
    date: Optional[str] = None,
    broker_id: str = "MLB",
) -> List[CandleWithBroker]:
    """Fetch candles and broker flows for a specific symbol, timeframe, date and broker from PostgreSQL."""
    pg_mgr = PostgresConnectionManager()
    target_date = date or "2026-09-14"

    # Verify if candles exist in PG; if not, dynamically generate them on-the-fly!
    conn = pg_mgr.get_connection()
    try:
        with conn.cursor() as cur:
            cur.execute(
                """
                SELECT count(*) FROM market_candles 
                WHERE timeframe = %s AND symbol = %s 
                  AND bucket_start >= %s AND bucket_start <= %s;
            """,
                (timeframe, symbol, f"{target_date} 00:00:00", f"{target_date} 23:59:59"),
            )
            count = cur.fetchone()[0]
    finally:
        conn.close()

    if count == 0:
        logger.info(f"Generating on-the-fly candles for {symbol} {target_date} [{timeframe}]...")
        try:
            if symbol == "XU030":
                from mdk_trading_oracle.data.bist30.index_engine import BIST30IndexEngine

                bist30_engine = BIST30IndexEngine()
                check_conn = pg_mgr.get_connection()
                try:
                    with check_conn.cursor() as cur:
                        cur.execute(
                            "SELECT count(*) FROM market_candles WHERE timeframe = %s AND bucket_start >= %s AND bucket_start <= %s;",
                            (timeframe, f"{target_date} 00:00:00", f"{target_date} 23:59:59"),
                        )
                        c_count = cur.fetchone()[0]
                finally:
                    check_conn.close()
                if c_count == 0:
                    engine = MultiTimeframeCandleEngine(pg_manager=pg_mgr)
                    engine.process_date_timeframe(target_date, timeframe)
                bist30_engine.generate_xu030_candles(target_date, timeframe)
            else:
                engine = MultiTimeframeCandleEngine(pg_manager=pg_mgr)
                engine.process_date_timeframe(target_date, timeframe)
        except Exception as e:
            logger.warning(f"Failed to dynamically generate candles for {symbol}: {e}")

    # Query PG for combined market candles + broker flows
    conn = pg_mgr.get_connection()
    results: List[CandleWithBroker] = []
    try:
        query = """
            SELECT 
                c.timeframe,
                to_char(c.bucket_start, 'YYYY-MM-DD HH24:MI:SS') as bucket_start,
                to_char(c.bucket_end, 'YYYY-MM-DD HH24:MI:SS') as bucket_end,
                c.symbol,
                c.open,
                c.high,
                c.low,
                c.close,
                c.volume,
                c.turnover_tl,
                c.vwap,
                c.trade_count,
                COALESCE(b.buy_volume, 0) as buy_volume,
                COALESCE(b.buy_turnover_tl, 0) as buy_turnover_tl,
                b.buy_vwap,
                COALESCE(b.sell_volume, 0) as sell_volume,
                COALESCE(b.sell_turnover_tl, 0) as sell_turnover_tl,
                b.sell_vwap,
                COALESCE(b.net_volume, 0) as net_volume,
                COALESCE(b.net_flow_tl, 0) as net_flow_tl,
                COALESCE(b.matched_volume, 0) as matched_volume,
                COALESCE(b.realized_pnl_tl, 0) as realized_pnl_tl,
                CASE 
                    WHEN c.turnover_tl > 0 THEN (COALESCE(b.buy_turnover_tl, 0) + COALESCE(b.sell_turnover_tl, 0)) / (2.0 * c.turnover_tl) * 100.0
                    ELSE 0 
                END as broker_share_pct
            FROM market_candles c
            LEFT JOIN candle_broker_flows b
                ON c.timeframe = b.timeframe 
                AND c.symbol = b.symbol 
                AND c.bucket_start = b.bucket_start 
                AND b.broker_id = %s
            WHERE c.symbol = %s 
              AND c.timeframe = %s 
              AND c.bucket_start >= %s 
              AND c.bucket_start <= %s
            ORDER BY c.bucket_start ASC;
        """
        with conn.cursor() as cur:
            cur.execute(
                query,
                (
                    broker_id,
                    symbol,
                    timeframe,
                    f"{target_date} 00:00:00",
                    f"{target_date} 23:59:59",
                ),
            )
            rows = cur.fetchall()
            for r in rows:
                results.append(
                    CandleWithBroker(
                        timeframe=r[0],
                        bucket_start=r[1],
                        bucket_end=r[2],
                        symbol=r[3],
                        open=float(r[4]),
                        high=float(r[5]),
                        low=float(r[6]),
                        close=float(r[7]),
                        volume=float(r[8]),
                        turnover_tl=float(r[9]),
                        vwap=float(r[10]),
                        trade_count=int(r[11]),
                        broker_id=broker_id,
                        buy_volume=float(r[12]),
                        buy_turnover_tl=float(r[13]),
                        buy_vwap=float(r[14]) if r[14] is not None else None,
                        sell_volume=float(r[15]),
                        sell_turnover_tl=float(r[16]),
                        sell_vwap=float(r[17]) if r[17] is not None else None,
                        net_volume=float(r[18]),
                        net_flow_tl=float(r[19]),
                        matched_volume=float(r[20]),
                        realized_pnl_tl=float(r[21]),
                        broker_share_pct=float(r[22]),
                    )
                )
    finally:
        conn.close()
    return results


def get_broker_summary(
    date: Optional[str] = None,
    broker_id: str = "MLB",
    symbol: Optional[str] = None,
) -> BrokerSummary:
    """Compute broker day summary for a specific stock or aggregate across all stocks via PostgreSQL."""
    target_date = date or "2026-09-14"
    is_macro = (not symbol) or (symbol in ("XU030", "ALL"))

    pm = PostgresConnectionManager()
    try:
        with pm.get_connection() as conn:
            with conn.cursor() as cur:
                if is_macro:
                    cur.execute(
                        """
                        SELECT 
                            broker_id,
                            to_char(trade_date, 'YYYY-MM-DD'),
                            sum(buy_volume) as total_buy_vol,
                            sum(buy_turnover_tl) as total_buy_tl,
                            sum(sell_volume) as total_sell_vol,
                            sum(sell_turnover_tl) as total_sell_tl,
                            sum(buy_turnover_tl) - sum(sell_turnover_tl) as net_flow,
                            sum(matched_volume) as matched_vol,
                            sum(intraday_realized_pnl_tl) as intra_pnl,
                            sum(carry_fifo_realized_pnl_tl) as carry_pnl,
                            sum(daily_realized_pnl_tl) as daily_pnl,
                            sum(buy_volume) - sum(sell_volume) as net_vol,
                            sum(open_stock_quantity) as open_stock_quantity
                        FROM broker_fifo_daily
                        WHERE broker_id = %s AND trade_date = %s
                        GROUP BY broker_id, trade_date;
                    """,
                        (broker_id, target_date),
                    )
                else:
                    cur.execute(
                        """
                        SELECT 
                            broker_id,
                            to_char(trade_date, 'YYYY-MM-DD'),
                            buy_volume as total_buy_vol,
                            buy_turnover_tl as total_buy_tl,
                            sell_volume as total_sell_vol,
                            sell_turnover_tl as total_sell_tl,
                            buy_turnover_tl - sell_turnover_tl as net_flow,
                            matched_volume as matched_vol,
                            intraday_realized_pnl_tl as intra_pnl,
                            carry_fifo_realized_pnl_tl as carry_pnl,
                            daily_realized_pnl_tl as daily_pnl,
                            buy_volume - sell_volume as net_vol,
                            open_stock_quantity
                        FROM broker_fifo_daily
                        WHERE broker_id = %s AND trade_date = %s AND symbol = %s;
                    """,
                        (broker_id, target_date, symbol),
                    )
                res = cur.fetchone()
                if res:
                    net_flow = float(res[6] or 0)
                    net_vol = float(res[11]) if res[11] is not None else 0.0
                    open_qty = float(res[12]) if res[12] is not None else None
                    if is_macro:
                        bias = (
                            "STRONG_BUY"
                            if net_flow > 500_000_000
                            else ("BUY" if net_flow > 0 else ("STRONG_SELL" if net_flow < -500_000_000 else "SELL"))
                        )
                    else:
                        bias = "BUY" if net_vol > 0 else ("SELL" if net_vol < 0 else "NEUTRAL")

                    return BrokerSummary(
                        broker_id=res[0],
                        trade_date=str(res[1]),
                        total_buy_volume=float(res[2] or 0),
                        total_buy_turnover_tl=float(res[3] or 0),
                        total_sell_volume=float(res[4] or 0),
                        total_sell_turnover_tl=float(res[5] or 0),
                        net_flow_tl=net_flow,
                        matched_volume=float(res[7] or 0),
                        intraday_pnl_tl=float(res[8] or 0),
                        carry_fifo_pnl_tl=float(res[9] or 0),
                        realized_pnl_tl=float(res[10] or 0),
                        position_bias=bias,
                        symbol=symbol or "XU030",
                        net_volume=net_vol,
                        open_stock_quantity=open_qty,
                    )
    except Exception as e:
        logger.warning(f"PostgreSQL FIFO summary query failed: {e}")

    # Fallback to candle_broker_flows
    pg_mgr = PostgresConnectionManager()
    try:
        conn = pg_mgr.get_connection()
        try:
            with conn.cursor() as cur:
                if is_macro:
                    cur.execute(
                        """
                        SELECT 
                            broker_id,
                            COALESCE(sum(buy_volume), 0),
                            COALESCE(sum(buy_turnover_tl), 0),
                            COALESCE(sum(sell_volume), 0),
                            COALESCE(sum(sell_turnover_tl), 0),
                            COALESCE(sum(net_flow_tl), 0),
                            COALESCE(sum(matched_volume), 0),
                            COALESCE(sum(realized_pnl_tl), 0),
                            COALESCE(sum(net_volume), 0)
                        FROM candle_broker_flows
                        WHERE timeframe = '8h' AND broker_id = %s 
                          AND bucket_start >= %s AND bucket_start <= %s
                        GROUP BY broker_id;
                    """,
                        (broker_id, f"{target_date} 00:00:00", f"{target_date} 23:59:59"),
                    )
                else:
                    cur.execute(
                        """
                        SELECT 
                            broker_id,
                            COALESCE(sum(buy_volume), 0),
                            COALESCE(sum(buy_turnover_tl), 0),
                            COALESCE(sum(sell_volume), 0),
                            COALESCE(sum(sell_turnover_tl), 0),
                            COALESCE(sum(net_flow_tl), 0),
                            COALESCE(sum(matched_volume), 0),
                            COALESCE(sum(realized_pnl_tl), 0),
                            COALESCE(sum(net_volume), 0)
                        FROM candle_broker_flows
                        WHERE timeframe = '8h' AND broker_id = %s AND symbol = %s
                          AND bucket_start >= %s AND bucket_start <= %s
                        GROUP BY broker_id;
                    """,
                        (broker_id, symbol, f"{target_date} 00:00:00", f"{target_date} 23:59:59"),
                    )
                row = cur.fetchone()
                if row:
                    net_flow = float(row[5])
                    net_vol = float(row[8]) if row[8] is not None else 0.0
                    if is_macro:
                        bias = (
                            "STRONG_BUY"
                            if net_flow > 500_000_000
                            else ("BUY" if net_flow > 0 else ("STRONG_SELL" if net_flow < -500_000_000 else "SELL"))
                        )
                    else:
                        bias = "BUY" if net_vol > 0 else ("SELL" if net_vol < 0 else "NEUTRAL")

                    return BrokerSummary(
                        broker_id=row[0],
                        trade_date=target_date,
                        total_buy_volume=float(row[1]),
                        total_buy_turnover_tl=float(row[2]),
                        total_sell_volume=float(row[3]),
                        total_sell_turnover_tl=float(row[4]),
                        net_flow_tl=net_flow,
                        matched_volume=float(row[6]),
                        intraday_pnl_tl=float(row[7]),
                        carry_fifo_pnl_tl=0.0,
                        realized_pnl_tl=float(row[7]),
                        position_bias=bias,
                        symbol=symbol or "XU030",
                        net_volume=net_vol,
                        open_stock_quantity=None,
                    )
        finally:
            conn.close()
    except Exception as e:
        logger.error(f"Error fetching broker summary from PostgreSQL: {e}")

    return BrokerSummary(
        broker_id=broker_id,
        trade_date=target_date,
        total_buy_volume=0.0,
        total_buy_turnover_tl=0.0,
        total_sell_volume=0.0,
        total_sell_turnover_tl=0.0,
        net_flow_tl=0.0,
        matched_volume=0.0,
        intraday_pnl_tl=0.0,
        carry_fifo_pnl_tl=0.0,
        realized_pnl_tl=0.0,
        position_bias="NEUTRAL",
        symbol=symbol or "XU030",
        net_volume=0.0,
        open_stock_quantity=None,
    )


def get_daily_fifo(
    broker_id: str = "MLB",
    symbol: Optional[str] = None,
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
    limit: int = 100,
) -> List[DailyFifoRecord]:
    """Return historical daily FIFO ledger records from PostgreSQL broker_fifo_daily."""
    pm = PostgresConnectionManager()
    results: List[DailyFifoRecord] = []
    with pm.get_connection() as conn:
        with conn.cursor() as cur:
            conditions = ["broker_id = %s"]
            params: list = [broker_id]
            is_xu030 = (symbol == "XU030")
            if is_xu030:
                conditions.append(
                    "symbol IN (SELECT symbol FROM instruments WHERE index_name = 'BIST30' AND symbol != 'XU030')"
                )
            elif symbol and symbol != "ALL":
                conditions.append("symbol = %s")
                params.append(symbol)
            if start_date:
                conditions.append("trade_date >= %s")
                params.append(start_date)
            if end_date:
                conditions.append("trade_date <= %s")
                params.append(end_date)

            where_clause = " WHERE " + " AND ".join(conditions)
            if is_xu030:
                query = f"""
                    SELECT 
                        to_char(trade_date, 'YYYY-MM-DD'), 'XU030' as symbol, 'BIST 30 Endeksi' as symbol_name, 'ENDEKS' as sector, broker_id,
                        sum(buy_volume) as buy_volume,
                        sum(buy_turnover_tl) as buy_turnover_tl,
                        CASE WHEN sum(buy_volume) > 0 THEN sum(buy_turnover_tl) / sum(buy_volume) ELSE NULL END as buy_vwap,
                        sum(sell_volume) as sell_volume,
                        sum(sell_turnover_tl) as sell_turnover_tl,
                        CASE WHEN sum(sell_volume) > 0 THEN sum(sell_turnover_tl) / sum(sell_volume) ELSE NULL END as sell_vwap,
                        sum(matched_volume) as matched_volume,
                        sum(intraday_realized_pnl_tl) as intraday_realized_pnl_tl,
                        sum(carry_fifo_realized_pnl_tl) as carry_fifo_realized_pnl_tl,
                        sum(daily_realized_pnl_tl) as daily_realized_pnl_tl,
                        CASE WHEN sum(market_value_tl) > 0 THEN 'LONG' WHEN sum(market_value_tl) < 0 THEN 'SHORT' ELSE 'FLAT' END as position_side,
                        sum(open_stock_quantity) as open_stock_quantity,
                        sum(open_fifo_cost_tl) as open_fifo_cost_tl,
                        NULL as fifo_avg_cost,
                        NULL as market_close_price,
                        sum(market_value_tl) as market_value_tl,
                        sum(unrealized_pnl_tl) as unrealized_pnl_tl,
                        sum(total_daily_pnl_tl) as total_daily_pnl_tl,
                        sum(cumulative_realized_pnl_tl) as cumulative_realized_pnl_tl
                    FROM broker_fifo_daily
                    {where_clause}
                    GROUP BY trade_date, broker_id
                    ORDER BY trade_date DESC
                    LIMIT {limit};
                """
            else:
                query = f"""
                    SELECT 
                        to_char(trade_date, 'YYYY-MM-DD'), symbol, symbol_name, sector, broker_id,
                        buy_volume, buy_turnover_tl, buy_vwap,
                        sell_volume, sell_turnover_tl, sell_vwap,
                        matched_volume, intraday_realized_pnl_tl,
                        carry_fifo_realized_pnl_tl, daily_realized_pnl_tl,
                        position_side, open_stock_quantity, open_fifo_cost_tl, fifo_avg_cost,
                        market_close_price, market_value_tl, unrealized_pnl_tl,
                        total_daily_pnl_tl, cumulative_realized_pnl_tl
                    FROM broker_fifo_daily
                    {where_clause}
                    ORDER BY trade_date DESC, symbol ASC
                    LIMIT {limit};
                """
            cur.execute(query, tuple(params))
            rows = cur.fetchall()
            for r in rows:
                results.append(
                    DailyFifoRecord(
                        trade_date=str(r[0]),
                        symbol=r[1],
                        symbol_name=r[2] or r[1],
                        sector=r[3] or "GENEL",
                        broker_id=r[4],
                        buy_volume=float(r[5] or 0),
                        buy_turnover_tl=float(r[6] or 0),
                        buy_vwap=float(r[7]) if r[7] is not None else None,
                        sell_volume=float(r[8] or 0),
                        sell_turnover_tl=float(r[9] or 0),
                        sell_vwap=float(r[10]) if r[10] is not None else None,
                        matched_volume=float(r[11] or 0),
                        intraday_realized_pnl_tl=float(r[12] or 0),
                        carry_fifo_realized_pnl_tl=float(r[13] or 0),
                        daily_realized_pnl_tl=float(r[14] or 0),
                        position_side=r[15] or "FLAT",
                        open_stock_quantity=float(r[16] or 0),
                        open_fifo_cost_tl=float(r[17] or 0),
                        fifo_avg_cost=float(r[18]) if r[18] is not None else None,
                        market_close_price=float(r[19]) if r[19] is not None else None,
                        market_value_tl=float(r[20] or 0),
                        unrealized_pnl_tl=float(r[21] or 0),
                        total_daily_pnl_tl=float(r[22] or 0),
                        cumulative_realized_pnl_tl=float(r[23] or 0),
                    )
                )
    return results


def get_tertip_lots(
    broker_id: str = "MLB",
    symbol: Optional[str] = None,
    status: Optional[str] = None,
    limit: int = 100,
) -> List[TertipLot]:
    """Return individual FIFO lot records from PostgreSQL broker_fifo_lot_lifecycle."""
    pm = PostgresConnectionManager()
    results: List[TertipLot] = []
    with pm.get_connection() as conn:
        with conn.cursor() as cur:
            conditions = ["broker_id = %s"]
            params: list = [broker_id]
            if symbol == "XU030":
                conditions.append(
                    "symbol IN (SELECT symbol FROM instruments WHERE index_name = 'BIST30' AND symbol != 'XU030')"
                )
            elif symbol and symbol != "ALL":
                conditions.append("symbol = %s")
                params.append(symbol)
            if status and status != "ALL":
                conditions.append("status = %s")
                params.append(status.upper())

            where_clause = " WHERE " + " AND ".join(conditions)
            query = f"""
                SELECT 
                    lot_id, broker_id, symbol, direction, to_char(open_date, 'YYYY-MM-DD'),
                    opened_quantity, opened_value_tl, opened_unit_cost,
                    status, to_char(closed_date, 'YYYY-MM-DD'), total_quantity_closed,
                    total_closing_value_tl, total_realized_pnl_tl,
                    remaining_quantity, remaining_value_tl
                FROM broker_fifo_lot_lifecycle
                {where_clause}
                ORDER BY open_date DESC
                LIMIT {limit};
            """
            cur.execute(query, tuple(params))
            rows = cur.fetchall()
            for r in rows:
                results.append(
                    TertipLot(
                        lot_id=r[0],
                        broker_id=r[1],
                        symbol=r[2],
                        direction=r[3],
                        open_date=str(r[4]),
                        opened_quantity=float(r[5] or 0),
                        opened_value_tl=float(r[6] or 0),
                        opened_unit_cost=float(r[7] or 0),
                        status=r[8],
                        closed_date=str(r[9]) if r[9] is not None else None,
                        total_quantity_closed=float(r[10] or 0),
                        total_closing_value_tl=float(r[11] or 0),
                        total_realized_pnl_tl=float(r[12] or 0),
                        remaining_quantity=float(r[13] or 0),
                        remaining_value_tl=float(r[14] or 0),
                    )
                )
    return results


def get_tertip_summary(
    broker_id: str = "MLB",
    date: Optional[str] = None,
) -> TertipExecutiveSummary:
    """Return executive summary metrics for tertip inventory and PnL from PostgreSQL."""
    target_date = date or "2026-09-14"
    pm = PostgresConnectionManager()
    with pm.get_connection() as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                SELECT 
                    broker_id,
                    to_char(trade_date, 'YYYY-MM-DD'),
                    sum(cumulative_realized_pnl_tl),
                    sum(daily_realized_pnl_tl),
                    sum(intraday_realized_pnl_tl),
                    sum(carry_fifo_realized_pnl_tl),
                    sum(open_fifo_cost_tl),
                    sum(market_value_tl),
                    sum(unrealized_pnl_tl),
                    sum(total_daily_pnl_tl)
                FROM broker_fifo_daily
                WHERE broker_id = %s AND trade_date = %s
                GROUP BY broker_id, trade_date;
            """,
                (broker_id, target_date),
            )
            row = cur.fetchone()

            cur.execute(
                """
                SELECT count(*) 
                FROM broker_fifo_lots 
                WHERE broker_id = %s;
            """,
                (broker_id,),
            )
            open_lots_count = cur.fetchone()[0]

            if row:
                return TertipExecutiveSummary(
                    broker_id=row[0],
                    trade_date=str(row[1]),
                    cumulative_realized_pnl_tl=float(row[2] or 0),
                    daily_realized_pnl_tl=float(row[3] or 0),
                    intraday_realized_pnl_tl=float(row[4] or 0),
                    carry_fifo_realized_pnl_tl=float(row[5] or 0),
                    open_inventory_cost_tl=float(row[6] or 0),
                    open_inventory_value_tl=float(row[7] or 0),
                    unrealized_pnl_tl=float(row[8] or 0),
                    total_daily_pnl_tl=float(row[9] or 0),
                    total_open_lots_count=int(open_lots_count),
                )

    return TertipExecutiveSummary(
        broker_id=broker_id,
        trade_date=target_date,
        cumulative_realized_pnl_tl=0.0,
        daily_realized_pnl_tl=0.0,
        intraday_realized_pnl_tl=0.0,
        carry_fifo_realized_pnl_tl=0.0,
        open_inventory_cost_tl=0.0,
        open_inventory_value_tl=0.0,
        unrealized_pnl_tl=0.0,
        total_daily_pnl_tl=0.0,
        total_open_lots_count=0,
    )


def get_event_study(
    symbol: str,
    condition_type: str = "DAILY_RETURN",
    min_value: Optional[float] = None,
    max_value: Optional[float] = None,
    direction: Optional[str] = None,
    forward_days: int = 5,
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
    limit: int = 100,
) -> EventStudyResult:
    """Analyze historical stock movements and compute forward subsequent returns via PostgreSQL."""
    pm = PostgresConnectionManager()
    clamped_forward_days = max(1, min(int(forward_days), 30))
    limit_val = max(1, min(int(limit), 500))

    lead_cols = []
    for k in range(1, clamped_forward_days + 1):
        lead_cols.append(f"LEAD(trade_date, {k}) OVER w as lead_date_{k}")
        lead_cols.append(f"LEAD(adj_close_price, {k}) OVER w as lead_price_{k}")
    leads_sql = ", ".join(lead_cols)

    cond_clauses = ["symbol = %s"]
    params: list = [symbol]

    cond_upper = condition_type.upper()
    dir_upper = (direction or "").upper()

    if cond_upper == "DAILY_RETURN":
        if dir_upper == "DOWN":
            val = min_value if min_value is not None else max_value
            if val is not None:
                cond_clauses.append("return_from_prev_close <= %s")
                params.append(-abs(float(val)) / 100.0)
            else:
                cond_clauses.append("return_from_prev_close <= %s")
                params.append(0.0)
        elif dir_upper == "UP":
            val = min_value if min_value is not None else max_value
            if val is not None:
                cond_clauses.append("return_from_prev_close >= %s")
                params.append(abs(float(val)) / 100.0)
            else:
                cond_clauses.append("return_from_prev_close >= %s")
                params.append(0.0)
        else:
            if min_value is not None and min_value < 0 and max_value is None:
                cond_clauses.append("return_from_prev_close <= %s")
                params.append(float(min_value) / 100.0)
            else:
                if min_value is not None:
                    cond_clauses.append("return_from_prev_close >= %s")
                    params.append(float(min_value) / 100.0)
                if max_value is not None:
                    cond_clauses.append("return_from_prev_close <= %s")
                    params.append(float(max_value) / 100.0)
    elif cond_upper == "BOFA_NET_FLOW":
        if dir_upper == "DOWN":
            val = min_value if min_value is not None else max_value
            if val is not None:
                cond_clauses.append("bofa_net_flow_tl <= %s")
                params.append(-abs(float(val)))
            else:
                cond_clauses.append("bofa_net_flow_tl <= %s")
                params.append(0.0)
        elif dir_upper == "UP":
            val = min_value if min_value is not None else max_value
            if val is not None:
                cond_clauses.append("bofa_net_flow_tl >= %s")
                params.append(abs(float(val)))
            else:
                cond_clauses.append("bofa_net_flow_tl >= %s")
                params.append(0.0)
        else:
            if min_value is not None and min_value < 0 and max_value is None:
                cond_clauses.append("bofa_net_flow_tl <= %s")
                params.append(float(min_value))
            else:
                if min_value is not None:
                    cond_clauses.append("bofa_net_flow_tl >= %s")
                    params.append(float(min_value))
                if max_value is not None:
                    cond_clauses.append("bofa_net_flow_tl <= %s")
                    params.append(float(max_value))
    elif cond_upper == "PRICE_RANGE":
        if min_value is not None:
            cond_clauses.append("price_range_pct >= %s")
            params.append(float(min_value) / 100.0)
        if max_value is not None:
            cond_clauses.append("price_range_pct <= %s")
            params.append(float(max_value) / 100.0)
    elif cond_upper == "VOLUME_SURGE":
        if min_value is not None:
            cond_clauses.append("total_turnover_tl >= %s")
            params.append(float(min_value))
        if max_value is not None:
            cond_clauses.append("total_turnover_tl <= %s")
            params.append(float(max_value))

    if start_date:
        cond_clauses.append("trade_date >= %s")
        params.append(start_date)
    if end_date:
        cond_clauses.append("trade_date <= %s")
        params.append(end_date)

    where_sql = " AND ".join(cond_clauses)

    query = f"""
        WITH base AS (
            SELECT 
                to_char(trade_date, 'YYYY-MM-DD') as trade_date,
                symbol,
                adj_open_price,
                adj_close_price,
                adj_daily_return_pct,
                bofa_net_flow_tl,
                price_range_pct,
                total_turnover_tl,
                LAG(adj_close_price, 1) OVER w as prev_close_price,
                CASE 
                    WHEN LAG(adj_close_price, 1) OVER w > 0 
                    THEN (adj_close_price - LAG(adj_close_price, 1) OVER w) / LAG(adj_close_price, 1) OVER w 
                    ELSE adj_daily_return_pct 
                END as return_from_prev_close,
                {leads_sql}
            FROM daily_stock_summary
            WHERE symbol = %s
            WINDOW w AS (ORDER BY trade_date ASC)
        )
        SELECT * FROM base
        WHERE {where_sql}
        ORDER BY trade_date DESC
        LIMIT {limit_val};
    """

    count_query = f"""
        WITH base AS (
            SELECT 
                to_char(trade_date, 'YYYY-MM-DD') as trade_date,
                symbol,
                adj_close_price,
                adj_daily_return_pct,
                bofa_net_flow_tl,
                price_range_pct,
                total_turnover_tl,
                CASE 
                    WHEN LAG(adj_close_price, 1) OVER w > 0 
                    THEN (adj_close_price - LAG(adj_close_price, 1) OVER w) / LAG(adj_close_price, 1) OVER w 
                    ELSE adj_daily_return_pct 
                END as return_from_prev_close
            FROM daily_stock_summary
            WHERE symbol = %s
            WINDOW w AS (ORDER BY trade_date ASC)
        )
        SELECT count(*) FROM base WHERE {where_sql};
    """

    with pm.get_connection() as conn:
        with conn.cursor() as cur:
            full_params = [symbol] + params
            cur.execute(query, tuple(full_params))
            cols = [d[0] for d in cur.description]
            df_rows = [dict(zip(cols, r)) for r in cur.fetchall()]

            cur.execute(count_query, tuple(full_params))
            total_occurrences = cur.fetchone()[0]

    occurrences: list[EventStudyOccurrence] = []
    horizon_daily_returns: dict[int, list[float]] = {k: [] for k in range(1, clamped_forward_days + 1)}
    horizon_cumul_returns: dict[int, list[float]] = {k: [] for k in range(1, clamped_forward_days + 1)}

    for row in df_rows:
        p0 = clean_float(row["adj_close_price"]) or 0.0
        p_prev = clean_float(row["prev_close_price"])
        p_open = clean_float(row["adj_open_price"])

        p_change_tl = clean_float(p0 - p_prev, 2) if (p_prev is not None and p0 is not None) else None
        p_change_pct = (
            clean_float(((p0 - p_prev) / p_prev) * 100.0, 2)
            if (p_prev is not None and p_prev > 0)
            else clean_float(float(row["adj_daily_return_pct"] or 0) * 100.0, 2)
        )

        if cond_upper == "DAILY_RETURN":
            movement_val = p_change_pct or 0.0
        elif cond_upper == "BOFA_NET_FLOW":
            movement_val = clean_float(row["bofa_net_flow_tl"], 2) or 0.0
        elif cond_upper == "PRICE_RANGE":
            movement_val = clean_float(float(row["price_range_pct"] or 0) * 100.0, 2) or 0.0
        elif cond_upper == "VOLUME_SURGE":
            movement_val = clean_float(row["total_turnover_tl"], 2) or 0.0
        else:
            movement_val = p_change_pct or 0.0

        fwd_points: list[ForwardReturnPoint] = []
        for k in range(1, clamped_forward_days + 1):
            pk = clean_float(row[f"lead_price_{k}"])
            dk = row[f"lead_date_{k}"]

            p_prev_day = p0 if k == 1 else clean_float(row[f"lead_price_{k - 1}"])

            daily_ret: Optional[float] = None
            cumul_ret: Optional[float] = None

            is_pk_valid = pk is not None
            is_prev_valid = p_prev_day is not None and p_prev_day > 0

            if is_pk_valid and is_prev_valid:
                daily_ret = clean_float(((pk - p_prev_day) / p_prev_day) * 100.0, 2)
                if daily_ret is not None:
                    horizon_daily_returns[k].append(daily_ret)

            if is_pk_valid and p0 > 0:
                cumul_ret = clean_float(((pk - p0) / p0) * 100.0, 2)
                if cumul_ret is not None:
                    horizon_cumul_returns[k].append(cumul_ret)

            fwd_points.append(
                ForwardReturnPoint(
                    day_offset=k,
                    date=str(dk)[:10] if dk is not None and dk == dk else None,
                    prev_close_price=p_prev_day if is_prev_valid else None,
                    close_price=pk if is_pk_valid else None,
                    return_pct=daily_ret,
                    daily_return_pct=daily_ret,
                    cumulative_return_pct=cumul_ret,
                )
            )

        occurrences.append(
            EventStudyOccurrence(
                event_date=str(row["trade_date"])[:10],
                prev_close_price=p_prev,
                open_price=p_open,
                close_price=p0,
                price_change_tl=p_change_tl,
                price_change_pct=p_change_pct,
                movement_value=movement_val,
                bofa_net_flow_tl=clean_float(row["bofa_net_flow_tl"], 2) or 0.0,
                total_turnover_tl=clean_float(row["total_turnover_tl"], 2) or 0.0,
                forward_returns=fwd_points,
            )
        )

    horizon_stats: list[ForwardHorizonStat] = []
    for k in range(1, clamped_forward_days + 1):
        d_rets = horizon_daily_returns[k]
        c_rets = horizon_cumul_returns[k]

        if d_rets:
            avg_d = round(sum(d_rets) / len(d_rets), 2)
            win_d = round((sum(1 for r in d_rets if r > 0) / len(d_rets)) * 100.0, 1)
            med_d = round(statistics.median(d_rets), 2)
            max_g = round(max(d_rets), 2)
            max_l = round(min(d_rets), 2)
        else:
            avg_d = 0.0
            win_d = 0.0
            med_d = 0.0
            max_g = 0.0
            max_l = 0.0

        if c_rets:
            avg_c = round(sum(c_rets) / len(c_rets), 2)
            win_c = round((sum(1 for r in c_rets if r > 0) / len(c_rets)) * 100.0, 1)
        else:
            avg_c = 0.0
            win_c = 0.0

        horizon_stats.append(
            ForwardHorizonStat(
                day_offset=k,
                avg_return_pct=avg_d,
                win_rate_pct=win_d,
                median_return_pct=med_d,
                max_gain_pct=max_g,
                max_loss_pct=max_l,
                cumul_avg_return_pct=avg_c,
                cumul_win_rate_pct=win_c,
                sample_count=len(d_rets),
            )
        )

    return EventStudyResult(
        symbol=symbol,
        condition_type=cond_upper,
        min_value=min_value,
        max_value=max_value,
        direction=dir_upper or None,
        forward_days=clamped_forward_days,
        total_occurrences=int(total_occurrences),
        horizon_stats=horizon_stats,
        occurrences=occurrences,
    )


def get_time_window_analysis(
    symbol: str = "AKBNK",
    broker_id: str = "MLB",
    start_datetime: str = "2026-09-14 09:55:00",
    end_datetime: str = "2026-09-14 18:08:00",
    timeframe: str = "5m",
) -> TimeWindowAnalysisResult:
    """Analyze a custom date and time window with opening/closing auctions and broker flows."""
    pm = PostgresConnectionManager()
    is_xu030 = (symbol == "XU030")

    # 1. Resolve Instrument Name and Broker Name
    symbol_name = symbol
    broker_name = broker_id
    with pm.get_connection() as conn:
        with conn.cursor() as cur:
            cur.execute("SELECT name FROM instruments WHERE symbol = %s;", (symbol,))
            r = cur.fetchone()
            if r and r[0]:
                symbol_name = r[0]
            elif is_xu030:
                symbol_name = "BIST 30 Endeksi"

            cur.execute("SELECT broker_name FROM brokers WHERE broker_id = %s;", (broker_id,))
            r = cur.fetchone()
            if r and r[0]:
                broker_name = r[0]

    # Normalize datetime strings (e.g. '2026-09-14T09:55' -> '2026-09-14 09:55:00')
    clean_start = start_datetime.replace("T", " ").strip()
    if len(clean_start) == 16:
        clean_start += ":00"
    clean_end = end_datetime.replace("T", " ").strip()
    if len(clean_end) == 16:
        clean_end += ":00"

    start_date = clean_start[:10]
    end_date = clean_end[:10]

    # 2. Opening Auction (09:55 - 09:56:30 on start_date if window starts around or before 10:05)
    opening_auction: Optional[TimeWindowAuctionDetail] = None
    start_time_part = clean_start[11:19]
    if start_time_part <= "10:05:00":
        with pm.get_connection() as conn:
            with conn.cursor() as cur:
                if is_xu030:
                    cur.execute(
                        """
                        SELECT 
                            to_char(min(timestamp), 'YYYY-MM-DD HH24:MI:SS'),
                            sum(volume),
                            sum(price * volume),
                            sum(CASE WHEN buyer_broker_id = %s THEN volume ELSE 0 END),
                            sum(CASE WHEN seller_broker_id = %s THEN volume ELSE 0 END),
                            sum(CASE WHEN buyer_broker_id = %s THEN price * volume ELSE 0 END) -
                            sum(CASE WHEN seller_broker_id = %s THEN price * volume ELSE 0 END)
                        FROM raw_trades
                        WHERE symbol IN (SELECT symbol FROM instruments WHERE index_name = 'BIST30')
                          AND timestamp >= %s AND timestamp <= %s;
                    """,
                        (
                            broker_id,
                            broker_id,
                            broker_id,
                            broker_id,
                            f"{start_date} 09:55:00",
                            f"{start_date} 09:56:30",
                        ),
                    )
                    r = cur.fetchone()
                    if r and r[1] and r[1] > 0:
                        tot_vol = float(r[1] or 0)
                        tot_to = float(r[2] or 0)
                        b_buy = float(r[3] or 0)
                        b_sell = float(r[4] or 0)
                        b_net = float(r[5] or 0)
                        b_net_vol = b_buy - b_sell
                        share = ((b_buy + b_sell) / (2.0 * tot_vol) * 100.0) if tot_vol > 0 else 0.0
                        opening_auction = TimeWindowAuctionDetail(
                            auction_type="OPENING_AUCTION",
                            match_time=str(r[0] or f"{start_date} 09:55:25"),
                            match_price=tot_to / tot_vol if tot_vol > 0 else 0.0,
                            total_volume=tot_vol,
                            total_turnover_tl=tot_to,
                            broker_buy_volume=b_buy,
                            broker_sell_volume=b_sell,
                            broker_net_volume=b_net_vol,
                            broker_net_flow_tl=b_net,
                            broker_share_pct=round(share, 2),
                        )
                else:
                    cur.execute(
                        """
                        SELECT 
                            to_char(min(timestamp), 'YYYY-MM-DD HH24:MI:SS'),
                            min(price),
                            sum(volume),
                            sum(price * volume),
                            sum(CASE WHEN buyer_broker_id = %s THEN volume ELSE 0 END),
                            sum(CASE WHEN seller_broker_id = %s THEN volume ELSE 0 END),
                            sum(CASE WHEN buyer_broker_id = %s THEN price * volume ELSE 0 END) -
                            sum(CASE WHEN seller_broker_id = %s THEN price * volume ELSE 0 END)
                        FROM raw_trades
                        WHERE symbol = %s AND timestamp >= %s AND timestamp <= %s;
                    """,
                        (
                            broker_id,
                            broker_id,
                            broker_id,
                            broker_id,
                            symbol,
                            f"{start_date} 09:55:00",
                            f"{start_date} 09:56:30",
                        ),
                    )
                    r = cur.fetchone()
                    if r and r[2] and r[2] > 0:
                        tot_vol = float(r[2] or 0)
                        tot_to = float(r[3] or 0)
                        b_buy = float(r[4] or 0)
                        b_sell = float(r[5] or 0)
                        b_net = float(r[6] or 0)
                        b_net_vol = b_buy - b_sell
                        share = ((b_buy + b_sell) / (2.0 * tot_vol) * 100.0) if tot_vol > 0 else 0.0
                        opening_auction = TimeWindowAuctionDetail(
                            auction_type="OPENING_AUCTION",
                            match_time=str(r[0] or f"{start_date} 09:55:25"),
                            match_price=float(r[1] or 0.0),
                            total_volume=tot_vol,
                            total_turnover_tl=tot_to,
                            broker_buy_volume=b_buy,
                            broker_sell_volume=b_sell,
                            broker_net_volume=b_net_vol,
                            broker_net_flow_tl=b_net,
                            broker_share_pct=round(share, 2),
                        )

    # 3. Closing Auction (18:05 - 18:10 on end_date if window ends at or after 18:00)
    closing_auction: Optional[TimeWindowAuctionDetail] = None
    end_time_part = clean_end[11:19]
    if end_time_part >= "18:00:00":
        with pm.get_connection() as conn:
            with conn.cursor() as cur:
                if is_xu030:
                    cur.execute(
                        """
                        SELECT 
                            to_char(min(timestamp), 'YYYY-MM-DD HH24:MI:SS'),
                            sum(volume),
                            sum(price * volume),
                            sum(CASE WHEN buyer_broker_id = %s THEN volume ELSE 0 END),
                            sum(CASE WHEN seller_broker_id = %s THEN volume ELSE 0 END),
                            sum(CASE WHEN buyer_broker_id = %s THEN price * volume ELSE 0 END) -
                            sum(CASE WHEN seller_broker_id = %s THEN price * volume ELSE 0 END)
                        FROM raw_trades
                        WHERE symbol IN (SELECT symbol FROM instruments WHERE index_name = 'BIST30')
                          AND timestamp >= %s AND timestamp <= %s;
                    """,
                        (
                            broker_id,
                            broker_id,
                            broker_id,
                            broker_id,
                            f"{end_date} 18:05:00",
                            f"{end_date} 18:10:00",
                        ),
                    )
                    r = cur.fetchone()
                    if r and r[1] and r[1] > 0:
                        tot_vol = float(r[1] or 0)
                        tot_to = float(r[2] or 0)
                        b_buy = float(r[3] or 0)
                        b_sell = float(r[4] or 0)
                        b_net = float(r[5] or 0)
                        b_net_vol = b_buy - b_sell
                        share = ((b_buy + b_sell) / (2.0 * tot_vol) * 100.0) if tot_vol > 0 else 0.0
                        closing_auction = TimeWindowAuctionDetail(
                            auction_type="CLOSING_AUCTION",
                            match_time=str(r[0] or f"{end_date} 18:05:15"),
                            match_price=tot_to / tot_vol if tot_vol > 0 else 0.0,
                            total_volume=tot_vol,
                            total_turnover_tl=tot_to,
                            broker_buy_volume=b_buy,
                            broker_sell_volume=b_sell,
                            broker_net_volume=b_net_vol,
                            broker_net_flow_tl=b_net,
                            broker_share_pct=round(share, 2),
                        )
                else:
                    cur.execute(
                        """
                        SELECT 
                            to_char(min(timestamp), 'YYYY-MM-DD HH24:MI:SS'),
                            min(price),
                            sum(volume),
                            sum(price * volume),
                            sum(CASE WHEN buyer_broker_id = %s THEN volume ELSE 0 END),
                            sum(CASE WHEN seller_broker_id = %s THEN volume ELSE 0 END),
                            sum(CASE WHEN buyer_broker_id = %s THEN price * volume ELSE 0 END) -
                            sum(CASE WHEN seller_broker_id = %s THEN price * volume ELSE 0 END)
                        FROM raw_trades
                        WHERE symbol = %s AND timestamp >= %s AND timestamp <= %s;
                    """,
                        (
                            broker_id,
                            broker_id,
                            broker_id,
                            broker_id,
                            symbol,
                            f"{end_date} 18:05:00",
                            f"{end_date} 18:10:00",
                        ),
                    )
                    r = cur.fetchone()
                    if r and r[2] and r[2] > 0:
                        tot_vol = float(r[2] or 0)
                        tot_to = float(r[3] or 0)
                        b_buy = float(r[4] or 0)
                        b_sell = float(r[5] or 0)
                        b_net = float(r[6] or 0)
                        b_net_vol = b_buy - b_sell
                        share = ((b_buy + b_sell) / (2.0 * tot_vol) * 100.0) if tot_vol > 0 else 0.0
                        closing_auction = TimeWindowAuctionDetail(
                            auction_type="CLOSING_AUCTION",
                            match_time=str(r[0] or f"{end_date} 18:05:15"),
                            match_price=float(r[1] or 0.0),
                            total_volume=tot_vol,
                            total_turnover_tl=tot_to,
                            broker_buy_volume=b_buy,
                            broker_sell_volume=b_sell,
                            broker_net_volume=b_net_vol,
                            broker_net_flow_tl=b_net,
                            broker_share_pct=round(share, 2),
                        )

    # 4. Candlesticks and Broker Flows inside the Window
    interval_map = {
        "1m": "1 minute",
        "5m": "5 minutes",
        "15m": "15 minutes",
        "30m": "30 minutes",
        "60m": "60 minutes",
        "120m": "120 minutes",
        "240m": "240 minutes",
        "8h": "8 hours",
    }
    tf_interval = interval_map.get(timeframe, "5 minutes")

    candles: list[TimeWindowCandle] = []

    # First attempt: market_candles + candle_broker_flows
    with pm.get_connection() as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                SELECT 
                    to_char(mc.bucket_start, 'YYYY-MM-DD HH24:MI:SS'),
                    to_char(mc.bucket_end, 'YYYY-MM-DD HH24:MI:SS'),
                    mc.open,
                    mc.high,
                    mc.low,
                    mc.close,
                    mc.volume,
                    mc.turnover_tl,
                    mc.vwap,
                    mc.trade_count,
                    COALESCE(cbf.buy_volume, 0),
                    COALESCE(cbf.sell_volume, 0),
                    COALESCE(cbf.net_volume, 0),
                    COALESCE(cbf.net_flow_tl, 0),
                    COALESCE(cbf.realized_pnl_tl, 0),
                    CASE WHEN mc.turnover_tl > 0 THEN (COALESCE(cbf.buy_turnover_tl, 0) + COALESCE(cbf.sell_turnover_tl, 0)) / (2.0 * mc.turnover_tl) * 100.0 ELSE 0 END
                FROM market_candles mc
                LEFT JOIN candle_broker_flows cbf 
                    ON mc.timeframe = cbf.timeframe 
                   AND mc.symbol = cbf.symbol 
                   AND mc.bucket_start = cbf.bucket_start 
                   AND cbf.broker_id = %s
                WHERE mc.symbol = %s 
                  AND mc.timeframe = %s
                  AND mc.bucket_start >= %s 
                  AND mc.bucket_start <= %s
                ORDER BY mc.bucket_start ASC;
            """,
                (broker_id, symbol, timeframe, clean_start, clean_end),
            )
            rows = cur.fetchall()
            for r in rows:
                candles.append(
                    TimeWindowCandle(
                        bucket_start=r[0],
                        bucket_end=r[1],
                        open=float(r[2]),
                        high=float(r[3]),
                        low=float(r[4]),
                        close=float(r[5]),
                        volume=float(r[6]),
                        turnover_tl=float(r[7]),
                        vwap=float(r[8]),
                        trades_count=int(r[9]),
                        broker_buy_volume=float(r[10]),
                        broker_sell_volume=float(r[11]),
                        broker_net_volume=float(r[12]),
                        broker_net_flow_tl=float(r[13]),
                        broker_realized_pnl_tl=float(r[14]),
                        broker_share_pct=round(float(r[15]), 2),
                    )
                )

    # Fallback to raw_trades with date_bin if precomputed market_candles has 0 rows
    if not candles:
        with pm.get_connection() as conn:
            with conn.cursor() as cur:
                sym_filter = (
                    "symbol IN (SELECT symbol FROM instruments WHERE index_name = 'BIST30')"
                    if is_xu030
                    else "symbol = %s"
                )
                sym_param = [] if is_xu030 else [symbol]
                raw_sql = f"""
                    SELECT 
                        to_char(date_bin('{tf_interval}'::interval, timestamp, TIMESTAMP '2026-01-01 00:00:00'), 'YYYY-MM-DD HH24:MI:SS') as b_start,
                        min(price) as low,
                        max(price) as high,
                        (array_agg(price ORDER BY timestamp ASC))[1] as open,
                        (array_agg(price ORDER BY timestamp DESC))[1] as close,
                        sum(volume) as vol,
                        sum(price * volume) as turnover,
                        count(*) as trades_count,
                        sum(CASE WHEN buyer_broker_id = %s THEN volume ELSE 0 END) as b_buy_vol,
                        sum(CASE WHEN seller_broker_id = %s THEN volume ELSE 0 END) as b_sell_vol,
                        sum(CASE WHEN buyer_broker_id = %s THEN price * volume ELSE 0 END) -
                        sum(CASE WHEN seller_broker_id = %s THEN price * volume ELSE 0 END) as b_net_flow
                    FROM raw_trades
                    WHERE {sym_filter}
                      AND timestamp >= %s AND timestamp <= %s
                    GROUP BY b_start
                    ORDER BY b_start ASC;
                """
                params = [broker_id, broker_id, broker_id, broker_id] + sym_param + [clean_start, clean_end]
                cur.execute(raw_sql, tuple(params))
                raw_rows = cur.fetchall()
                for r in raw_rows:
                    v = float(r[5] or 0)
                    to = float(r[6] or 0)
                    bb = float(r[8] or 0)
                    bs = float(r[9] or 0)
                    sh = ((bb + bs) / (2.0 * v) * 100.0) if v > 0 else 0.0
                    candles.append(
                        TimeWindowCandle(
                            bucket_start=r[0],
                            bucket_end=r[0],
                            open=float(r[3] or 0),
                            high=float(r[2] or 0),
                            low=float(r[1] or 0),
                            close=float(r[4] or 0),
                            volume=v,
                            turnover_tl=to,
                            vwap=to / v if v > 0 else float(r[4] or 0),
                            trades_count=int(r[7] or 0),
                            broker_buy_volume=bb,
                            broker_sell_volume=bs,
                            broker_net_volume=bb - bs,
                            broker_net_flow_tl=float(r[10] or 0),
                            broker_realized_pnl_tl=0.0,
                            broker_share_pct=round(sh, 2),
                        )
                    )

    # 5. Aggregate Summary Statistics for the Window
    w_open = candles[0].open if candles else None
    w_close = candles[-1].close if candles else None
    price_change_tl = round(w_close - w_open, 2) if (w_close is not None and w_open is not None) else None
    price_change_pct = (
        round(((w_close - w_open) / w_open) * 100.0, 2)
        if (w_close is not None and w_open is not None and w_open > 0)
        else None
    )

    w_high = max((c.high for c in candles), default=None) if candles else None
    w_low = min((c.low for c in candles), default=None) if candles else None
    price_range_pct = (
        round(((w_high - w_low) / w_low) * 100.0, 2)
        if (w_high is not None and w_low is not None and w_low > 0)
        else None
    )

    tot_vol = sum(c.volume for c in candles)
    tot_to = sum(c.turnover_tl for c in candles)
    tot_trades = sum(c.trades_count for c in candles)

    brk_buy_vol = sum(c.broker_buy_volume for c in candles)
    brk_sell_vol = sum(c.broker_sell_volume for c in candles)
    brk_net_vol = brk_buy_vol - brk_sell_vol
    brk_net_flow = sum(c.broker_net_flow_tl for c in candles)
    brk_realized_pnl = sum(c.broker_realized_pnl_tl for c in candles)
    brk_share = round(((brk_buy_vol + brk_sell_vol) / (2.0 * tot_vol) * 100.0), 2) if tot_vol > 0 else 0.0

    return TimeWindowAnalysisResult(
        symbol=symbol,
        symbol_name=symbol_name,
        broker_id=broker_id,
        broker_name=broker_name,
        start_datetime=clean_start,
        end_datetime=clean_end,
        timeframe=timeframe,
        window_open_price=w_open,
        window_close_price=w_close,
        price_change_tl=price_change_tl,
        price_change_pct=price_change_pct,
        window_high_price=w_high,
        window_low_price=w_low,
        price_range_pct=price_range_pct,
        total_volume=tot_vol,
        total_turnover_tl=tot_to,
        total_trades_count=tot_trades,
        broker_buy_volume=brk_buy_vol,
        broker_buy_turnover_tl=0.0,
        broker_buy_vwap=None,
        broker_sell_volume=brk_sell_vol,
        broker_sell_turnover_tl=0.0,
        broker_sell_vwap=None,
        broker_net_volume=brk_net_vol,
        broker_net_flow_tl=brk_net_flow,
        broker_realized_pnl_tl=brk_realized_pnl,
        broker_market_share_pct=brk_share,
        opening_auction=opening_auction,
        closing_auction=closing_auction,
        candles=candles,
    )
