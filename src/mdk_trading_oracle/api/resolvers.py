"""Database resolvers for GraphQL queries."""

import logging
import statistics
from typing import List, Optional

import duckdb
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
)
from mdk_trading_oracle.core.config import get_settings
from mdk_trading_oracle.data.postgres.candle_engine import MultiTimeframeCandleEngine
from mdk_trading_oracle.data.postgres.connection import PostgresConnectionManager

logger = logging.getLogger(__name__)


def get_instruments() -> List[Instrument]:
    """Return list of active instruments from config/instruments.yaml."""
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
        Instrument(symbol="THYAO", name="Türk Hava Yolları", sector="Transportation"),
        Instrument(symbol="AKBNK", name="Akbank", sector="Banking"),
        Instrument(symbol="ASELS", name="Aselsan", sector="Defense"),
    ]


def get_brokers() -> List[Broker]:
    """Return list of active brokerages from config/brokers.yaml."""
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
    """Return distinct dates with available trade data."""
    settings = get_settings()
    duckdb_path = str(settings.duckdb_path)
    try:
        conn = duckdb.connect(duckdb_path, read_only=True)
        try:
            rows = conn.execute("""
                SELECT DISTINCT strftime(trade_date, '%Y-%m-%d') as d 
                FROM silver_broker_fifo_daily 
                ORDER BY d DESC 
                LIMIT 60;
            """).fetchall()
            if rows:
                return [r[0] for r in rows if r[0]]
        finally:
            conn.close()
    except Exception as e:
        logger.warning(f"Failed to fetch dates from DuckDB: {e}")

    # Fallback to PostgreSQL
    pg_mgr = PostgresConnectionManager()
    try:
        conn = pg_mgr.get_connection()
        try:
            with conn.cursor() as cur:
                cur.execute("""
                    SELECT DISTINCT to_char(bucket_start, 'YYYY-MM-DD') as d 
                    FROM market_candles 
                    ORDER BY d DESC;
                """)
                rows = cur.fetchall()
                dates = [r[0] for r in rows if r[0]]
                if dates:
                    return dates
        finally:
            conn.close()
    except Exception:
        pass

    return ["2026-09-14", "2026-03-02"]


def get_candles(
    symbol: str,
    timeframe: str = "60m",
    date: Optional[str] = None,
    broker_id: str = "MLB",
) -> List[CandleWithBroker]:
    """Fetch candles and broker flows for a specific symbol, timeframe, date and broker."""
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
        logger.info(f"Generating on-the-fly candles for {target_date} [{timeframe}]...")
        try:
            engine = MultiTimeframeCandleEngine(pg_manager=pg_mgr)
            engine.process_date_timeframe(target_date, timeframe)
        except Exception as e:
            logger.warning(f"Failed to dynamically generate candles: {e}")

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
) -> BrokerSummary:
    """Compute aggregate broker day summary across all stocks using audited FIFO ledger."""
    settings = get_settings()
    duckdb_path = str(settings.duckdb_path)
    target_date = date or "2026-09-14"

    # 1. First attempt: Query silver_broker_fifo_daily for full fidelity PnL
    try:
        conn = duckdb.connect(duckdb_path, read_only=True)
        try:
            res = conn.execute(
                """
                SELECT 
                    broker_id,
                    trade_date,
                    sum(buy_volume) as total_buy_vol,
                    sum(buy_turnover_tl) as total_buy_tl,
                    sum(sell_volume) as total_sell_vol,
                    sum(sell_turnover_tl) as total_sell_tl,
                    sum(buy_turnover_tl) - sum(sell_turnover_tl) as net_flow,
                    sum(matched_volume) as matched_vol,
                    sum(intraday_realized_pnl_tl) as intra_pnl,
                    sum(carry_fifo_realized_pnl_tl) as carry_pnl,
                    sum(daily_realized_pnl_tl) as daily_pnl
                FROM silver_broker_fifo_daily
                WHERE broker_id = ? AND trade_date = ?
                GROUP BY broker_id, trade_date;
            """,
                [broker_id, target_date],
            ).fetchone()

            if res:
                net_flow = float(res[6])
                bias = (
                    "STRONG_BUY"
                    if net_flow > 500_000_000
                    else ("BUY" if net_flow > 0 else ("STRONG_SELL" if net_flow < -500_000_000 else "SELL"))
                )
                return BrokerSummary(
                    broker_id=res[0],
                    trade_date=str(res[1]),
                    total_buy_volume=float(res[2]),
                    total_buy_turnover_tl=float(res[3]),
                    total_sell_volume=float(res[4]),
                    total_sell_turnover_tl=float(res[5]),
                    net_flow_tl=net_flow,
                    matched_volume=float(res[7]),
                    intraday_pnl_tl=float(res[8]),
                    carry_fifo_pnl_tl=float(res[9]),
                    realized_pnl_tl=float(res[10]),
                    position_bias=bias,
                )
        finally:
            conn.close()
    except Exception as e:
        logger.warning(f"DuckDB FIFO summary query failed: {e}")

    # Fallback to PostgreSQL 8h candle summary
    pg_mgr = PostgresConnectionManager()
    try:
        conn = pg_mgr.get_connection()
        try:
            with conn.cursor() as cur:
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
                        COALESCE(sum(realized_pnl_tl), 0)
                    FROM candle_broker_flows
                    WHERE timeframe = '8h' AND broker_id = %s 
                      AND bucket_start >= %s AND bucket_start <= %s
                    GROUP BY broker_id;
                """,
                    (broker_id, f"{target_date} 00:00:00", f"{target_date} 23:59:59"),
                )
                row = cur.fetchone()
                if row:
                    net_flow = float(row[5])
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
                        position_bias="BUY" if net_flow >= 0 else "SELL",
                    )
        finally:
            conn.close()
    except Exception:
        pass

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
    )


def get_daily_fifo(
    broker_id: str = "MLB",
    symbol: Optional[str] = None,
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
    limit: int = 100,
) -> List[DailyFifoRecord]:
    """Return historical daily FIFO ledger records from silver_broker_fifo_daily."""
    settings = get_settings()
    duckdb_path = str(settings.duckdb_path)
    conn = duckdb.connect(duckdb_path, read_only=True)
    results: List[DailyFifoRecord] = []
    try:
        conditions = ["broker_id = ?"]
        params = [broker_id]
        if symbol and symbol != "ALL":
            conditions.append("symbol = ?")
            params.append(symbol)
        if start_date:
            conditions.append("trade_date >= ?")
            params.append(start_date)
        if end_date:
            conditions.append("trade_date <= ?")
            params.append(end_date)

        where_clause = " WHERE " + " AND ".join(conditions)
        query = f"""
            SELECT 
                trade_date, symbol, symbol_name, sector, broker_id,
                buy_volume, buy_turnover_tl, buy_vwap,
                sell_volume, sell_turnover_tl, sell_vwap,
                matched_volume, intraday_realized_pnl_tl,
                carry_fifo_realized_pnl_tl, daily_realized_pnl_tl,
                position_side, open_stock_quantity, open_fifo_cost_tl, fifo_avg_cost,
                market_close_price, market_value_tl, unrealized_pnl_tl,
                total_daily_pnl_tl, cumulative_realized_pnl_tl
            FROM silver_broker_fifo_daily
            {where_clause}
            ORDER BY trade_date DESC, symbol ASC
            LIMIT {limit};
        """
        rows = conn.execute(query, params).fetchall()
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
    finally:
        conn.close()
    return results


def get_tertip_lots(
    broker_id: str = "MLB",
    symbol: Optional[str] = None,
    status: Optional[str] = None,
    limit: int = 100,
) -> List[TertipLot]:
    """Return historical tertip lot lifecycles from silver_broker_fifo_lot_lifecycle."""
    settings = get_settings()
    duckdb_path = str(settings.duckdb_path)
    conn = duckdb.connect(duckdb_path, read_only=True)
    results: List[TertipLot] = []
    try:
        conditions = ["broker_id = ?"]
        params = [broker_id]
        if symbol and symbol != "ALL":
            conditions.append("symbol = ?")
            params.append(symbol)
        if status and status != "ALL":
            conditions.append("status = ?")
            params.append(status.upper())

        where_clause = " WHERE " + " AND ".join(conditions)
        query = f"""
            SELECT 
                lot_id, broker_id, symbol, direction, open_date,
                opened_quantity, opened_value_tl, opened_unit_cost,
                status, closed_date, total_quantity_closed,
                total_closing_value_tl, total_realized_pnl_tl,
                remaining_quantity, remaining_value_tl
            FROM silver_broker_fifo_lot_lifecycle
            {where_clause}
            ORDER BY open_date DESC
            LIMIT {limit};
        """
        rows = conn.execute(query, params).fetchall()
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
    finally:
        conn.close()
    return results


def get_tertip_summary(
    broker_id: str = "MLB",
    date: Optional[str] = None,
) -> TertipExecutiveSummary:
    """Return executive summary metrics for tertip inventory and PnL."""
    settings = get_settings()
    duckdb_path = str(settings.duckdb_path)
    target_date = date or "2026-09-14"

    conn = duckdb.connect(duckdb_path, read_only=True)
    try:
        # Sum metrics from daily FIFO
        row = conn.execute(
            """
            SELECT 
                broker_id,
                trade_date,
                sum(cumulative_realized_pnl_tl),
                sum(daily_realized_pnl_tl),
                sum(intraday_realized_pnl_tl),
                sum(carry_fifo_realized_pnl_tl),
                sum(open_fifo_cost_tl),
                sum(market_value_tl),
                sum(unrealized_pnl_tl),
                sum(total_daily_pnl_tl)
            FROM silver_broker_fifo_daily
            WHERE broker_id = ? AND trade_date = ?
            GROUP BY broker_id, trade_date;
        """,
            [broker_id, target_date],
        ).fetchone()

        # Count active open lots
        open_lots_count = conn.execute(
            """
            SELECT count(*) 
            FROM silver_broker_fifo_lots 
            WHERE broker_id = ?;
        """,
            [broker_id],
        ).fetchone()[0]

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
    finally:
        conn.close()

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
    forward_days: int = 5,
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
    limit: int = 100,
) -> EventStudyResult:
    """Analyze historical stock movements and compute forward subsequent returns."""
    settings = get_settings()
    duckdb_path = str(settings.duckdb_path)
    conn = duckdb.connect(duckdb_path, read_only=True)

    clamped_forward_days = max(1, min(int(forward_days), 30))
    limit_val = max(1, min(int(limit), 500))

    lead_cols = []
    for k in range(1, clamped_forward_days + 1):
        lead_cols.append(f"LEAD(trade_date, {k}) OVER w as lead_date_{k}")
        lead_cols.append(f"LEAD(adj_close_price, {k}) OVER w as lead_price_{k}")
    leads_sql = ", ".join(lead_cols)

    cond_clauses = ["symbol = ?"]
    params: list = [symbol]

    cond_upper = condition_type.upper()
    if cond_upper == "DAILY_RETURN":
        if min_value is not None:
            cond_clauses.append("adj_daily_return_pct >= ?")
            params.append(float(min_value) / 100.0)
        if max_value is not None:
            cond_clauses.append("adj_daily_return_pct <= ?")
            params.append(float(max_value) / 100.0)
    elif cond_upper == "BOFA_NET_FLOW":
        if min_value is not None:
            cond_clauses.append("bofa_net_flow_tl >= ?")
            params.append(float(min_value))
        if max_value is not None:
            cond_clauses.append("bofa_net_flow_tl <= ?")
            params.append(float(max_value))
    elif cond_upper == "PRICE_RANGE":
        if min_value is not None:
            cond_clauses.append("price_range_pct >= ?")
            params.append(float(min_value) / 100.0)
        if max_value is not None:
            cond_clauses.append("price_range_pct <= ?")
            params.append(float(max_value) / 100.0)
    elif cond_upper == "VOLUME_SURGE":
        if min_value is not None:
            cond_clauses.append("total_turnover_tl >= ?")
            params.append(float(min_value))
        if max_value is not None:
            cond_clauses.append("total_turnover_tl <= ?")
            params.append(float(max_value))

    if start_date:
        cond_clauses.append("trade_date >= ?")
        params.append(start_date)
    if end_date:
        cond_clauses.append("trade_date <= ?")
        params.append(end_date)

    where_sql = " AND ".join(cond_clauses)

    query = f"""
        WITH base AS (
            SELECT 
                trade_date,
                symbol,
                adj_close_price,
                adj_daily_return_pct,
                bofa_net_flow_tl,
                price_range_pct,
                total_turnover_tl,
                {leads_sql}
            FROM silver_daily_stock_summary
            WHERE symbol = ?
            WINDOW w AS (ORDER BY trade_date ASC)
        )
        SELECT * FROM base
        WHERE {where_sql}
        ORDER BY trade_date DESC
        LIMIT {limit_val};
    """

    try:
        full_params = [symbol] + params
        df = conn.execute(query, full_params).fetchdf()

        count_query = f"""
            WITH base AS (
                SELECT 
                    trade_date,
                    symbol,
                    adj_close_price,
                    adj_daily_return_pct,
                    bofa_net_flow_tl,
                    price_range_pct,
                    total_turnover_tl
                FROM silver_daily_stock_summary
                WHERE symbol = ?
            )
            SELECT count(*) FROM base WHERE {where_sql};
        """
        total_occurrences = conn.execute(count_query, full_params).fetchone()[0]
    finally:
        conn.close()

    occurrences: list[EventStudyOccurrence] = []
    horizon_returns: dict[int, list[float]] = {k: [] for k in range(1, clamped_forward_days + 1)}

    for _, row in df.iterrows():
        p0 = float(row["adj_close_price"]) if row["adj_close_price"] is not None else 0.0

        if cond_upper == "DAILY_RETURN":
            movement_val = round(float(row["adj_daily_return_pct"] or 0) * 100.0, 2)
        elif cond_upper == "BOFA_NET_FLOW":
            movement_val = round(float(row["bofa_net_flow_tl"] or 0), 2)
        elif cond_upper == "PRICE_RANGE":
            movement_val = round(float(row["price_range_pct"] or 0) * 100.0, 2)
        elif cond_upper == "VOLUME_SURGE":
            movement_val = round(float(row["total_turnover_tl"] or 0), 2)
        else:
            movement_val = round(float(row["adj_daily_return_pct"] or 0) * 100.0, 2)

        fwd_points: list[ForwardReturnPoint] = []
        for k in range(1, clamped_forward_days + 1):
            pk = row[f"lead_price_{k}"]
            dk = row[f"lead_date_{k}"]
            ret_pct: Optional[float] = None
            if pk is not None and not (isinstance(pk, float) and (pk != pk)) and p0 > 0:
                ret_pct = round(((float(pk) - p0) / p0) * 100.0, 2)
                horizon_returns[k].append(ret_pct)

            fwd_points.append(
                ForwardReturnPoint(
                    day_offset=k,
                    date=str(dk)[:10] if dk is not None and dk == dk else None,
                    close_price=float(pk) if (pk is not None and pk == pk) else None,
                    return_pct=ret_pct,
                )
            )

        occurrences.append(
            EventStudyOccurrence(
                event_date=str(row["trade_date"])[:10],
                close_price=p0,
                movement_value=movement_val,
                bofa_net_flow_tl=float(row["bofa_net_flow_tl"] or 0),
                total_turnover_tl=float(row["total_turnover_tl"] or 0),
                forward_returns=fwd_points,
            )
        )

    horizon_stats: list[ForwardHorizonStat] = []
    for k in range(1, clamped_forward_days + 1):
        rets = horizon_returns[k]
        if rets:
            avg_ret = round(sum(rets) / len(rets), 2)
            win_rate = round((sum(1 for r in rets if r > 0) / len(rets)) * 100.0, 1)
            med_ret = round(statistics.median(rets), 2)
            max_g = round(max(rets), 2)
            max_l = round(min(rets), 2)
        else:
            avg_ret = 0.0
            win_rate = 0.0
            med_ret = 0.0
            max_g = 0.0
            max_l = 0.0

        horizon_stats.append(
            ForwardHorizonStat(
                day_offset=k,
                avg_return_pct=avg_ret,
                win_rate_pct=win_rate,
                median_return_pct=med_ret,
                max_gain_pct=max_g,
                max_loss_pct=max_l,
                sample_count=len(rets),
            )
        )

    return EventStudyResult(
        symbol=symbol,
        condition_type=cond_upper,
        min_value=min_value,
        max_value=max_value,
        forward_days=clamped_forward_days,
        total_occurrences=int(total_occurrences),
        horizon_stats=horizon_stats,
        occurrences=occurrences,
    )
