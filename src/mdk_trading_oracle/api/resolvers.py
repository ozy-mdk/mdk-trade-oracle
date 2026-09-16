"""Database resolvers for GraphQL queries."""

import logging
from typing import List, Optional

import psycopg2
import yaml

from mdk_trading_oracle.api.types import Broker, BrokerSummary, CandleWithBroker, Instrument
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
        Instrument(symbol="THYAO", name="Türk Hava Yolları", sector="ULASTIRMA"),
        Instrument(symbol="AKBNK", name="Akbank", sector="BANKA"),
        Instrument(symbol="ASELS", name="Aselsan", sector="SAVUNMA"),
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
            key=lambda x: (0 if x.get("code") == "MLB" or x.get("broker_id") == "MLB" else 1, x.get("code", x.get("broker_id", ""))),
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
    pg_mgr = PostgresConnectionManager()
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
    except Exception as e:
        logger.warning(f"Failed to fetch dates from PG: {e}")
    finally:
        conn.close()

    # Fallback to candle engine discovery from DuckDB
    try:
        engine = MultiTimeframeCandleEngine(pg_manager=pg_mgr)
        dates = engine.get_available_dates()
        return list(reversed(dates[-30:]))  # Most recent 30 trading days
    except Exception:
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
    """Compute aggregate broker day summary across all stocks."""
    pg_mgr = PostgresConnectionManager()
    target_date = date or "2026-09-14"

    conn = pg_mgr.get_connection()
    try:
        query = """
            SELECT 
                broker_id,
                COALESCE(sum(buy_volume), 0) as total_buy_vol,
                COALESCE(sum(buy_turnover_tl), 0) as total_buy_tl,
                COALESCE(sum(sell_volume), 0) as total_sell_vol,
                COALESCE(sum(sell_turnover_tl), 0) as total_sell_tl,
                COALESCE(sum(net_flow_tl), 0) as net_flow,
                COALESCE(sum(matched_volume), 0) as matched_vol,
                COALESCE(sum(realized_pnl_tl), 0) as realized_pnl
            FROM candle_broker_flows
            WHERE timeframe = '8h' 
              AND broker_id = %s 
              AND bucket_start >= %s 
              AND bucket_start <= %s
            GROUP BY broker_id;
        """
        with conn.cursor() as cur:
            cur.execute(query, (broker_id, f"{target_date} 00:00:00", f"{target_date} 23:59:59"))
            row = cur.fetchone()
            if row:
                net_flow = float(row[5])
                bias = "STRONG_BUY" if net_flow > 500_000_000 else (
                    "BUY" if net_flow > 0 else (
                        "STRONG_SELL" if net_flow < -500_000_000 else "SELL"
                    )
                )
                return BrokerSummary(
                    broker_id=row[0],
                    trade_date=target_date,
                    total_buy_volume=float(row[1]),
                    total_buy_turnover_tl=float(row[2]),
                    total_sell_volume=float(row[3]),
                    total_sell_turnover_tl=float(row[4]),
                    net_flow_tl=net_flow,
                    matched_volume=float(row[6]),
                    realized_pnl_tl=float(row[7]),
                    position_bias=bias,
                )
    finally:
        conn.close()

    return BrokerSummary(
        broker_id=broker_id,
        trade_date=target_date,
        total_buy_volume=0.0,
        total_buy_turnover_tl=0.0,
        total_sell_volume=0.0,
        total_sell_turnover_tl=0.0,
        net_flow_tl=0.0,
        matched_volume=0.0,
        realized_pnl_tl=0.0,
        position_bias="NEUTRAL",
    )
