"""BIST 30 (XU030) Synthetic Index & Instrument Calculation Engine.

Computes the BIST 30 index from its 30 constituent equities using quarterly
membership snapshots and calibrated weights, generating multi-timeframe
candles, broker flows, and daily summaries under symbol 'XU030'.
"""

import logging
from datetime import datetime
from typing import Dict, List, Optional

import duckdb
import numpy as np
import pandas as pd
from scipy.optimize import nnls

from mdk_trading_oracle.core.config import get_settings
from mdk_trading_oracle.data.postgres.connection import PostgresConnectionManager

logger = logging.getLogger(__name__)


class BIST30IndexEngine:
    """Calculates synthetic BIST 30 (XU030) candles, flows, and metrics."""

    def __init__(
        self,
        duckdb_path: Optional[str] = None,
        pg_manager: Optional[PostgresConnectionManager] = None,
    ):
        settings = get_settings()
        self.duckdb_path = duckdb_path or str(settings.duckdb_path)
        self.pg_manager = pg_manager or PostgresConnectionManager()
        self._weights_cache: Dict[str, Dict[str, float]] = {}

    def get_constituents(self, trade_date: str) -> List[str]:
        """Return list of active 30 constituent stock symbols for a given date."""
        conn = duckdb.connect(self.duckdb_path, read_only=True)
        try:
            target_date = trade_date[:10]
            # Query bronze_bist30_membership for exact date
            rows = conn.execute(
                """
                SELECT DISTINCT symbol 
                FROM bronze_bist30_membership 
                WHERE start_date <= ? AND (end_date IS NULL OR end_date >= ?)
                ORDER BY symbol;
            """,
                [target_date, target_date],
            ).fetchall()

            if rows and len(rows) >= 25:
                return [r[0] for r in rows if r[0]]

            # Fallback to latest membership snapshot
            rows = conn.execute(
                """
                SELECT DISTINCT symbol 
                FROM bronze_bist30_membership 
                WHERE start_date = (SELECT max(start_date) FROM bronze_bist30_membership)
                ORDER BY symbol;
            """
            ).fetchall()

            if rows and len(rows) >= 25:
                return [r[0] for r in rows if r[0]]

            # Fallback to bronze_instruments index_name = 'BIST30'
            rows = conn.execute(
                """
                SELECT DISTINCT symbol 
                FROM bronze_instruments 
                WHERE index_name = 'BIST30' AND symbol != 'XU030'
                ORDER BY symbol;
            """
            ).fetchall()

            return [r[0] for r in rows if r[0]]
        finally:
            conn.close()

    def get_constituent_weights(self, trade_date: str) -> Dict[str, float]:
        """Compute or retrieve calibrated constituent weights for the date's quarter."""
        dt = datetime.strptime(trade_date[:10], "%Y-%m-%d").date()
        quarter = (dt.month - 1) // 3 + 1
        quarter_key = f"{dt.year}-Q{quarter}"

        if quarter_key in self._weights_cache:
            return self._weights_cache[quarter_key]

        constituents = self.get_constituents(trade_date)
        if not constituents:
            logger.warning(f"No constituents found for {trade_date}")
            return {}

        conn = duckdb.connect(self.duckdb_path, read_only=True)
        try:
            # Lookback up to 6 months to ensure solid regression sample
            lookback_start = f"{dt.year - 1}-10-01" if dt.month <= 3 else f"{dt.year}-01-01"

            stocks_df = conn.execute(
                """
                SELECT trade_date, symbol, adj_close_price
                FROM silver_daily_stock_summary
                WHERE trade_date >= ? AND trade_date <= ?
                  AND symbol IN (SELECT unnest(?))
                ORDER BY trade_date, symbol;
            """,
                [lookback_start, trade_date[:10], constituents],
            ).fetchdf()

            index_df = conn.execute(
                """
                SELECT trade_date, close_price as index_close
                FROM bronze_bist_index_benchmarks
                WHERE trade_date >= ? AND trade_date <= ?
                ORDER BY trade_date;
            """,
                [lookback_start, trade_date[:10]],
            ).fetchdf()

            if stocks_df.empty or index_df.empty:
                logger.warning("Insufficient data to fit weights; using equal weights.")
                eq_weight = 1.0 / len(constituents)
                return {s: eq_weight for s in constituents}

            pivot = stocks_df.pivot(index="trade_date", columns="symbol", values="adj_close_price").dropna()
            merged = pivot.join(index_df.set_index("trade_date"), how="inner").dropna()

            if len(merged) < 10:
                merged = pivot.join(index_df.set_index("trade_date"), how="inner").fillna(method="ffill").dropna()

            symbols_present = [s for s in pivot.columns if s in constituents]
            X = merged[symbols_present].values
            y = merged["index_close"].values

            weights, _ = nnls(X, y)
            weight_map = dict(zip(symbols_present, [float(w) for w in weights]))

            y_pred = X @ weights
            ss_tot = np.sum((y - np.mean(y)) ** 2)
            r2 = 1 - (np.sum((y - y_pred) ** 2) / (ss_tot + 1e-8))
            logger.info(
                f"Fitted BIST 30 weights for {quarter_key} ({len(symbols_present)} stocks, {len(merged)} sessions, R2={r2:.5f})"
            )

            self._weights_cache[quarter_key] = weight_map
            return weight_map
        finally:
            conn.close()

    def generate_xu030_candles(self, trade_date: str, timeframe: str = "60m") -> int:
        """Generate synthetic XU030 candles and broker flows in PostgreSQL for given date & timeframe."""
        target_date = trade_date[:10]
        constituents = self.get_constituents(target_date)
        if not constituents:
            logger.warning(f"No constituents found for {target_date}")
            return 0

        weights = self.get_constituent_weights(target_date)
        if not weights:
            logger.warning(f"No weights available for {target_date}")
            return 0

        total_weight_sum = sum(weights.values())

        # Query constituent candles from PostgreSQL
        pg_conn = self.pg_manager.get_connection()
        try:
            with pg_conn.cursor() as cur:
                cur.execute(
                    """
                    SELECT 
                        bucket_start, bucket_end, symbol, 
                        open, high, low, close, volume, turnover_tl, trade_count
                    FROM market_candles
                    WHERE timeframe = %s AND symbol = ANY(%s)
                      AND bucket_start >= %s AND bucket_start <= %s
                    ORDER BY bucket_start ASC, symbol ASC;
                """,
                    (
                        timeframe,
                        constituents,
                        f"{target_date} 00:00:00",
                        f"{target_date} 23:59:59",
                    ),
                )
                rows = cur.fetchall()

            if not rows:
                logger.info(f"No constituent candles found in PG for {target_date} [{timeframe}].")
                return 0

            df = pd.DataFrame(
                rows,
                columns=[
                    "bucket_start",
                    "bucket_end",
                    "symbol",
                    "open",
                    "high",
                    "low",
                    "close",
                    "volume",
                    "turnover_tl",
                    "trade_count",
                ],
            )

            # Query official day benchmark OHLC from DuckDB if available for precision calibration
            bench_ohlc = None
            try:
                duck_conn = duckdb.connect(self.duckdb_path, read_only=True)
                res = duck_conn.execute(
                    "SELECT open_price, high_price, low_price, close_price FROM bronze_bist_index_benchmarks WHERE trade_date = ?",
                    [target_date],
                ).fetchone()
                if res and res[0] is not None:
                    bench_ohlc = {
                        "open": float(res[0]),
                        "high": float(res[1]),
                        "low": float(res[2]),
                        "close": float(res[3]),
                    }
                duck_conn.close()
            except Exception:
                pass

            # Calculate synthetic XU030 candle per bucket
            xu030_candles = []
            buckets = sorted(df["bucket_start"].unique())

            for i, b_start in enumerate(buckets):
                b_df = df[df["bucket_start"] == b_start].copy()
                b_end = b_df["bucket_end"].iloc[0]

                # Map weights
                b_df["weight"] = b_df["symbol"].map(weights).fillna(0.0)

                # Active constituents in this candle
                active_weights_sum = b_df["weight"].sum()
                norm_factor = 1.0
                if active_weights_sum > 0 and total_weight_sum > 0:
                    norm_factor = total_weight_sum / active_weights_sum

                c_open = float((b_df["open"] * b_df["weight"] * norm_factor).sum())
                c_close = float((b_df["close"] * b_df["weight"] * norm_factor).sum())
                c_high = float(max(c_open, c_close, (b_df["high"] * b_df["weight"] * norm_factor).sum()))
                c_low = float(min(c_open, c_close, (b_df["low"] * b_df["weight"] * norm_factor).sum()))

                # Special hours precision handling:
                # 1. Opening candle (first bucket of day): If bench_ohlc exists, align open
                is_first_bucket = i == 0
                is_last_bucket = i == len(buckets) - 1

                if is_first_bucket and bench_ohlc is not None:
                    # Keep synthetic close/high/low consistent with official open
                    if abs(c_open - bench_ohlc["open"]) / bench_ohlc["open"] < 0.005:
                        c_open = bench_ohlc["open"]
                        c_high = max(c_high, c_open)
                        c_low = min(c_low, c_open)

                # 2. Closing auction candle (last bucket of day e.g. 18:00 or 18:05):
                if is_last_bucket and bench_ohlc is not None:
                    if abs(c_close - bench_ohlc["close"]) / bench_ohlc["close"] < 0.005:
                        c_close = bench_ohlc["close"]
                        c_high = max(c_high, c_close)
                        c_low = min(c_low, c_close)

                c_vol = float(b_df["volume"].sum())
                c_turnover = float(b_df["turnover_tl"].sum())
                c_trades = int(b_df["trade_count"].sum())
                c_vwap = round(c_turnover / c_vol, 2) if c_vol > 0 else round((c_open + c_close) / 2.0, 2)

                xu030_candles.append(
                    (
                        timeframe,
                        b_start,
                        b_end,
                        "XU030",
                        round(c_open, 2),
                        round(c_high, 2),
                        round(c_low, 2),
                        round(c_close, 2),
                        round(c_vol, 2),
                        round(c_turnover, 2),
                        c_vwap,
                        c_trades,
                    )
                )

            # Upsert into PostgreSQL market_candles
            with pg_conn.cursor() as cur:
                cur.executemany(
                    """
                    INSERT INTO market_candles (
                        timeframe, bucket_start, bucket_end, symbol,
                        open, high, low, close, volume, turnover_tl, vwap, trade_count
                    ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                    ON CONFLICT (timeframe, symbol, bucket_start) DO UPDATE SET
                        open = EXCLUDED.open,
                        high = EXCLUDED.high,
                        low = EXCLUDED.low,
                        close = EXCLUDED.close,
                        volume = EXCLUDED.volume,
                        turnover_tl = EXCLUDED.turnover_tl,
                        vwap = EXCLUDED.vwap,
                        trade_count = EXCLUDED.trade_count;
                """,
                    xu030_candles,
                )

                # Aggregate institutional broker flows for XU030 across the 30 constituent stocks
                cur.execute(
                    """
                    INSERT INTO candle_broker_flows (
                        timeframe, bucket_start, symbol, broker_id,
                        buy_volume, buy_turnover_tl, buy_vwap,
                        sell_volume, sell_turnover_tl, sell_vwap,
                        net_volume, net_flow_tl, matched_volume,
                        matched_buy_value_tl, matched_sell_value_tl, realized_pnl_tl
                    )
                    SELECT 
                        timeframe,
                        bucket_start,
                        'XU030' as symbol,
                        broker_id,
                        sum(buy_volume) as buy_volume,
                        sum(buy_turnover_tl) as buy_turnover_tl,
                        CASE WHEN sum(buy_volume) > 0 THEN sum(buy_turnover_tl) / sum(buy_volume) ELSE NULL END as buy_vwap,
                        sum(sell_volume) as sell_volume,
                        sum(sell_turnover_tl) as sell_turnover_tl,
                        CASE WHEN sum(sell_volume) > 0 THEN sum(sell_turnover_tl) / sum(sell_volume) ELSE NULL END as sell_vwap,
                        sum(net_volume) as net_volume,
                        sum(net_flow_tl) as net_flow_tl,
                        sum(matched_volume) as matched_volume,
                        sum(matched_buy_value_tl) as matched_buy_value_tl,
                        sum(matched_sell_value_tl) as matched_sell_value_tl,
                        sum(realized_pnl_tl) as realized_pnl_tl
                    FROM candle_broker_flows
                    WHERE timeframe = %s AND symbol = ANY(%s)
                      AND bucket_start >= %s AND bucket_start <= %s
                    GROUP BY timeframe, bucket_start, broker_id
                    ON CONFLICT (timeframe, symbol, broker_id, bucket_start) DO UPDATE SET
                        buy_volume = EXCLUDED.buy_volume,
                        buy_turnover_tl = EXCLUDED.buy_turnover_tl,
                        buy_vwap = EXCLUDED.buy_vwap,
                        sell_volume = EXCLUDED.sell_volume,
                        sell_turnover_tl = EXCLUDED.sell_turnover_tl,
                        sell_vwap = EXCLUDED.sell_vwap,
                        net_volume = EXCLUDED.net_volume,
                        net_flow_tl = EXCLUDED.net_flow_tl,
                        matched_volume = EXCLUDED.matched_volume,
                        matched_buy_value_tl = EXCLUDED.matched_buy_value_tl,
                        matched_sell_value_tl = EXCLUDED.matched_sell_value_tl,
                        realized_pnl_tl = EXCLUDED.realized_pnl_tl;
                """,
                    (
                        timeframe,
                        constituents,
                        f"{target_date} 00:00:00",
                        f"{target_date} 23:59:59",
                    ),
                )
                pg_conn.commit()

            logger.info(
                f"Generated {len(xu030_candles)} XU030 candles [{timeframe}] for {target_date} with consolidated broker flows."
            )
            return len(xu030_candles)
        finally:
            pg_conn.close()

    def generate_xu030_daily_summary(self, trade_date: Optional[str] = None) -> int:
        """Insert/update daily XU030 records in DuckDB silver_daily_stock_summary."""
        conn = duckdb.connect(self.duckdb_path, read_only=False)
        try:
            if trade_date:
                conn.execute(
                    "DELETE FROM silver_daily_stock_summary WHERE symbol = 'XU030' AND trade_date = ?",
                    [trade_date[:10]],
                )
                date_filter = f"WHERE b.trade_date = '{trade_date[:10]}'"
            else:
                conn.execute("DELETE FROM silver_daily_stock_summary WHERE symbol = 'XU030'")
                date_filter = ""

            insert_sql = f"""
                INSERT INTO silver_daily_stock_summary (
                    trade_date,
                    day_of_week,
                    is_monday,
                    is_friday,
                    symbol,
                    canonical_symbol,
                    symbol_name,
                    sector,
                    index_name,
                    open_price,
                    high_price,
                    low_price,
                    close_price,
                    market_vwap,
                    daily_return_pct,
                    price_range_pct,
                    total_volume,
                    total_turnover_tl,
                    total_trades,
                    active_brokers_count,
                    quantity_factor,
                    has_unresolved_paid_action,
                    adj_open_price,
                    adj_high_price,
                    adj_low_price,
                    adj_close_price,
                    adj_market_vwap,
                    adj_total_volume,
                    adj_daily_return_pct,
                    top_buyer_broker_id,
                    top_buyer_turnover_tl,
                    top_buyer_share,
                    top_seller_broker_id,
                    top_seller_turnover_tl,
                    top_seller_share,
                    top_5_buyers_net_flow_tl,
                    top_5_sellers_net_flow_tl,
                    top_5_concentration_ratio,
                    top_5_domestic_net_flow_tl,
                    bofa_buy_turnover_tl,
                    bofa_sell_turnover_tl,
                    bofa_net_flow_tl,
                    bofa_stock_turnover_share,
                    bofa_buy_vwap,
                    bofa_sell_vwap,
                    bofa_total_vwap,
                    bofa_vwap_spread_pct,
                    adj_bofa_buy_vwap,
                    adj_bofa_sell_vwap,
                    adj_bofa_total_vwap,
                    bofa_rank_in_stock,
                    calculated_at
                )
                WITH agg_flows AS (
                    SELECT 
                        trade_date,
                        sum(total_volume) as total_volume,
                        sum(total_turnover_tl) as total_turnover_tl,
                        sum(total_trades) as total_trades,
                        max(active_brokers_count) as active_brokers_count,
                        sum(bofa_buy_turnover_tl) as bofa_buy_tl,
                        sum(bofa_sell_turnover_tl) as bofa_sell_tl,
                        sum(bofa_net_flow_tl) as bofa_net_tl
                    FROM silver_daily_stock_summary
                    WHERE symbol != 'XU030'
                    GROUP BY trade_date
                )
                SELECT 
                    b.trade_date,
                    extract(dow from b.trade_date) as day_of_week,
                    extract(dow from b.trade_date) = 1 as is_monday,
                    extract(dow from b.trade_date) = 5 as is_friday,
                    'XU030' as symbol,
                    'XU030' as canonical_symbol,
                    'BIST 30 Endeksi' as symbol_name,
                    'ENDEKS' as sector,
                    'BIST30' as index_name,
                    b.open_price,
                    b.high_price,
                    b.low_price,
                    b.close_price,
                    round((b.open_price + b.close_price) / 2.0, 2) as market_vwap,
                    b.daily_return_pct,
                    b.price_range_pct,
                    coalesce(a.total_volume, b.volume) as total_volume,
                    coalesce(a.total_turnover_tl, 0.0) as total_turnover_tl,
                    coalesce(a.total_trades, 0) as total_trades,
                    coalesce(a.active_brokers_count, 50) as active_brokers_count,
                    1.0 as quantity_factor,
                    false as has_unresolved_paid_action,
                    b.open_price as adj_open_price,
                    b.high_price as adj_high_price,
                    b.low_price as adj_low_price,
                    b.close_price as adj_close_price,
                    round((b.open_price + b.close_price) / 2.0, 2) as adj_market_vwap,
                    coalesce(a.total_volume, b.volume) as adj_total_volume,
                    b.daily_return_pct as adj_daily_return_pct,
                    NULL as top_buyer_broker_id,
                    NULL as top_buyer_turnover_tl,
                    NULL as top_buyer_share,
                    NULL as top_seller_broker_id,
                    NULL as top_seller_turnover_tl,
                    NULL as top_seller_share,
                    NULL as top_5_buyers_net_flow_tl,
                    NULL as top_5_sellers_net_flow_tl,
                    NULL as top_5_concentration_ratio,
                    NULL as top_5_domestic_net_flow_tl,
                    coalesce(a.bofa_buy_tl, 0.0) as bofa_buy_turnover_tl,
                    coalesce(a.bofa_sell_tl, 0.0) as bofa_sell_turnover_tl,
                    coalesce(a.bofa_net_tl, 0.0) as bofa_net_flow_tl,
                    CASE WHEN a.total_turnover_tl > 0 THEN (coalesce(a.bofa_buy_tl, 0) + coalesce(a.bofa_sell_tl, 0)) / (2.0 * a.total_turnover_tl) * 100.0 ELSE 0.0 END as bofa_stock_turnover_share,
                    b.close_price as bofa_buy_vwap,
                    b.close_price as bofa_sell_vwap,
                    b.close_price as bofa_total_vwap,
                    0.0 as bofa_vwap_spread_pct,
                    b.close_price as adj_bofa_buy_vwap,
                    b.close_price as adj_bofa_sell_vwap,
                    b.close_price as adj_bofa_total_vwap,
                    1 as bofa_rank_in_stock,
                    current_timestamp as calculated_at
                FROM bronze_bist_index_benchmarks b
                LEFT JOIN agg_flows a ON b.trade_date = a.trade_date
                {date_filter};
            """
            conn.execute(insert_sql)
            count = conn.execute(
                "SELECT count(*) FROM silver_daily_stock_summary WHERE symbol = 'XU030';"
            ).fetchone()[0]
            logger.info(f"Populated silver_daily_stock_summary with {count} daily records for XU030.")
            return count
        finally:
            conn.close()
