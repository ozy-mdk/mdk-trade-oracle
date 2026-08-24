"""Silver Layer: Per-Stock Per-Window Intraday Return Percentile Thresholds.

Computes empirical return distribution quantiles (P25, P50, P85) for each BIST30 stock
across three reaction windows (W2: first_reaction, W3: midday_followup, W5: closing_session).

Used by Model 3 (StockReactionForecaster) for dynamic direction classification:
- STRONG_RALLY:   return >= up_p85_pct
- RALLY:          up_p50_pct <= return < up_p85_pct
- WEAK_RALLY:     up_p25_pct <= return < up_p50_pct
- NEUTRAL:        |return| < min(up_p25_pct, down_p25_pct)
- WEAK_DECLINE:   down_p25_pct <= |return| < down_p50_pct  (and return < 0)
- DECLINE:        down_p50_pct <= |return| < down_p85_pct  (and return < 0)
- STRONG_DECLINE: |return| >= down_p85_pct                 (and return < 0)

Return measurement: W1 VWAP price -> window-end VWAP price (execution-quality adjusted).
"""

from typing import Any

import polars as pl

from mdk_trading_oracle.core.db import DuckDBManager
from mdk_trading_oracle.core.logger import get_logger

logger = get_logger("mdk_oracle.data.silver.stock_reaction_thresholds")

# Windows targeted by Model 3 reaction forecasting
# W1 (day_start) is the input signal window — not a target
REACTION_WINDOWS = ["first_reaction", "midday_followup", "closing_session"]


class StockReactionThresholdEngine:
    """Computes and persists per-stock per-window empirical return percentile thresholds.

    The return for each window is measured as:
        (window_end_market_vwap - w1_ref_price) / w1_ref_price * 100

    where w1_ref_price is the BofA W1 buy VWAP (or market W1 VWAP as fallback, then adj_open_price).
    This provides an execution-aware return: 'if you filled alongside BofA at open,
    where is the stock trading by window end?'
    """

    def __init__(self, db: DuckDBManager) -> None:
        self.db = db

    def compute_and_persist(self) -> dict[str, Any]:
        """Compute per-stock per-window return quantiles and persist to silver_stock_reaction_thresholds."""
        conn = self.db.get_connection()
        logger.info(
            "Computing `silver_stock_reaction_thresholds` "
            "(per-stock per-window empirical return percentiles)..."
        )

        query = """
            WITH w1_bofa_vwap AS (
                -- BofA W1 buy VWAP as the execution reference price (primary)
                SELECT
                    trade_date,
                    symbol,
                    CASE
                        WHEN buy_volume > 0 THEN buy_turnover_tl / buy_volume
                        ELSE NULL
                    END AS bofa_w1_buy_vwap
                FROM silver_intraday_broker_window_summary
                WHERE window_name = 'day_start'
                  AND broker_id = 'MLB'
            ),
            w1_market_vwap AS (
                -- Market-wide W1 VWAP as fallback (all brokers aggregated)
                SELECT
                    trade_date,
                    symbol,
                    CASE
                        WHEN SUM(buy_volume + sell_volume) > 0
                        THEN SUM(buy_turnover_tl + sell_turnover_tl) / SUM(buy_volume + sell_volume)
                        ELSE NULL
                    END AS market_w1_vwap
                FROM silver_intraday_broker_window_summary
                WHERE window_name = 'day_start'
                GROUP BY trade_date, symbol
            ),
            ref_prices AS (
                -- Resolve best entry reference price per (trade_date, symbol)
                SELECT
                    stk.trade_date,
                    stk.symbol,
                    COALESCE(
                        b.bofa_w1_buy_vwap,
                        m.market_w1_vwap,
                        stk.adj_open_price,
                        stk.open_price
                    ) AS entry_price
                FROM silver_daily_stock_summary stk
                LEFT JOIN w1_bofa_vwap b
                    ON stk.trade_date = b.trade_date AND stk.symbol = b.symbol
                LEFT JOIN w1_market_vwap m
                    ON stk.trade_date = m.trade_date AND stk.symbol = m.symbol
            ),
            window_vwaps AS (
                -- Market VWAP for each of the 3 reaction windows per (trade_date, symbol)
                SELECT
                    trade_date,
                    symbol,
                    window_name,
                    CASE
                        WHEN SUM(buy_volume + sell_volume) > 0
                        THEN SUM(buy_turnover_tl + sell_turnover_tl) / SUM(buy_volume + sell_volume)
                        ELSE NULL
                    END AS window_market_vwap
                FROM silver_intraday_broker_window_summary
                WHERE window_name IN ('first_reaction', 'midday_followup', 'closing_session')
                GROUP BY trade_date, symbol, window_name
            ),
            returns AS (
                -- Execution-aware return% per (trade_date, symbol, window)
                SELECT
                    wv.trade_date,
                    wv.symbol,
                    wv.window_name,
                    rp.entry_price,
                    wv.window_market_vwap,
                    CASE
                        WHEN rp.entry_price > 0 AND wv.window_market_vwap IS NOT NULL
                        THEN (wv.window_market_vwap - rp.entry_price) / rp.entry_price * 100.0
                        ELSE NULL
                    END AS intraday_return_pct
                FROM window_vwaps wv
                INNER JOIN ref_prices rp
                    ON wv.trade_date = rp.trade_date AND wv.symbol = rp.symbol
                WHERE rp.entry_price > 0
            )
            SELECT symbol, window_name, intraday_return_pct
            FROM returns
            WHERE intraday_return_pct IS NOT NULL
            ORDER BY symbol, window_name, trade_date;
        """

        df = conn.execute(query).pl()
        logger.info(
            f"Fetched {len(df):,} return observations "
            f"for {df['symbol'].n_unique()} stocks × {df['window_name'].n_unique()} windows"
        )

        if df.is_empty():
            logger.warning(
                "No return data available — `silver_stock_reaction_thresholds` will have no rows. "
                "Ensure Silver intraday window tables are populated."
            )
            return {"table": "silver_stock_reaction_thresholds", "rows": 0, "status": "empty"}

        # Compute quantiles per (symbol, window_name)
        threshold_rows = []
        for keys, group in df.group_by(["symbol", "window_name"]):
            symbol, window_name = keys[0], keys[1]
            returns_series = group["intraday_return_pct"].drop_nulls()
            pos = returns_series.filter(returns_series > 0)
            neg = returns_series.filter(returns_series < 0).abs()

            total_sessions = len(returns_series)
            up_count = len(pos)
            down_count = len(neg)

            # Minimum 3 sessions per direction for meaningful quantiles; use fallbacks otherwise
            up_p25 = float(pos.quantile(0.25)) if up_count >= 3 else 0.20
            up_p50 = float(pos.quantile(0.50)) if up_count >= 3 else 0.50
            up_p85 = float(pos.quantile(0.85)) if up_count >= 3 else 1.50

            down_p25 = float(neg.quantile(0.25)) if down_count >= 3 else 0.20
            down_p50 = float(neg.quantile(0.50)) if down_count >= 3 else 0.50
            down_p85 = float(neg.quantile(0.85)) if down_count >= 3 else 1.50

            threshold_rows.append({
                "symbol": symbol,
                "window_name": window_name,
                "up_p25_pct": up_p25,
                "up_p50_pct": up_p50,
                "up_p85_pct": up_p85,
                "down_p25_pct": down_p25,
                "down_p50_pct": down_p50,
                "down_p85_pct": down_p85,
                "up_session_count": up_count,
                "down_session_count": down_count,
                "total_sessions": total_sessions,
            })

        pl_thresholds = pl.DataFrame(threshold_rows)
        conn.register("df_stock_rxn_thresh_temp", pl_thresholds)
        conn.execute("""
            CREATE OR REPLACE TABLE silver_stock_reaction_thresholds AS
            SELECT
                symbol,
                window_name,
                up_p25_pct,
                up_p50_pct,
                up_p85_pct,
                down_p25_pct,
                down_p50_pct,
                down_p85_pct,
                CAST(up_session_count AS INTEGER) AS up_session_count,
                CAST(down_session_count AS INTEGER) AS down_session_count,
                CAST(total_sessions AS INTEGER) AS total_sessions,
                CURRENT_TIMESTAMP AS calculated_at
            FROM df_stock_rxn_thresh_temp
            ORDER BY symbol, window_name;
        """)
        conn.unregister("df_stock_rxn_thresh_temp")

        rows = len(pl_thresholds)
        symbols_count = pl_thresholds["symbol"].n_unique()
        windows_count = pl_thresholds["window_name"].n_unique()

        logger.info(
            f"Successfully built `silver_stock_reaction_thresholds`: "
            f"{rows:,} threshold profiles | {symbols_count} stocks x {windows_count} windows"
        )
        return {
            "table": "silver_stock_reaction_thresholds",
            "rows": rows,
            "symbols": symbols_count,
            "windows": windows_count,
            "status": "success",
        }
