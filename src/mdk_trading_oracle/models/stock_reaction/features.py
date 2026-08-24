"""Stock Reaction Feature Extraction Engine: 8 Clusters for W2/W3/W5 Return Prediction.

Features are computed using ONLY data available at T_feature_cutoff = W1 end (10:30 TRT):
  - Same-day W1 execution data (BofA + 6 tracked brokers in that window)
  - T-1 end-of-day inventory, FIFO cost basis, stock returns, macro rates
  - Rolling multi-day (5d/20d) momentum, inventory, and breadth from T-1 backward

Zero lookahead guarantee: no W2/W3/W4/W5 data from the same trade_date is used as a feature.
"""

from datetime import date, timedelta
from typing import List, Optional

import polars as pl

from mdk_trading_oracle.core.config import get_settings
from mdk_trading_oracle.core.db import DuckDBManager
from mdk_trading_oracle.core.logger import get_logger
from mdk_trading_oracle.models.base import BaseFeatureExtractor

logger = get_logger("mdk_oracle.models.stock_reaction.features")

# Tracked broker universe for FIFO inventory signals (from config, loaded at runtime)
DEFAULT_TRACKED_BROKERS = ["MLB", "IYM", "YKR", "AKM", "GRM", "ZRY", "TRA"]

# Window name -> target column alias
WINDOW_TARGET_COLS = {
    "first_reaction": "target_w2_return_pct",
    "midday_followup": "target_w3_return_pct",
    "closing_session": "target_w5_return_pct",
}


class StockReactionFeatureExtractor(BaseFeatureExtractor):
    """Extracts 8 microstructure feature clusters for BIST30 stock intraday reaction prediction.

    Feature Clusters:
        1. BofA W1 Execution Signal       — net flow TL, volume share, direction, velocity
        2. Multi-Broker W1 Alignment      — Top-5 domestic + TRA W1 net flows, imbalance vs BofA
        3. T-1 Stock Momentum             — adj_daily_return_pct 1d/5d/20d, adj_close vs SMA20
        4. T-1 FIFO Inventory Posture     — open_stock_quantity, fifo_avg_cost, unrealized_pnl_tl
        5. T-1 Multi-Day Accumulation     — 5d/20d rolling BofA net flow, Z-score vs 20d mean
        6. Sector Breadth & Peer Spread   — sector daily return 1d, sector BofA flow vs peer median
        7. Macro & Carry Context          — TCMB repo rate, days since last rate change, carry cost
        8. Calendar & Temporal            — day_of_week, is_monday, is_friday, days_to_settlement
    """

    CLUSTER_NAMES = [
        "bofa_w1_execution",
        "multi_broker_w1_alignment",
        "stock_momentum",
        "fifo_inventory",
        "multi_day_accumulation",
        "sector_breadth",
        "macro_carry",
        "calendar",
    ]

    def __init__(
        self,
        symbol: str,
        db: Optional[DuckDBManager] = None,
        lookback_months: Optional[int] = None,
        tracked_brokers: Optional[List[str]] = None,
    ):
        self.symbol = symbol.upper()
        self.db = db or DuckDBManager(read_only=True)
        self.settings = get_settings()
        cfg = self.settings.get_model_config("stock_reaction") or {}
        self.lookback_months = lookback_months if lookback_months is not None else cfg.get("lookback_months", 12)
        self.tracked_brokers = tracked_brokers or cfg.get("fifo_brokers", DEFAULT_TRACKED_BROKERS)
        # Brokers other than BofA used for W1 alignment features
        self.competitor_brokers = [b for b in self.tracked_brokers if b != "MLB"]

    def extract_features(
        self,
        start_date: Optional[date] = None,
        end_date: Optional[date] = None,
    ) -> pl.DataFrame:
        """Extract 8-cluster feature matrix joined with W2/W3/W5 return targets.

        Args:
            start_date: Earliest trade_date to include (for walk-forward windowing).
            end_date:   Latest trade_date to include (strict T-1 cutoff enforced internally).

        Returns:
            pl.DataFrame with columns: trade_date, symbol, feat_*, target_w2_return_pct,
            target_w3_return_pct, target_w5_return_pct.
        """
        conn = self.db.get_connection()

        # Build competitor broker IN clause for SQL
        comp_brokers_str = ", ".join(f"'{b}'" for b in self.competitor_brokers)
        tracked_brokers_str = ", ".join(f"'{b}'" for b in self.tracked_brokers)

        date_filter = f"AND w1_bofa.trade_date >= '{start_date}'" if start_date else ""
        end_filter = f"AND w1_bofa.trade_date <= '{end_date}'" if end_date else ""

        query = f"""
            WITH

            -- ── CLUSTER 1: BofA W1 Execution Signal ────────────────────────────────────────────
            w1_bofa AS (
                SELECT
                    trade_date,
                    symbol,
                    buy_volume                                              AS feat_bofa_w1_buy_vol,
                    sell_volume                                             AS feat_bofa_w1_sell_vol,
                    buy_turnover_tl                                         AS feat_bofa_w1_buy_tl,
                    sell_turnover_tl                                        AS feat_bofa_w1_sell_tl,
                    COALESCE(buy_turnover_tl, 0) - COALESCE(sell_turnover_tl, 0)
                                                                            AS feat_bofa_w1_net_flow_tl,
                    COALESCE(buy_volume, 0) - COALESCE(sell_volume, 0)     AS feat_bofa_w1_net_vol,
                    CASE
                        WHEN total_volume > 0
                        THEN (COALESCE(buy_volume,0) + COALESCE(sell_volume,0)) / total_volume
                        ELSE 0
                    END                                                     AS feat_bofa_w1_vol_share,
                    CASE
                        WHEN COALESCE(buy_turnover_tl,0) > COALESCE(sell_turnover_tl,0) THEN 1.0
                        WHEN COALESCE(buy_turnover_tl,0) < COALESCE(sell_turnover_tl,0) THEN -1.0
                        ELSE 0.0
                    END                                                     AS feat_bofa_w1_direction_sign,
                    CASE
                        WHEN total_volume > 0
                        THEN (buy_turnover_tl + sell_turnover_tl) / total_volume
                        ELSE NULL
                    END                                                     AS feat_bofa_w1_market_vwap
                FROM silver_intraday_broker_window_summary
                WHERE window_name = 'day_start'
                  AND broker_id = 'MLB'
                  AND symbol = '{self.symbol}'
                  {date_filter}
                  {end_filter}
            ),

            -- ── CLUSTER 2: Multi-Broker W1 Alignment ────────────────────────────────────────────
            w1_competitors AS (
                SELECT
                    trade_date,
                    symbol,
                    SUM(CASE WHEN broker_id IN ({comp_brokers_str}) THEN COALESCE(buy_turnover_tl,0) - COALESCE(sell_turnover_tl,0) ELSE 0 END)
                                                                            AS feat_comp_w1_net_flow_tl,
                    SUM(CASE WHEN broker_id = 'IYM' THEN COALESCE(buy_turnover_tl,0) - COALESCE(sell_turnover_tl,0) ELSE 0 END)
                                                                            AS feat_iym_w1_net_flow_tl,
                    SUM(CASE WHEN broker_id = 'YKR' THEN COALESCE(buy_turnover_tl,0) - COALESCE(sell_turnover_tl,0) ELSE 0 END)
                                                                            AS feat_ykr_w1_net_flow_tl,
                    SUM(CASE WHEN broker_id = 'AKM' THEN COALESCE(buy_turnover_tl,0) - COALESCE(sell_turnover_tl,0) ELSE 0 END)
                                                                            AS feat_akm_w1_net_flow_tl,
                    SUM(CASE WHEN broker_id = 'GRM' THEN COALESCE(buy_turnover_tl,0) - COALESCE(sell_turnover_tl,0) ELSE 0 END)
                                                                            AS feat_grm_w1_net_flow_tl,
                    SUM(CASE WHEN broker_id = 'ZRY' THEN COALESCE(buy_turnover_tl,0) - COALESCE(sell_turnover_tl,0) ELSE 0 END)
                                                                            AS feat_zry_w1_net_flow_tl,
                    SUM(CASE WHEN broker_id = 'TRA' THEN COALESCE(buy_turnover_tl,0) - COALESCE(sell_turnover_tl,0) ELSE 0 END)
                                                                            AS feat_tra_w1_net_flow_tl
                FROM silver_intraday_broker_window_summary
                WHERE window_name = 'day_start'
                  AND broker_id IN ({tracked_brokers_str})
                  AND symbol = '{self.symbol}'
                GROUP BY trade_date, symbol
            ),

            -- ── CLUSTER 3: T-1 Stock Momentum ────────────────────────────────────────────────────
            stock_momentum AS (
                SELECT
                    stk.trade_date,
                    stk.symbol,
                    stk.adj_daily_return_pct                                AS feat_stock_ret_1d,
                    -- 5-day rolling return (from T-5 close to T-1 close)
                    stk.adj_daily_return_pct
                        + LAG(stk.adj_daily_return_pct, 1) OVER (PARTITION BY stk.symbol ORDER BY stk.trade_date)
                        + LAG(stk.adj_daily_return_pct, 2) OVER (PARTITION BY stk.symbol ORDER BY stk.trade_date)
                        + LAG(stk.adj_daily_return_pct, 3) OVER (PARTITION BY stk.symbol ORDER BY stk.trade_date)
                        + LAG(stk.adj_daily_return_pct, 4) OVER (PARTITION BY stk.symbol ORDER BY stk.trade_date)
                                                                            AS feat_stock_ret_5d,
                    -- 20-day rolling return
                    stk.adj_close_price / NULLIF(
                        LAG(stk.adj_close_price, 20) OVER (PARTITION BY stk.symbol ORDER BY stk.trade_date), 0
                    ) * 100 - 100                                           AS feat_stock_ret_20d,
                    -- Distance from 20-day SMA (mean reversion signal)
                    stk.adj_close_price / NULLIF(
                        AVG(stk.adj_close_price) OVER (
                            PARTITION BY stk.symbol
                            ORDER BY stk.trade_date
                            ROWS BETWEEN 19 PRECEDING AND CURRENT ROW
                        ), 0
                    ) * 100 - 100                                           AS feat_stock_dist_sma20_pct,
                    -- Daily volatility proxy (20-day rolling stdev of returns)
                    STDDEV_POP(stk.adj_daily_return_pct) OVER (
                        PARTITION BY stk.symbol
                        ORDER BY stk.trade_date
                        ROWS BETWEEN 19 PRECEDING AND CURRENT ROW
                    )                                                       AS feat_stock_vol_20d,
                    -- Open-to-T1-close return (gap measure)
                    CASE
                        WHEN stk.open_price > 0
                        THEN (stk.adj_close_price - stk.open_price) / stk.open_price * 100
                        ELSE NULL
                    END                                                     AS feat_stock_intraday_range_pct
                FROM silver_daily_stock_summary stk
                WHERE stk.symbol = '{self.symbol}'
            ),

            -- ── CLUSTER 4: T-1 FIFO Inventory Posture ────────────────────────────────────────────
            fifo_posture AS (
                SELECT
                    f.trade_date,
                    f.broker_id,
                    f.symbol,
                    COALESCE(f.open_stock_quantity, 0)                      AS feat_fifo_open_qty,
                    COALESCE(f.fifo_avg_cost, 0)                            AS feat_fifo_avg_cost,
                    COALESCE(f.unrealized_pnl_tl, 0)                        AS feat_fifo_unrealized_pnl_tl,
                    COALESCE(f.market_close_price, 0)                       AS feat_fifo_close_price,
                    COALESCE(f.cumulative_realized_pnl_tl, 0)               AS feat_fifo_cum_realized_pnl_tl,
                    -- Cost basis spread: how far close is from average buy cost (%)
                    CASE
                        WHEN f.fifo_avg_cost > 0
                        THEN (f.market_close_price - f.fifo_avg_cost) / f.fifo_avg_cost * 100
                        ELSE 0
                    END                                                     AS feat_fifo_cost_basis_spread_pct
                FROM silver_broker_fifo_daily f
                WHERE f.symbol = '{self.symbol}'
                  AND f.broker_id IN ({tracked_brokers_str})
            ),

            -- FIFO pivot: one row per (trade_date, symbol) with per-broker columns
            fifo_pivot AS (
                SELECT
                    trade_date,
                    symbol,
                    SUM(CASE WHEN broker_id = 'MLB' THEN feat_fifo_open_qty ELSE 0 END)
                                                                            AS feat_bofa_t1_open_qty,
                    SUM(CASE WHEN broker_id = 'MLB' THEN feat_fifo_cost_basis_spread_pct ELSE 0 END)
                                                                            AS feat_bofa_t1_cost_spread_pct,
                    SUM(CASE WHEN broker_id = 'MLB' THEN feat_fifo_unrealized_pnl_tl ELSE 0 END)
                                                                            AS feat_bofa_t1_unrealized_pnl_tl,
                    SUM(CASE WHEN broker_id = 'TRA' THEN feat_fifo_open_qty ELSE 0 END)
                                                                            AS feat_tra_t1_open_qty,
                    SUM(CASE WHEN broker_id = 'TRA' THEN feat_fifo_cost_basis_spread_pct ELSE 0 END)
                                                                            AS feat_tra_t1_cost_spread_pct,
                    -- Aggregate domestic big-5 open inventory
                    SUM(CASE WHEN broker_id IN ('IYM','YKR','AKM','GRM','ZRY') THEN feat_fifo_open_qty ELSE 0 END)
                                                                            AS feat_dom5_t1_open_qty,
                    SUM(CASE WHEN broker_id IN ('IYM','YKR','AKM','GRM','ZRY') THEN feat_fifo_unrealized_pnl_tl ELSE 0 END)
                                                                            AS feat_dom5_t1_unrealized_pnl_tl
                FROM fifo_posture
                GROUP BY trade_date, symbol
            ),

            -- ── CLUSTER 5: Multi-Day Accumulation ────────────────────────────────────────────────
            daily_broker_stock AS (
                SELECT
                    dbs.trade_date,
                    dbs.symbol,
                    SUM(CASE WHEN dbs.broker_id = 'MLB' THEN COALESCE(dbs.buy_turnover_tl,0) - COALESCE(dbs.sell_turnover_tl,0) ELSE 0 END)
                                                                            AS bofa_daily_net_flow_tl,
                    SUM(CASE WHEN dbs.broker_id IN ({comp_brokers_str}) THEN COALESCE(dbs.buy_turnover_tl,0) - COALESCE(dbs.sell_turnover_tl,0) ELSE 0 END)
                                                                            AS comp_daily_net_flow_tl
                FROM silver_daily_broker_summary dbs
                WHERE dbs.symbol = '{self.symbol}'
                GROUP BY dbs.trade_date, dbs.symbol
            ),
            accumulation AS (
                SELECT
                    trade_date,
                    symbol,
                    SUM(bofa_daily_net_flow_tl) OVER (
                        PARTITION BY symbol ORDER BY trade_date
                        ROWS BETWEEN 4 PRECEDING AND CURRENT ROW
                    )                                                       AS feat_bofa_accum_5d_tl,
                    SUM(bofa_daily_net_flow_tl) OVER (
                        PARTITION BY symbol ORDER BY trade_date
                        ROWS BETWEEN 19 PRECEDING AND CURRENT ROW
                    )                                                       AS feat_bofa_accum_20d_tl,
                    -- Z-score of today's flow vs 20-day mean and stdev
                    (bofa_daily_net_flow_tl - AVG(bofa_daily_net_flow_tl) OVER (
                        PARTITION BY symbol ORDER BY trade_date
                        ROWS BETWEEN 19 PRECEDING AND CURRENT ROW
                    )) / NULLIF(STDDEV_POP(bofa_daily_net_flow_tl) OVER (
                        PARTITION BY symbol ORDER BY trade_date
                        ROWS BETWEEN 19 PRECEDING AND CURRENT ROW
                    ), 0)                                                   AS feat_bofa_flow_zscore_20d,
                    SUM(comp_daily_net_flow_tl) OVER (
                        PARTITION BY symbol ORDER BY trade_date
                        ROWS BETWEEN 4 PRECEDING AND CURRENT ROW
                    )                                                       AS feat_comp_accum_5d_tl,
                    -- Divergence: BofA buying while competitors selling (or vice versa)
                    bofa_daily_net_flow_tl - comp_daily_net_flow_tl        AS feat_bofa_comp_flow_delta_tl
                FROM daily_broker_stock
            ),

            -- ── CLUSTER 6: Sector Breadth & Peer Spread ───────────────────────────────────────────
            sector_breadth AS (
                SELECT
                    stk.trade_date,
                    stk.symbol,
                    stk.sector,
                    sect.daily_return_pct                                   AS feat_sector_ret_1d,
                    sect.bofa_sector_net_flow_tl                            AS feat_sector_bofa_flow_tl,
                    -- Peer spread: stock return minus sector return
                    stk.adj_daily_return_pct - sect.daily_return_pct        AS feat_peer_spread_1d,
                    sect.stock_count                                         AS feat_sector_stock_count
                FROM silver_daily_stock_summary stk
                LEFT JOIN silver_daily_sector_summary sect
                    ON stk.trade_date = sect.trade_date AND stk.sector = sect.sector
                WHERE stk.symbol = '{self.symbol}'
            ),

            -- ── CLUSTER 7: Macro & Carry Context ─────────────────────────────────────────────────
            macro AS (
                SELECT
                    trade_date,
                    weekly_repo_rate_pct                                    AS feat_macro_repo_rate,
                    rate_change_delta                                        AS feat_macro_rate_delta,
                    days_since_last_decision                                 AS feat_macro_days_since_decision,
                    daily_carry_cost_pct                                     AS feat_macro_daily_carry_pct,
                    rate_spread_vs_30d_mean                                  AS feat_macro_rate_spread_30d
                FROM silver_daily_macro_rates
            ),

            -- ── CLUSTER 8: Calendar ───────────────────────────────────────────────────────────────
            -- (derived inline from trade_date in final assembly)

            -- ── TARGET VARIABLES: W2/W3/W5 Return% ───────────────────────────────────────────────
            -- These are the SAME-DAY window returns.
            -- Zero-lookahead: they become the target y during training.
            -- During live inference, these columns are NULL for the upcoming forecast date.
            window_returns AS (
                SELECT
                    trade_date,
                    symbol,
                    -- W1 reference price: BofA buy VWAP, fallback market W1 VWAP, fallback open price
                    COALESCE(
                        NULLIF(MAX(CASE WHEN broker_id='MLB' AND buy_volume > 0
                            THEN buy_turnover_tl / buy_volume ELSE NULL END), 0),
                        NULLIF(SUM(buy_turnover_tl + sell_turnover_tl) / NULLIF(SUM(buy_volume + sell_volume), 0), 0)
                    )                                                        AS w1_ref_price,
                    MAX(CASE WHEN window_name = 'first_reaction' AND (buy_volume+sell_volume) > 0
                        THEN (buy_turnover_tl+sell_turnover_tl)/(buy_volume+sell_volume) ELSE NULL END)
                                                                             AS w2_vwap,
                    MAX(CASE WHEN window_name = 'midday_followup' AND (buy_volume+sell_volume) > 0
                        THEN (buy_turnover_tl+sell_turnover_tl)/(buy_volume+sell_volume) ELSE NULL END)
                                                                             AS w3_vwap,
                    MAX(CASE WHEN window_name = 'closing_session' AND (buy_volume+sell_volume) > 0
                        THEN (buy_turnover_tl+sell_turnover_tl)/(buy_volume+sell_volume) ELSE NULL END)
                                                                             AS w5_vwap
                FROM silver_intraday_broker_window_summary
                WHERE symbol = '{self.symbol}'
                  AND window_name IN ('day_start', 'first_reaction', 'midday_followup', 'closing_session')
                GROUP BY trade_date, symbol
            ),
            targets AS (
                SELECT
                    trade_date,
                    symbol,
                    CASE WHEN w1_ref_price > 0 AND w2_vwap IS NOT NULL
                        THEN (w2_vwap - w1_ref_price) / w1_ref_price * 100 ELSE NULL END
                                                                             AS target_w2_return_pct,
                    CASE WHEN w1_ref_price > 0 AND w3_vwap IS NOT NULL
                        THEN (w3_vwap - w1_ref_price) / w1_ref_price * 100 ELSE NULL END
                                                                             AS target_w3_return_pct,
                    CASE WHEN w1_ref_price > 0 AND w5_vwap IS NOT NULL
                        THEN (w5_vwap - w1_ref_price) / w1_ref_price * 100 ELSE NULL END
                                                                             AS target_w5_return_pct
                FROM window_returns
            )

            -- ── FINAL ASSEMBLY ────────────────────────────────────────────────────────────────────
            -- Join all clusters. T-1 lag is applied to all inventory/momentum clusters.
            -- W1 execution clusters (same-day) are zero-lag — available at 10:30 TRT.
            SELECT
                w1_bofa.trade_date,
                w1_bofa.symbol,

                -- Cluster 1: BofA W1
                w1_bofa.feat_bofa_w1_buy_vol,
                w1_bofa.feat_bofa_w1_sell_vol,
                w1_bofa.feat_bofa_w1_buy_tl,
                w1_bofa.feat_bofa_w1_sell_tl,
                w1_bofa.feat_bofa_w1_net_flow_tl,
                w1_bofa.feat_bofa_w1_net_vol,
                w1_bofa.feat_bofa_w1_vol_share,
                w1_bofa.feat_bofa_w1_direction_sign,
                w1_bofa.feat_bofa_w1_market_vwap,

                -- Cluster 2: Competitor W1 Alignment
                w1_comp.feat_comp_w1_net_flow_tl,
                w1_comp.feat_iym_w1_net_flow_tl,
                w1_comp.feat_ykr_w1_net_flow_tl,
                w1_comp.feat_akm_w1_net_flow_tl,
                w1_comp.feat_grm_w1_net_flow_tl,
                w1_comp.feat_zry_w1_net_flow_tl,
                w1_comp.feat_tra_w1_net_flow_tl,
                -- Alignment: BofA vs competitors (positive = aligned, negative = divergence)
                CASE
                    WHEN w1_bofa.feat_bofa_w1_net_flow_tl * COALESCE(w1_comp.feat_comp_w1_net_flow_tl, 0) > 0
                    THEN 1.0
                    WHEN w1_bofa.feat_bofa_w1_net_flow_tl * COALESCE(w1_comp.feat_comp_w1_net_flow_tl, 0) < 0
                    THEN -1.0
                    ELSE 0.0
                END                                                         AS feat_w1_bofa_comp_alignment,
                -- TRA contra-sign: if TRA selling while BofA buying = retail panic exit (bullish)
                CASE
                    WHEN w1_bofa.feat_bofa_w1_net_flow_tl > 0 AND COALESCE(w1_comp.feat_tra_w1_net_flow_tl, 0) < 0
                    THEN 1.0
                    WHEN w1_bofa.feat_bofa_w1_net_flow_tl < 0 AND COALESCE(w1_comp.feat_tra_w1_net_flow_tl, 0) > 0
                    THEN -1.0
                    ELSE 0.0
                END                                                         AS feat_w1_bofa_tra_contra_signal,

                -- Cluster 3: T-1 Stock Momentum (lagged by 1 day — T-1 values)
                LAG(sm.feat_stock_ret_1d, 1) OVER (PARTITION BY w1_bofa.symbol ORDER BY w1_bofa.trade_date)
                                                                            AS feat_stock_ret_t1_1d,
                LAG(sm.feat_stock_ret_5d, 1) OVER (PARTITION BY w1_bofa.symbol ORDER BY w1_bofa.trade_date)
                                                                            AS feat_stock_ret_t1_5d,
                LAG(sm.feat_stock_ret_20d, 1) OVER (PARTITION BY w1_bofa.symbol ORDER BY w1_bofa.trade_date)
                                                                            AS feat_stock_ret_t1_20d,
                LAG(sm.feat_stock_dist_sma20_pct, 1) OVER (PARTITION BY w1_bofa.symbol ORDER BY w1_bofa.trade_date)
                                                                            AS feat_stock_dist_sma20_t1,
                LAG(sm.feat_stock_vol_20d, 1) OVER (PARTITION BY w1_bofa.symbol ORDER BY w1_bofa.trade_date)
                                                                            AS feat_stock_vol_20d_t1,
                LAG(sm.feat_stock_intraday_range_pct, 1) OVER (PARTITION BY w1_bofa.symbol ORDER BY w1_bofa.trade_date)
                                                                            AS feat_stock_intraday_range_t1,

                -- Cluster 4: T-1 FIFO Inventory (lagged — yesterday's end-of-day open positions)
                LAG(fp.feat_bofa_t1_open_qty, 1) OVER (PARTITION BY w1_bofa.symbol ORDER BY w1_bofa.trade_date)
                                                                            AS feat_bofa_t1_open_qty,
                LAG(fp.feat_bofa_t1_cost_spread_pct, 1) OVER (PARTITION BY w1_bofa.symbol ORDER BY w1_bofa.trade_date)
                                                                            AS feat_bofa_t1_cost_spread_pct,
                LAG(fp.feat_bofa_t1_unrealized_pnl_tl, 1) OVER (PARTITION BY w1_bofa.symbol ORDER BY w1_bofa.trade_date)
                                                                            AS feat_bofa_t1_unrealized_pnl_tl,
                LAG(fp.feat_tra_t1_open_qty, 1) OVER (PARTITION BY w1_bofa.symbol ORDER BY w1_bofa.trade_date)
                                                                            AS feat_tra_t1_open_qty,
                LAG(fp.feat_tra_t1_cost_spread_pct, 1) OVER (PARTITION BY w1_bofa.symbol ORDER BY w1_bofa.trade_date)
                                                                            AS feat_tra_t1_cost_spread_pct,
                LAG(fp.feat_dom5_t1_open_qty, 1) OVER (PARTITION BY w1_bofa.symbol ORDER BY w1_bofa.trade_date)
                                                                            AS feat_dom5_t1_open_qty,
                LAG(fp.feat_dom5_t1_unrealized_pnl_tl, 1) OVER (PARTITION BY w1_bofa.symbol ORDER BY w1_bofa.trade_date)
                                                                            AS feat_dom5_t1_unrealized_pnl_tl,

                -- Cluster 5: Multi-Day Accumulation (T-1 rolling window — already lagged in accumulation CTE)
                LAG(acc.feat_bofa_accum_5d_tl, 1) OVER (PARTITION BY w1_bofa.symbol ORDER BY w1_bofa.trade_date)
                                                                            AS feat_bofa_accum_5d_t1_tl,
                LAG(acc.feat_bofa_accum_20d_tl, 1) OVER (PARTITION BY w1_bofa.symbol ORDER BY w1_bofa.trade_date)
                                                                            AS feat_bofa_accum_20d_t1_tl,
                LAG(acc.feat_bofa_flow_zscore_20d, 1) OVER (PARTITION BY w1_bofa.symbol ORDER BY w1_bofa.trade_date)
                                                                            AS feat_bofa_flow_zscore_t1,
                LAG(acc.feat_comp_accum_5d_tl, 1) OVER (PARTITION BY w1_bofa.symbol ORDER BY w1_bofa.trade_date)
                                                                            AS feat_comp_accum_5d_t1_tl,
                LAG(acc.feat_bofa_comp_flow_delta_tl, 1) OVER (PARTITION BY w1_bofa.symbol ORDER BY w1_bofa.trade_date)
                                                                            AS feat_bofa_comp_delta_t1_tl,

                -- Cluster 6: Sector Breadth (T-1 — yesterday's sector-level data)
                LAG(sb.feat_sector_ret_1d, 1) OVER (PARTITION BY w1_bofa.symbol ORDER BY w1_bofa.trade_date)
                                                                            AS feat_sector_ret_t1,
                LAG(sb.feat_sector_bofa_flow_tl, 1) OVER (PARTITION BY w1_bofa.symbol ORDER BY w1_bofa.trade_date)
                                                                            AS feat_sector_bofa_flow_t1,
                LAG(sb.feat_peer_spread_1d, 1) OVER (PARTITION BY w1_bofa.symbol ORDER BY w1_bofa.trade_date)
                                                                            AS feat_peer_spread_t1,

                -- Cluster 7: Macro Context (T-1 — yesterday's settled macro rates)
                LAG(mc.feat_macro_repo_rate, 1) OVER (PARTITION BY w1_bofa.symbol ORDER BY w1_bofa.trade_date)
                                                                            AS feat_macro_repo_rate_t1,
                LAG(mc.feat_macro_rate_delta, 1) OVER (PARTITION BY w1_bofa.symbol ORDER BY w1_bofa.trade_date)
                                                                            AS feat_macro_rate_delta_t1,
                LAG(mc.feat_macro_days_since_decision, 1) OVER (PARTITION BY w1_bofa.symbol ORDER BY w1_bofa.trade_date)
                                                                            AS feat_macro_days_since_decision_t1,
                LAG(mc.feat_macro_daily_carry_pct, 1) OVER (PARTITION BY w1_bofa.symbol ORDER BY w1_bofa.trade_date)
                                                                            AS feat_macro_carry_t1,

                -- Cluster 8: Calendar (same-day — always available at session start)
                DAYOFWEEK(w1_bofa.trade_date)                               AS feat_day_of_week,
                CASE WHEN DAYOFWEEK(w1_bofa.trade_date) = 2 THEN 1 ELSE 0 END
                                                                            AS feat_is_monday,
                CASE WHEN DAYOFWEEK(w1_bofa.trade_date) = 6 THEN 1 ELSE 0 END
                                                                            AS feat_is_friday,
                DAYOFMONTH(w1_bofa.trade_date)                              AS feat_day_of_month,

                -- Target variables (NULL for the live inference date — filled in by reconcile)
                tgt.target_w2_return_pct,
                tgt.target_w3_return_pct,
                tgt.target_w5_return_pct

            FROM w1_bofa
            LEFT JOIN w1_competitors w1_comp
                ON w1_bofa.trade_date = w1_comp.trade_date AND w1_bofa.symbol = w1_comp.symbol
            LEFT JOIN stock_momentum sm
                ON w1_bofa.trade_date = sm.trade_date AND w1_bofa.symbol = sm.symbol
            LEFT JOIN fifo_pivot fp
                ON w1_bofa.trade_date = fp.trade_date AND w1_bofa.symbol = fp.symbol
            LEFT JOIN accumulation acc
                ON w1_bofa.trade_date = acc.trade_date AND w1_bofa.symbol = acc.symbol
            LEFT JOIN sector_breadth sb
                ON w1_bofa.trade_date = sb.trade_date AND w1_bofa.symbol = sb.symbol
            LEFT JOIN macro mc
                ON w1_bofa.trade_date = mc.trade_date
            LEFT JOIN targets tgt
                ON w1_bofa.trade_date = tgt.trade_date AND w1_bofa.symbol = tgt.symbol

            ORDER BY w1_bofa.trade_date ASC;
        """

        df = conn.execute(query).pl()
        logger.info(
            f"[{self.symbol}] Extracted {len(df):,} rows × {len(df.columns)} columns "
            f"(feature clusters: 8, lookback: {self.lookback_months}m)"
        )
        return df

    def get_feature_columns(self) -> List[str]:
        """Return list of all engineered feature column names (excludes trade_date, symbol, target_*)."""
        return [
            # Cluster 1: BofA W1
            "feat_bofa_w1_buy_vol", "feat_bofa_w1_sell_vol", "feat_bofa_w1_buy_tl",
            "feat_bofa_w1_sell_tl", "feat_bofa_w1_net_flow_tl", "feat_bofa_w1_net_vol",
            "feat_bofa_w1_vol_share", "feat_bofa_w1_direction_sign", "feat_bofa_w1_market_vwap",
            # Cluster 2: Multi-broker W1 alignment
            "feat_comp_w1_net_flow_tl", "feat_iym_w1_net_flow_tl", "feat_ykr_w1_net_flow_tl",
            "feat_akm_w1_net_flow_tl", "feat_grm_w1_net_flow_tl", "feat_zry_w1_net_flow_tl",
            "feat_tra_w1_net_flow_tl", "feat_w1_bofa_comp_alignment", "feat_w1_bofa_tra_contra_signal",
            # Cluster 3: T-1 stock momentum
            "feat_stock_ret_t1_1d", "feat_stock_ret_t1_5d", "feat_stock_ret_t1_20d",
            "feat_stock_dist_sma20_t1", "feat_stock_vol_20d_t1", "feat_stock_intraday_range_t1",
            # Cluster 4: FIFO inventory
            "feat_bofa_t1_open_qty", "feat_bofa_t1_cost_spread_pct", "feat_bofa_t1_unrealized_pnl_tl",
            "feat_tra_t1_open_qty", "feat_tra_t1_cost_spread_pct",
            "feat_dom5_t1_open_qty", "feat_dom5_t1_unrealized_pnl_tl",
            # Cluster 5: Multi-day accumulation
            "feat_bofa_accum_5d_t1_tl", "feat_bofa_accum_20d_t1_tl", "feat_bofa_flow_zscore_t1",
            "feat_comp_accum_5d_t1_tl", "feat_bofa_comp_delta_t1_tl",
            # Cluster 6: Sector breadth
            "feat_sector_ret_t1", "feat_sector_bofa_flow_t1", "feat_peer_spread_t1",
            # Cluster 7: Macro carry
            "feat_macro_repo_rate_t1", "feat_macro_rate_delta_t1",
            "feat_macro_days_since_decision_t1", "feat_macro_carry_t1",
            # Cluster 8: Calendar
            "feat_day_of_week", "feat_is_monday", "feat_is_friday", "feat_day_of_month",
        ]

    def get_target_column(self, window: str) -> str:
        """Return target column name for a given window identifier."""
        mapping = {
            "w2": "target_w2_return_pct",
            "w3": "target_w3_return_pct",
            "w5": "target_w5_return_pct",
            "first_reaction": "target_w2_return_pct",
            "midday_followup": "target_w3_return_pct",
            "closing_session": "target_w5_return_pct",
        }
        if window not in mapping:
            raise ValueError(f"Unknown window '{window}'. Valid: {list(mapping.keys())}")
        return mapping[window]
