"""Day-Start Feature Extraction Engine: Assembles the 7 Feature Clusters from Silver fact tables."""

from datetime import date
from typing import Optional

import polars as pl

from mdk_trading_oracle.core.config import get_settings
from mdk_trading_oracle.core.db import DuckDBManager
from mdk_trading_oracle.core.logger import get_logger
from mdk_trading_oracle.models.base import BaseFeatureExtractor

logger = get_logger("mdk_oracle.models.day_start.features")


class DayStartFeatureExtractor(BaseFeatureExtractor):
    """Extracts, engineers, and temporal-aligns the 7 Feature Clusters at T-1 Close to predict T_open (Window 1).
    
    Clusters:
        1. Prior Closing Window Momentum (Window 4 net flow & acceleration)
        2. Multi-Day Inventory & Sector Saturation (5d/20d rolling flows & Z-scores)
        3. Institutional Cost Basis & Unrealized PnL (Close vs 20d Buy VWAP spread)
        4. Top-5 Competitor Closing Posture & Imbalance (Domestic banks W4 flow & delta)
        5. Competitor Alignment & Institutional Hegemony (Combined market share)
        6. Sector Cross-Sectional Stress & Crash Flags (Daily returns & breadth)
        7. Calendar & Temporal Seasonality (is_monday, is_friday, day_of_week)
    """

    def __init__(
        self,
        db: Optional[DuckDBManager] = None,
        target_broker_id: str = "MLB",
        lookback_months: Optional[int] = None,
    ):
        self.db = db or DuckDBManager(read_only=True)
        self.target_broker = target_broker_id
        self.settings = get_settings()
        cfg = self.settings.get_model_config("day_start")
        self.lookback_months = lookback_months if lookback_months is not None else cfg.get("lookback_months", 12)

    def extract_features(self, start_date: Optional[date] = None, end_date: Optional[date] = None) -> pl.DataFrame:
        """Extract multi-cluster feature matrix from DuckDB Silver tables with optional lookback filtering.
        
        Args:
            start_date: Optional filter for minimum trade date.
            end_date: Optional filter for maximum trade date.
            
        Returns:
            pl.DataFrame: Clean dataset where each row is a trading day T with features computed 
                          strictly from historical data up to T-1 Close, paired with actual Window 1 target outcomes on Day T.
        """
        conn = self.db.get_connection()
        logger.info(f"Extracting Day-Start 7 Feature Clusters for broker '{self.target_broker}' (lookback_months={self.lookback_months})...")

        # Determine effective start_date from lookback_months if not explicitly given
        effective_start = start_date
        if effective_start is None and self.lookback_months is not None:
            max_d_res = conn.execute("SELECT MAX(trade_date) FROM silver_daily_stock_summary;").fetchone()
            if max_d_res and max_d_res[0]:
                from dateutil.relativedelta import relativedelta
                effective_start = max_d_res[0] - relativedelta(months=self.lookback_months)

        query = f"""
            WITH daily_dates AS (
                SELECT DISTINCT trade_date
                FROM silver_daily_stock_summary
                ORDER BY trade_date ASC
            ),
            -- 1. Target Variables on Day T (Window 1: day_start)
            day_t_targets AS (
                SELECT 
                    trade_date,
                    SUM(CASE WHEN broker_id = '{self.target_broker}' THEN net_flow_tl ELSE 0.0 END) AS target_open_net_flow_tl
                FROM silver_intraday_broker_window_summary
                WHERE window_name = 'day_start'
                GROUP BY trade_date
            ),
            -- 2. Prior Day Closing Window 4 Flow (T-1)
            prev_day_w4_flows AS (
                SELECT 
                    trade_date,
                    SUM(CASE WHEN broker_id = '{self.target_broker}' THEN net_flow_tl ELSE 0.0 END) AS bofa_w4_net_flow_tl,
                    SUM(CASE WHEN broker_id = '{self.target_broker}' THEN total_turnover_tl ELSE 0.0 END) AS bofa_w4_turnover_tl,
                    SUM(CASE WHEN broker_id IN ('IYM', 'YKR', 'AKM', 'GRM', 'ZRY') THEN net_flow_tl ELSE 0.0 END) AS top5_domestic_w4_net_flow_tl,
                    SUM(CASE WHEN broker_id IN ('IYM', 'YKR', 'AKM', 'GRM', 'ZRY') THEN total_turnover_tl ELSE 0.0 END) AS top5_domestic_w4_turnover_tl
                FROM silver_intraday_broker_window_summary
                WHERE window_name = 'closing_session'
                GROUP BY trade_date
            ),
            -- 3. Prior Day Full Macro Overview (T-1)
            prev_day_broker_overview AS (
                SELECT 
                    trade_date,
                    MAX(CASE WHEN broker_id = '{self.target_broker}' THEN net_flow_tl ELSE NULL END) AS bofa_prev_day_net_flow_tl,
                    MAX(CASE WHEN broker_id = '{self.target_broker}' THEN total_turnover_tl ELSE NULL END) AS bofa_prev_day_turnover_tl,
                    MAX(CASE WHEN broker_id = '{self.target_broker}' THEN market_turnover_share ELSE NULL END) AS bofa_prev_day_market_share,
                    MAX(CASE WHEN broker_id = '{self.target_broker}' THEN market_turnover_rank ELSE NULL END) AS bofa_prev_day_turnover_rank,
                    SUM(CASE WHEN broker_id IN ('IYM', 'YKR', 'AKM', 'GRM', 'ZRY') THEN net_flow_tl ELSE 0.0 END) AS top5_domestic_prev_day_net_flow_tl,
                    SUM(CASE WHEN broker_id IN ('IYM', 'YKR', 'AKM', 'GRM', 'ZRY') THEN total_turnover_tl ELSE 0.0 END) AS top5_domestic_prev_day_turnover_tl,
                    SUM(total_turnover_tl) / 2.0 AS market_total_turnover_tl
                FROM silver_daily_broker_overview
                GROUP BY trade_date
            ),
            -- 4. Prior Day Market Aggregates (T-1 Close, VWAP, Returns)
            prev_day_market_summary AS (
                SELECT 
                    trade_date,
                    AVG(daily_return_pct) AS market_avg_return_pct,
                    AVG(price_range_pct) AS market_avg_range_pct,
                    SUM(total_turnover_tl) AS total_market_turnover_tl,
                    AVG(top_5_concentration_ratio) AS avg_cr5_concentration,
                    SUM(bofa_net_flow_tl) AS market_bofa_net_flow_tl,
                    -- Cost Basis aggregations
                    SUM(bofa_buy_turnover_tl) AS bofa_total_buy_turnover_tl,
                    SUM(CASE WHEN bofa_buy_vwap > 0 THEN bofa_buy_vwap * total_volume ELSE 0.0 END) / 
                        NULLIF(SUM(CASE WHEN bofa_buy_vwap > 0 THEN total_volume ELSE 0.0 END), 0.0) AS bofa_daily_buy_vwap,
                    AVG(close_price) AS market_avg_close_price,
                    AVG(market_vwap) AS market_avg_vwap
                FROM silver_daily_stock_summary
                GROUP BY trade_date
            ),
            -- 5. Prior Day Sector Flows (T-1)
            prev_day_sector_flows AS (
                SELECT 
                    trade_date,
                    SUM(CASE WHEN sector = 'Banking' AND broker_id = '{self.target_broker}' THEN net_flow_tl ELSE 0.0 END) AS bofa_banking_flow_prev_day,
                    SUM(CASE WHEN sector = 'Transportation' AND broker_id = '{self.target_broker}' THEN net_flow_tl ELSE 0.0 END) AS bofa_transport_flow_prev_day,
                    SUM(CASE WHEN sector = 'Holding' AND broker_id = '{self.target_broker}' THEN net_flow_tl ELSE 0.0 END) AS bofa_holding_flow_prev_day,
                    SUM(CASE WHEN sector = 'Energy & Refining' AND broker_id = '{self.target_broker}' THEN net_flow_tl ELSE 0.0 END) AS bofa_energy_flow_prev_day,
                    SUM(CASE WHEN sector = 'Defense & Tech' AND broker_id = '{self.target_broker}' THEN net_flow_tl ELSE 0.0 END) AS bofa_defense_flow_prev_day,
                    SUM(CASE WHEN sector = 'Banking' AND broker_id IN ('IYM', 'YKR', 'AKM', 'GRM', 'ZRY') THEN net_flow_tl ELSE 0.0 END) AS top5_banking_flow_prev_day
                FROM silver_daily_sector_summary
                GROUP BY trade_date
            ),
            -- Combine Prior Day T-1 Features
            daily_feature_base AS (
                SELECT 
                    d.trade_date,
                    EXTRACT(DOW FROM d.trade_date) AS day_of_week,
                    (EXTRACT(DOW FROM d.trade_date) = 1) AS is_monday,
                    (EXTRACT(DOW FROM d.trade_date) = 5) AS is_friday,
                    -- Cluster 1: Closing Window Momentum (W4)
                    COALESCE(w4.bofa_w4_net_flow_tl, 0.0) AS bofa_w4_net_flow_tl,
                    COALESCE(w4.bofa_w4_turnover_tl, 0.0) AS bofa_w4_turnover_tl,
                    COALESCE(w4.bofa_w4_net_flow_tl, 0.0) / 
                        NULLIF(ABS(COALESCE(bo.bofa_prev_day_net_flow_tl, 0.0)), 0.0) AS w4_flow_acceleration_ratio,
                    -- Cluster 2 & 4: Macro & Competitor Overview
                    COALESCE(bo.bofa_prev_day_net_flow_tl, 0.0) AS bofa_prev_day_net_flow_tl,
                    COALESCE(bo.bofa_prev_day_turnover_tl, 0.0) AS bofa_prev_day_turnover_tl,
                    COALESCE(bo.bofa_prev_day_market_share, 0.0) AS bofa_prev_day_market_share,
                    COALESCE(bo.bofa_prev_day_turnover_rank, 10) AS bofa_prev_day_turnover_rank,
                    COALESCE(w4.top5_domestic_w4_net_flow_tl, 0.0) AS top5_domestic_w4_net_flow_tl,
                    COALESCE(bo.top5_domestic_prev_day_net_flow_tl, 0.0) AS top5_domestic_prev_day_net_flow_tl,
                    -- Imbalance / Divergence Delta
                    COALESCE(w4.bofa_w4_net_flow_tl, 0.0) - COALESCE(w4.top5_domestic_w4_net_flow_tl, 0.0) AS bofa_vs_top5_w4_flow_delta_tl,
                    COALESCE(bo.bofa_prev_day_net_flow_tl, 0.0) - COALESCE(bo.top5_domestic_prev_day_net_flow_tl, 0.0) AS bofa_vs_top5_total_flow_delta_tl,
                    -- Cluster 5: Institutional Hegemony
                    (COALESCE(bo.bofa_prev_day_turnover_tl, 0.0) + COALESCE(bo.top5_domestic_prev_day_turnover_tl, 0.0)) / 
                        NULLIF(COALESCE(ms.total_market_turnover_tl, 1.0), 0.0) AS institutional_hegemony_share,
                    COALESCE(ms.avg_cr5_concentration, 0.0) AS avg_cr5_concentration,
                    -- Cluster 3 & 6: Market Returns & Cost Basis
                    COALESCE(ms.market_avg_return_pct, 0.0) AS market_avg_return_pct,
                    COALESCE(ms.market_avg_range_pct, 0.0) AS market_avg_range_pct,
                    (COALESCE(ms.market_avg_close_price, 0.0) - COALESCE(ms.market_avg_vwap, 0.0)) / 
                        NULLIF(COALESCE(ms.market_avg_vwap, 1.0), 0.0) AS prev_day_close_vs_vwap_spread_pct,
                    COALESCE(ms.bofa_daily_buy_vwap, 0.0) AS bofa_daily_buy_vwap,
                    COALESCE(ms.market_avg_close_price, 0.0) AS market_avg_close_price,
                    -- Cluster 2: Sector specific flows
                    COALESCE(sf.bofa_banking_flow_prev_day, 0.0) AS bofa_banking_flow_prev_day,
                    COALESCE(sf.bofa_transport_flow_prev_day, 0.0) AS bofa_transport_flow_prev_day,
                    COALESCE(sf.bofa_holding_flow_prev_day, 0.0) AS bofa_holding_flow_prev_day,
                    COALESCE(sf.bofa_energy_flow_prev_day, 0.0) AS bofa_energy_flow_prev_day,
                    COALESCE(sf.bofa_defense_flow_prev_day, 0.0) AS bofa_defense_flow_prev_day,
                    COALESCE(sf.top5_banking_flow_prev_day, 0.0) AS top5_banking_flow_prev_day
                FROM daily_dates d
                LEFT JOIN prev_day_w4_flows w4 ON d.trade_date = w4.trade_date
                LEFT JOIN prev_day_broker_overview bo ON d.trade_date = bo.trade_date
                LEFT JOIN prev_day_market_summary ms ON d.trade_date = ms.trade_date
                LEFT JOIN prev_day_sector_flows sf ON d.trade_date = sf.trade_date
            ),
            -- Intermediate Rolling Multi-Day Signals (Unlagged)
            unlagged_rolling AS (
                SELECT 
                    *,
                    SUM(bofa_prev_day_net_flow_tl) OVER (
                        ORDER BY trade_date ROWS BETWEEN 4 PRECEDING AND CURRENT ROW
                    ) AS bofa_cum_net_flow_5d_tl,
                    SUM(top5_domestic_prev_day_net_flow_tl) OVER (
                        ORDER BY trade_date ROWS BETWEEN 4 PRECEDING AND CURRENT ROW
                    ) AS top5_cum_net_flow_5d_tl,
                    AVG(bofa_prev_day_net_flow_tl) OVER (
                        ORDER BY trade_date ROWS BETWEEN 19 PRECEDING AND CURRENT ROW
                    ) AS bofa_mean_20d,
                    STDDEV(bofa_prev_day_net_flow_tl) OVER (
                        ORDER BY trade_date ROWS BETWEEN 19 PRECEDING AND CURRENT ROW
                    ) AS bofa_std_20d,
                    AVG(bofa_daily_buy_vwap) OVER (
                        ORDER BY trade_date ROWS BETWEEN 19 PRECEDING AND CURRENT ROW
                    ) AS bofa_buy_vwap_20d
                FROM daily_feature_base
            ),
            -- Lag Alignment (Features of Day T come strictly from T-1 Close)
            lagged_features AS (
                SELECT 
                    trade_date,
                    day_of_week,
                    is_monday,
                    is_friday,
                    LAG(bofa_w4_net_flow_tl, 1) OVER (ORDER BY trade_date) AS feat_bofa_w4_net_flow_tl,
                    LAG(bofa_w4_turnover_tl, 1) OVER (ORDER BY trade_date) AS feat_bofa_w4_turnover_tl,
                    LAG(w4_flow_acceleration_ratio, 1) OVER (ORDER BY trade_date) AS feat_w4_flow_acceleration_ratio,
                    LAG(bofa_prev_day_net_flow_tl, 1) OVER (ORDER BY trade_date) AS feat_bofa_prev_day_net_flow_tl,
                    LAG(bofa_prev_day_turnover_tl, 1) OVER (ORDER BY trade_date) AS feat_bofa_prev_day_turnover_tl,
                    LAG(bofa_prev_day_market_share, 1) OVER (ORDER BY trade_date) AS feat_bofa_prev_day_market_share,
                    LAG(bofa_prev_day_turnover_rank, 1) OVER (ORDER BY trade_date) AS feat_bofa_prev_day_turnover_rank,
                    LAG(top5_domestic_w4_net_flow_tl, 1) OVER (ORDER BY trade_date) AS feat_top5_domestic_w4_net_flow_tl,
                    LAG(top5_domestic_prev_day_net_flow_tl, 1) OVER (ORDER BY trade_date) AS feat_top5_domestic_prev_day_net_flow_tl,
                    LAG(bofa_vs_top5_w4_flow_delta_tl, 1) OVER (ORDER BY trade_date) AS feat_bofa_vs_top5_w4_flow_delta_tl,
                    LAG(bofa_vs_top5_total_flow_delta_tl, 1) OVER (ORDER BY trade_date) AS feat_bofa_vs_top5_total_flow_delta_tl,
                    LAG(institutional_hegemony_share, 1) OVER (ORDER BY trade_date) AS feat_institutional_hegemony_share,
                    LAG(avg_cr5_concentration, 1) OVER (ORDER BY trade_date) AS feat_avg_cr5_concentration,
                    LAG(market_avg_return_pct, 1) OVER (ORDER BY trade_date) AS feat_market_avg_return_pct,
                    LAG(market_avg_range_pct, 1) OVER (ORDER BY trade_date) AS feat_market_avg_range_pct,
                    LAG(prev_day_close_vs_vwap_spread_pct, 1) OVER (ORDER BY trade_date) AS feat_prev_day_close_vs_vwap_spread_pct,
                    LAG(bofa_banking_flow_prev_day, 1) OVER (ORDER BY trade_date) AS feat_bofa_banking_flow_prev_day,
                    LAG(bofa_transport_flow_prev_day, 1) OVER (ORDER BY trade_date) AS feat_bofa_transport_flow_prev_day,
                    LAG(bofa_holding_flow_prev_day, 1) OVER (ORDER BY trade_date) AS feat_bofa_holding_flow_prev_day,
                    LAG(bofa_energy_flow_prev_day, 1) OVER (ORDER BY trade_date) AS feat_bofa_energy_flow_prev_day,
                    LAG(bofa_defense_flow_prev_day, 1) OVER (ORDER BY trade_date) AS feat_bofa_defense_flow_prev_day,
                    LAG(top5_banking_flow_prev_day, 1) OVER (ORDER BY trade_date) AS feat_top5_banking_flow_prev_day,
                    LAG(bofa_cum_net_flow_5d_tl, 1) OVER (ORDER BY trade_date) AS feat_bofa_cum_net_flow_5d_tl,
                    LAG(top5_cum_net_flow_5d_tl, 1) OVER (ORDER BY trade_date) AS feat_top5_cum_net_flow_5d_tl,
                    LAG(CASE WHEN bofa_std_20d > 0 THEN (bofa_prev_day_net_flow_tl - bofa_mean_20d) / bofa_std_20d ELSE 0.0 END, 1) 
                        OVER (ORDER BY trade_date) AS feat_bofa_flow_zscore_20d,
                    LAG(CASE WHEN bofa_buy_vwap_20d > 0 THEN (market_avg_close_price - bofa_buy_vwap_20d) / bofa_buy_vwap_20d ELSE 0.0 END, 1) 
                        OVER (ORDER BY trade_date) AS feat_bofa_cost_basis_spread_20d_pct
                FROM unlagged_rolling
            )
            SELECT 
                r.trade_date,
                r.day_of_week,
                r.is_monday,
                r.is_friday,
                COALESCE(r.feat_bofa_w4_net_flow_tl, 0.0) AS feat_bofa_w4_net_flow_tl,
                COALESCE(r.feat_bofa_w4_turnover_tl, 0.0) AS feat_bofa_w4_turnover_tl,
                COALESCE(r.feat_w4_flow_acceleration_ratio, 0.0) AS feat_w4_flow_acceleration_ratio,
                COALESCE(r.feat_bofa_prev_day_net_flow_tl, 0.0) AS feat_bofa_prev_day_net_flow_tl,
                COALESCE(r.feat_bofa_prev_day_turnover_tl, 0.0) AS feat_bofa_prev_day_turnover_tl,
                COALESCE(r.feat_bofa_prev_day_market_share, 0.0) AS feat_bofa_prev_day_market_share,
                COALESCE(r.feat_bofa_prev_day_turnover_rank, 10) AS feat_bofa_prev_day_turnover_rank,
                COALESCE(r.feat_top5_domestic_w4_net_flow_tl, 0.0) AS feat_top5_domestic_w4_net_flow_tl,
                COALESCE(r.feat_top5_domestic_prev_day_net_flow_tl, 0.0) AS feat_top5_domestic_prev_day_net_flow_tl,
                COALESCE(r.feat_bofa_vs_top5_w4_flow_delta_tl, 0.0) AS feat_bofa_vs_top5_w4_flow_delta_tl,
                COALESCE(r.feat_bofa_vs_top5_total_flow_delta_tl, 0.0) AS feat_bofa_vs_top5_total_flow_delta_tl,
                COALESCE(r.feat_institutional_hegemony_share, 0.0) AS feat_institutional_hegemony_share,
                COALESCE(r.feat_avg_cr5_concentration, 0.0) AS feat_avg_cr5_concentration,
                COALESCE(r.feat_market_avg_return_pct, 0.0) AS feat_market_avg_return_pct,
                COALESCE(r.feat_market_avg_range_pct, 0.0) AS feat_market_avg_range_pct,
                COALESCE(r.feat_prev_day_close_vs_vwap_spread_pct, 0.0) AS feat_prev_day_close_vs_vwap_spread_pct,
                COALESCE(r.feat_bofa_banking_flow_prev_day, 0.0) AS feat_bofa_banking_flow_prev_day,
                COALESCE(r.feat_bofa_transport_flow_prev_day, 0.0) AS feat_bofa_transport_flow_prev_day,
                COALESCE(r.feat_bofa_holding_flow_prev_day, 0.0) AS feat_bofa_holding_flow_prev_day,
                COALESCE(r.feat_bofa_energy_flow_prev_day, 0.0) AS feat_bofa_energy_flow_prev_day,
                COALESCE(r.feat_bofa_defense_flow_prev_day, 0.0) AS feat_bofa_defense_flow_prev_day,
                COALESCE(r.feat_top5_banking_flow_prev_day, 0.0) AS feat_top5_banking_flow_prev_day,
                COALESCE(r.feat_bofa_cum_net_flow_5d_tl, 0.0) AS feat_bofa_cum_net_flow_5d_tl,
                COALESCE(r.feat_top5_cum_net_flow_5d_tl, 0.0) AS feat_top5_cum_net_flow_5d_tl,
                COALESCE(r.feat_bofa_flow_zscore_20d, 0.0) AS feat_bofa_flow_zscore_20d,
                COALESCE(r.feat_bofa_cost_basis_spread_20d_pct, 0.0) AS feat_bofa_cost_basis_spread_20d_pct,
                -- Target Columns on Day T
                COALESCE(t.target_open_net_flow_tl, 0.0) AS target_open_net_flow_tl,
                CASE WHEN COALESCE(t.target_open_net_flow_tl, 0.0) > 0 THEN 'BUY' ELSE 'SELL' END AS target_open_direction
            FROM lagged_features r
            LEFT JOIN day_t_targets t ON r.trade_date = t.trade_date
            WHERE r.feat_bofa_prev_day_net_flow_tl IS NOT NULL
              AND (? IS NULL OR r.trade_date >= ?)
              AND (? IS NULL OR r.trade_date <= ?)
            ORDER BY r.trade_date ASC;
        """
        df = conn.execute(query, [effective_start, effective_start, end_date, end_date]).pl()
        logger.info(f"Extracted {df.height} historical daily observations with {len(df.columns)} features.")
        return df
