"""Sector Day-Start Feature Extraction Engine: Assembles sector-specific feature clusters from Silver fact tables."""

from datetime import date
from typing import List, Optional

import polars as pl

from mdk_trading_oracle.core.config import get_settings
from mdk_trading_oracle.core.db import DuckDBManager
from mdk_trading_oracle.core.logger import get_logger
from mdk_trading_oracle.models.base import BaseFeatureExtractor
from mdk_trading_oracle.models.day_start.features import get_next_trading_day

logger = get_logger("mdk_oracle.models.sector_day_start.features")


class SectorDayStartFeatureExtractor(BaseFeatureExtractor):
    """Extracts, engineers, and temporal-aligns sector-specific features at T-1 Close to predict T_open (Window 1) per sector.
    
    Clusters:
        1. Sector Prior Closing Window Momentum (Window 4 net flow & turnover)
        2. Sector Competitor Imbalance (BofA vs Top-5 domestic desk deltas in that sector)
        3. Sector Dominance & Share of Wallet (BofA sector share vs total BofA flow)
        4. Sector Multi-Day Accumulation & Saturation (5d/20d rolling flows & Z-scores)
        5. Macro Context & Calendar Seasonality (is_monday, is_friday, macro BofA flow)
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
        cfg = self.settings.get_model_config("sector_day_start")
        self.lookback_months = lookback_months if lookback_months is not None else cfg.get("lookback_months", 12)

    def get_tracked_sectors(self, min_session_count: int = 15) -> List[str]:
        """Retrieve list of distinct liquid sectors available in DuckDB."""
        conn = self.db.get_connection()
        query = f"""
            SELECT sector, COUNT(DISTINCT trade_date) as session_count
            FROM silver_daily_sector_summary
            WHERE broker_id = '{self.target_broker}'
            GROUP BY sector
            HAVING COUNT(DISTINCT trade_date) >= {min_session_count}
            ORDER BY SUM(total_turnover_tl) DESC;
        """
        df = conn.execute(query).df()
        return df["sector"].tolist()

    def extract_features(
        self,
        sector: Optional[str] = None,
        start_date: Optional[date] = None,
        end_date: Optional[date] = None,
    ) -> pl.DataFrame:
        """Extract multi-cluster feature matrix for a specific sector (or all sectors) from DuckDB Silver tables with lookback support.
        
        Args:
            sector: Optional specific sector name to filter (e.g. 'Banking').
            start_date: Optional filter for minimum trade date.
            end_date: Optional filter for maximum trade date.
            
        Returns:
            pl.DataFrame: Clean dataset where each row is (trade_date T, sector s) with features computed 
                          strictly from historical data up to T-1 Close, paired with actual Window 1 target outcomes on Day T.
        """
        conn = self.db.get_connection()
        sector_filter_w4 = f"AND sector = '{sector}'" if sector else ""
        sector_filter_macro = f"AND sector = '{sector}'" if sector else ""

        logger.info(f"Extracting Sector Day-Start Features for broker '{self.target_broker}' (Sector: {sector or 'ALL'}, lookback_months={self.lookback_months})...")

        # Determine effective start_date from lookback_months if not explicitly given
        effective_start = start_date
        if effective_start is None and self.lookback_months is not None:
            max_d_res = conn.execute("SELECT MAX(trade_date) FROM silver_daily_sector_summary;").fetchone()
            if max_d_res and max_d_res[0]:
                from dateutil.relativedelta import relativedelta
                effective_start = max_d_res[0] - relativedelta(months=self.lookback_months)

        query = f"""
            WITH daily_dates AS (
                SELECT DISTINCT trade_date
                FROM silver_daily_stock_summary
                ORDER BY trade_date ASC
            ),
            sector_universe AS (
                SELECT DISTINCT sector
                FROM silver_daily_sector_summary
                WHERE sector IS NOT NULL AND sector != '' {sector_filter_macro}
            ),
            date_sector_grid AS (
                SELECT d.trade_date, s.sector
                FROM daily_dates d
                CROSS JOIN sector_universe s
            ),
            -- 1. Sector Target Variables on Day T (Window 1: day_start)
            day_t_sector_targets AS (
                SELECT 
                    trade_date,
                    sector,
                    SUM(CASE WHEN broker_id = '{self.target_broker}' THEN net_flow_tl ELSE 0.0 END) AS target_sector_open_net_flow_tl
                FROM silver_intraday_sector_window_summary
                WHERE window_name = 'day_start'
                GROUP BY trade_date, sector
            ),
            -- 2. Prior Day Sector Closing Window 4 Flow (T-1)
            prev_day_sector_w4 AS (
                SELECT 
                    trade_date,
                    sector,
                    SUM(CASE WHEN broker_id = '{self.target_broker}' THEN net_flow_tl ELSE 0.0 END) AS bofa_sector_w4_net_flow_tl,
                    SUM(CASE WHEN broker_id = '{self.target_broker}' THEN total_turnover_tl ELSE 0.0 END) AS bofa_sector_w4_turnover_tl,
                    SUM(CASE WHEN broker_id IN ('IYM', 'YKR', 'AKM', 'GRM', 'ZRY') THEN net_flow_tl ELSE 0.0 END) AS top5_sector_w4_net_flow_tl,
                    SUM(CASE WHEN broker_id IN ('IYM', 'YKR', 'AKM', 'GRM', 'ZRY') THEN total_turnover_tl ELSE 0.0 END) AS top5_sector_w4_turnover_tl
                FROM silver_intraday_sector_window_summary
                WHERE window_name = 'closing_session' {sector_filter_w4}
                GROUP BY trade_date, sector
            ),
            -- 3. Prior Day Full Daily Sector Aggregates (T-1)
            prev_day_sector_daily AS (
                SELECT 
                    trade_date,
                    sector,
                    SUM(CASE WHEN broker_id = '{self.target_broker}' THEN net_flow_tl ELSE 0.0 END) AS bofa_sector_prev_day_net_flow_tl,
                    SUM(CASE WHEN broker_id = '{self.target_broker}' THEN total_turnover_tl ELSE 0.0 END) AS bofa_sector_prev_day_turnover_tl,
                    SUM(CASE WHEN broker_id IN ('IYM', 'YKR', 'AKM', 'GRM', 'ZRY') THEN net_flow_tl ELSE 0.0 END) AS top5_sector_prev_day_net_flow_tl,
                    SUM(CASE WHEN broker_id IN ('IYM', 'YKR', 'AKM', 'GRM', 'ZRY') THEN total_turnover_tl ELSE 0.0 END) AS top5_sector_prev_day_turnover_tl,
                    SUM(total_turnover_tl) / 2.0 AS total_sector_turnover_tl
                FROM silver_daily_sector_summary
                WHERE 1=1 {sector_filter_macro}
                GROUP BY trade_date, sector
            ),
            -- 4. Macro Context (Total BofA Daily Net Flow across all sectors)
            prev_day_macro_overview AS (
                SELECT 
                    trade_date,
                    MAX(CASE WHEN broker_id = '{self.target_broker}' THEN net_flow_tl ELSE NULL END) AS bofa_macro_prev_day_net_flow_tl,
                    MAX(CASE WHEN broker_id = '{self.target_broker}' THEN total_turnover_tl ELSE NULL END) AS bofa_macro_prev_day_turnover_tl,
                    SUM(CASE WHEN broker_id IN ('IYM', 'YKR', 'AKM', 'GRM', 'ZRY') THEN net_flow_tl ELSE 0.0 END) AS top5_macro_prev_day_net_flow_tl
                FROM silver_daily_broker_overview
                GROUP BY trade_date
            ),
            -- Combine base features
            sector_feature_base AS (
                SELECT 
                    g.trade_date,
                    g.sector,
                    EXTRACT(DOW FROM g.trade_date) AS day_of_week,
                    (EXTRACT(DOW FROM g.trade_date) = 1) AS is_monday,
                    (EXTRACT(DOW FROM g.trade_date) = 5) AS is_friday,
                    -- Cluster 1: Sector Closing Window Momentum
                    COALESCE(w4.bofa_sector_w4_net_flow_tl, 0.0) AS bofa_sector_w4_net_flow_tl,
                    COALESCE(w4.bofa_sector_w4_turnover_tl, 0.0) AS bofa_sector_w4_turnover_tl,
                    -- Cluster 2: Sector Competitor Posture & Deltas
                    COALESCE(w4.top5_sector_w4_net_flow_tl, 0.0) AS top5_sector_w4_net_flow_tl,
                    COALESCE(w4.bofa_sector_w4_net_flow_tl, 0.0) - COALESCE(w4.top5_sector_w4_net_flow_tl, 0.0) AS bofa_vs_top5_sector_w4_delta_tl,
                    COALESCE(sd.bofa_sector_prev_day_net_flow_tl, 0.0) AS bofa_sector_prev_day_net_flow_tl,
                    COALESCE(sd.bofa_sector_prev_day_turnover_tl, 0.0) AS bofa_sector_prev_day_turnover_tl,
                    COALESCE(sd.top5_sector_prev_day_net_flow_tl, 0.0) AS top5_sector_prev_day_net_flow_tl,
                    COALESCE(sd.bofa_sector_prev_day_net_flow_tl, 0.0) - COALESCE(sd.top5_sector_prev_day_net_flow_tl, 0.0) AS bofa_vs_top5_sector_daily_delta_tl,
                    -- Cluster 3: Sector Market Share & Share of BofA Wallet
                    COALESCE(sd.bofa_sector_prev_day_turnover_tl, 0.0) / 
                        NULLIF(COALESCE(sd.total_sector_turnover_tl, 1.0), 0.0) AS bofa_sector_market_share,
                    COALESCE(sd.bofa_sector_prev_day_turnover_tl, 0.0) / 
                        NULLIF(COALESCE(mo.bofa_macro_prev_day_turnover_tl, 1.0), 0.0) AS bofa_sector_share_of_wallet,
                    -- Cluster 5: Macro Flow Context
                    COALESCE(mo.bofa_macro_prev_day_net_flow_tl, 0.0) AS bofa_macro_prev_day_net_flow_tl,
                    COALESCE(mo.top5_macro_prev_day_net_flow_tl, 0.0) AS top5_macro_prev_day_net_flow_tl
                FROM date_sector_grid g
                LEFT JOIN prev_day_sector_w4 w4 ON g.trade_date = w4.trade_date AND g.sector = w4.sector
                LEFT JOIN prev_day_sector_daily sd ON g.trade_date = sd.trade_date AND g.sector = sd.sector
                LEFT JOIN prev_day_macro_overview mo ON g.trade_date = mo.trade_date
            ),
            -- Multi-Day Rolling Aggregations per Sector (Unlagged)
            unlagged_sector_rolling AS (
                SELECT 
                    *,
                    SUM(bofa_sector_prev_day_net_flow_tl) OVER (
                        PARTITION BY sector ORDER BY trade_date ROWS BETWEEN 4 PRECEDING AND CURRENT ROW
                    ) AS bofa_sector_cum_net_flow_5d_tl,
                    SUM(top5_sector_prev_day_net_flow_tl) OVER (
                        PARTITION BY sector ORDER BY trade_date ROWS BETWEEN 4 PRECEDING AND CURRENT ROW
                    ) AS top5_sector_cum_net_flow_5d_tl,
                    AVG(bofa_sector_prev_day_net_flow_tl) OVER (
                        PARTITION BY sector ORDER BY trade_date ROWS BETWEEN 19 PRECEDING AND CURRENT ROW
                    ) AS bofa_sector_mean_20d,
                    STDDEV(bofa_sector_prev_day_net_flow_tl) OVER (
                        PARTITION BY sector ORDER BY trade_date ROWS BETWEEN 19 PRECEDING AND CURRENT ROW
                    ) AS bofa_sector_std_20d
                FROM sector_feature_base
            ),
            -- Lag Alignment (Features of Day T for Sector s come strictly from T-1 Close)
            lagged_sector_features AS (
                SELECT 
                    trade_date,
                    sector,
                    day_of_week,
                    is_monday,
                    is_friday,
                    LAG(bofa_sector_w4_net_flow_tl, 1) OVER (PARTITION BY sector ORDER BY trade_date) AS feat_sector_bofa_w4_net_flow_tl,
                    LAG(bofa_sector_w4_turnover_tl, 1) OVER (PARTITION BY sector ORDER BY trade_date) AS feat_sector_bofa_w4_turnover_tl,
                    LAG(top5_sector_w4_net_flow_tl, 1) OVER (PARTITION BY sector ORDER BY trade_date) AS feat_sector_top5_w4_net_flow_tl,
                    LAG(bofa_vs_top5_sector_w4_delta_tl, 1) OVER (PARTITION BY sector ORDER BY trade_date) AS feat_sector_bofa_vs_top5_w4_delta_tl,
                    LAG(bofa_sector_prev_day_net_flow_tl, 1) OVER (PARTITION BY sector ORDER BY trade_date) AS feat_sector_bofa_prev_day_net_flow_tl,
                    LAG(bofa_sector_prev_day_turnover_tl, 1) OVER (PARTITION BY sector ORDER BY trade_date) AS feat_sector_bofa_prev_day_turnover_tl,
                    LAG(top5_sector_prev_day_net_flow_tl, 1) OVER (PARTITION BY sector ORDER BY trade_date) AS feat_sector_top5_prev_day_net_flow_tl,
                    LAG(bofa_vs_top5_sector_daily_delta_tl, 1) OVER (PARTITION BY sector ORDER BY trade_date) AS feat_sector_bofa_vs_top5_daily_delta_tl,
                    LAG(bofa_sector_market_share, 1) OVER (PARTITION BY sector ORDER BY trade_date) AS feat_sector_bofa_market_share,
                    LAG(bofa_sector_share_of_wallet, 1) OVER (PARTITION BY sector ORDER BY trade_date) AS feat_sector_bofa_share_of_wallet,
                    LAG(bofa_macro_prev_day_net_flow_tl, 1) OVER (PARTITION BY sector ORDER BY trade_date) AS feat_macro_bofa_prev_day_net_flow_tl,
                    LAG(top5_macro_prev_day_net_flow_tl, 1) OVER (PARTITION BY sector ORDER BY trade_date) AS feat_macro_top5_prev_day_net_flow_tl,
                    LAG(bofa_sector_cum_net_flow_5d_tl, 1) OVER (PARTITION BY sector ORDER BY trade_date) AS feat_sector_bofa_cum_net_flow_5d_tl,
                    LAG(top5_sector_cum_net_flow_5d_tl, 1) OVER (PARTITION BY sector ORDER BY trade_date) AS feat_sector_top5_cum_net_flow_5d_tl,
                    LAG(CASE WHEN bofa_sector_std_20d > 0 THEN (bofa_sector_prev_day_net_flow_tl - bofa_sector_mean_20d) / bofa_sector_std_20d ELSE 0.0 END, 1) 
                        OVER (PARTITION BY sector ORDER BY trade_date) AS feat_sector_bofa_flow_zscore_20d
                FROM unlagged_sector_rolling
            )
            SELECT 
                r.trade_date,
                r.sector,
                r.day_of_week,
                r.is_monday,
                r.is_friday,
                COALESCE(r.feat_sector_bofa_w4_net_flow_tl, 0.0) AS feat_sector_bofa_w4_net_flow_tl,
                COALESCE(r.feat_sector_bofa_w4_turnover_tl, 0.0) AS feat_sector_bofa_w4_turnover_tl,
                COALESCE(r.feat_sector_top5_w4_net_flow_tl, 0.0) AS feat_sector_top5_w4_net_flow_tl,
                COALESCE(r.feat_sector_bofa_vs_top5_w4_delta_tl, 0.0) AS feat_sector_bofa_vs_top5_w4_delta_tl,
                COALESCE(r.feat_sector_bofa_prev_day_net_flow_tl, 0.0) AS feat_sector_bofa_prev_day_net_flow_tl,
                COALESCE(r.feat_sector_bofa_prev_day_turnover_tl, 0.0) AS feat_sector_bofa_prev_day_turnover_tl,
                COALESCE(r.feat_sector_top5_prev_day_net_flow_tl, 0.0) AS feat_sector_top5_prev_day_net_flow_tl,
                COALESCE(r.feat_sector_bofa_vs_top5_daily_delta_tl, 0.0) AS feat_sector_bofa_vs_top5_daily_delta_tl,
                COALESCE(r.feat_sector_bofa_market_share, 0.0) AS feat_sector_bofa_market_share,
                COALESCE(r.feat_sector_bofa_share_of_wallet, 0.0) AS feat_sector_bofa_share_of_wallet,
                COALESCE(r.feat_macro_bofa_prev_day_net_flow_tl, 0.0) AS feat_macro_bofa_prev_day_net_flow_tl,
                COALESCE(r.feat_macro_top5_prev_day_net_flow_tl, 0.0) AS feat_macro_top5_prev_day_net_flow_tl,
                COALESCE(r.feat_sector_bofa_cum_net_flow_5d_tl, 0.0) AS feat_sector_bofa_cum_net_flow_5d_tl,
                COALESCE(r.feat_sector_top5_cum_net_flow_5d_tl, 0.0) AS feat_sector_top5_cum_net_flow_5d_tl,
                COALESCE(r.feat_sector_bofa_flow_zscore_20d, 0.0) AS feat_sector_bofa_flow_zscore_20d,
                -- Target Columns on Day T for Sector s
                COALESCE(t.target_sector_open_net_flow_tl, 0.0) AS target_sector_open_net_flow_tl,
                CASE WHEN COALESCE(t.target_sector_open_net_flow_tl, 0.0) > 0 THEN 'BUY' ELSE 'SELL' END AS target_sector_open_direction
            FROM lagged_sector_features r
            LEFT JOIN day_t_sector_targets t ON r.trade_date = t.trade_date AND r.sector = t.sector
            WHERE r.feat_sector_bofa_prev_day_net_flow_tl IS NOT NULL
              AND (? IS NULL OR r.trade_date >= ?)
              AND (? IS NULL OR r.trade_date <= ?)
            ORDER BY r.trade_date ASC, r.sector ASC;
        """
        df = conn.execute(query, [effective_start, effective_start, end_date, end_date]).pl()
        logger.info(f"Extracted {df.height} historical sector observations with {len(df.columns)} columns.")
        return df

    def extract_next_day_features(
        self,
        sectors: Optional[List[str]] = None,
        sector: Optional[str] = None,
        as_of_date: Optional[date] = None,
    ) -> pl.DataFrame:
        """Extract multi-sector feature rows for the upcoming trading session (T_next) based on T_close.
        
        Zero lookahead leakage: all sector metrics are computed from completed data strictly up to `as_of_date`
        (or the latest date in DuckDB if as_of_date is None).
        The trade_date is automatically computed as the next trading business day.
        
        Args:
            sectors: Optional list of sectors to include.
            sector: Optional single sector filter.
            as_of_date: Optional reference date. If provided, data after this date is completely hidden.
            
        Returns:
            pl.DataFrame: Multi-sector feature matrix (1 row per tracked sector) ready for live inference.
        """
        conn = self.db.get_connection()
        logger.info(f"Extracting Next-Day Sector Features for broker '{self.target_broker}' (Sector: {sector or (len(sectors) if sectors else 'ALL')}, as_of_date={as_of_date or 'LATEST'})...")

        sector_filter = ""
        if sector:
            sector_filter = f"AND sector = '{sector}'"
        elif sectors:
            formatted_sectors = "', '".join(sectors)
            sector_filter = f"AND sector IN ('{formatted_sectors}')"

        date_filter = f"WHERE trade_date <= '{as_of_date}'" if as_of_date is not None else ""

        query = f"""
            WITH daily_dates AS (
                SELECT DISTINCT trade_date
                FROM silver_daily_stock_summary
                {date_filter}
                ORDER BY trade_date ASC
            ),
            sector_universe AS (
                SELECT DISTINCT sector
                FROM silver_daily_sector_summary
                WHERE sector IS NOT NULL AND sector != '' {sector_filter}
            ),
            date_sector_grid AS (
                SELECT d.trade_date, s.sector
                FROM daily_dates d
                CROSS JOIN sector_universe s
            ),
            prev_day_sector_w4 AS (
                SELECT 
                    trade_date,
                    sector,
                    SUM(CASE WHEN broker_id = '{self.target_broker}' THEN net_flow_tl ELSE 0.0 END) AS bofa_sector_w4_net_flow_tl,
                    SUM(CASE WHEN broker_id = '{self.target_broker}' THEN total_turnover_tl ELSE 0.0 END) AS bofa_sector_w4_turnover_tl,
                    SUM(CASE WHEN broker_id IN ('IYM', 'YKR', 'AKM', 'GRM', 'ZRY') THEN net_flow_tl ELSE 0.0 END) AS top5_sector_w4_net_flow_tl,
                    SUM(CASE WHEN broker_id IN ('IYM', 'YKR', 'AKM', 'GRM', 'ZRY') THEN total_turnover_tl ELSE 0.0 END) AS top5_sector_w4_turnover_tl
                FROM silver_intraday_sector_window_summary
                WHERE window_name = 'closing_session' {sector_filter}
                GROUP BY trade_date, sector
            ),
            prev_day_sector_daily AS (
                SELECT 
                    trade_date,
                    sector,
                    SUM(CASE WHEN broker_id = '{self.target_broker}' THEN net_flow_tl ELSE 0.0 END) AS bofa_sector_prev_day_net_flow_tl,
                    SUM(CASE WHEN broker_id = '{self.target_broker}' THEN total_turnover_tl ELSE 0.0 END) AS bofa_sector_prev_day_turnover_tl,
                    SUM(CASE WHEN broker_id IN ('IYM', 'YKR', 'AKM', 'GRM', 'ZRY') THEN net_flow_tl ELSE 0.0 END) AS top5_sector_prev_day_net_flow_tl,
                    SUM(CASE WHEN broker_id IN ('IYM', 'YKR', 'AKM', 'GRM', 'ZRY') THEN total_turnover_tl ELSE 0.0 END) AS top5_sector_prev_day_turnover_tl,
                    SUM(total_turnover_tl) / 2.0 AS total_sector_turnover_tl
                FROM silver_daily_sector_summary
                WHERE 1=1 {sector_filter}
                GROUP BY trade_date, sector
            ),
            prev_day_macro_overview AS (
                SELECT 
                    trade_date,
                    MAX(CASE WHEN broker_id = '{self.target_broker}' THEN net_flow_tl ELSE NULL END) AS bofa_macro_prev_day_net_flow_tl,
                    MAX(CASE WHEN broker_id = '{self.target_broker}' THEN total_turnover_tl ELSE NULL END) AS bofa_macro_prev_day_turnover_tl,
                    SUM(CASE WHEN broker_id IN ('IYM', 'YKR', 'AKM', 'GRM', 'ZRY') THEN net_flow_tl ELSE 0.0 END) AS top5_macro_prev_day_net_flow_tl
                FROM silver_daily_broker_overview
                GROUP BY trade_date
            ),
            sector_feature_base AS (
                SELECT 
                    g.trade_date,
                    g.sector,
                    COALESCE(w4.bofa_sector_w4_net_flow_tl, 0.0) AS bofa_sector_w4_net_flow_tl,
                    COALESCE(w4.bofa_sector_w4_turnover_tl, 0.0) AS bofa_sector_w4_turnover_tl,
                    COALESCE(w4.top5_sector_w4_net_flow_tl, 0.0) AS top5_sector_w4_net_flow_tl,
                    COALESCE(w4.bofa_sector_w4_net_flow_tl, 0.0) - COALESCE(w4.top5_sector_w4_net_flow_tl, 0.0) AS bofa_vs_top5_sector_w4_delta_tl,
                    COALESCE(sd.bofa_sector_prev_day_net_flow_tl, 0.0) AS bofa_sector_prev_day_net_flow_tl,
                    COALESCE(sd.bofa_sector_prev_day_turnover_tl, 0.0) AS bofa_sector_prev_day_turnover_tl,
                    COALESCE(sd.top5_sector_prev_day_net_flow_tl, 0.0) AS top5_sector_prev_day_net_flow_tl,
                    COALESCE(sd.bofa_sector_prev_day_net_flow_tl, 0.0) - COALESCE(sd.top5_sector_prev_day_net_flow_tl, 0.0) AS bofa_vs_top5_sector_daily_delta_tl,
                    COALESCE(sd.bofa_sector_prev_day_turnover_tl, 0.0) / 
                        NULLIF(COALESCE(sd.total_sector_turnover_tl, 1.0), 0.0) AS bofa_sector_market_share,
                    COALESCE(sd.bofa_sector_prev_day_turnover_tl, 0.0) / 
                        NULLIF(COALESCE(mo.bofa_macro_prev_day_turnover_tl, 1.0), 0.0) AS bofa_sector_share_of_wallet,
                    COALESCE(mo.bofa_macro_prev_day_net_flow_tl, 0.0) AS bofa_macro_prev_day_net_flow_tl,
                    COALESCE(mo.top5_macro_prev_day_net_flow_tl, 0.0) AS top5_macro_prev_day_net_flow_tl
                FROM date_sector_grid g
                LEFT JOIN prev_day_sector_w4 w4 ON g.trade_date = w4.trade_date AND g.sector = w4.sector
                LEFT JOIN prev_day_sector_daily sd ON g.trade_date = sd.trade_date AND g.sector = sd.sector
                LEFT JOIN prev_day_macro_overview mo ON g.trade_date = mo.trade_date
            ),
            unlagged_sector_rolling AS (
                SELECT 
                    *,
                    SUM(bofa_sector_prev_day_net_flow_tl) OVER (
                        PARTITION BY sector ORDER BY trade_date ROWS BETWEEN 4 PRECEDING AND CURRENT ROW
                    ) AS bofa_sector_cum_net_flow_5d_tl,
                    SUM(top5_sector_prev_day_net_flow_tl) OVER (
                        PARTITION BY sector ORDER BY trade_date ROWS BETWEEN 4 PRECEDING AND CURRENT ROW
                    ) AS top5_sector_cum_net_flow_5d_tl,
                    AVG(bofa_sector_prev_day_net_flow_tl) OVER (
                        PARTITION BY sector ORDER BY trade_date ROWS BETWEEN 19 PRECEDING AND CURRENT ROW
                    ) AS bofa_sector_mean_20d,
                    STDDEV(bofa_sector_prev_day_net_flow_tl) OVER (
                        PARTITION BY sector ORDER BY trade_date ROWS BETWEEN 19 PRECEDING AND CURRENT ROW
                    ) AS bofa_sector_std_20d
                FROM sector_feature_base
            ),
            latest_date_cte AS (
                SELECT MAX(trade_date) AS max_trade_date FROM silver_daily_stock_summary {date_filter}
            )
            SELECT 
                r.trade_date AS source_date,
                r.sector,
                COALESCE(r.bofa_sector_w4_net_flow_tl, 0.0) AS feat_sector_bofa_w4_net_flow_tl,
                COALESCE(r.bofa_sector_w4_turnover_tl, 0.0) AS feat_sector_bofa_w4_turnover_tl,
                COALESCE(r.top5_sector_w4_net_flow_tl, 0.0) AS feat_sector_top5_w4_net_flow_tl,
                COALESCE(r.bofa_vs_top5_sector_w4_delta_tl, 0.0) AS feat_sector_bofa_vs_top5_w4_delta_tl,
                COALESCE(r.bofa_sector_prev_day_net_flow_tl, 0.0) AS feat_sector_bofa_prev_day_net_flow_tl,
                COALESCE(r.bofa_sector_prev_day_turnover_tl, 0.0) AS feat_sector_bofa_prev_day_turnover_tl,
                COALESCE(r.top5_sector_prev_day_net_flow_tl, 0.0) AS feat_sector_top5_prev_day_net_flow_tl,
                COALESCE(r.bofa_vs_top5_sector_daily_delta_tl, 0.0) AS feat_sector_bofa_vs_top5_daily_delta_tl,
                COALESCE(r.bofa_sector_market_share, 0.0) AS feat_sector_bofa_market_share,
                COALESCE(r.bofa_sector_share_of_wallet, 0.0) AS feat_sector_bofa_share_of_wallet,
                COALESCE(r.bofa_macro_prev_day_net_flow_tl, 0.0) AS feat_macro_bofa_prev_day_net_flow_tl,
                COALESCE(r.top5_macro_prev_day_net_flow_tl, 0.0) AS feat_macro_top5_prev_day_net_flow_tl,
                COALESCE(r.bofa_sector_cum_net_flow_5d_tl, 0.0) AS feat_sector_bofa_cum_net_flow_5d_tl,
                COALESCE(r.top5_sector_cum_net_flow_5d_tl, 0.0) AS feat_sector_top5_cum_net_flow_5d_tl,
                COALESCE(CASE WHEN r.bofa_sector_std_20d > 0 THEN (r.bofa_sector_prev_day_net_flow_tl - r.bofa_sector_mean_20d) / r.bofa_sector_std_20d ELSE 0.0 END, 0.0) AS feat_sector_bofa_flow_zscore_20d
            FROM unlagged_sector_rolling r
            JOIN latest_date_cte l ON r.trade_date = l.max_trade_date
            ORDER BY r.sector ASC;
        """
        df_latest = conn.execute(query).pl()
        if df_latest.height == 0:
            logger.warning("No historical sector sessions found to extract next-day features.")
            return pl.DataFrame()

        source_date = df_latest["source_date"][0]
        next_date = get_next_trading_day(source_date)
        dow = next_date.isoweekday()
        is_mon = (dow == 1)
        is_fri = (dow == 5)

        df_next = df_latest.with_columns([
            pl.lit(next_date).alias("trade_date"),
            pl.lit(dow).cast(pl.Int64).alias("day_of_week"),
            pl.lit(is_mon).alias("is_monday"),
            pl.lit(is_fri).alias("is_friday"),
        ]).drop("source_date")

        logger.info(
            f"Assembled next-day features across {df_next.height} sectors for {next_date} "
            f"(Source: {source_date} Close, Day of Week: {dow}, is_monday: {is_mon})."
        )
        return df_next


