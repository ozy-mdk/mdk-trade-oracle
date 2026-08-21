from datetime import date
from typing import Any, List, Optional, Union

from mdk_trading_oracle.core.config import get_settings
from mdk_trading_oracle.core.db import DuckDBManager
from mdk_trading_oracle.core.logger import get_logger
from mdk_trading_oracle.data.gold.schema import initialize_gold_schema
from mdk_trading_oracle.models.day_start.forecaster import DayStartForecaster
from mdk_trading_oracle.models.sector_day_start.forecaster import SectorDayStartForecaster

logger = get_logger("mdk_oracle.data.gold.feature_engineering")


class GoldFeatureEngineer:
    """Computes high-alpha feature representations and executes predictive Gold models."""

    def __init__(self, db: DuckDBManager):
        self.db = db
        self.settings = get_settings()

    def compute_institutional_signals(self) -> dict[str, Any]:
        """Compute rolling 5-day and 20-day BofA flows and Z-scores from `silver_daily_stock_summary`."""
        conn = self.db.get_connection()
        logger.info("Computing `gold_institutional_daily_signals` from Silver tables...")

        query = """
            CREATE OR REPLACE TABLE gold_institutional_daily_signals AS
            WITH base AS (
                SELECT 
                    trade_date,
                    symbol,
                    bofa_net_flow_tl,
                    bofa_stock_turnover_share AS bofa_volume_share,
                    market_vwap,
                    close_price,
                    SUM(bofa_net_flow_tl) OVER (
                        PARTITION BY symbol ORDER BY trade_date ROWS BETWEEN 4 PRECEDING AND CURRENT ROW
                    ) AS bofa_accum_5d_tl,
                    SUM(bofa_net_flow_tl) OVER (
                        PARTITION BY symbol ORDER BY trade_date ROWS BETWEEN 19 PRECEDING AND CURRENT ROW
                    ) AS bofa_accum_20d_tl,
                    AVG(bofa_net_flow_tl) OVER (
                        PARTITION BY symbol ORDER BY trade_date ROWS BETWEEN 19 PRECEDING AND CURRENT ROW
                    ) AS mean_flow_20d,
                    STDDEV(bofa_net_flow_tl) OVER (
                        PARTITION BY symbol ORDER BY trade_date ROWS BETWEEN 19 PRECEDING AND CURRENT ROW
                    ) AS std_flow_20d
                FROM silver_daily_stock_summary
            )
            SELECT 
                trade_date,
                symbol,
                bofa_net_flow_tl,
                bofa_volume_share,
                CASE 
                    WHEN std_flow_20d > 0 THEN (bofa_net_flow_tl - mean_flow_20d) / std_flow_20d
                    ELSE 0.0
                END AS bofa_flow_zscore_20d,
                bofa_accum_5d_tl,
                bofa_accum_20d_tl,
                market_vwap,
                close_price,
                CURRENT_TIMESTAMP AS calculated_at
            FROM base;
        """
        conn.execute(query)

        rows = conn.execute("SELECT COUNT(*) FROM gold_institutional_daily_signals;").fetchone()[0]
        logger.info(f"Successfully populated `gold_institutional_daily_signals`: {rows:,} rows.")
        return {"table": "gold_institutional_daily_signals", "rows": rows, "status": "success"}

    def run_day_start_forecasting(
        self,
        backfill_dates: Optional[List[Union[str, date]]] = None,
        all_missing: bool = False,
        backfill_lookback_months: Optional[int] = None,
        backfill_lookback_days: Optional[int] = None,
    ) -> dict[str, Any]:
        """Execute Model 1 (Day-Start Forecaster) with Auto-Champion Selection, reconcile performance ledger, and persist live forecasts."""
        logger.info("Executing Gold Layer Model 1: Day-Start Forecaster (Auto-Champion Mode)...")
        forecaster = DayStartForecaster(self.db, model_type="auto")
        
        # 1. Reconcile Completed Historical Sessions into Performance Ledger
        saved_performance = forecaster.reconcile_and_update_performance_ledger()
        
        # 2. Point-in-Time Backfill for specified missed dates if requested
        if backfill_dates or all_missing:
            saved_performance = forecaster.backfill_historical_performance(
                target_dates=backfill_dates,
                all_missing=all_missing,
                lookback_months=backfill_lookback_months,
                lookback_days=backfill_lookback_days,
            )

        # 3. Live Next-Day Forecast (strictly upcoming T+1)
        saved_forecasts = 0
        try:
            next_day_forecast = forecaster.forecast_next_day()
            saved_forecasts = forecaster.save_forecasts_to_gold(next_day_forecast, replace_active=True)
        except Exception as e:
            logger.warning(f"Could not generate live day-start forecast (insufficient data): {e}")
        
        # 4. Historical Out-of-Sample Walk-Forward Backtests (1..T)
        saved_backtests = forecaster.save_backtests_to_gold()
        
        return {
            "forecast_table": "gold_bofa_day_start_forecasts",
            "forecast_rows": saved_forecasts,
            "performance_table": "gold_bofa_day_start_performance",
            "performance_rows": saved_performance,
            "backtest_table": "gold_bofa_day_start_backtests",
            "backtest_rows": saved_backtests,
            "champion_model": forecaster.champion_name,
            "status": "success",
        }

    def run_sector_day_start_forecasting(
        self,
        backfill_dates: Optional[List[Union[str, date]]] = None,
        all_missing: bool = False,
        backfill_lookback_months: Optional[int] = None,
        backfill_lookback_days: Optional[int] = None,
    ) -> dict[str, Any]:
        """Execute Model 2 (Sector Day-Start Forecaster), reconcile sector performance ledger, and persist live sector forecasts."""
        logger.info("Executing Gold Layer Model 2: Sector Day-Start Forecaster (Auto-Champion Mode)...")
        forecaster = SectorDayStartForecaster(self.db, model_type="auto")
        
        # 1. Reconcile Completed Historical Sessions into Sector Performance Ledger
        saved_sector_performance = forecaster.reconcile_and_update_performance_ledger()
        
        # 2. Point-in-Time Backfill for specified missed dates if requested
        if backfill_dates or all_missing:
            saved_sector_performance = forecaster.backfill_historical_performance(
                target_dates=backfill_dates,
                all_missing=all_missing,
                lookback_months=backfill_lookback_months,
                lookback_days=backfill_lookback_days,
            )

        # 3. Live Next-Day Sector Forecasts (strictly upcoming T+1 across 26 sectors)
        saved_sector_forecasts = 0
        try:
            next_day_sector_forecasts = forecaster.forecast_next_day()
            if next_day_sector_forecasts:
                saved_sector_forecasts = forecaster.save_forecasts_to_gold(next_day_sector_forecasts, replace_active=True)
        except Exception as e:
            logger.warning(f"Could not generate live sector day-start forecasts: {e}")
        
        # 4. Historical Sector Backtests across tracked sectors
        saved_sector_backtests = forecaster.save_backtests_to_gold()
        
        return {
            "forecast_table": "gold_bofa_sector_day_start_forecasts",
            "forecast_rows": saved_sector_forecasts,
            "performance_table": "gold_bofa_sector_day_start_performance",
            "performance_rows": saved_sector_performance,
            "backtest_table": "gold_bofa_sector_day_start_backtests",
            "backtest_rows": saved_sector_backtests,
            "champion_model": forecaster.champion_name,
            "status": "success",
        }

    def run_all(
        self,
        backfill_dates: Optional[List[Union[str, date]]] = None,
        all_missing: bool = False,
        backfill_lookback_months: Optional[int] = None,
        backfill_lookback_days: Optional[int] = None,
    ) -> dict[str, Any]:
        """Run full Gold feature pipeline and predictive models."""
        initialize_gold_schema(self.db)
        res_signals = self.compute_institutional_signals()
        res_day_start = self.run_day_start_forecasting(
            backfill_dates=backfill_dates,
            all_missing=all_missing,
            backfill_lookback_months=backfill_lookback_months,
            backfill_lookback_days=backfill_lookback_days,
        )
        res_sector_day_start = self.run_sector_day_start_forecasting(
            backfill_dates=backfill_dates,
            all_missing=all_missing,
            backfill_lookback_months=backfill_lookback_months,
            backfill_lookback_days=backfill_lookback_days,
        )

        return {
            "gold_institutional_daily_signals": res_signals,
            "gold_bofa_day_start_forecasts": res_day_start,
            "gold_bofa_sector_day_start_forecasts": res_sector_day_start,
            "status": "success",
        }


