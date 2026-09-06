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
        disabled_clusters: Optional[List[str]] = None,
        enabled_clusters: Optional[List[str]] = None,
        include_features: Optional[List[str]] = None,
        exclude_features: Optional[List[str]] = None,
    ) -> dict[str, Any]:
        """Execute Model 1 (Day-Start Forecaster) with Auto-Champion Selection, reconcile performance ledger, and persist live forecasts."""
        logger.info("Executing Gold Layer Model 1: Day-Start Forecaster (Auto-Champion Mode)...")
        forecaster = DayStartForecaster(
            self.db,
            model_type="auto",
            disabled_clusters=disabled_clusters,
            enabled_clusters=enabled_clusters,
            include_features=include_features,
            exclude_features=exclude_features,
        )

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
        disabled_clusters: Optional[List[str]] = None,
        enabled_clusters: Optional[List[str]] = None,
        include_features: Optional[List[str]] = None,
        exclude_features: Optional[List[str]] = None,
    ) -> dict[str, Any]:
        """Execute Model 2 (Sector Day-Start Forecaster), reconcile sector performance ledger, and persist live sector forecasts."""
        logger.info("Executing Gold Layer Model 2: Sector Day-Start Forecaster (Auto-Champion Mode)...")
        forecaster = SectorDayStartForecaster(
            self.db,
            model_type="auto",
            disabled_clusters=disabled_clusters,
            enabled_clusters=enabled_clusters,
            include_features=include_features,
            exclude_features=exclude_features,
        )

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
                saved_sector_forecasts = forecaster.save_forecasts_to_gold(
                    next_day_sector_forecasts, replace_active=True
                )
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

    def run_stock_reaction_forecasting(
        self,
        symbols: Optional[List[str]] = None,
        windows: Optional[List[str]] = None,
        run_backtest: bool = False,
        backfill_dates: Optional[List[Union[str, date]]] = None,
        all_missing: bool = False,
        backfill_lookback_months: Optional[int] = None,
        backfill_lookback_days: Optional[int] = None,
    ) -> dict[str, Any]:
        """Execute Model 3 (Stock Intraday Reaction Forecaster) across BIST30 stocks x W2/W3/W5.

        Args:
            symbols: Explicit symbol list override (None = use config/default fallback to all BIST30).
            windows: Explicit window list override (None = ['w2', 'w3', 'w5']).
            run_backtest: Also run full walk-forward backtests (slow — typically run once on setup).
            backfill_dates: Specific dates to point-in-time backfill into performance ledgers.
            all_missing: If True, backfill all missing historical sessions within lookback window.
            backfill_lookback_months: Number of trailing months to look back for missing sessions.
            backfill_lookback_days: Number of trailing days to look back for missing sessions.
        """
        # Lazy import to avoid circular imports and optional heavy dependency loading
        from mdk_trading_oracle.models.stock_reaction.orchestrator import StockReactionOrchestrator

        logger.info(
            "Executing Gold Layer Model 3: Stock Intraday Reaction Forecaster "
            f"(symbols={'ALL' if not symbols else symbols}, windows={windows or 'ALL'})..."
        )
        orchestrator = StockReactionOrchestrator(
            db=self.db,
            symbols=symbols,
            windows=windows,
        )

        backfill_count = 0
        if backfill_dates or all_missing:
            backfill_res = orchestrator.backfill_historical_performance(
                target_dates=backfill_dates,
                symbols=symbols,
                windows=windows,
                all_missing=all_missing,
                lookback_months=backfill_lookback_months,
                lookback_days=backfill_lookback_days,
            )
            backfill_count = backfill_res.get("total_backfilled", 0)

        result = orchestrator.run_all_windows(run_backtest=run_backtest)
        return {
            "model": "stock_reaction_forecaster",
            "symbols_run": result["symbols_run"],
            "windows_run": result["windows_run"],
            "total_runs": result["total_runs"],
            "success_count": result["success_count"],
            "error_count": result["error_count"],
            "backfill_rows": backfill_count,
            "status": "success" if result["error_count"] == 0 else "partial",
        }

    def run_all(
        self,
        backfill_dates: Optional[List[Union[str, date]]] = None,
        all_missing: bool = False,
        backfill_lookback_months: Optional[int] = None,
        backfill_lookback_days: Optional[int] = None,
        disabled_clusters: Optional[List[str]] = None,
        enabled_clusters: Optional[List[str]] = None,
        include_features: Optional[List[str]] = None,
        exclude_features: Optional[List[str]] = None,
        # Model 3 specific
        stock_reaction_symbols: Optional[List[str]] = None,
        stock_reaction_windows: Optional[List[str]] = None,
        stock_reaction_backtest: bool = False,
    ) -> dict[str, Any]:
        """Execute all Gold layer feature tables, institutional signals, predictive models, and ledgers."""
        initialize_gold_schema(self.db)
        sig_res = self.compute_institutional_signals()
        day_start_res = self.run_day_start_forecasting(
            backfill_dates=backfill_dates,
            all_missing=all_missing,
            backfill_lookback_months=backfill_lookback_months,
            backfill_lookback_days=backfill_lookback_days,
            disabled_clusters=disabled_clusters,
            enabled_clusters=enabled_clusters,
            include_features=include_features,
            exclude_features=exclude_features,
        )
        sector_day_start_res = self.run_sector_day_start_forecasting(
            backfill_dates=backfill_dates,
            all_missing=all_missing,
            backfill_lookback_months=backfill_lookback_months,
            backfill_lookback_days=backfill_lookback_days,
            disabled_clusters=disabled_clusters,
            enabled_clusters=enabled_clusters,
            include_features=include_features,
            exclude_features=exclude_features,
        )
        # Model 3: Stock Intraday Reaction (BIST30 or configured symbol subset)
        stock_reaction_res = self.run_stock_reaction_forecasting(
            symbols=stock_reaction_symbols,
            windows=stock_reaction_windows,
            run_backtest=stock_reaction_backtest,
            backfill_dates=backfill_dates,
            all_missing=all_missing,
            backfill_lookback_months=backfill_lookback_months,
            backfill_lookback_days=backfill_lookback_days,
        )
        return {
            "institutional_signals": sig_res,
            "day_start_macro_forecaster": day_start_res,
            "sector_day_start_forecaster": sector_day_start_res,
            "stock_reaction_forecaster": stock_reaction_res,
            "status": "success",
        }
