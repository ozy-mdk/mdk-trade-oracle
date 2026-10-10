"""Medallion Lakehouse Pipeline Orchestrator (Bronze -> Silver -> Gold)."""

from datetime import datetime
from pathlib import Path
from typing import Any, Optional, Union

from rich.console import Console
from rich.table import Table

from mdk_trading_oracle.core.config import get_settings
from mdk_trading_oracle.core.db import PostgresManager
from mdk_trading_oracle.core.logger import get_logger
from mdk_trading_oracle.data.bronze import BronzeIngestor, initialize_bronze_schema
from mdk_trading_oracle.data.discovery import RawDataInspector
from mdk_trading_oracle.data.gold import GoldFeatureEngineer, initialize_gold_schema
from mdk_trading_oracle.data.silver import SilverTransformer, initialize_silver_schema

logger = get_logger("mdk_oracle.data.pipeline")
console = Console()


class MedallionPipeline:
    """Orchestrates Bronze, Silver, and Gold transformations with automatic dependency DAG resolution."""

    VALID_LAYERS = ["catalog", "bronze", "silver", "gold", "all"]

    def __init__(self, db: Optional[PostgresManager] = None):
        self.db = db or PostgresManager()
        self.settings = get_settings()
        self.bronze_ingestor = BronzeIngestor(self.db)
        self.silver_transformer = SilverTransformer(self.db)
        self.gold_engineer = GoldFeatureEngineer(self.db)

    def _resolve_layers(self, target: Union[str, list[str]], resolve_dependencies: bool = True) -> list[str]:
        """Resolve requested target(s) into ordered execution layers."""
        if isinstance(target, str):
            if "," in target:
                layers = [t.strip().lower() for t in target.split(",") if t.strip()]
            else:
                target_lower = target.lower()
                if target_lower == "all":
                    return ["bronze", "silver", "gold"]
                layers = [target_lower]
        else:
            layers = [t.lower() for t in target]

        for layer in layers:
            if layer not in self.VALID_LAYERS:
                raise ValueError(f"Invalid pipeline layer: '{layer}'. Must be one of {self.VALID_LAYERS}")

        if not resolve_dependencies:
            return layers

        # Dependency DAG: gold requires silver, silver requires bronze
        resolved = []
        if "catalog" in layers:
            resolved.append("catalog")
        if "gold" in layers:
            for dep in ["bronze", "silver", "gold"]:
                if dep not in resolved:
                    resolved.append(dep)
        elif "silver" in layers:
            for dep in ["bronze", "silver"]:
                if dep not in resolved:
                    resolved.append(dep)
        elif "bronze" in layers and "bronze" not in resolved:
            resolved.append("bronze")

        return resolved

    def run_catalog_sync(self, raw_glob: Optional[str] = None) -> dict[str, Any]:
        """Execute raw data discovery and synchronize YAML catalogs (instruments & brokers)."""
        logger.info("Starting Data Discovery & Catalog Synchronization...")
        start_time = datetime.now()

        inspector = RawDataInspector(raw_glob=raw_glob)
        res = inspector.sync_to_yaml_catalogs()
        elapsed = (datetime.now() - start_time).total_seconds()

        logger.info(
            f"Catalog Sync completed in {elapsed:.2f}s | "
            f"Instruments: {res['instruments_count']} | Brokers: {res['brokers_count']}"
        )
        return {
            "layer": "catalog",
            "elapsed_sec": elapsed,
            "metrics": {
                "config/instruments.yaml": res["instruments_count"],
                "config/brokers.yaml": res["brokers_count"],
            },
            "details": res,
            "status": "success",
        }

    def run_bronze(
        self,
        raw_glob: Optional[str] = None,
        target_date: Optional[str] = None,
        target_month: Optional[str] = None,
        target_file: Optional[Union[str, Path]] = None,
        force: bool = False,
    ) -> dict[str, Any]:
        """Execute Bronze schema initialization and raw data ingestion (incremental or selective partition)."""
        logger.info("Starting Bronze Layer Ingestion...")
        start_time = datetime.now()

        initialize_bronze_schema(self.db)

        if target_date:
            logger.info(f"Bronze Ingestion targeting single date: {target_date}")
            ingest_res = self.bronze_ingestor.ingest_date(target_date)
        elif target_month:
            logger.info(f"Bronze Ingestion targeting month: {target_month}")
            ingest_res = self.bronze_ingestor.ingest_month(target_month)
        elif target_file:
            logger.info(f"Bronze Ingestion targeting single file: {target_file}")
            ingest_res = self.bronze_ingestor.ingest_file(target_file, force=force)
        elif raw_glob:
            logger.info(f"Bronze Ingestion targeting glob pattern: {raw_glob}")
            ingest_res = self.bronze_ingestor.ingest_bist_raw_csv_glob(glob_pattern=raw_glob)
        else:
            logger.info(f"Bronze Ingestion running incremental discovery (force={force})...")
            ingest_res = self.bronze_ingestor.ingest_all(force=force)

        # Ingest Central Bank policy interest rates and sync/forward-fill to latest market date
        cbrt_res = self.bronze_ingestor.ingest_central_bank_rates(force=force, sync_market_dates=True)

        # Ingest official BIST 30 benchmark index data and sync/forward-fill to latest market date
        bench_res = self.bronze_ingestor.ingest_bist30_benchmarks(force=force, sync_market_dates=True)

        # Ingest corporate action events (splits, bonus issues, ticker changes, rights notes)
        actions_res = self.bronze_ingestor.ingest_corporate_actions(force=force)

        # Ingest BIST 30 membership snapshots, historical changes, and stock periods
        bist30_res = self.bronze_ingestor.ingest_bist30_membership(force=force)

        conn = self.db.get_connection()
        trades_count = conn.execute("SELECT COUNT(*) FROM bronze_raw_trades;").fetchone()[0]
        brokers_count = conn.execute("SELECT COUNT(*) FROM bronze_brokers;").fetchone()[0]
        instruments_count = conn.execute("SELECT COUNT(*) FROM bronze_instruments;").fetchone()[0]
        log_count = conn.execute("SELECT COUNT(*) FROM bronze_ingestion_log;").fetchone()[0]
        cbrt_rates_count = conn.execute("SELECT COUNT(*) FROM bronze_central_bank_rates;").fetchone()[0]
        benchmarks_count = conn.execute("SELECT COUNT(*) FROM bronze_bist_index_benchmarks;").fetchone()[0]
        actions_count = conn.execute("SELECT COUNT(*) FROM bronze_corporate_actions;").fetchone()[0]
        bist30_count = conn.execute("SELECT COUNT(*) FROM bronze_bist30_membership;").fetchone()[0]
        elapsed = (datetime.now() - start_time).total_seconds()

        logger.info(
            f"Bronze Layer completed in {elapsed:.2f}s | "
            f"Raw Trades: {trades_count:,} | CBRT Rates: {cbrt_rates_count:,} | Benchmarks: {benchmarks_count:,} | "
            f"Corporate Actions: {actions_count:,} | BIST30 Membership: {bist30_count:,} | Ingested Files Logged: {log_count:,}"
        )
        return {
            "layer": "bronze",
            "elapsed_sec": elapsed,
            "metrics": {
                "bronze_raw_trades": trades_count,
                "bronze_central_bank_rates": cbrt_rates_count,
                "bronze_bist_index_benchmarks": benchmarks_count,
                "bronze_corporate_actions": actions_count,
                "bronze_bist30_membership": bist30_count,
                "bronze_ingestion_log": log_count,
                "bronze_brokers": brokers_count,
                "bronze_instruments": instruments_count,
            },
            "details": {
                "trades": ingest_res,
                "central_bank_rates": cbrt_res,
                "benchmarks": bench_res,
                "corporate_actions": actions_res,
                "bist30_membership": bist30_res,
            },
            "status": "success",
        }

    def run_silver(self, target_dates: Optional[list[str]] = None) -> dict[str, Any]:
        """Execute Silver layer aggregations (daily broker summaries, overview, stock summary, sector, and intraday windows)."""
        logger.info(f"Starting Silver Layer Transformations (target_dates={target_dates})...")
        start_time = datetime.now()

        initialize_silver_schema(self.db)
        silver_res = self.silver_transformer.run_all(target_dates=target_dates)

        conn = self.db.get_connection()
        adj_periods_count = conn.execute("SELECT COUNT(*) FROM silver_corporate_action_adjustment_periods;").fetchone()[
            0
        ]
        broker_summary_count = conn.execute("SELECT COUNT(*) FROM silver_daily_broker_summary;").fetchone()[0]
        broker_overview_count = conn.execute("SELECT COUNT(*) FROM silver_daily_broker_overview;").fetchone()[0]
        stock_summary_count = conn.execute("SELECT COUNT(*) FROM silver_daily_stock_summary;").fetchone()[0]
        sector_summary_count = conn.execute("SELECT COUNT(*) FROM silver_daily_sector_summary;").fetchone()[0]
        win_broker_count = conn.execute("SELECT COUNT(*) FROM silver_intraday_broker_window_summary;").fetchone()[0]
        win_sector_count = conn.execute("SELECT COUNT(*) FROM silver_intraday_sector_window_summary;").fetchone()[0]
        macro_rates_count = conn.execute("SELECT COUNT(*) FROM silver_daily_macro_rates;").fetchone()[0]
        benchmark_count = conn.execute("SELECT COUNT(*) FROM silver_daily_benchmark_index;").fetchone()[0]
        thresholds_count = conn.execute("SELECT COUNT(*) FROM silver_bofa_historical_flow_thresholds;").fetchone()[0]
        fifo_daily_count = conn.execute("SELECT COUNT(*) FROM silver_broker_fifo_daily;").fetchone()[0]
        fifo_entries_count = conn.execute("SELECT COUNT(*) FROM silver_broker_fifo_lot_entries;").fetchone()[0]
        fifo_lots_count = conn.execute("SELECT COUNT(*) FROM silver_broker_fifo_lots;").fetchone()[0]
        fifo_realizations_count = conn.execute("SELECT COUNT(*) FROM silver_broker_fifo_lot_realizations;").fetchone()[
            0
        ]
        fifo_lifecycle_count = conn.execute("SELECT COUNT(*) FROM silver_broker_fifo_lot_lifecycle;").fetchone()[0]
        elapsed = (datetime.now() - start_time).total_seconds()

        logger.info(
            f"Silver Layer completed in {elapsed:.2f}s | "
            f"Adjustment Periods: {adj_periods_count:,} | Stock Summary: {stock_summary_count:,} | "
            f"Stock-Broker: {broker_summary_count:,} | Broker Overview: {broker_overview_count:,} | "
            f"Sector: {sector_summary_count:,} | Intraday Windows: {win_broker_count:,} | "
            f"Macro Rates: {macro_rates_count:,} | Benchmark Days: {benchmark_count:,} | "
            f"Threshold Profiles: {thresholds_count:,} | FIFO Daily: {fifo_daily_count:,} | "
            f"Lot Entries: {fifo_entries_count:,} | Active Lots: {fifo_lots_count:,}"
        )
        return {
            "layer": "silver",
            "elapsed_sec": elapsed,
            "metrics": {
                "silver_corporate_action_adjustment_periods": adj_periods_count,
                "silver_daily_stock_summary": stock_summary_count,
                "silver_daily_broker_summary": broker_summary_count,
                "silver_daily_broker_overview": broker_overview_count,
                "silver_daily_sector_summary": sector_summary_count,
                "silver_intraday_broker_window_summary": win_broker_count,
                "silver_intraday_sector_window_summary": win_sector_count,
                "silver_daily_macro_rates": macro_rates_count,
                "silver_daily_benchmark_index": benchmark_count,
                "silver_bofa_historical_flow_thresholds": thresholds_count,
                "silver_broker_fifo_daily": fifo_daily_count,
                "silver_broker_fifo_lot_entries": fifo_entries_count,
                "silver_broker_fifo_lots": fifo_lots_count,
                "silver_broker_fifo_lot_realizations": fifo_realizations_count,
                "silver_broker_fifo_lot_lifecycle": fifo_lifecycle_count,
            },
            "details": silver_res,
            "status": "success",
        }

    def run_gold(self, **kwargs: Any) -> dict[str, Any]:
        """Execute Gold layer feature engineering and institutional daily signals."""
        logger.info("Starting Gold Layer Institutional Signals...")
        start_time = datetime.now()

        initialize_gold_schema(self.db)
        gold_res = self.gold_engineer.run_all()

        # Update Gold Layer Tertip ML Forecasts & Walk-Forward Backtests for the Frontend
        try:
            logger.info("Updating Gold Layer Tertip ML Forecasts & Opportunity Hub...")
            from mdk_trading_oracle.data.gold.tertip_forecasts import update_gold_tertip_forecasts

            tertip_res = update_gold_tertip_forecasts(self.db)
            gold_res["tertip_forecasts"] = tertip_res
        except Exception as e_fc:
            logger.warning(f"Could not update Gold Tertip forecasts: {e_fc}")

        # Update Gold Layer Week Start Forecasts & Backtests for the Frontend
        try:
            logger.info("Updating Gold Layer Week Start Forecasts & Hub...")
            from mdk_trading_oracle.data.gold.week_start_forecasts import sync_week_start_forecasts_from_yaml

            ws_res = sync_week_start_forecasts_from_yaml(self.db)
            gold_res["week_start_forecasts"] = ws_res
        except Exception as e_ws:
            logger.warning(f"Could not update Gold Week Start forecasts: {e_ws}")

        # Update Gold Layer Multi-Horizon Forecasts & Backtests (3d, 5d, 10d, 15d, 30d)
        try:
            logger.info("Updating Gold Layer Multi-Horizon Forecasts (3d, 5d, 10d, 15d, 30d)...")
            from mdk_trading_oracle.data.gold.tertip_multi_horizon import (
                update_gold_tertip_multi_horizon_forecasts,
            )

            mh_res = update_gold_tertip_multi_horizon_forecasts(self.db)
            gold_res["multi_horizon_forecasts"] = mh_res
        except Exception as e_mh:
            logger.warning(f"Could not update Gold Multi-Horizon forecasts: {e_mh}")

        conn = self.db.get_connection()
        signals_count = conn.execute("SELECT COUNT(*) FROM gold_institutional_daily_signals;").fetchone()[0]
        elapsed = (datetime.now() - start_time).total_seconds()

        logger.info(f"Gold Layer completed in {elapsed:.2f}s | Signals: {signals_count:,}")
        return {
            "layer": "gold",
            "elapsed_sec": elapsed,
            "metrics": {
                "gold_institutional_daily_signals": signals_count,
            },
            "details": gold_res,
            "status": "success",
        }

    def run(
        self,
        target: Union[str, list[str]] = "all",
        raw_glob: Optional[str] = None,
        target_date: Optional[str] = None,
        target_month: Optional[str] = None,
        target_file: Optional[Union[str, Path]] = None,
        force: bool = False,
        sync_catalog: bool = False,
        resolve_dependencies: bool = True,
        print_summary: bool = True,
        **kwargs: Any,
    ) -> dict[str, Any]:
        """Execute the Medallion Pipeline for requested target layers."""
        pipeline_start = datetime.now()
        layers_to_run = self._resolve_layers(target, resolve_dependencies=resolve_dependencies)

        logger.info(
            f"Executing Medallion Pipeline DAG for layers: {layers_to_run} "
            f"(target_date={target_date}, target_month={target_month}, force={force}, sync_catalog={sync_catalog})"
        )

        results: dict[str, Any] = {}

        if sync_catalog and "catalog" not in layers_to_run:
            results["catalog"] = self.run_catalog_sync(raw_glob=raw_glob)

        for layer in layers_to_run:
            if layer == "catalog":
                results["catalog"] = self.run_catalog_sync(raw_glob=raw_glob)
            elif layer == "bronze":
                results["bronze"] = self.run_bronze(
                    raw_glob=raw_glob,
                    target_date=target_date,
                    target_month=target_month,
                    target_file=target_file,
                    force=force,
                )
            elif layer == "silver":
                silver_target_dates = [target_date] if target_date else None
                if not silver_target_dates and "bronze" in results:
                    bronze_ingested = results["bronze"].get("details", {}).get("trades", {}).get("ingested_dates")
                    if bronze_ingested:
                        silver_target_dates = bronze_ingested
                results["silver"] = self.run_silver(target_dates=silver_target_dates)
                # Clear Tertip ML Forecaster cache on new silver data
                try:
                    from mdk_trading_oracle.data.silver.tertip_ml_forecaster import clear_forecast_cache
                    clear_forecast_cache()
                except Exception as e_cache:
                    logger.debug(f"Could not clear tertip forecast cache: {e_cache}")
            elif layer == "gold":
                results["gold"] = self.run_gold()

        total_elapsed = (datetime.now() - pipeline_start).total_seconds()
        results["total_elapsed_sec"] = total_elapsed
        results["status"] = "success"

        if print_summary:
            all_executed_layers = list(results.keys())
            if "total_elapsed_sec" in all_executed_layers:
                all_executed_layers.remove("total_elapsed_sec")
            if "status" in all_executed_layers:
                all_executed_layers.remove("status")
            self._print_execution_table(results, all_executed_layers)

        return results

    def _print_execution_table(self, results: dict[str, Any], layers: list[str]) -> None:
        """Render a formatted execution table using Rich."""
        table = Table(
            title="⚡ Medallion Lakehouse Pipeline Execution Summary",
            title_style="bold cyan",
            border_style="cyan",
        )
        table.add_column("Layer / Stage", style="bold yellow")
        table.add_column("Table / Output", style="green")
        table.add_column("Entity / Row Count", justify="right", style="cyan")
        table.add_column("Duration", justify="right", style="magenta")
        table.add_column("Status", justify="center", style="bold green")

        for layer in layers:
            res = results.get(layer, {})
            duration = f"{res.get('elapsed_sec', 0.0):.2f}s"
            metrics = res.get("metrics", {})
            first = True
            for tbl, count in metrics.items():
                layer_label = layer.capitalize() if first else ""
                table.add_row(layer_label, tbl, f"{count:,}", duration if first else "", "✅ SUCCESS")
                first = False

        table.add_section()
        table.add_row(
            "TOTAL", "Full Pipeline Execution", "", f"{results.get('total_elapsed_sec', 0.0):.2f}s", "🚀 COMPLETE"
        )
        console.print(table)
