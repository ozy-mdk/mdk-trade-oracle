"""Medallion Lakehouse Pipeline Orchestrator (Bronze -> Silver -> Gold)."""

from datetime import datetime
from typing import Any, Optional, Union

from rich.console import Console
from rich.table import Table

from mdk_trading_oracle.core.config import get_settings
from mdk_trading_oracle.core.db import DuckDBManager
from mdk_trading_oracle.core.logger import get_logger
from mdk_trading_oracle.data.bronze import BronzeIngestor, initialize_bronze_schema
from mdk_trading_oracle.data.gold import GoldFeatureEngineer, initialize_gold_schema
from mdk_trading_oracle.data.silver import SilverTransformer, initialize_silver_schema

logger = get_logger("mdk_oracle.data.pipeline")
console = Console()


class MedallionPipeline:
    """Orchestrates Bronze, Silver, and Gold transformations with automatic dependency DAG resolution."""

    VALID_LAYERS = ["bronze", "silver", "gold", "all"]

    def __init__(self, db: Optional[DuckDBManager] = None):
        self.db = db or DuckDBManager()
        self.settings = get_settings()
        self.bronze_ingestor = BronzeIngestor(self.db)
        self.silver_transformer = SilverTransformer(self.db)
        self.gold_engineer = GoldFeatureEngineer(self.db)

    def _resolve_layers(self, target: Union[str, list[str]], resolve_dependencies: bool = True) -> list[str]:
        """Resolve requested target(s) into ordered execution layers [bronze, silver, gold]."""
        if isinstance(target, str):
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
        if "gold" in layers:
            for dep in ["bronze", "silver", "gold"]:
                if dep not in resolved:
                    resolved.append(dep)
        elif "silver" in layers:
            for dep in ["bronze", "silver"]:
                if dep not in resolved:
                    resolved.append(dep)
        elif "bronze" in layers:
            resolved.append("bronze")

        return resolved

    def run_bronze(self, raw_glob: Optional[str] = None, raw_source_label: str = "bist_2026_03_march") -> dict[str, Any]:
        """Execute Bronze schema initialization and raw data ingestion."""
        logger.info("Starting Bronze Layer Ingestion...")
        start_time = datetime.now()

        initialize_bronze_schema(self.db)
        target_glob = raw_glob or (self.settings.raw_data_dir / "2026/03_march/raw_csv/**/*.csv").as_posix()
        ingest_res = self.bronze_ingestor.ingest_bist_raw_csv_glob(
            glob_pattern=target_glob,
            raw_source_label=raw_source_label,
        )

        conn = self.db.get_connection()
        trades_count = conn.execute("SELECT COUNT(*) FROM bronze_raw_trades;").fetchone()[0]
        brokers_count = conn.execute("SELECT COUNT(*) FROM bronze_brokers;").fetchone()[0]
        instruments_count = conn.execute("SELECT COUNT(*) FROM bronze_instruments;").fetchone()[0]
        elapsed = (datetime.now() - start_time).total_seconds()

        logger.info(f"Bronze Layer completed in {elapsed:.2f}s | Raw Trades: {trades_count:,}")
        return {
            "layer": "bronze",
            "elapsed_sec": elapsed,
            "metrics": {
                "bronze_raw_trades": trades_count,
                "bronze_brokers": brokers_count,
                "bronze_instruments": instruments_count,
            },
            "details": ingest_res,
            "status": "success",
        }

    def run_silver(self) -> dict[str, Any]:
        """Execute Silver layer aggregations (daily broker summaries and market OHLCV)."""
        logger.info("Starting Silver Layer Transformations...")
        start_time = datetime.now()

        initialize_silver_schema(self.db)
        silver_res = self.silver_transformer.run_all()

        conn = self.db.get_connection()
        broker_summary_count = conn.execute("SELECT COUNT(*) FROM silver_daily_broker_summary;").fetchone()[0]
        market_daily_count = conn.execute("SELECT COUNT(*) FROM silver_market_daily;").fetchone()[0]
        elapsed = (datetime.now() - start_time).total_seconds()

        logger.info(
            f"Silver Layer completed in {elapsed:.2f}s | "
            f"Broker Summaries: {broker_summary_count:,} | Market Daily: {market_daily_count:,}"
        )
        return {
            "layer": "silver",
            "elapsed_sec": elapsed,
            "metrics": {
                "silver_daily_broker_summary": broker_summary_count,
                "silver_market_daily": market_daily_count,
            },
            "details": silver_res,
            "status": "success",
        }

    def run_gold(self) -> dict[str, Any]:
        """Execute Gold layer feature engineering and institutional flow signals."""
        logger.info("Starting Gold Layer Feature Engineering...")
        start_time = datetime.now()

        initialize_gold_schema(self.db)
        gold_res = self.gold_engineer.run_all()

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
        resolve_dependencies: bool = True,
        print_summary: bool = True,
    ) -> dict[str, Any]:
        """Execute the Medallion Pipeline for requested target layers."""
        pipeline_start = datetime.now()
        layers_to_run = self._resolve_layers(target, resolve_dependencies=resolve_dependencies)

        logger.info(f"Executing Medallion Pipeline DAG for layers: {layers_to_run}")

        results: dict[str, Any] = {}

        for layer in layers_to_run:
            if layer == "bronze":
                results["bronze"] = self.run_bronze(raw_glob=raw_glob)
            elif layer == "silver":
                results["silver"] = self.run_silver()
            elif layer == "gold":
                results["gold"] = self.run_gold()

        total_elapsed = (datetime.now() - pipeline_start).total_seconds()
        results["total_elapsed_sec"] = total_elapsed
        results["status"] = "success"

        if print_summary:
            self._print_execution_table(results, layers_to_run)

        return results

    def _print_execution_table(self, results: dict[str, Any], layers: list[str]) -> None:
        """Render a formatted execution table using Rich."""
        table = Table(
            title="⚡ Medallion Lakehouse Pipeline Execution Summary",
            title_style="bold cyan",
            border_style="cyan",
        )
        table.add_column("Layer", style="bold yellow")
        table.add_column("Table / Output", style="green")
        table.add_column("Row Count", justify="right", style="cyan")
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
        table.add_row("TOTAL", "Full Pipeline Execution", "", f"{results.get('total_elapsed_sec', 0.0):.2f}s", "🚀 COMPLETE")
        console.print(table)
