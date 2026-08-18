"""Local file ingestor for Bronze layer."""

from pathlib import Path
from typing import Any, Dict, List, Union
import polars as pl
from mdk_trading_oracle.core.config import get_settings
from mdk_trading_oracle.core.db import DuckDBManager
from mdk_trading_oracle.core.logger import get_logger
from mdk_trading_oracle.ingestion.base import BaseIngestor

logger = get_logger("mdk_oracle.ingestion")


class FileIngestor(BaseIngestor):
    """Loads CSV and Parquet trade dump files into the Bronze table."""

    def __init__(self, db_manager: DuckDBManager):
        super().__init__(db_manager)
        self.settings = get_settings()

    def ingest(self, source_path: Union[str, Path], **kwargs) -> Dict[str, Any]:
        """Ingest a CSV or Parquet file into `bronze_raw_trades`."""
        path = Path(source_path)
        if not path.exists():
            raise FileNotFoundError(f"Source file does not exist: {path}")

        conn = self.db.get_connection()
        file_ext = path.suffix.lower()

        logger.info(f"Ingesting raw data from: {path.name}")

        if file_ext == ".parquet":
            query = f"""
                INSERT INTO bronze_raw_trades (
                    trade_id, timestamp, symbol, price, volume, buyer_broker_id, seller_broker_id, raw_source
                )
                SELECT 
                    CAST(trade_id AS VARCHAR),
                    CAST(timestamp AS TIMESTAMP),
                    CAST(symbol AS VARCHAR),
                    CAST(price AS DOUBLE),
                    CAST(volume AS DOUBLE),
                    CAST(buyer_broker_id AS VARCHAR),
                    CAST(seller_broker_id AS VARCHAR),
                    '{path.name}' AS raw_source
                FROM read_parquet('{path.as_posix()}');
            """
        elif file_ext in [".csv", ".txt"]:
            query = f"""
                INSERT INTO bronze_raw_trades (
                    trade_id, timestamp, symbol, price, volume, buyer_broker_id, seller_broker_id, raw_source
                )
                SELECT 
                    CAST(trade_id AS VARCHAR),
                    CAST(timestamp AS TIMESTAMP),
                    CAST(symbol AS VARCHAR),
                    CAST(price AS DOUBLE),
                    CAST(volume AS DOUBLE),
                    CAST(buyer_broker_id AS VARCHAR),
                    CAST(seller_broker_id AS VARCHAR),
                    '{path.name}' AS raw_source
                FROM read_csv_auto('{path.as_posix()}');
            """
        else:
            raise ValueError(f"Unsupported file format: {file_ext}")

        conn.execute(query)
        count_res = conn.execute(
            "SELECT COUNT(*) FROM bronze_raw_trades WHERE raw_source = ?", [path.name]
        ).fetchone()
        row_count = count_res[0] if count_res else 0

        logger.info(f"Successfully ingested {row_count} rows from {path.name} into Bronze.")
        return {
            "source_file": path.name,
            "rows_ingested": row_count,
            "status": "success",
        }

    def ingest_all_bronze(self) -> List[Dict[str, Any]]:
        """Ingest all pending raw files in `data/01_bronze/`."""
        bronze_dir = self.settings.bronze_dir
        results = []
        files = sorted(
            list(bronze_dir.glob("*.csv")) + list(bronze_dir.glob("*.parquet"))
        )
        if not files:
            logger.warning(f"No CSV or Parquet files found in {bronze_dir}")
            return results

        for file_path in files:
            res = self.ingest(file_path)
            results.append(res)

        return results
