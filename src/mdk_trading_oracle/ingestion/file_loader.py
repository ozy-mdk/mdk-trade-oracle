"""Local file ingestor for Bronze layer supporting BIST raw formats and recursion."""

from pathlib import Path
from typing import Any, Dict, List, Optional, Union

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

    def ingest_bist_raw_csv_glob(self, glob_pattern: str, raw_source_label: str = "bist_raw_feed") -> Dict[str, Any]:
        """Ingest multiple BIST CSV files matching a glob pattern directly via DuckDB."""
        conn = self.db.get_connection()

        # Check if already ingested
        existing = conn.execute(
            "SELECT COUNT(*) FROM bronze_raw_trades WHERE raw_source = ?", [raw_source_label]
        ).fetchone()[0]
        if existing > 0:
            logger.info(f"Bronze layer already has {existing:,} trades for {raw_source_label}. Skipping re-ingestion.")
            return {
                "glob_pattern": glob_pattern,
                "rows_ingested": existing,
                "status": "already_ingested",
            }

        logger.info(f"Ingesting BIST trade CSVs matching glob: {glob_pattern}")

        # DuckDB can ingest all matched CSVs in parallel while normalizing columns
        query = f"""
            INSERT INTO bronze_raw_trades (
                trade_id, timestamp, symbol, price, volume, buyer_broker_id, seller_broker_id, raw_source
            )
            SELECT 
                md5(symbol || '_' || signal_time_text || '_' || CAST(price AS VARCHAR) || '_' || CAST(quantity AS VARCHAR) || '_' || buyer || '_' || seller) AS trade_id,
                TRY_CAST(signal_time_text AS TIMESTAMP) AS timestamp,
                CAST(symbol AS VARCHAR) AS symbol,
                CAST(price AS DOUBLE) AS price,
                CAST(quantity AS DOUBLE) AS volume,
                CAST(buyer AS VARCHAR) AS buyer_broker_id,
                CAST(seller AS VARCHAR) AS seller_broker_id,
                '{raw_source_label}' AS raw_source
            FROM read_csv_auto('{glob_pattern}', union_by_name=true, header=true)
            WHERE symbol IS NOT NULL AND price > 0;
        """
        conn.execute(query)

        count_res = conn.execute(
            "SELECT COUNT(*) FROM bronze_raw_trades WHERE raw_source = ?", [raw_source_label]
        ).fetchone()
        row_count = count_res[0] if count_res else 0

        logger.info(f"Successfully ingested {row_count:,} rows into Bronze via glob.")
        return {
            "glob_pattern": glob_pattern,
            "rows_ingested": row_count,
            "status": "success",
        }

    def ingest(self, source_path: Union[str, Path], **kwargs) -> Dict[str, Any]:
        """Ingest a single CSV or Parquet file into `bronze_raw_trades`."""
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
            # Check CSV header columns to detect schema
            sample_df = conn.execute(f"SELECT * FROM read_csv_auto('{path.as_posix()}', limit=2);").fetch_df()
            cols = [c.lower() for c in sample_df.columns]

            if "signal_time_text" in cols and "buyer" in cols and "seller" in cols:
                # Real BIST CSV format
                query = f"""
                    INSERT INTO bronze_raw_trades (
                        trade_id, timestamp, symbol, price, volume, buyer_broker_id, seller_broker_id, raw_source
                    )
                    SELECT 
                        md5(symbol || '_' || signal_time_text || '_' || CAST(price AS VARCHAR) || '_' || CAST(quantity AS VARCHAR) || '_' || buyer || '_' || seller) AS trade_id,
                        TRY_CAST(signal_time_text AS TIMESTAMP) AS timestamp,
                        CAST(symbol AS VARCHAR) AS symbol,
                        CAST(price AS DOUBLE) AS price,
                        CAST(quantity AS DOUBLE) AS volume,
                        CAST(buyer AS VARCHAR) AS buyer_broker_id,
                        CAST(seller AS VARCHAR) AS seller_broker_id,
                        '{path.name}' AS raw_source
                    FROM read_csv_auto('{path.as_posix()}', header=true)
                    WHERE symbol IS NOT NULL AND price > 0;
                """
            else:
                # Standard format
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

        logger.info(f"Successfully ingested {row_count:,} rows from {path.name} into Bronze.")
        return {
            "source_file": path.name,
            "rows_ingested": row_count,
            "status": "success",
        }

    def ingest_all_bronze(self, target_dir: Optional[Path] = None) -> List[Dict[str, Any]]:
        """Ingest all pending raw files recursively in `data/00_raw_data/` or a target folder."""
        search_dir = target_dir or (self.settings.raw_data_dir if self.settings.raw_data_dir.exists() else self.settings.bronze_dir)
        results = []

        # If a raw_csv folder exists with BIST daily partitions, use high-speed glob
        bist_csvs = list(search_dir.rglob("raw_csv/**/*.csv"))
        if bist_csvs:
            glob_path = (search_dir / "2026/03_march/raw_csv/**/*.csv").as_posix()
            res = self.ingest_bist_raw_csv_glob(glob_path, raw_source_label="bist_2026_03_march")
            results.append(res)
            return results

        # Otherwise ingest individual files
        files = sorted(
            list(search_dir.rglob("*.csv")) + list(search_dir.rglob("*.parquet"))
        )
        if not files:
            logger.warning(f"No CSV or Parquet files found in {search_dir}")
            return results

        for file_path in files:
            res = self.ingest(file_path)
            results.append(res)

        return results
