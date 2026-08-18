"""Bronze Layer Ingestor for BIST raw files (CSVs and Parquets)."""

from pathlib import Path
from typing import Any, Optional, Union

from mdk_trading_oracle.core.config import get_settings
from mdk_trading_oracle.core.db import DuckDBManager
from mdk_trading_oracle.core.logger import get_logger

logger = get_logger("mdk_oracle.data.bronze.ingestor")


class BronzeIngestor:
    """Ingests raw trade CSV and Parquet files into the DuckDB `bronze_raw_trades` table."""

    def __init__(self, db: DuckDBManager):
        self.db = db
        self.settings = get_settings()

    def ingest_bist_raw_csv_glob(self, glob_pattern: str, raw_source_label: str = "bist_raw_feed") -> dict[str, Any]:
        """Ingest multiple BIST CSV files matching a glob pattern directly via DuckDB multi-threaded parser."""
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

        # DuckDB ingests all matched CSVs in parallel while normalizing columns
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

    def ingest_file(self, source_path: Union[str, Path]) -> dict[str, Any]:
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
            sample_df = conn.execute(f"SELECT * FROM read_csv_auto('{path.as_posix()}', limit=2);").fetch_df()
            cols = [c.lower() for c in sample_df.columns]

            if "signal_time_text" in cols and "buyer" in cols and "seller" in cols:
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

    def ingest_all(self, target_dir: Optional[Path] = None) -> list[dict[str, Any]]:
        """Ingest all pending raw files in `00_raw_data/` or a custom directory."""
        search_dir = target_dir or self.settings.raw_data_dir
        results = []

        bist_csvs = list(search_dir.rglob("raw_csv/**/*.csv"))
        if bist_csvs:
            glob_path = (search_dir / "2026/03_march/raw_csv/**/*.csv").as_posix()
            res = self.ingest_bist_raw_csv_glob(glob_path, raw_source_label="bist_2026_03_march")
            results.append(res)
            return results

        files = sorted(list(search_dir.rglob("*.csv")) + list(search_dir.rglob("*.parquet")))
        if not files:
            logger.warning(f"No CSV or Parquet files found in {search_dir}")
            return results

        for file_path in files:
            res = self.ingest_file(file_path)
            results.append(res)

        return results
