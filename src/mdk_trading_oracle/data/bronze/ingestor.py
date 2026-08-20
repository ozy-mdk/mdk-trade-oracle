"""Bronze Layer Ingestor for BIST raw files (CSVs and Parquets)."""

import csv
import tempfile
import zipfile
from pathlib import Path, PurePosixPath
from typing import Any, BinaryIO, Optional, Union

from mdk_trading_oracle.core.config import get_settings
from mdk_trading_oracle.core.db import DuckDBManager
from mdk_trading_oracle.core.logger import get_logger

logger = get_logger("mdk_oracle.data.bronze.ingestor")

EXPECTED_BIST_HEADER = [
    "symbol",
    "signal_time_text",
    "price",
    "quantity",
    "bidask",
    "buyer",
    "seller",
]


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
                trade_id, timestamp, symbol, price, volume, bidask,
                buyer_broker_id, seller_broker_id, raw_source
            )
            SELECT 
                md5(symbol || '_' || signal_time_text || '_' || CAST(price AS VARCHAR) || '_' || CAST(quantity AS VARCHAR) || '_' || buyer || '_' || seller) AS trade_id,
                TRY_CAST(signal_time_text AS TIMESTAMP) AS timestamp,
                CAST(symbol AS VARCHAR) AS symbol,
                CAST(price AS DOUBLE) AS price,
                CAST(quantity AS BIGINT) AS volume,
                NULLIF(CAST(bidask AS VARCHAR), '') AS bidask,
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
                    trade_id, timestamp, symbol, price, volume, bidask,
                    buyer_broker_id, seller_broker_id, raw_source
                )
                SELECT 
                    CAST(trade_id AS VARCHAR),
                    CAST(timestamp AS TIMESTAMP),
                    CAST(symbol AS VARCHAR),
                    CAST(price AS DOUBLE),
                    CAST(volume AS BIGINT),
                    NULL,
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
                        trade_id, timestamp, symbol, price, volume, bidask,
                        buyer_broker_id, seller_broker_id, raw_source
                    )
                    SELECT 
                        md5(symbol || '_' || signal_time_text || '_' || CAST(price AS VARCHAR) || '_' || CAST(quantity AS VARCHAR) || '_' || buyer || '_' || seller) AS trade_id,
                        TRY_CAST(signal_time_text AS TIMESTAMP) AS timestamp,
                        CAST(symbol AS VARCHAR) AS symbol,
                        CAST(price AS DOUBLE) AS price,
                        CAST(quantity AS BIGINT) AS volume,
                        NULLIF(CAST(bidask AS VARCHAR), '') AS bidask,
                        CAST(buyer AS VARCHAR) AS buyer_broker_id,
                        CAST(seller AS VARCHAR) AS seller_broker_id,
                        '{path.name}' AS raw_source
                    FROM read_csv_auto('{path.as_posix()}', header=true)
                    WHERE symbol IS NOT NULL AND price > 0;
                """
            else:
                query = f"""
                    INSERT INTO bronze_raw_trades (
                        trade_id, timestamp, symbol, price, volume, bidask,
                        buyer_broker_id, seller_broker_id, raw_source
                    )
                    SELECT 
                        CAST(trade_id AS VARCHAR),
                        CAST(timestamp AS TIMESTAMP),
                        CAST(symbol AS VARCHAR),
                        CAST(price AS DOUBLE),
                        CAST(volume AS BIGINT),
                        NULL,
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

    @staticmethod
    def _append_zip_member(
        archive: zipfile.ZipFile,
        info: zipfile.ZipInfo,
        output: BinaryIO,
    ) -> int:
        """Append one headerless CSV member and return its data-row count."""
        with archive.open(info) as source:
            header = source.readline().decode("utf-8-sig", "strict").strip("\r\n")
            if next(csv.reader([header])) != EXPECTED_BIST_HEADER:
                raise ValueError(f"Unexpected CSV header in {info.filename}: {header}")

            row_count = 0
            copied_any = False
            last_byte = b""
            while True:
                block = source.read(8 * 1024 * 1024)
                if not block:
                    break
                output.write(block)
                row_count += block.count(b"\n")
                copied_any = True
                last_byte = block[-1:]

            if copied_any and last_byte not in (b"\n", b"\r"):
                output.write(b"\n")
                row_count += 1
            return row_count

    def _ingest_zip_batch(
        self,
        archive_name: str,
        temp_path: Path,
        members: list[tuple[zipfile.ZipInfo, int]],
    ) -> int:
        """Commit a prepared CSV batch and its member ledger atomically."""
        conn = self.db.get_connection()
        escaped_path = temp_path.as_posix().replace("'", "''")
        escaped_archive = archive_name.replace("'", "''")
        query = f"""
            INSERT INTO bronze_raw_trades (
                trade_id, timestamp, symbol, price, volume, bidask,
                buyer_broker_id, seller_broker_id, raw_source
            )
            SELECT
                NULL AS trade_id,
                CAST(SUBSTR(signal_time_text, 1, 23) AS TIMESTAMP) AS timestamp,
                symbol,
                price,
                quantity AS volume,
                bidask,
                buyer AS buyer_broker_id,
                seller AS seller_broker_id,
                '{escaped_archive}' AS raw_source
            FROM read_csv(
                '{escaped_path}',
                header=true,
                columns={{
                    'symbol': 'VARCHAR',
                    'signal_time_text': 'VARCHAR',
                    'price': 'DECIMAL(20,6)',
                    'quantity': 'BIGINT',
                    'bidask': 'VARCHAR',
                    'buyer': 'VARCHAR',
                    'seller': 'VARCHAR'
                }},
                nullstr='',
                strict_mode=true
            )
            WHERE symbol IS NOT NULL AND price > 0;
        """

        conn.execute("BEGIN TRANSACTION;")
        try:
            inserted = conn.execute(query).fetchone()
            conn.executemany(
                """
                INSERT INTO bronze_loaded_files (
                    archive_name, member_name, crc32, byte_size, row_count
                ) VALUES (?, ?, ?, ?, ?);
                """,
                [
                    [archive_name, info.filename, info.CRC, info.file_size, row_count]
                    for info, row_count in members
                ],
            )
            conn.execute("COMMIT;")
        except Exception:
            conn.execute("ROLLBACK;")
            raise

        return int(inserted[0]) if inserted else sum(row_count for _, row_count in members)

    def ingest_bist_zip_archives(
        self,
        archive_dir: Union[str, Path],
        temp_dir: Optional[Union[str, Path]] = None,
        batch_mb: int = 512,
        limit_files: Optional[int] = None,
    ) -> dict[str, Any]:
        """Stream annual BIST ZIP archives into Bronze without permanent extraction."""
        source_dir = Path(archive_dir).resolve()
        if not source_dir.is_dir():
            raise FileNotFoundError(f"ZIP archive directory does not exist: {source_dir}")
        if batch_mb <= 0:
            raise ValueError("batch_mb must be positive")
        if limit_files is not None and limit_files <= 0:
            raise ValueError("limit_files must be positive when provided")

        staging_dir = Path(temp_dir).resolve() if temp_dir else self.settings.database_dir / "tmp" / "zip_ingest"
        staging_dir.mkdir(parents=True, exist_ok=True)
        conn = self.db.get_connection()
        loaded = {
            (row[0], row[1])
            for row in conn.execute("SELECT archive_name, member_name FROM bronze_loaded_files;").fetchall()
        }
        archives = sorted(source_dir.rglob("*.zip"))
        if not archives:
            raise FileNotFoundError(f"No ZIP archives found under: {source_dir}")

        target_bytes = batch_mb * 1024 * 1024
        processed_files = 0
        rows_ingested = 0
        batches = 0

        for archive_path in archives:
            with zipfile.ZipFile(archive_path) as archive:
                pending = [
                    info
                    for info in archive.infolist()
                    if not info.is_dir()
                    and not info.filename.startswith("__MACOSX/")
                    and PurePosixPath(info.filename).suffix.lower() == ".csv"
                    and (archive_path.name, info.filename) not in loaded
                ]
                member_index = 0
                while member_index < len(pending):
                    if limit_files is not None and processed_files >= limit_files:
                        break
                    temp_file = tempfile.NamedTemporaryFile(
                        prefix="mdk_zip_ingest_",
                        suffix=".csv",
                        dir=staging_dir,
                        delete=False,
                    )
                    temp_path = Path(temp_file.name)
                    batch: list[tuple[zipfile.ZipInfo, int]] = []
                    try:
                        temp_file.write((",".join(EXPECTED_BIST_HEADER) + "\n").encode("utf-8"))
                        while member_index < len(pending) and temp_file.tell() < target_bytes:
                            if limit_files is not None and processed_files + len(batch) >= limit_files:
                                break
                            info = pending[member_index]
                            row_count = self._append_zip_member(archive, info, temp_file)
                            batch.append((info, row_count))
                            member_index += 1
                        temp_file.close()
                        if not batch:
                            break

                        inserted = self._ingest_zip_batch(archive_path.name, temp_path, batch)
                        batches += 1
                        processed_files += len(batch)
                        rows_ingested += inserted
                        logger.info(
                            "ZIP Bronze batch %s committed | archive=%s files=%s rows=%s total_files=%s",
                            batches,
                            archive_path.name,
                            len(batch),
                            f"{inserted:,}",
                            f"{processed_files:,}",
                        )
                    finally:
                        if not temp_file.closed:
                            temp_file.close()
                        temp_path.unlink(missing_ok=True)

                if limit_files is not None and processed_files >= limit_files:
                    break

        total_rows = conn.execute("SELECT COUNT(*) FROM bronze_raw_trades;").fetchone()[0]
        total_files = conn.execute("SELECT COUNT(*) FROM bronze_loaded_files;").fetchone()[0]
        return {
            "archive_dir": str(source_dir),
            "processed_files": processed_files,
            "rows_ingested": rows_ingested,
            "batches": batches,
            "total_loaded_files": total_files,
            "total_bronze_rows": total_rows,
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
