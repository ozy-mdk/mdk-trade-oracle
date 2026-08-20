"""Bronze Layer Ingestor for BIST raw files (CSVs and Parquets).

Supports:
- Incremental file discovery & ingestion (only ingest new or modified files)
- Ingestion tracking log (`bronze_ingestion_log`)
- Selective partition updates (by single date `YYYY-MM-DD`, month `YYYY-MM`, or individual file)
- Multi-threaded DuckDB bulk ingestion
- Resumable annual ZIP ingestion without permanent extraction
"""

import csv
import re
import tempfile
import zipfile
from datetime import date
from pathlib import Path, PurePosixPath
from typing import Any, BinaryIO, Dict, List, Optional, Tuple, Union

from mdk_trading_oracle.core.config import get_settings
from mdk_trading_oracle.core.db import DuckDBManager
from mdk_trading_oracle.core.logger import get_logger
from mdk_trading_oracle.data.bronze.schema import initialize_bronze_schema

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
    """Ingests raw trade CSV and Parquet files into the DuckDB `bronze_raw_trades` table with tracking."""

    def __init__(self, db: DuckDBManager):
        self.db = db
        self.settings = get_settings()

    def _extract_file_metadata(self, file_path: Path) -> Dict[str, Any]:
        """Extract partition information, size, mtime, and trade date from a raw file path."""
        path_str = file_path.resolve().as_posix()
        stat = file_path.stat()
        file_size = stat.st_size
        file_mtime = stat.st_mtime

        # Extract date (e.g. '2026-03-09' or '20260309')
        date_match = re.search(r"(\d{4}-\d{2}-\d{2})", path_str)
        if date_match:
            trade_date_str = date_match.group(1)
        else:
            compact_date = re.search(r"/(\d{8})/", path_str)
            if compact_date:
                raw_d = compact_date.group(1)
                trade_date_str = f"{raw_d[:4]}-{raw_d[4:6]}-{raw_d[6:8]}"
            else:
                trade_date_str = None

        # Extract year_month (e.g. '2026-03' or '2026/03_march')
        month_match = re.search(r"(\d{4})/(\d{2}[_\w]*)", path_str)
        if month_match:
            year_month = f"{month_match.group(1)}-{month_match.group(2)}"
        elif trade_date_str:
            year_month = trade_date_str[:7]
        else:
            year_month = "unknown"

        return {
            "file_path": path_str,
            "file_name": file_path.name,
            "file_size_bytes": file_size,
            "file_mtime_epoch": file_mtime,
            "trade_date": trade_date_str,
            "year_month": year_month,
            "extension": file_path.suffix.lower(),
        }

    def discover_raw_files(self, search_dir: Optional[Path] = None) -> List[Dict[str, Any]]:
        """Recursively scan directory for raw CSV and Parquet files and extract metadata."""
        base_dir = search_dir or self.settings.raw_data_dir
        if not base_dir.exists():
            logger.warning(f"Raw data directory does not exist: {base_dir}")
            return []

        discovered: List[Dict[str, Any]] = []
        for ext in ["*.csv", "*.parquet", "*.txt"]:
            for f in base_dir.rglob(ext):
                # Ignore hidden files, temporary files, and mysql dump directories
                if f.name.startswith(".") or "/mysql/" in f.as_posix():
                    continue
                discovered.append(self._extract_file_metadata(f))

        logger.debug(f"Discovered {len(discovered)} raw trade files under {base_dir}")
        return sorted(discovered, key=lambda x: x["file_path"])

    def _ensure_ingestion_log_synced(self, discovered_files: List[Dict[str, Any]]) -> None:
        """Backfill `bronze_ingestion_log` if database already contains legacy historical trades."""
        conn = self.db.get_connection()
        initialize_bronze_schema(self.db)

        log_count = conn.execute("SELECT COUNT(*) FROM bronze_ingestion_log;").fetchone()[0]
        trades_count = conn.execute("SELECT COUNT(*) FROM bronze_raw_trades;").fetchone()[0]

        if log_count == 0 and trades_count > 0:
            logger.info("Syncing existing historical trades into `bronze_ingestion_log`...")
            # Check if historical load was 'bist_2026_03_march'
            legacy_sources = [
                r[0] for r in conn.execute("SELECT DISTINCT raw_source FROM bronze_raw_trades;").fetchall()
            ]
            for meta in discovered_files:
                conn.execute("""
                    INSERT OR IGNORE INTO bronze_ingestion_log (
                        file_path, file_name, file_size_bytes, file_mtime_epoch, trade_date, year_month, rows_ingested, raw_source_label
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?);
                """, [
                    meta["file_path"],
                    meta["file_name"],
                    meta["file_size_bytes"],
                    meta["file_mtime_epoch"],
                    meta["trade_date"],
                    meta["year_month"],
                    0,
                    legacy_sources[0] if legacy_sources else "historical_feed",
                ])
            logger.info(f"Backfilled {len(discovered_files)} file entries into `bronze_ingestion_log`.")

    def get_pending_files(self, discovered_files: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Filter discovered files down to only those not yet ingested or modified since last ingestion."""
        initialize_bronze_schema(self.db)
        self._ensure_ingestion_log_synced(discovered_files)

        conn = self.db.get_connection()
        logged_rows = conn.execute(
            "SELECT file_path, file_size_bytes, file_mtime_epoch FROM bronze_ingestion_log;"
        ).fetchall()

        logged_map: Dict[str, Tuple[int, float]] = {
            r[0]: (int(r[1]), float(r[2])) for r in logged_rows
        }

        pending: List[Dict[str, Any]] = []
        for meta in discovered_files:
            fp = meta["file_path"]
            if fp not in logged_map:
                pending.append(meta)
            else:
                log_size, log_mtime = logged_map[fp]
                # Check if file size or mtime has changed
                if meta["file_size_bytes"] != log_size or abs(meta["file_mtime_epoch"] - log_mtime) > 1e-2:
                    pending.append(meta)

        return pending

    def _ingest_file_batch(self, file_metas: List[Dict[str, Any]], batch_size: int = 50) -> int:
        """Ingest a batch of files in parallel via DuckDB multi-file parser and log entries."""
        if not file_metas:
            return 0

        conn = self.db.get_connection()
        total_rows_inserted = 0

        # Group by extension
        csv_metas = [m for m in file_metas if m["extension"] in [".csv", ".txt"]]
        parquet_metas = [m for m in file_metas if m["extension"] == ".parquet"]

        # 1. Process CSV files in chunks
        for i in range(0, len(csv_metas), batch_size):
            chunk = csv_metas[i : i + batch_size]
            paths = [m["file_path"] for m in chunk]

            # Ingest chunk into bronze_raw_trades
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
                    filename AS raw_source
                FROM read_csv_auto({paths}, union_by_name=true, header=true, filename=true)
                WHERE symbol IS NOT NULL AND price > 0;
            """
            conn.execute(query)

            # Record each ingested file in bronze_ingestion_log
            for meta in chunk:
                conn.execute("""
                    INSERT OR REPLACE INTO bronze_ingestion_log (
                        file_path, file_name, file_size_bytes, file_mtime_epoch, trade_date, year_month, rows_ingested, raw_source_label
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?);
                """, [
                    meta["file_path"],
                    meta["file_name"],
                    meta["file_size_bytes"],
                    meta["file_mtime_epoch"],
                    meta["trade_date"],
                    meta["year_month"],
                    0,
                    meta["file_name"],
                ])

            total_rows_inserted += len(chunk)
            logger.debug(f"Ingested CSV chunk {i // batch_size + 1}/{(len(csv_metas) - 1) // batch_size + 1} ({len(chunk)} files)")

        # 2. Process Parquet files
        for meta in parquet_metas:
            p = meta["file_path"]
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
                    '{meta["file_name"]}' AS raw_source
                FROM read_parquet('{p}');
            """
            conn.execute(query)
            conn.execute("""
                INSERT OR REPLACE INTO bronze_ingestion_log (
                    file_path, file_name, file_size_bytes, file_mtime_epoch, trade_date, year_month, rows_ingested, raw_source_label
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?);
            """, [
                meta["file_path"],
                meta["file_name"],
                meta["file_size_bytes"],
                meta["file_mtime_epoch"],
                meta["trade_date"],
                meta["year_month"],
                0,
                meta["file_name"],
            ])
            total_rows_inserted += 1

        return total_rows_inserted

    def ingest_incremental(self, search_dir: Optional[Path] = None) -> Dict[str, Any]:
        """Discover all raw files and ingest only newly arrived or modified files."""
        initialize_bronze_schema(self.db)
        discovered = self.discover_raw_files(search_dir)
        pending = self.get_pending_files(discovered)

        if not pending:
            logger.info(f"Bronze layer is up to date. Discovered {len(discovered)} files, 0 pending ingestion.")
            return {
                "status": "up_to_date",
                "total_discovered": len(discovered),
                "pending_files": 0,
                "rows_ingested": 0,
            }

        logger.info(f"Discovered {len(pending)} new/modified files to ingest into Bronze layer.")
        self._ingest_file_batch(pending)

        conn = self.db.get_connection()
        total_trades = conn.execute("SELECT COUNT(*) FROM bronze_raw_trades;").fetchone()[0]

        logger.info(f"Successfully ingested {len(pending)} files. Total Bronze trades: {total_trades:,}")
        return {
            "status": "success",
            "total_discovered": len(discovered),
            "pending_files": len(pending),
            "files_ingested": len(pending),
            "total_trades": total_trades,
        }

    def ingest_date(self, target_date: Union[str, date], search_dir: Optional[Path] = None) -> Dict[str, Any]:
        """Atomically delete and re-ingest all trade data for a specific single date (YYYY-MM-DD)."""
        date_str = target_date.strftime("%Y-%m-%d") if isinstance(target_date, date) else str(target_date)
        initialize_bronze_schema(self.db)
        conn = self.db.get_connection()

        logger.info(f"Targeted Date Ingestion requested for date: {date_str}")

        # 1. Delete existing rows for this date
        deleted_trades = conn.execute(
            "SELECT COUNT(*) FROM bronze_raw_trades WHERE CAST(timestamp AS DATE) = CAST(? AS DATE);", [date_str]
        ).fetchone()[0]
        conn.execute("DELETE FROM bronze_raw_trades WHERE CAST(timestamp AS DATE) = CAST(? AS DATE);", [date_str])
        conn.execute("DELETE FROM bronze_ingestion_log WHERE trade_date = CAST(? AS DATE) OR file_path LIKE ?;", [date_str, f"%{date_str}%"])

        logger.info(f"Cleared {deleted_trades:,} previous trades for date {date_str}.")

        # 2. Discover files matching target_date
        all_files = self.discover_raw_files(search_dir)
        target_files = [
            f for f in all_files if f["trade_date"] == date_str or f"/{date_str}/" in f["file_path"]
        ]

        if not target_files:
            logger.warning(f"No raw files found for date: {date_str}")
            return {
                "status": "not_found",
                "target_date": date_str,
                "files_ingested": 0,
                "deleted_trades": deleted_trades,
            }

        logger.info(f"Found {len(target_files)} raw files for date {date_str}. Ingesting...")
        self._ingest_file_batch(target_files)

        new_count = conn.execute(
            "SELECT COUNT(*) FROM bronze_raw_trades WHERE CAST(timestamp AS DATE) = CAST(? AS DATE);", [date_str]
        ).fetchone()[0]

        logger.info(f"Successfully re-ingested {new_count:,} trades for date {date_str} across {len(target_files)} files.")
        return {
            "status": "success",
            "target_date": date_str,
            "files_ingested": len(target_files),
            "deleted_previous_trades": deleted_trades,
            "new_trades_ingested": new_count,
        }

    def ingest_month(self, year_month: str, search_dir: Optional[Path] = None) -> Dict[str, Any]:
        """Atomically delete and re-ingest all trade data for a specific month (e.g. '2026-03' or '03_march')."""
        initialize_bronze_schema(self.db)
        conn = self.db.get_connection()

        logger.info(f"Targeted Month Ingestion requested for: {year_month}")

        # Match files for this month
        all_files = self.discover_raw_files(search_dir)
        target_files = [
            f for f in all_files if year_month in f["year_month"] or f"/{year_month}/" in f["file_path"] or year_month in f["file_path"]
        ]

        if not target_files:
            logger.warning(f"No raw files found matching month: {year_month}")
            return {"status": "not_found", "year_month": year_month, "files_ingested": 0}

        # Collect dates involved
        dates = sorted(list({f["trade_date"] for f in target_files if f["trade_date"]}))

        # Delete existing data for matching dates or raw_sources
        if dates:
            conn.execute(
                f"DELETE FROM bronze_raw_trades WHERE CAST(timestamp AS DATE) >= '{dates[0]}' AND CAST(timestamp AS DATE) <= '{dates[-1]}';"
            )
            conn.execute(
                f"DELETE FROM bronze_ingestion_log WHERE trade_date >= '{dates[0]}' AND trade_date <= '{dates[-1]}';"
            )

        logger.info(f"Cleared existing records. Ingesting {len(target_files)} files for {year_month}...")
        self._ingest_file_batch(target_files)

        total_ingested = conn.execute("SELECT COUNT(*) FROM bronze_raw_trades;").fetchone()[0]
        return {
            "status": "success",
            "year_month": year_month,
            "files_ingested": len(target_files),
            "trading_days": len(dates),
            "total_bronze_trades": total_ingested,
        }

    def ingest_file(self, source_path: Union[str, Path], force: bool = False) -> Dict[str, Any]:
        """Ingest a single CSV or Parquet file into `bronze_raw_trades`."""
        path = Path(source_path).resolve()
        if not path.exists():
            raise FileNotFoundError(f"Source file does not exist: {path}")

        initialize_bronze_schema(self.db)
        conn = self.db.get_connection()
        meta = self._extract_file_metadata(path)

        if force:
            conn.execute("DELETE FROM bronze_raw_trades WHERE raw_source = ? OR raw_source = ?;", [path.as_posix(), path.name])
            conn.execute("DELETE FROM bronze_ingestion_log WHERE file_path = ?;", [path.as_posix()])

        self._ingest_file_batch([meta])

        row_count = conn.execute(
            "SELECT COUNT(*) FROM bronze_raw_trades WHERE raw_source = ? OR raw_source = ?;",
            [path.as_posix(), path.name]
        ).fetchone()[0]

        logger.info(f"Successfully ingested {row_count:,} rows from {path.name} into Bronze.")
        return {
            "source_file": path.name,
            "file_path": path.as_posix(),
            "rows_ingested": row_count,
            "status": "success",
        }

    def ingest_bist_raw_csv_glob(self, glob_pattern: str, raw_source_label: str = "bist_raw_feed") -> Dict[str, Any]:
        """Ingest multiple BIST CSV files matching a glob pattern directly via DuckDB multi-threaded parser."""
        initialize_bronze_schema(self.db)
        conn = self.db.get_connection()

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
        members: List[Tuple[zipfile.ZipInfo, int]],
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
    ) -> Dict[str, Any]:
        """Stream annual BIST ZIP archives into Bronze without permanent extraction."""
        initialize_bronze_schema(self.db)
        source_dir = Path(archive_dir).resolve()
        if not source_dir.is_dir():
            raise FileNotFoundError(f"ZIP archive directory does not exist: {source_dir}")
        if batch_mb <= 0:
            raise ValueError("batch_mb must be positive")
        if limit_files is not None and limit_files <= 0:
            raise ValueError("limit_files must be positive when provided")

        staging_dir = (
            Path(temp_dir).resolve()
            if temp_dir
            else self.settings.database_dir / "tmp" / "zip_ingest"
        )
        staging_dir.mkdir(parents=True, exist_ok=True)
        conn = self.db.get_connection()
        loaded = {
            (row[0], row[1])
            for row in conn.execute(
                "SELECT archive_name, member_name FROM bronze_loaded_files;"
            ).fetchall()
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
                    batch: List[Tuple[zipfile.ZipInfo, int]] = []
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
                            "ZIP Bronze batch %s committed | archive=%s files=%s "
                            "rows=%s total_files=%s",
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

    def ingest_all(self, target_dir: Optional[Path] = None, force: bool = False) -> Dict[str, Any]:
        """Ingest all discovered raw files (incremental by default, or full rebuild if force=True)."""
        initialize_bronze_schema(self.db)
        if force:
            conn = self.db.get_connection()
            logger.warning(
                "Force rebuild requested: Truncating Bronze trades and ingestion ledgers..."
            )
            conn.execute("DELETE FROM bronze_raw_trades;")
            conn.execute("DELETE FROM bronze_ingestion_log;")
            conn.execute("DELETE FROM bronze_loaded_files;")

        return self.ingest_incremental(target_dir)
