"""Bronze Layer Ingestor for BIST raw files (CSVs and Parquets).

Supports:
- Incremental file discovery & ingestion (only ingest new or modified files)
- Ingestion tracking log (`bronze_ingestion_log`)
- Selective partition updates (by single date `YYYY-MM-DD`, month `YYYY-MM`, or individual file)
- Multi-threaded DuckDB bulk ingestion
"""

import re
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import pandas as pd

from mdk_trading_oracle.core.config import get_settings
from mdk_trading_oracle.core.db import DuckDBManager
from mdk_trading_oracle.core.logger import get_logger
from mdk_trading_oracle.data.bronze.schema import initialize_bronze_schema

logger = get_logger("mdk_oracle.data.bronze.ingestor")


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
                # Ignore hidden files, temporary files, mysql dump directories, and central bank rates
                path_str = f.as_posix()
                if (
                    f.name.startswith(".")
                    or "/mysql/" in path_str
                    or "/central_bank_interest_rates/" in path_str
                ):
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

    def ingest_all(self, target_dir: Optional[Path] = None, force: bool = False) -> Dict[str, Any]:
        """Ingest all discovered raw files (incremental by default, or full rebuild if force=True)."""
        initialize_bronze_schema(self.db)
        if force:
            conn = self.db.get_connection()
            logger.warning("Force rebuild requested: Truncating `bronze_raw_trades` and `bronze_ingestion_log`...")
            conn.execute("DELETE FROM bronze_raw_trades;")
            conn.execute("DELETE FROM bronze_ingestion_log;")

        return self.ingest_incremental(target_dir)

    def discover_central_bank_files(self, search_dir: Optional[Path] = None) -> List[Dict[str, Any]]:
        """Recursively scan directory for Central Bank interest rate files (.xlsx, .xls, .csv, .parquet)."""
        base_dir = search_dir or (self.settings.raw_data_dir / "central_bank_interest_rates")
        if not base_dir.exists():
            logger.debug(f"Central bank rates directory does not exist: {base_dir}")
            return []

        discovered: List[Dict[str, Any]] = []
        for ext in ["*.xlsx", "*.xls", "*.csv", "*.parquet"]:
            for f in base_dir.rglob(ext):
                if f.name.startswith("."):
                    continue
                path_str = f.resolve().as_posix()
                stat = f.stat()
                discovered.append({
                    "file_path": path_str,
                    "file_name": f.name,
                    "file_size_bytes": stat.st_size,
                    "file_mtime_epoch": stat.st_mtime,
                    "extension": f.suffix.lower(),
                })

        logger.debug(f"Discovered {len(discovered)} central bank rate files under {base_dir}")
        return sorted(discovered, key=lambda x: x["file_path"])

    def ingest_central_bank_rates(
        self,
        file_path: Optional[Union[str, Path]] = None,
        force: bool = False,
        sync_market_dates: bool = True,
    ) -> Dict[str, Any]:
        """Ingest Central Bank interest rate files into `bronze_central_bank_rates` with upserting and logging."""
        initialize_bronze_schema(self.db)
        conn = self.db.get_connection()

        if file_path:
            target_files = [Path(file_path)]
        else:
            disc = self.discover_central_bank_files()
            if not disc:
                logger.info("No central bank rate files found to ingest.")
                sync_res = {}
                if sync_market_dates:
                    sync_res = self.sync_central_bank_rates_to_market()
                return {"files_processed": 0, "rows_ingested": 0, "market_sync": sync_res, "status": "no_files"}

            if force:
                target_files = [Path(d["file_path"]) for d in disc]
            else:
                logged = {
                    r[0]: r[1]
                    for r in conn.execute(
                        "SELECT file_path, file_mtime_epoch FROM bronze_ingestion_log WHERE raw_source_label = 'cbrt_interest_rates';"
                    ).fetchall()
                }
                target_files = [
                    Path(d["file_path"])
                    for d in disc
                    if d["file_path"] not in logged or abs(logged[d["file_path"]] - d["file_mtime_epoch"]) > 1e-3
                ]

        if not target_files:
            logger.info("All central bank rate files are already up-to-date in Bronze.")
            sync_res = {}
            if sync_market_dates:
                sync_res = self.sync_central_bank_rates_to_market()
            total_rows = conn.execute("SELECT COUNT(*) FROM bronze_central_bank_rates;").fetchone()[0]
            return {"files_processed": 0, "rows_in_table": total_rows, "market_sync": sync_res, "status": "already_up_to_date"}

        total_ingested = 0
        for fpath in target_files:
            if not fpath.exists():
                logger.warning(f"Central bank rate file not found: {fpath}")
                continue

            logger.info(f"Ingesting central bank rate file: {fpath.name}")
            ext = fpath.suffix.lower()
            try:
                if ext in [".xlsx", ".xls"]:
                    df = pd.read_excel(fpath)
                elif ext == ".csv":
                    df = pd.read_csv(fpath)
                elif ext == ".parquet":
                    df = pd.read_parquet(fpath)
                else:
                    logger.warning(f"Unsupported file format for CBRT rates: {fpath}")
                    continue
            except Exception as e:
                logger.error(f"Failed to read central bank file {fpath}: {e}")
                continue

            # Identify date and rate columns
            date_col = None
            rate_col = None
            for c in df.columns:
                c_clean = str(c).strip().lower()
                if any(k in c_clean for k in ["tarih", "date", "time", "gun"]):
                    date_col = c
                if any(k in c_clean for k in ["repo", "faiz", "rate", "interest", "deger", "value"]):
                    rate_col = c

            if not date_col or not rate_col:
                if len(df.columns) >= 2:
                    date_col = df.columns[0]
                    rate_col = df.columns[1]
                else:
                    logger.error(f"Could not identify date and rate columns in {fpath.name}: {df.columns.tolist()}")
                    continue

            df_clean = df[[date_col, rate_col]].dropna().copy()
            df_clean["rate_date"] = pd.to_datetime(df_clean[date_col]).dt.strftime("%Y-%m-%d")
            df_clean["interest_rate"] = pd.to_numeric(df_clean[rate_col], errors="coerce")
            df_clean = df_clean.dropna(subset=["rate_date", "interest_rate"])
            df_clean = df_clean.drop_duplicates(subset=["rate_date"]).sort_values("rate_date")

            if df_clean.empty:
                logger.warning(f"No valid rows found in central bank rate file: {fpath.name}")
                continue

            records = []
            for _, row in df_clean.iterrows():
                records.append((
                    row["rate_date"],
                    "1_week_repo",
                    float(row["interest_rate"]),
                    0.0,
                    False,
                    False,
                    fpath.name,
                ))

            conn.executemany("""
                INSERT OR REPLACE INTO bronze_central_bank_rates (
                    rate_date, rate_type, interest_rate, rate_change, is_rate_change_day, is_forward_filled, raw_source, ingested_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, CURRENT_TIMESTAMP);
            """, records)

            stat = fpath.stat()
            conn.execute("""
                INSERT OR REPLACE INTO bronze_ingestion_log (
                    file_path, file_name, file_size_bytes, file_mtime_epoch, trade_date, year_month, rows_ingested, raw_source_label, ingested_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, CURRENT_TIMESTAMP);
            """, [
                fpath.resolve().as_posix(),
                fpath.name,
                stat.st_size,
                stat.st_mtime,
                df_clean["rate_date"].max(),
                df_clean["rate_date"].max()[:7],
                len(df_clean),
                "cbrt_interest_rates",
            ])
            total_ingested += len(df_clean)

        # Globally recalculate rate_change and is_rate_change_day for all records to maintain continuity
        conn.execute("""
            CREATE OR REPLACE TEMP TABLE _cbrt_recalculated AS
            WITH ordered AS (
                SELECT 
                    rate_date,
                    rate_type,
                    interest_rate,
                    COALESCE(interest_rate - LAG(interest_rate) OVER (PARTITION BY rate_type ORDER BY rate_date), 0.0) AS calculated_change,
                    CASE 
                        WHEN LAG(interest_rate) OVER (PARTITION BY rate_type ORDER BY rate_date) IS NOT NULL 
                             AND ABS(interest_rate - LAG(interest_rate) OVER (PARTITION BY rate_type ORDER BY rate_date)) > 1e-6 
                        THEN TRUE 
                        ELSE FALSE 
                    END AS calculated_is_change,
                    is_forward_filled,
                    raw_source,
                    ingested_at
                FROM bronze_central_bank_rates
            )
            SELECT 
                rate_date,
                rate_type,
                interest_rate,
                calculated_change AS rate_change,
                calculated_is_change AS is_rate_change_day,
                is_forward_filled,
                raw_source,
                ingested_at
            FROM ordered;

            DELETE FROM bronze_central_bank_rates;
            INSERT INTO bronze_central_bank_rates SELECT * FROM _cbrt_recalculated;
        """)

        sync_res = {}
        if sync_market_dates:
            sync_res = self.sync_central_bank_rates_to_market()

        total_rows = conn.execute("SELECT COUNT(*) FROM bronze_central_bank_rates;").fetchone()[0]
        logger.info(f"Central bank rates ingestion complete. Total rows in table: {total_rows:,}")

        return {
            "files_processed": len(target_files),
            "rows_ingested": total_ingested,
            "rows_in_table": total_rows,
            "market_sync": sync_res,
            "status": "success",
        }

    def sync_central_bank_rates_to_market(
        self,
        target_end_date: Optional[Union[date, str]] = None,
    ) -> Dict[str, Any]:
        """Synchronize and forward-fill Central Bank rates up to the latest market date or target_end_date."""
        initialize_bronze_schema(self.db)
        conn = self.db.get_connection()

        # Check existing max CBRT rate date from non-forward-filled data first, then table max
        cbrt_max_res = conn.execute(
            "SELECT MAX(rate_date) FROM bronze_central_bank_rates WHERE is_forward_filled = FALSE;"
        ).fetchone()
        if not cbrt_max_res or not cbrt_max_res[0]:
            cbrt_max_res = conn.execute("SELECT MAX(rate_date) FROM bronze_central_bank_rates;").fetchone()
            if not cbrt_max_res or not cbrt_max_res[0]:
                logger.debug("No base central bank rates present to forward-fill from.")
                return {"forward_filled_count": 0, "status": "no_base_rates"}

        max_cbrt_date = cbrt_max_res[0]
        if isinstance(max_cbrt_date, str):
            max_cbrt_date = datetime.strptime(max_cbrt_date, "%Y-%m-%d").date()

        if target_end_date is not None:
            if isinstance(target_end_date, str):
                eff_target_date = datetime.strptime(target_end_date, "%Y-%m-%d").date()
            else:
                eff_target_date = target_end_date
        else:
            has_stock_sum = conn.execute(
                "SELECT COUNT(*) FROM information_schema.tables WHERE table_name = 'silver_daily_stock_summary';"
            ).fetchone()[0]
            max_mkt_date = None
            if has_stock_sum > 0:
                mkt_res = conn.execute("SELECT MAX(trade_date) FROM silver_daily_stock_summary;").fetchone()
                if mkt_res and mkt_res[0]:
                    max_mkt_date = mkt_res[0]
            if max_mkt_date is None:
                has_trades = conn.execute(
                    "SELECT COUNT(*) FROM information_schema.tables WHERE table_name = 'bronze_raw_trades';"
                ).fetchone()[0]
                if has_trades > 0:
                    trade_res = conn.execute("SELECT CAST(MAX(timestamp) AS DATE) FROM bronze_raw_trades;").fetchone()
                    if trade_res and trade_res[0]:
                        max_mkt_date = trade_res[0]

            eff_target_date = max_mkt_date

        if eff_target_date is None or eff_target_date <= max_cbrt_date:
            logger.debug(f"Central bank rates are already synchronized through {max_cbrt_date} (target={eff_target_date}).")
            return {"forward_filled_count": 0, "max_rate_date": str(max_cbrt_date), "status": "already_synced"}

        latest_rate = conn.execute("""
            SELECT interest_rate 
            FROM bronze_central_bank_rates 
            WHERE rate_date = ? 
            LIMIT 1;
        """, [max_cbrt_date]).fetchone()[0]

        logger.info(
            f"Forward-filling Central Bank rates from {max_cbrt_date + timedelta(days=1)} "
            f"through {eff_target_date} at rate {latest_rate}%..."
        )

        start_fill = max_cbrt_date + timedelta(days=1)
        conn.execute("""
            INSERT OR REPLACE INTO bronze_central_bank_rates (
                rate_date, rate_type, interest_rate, rate_change, is_rate_change_day, is_forward_filled, raw_source, ingested_at
            )
            SELECT 
                d::DATE AS rate_date,
                '1_week_repo' AS rate_type,
                ? AS interest_rate,
                0.0 AS rate_change,
                FALSE AS is_rate_change_day,
                TRUE AS is_forward_filled,
                'forward_fill_market_sync' AS raw_source,
                CURRENT_TIMESTAMP AS ingested_at
            FROM generate_series(?::DATE, ?::DATE, INTERVAL 1 DAY) t(d);
        """, [latest_rate, start_fill, eff_target_date])

        filled_count = (eff_target_date - max_cbrt_date).days
        logger.info(f"Forward-filled {filled_count} missing date(s) in `bronze_central_bank_rates`.")

        return {
            "forward_filled_count": filled_count,
            "start_date": str(start_fill),
            "end_date": str(eff_target_date),
            "interest_rate": float(latest_rate),
            "status": "success",
        }
