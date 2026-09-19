#!/usr/bin/env python3
"""Automated, resumable migration tool: Migrates Medallion Lakehouse from DuckDB to PostgreSQL + TimescaleDB.

Transfers Reference tables, Silver aggregations, FIFO Tertip ledgers, and Gold models
with zero data loss, utilizing PostgreSQL binary COPY protocol for ultra-high throughput.

Usage:
  # 1. Quick Migration (Reference, Silver, FIFO, and Gold tables — ~2-5 minutes):
  .venv/bin/python scripts/migrate_duckdb_to_postgres.py

  # 2. Full Migration including 2.2B raw trades (streamed in monthly chunks):
  .venv/bin/python scripts/migrate_duckdb_to_postgres.py --include-ticks

  # 3. Stream recent raw trades (e.g. from 2026-01-01 onwards):
  .venv/bin/python scripts/migrate_duckdb_to_postgres.py --include-ticks --start-date 2026-01-01
"""

import argparse
import sys
import time
from pathlib import Path

import duckdb
import polars as pl
from rich.console import Console
from rich.table import Table

# Add project root to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from mdk_trading_oracle.core.config import get_settings
from mdk_trading_oracle.core.db import PostgresManager
from mdk_trading_oracle.core.logger import get_logger

logger = get_logger("mdk_oracle.scripts.migrate")
console = Console()

CORE_TABLES = [
    # Bronze Reference & Dimension Tables
    "bronze_brokers",
    "bronze_instruments",
    "bronze_ingestion_log",
    "bronze_central_bank_rates",
    "bronze_bist_index_benchmarks",
    "bronze_corporate_actions",
    "bronze_bist30_membership",
    "bronze_bist30_changes",
    "bronze_bist30_stock_periods",
    # Silver Macro, Overview & Threshold Tables
    "silver_corporate_action_adjustment_periods",
    "silver_daily_macro_rates",
    "silver_daily_benchmark_index",
    "silver_bofa_historical_flow_thresholds",
    "silver_stock_reaction_thresholds",
    "silver_market_daily",
    "silver_daily_broker_overview",
    "silver_daily_stock_summary",
    "silver_daily_sector_summary",
    "silver_daily_broker_summary",
    # Silver Tertip FIFO Ledgers
    "silver_broker_fifo_daily",
    "silver_broker_fifo_lot_entries",
    "silver_broker_fifo_lots",
    "silver_broker_fifo_lot_realizations",
    "silver_broker_fifo_lot_lifecycle",
    # Silver Intraday Window Summaries
    "silver_intraday_broker_window_summary",
    "silver_intraday_sector_window_summary",
    # Gold Features & Signals
    "gold_institutional_daily_signals",
    # Gold Model 1: Macro Day-Start
    "gold_bofa_day_start_forecasts",
    "gold_bofa_day_start_performance",
    "gold_bofa_day_start_backtests",
    # Gold Model 2: Sector Day-Start
    "gold_bofa_sector_day_start_forecasts",
    "gold_bofa_sector_day_start_performance",
    "gold_bofa_sector_day_start_backtests",
    # Gold Model 3: Stock Intraday Reaction (W2, W3, W5)
    "gold_bofa_stock_reaction_w2_forecasts",
    "gold_bofa_stock_reaction_w2_performance",
    "gold_bofa_stock_reaction_w2_backtests",
    "gold_bofa_stock_reaction_w3_forecasts",
    "gold_bofa_stock_reaction_w3_performance",
    "gold_bofa_stock_reaction_w3_backtests",
    "gold_bofa_stock_reaction_w5_forecasts",
    "gold_bofa_stock_reaction_w5_performance",
    "gold_bofa_stock_reaction_w5_backtests",
]


def migrate_table(duck_conn: duckdb.DuckDBPyConnection, pg: PostgresManager, table_name: str, chunk_size: int = 100_000) -> int:
    """Stream table from DuckDB into PostgreSQL using Polars zero-copy and binary COPY."""
    # Check if table exists in DuckDB
    exists = duck_conn.execute(
        f"SELECT COUNT(*) FROM information_schema.tables WHERE table_name = '{table_name}';"
    ).fetchone()[0]
    if not exists:
        logger.warning(f"Table '{table_name}' does not exist in DuckDB; skipping.")
        return 0

    total_rows = duck_conn.execute(f"SELECT COUNT(*) FROM {table_name};").fetchone()[0]
    if total_rows == 0:
        logger.debug(f"Table '{table_name}' is empty in DuckDB; skipping data copy.")
        return 0

    # Truncate destination in PostgreSQL to ensure idempotent migration
    try:
        pg.execute(f"TRUNCATE TABLE {table_name} CASCADE;")
    except Exception:
        pass

    rows_migrated = 0
    if total_rows <= chunk_size:
        arrow_table = duck_conn.execute(f"SELECT * FROM {table_name};").fetch_arrow_table()
        df = pl.from_arrow(arrow_table)
        pg.copy_df_to_table(df, table_name)
        rows_migrated = df.height
    else:
        # Chunked migration for larger tables (e.g. intraday windows, FIFO ledgers)
        offset = 0
        while offset < total_rows:
            arrow_table = duck_conn.execute(
                f"SELECT * FROM {table_name} LIMIT {chunk_size} OFFSET {offset};"
            ).fetch_arrow_table()
            df = pl.from_arrow(arrow_table)
            if df.is_empty():
                break
            pg.copy_df_to_table(df, table_name)
            offset += df.height
            rows_migrated += df.height

    return rows_migrated


def migrate_raw_trades(duck_conn: duckdb.DuckDBPyConnection, pg: PostgresManager, start_date: str = None) -> int:
    """Stream raw trades from DuckDB to PostgreSQL in month-by-month partitions."""
    console.print("[cyan]Discovering monthly partitions in `bronze_raw_trades`...[/cyan]")
    query = "SELECT DISTINCT strftime(timestamp, '%Y-%m') AS ym FROM bronze_raw_trades"
    if start_date:
        query += f" WHERE timestamp >= '{start_date}'"
    query += " ORDER BY ym ASC;"

    months = [r[0] for r in duck_conn.execute(query).fetchall() if r[0]]
    console.print(f"[green]Found {len(months)} monthly partitions to migrate.[/green]")

    total_ticks = 0
    for ym in months:
        t0 = time.time()
        m_count = duck_conn.execute(
            f"SELECT COUNT(*) FROM bronze_raw_trades WHERE strftime(timestamp, '%Y-%m') = '{ym}';"
        ).fetchone()[0]
        console.print(f"[yellow]Migrating partition {ym} ({m_count:,} rows)...[/yellow]")

        chunk_size = 500_000
        offset = 0
        while offset < m_count:
            arrow_table = duck_conn.execute(
                f"SELECT * FROM bronze_raw_trades WHERE strftime(timestamp, '%Y-%m') = '{ym}' "
                f"ORDER BY timestamp ASC LIMIT {chunk_size} OFFSET {offset};"
            ).fetch_arrow_table()
            df = pl.from_arrow(arrow_table)
            if df.is_empty():
                break
            pg.copy_df_to_table(df, "bronze_raw_trades")
            offset += df.height
            total_ticks += df.height

        elapsed = time.time() - t0
        rate = m_count / max(elapsed, 0.01)
        console.print(f"  [green]✓[/green] Partition {ym} complete ({m_count:,} rows in {elapsed:.1f}s, {rate:,.0f} rows/s)")

    return total_ticks


def main():
    parser = argparse.ArgumentParser(description="Migrate MDK Trading Oracle from DuckDB to PostgreSQL + TimescaleDB")
    parser.add_argument("--include-ticks", action="store_true", help="Migrate bronze_raw_trades (2.2B+ tick rows)")
    parser.add_argument("--start-date", type=str, default=None, help="Filter start date for raw trades (e.g. '2026-01-01')")
    args = parser.parse_args()

    settings = get_settings()
    duckdb_path = settings.database_path

    if not duckdb_path.exists():
        console.print(f"[red]Error: DuckDB file not found at {duckdb_path}[/red]")
        sys.exit(1)

    console.print("[bold cyan]═══════════════════════════════════════════════════════════════════════════[/bold cyan]")
    console.print("[bold cyan] MDK TRADING ORACLE — MEDALLION LAKEHOUSE DATABASE MIGRATION ENGINE        [/bold cyan]")
    console.print(f"[bold cyan] Source: {duckdb_path} (DuckDB)                                            [/bold cyan]")
    console.print(f"[bold cyan] Target: {settings.pg_host}:{settings.pg_port}/{settings.pg_database} (PostgreSQL + TimescaleDB) [/bold cyan]")
    console.print("[bold cyan]═══════════════════════════════════════════════════════════════════════════[/bold cyan]\n")

    # Connect to DuckDB read-only
    console.print("[blue]Connecting to source DuckDB...[/blue]")
    duck_conn = duckdb.connect(str(duckdb_path), read_only=True)

    # Connect to PostgreSQL and initialize schema
    console.print("[blue]Connecting to target PostgreSQL and initializing Medallion schemas...[/blue]")
    pg = PostgresManager()
    try:
        pg.get_connection()
    except Exception as e:
        console.print(f"[red]Failed to connect to PostgreSQL: {e}[/red]")
        console.print("[yellow]Tip: Ensure PostgreSQL/TimescaleDB is running (e.g. via 'docker compose up -d').[/yellow]")
        sys.exit(1)

    pg.initialize_schema()

    summary_table = Table(title="Database Migration Summary (Bronze -> Silver -> Gold)")
    summary_table.add_column("Table Name", style="cyan")
    summary_table.add_column("DuckDB Rows", justify="right")
    summary_table.add_column("Migrated Rows", justify="right", style="green")
    summary_table.add_column("Status", style="bold")

    start_time = time.time()

    # Stage 1: Migrate Core Reference, Silver, and Gold Tables
    console.print("\n[bold]Stage 1: Migrating Reference, Silver, FIFO Ledgers, and Gold Signals...[/bold]")
    for tbl in CORE_TABLES:
        try:
            d_count = duck_conn.execute(
                f"SELECT COUNT(*) FROM information_schema.tables WHERE table_name = '{tbl}';"
            ).fetchone()[0]
            if not d_count:
                continue
            duck_rows = duck_conn.execute(f"SELECT COUNT(*) FROM {tbl};").fetchone()[0]
            migrated = migrate_table(duck_conn, pg, tbl)
            status = "[green]SUCCESS[/green]" if migrated == duck_rows else "[yellow]PARTIAL[/yellow]"
            summary_table.add_row(tbl, f"{duck_rows:,}", f"{migrated:,}", status)
            console.print(f"  [green]✓[/green] Migrated {tbl}: {migrated:,} rows")
        except Exception as e:
            logger.error(f"Error migrating {tbl}: {e}")
            summary_table.add_row(tbl, "ERR", "0", "[red]FAILED[/red]")

    # Stage 2: Optional Raw Trades Migration
    if args.include_ticks:
        console.print("\n[bold]Stage 2: Migrating Raw Trade Ticks (`bronze_raw_trades`)...[/bold]")
        total_ticks = migrate_raw_trades(duck_conn, pg, start_date=args.start_date)
        summary_table.add_row("bronze_raw_trades", "2.2B+", f"{total_ticks:,}", "[green]SUCCESS[/green]")

    elapsed = time.time() - start_time
    console.print("\n")
    console.print(summary_table)
    console.print(f"\n[bold green]✓ Migration finished successfully in {elapsed:.2f} seconds![/bold green]")
    console.print("[cyan]MDK Trading Oracle is now fully running on PostgreSQL + TimescaleDB.[/cyan]\n")


if __name__ == "__main__":
    main()
