#!/usr/bin/env python3
"""Initialize PostgreSQL 16 + TimescaleDB Medallion Lakehouse Architecture.

Creates all Bronze hypertables with columnar compression, Silver continuous aggregate
candlestick views (1m, 5m), FIFO lot tracking ledgers, and Gold predictive model schemas.

Usage:
  .venv/bin/python scripts/setup_postgres_lakehouse.py
"""

import sys
import time
from pathlib import Path

from rich.console import Console
from rich.panel import Panel
from rich.table import Table

# Add project root to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from mdk_trading_oracle.core.config import get_settings
from mdk_trading_oracle.core.db import PostgresManager
from mdk_trading_oracle.core.logger import get_logger

logger = get_logger("mdk_oracle.scripts.setup")
console = Console()


def main():
    settings = get_settings()
    console.print(
        Panel.fit(
            f"[bold cyan]MDK Trading Oracle — Lakehouse Database Initializer[/bold cyan]\n"
            f"[dim]Target: PostgreSQL 16 + TimescaleDB on Apple Silicon M5 Mac Pro[/dim]\n"
            f"Host: [bold green]{settings.pg_host}:{settings.pg_port}[/bold green] | "
            f"Database: [bold green]{settings.pg_database}[/bold green] | User: [bold green]{settings.pg_user}[/bold green]",
            border_style="cyan",
        )
    )

    t0 = time.time()
    pg = PostgresManager()

    # Verify live connection
    try:
        conn = pg.get_connection()
        with conn.cursor() as cur:
            cur.execute("SELECT version();")
            pg_ver = cur.fetchone()[0].split(",")[0]
            cur.execute("SELECT extversion FROM pg_extension WHERE extname = 'timescaledb';")
            ts_row = cur.fetchone()
            ts_ver = ts_row[0] if ts_row else "Not enabled yet (will enable)"
        console.print(f"[bold green]Connected successfully![/bold green] {pg_ver} | TimescaleDB: {ts_ver}")
    except Exception as e:
        console.print(f"[bold red]Connection to PostgreSQL failed:[/bold red] {e}")
        console.print("[yellow]Please ensure Docker Desktop / container is running via `docker compose up -d`.[/yellow]")
        sys.exit(1)

    # 1. Initialize all layers
    console.print("\n[bold]1. Initializing Medallion Schemas (Bronze, Silver, Gold)...[/bold]")
    pg.initialize_schema()

    # 2. Sync Reference Data from YAML configs
    console.print("\n[bold]2. Synchronizing Broker & Instrument Catalogs...[/bold]")
    pg.sync_reference_data()

    elapsed = time.time() - t0

    # 3. Inspect and display all created tables
    query = """
        SELECT 
            table_name,
            table_type
        FROM information_schema.tables
        WHERE table_schema = 'public'
        ORDER BY table_name;
    """
    tables = pg.execute(query).fetchall()

    table_report = Table(title="PostgreSQL 16 + TimescaleDB Lakehouse Tables", border_style="green")
    table_report.add_column("Layer", style="bold cyan")
    table_report.add_column("Table Name", style="bold")
    table_report.add_column("Type", style="dim")

    for t_name, t_type in tables:
        layer = "Other"
        if t_name.startswith("bronze_"):
            layer = "Bronze (Raw / Dims)"
        elif t_name.startswith("silver_"):
            layer = "Silver (Agg / FIFO)"
        elif t_name.startswith("gold_"):
            layer = "Gold (Models / Signals)"
        table_report.add_row(layer, t_name, t_type)

    console.print(table_report)
    console.print(
        f"\n[bold green]Database structure successfully created in {elapsed:.2f}s with {len(tables)} tables/views![/bold green]\n"
    )


if __name__ == "__main__":
    main()
