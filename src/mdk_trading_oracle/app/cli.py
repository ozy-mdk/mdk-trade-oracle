"""Typer CLI interface for MDK Trading Oracle."""

from datetime import datetime
from typing import Optional

import duckdb
import typer
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from mdk_trading_oracle.core.config import get_settings
from mdk_trading_oracle.core.db import DuckDBManager
from mdk_trading_oracle.ingestion.file_loader import FileIngestor

app = typer.Typer(
    name="mdk-oracle",
    help="MDK Trading Oracle - Quantitative Decision Support & Institutional Flow Analyzer",
    add_completion=False,
)
console = Console()


@app.command()
def info():
    """Display environment, external data directories, and database status."""
    settings = get_settings()

    console.print(
        Panel.fit(
            "[bold cyan]MDK Trading Oracle[/bold cyan] [green]v0.1.0[/green]\n"
            "[italic]BIST Institutional Flow & Bank of America (MLB) Tracker[/italic]",
            border_style="cyan",
        )
    )

    # Config & Storage Table
    table = Table(title="🏛 System & Storage Configuration", title_style="bold yellow")
    table.add_column("Property", style="bold cyan")
    table.add_column("Value", style="green")

    table.add_row("App Environment", settings.app_env)
    table.add_row("Primary Institution", f"{settings.primary_institution} (Bank of America / Merrill Lynch)")
    table.add_row("Project Root (Code)", str(settings.project_root))
    table.add_row("External Data Lakehouse", str(settings.data_dir))
    table.add_row("Raw Landing Zone (00_raw_data)", str(settings.raw_data_dir))
    table.add_row("DuckDB Database File", str(settings.database_path))
    table.add_row("Config Directory", str(settings.config_dir))
    console.print(table)

    # Database Statistics Table
    if settings.database_path.exists():
        conn = duckdb.connect(str(settings.database_path), read_only=True)
        tables = conn.execute("SHOW TABLES;").fetchall()
        
        db_table = Table(title="📊 DuckDB Lakehouse Tables", title_style="bold magenta")
        db_table.add_column("Table Name", style="bold")
        db_table.add_column("Row Count", style="cyan", justify="right")
        db_table.add_column("Layer", style="yellow")

        for (tbl_name,) in tables:
            count = conn.execute(f"SELECT COUNT(*) FROM {tbl_name};").fetchone()[0]
            layer = "Bronze" if tbl_name.startswith("bronze_") else ("Silver" if tbl_name.startswith("silver_") else "Gold")
            db_table.add_row(tbl_name, f"{count:,}", layer)

        console.print(db_table)
    else:
        console.print("[bold red]⚠️ DuckDB database not yet initialized. Run 'mdk-oracle load-bronze' to initialize.[/bold red]")


@app.command()
def load_bronze(
    glob_pattern: Optional[str] = typer.Option(
        None,
        "--glob",
        "-g",
        help="Optional custom glob pattern for raw CSVs (defaults to settings.raw_data_dir/2026/03_march/raw_csv/**/*.csv)",
    )
):
    """Ingest raw BIST trade CSV feeds into the Bronze DuckDB layer."""
    start_time = datetime.now()
    settings = get_settings()

    console.print("[bold cyan]🔄 Initializing DuckDB and Bronze Schemas...[/bold cyan]")
    db = DuckDBManager()
    db.initialize_schema()

    ingestor = FileIngestor(db)
    target_glob = glob_pattern or (settings.raw_data_dir / "2026/03_march/raw_csv/**/*.csv").as_posix()

    console.print(f"[bold yellow]📂 Ingesting BIST raw files matching:[/bold yellow] {target_glob}")

    ingestor.ingest_bist_raw_csv_glob(
        glob_pattern=target_glob,
        raw_source_label="bist_2026_03_march",
    )

    conn = db.get_connection()
    total_trades = conn.execute("SELECT COUNT(*) FROM bronze_raw_trades;").fetchone()[0]
    total_brokers = conn.execute("SELECT COUNT(*) FROM bronze_brokers;").fetchone()[0]
    total_instruments = conn.execute("SELECT COUNT(*) FROM bronze_instruments;").fetchone()[0]

    elapsed = (datetime.now() - start_time).total_seconds()
    console.print(
        Panel.fit(
            f"[bold green] Bronze Ingestion Complete in {elapsed:.1f}s[/bold green]\n\n"
            f"• [bold]bronze_raw_trades[/bold]: [cyan]{total_trades:,}[/cyan] rows\n"
            f"• [bold]bronze_brokers[/bold]: [cyan]{total_brokers}[/cyan] brokers\n"
            f"• [bold]bronze_instruments[/bold]: [cyan]{total_instruments}[/cyan] instruments",
            title="Ingestion Summary",
            border_style="green",
        )
    )


if __name__ == "__main__":
    app()
