"""Typer CLI interface for MDK Trading Oracle."""

from datetime import datetime
from pathlib import Path
from typing import Optional

import duckdb
import typer
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from mdk_trading_oracle.core.config import get_settings
from mdk_trading_oracle.core.db import DuckDBManager
from mdk_trading_oracle.data.bronze import BronzeIngestor
from mdk_trading_oracle.data.discovery import RawDataInspector
from mdk_trading_oracle.data.gold import GoldFeatureEngineer
from mdk_trading_oracle.data.pipeline import MedallionPipeline
from mdk_trading_oracle.data.silver import SilverTransformer

app = typer.Typer(
    name="mdk-oracle",
    help="MDK Trading Oracle - Quantitative Decision Support & Institutional Flow Analyzer",
    add_completion=False,
)
pipeline_app = typer.Typer(
    name="pipeline",
    help="Medallion Lakehouse Pipeline Orchestration (Bronze -> Silver -> Gold)",
    add_completion=False,
)
data_app = typer.Typer(
    name="data",
    help="Raw Data Discovery, Catalog Preparation, and Reference Metadata Sync",
    add_completion=False,
)
app.add_typer(pipeline_app, name="pipeline")
app.add_typer(data_app, name="data")

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
            layer = (
                "Bronze"
                if tbl_name.startswith("bronze_")
                else ("Silver" if tbl_name.startswith("silver_") else "Gold")
            )
            db_table.add_row(tbl_name, f"{count:,}", layer)

        console.print(db_table)
    else:
        console.print(
            "[bold red]⚠️ DuckDB database not yet initialized. Run 'mdk-oracle load-bronze' to initialize.[/bold red]"
        )


@data_app.command("inspect")
def data_inspect(
    glob_pattern: Optional[str] = typer.Option(
        None,
        "--glob",
        "-g",
        help="Optional custom glob pattern for raw CSVs",
    ),
):
    """Scan raw CSV feeds and render interactive tables of discovered instruments and brokers."""
    inspector = RawDataInspector(raw_glob=glob_pattern)
    inspector.print_interactive_report()


@data_app.command("sync-catalog")
def data_sync_catalog(
    glob_pattern: Optional[str] = typer.Option(
        None,
        "--glob",
        "-g",
        help="Optional custom glob pattern for raw CSVs",
    ),
):
    """Extract all instruments and brokerages from raw feeds and synchronize YAML catalogs."""
    inspector = RawDataInspector(raw_glob=glob_pattern)
    console.print("[bold cyan]🔄 Discovering entities and synchronizing YAML catalogs...[/bold cyan]")
    res = inspector.sync_to_yaml_catalogs()
    console.print(
        Panel.fit(
            f"[bold green]✨ Catalogs Synchronized Successfully![/bold green]\n\n"
            f"• [bold]Instruments[/bold]: [cyan]{res['instruments_count']}[/cyan] symbols synced to `{res['instruments_file']}`\n"
            f"• [bold]Brokers[/bold]: [cyan]{res['brokers_count']}[/cyan] brokerages synced to `{res['brokers_file']}`",
            title="Catalog Synchronization Summary",
            border_style="green",
        )
    )


@app.command()
def load_bronze(
    glob_pattern: Optional[str] = typer.Option(
        None,
        "--glob",
        "-g",
        help="Optional custom glob pattern for raw CSVs (defaults to settings.raw_data_dir/2026/03_march/raw_csv/**/*.csv)",
    )
):
    """Ingest raw BIST trade CSV feeds and Central Bank interest rates into the Bronze DuckDB layer."""
    start_time = datetime.now()
    settings = get_settings()

    console.print("[bold cyan]🔄 Initializing DuckDB and Bronze Schemas...[/bold cyan]")
    db = DuckDBManager()
    db.initialize_schema()

    ingestor = BronzeIngestor(db)
    target_glob = glob_pattern or (settings.raw_data_dir / "2026/03_march/raw_csv/**/*.csv").as_posix()

    console.print(f"[bold yellow]📂 Ingesting BIST raw files matching:[/bold yellow] {target_glob}")

    ingestor.ingest_bist_raw_csv_glob(
        glob_pattern=target_glob,
        raw_source_label="bist_2026_03_march",
    )

    console.print("[bold yellow]🏛️ Ingesting Central Bank interest rates and synchronizing market dates...[/bold yellow]")
    ingestor.ingest_central_bank_rates(sync_market_dates=True)

    console.print("[bold yellow]📈 Ingesting BIST 30 benchmark data and synchronizing market dates...[/bold yellow]")
    ingestor.ingest_bist30_benchmarks(sync_market_dates=True)

    conn = db.get_connection()
    total_trades = conn.execute("SELECT COUNT(*) FROM bronze_raw_trades;").fetchone()[0]
    total_cbrt = conn.execute("SELECT COUNT(*) FROM bronze_central_bank_rates;").fetchone()[0]
    total_bench = conn.execute("SELECT COUNT(*) FROM bronze_bist_index_benchmarks;").fetchone()[0]
    total_files = conn.execute("SELECT COUNT(*) FROM bronze_ingestion_log;").fetchone()[0]

    elapsed = (datetime.now() - start_time).total_seconds()
    console.print(
        Panel.fit(
            f"[bold green]✨ Bronze Layer Ingestion Complete in {elapsed:.1f}s[/bold green]\n\n"
            f"• [bold]bronze_raw_trades[/bold]: [cyan]{total_trades:,}[/cyan] rows\n"
            f"• [bold]bronze_central_bank_rates[/bold]: [cyan]{total_cbrt:,}[/cyan] rows\n"
            f"• [bold]bronze_bist_index_benchmarks[/bold]: [cyan]{total_bench:,}[/cyan] rows\n"
            f"• [bold]bronze_ingestion_log[/bold]: [cyan]{total_files:,}[/cyan] files logged",
            title="Bronze Summary",
            border_style="green",
        )
    )


@app.command()
def load_rates(
    file_path: Optional[Path] = typer.Option(
        None,
        "--file",
        "-f",
        help="Optional path to a specific Central Bank Excel/CSV/Parquet rate file",
    ),
    force: bool = typer.Option(
        False,
        "--force",
        help="Force re-ingestion of already processed rate files",
    ),
):
    """Ingest Central Bank policy interest rates and synchronize with market trading dates."""
    start_time = datetime.now()
    console.print("[bold cyan]🏛️ Ingesting Central Bank interest rates...[/bold cyan]")

    db = DuckDBManager()
    ingestor = BronzeIngestor(db)
    res = ingestor.ingest_central_bank_rates(file_path=file_path, force=force, sync_market_dates=True)

    conn = db.get_connection()
    total_rates = conn.execute("SELECT COUNT(*) FROM bronze_central_bank_rates;").fetchone()[0]
    rate_range = conn.execute("SELECT MIN(rate_date), MAX(rate_date) FROM bronze_central_bank_rates;").fetchone()

    elapsed = (datetime.now() - start_time).total_seconds()
    console.print(
        Panel.fit(
            f"[bold green]✨ Central Bank Rates Ingested & Synced in {elapsed:.1f}s[/bold green]\n\n"
            f"• [bold]bronze_central_bank_rates[/bold]: [cyan]{total_rates:,}[/cyan] rows\n"
            f"• [bold]Date Range[/bold]: [cyan]{rate_range[0]}[/cyan] to [cyan]{rate_range[1]}[/cyan]\n"
            f"• [bold]Files Processed[/bold]: [cyan]{res.get('files_processed', 0)}[/cyan]",
            title="CBRT Rates Summary",
            border_style="green",
        )
    )


@app.command()
def load_benchmark(
    years: int = typer.Option(
        5,
        "--years",
        "-y",
        help="Number of historical years to fetch (default: 5)",
    ),
    ticker: str = typer.Option(
        "XU030.IS",
        "--ticker",
        "-t",
        help="Benchmark ticker symbol on Yahoo Finance (default: XU030.IS)",
    ),
    force: bool = typer.Option(
        False,
        "--force",
        help="Force re-download and re-ingestion of benchmark data",
    ),
):
    """Ingest official BIST 30 (XU030.IS) benchmark historical data and synchronize with market dates."""
    start_time = datetime.now()
    console.print(f"[bold cyan]📈 Ingesting {years}-Year BIST 30 Benchmark Data ({ticker})...[/bold cyan]")

    db = DuckDBManager()
    ingestor = BronzeIngestor(db)
    res = ingestor.ingest_bist30_benchmarks(years=years, ticker_symbol=ticker, force=force, sync_market_dates=True)

    conn = db.get_connection()
    total_bench = conn.execute("SELECT COUNT(*) FROM bronze_bist_index_benchmarks;").fetchone()[0]
    bench_range = conn.execute("SELECT MIN(trade_date), MAX(trade_date) FROM bronze_bist_index_benchmarks;").fetchone()

    elapsed = (datetime.now() - start_time).total_seconds()
    console.print(
        Panel.fit(
            f"[bold green]✨ BIST 30 Benchmark Data Ingested & Synced in {elapsed:.1f}s[/bold green]\n\n"
            f"• [bold]bronze_bist_index_benchmarks[/bold]: [cyan]{total_bench:,}[/cyan] rows\n"
            f"• [bold]Date Range[/bold]: [cyan]{bench_range[0]}[/cyan] to [cyan]{bench_range[1]}[/cyan]\n"
            f"• [bold]Status[/bold]: [cyan]{res.get('status', 'success')}[/cyan]",
            title="BIST 30 Benchmark Summary",
            border_style="green",
        )
    )


@app.command()
def build_silver():
    """Build Silver layer daily broker aggregations, macro rates, benchmark indicators, and market OHLCV metrics."""
    start_time = datetime.now()
    console.print("[bold cyan]🚀 Building Silver Lakehouse Layer...[/bold cyan]")

    db = DuckDBManager()
    transformer = SilverTransformer(db)
    transformer.run_all()

    conn = db.get_connection()
    broker_rows = conn.execute("SELECT COUNT(*) FROM silver_daily_broker_summary;").fetchone()[0]
    macro_rows = conn.execute("SELECT COUNT(*) FROM silver_daily_macro_rates;").fetchone()[0]
    bench_rows = conn.execute("SELECT COUNT(*) FROM silver_daily_benchmark_index;").fetchone()[0]
    market_rows = conn.execute("SELECT COUNT(*) FROM silver_market_daily;").fetchone()[0]

    elapsed = (datetime.now() - start_time).total_seconds()
    console.print(
        Panel.fit(
            f"[bold green]✨ Silver Layer Transformations Complete in {elapsed:.1f}s[/bold green]\n\n"
            f"• [bold]silver_daily_broker_summary[/bold]: [cyan]{broker_rows:,}[/cyan] rows\n"
            f"• [bold]silver_daily_macro_rates[/bold]: [cyan]{macro_rows:,}[/cyan] rows\n"
            f"• [bold]silver_daily_benchmark_index[/bold]: [cyan]{bench_rows:,}[/cyan] rows\n"
            f"• [bold]silver_market_daily[/bold]: [cyan]{market_rows:,}[/cyan] rows",
            title="Silver Summary",
            border_style="green",
        )
    )


@app.command()
def build_gold():
    """Build Gold layer institutional flow indicators and rolling signal metrics."""
    start_time = datetime.now()
    console.print("[bold cyan]🚀 Building Gold Lakehouse Layer...[/bold cyan]")

    db = DuckDBManager()
    engineer = GoldFeatureEngineer(db)
    engineer.run_all()

    conn = db.get_connection()
    signal_rows = conn.execute("SELECT COUNT(*) FROM gold_institutional_daily_signals;").fetchone()[0]

    elapsed = (datetime.now() - start_time).total_seconds()
    console.print(
        Panel.fit(
            f"[bold green]✨ Gold Layer Feature Engineering Complete in {elapsed:.1f}s[/bold green]\n\n"
            f"• [bold]gold_institutional_daily_signals[/bold]: [cyan]{signal_rows:,}[/cyan] rows",
            title="Gold Summary",
            border_style="green",
        )
    )


@app.command()
def build_all(
    glob_pattern: Optional[str] = typer.Option(
        None,
        "--glob",
        "-g",
        help="Optional custom glob pattern for raw CSVs",
    ),
    sync_catalog: bool = typer.Option(
        False,
        "--sync-catalog",
        "-s",
        help="Discover entities and synchronize YAML catalogs before running pipeline",
    ),
):
    """Run end-to-end Medallion pipeline: Bronze Ingestion -> Silver Layer -> Gold Layer."""
    db = DuckDBManager()
    pipeline = MedallionPipeline(db)
    pipeline.run(
        target="all",
        raw_glob=glob_pattern,
        sync_catalog=sync_catalog,
        print_summary=True,
    )


@pipeline_app.command("run")
def pipeline_run(
    target: str = typer.Option(
        "all",
        "--target",
        "-t",
        help="Target layer to execute: catalog, bronze, silver, gold, or all",
    ),
    glob_pattern: Optional[str] = typer.Option(
        None,
        "--glob",
        "-g",
        help="Optional custom glob pattern for raw CSVs",
    ),
    sync_catalog: bool = typer.Option(
        False,
        "--sync-catalog",
        "-s",
        help="Discover entities and synchronize YAML catalogs before running pipeline",
    ),
    no_deps: bool = typer.Option(
        False,
        "--no-deps",
        help="Disable automatic dependency resolution",
    ),
):
    """Execute the Medallion Pipeline with dependency DAG resolution."""
    db = DuckDBManager()
    pipeline = MedallionPipeline(db)
    pipeline.run(
        target=target,
        raw_glob=glob_pattern,
        sync_catalog=sync_catalog,
        resolve_dependencies=not no_deps,
        print_summary=True,
    )


if __name__ == "__main__":
    app()

