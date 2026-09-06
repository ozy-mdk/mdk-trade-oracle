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
                "Bronze" if tbl_name.startswith("bronze_") else ("Silver" if tbl_name.startswith("silver_") else "Gold")
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
    ),
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

    console.print(
        "[bold yellow]🏛️ Ingesting Central Bank interest rates and synchronizing market dates...[/bold yellow]"
    )
    ingestor.ingest_central_bank_rates(sync_market_dates=True)

    console.print("[bold yellow]📈 Ingesting BIST 30 benchmark data and synchronizing market dates...[/bold yellow]")
    ingestor.ingest_bist30_benchmarks(sync_market_dates=True)

    console.print("[bold yellow]📋 Ingesting BIST 30 membership list and rebalancing changes...[/bold yellow]")
    ingestor.ingest_bist30_membership()

    conn = db.get_connection()
    total_trades = conn.execute("SELECT COUNT(*) FROM bronze_raw_trades;").fetchone()[0]
    total_cbrt = conn.execute("SELECT COUNT(*) FROM bronze_central_bank_rates;").fetchone()[0]
    total_bench = conn.execute("SELECT COUNT(*) FROM bronze_bist_index_benchmarks;").fetchone()[0]
    total_bist30 = conn.execute("SELECT COUNT(*) FROM bronze_bist30_membership;").fetchone()[0]
    total_files = conn.execute("SELECT COUNT(*) FROM bronze_ingestion_log;").fetchone()[0]

    elapsed = (datetime.now() - start_time).total_seconds()
    console.print(
        Panel.fit(
            f"[bold green]✨ Bronze Layer Ingestion Complete in {elapsed:.1f}s[/bold green]\n\n"
            f"• [bold]bronze_raw_trades[/bold]: [cyan]{total_trades:,}[/cyan] rows\n"
            f"• [bold]bronze_central_bank_rates[/bold]: [cyan]{total_cbrt:,}[/cyan] rows\n"
            f"• [bold]bronze_bist_index_benchmarks[/bold]: [cyan]{total_bench:,}[/cyan] rows\n"
            f"• [bold]bronze_bist30_membership[/bold]: [cyan]{total_bist30:,}[/cyan] rows\n"
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
def load_corporate_actions(
    file_path: Optional[Path] = typer.Option(
        None,
        "--file",
        "-f",
        help="Optional path to a specific corporate_actions.csv file",
    ),
    force: bool = typer.Option(
        False,
        "--force",
        help="Force re-ingestion of corporate actions",
    ),
):
    """Ingest corporate actions (bonus share issues, splits, ticker changes, rights notes) into DuckDB Bronze."""
    start_time = datetime.now()
    console.print("[bold cyan]🔄 Ingesting Corporate Actions...[/bold cyan]")

    db = DuckDBManager()
    ingestor = BronzeIngestor(db)
    res = ingestor.ingest_corporate_actions(csv_path=file_path, force=force)

    conn = db.get_connection()
    total_actions = conn.execute("SELECT COUNT(*) FROM bronze_corporate_actions;").fetchone()[0]
    action_range = conn.execute("SELECT MIN(action_date), MAX(action_date) FROM bronze_corporate_actions;").fetchone()

    elapsed = (datetime.now() - start_time).total_seconds()
    console.print(
        Panel.fit(
            f"[bold green]✨ Corporate Actions Ingested Successfully in {elapsed:.1f}s[/bold green]\n\n"
            f"• [bold]bronze_corporate_actions[/bold]: [cyan]{total_actions:,}[/cyan] rows\n"
            f"• [bold]Date Range[/bold]: [cyan]{action_range[0]}[/cyan] to [cyan]{action_range[1]}[/cyan]\n"
            f"• [bold]Source File[/bold]: [cyan]{res.get('source_path', '')}[/cyan]",
            title="Corporate Actions Summary",
            border_style="green",
        )
    )


@app.command()
def load_bist30(
    file_path: Optional[Path] = typer.Option(
        None,
        "--file",
        "-f",
        help="Optional path to BIST30_uyelik_ve_degisim_tarihi.xlsx",
    ),
    force: bool = typer.Option(
        False,
        "--force",
        help="Force re-ingestion of BIST 30 membership list and changes",
    ),
):
    """Ingest BIST 30 index membership snapshots, quarterly rebalancing changes, and continuous stock periods."""
    start_time = datetime.now()
    console.print("[bold cyan]📋 Ingesting BIST 30 Membership List & Historical Changes...[/bold cyan]")

    db = DuckDBManager()
    ingestor = BronzeIngestor(db)
    res = ingestor.ingest_bist30_membership(file_path=file_path, force=force)

    conn = db.get_connection()
    total_mem = conn.execute("SELECT COUNT(*) FROM bronze_bist30_membership;").fetchone()[0]
    total_changes = conn.execute("SELECT COUNT(*) FROM bronze_bist30_changes;").fetchone()[0]
    total_periods = conn.execute("SELECT COUNT(*) FROM bronze_bist30_stock_periods;").fetchone()[0]
    active_count = conn.execute("SELECT COUNT(DISTINCT symbol) FROM bronze_bist30_membership WHERE is_active = TRUE;").fetchone()[0]

    elapsed = (datetime.now() - start_time).total_seconds()
    console.print(
        Panel.fit(
            f"[bold green]✨ BIST 30 Membership Dataset Ingested in {elapsed:.1f}s[/bold green]\n\n"
            f"• [bold]bronze_bist30_membership[/bold]: [cyan]{total_mem:,}[/cyan] rows (Active: [green]{active_count}[/green])\n"
            f"• [bold]bronze_bist30_changes[/bold]: [cyan]{total_changes:,}[/cyan] rebalancing events\n"
            f"• [bold]bronze_bist30_stock_periods[/bold]: [cyan]{total_periods:,}[/cyan] continuous periods\n"
            f"• [bold]Source File[/bold]: [cyan]{res.get('source_path', '')}[/cyan]",
            title="BIST 30 Membership Summary",
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
def build_gold(
    symbols: Optional[str] = typer.Option(
        None,
        "--symbols",
        help="Comma-separated BIST tickers for Model 3 (e.g. 'AKBNK,GARAN'). Default: all BIST30.",
    ),
    windows: Optional[str] = typer.Option(
        None,
        "--windows",
        help="Comma-separated windows for Model 3 (e.g. 'w2,w5'). Default: w2,w3,w5.",
    ),
    run_backtest: bool = typer.Option(
        False,
        "--run-backtest",
        help="Also run full historical walk-forward backtests for Model 3.",
    ),
):
    """Build Gold layer institutional flow indicators, Day-Start, Sector, and Stock Reaction predictive models."""
    start_time = datetime.now()
    console.print("[bold cyan]🚀 Building Gold Lakehouse Layer (Models 1, 2, 3)...[/bold cyan]")

    db = DuckDBManager()
    engineer = GoldFeatureEngineer(db)
    symbols_list = [s.strip().upper() for s in symbols.split(",") if s.strip()] if symbols else None
    windows_list = [w.strip() for w in windows.split(",") if w.strip()] if windows else None

    engineer.run_all(
        stock_reaction_symbols=symbols_list,
        stock_reaction_windows=windows_list,
        stock_reaction_backtest=run_backtest,
    )

    conn = db.get_connection()
    signal_rows = conn.execute("SELECT COUNT(*) FROM gold_institutional_daily_signals;").fetchone()[0]
    macro_fc = conn.execute("SELECT COUNT(*) FROM gold_bofa_day_start_forecasts;").fetchone()[0]
    sector_fc = conn.execute("SELECT COUNT(*) FROM gold_bofa_sector_day_start_forecasts;").fetchone()[0]
    try:
        w2_fc = conn.execute("SELECT COUNT(*) FROM gold_bofa_stock_reaction_w2_forecasts;").fetchone()[0]
        w3_fc = conn.execute("SELECT COUNT(*) FROM gold_bofa_stock_reaction_w3_forecasts;").fetchone()[0]
        w5_fc = conn.execute("SELECT COUNT(*) FROM gold_bofa_stock_reaction_w5_forecasts;").fetchone()[0]
        sr_fc = w2_fc + w3_fc + w5_fc
    except Exception:
        sr_fc = 0

    elapsed = (datetime.now() - start_time).total_seconds()
    console.print(
        Panel.fit(
            f"[bold green]✨ Gold Layer Feature Engineering & Inference Complete in {elapsed:.1f}s[/bold green]\n\n"
            f"• [bold]gold_institutional_daily_signals[/bold]: [cyan]{signal_rows:,}[/cyan] rows\n"
            f"• [bold]Model 1 (Day-Start Macro Live)[/bold]: [cyan]{macro_fc:,}[/cyan] forecasts\n"
            f"• [bold]Model 2 (Sector Allocation Live)[/bold]: [cyan]{sector_fc:,}[/cyan] forecasts\n"
            f"• [bold]Model 3 (Stock Intraday Reaction Live)[/bold]: [cyan]{sr_fc:,}[/cyan] forecasts across W2/W3/W5",
            title="Gold Summary",
            border_style="green",
        )
    )


@app.command("build-stock-reaction-gold")
def build_stock_reaction_gold(
    symbols: Optional[str] = typer.Option(
        None,
        "--symbols",
        help="Comma-separated BIST tickers (e.g. 'AKBNK,GARAN'). Default: all BIST30.",
    ),
    windows: Optional[str] = typer.Option(
        None,
        "--windows",
        help="Comma-separated windows (e.g. 'w2,w5'). Default: w2,w3,w5.",
    ),
    run_backtest: bool = typer.Option(
        False,
        "--run-backtest",
        help="Also run full historical walk-forward backtests (slow).",
    ),
):
    """Execute specifically Model 3: BIST30 Stock Intraday Reaction Forecaster."""
    start_time = datetime.now()
    console.print("[bold cyan]🚀 Running Model 3: BIST30 Stock Intraday Reaction Forecaster...[/bold cyan]")

    db = DuckDBManager()
    engineer = GoldFeatureEngineer(db)
    symbols_list = [s.strip().upper() for s in symbols.split(",") if s.strip()] if symbols else None
    windows_list = [w.strip() for w in windows.split(",") if w.strip()] if windows else None

    res = engineer.run_stock_reaction_forecasting(
        symbols=symbols_list,
        windows=windows_list,
        run_backtest=run_backtest,
    )

    elapsed = (datetime.now() - start_time).total_seconds()
    console.print(
        Panel.fit(
            f"[bold green]✨ Model 3 Execution Complete in {elapsed:.1f}s[/bold green]\n\n"
            f"• Symbols Evaluated: [cyan]{res['symbols_run']}[/cyan]\n"
            f"• Windows Evaluated: [cyan]{res['windows_run']}[/cyan]\n"
            f"• Total Forecaster Runs: [cyan]{res['total_runs']}[/cyan]\n"
            f"• Successful: [green]{res['success_count']}[/green] | Errors: [red]{res['error_count']}[/red]",
            title="Stock Reaction Summary",
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


@app.command("audit-features")
def audit_features(
    model: str = typer.Option(
        "day_start",
        "--model",
        "-m",
        help="Target model to audit: 'day_start' or 'sector_day_start'",
    ),
    sector: Optional[str] = typer.Option(
        None,
        "--sector",
        "-s",
        help="Optional specific sector for sector_day_start",
    ),
    collinearity: float = typer.Option(
        0.85,
        "--collinearity",
        "-c",
        help="Pairwise correlation threshold to flag redundancy (|r| >= threshold)",
    ),
):
    """Run out-of-sample permutation drop testing and collinearity screening for feature pruning."""
    console.print(f"[bold cyan]Running Feature Selection & Redundancy Audit for '{model}'...[/bold cyan]")

    db = DuckDBManager(read_only=True)

    if model == "day_start":
        from mdk_trading_oracle.models.day_start.forecaster import DayStartForecaster

        forecaster = DayStartForecaster(db=db)
        report = forecaster.audit_features(collinearity_threshold=collinearity)
    elif model == "sector_day_start":
        from mdk_trading_oracle.models.sector_day_start.forecaster import SectorDayStartForecaster

        forecaster = SectorDayStartForecaster(db=db)
        report = forecaster.audit_features(sector=sector, collinearity_threshold=collinearity)
    else:
        console.print(f"[bold red]Unsupported model: {model}[/bold red]")
        raise typer.Exit(code=1)

    console.print(
        Panel(
            f"[bold green]Feature Audit Complete[/bold green] | Model: [cyan]{report.model_name}[/cyan] | "
            f"Evaluated Sessions: {report.evaluated_sessions} | Features: {report.total_features}",
            border_style="green",
        )
    )

    # Top Drivers Table
    if report.top_drivers:
        table_top = Table(title="Top 10 Out-of-Sample Alpha Drivers", title_style="bold green")
        table_top.add_column("Rank", style="dim")
        table_top.add_column("Feature", style="bold cyan")
        table_top.add_column("Cluster", style="magenta")
        table_top.add_column("Permutation Score Drop", justify="right", style="green")

        for idx, d in enumerate(report.top_drivers, 1):
            table_top.add_row(str(idx), d["feature_name"], d["cluster_name"], f"{d['permutation_score_drop']:+.4f}")
        console.print(table_top)

    # Collinear Pairs
    if report.collinear_pairs:
        table_corr = Table(title="Collinear Feature Pairs (|r| >= threshold)", title_style="bold yellow")
        table_corr.add_column("Feature A", style="cyan")
        table_corr.add_column("Feature B", style="red")
        table_corr.add_column("Correlation |r|", justify="right", style="bold yellow")
        table_corr.add_column("Cluster A", style="dim")
        table_corr.add_column("Cluster B", style="dim")

        for p in report.collinear_pairs:
            table_corr.add_row(
                p["feature_a"], p["feature_b"], f"{p['correlation']:.4f}", p["cluster_a"], p["cluster_b"]
            )
        console.print(table_corr)

    # Prune Candidates Table
    if report.prune_candidates:
        table_prune = Table(title="Recommended Features to Exclude (Zero Alpha / Redundant)", title_style="bold red")
        table_prune.add_column("Feature", style="bold red")
        table_prune.add_column("Cluster", style="dim")
        table_prune.add_column("Permutation Drop", justify="right")
        table_prune.add_column("Reason", style="yellow")

        for c in report.prune_candidates:
            table_prune.add_row(
                c["feature_name"], c["cluster_name"], f"{c['permutation_score_drop']:+.4f}", c["reason"]
            )
        console.print(table_prune)

    if report.recommended_features_yaml:
        console.print("\n[bold cyan]Recommended YAML Snippet for config/features.yaml:[/bold cyan]")
        console.print(Panel(report.recommended_features_yaml, border_style="cyan"))


@app.command("explain")
def explain_forecast(
    model: str = typer.Option(
        "day_start",
        "--model",
        "-m",
        help="Target model: 'day_start' or 'sector_day_start'",
    ),
    sector: Optional[str] = typer.Option(
        None,
        "--sector",
        "-s",
        help="Target sector for sector_day_start",
    ),
):
    """Explain upcoming live T+1 forecast via SHAP waterfall attribution."""
    db = DuckDBManager(read_only=True)

    if model == "day_start":
        from mdk_trading_oracle.explainability import format_markdown_card
        from mdk_trading_oracle.models.day_start.forecaster import DayStartForecaster

        forecaster = DayStartForecaster(db=db)
        exp = forecaster.explain_forecast()
        console.print(format_markdown_card(exp))
    elif model == "sector_day_start":
        from mdk_trading_oracle.explainability import format_markdown_card
        from mdk_trading_oracle.models.sector_day_start.forecaster import SectorDayStartForecaster

        sec = sector or "Banking"
        forecaster = SectorDayStartForecaster(db=db)
        exp = forecaster.explain_sector_forecast(sector=sec)
        if exp:
            console.print(format_markdown_card(exp))
        else:
            console.print(f"[red]Could not generate explanation for sector '{sec}'[/red]")


if __name__ == "__main__":
    app()
