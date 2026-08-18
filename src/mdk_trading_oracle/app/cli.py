"""Typer CLI interface for MDK Trading Oracle."""

from pathlib import Path
from typing import Optional
import typer
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from mdk_trading_oracle.core.config import get_settings
from mdk_trading_oracle.core.db import DuckDBManager
from mdk_trading_oracle.core.logger import get_logger
from mdk_trading_oracle.ingestion.file_loader import FileIngestor
from mdk_trading_oracle.oracle.evaluator import OracleEvaluator
from mdk_trading_oracle.pipeline.bronze_to_silver import BronzeToSilverPipeline
from mdk_trading_oracle.pipeline.silver_to_gold import SilverToGoldPipeline

app = typer.Typer(
    name="mdk-oracle",
    help="MDK Trading Oracle - Institutional Flow Decision Support Engine",
    add_completion=False,
)
console = Console()
logger = get_logger("mdk_oracle.cli")


@app.command("init-db")
def init_db():
    """Initialize DuckDB database tables and sync config definitions."""
    db = DuckDBManager()
    db.initialize_schema()
    console.print("[bold green]✔ DuckDB schema and reference tables initialized successfully.[/bold green]")


@app.command("ingest")
def ingest(file_path: Optional[Path] = typer.Option(None, "--file", "-f", help="Path to raw CSV/Parquet file")):
    """Ingest raw trade dumps into Bronze layer."""
    db = DuckDBManager()
    db.initialize_schema()
    ingestor = FileIngestor(db)

    if file_path:
        res = ingestor.ingest(file_path)
        console.print(f"[bold green]✔ Ingested {res['rows_ingested']} rows from {res['source_file']}[/bold green]")
    else:
        results = ingestor.ingest_all_bronze()
        total_rows = sum(r["rows_ingested"] for r in results)
        console.print(f"[bold green]✔ Ingested {len(results)} files ({total_rows} total rows) into Bronze.[/bold green]")


@app.command("run-pipeline")
def run_pipeline():
    """Execute complete Bronze -> Silver -> Gold transformation."""
    db = DuckDBManager()
    db.initialize_schema()

    console.print("[bold blue]Starting MDK Medallion Data Pipeline...[/bold blue]")

    # Ingest pending files if any
    ingestor = FileIngestor(db)
    ingestor.ingest_all_bronze()

    # Step 1: Bronze -> Silver
    b2s = BronzeToSilverPipeline(db)
    s_res = b2s.run()
    console.print(f"[cyan]• Silver Transactions:[/cyan] {s_res['silver_broker_transactions_count']} rows")
    console.print(f"[cyan]• Silver Daily Summaries:[/cyan] {s_res['silver_daily_broker_summary_count']} rows")

    # Step 2: Silver -> Gold
    s2g = SilverToGoldPipeline(db)
    g_res = s2g.run()
    console.print(f"[cyan]• Gold Flow Records:[/cyan] {g_res['gold_bofa_flow_metrics_count']} rows")

    # Step 3: Oracle Signal Generation
    evaluator = OracleEvaluator(db)
    signals = evaluator.evaluate_latest_signals()

    console.print(f"[bold green]✔ Pipeline completed successfully! Generated {len(signals)} Oracle decision signals.[/bold green]")


@app.command("evaluate")
def evaluate(symbol: Optional[str] = typer.Option(None, "--symbol", "-s", help="Specific ticker symbol (e.g. AKBNK, GARAN)")):
    """Evaluate institutional flow and display Oracle trading signals."""
    db = DuckDBManager()
    evaluator = OracleEvaluator(db)

    if symbol:
        sig = evaluator.get_symbol_signal(symbol.upper())
        if not sig:
            console.print(f"[bold red]No Gold metrics found for symbol: {symbol}[/bold red]")
            raise typer.Exit(code=1)

        color = "green" if "BUY" in sig.signal.value else "red" if "SELL" in sig.signal.value else "yellow"
        panel_content = f"""[bold {color}]Signal:[/bold {color}] {sig.signal.value} (Confidence: {sig.confidence:.0%})
[bold]Date:[/bold] {sig.date_val}
[bold]BofA Net Flow:[/bold] {sig.bofa_net_tl:,.0f} TL
[bold]BofA Net Share:[/bold] {sig.bofa_net_share:.1%}
[bold]Flow Z-Score:[/bold] {sig.bofa_flow_zscore:+.2f}

[bold underline]Reasoning:[/bold underline]
""" + "\n".join(f"• {r}" for r in sig.reasons)

        console.print(Panel(panel_content, title=f"[bold]Oracle Decision: {sig.symbol}[/bold]", border_style=color))

    else:
        signals = evaluator.evaluate_latest_signals()
        if not signals:
            console.print("[yellow]No data available to evaluate. Run 'mdk-oracle run-pipeline' first.[/yellow]")
            return

        table = Table(title="MDK Trading Oracle - Latest Institutional Flow Signals")
        table.add_column("Symbol", style="bold cyan")
        table.add_column("Signal", style="bold")
        table.add_column("Confidence", justify="right")
        table.add_column("BofA Net (TL)", justify="right")
        table.add_column("BofA Net %", justify="right")
        table.add_column("Flow Z-Score", justify="right")
        table.add_column("Summary")

        for s in signals:
            color = "green" if "BUY" in s.signal.value else "red" if "SELL" in s.signal.value else "yellow"
            table.add_row(
                s.symbol,
                f"[{color}]{s.signal.value}[/{color}]",
                f"{s.confidence:.0%}",
                f"{s.bofa_net_tl:,.0f}",
                f"{s.bofa_net_share:+.1%}",
                f"{s.bofa_flow_zscore:+.2f}",
                s.summary,
            )

        console.print(table)


if __name__ == "__main__":
    app()
