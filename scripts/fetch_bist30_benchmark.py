"""Script to fetch official BIST 30 (XU030.IS) benchmark historical data from Yahoo Finance."""

import argparse
from datetime import datetime, timedelta
from pathlib import Path

import pandas as pd
import yfinance as yf
from rich.console import Console

from mdk_trading_oracle.core.config import get_settings
from mdk_trading_oracle.core.db import DuckDBManager
from mdk_trading_oracle.core.logger import get_logger

logger = get_logger("mdk_oracle.scripts.fetch_bist30")
console = Console()


def fetch_bist30_data(
    years: int = 5,
    ticker_symbol: str = "XU030.IS",
    save_csv: bool = True,
    save_excel: bool = True,
    save_db: bool = False,
    output_dir: Path = None,
) -> pd.DataFrame:
    """Download historical index data, format columns, and optionally save to CSV/Excel/DuckDB."""
    settings = get_settings()
    out_dir = output_dir or settings.data_dir / "00_raw_data" / "benchmarks"
    out_dir.mkdir(parents=True, exist_ok=True)

    end_date = datetime.today().strftime("%Y-%m-%d")
    start_date = (datetime.today() - timedelta(days=years * 365)).strftime("%Y-%m-%d")

    console.print(f"[bold cyan]Fetching {ticker_symbol} benchmark data from {start_date} to {end_date}...[/bold cyan]")

    # Download from yfinance
    df = yf.download(ticker_symbol, start=start_date, end=end_date, progress=False)

    if df.empty:
        console.print("[bold red]Failed to fetch data. Check internet connection or ticker symbol.[/bold red]")
        return pd.DataFrame()

    # Flatten MultiIndex column headers if present (common in modern yfinance versions)
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)

    # Reset Date index
    df = df.reset_index()
    df.columns.name = None

    # Clean and standardize column names
    df["Date"] = pd.to_datetime(df["Date"]).dt.strftime("%Y-%m-%d")
    df = df.sort_values(by="Date", ascending=True).reset_index(drop=True)

    # Add useful derived metrics
    df["Daily_Return_Pct"] = df["Close"].pct_change()
    df["Price_Range_Pct"] = (df["High"] - df["Low"]) / df["Low"].replace(0, pd.NA)

    # Reorder columns cleanly
    col_order = ["Date", "Open", "High", "Low", "Close", "Volume", "Daily_Return_Pct", "Price_Range_Pct"]
    existing_cols = [c for c in col_order if c in df.columns]
    remaining_cols = [c for c in df.columns if c not in existing_cols]
    df = df[existing_cols + remaining_cols]

    # Save to CSV
    csv_file = out_dir / "bist30_5year_data.csv"
    if save_csv:
        df.to_csv(csv_file, index=False)
        console.print(f"[green]Saved CSV to:[/green] `{csv_file}`")

    # Save to Excel
    excel_file = out_dir / "bist30_5year_data.xlsx"
    if save_excel:
        df.to_excel(excel_file, index=False)
        console.print(f"[green]Saved Excel to:[/green] `{excel_file}`")

    # Optionally persist to DuckDB
    if save_db:
        db = DuckDBManager()
        conn = db.get_connection()
        conn.execute("""
            CREATE TABLE IF NOT EXISTS bronze_bist_index_benchmarks (
                trade_date DATE PRIMARY KEY,
                index_code VARCHAR DEFAULT 'XU030',
                open_price DOUBLE,
                high_price DOUBLE,
                low_price DOUBLE,
                close_price DOUBLE NOT NULL,
                volume DOUBLE,
                daily_return_pct DOUBLE,
                price_range_pct DOUBLE,
                source VARCHAR DEFAULT 'yfinance_XU030.IS',
                ingested_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            );
        """)

        # Insert or replace records
        records_to_insert = [
            (
                row["Date"],
                "XU030",
                float(row["Open"]) if pd.notna(row.get("Open")) else None,
                float(row["High"]) if pd.notna(row.get("High")) else None,
                float(row["Low"]) if pd.notna(row.get("Low")) else None,
                float(row["Close"]),
                float(row["Volume"]) if pd.notna(row.get("Volume")) else None,
                float(row["Daily_Return_Pct"]) if pd.notna(row.get("Daily_Return_Pct")) else None,
                float(row["Price_Range_Pct"]) if pd.notna(row.get("Price_Range_Pct")) else None,
            )
            for _, row in df.iterrows()
        ]

        conn.executemany("""
            INSERT OR REPLACE INTO bronze_bist_index_benchmarks (
                trade_date, index_code, open_price, high_price, low_price, close_price, volume, daily_return_pct, price_range_pct
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?);
        """, records_to_insert)
        console.print(f"[green]Synced {len(records_to_insert)} records into DuckDB table `bronze_bist_index_benchmarks`[/green]")

    console.print(f"\n[bold green]Success! Downloaded {len(df)} trading days of data.[/bold green]")
    console.print("\n[bold]Data Preview (Most Recent Rows):[/bold]")
    console.print(df.tail())

    return df


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Fetch BIST 30 historical benchmark data from Yahoo Finance.")
    parser.add_argument("--years", type=int, default=5, help="Number of historical years to fetch (default: 5)")
    parser.add_argument("--ticker", type=str, default="XU030.IS", help="Yahoo Finance ticker symbol (default: XU030.IS)")
    parser.add_argument("--save-db", action="store_true", help="Sync data into DuckDB table `bronze_bist_index_benchmarks`")
    args = parser.parse_args()

    fetch_bist30_data(years=args.years, ticker_symbol=args.ticker, save_db=args.save_db)
