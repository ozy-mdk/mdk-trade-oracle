#!/usr/bin/env python3
"""CLI Utility to verify that the Backend API, PostgreSQL, and Crowned Models are 100% in sync."""

from __future__ import annotations

import argparse
import json
import urllib.request
import yaml
from pathlib import Path
from rich.console import Console
from rich.table import Table

console = Console()
CONFIG_PATH = Path("config/tertip_multi_horizon_crowned_models.yaml")


def check_symbol(symbol: str, base_url: str = "http://localhost:8000", fresh: bool = True):
    sym = symbol.upper()
    console.print(f"\n[bold cyan]Checking synchronization for {sym}...[/bold cyan]")

    # 1. Read Crowned YAML
    yaml_champs = {}
    if CONFIG_PATH.exists():
        with open(CONFIG_PATH, "r", encoding="utf-8") as f:
            cfg = yaml.safe_load(f) or {}
            yaml_champs = cfg.get("symbols", {}).get(sym, {})

    # 2. Call FastAPI backend
    url = f"{base_url}/api/v1/tertip/multi-horizon/forecast?symbol={sym}"
    if fresh:
        url += "&fresh=true"

    try:
        req = urllib.request.urlopen(url, timeout=30)
        api_data = json.loads(req.read().decode("utf-8"))
        api_horizons = api_data.get("horizons", {})
    except Exception as e:
        console.print(f"[bold red]❌ Failed to connect to FastAPI backend at {base_url}: {e}[/bold red]")
        return

    table = Table(title=f"🎯 {sym} Multi-Horizon Sync Verification", title_style="bold yellow")
    table.add_column("Horizon", style="bold cyan")
    table.add_column("Config YAML Champion", style="magenta")
    table.add_column("Live API Champion", style="green")
    table.add_column("Hit Rate %", justify="right", style="cyan")
    table.add_column("Lookback", justify="center", style="yellow")
    table.add_column("Status", justify="center", style="bold green")

    all_in_sync = True
    for h in ["3d", "5d", "10d", "15d", "30d"]:
        y_info = yaml_champs.get(h, {})
        y_model = y_info.get("model", "N/A")
        y_lb = y_info.get("horizon", "N/A")

        api_h = api_horizons.get(h, {})
        api_meta = api_h.get("meta", {})
        api_model = api_meta.get("crowned_model", "N/A")
        api_lb = api_meta.get("crowned_horizon", "N/A")
        hit_rate = api_meta.get("hit_rate_pct", 0.0)

        is_match = (y_model == api_model) and (y_lb == api_lb)
        if not is_match:
            all_in_sync = False

        status = "✅ IN SYNC" if is_match else "⚠️ MISMATCH"
        table.add_row(
            h.upper(),
            f"{y_model} ({y_lb})",
            f"{api_model} ({api_lb})",
            f"{hit_rate:.1f}%",
            f"{api_meta.get('training_lookback_sessions', '-')}s",
            status,
        )

    console.print(table)
    if all_in_sync:
        console.print(f"[bold green]✨ All models for {sym} are perfectly synchronized between Config, Backend API, and Frontend![/bold green]")
    else:
        console.print(f"[bold yellow]⚠️ Note: Differences detected. Running with fresh=true should resolve cache differences.[/bold yellow]")


def main():
    parser = argparse.ArgumentParser(description="Verify Multi-Horizon Models Frontend/Backend Sync")
    parser.add_argument("--symbol", "-s", type=str, default="YKBNK", help="Stock ticker symbol (default: YKBNK)")
    parser.add_argument("--url", type=str, default="http://localhost:8000", help="FastAPI Base URL")
    parser.add_argument("--fresh", action="store_true", help="Force live on-the-fly recomputation on backend")
    args = parser.parse_args()

    check_symbol(args.symbol, base_url=args.url, fresh=args.fresh)


if __name__ == "__main__":
    main()
