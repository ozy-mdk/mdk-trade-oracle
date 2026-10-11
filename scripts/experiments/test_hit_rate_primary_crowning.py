"""Experiment: Prioritizing Directional Hit Rate % as Primary Crowning Metric.

Compares:
1. Current Status Quo: Crowning by Composite Loss (with 2.5x miss_mae penalty).
2. Proposed System: Crowning primarily by Directional Hit Rate % (Recency-Weighted),
   with Recent-20-session Hit Rate as secondary, and MAE as tie-breaker.

Tested across:
ASELS, AKBNK, GARAN, ISCTR, YKBNK
Across all 5 horizons (3D, 5D, 10D, 15D, 30D).
"""

from __future__ import annotations

import os

os.environ["OMP_NUM_THREADS"] = "1"

import numpy as np
import pandas as pd
from rich.console import Console
from rich.table import Table

from mdk_trading_oracle.core.db import PostgresManager
from mdk_trading_oracle.data.silver.tertip_ml_forecaster import extract_3pillar_time_series
from mdk_trading_oracle.data.silver.tertip_multi_horizon_forecaster import (
    engineer_multi_horizon_features,
    evaluate_horizon_walk_forward,
    generate_live_horizon_forecast,
)

console = Console()

CANDIDATES = ["Ridge", "BayesianRidge", "Huber", "XGBoost", "LightGBM"]
LOOKBACKS = [("3m", 63), ("6m", 126), ("12m", 252), ("18m", 378), ("24m", 504)]
HORIZONS = ["3d", "5d", "10d", "15d", "30d"]
SYMBOLS = ["ASELS", "AKBNK", "GARAN", "ISCTR", "YKBNK"]


def evaluate_symbol(db: PostgresManager, symbol: str):
    df_raw = extract_3pillar_time_series(db, symbol, lookback_days=1100)
    df_feat, _ = engineer_multi_horizon_features(df_raw)

    results = []

    for h in HORIZONS:
        all_evals = []

        for m in CANDIDATES:
            for lb_name, lb_s in LOOKBACKS:
                res = evaluate_horizon_walk_forward(
                    df=df_feat,
                    horizon_key=h,
                    model_name=m,
                    train_lookback_sessions=lb_s,
                    n_eval_sessions=60,
                )
                ledger = res.get("ledger", [])
                recent_20 = ledger[-20:] if len(ledger) >= 20 else ledger
                rec_hits = sum(1 for r in recent_20 if r.get("is_hit"))
                rec_rate = (rec_hits / len(recent_20) * 100.0) if recent_20 else 0.0

                total_evals = len(ledger)
                weights = np.linspace(0.5, 1.5, total_evals) / np.mean(np.linspace(0.5, 1.5, total_evals)) if total_evals > 1 else np.ones(total_evals)
                hits_arr = np.array([1.0 if r.get("is_hit") else 0.0 for r in ledger])
                w_hit_rate = float(np.sum(hits_arr * weights) / np.sum(weights) * 100.0) if total_evals > 0 else 0.0

                all_evals.append({
                    "model": m,
                    "lookback": lb_name,
                    "sessions": lb_s,
                    "hit_rate_pct": res.get("hit_rate_pct", 0.0),
                    "weighted_hit_rate_pct": w_hit_rate,
                    "recent_20_hit_rate": rec_rate,
                    "recent_20_hits": rec_hits,
                    "hits": res.get("hits", 0),
                    "total_evals": total_evals,
                    "mae_pct": res.get("mae_pct", 0.0),
                    "hit_mae_pct": res.get("hit_mae_pct", 0.0),
                    "miss_mae_pct": res.get("miss_mae_pct", 0.0),
                    "loss": res.get("loss", 999.0),
                })

        # 1. Status Quo Champion (min composite loss)
        sq_champ = sorted(all_evals, key=lambda x: (x["loss"], -x["hit_rate_pct"]))[0]

        # 2. Proposed Champion: Primary = Weighted Hit Rate, Secondary = Recent 20 Hit Rate, Tie-breaker = MAE
        prop_champ = sorted(
            all_evals,
            key=lambda x: (
                -round(x["weighted_hit_rate_pct"], 1),
                -round(x["recent_20_hit_rate"], 1),
                x["mae_pct"],
            )
        )[0]

        # Generate live forecast for both to check sign and expected return
        sq_live = generate_live_horizon_forecast(
            df=df_feat,
            horizon_key=h,
            champion_model=sq_champ["model"],
            train_lookback_sessions=sq_champ["sessions"],
        )
        prop_live = generate_live_horizon_forecast(
            df=df_feat,
            horizon_key=h,
            champion_model=prop_champ["model"],
            train_lookback_sessions=prop_champ["sessions"],
        )

        results.append({
            "horizon": h,
            "sq_champ": f"{sq_champ['model']} ({sq_champ['lookback']})",
            "sq_win": sq_champ["hit_rate_pct"],
            "sq_recent": sq_champ["recent_20_hit_rate"],
            "sq_ret": prop_live.get("live_forecast", {}).get("expected_return_pct", 0.0) if sq_champ == prop_champ else sq_live.get("live_forecast", {}).get("expected_return_pct", 0.0),
            "sq_stance": sq_live.get("live_forecast", {}).get("stance", "N/A"),
            "prop_champ": f"{prop_champ['model']} ({prop_champ['lookback']})",
            "prop_win": prop_champ["hit_rate_pct"],
            "prop_recent": prop_champ["recent_20_hit_rate"],
            "prop_ret": prop_live.get("live_forecast", {}).get("expected_return_pct", 0.0),
            "prop_stance": prop_live.get("live_forecast", {}).get("stance", "N/A"),
        })

    return results


def main():
    db = PostgresManager()
    console.print("\n[bold yellow]" + "=" * 96 + "[/bold yellow]")
    console.print("[bold cyan]EXPERIMENT: HIT-RATE-PRIMARY CROWNING vs STATUS QUO[/bold cyan]")
    console.print("[bold yellow]" + "=" * 96 + "[/bold yellow]")

    for sym in SYMBOLS:
        res = evaluate_symbol(db, sym)
        table = Table(title=f"📊 {sym} Multi-Horizon Comparison", title_style="bold green")
        table.add_column("Horizon", style="bold cyan")
        table.add_column("Status Quo Model", style="magenta")
        table.add_column("SQ Win % (Full / Recent)", justify="right", style="magenta")
        table.add_column("SQ Signal", justify="center", style="magenta")
        table.add_column("Hit-Rate Primary Model", style="green")
        table.add_column("New Win % (Full / Recent)", justify="right", style="green")
        table.add_column("New Signal", justify="center", style="green")

        for r in res:
            sq_win_str = f"{r['sq_win']:.1f}% / {r['sq_recent']:.0f}%"
            prop_win_str = f"{r['prop_win']:.1f}% / {r['prop_recent']:.0f}%"
            sq_sig = f"{r['sq_ret']:+.1f}% ({r['sq_stance'][:4]})"
            prop_sig = f"{r['prop_ret']:+.1f}% ({r['prop_stance'][:4]})"

            table.add_row(
                r["horizon"].upper(),
                r["sq_champ"],
                sq_win_str,
                sq_sig,
                r["prop_champ"],
                prop_win_str,
                prop_sig,
            )
        console.print(table)


if __name__ == "__main__":
    main()
