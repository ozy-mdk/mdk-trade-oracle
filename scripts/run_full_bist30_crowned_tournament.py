"""Automated BIST 30 Full-Constituent Walk-Forward Model Tournament.

Evaluates all 30 BIST 30 equities across 5 model paradigms (XGBoost, LightGBM,
BayesianRidge, Huber, Ridge) and 3 dynamic lookback horizons (3M=63, 6M=126, 12M=252).

Crowning Rule (The 3 Core Champion Selection Criteria):
1. Directional Hit Rate % (±0.25% market consolidation deadband)
2. Hit MAE % (Calibration accuracy on correct calls)
3. Miss Drawdown MAE % (2.5x downside penalty on bad calls)

Composite Tournament Loss:
    Loss = (100.0 - hit_rate_pct) + 1.0 * hit_mae_pct + 2.5 * miss_mae_pct

Tie-Breaker Hierarchy:
    1. Significant Move Hit % (|Return| >= 1.0%)
    2. Directional Hit Rate %
    3. Price Calibration MAE %

Outputs:
- Updates config/tertip_crowned_models.yaml
- Updates PostgreSQL gold_tertip_daily_forecasts
- Clears in-memory forecast cache
"""

from __future__ import annotations

import argparse
import logging
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd
import yaml

from mdk_trading_oracle.core.db import PostgresManager
from mdk_trading_oracle.data.silver.tertip_ml_forecaster import (
    attach_prophet_rolling_features,
    clear_forecast_cache,
    extract_3pillar_time_series,
    run_30d_walk_forward_arena,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("bist30_tournament")

CONFIG_PATH = Path("config/tertip_crowned_models.yaml")
HORIZONS = {"3m": 63, "6m": 126, "12m": 252}
CANDIDATE_MODELS = ["XGBoost", "LightGBM", "BayesianRidge", "Huber", "Ridge"]


def run_tournament_for_symbol(
    db: PostgresManager,
    symbol: str,
    n_sessions: int = 30,
) -> dict[str, Any] | None:
    """Run walk-forward arena across all candidate models and horizons for a single equity."""
    sym = symbol.upper()
    df = extract_3pillar_time_series(db, sym, lookback_days=500)
    if df.empty or len(df) < (n_sessions + 35):
        logger.warning(f"Skipping {sym}: Insufficient historical data ({len(df)} rows)")
        return None

    df = attach_prophet_rolling_features(df, sym, n_history_needed=n_sessions + 40)

    best_config: dict[str, Any] | None = None
    best_key: tuple[float, float, float, float] | None = None

    for h_name, h_lb in HORIZONS.items():
        _, summary, _ = run_30d_walk_forward_arena(
            df,
            n_sessions=n_sessions,
            train_lookback_sessions=h_lb,
        )

        for model_name in CANDIDATE_MODELS:
            prefix = "bayesian_ridge" if model_name == "BayesianRidge" else model_name.lower()

            hits = summary.get(f"{prefix}_dir_hits", 0)
            hit_rate = summary.get(f"{prefix}_30d_hit_rate_pct", 0.0)
            mae = summary.get(f"{prefix}_mae_pct", 0.0)
            hit_mae = summary.get(f"{prefix}_hit_mae_pct", 0.0)
            miss_mae = summary.get(f"{prefix}_miss_mae_pct", 0.0)
            sig_hits = summary.get(f"{prefix}_30d_sig_move_hits", 0)
            sig_total = summary.get(f"{prefix}_30d_sig_move_total", 0)
            sig_rate = summary.get(f"{prefix}_30d_sig_move_hit_rate_pct", 0.0)
            big_hits = summary.get(f"{prefix}_30d_big_move_hits", 0)
            big_total = summary.get(f"{prefix}_30d_big_move_total", 0)
            big_rate = summary.get(f"{prefix}_30d_big_move_hit_rate_pct", 0.0)
            quiet_fa = summary.get(f"{prefix}_30d_quiet_false_alarms", 0)
            loss = summary.get(f"{prefix}_tournament_loss_30d", 999.0)

            cfg = {
                "symbol": sym,
                "model": model_name,
                "horizon": h_name,
                "training_lookback_sessions": h_lb,
                "recent_30d_hits": int(hits),
                "recent_30d_hit_rate_pct": float(round(hit_rate, 1)),
                "recent_30d_mae_pct": float(round(mae, 2)),
                "recent_30d_hit_mae_pct": float(round(hit_mae, 2)),
                "recent_30d_miss_mae_pct": float(round(miss_mae, 2)),
                "recent_30d_sig_move_hits": int(sig_hits),
                "recent_30d_sig_move_total": int(sig_total),
                "recent_30d_sig_move_hit_rate_pct": float(round(sig_rate, 1)),
                "recent_30d_quiet_false_alarms": int(quiet_fa),
                "recent_30d_big_move_hits": int(big_hits),
                "recent_30d_big_move_total": int(big_total),
                "recent_30d_big_move_hit_rate_pct": float(round(big_rate, 1)),
                "recent_30d_penalty_loss": float(round(loss, 2)),
            }

            # Minimization tuple: (Composite Loss, -Sig Move Hit %, -Overall Hit %, MAE)
            key = (loss, -sig_rate, -hit_rate, mae)
            if best_key is None or key < best_key:
                best_key = key
                best_config = cfg

    return best_config


def main() -> None:
    parser = argparse.ArgumentParser(description="Run Full BIST 30 Tournament")
    parser.add_argument("--symbols", nargs="*", help="Optional subset of symbols to evaluate")
    parser.add_argument("--dry-run", action="store_true", help="Print scorecard without writing to disk or DB")
    args = parser.parse_args()

    db = PostgresManager()

    # Load existing configuration to compare deltas
    old_config = {}
    if CONFIG_PATH.exists():
        with open(CONFIG_PATH, "r", encoding="utf-8") as f:
            old_config = yaml.safe_load(f).get("symbols", {})

    # Determine symbols universe
    if args.symbols:
        eval_symbols = [s.upper() for s in args.symbols]
    else:
        eval_symbols = sorted(list(old_config.keys())) if old_config else [
            "AEFES", "AKBNK", "ASELS", "ASTOR", "BIMAS", "DSTKF", "EKGYO", "ENKAI", "EREGL", "FROTO",
            "GARAN", "GUBRF", "ISCTR", "KCHOL", "KRDMD", "MGROS", "PETKM", "PGSUS", "SAHOL", "SASA",
            "SISE", "TAVHL", "TCELL", "THYAO", "TOASO", "TRALT", "TTKOM", "TUPRS", "VAKBN", "YKBNK"
        ]

    logger.info(f"Starting Full BIST 30 Tournament for {len(eval_symbols)} equities...")
    logger.info("Candidates: XGBoost, LightGBM, BayesianRidge, Huber, Ridge across [3M, 6M, 12M]")
    t_start = time.time()

    new_results: dict[str, Any] = {}
    scorecard_rows: list[dict[str, Any]] = []

    for idx, sym in enumerate(eval_symbols, 1):
        sym_t0 = time.time()
        print(f"\n[{idx:02d}/{len(eval_symbols):02d}] Benchmarking {sym} across 15 configurations...", flush=True)
        res = run_tournament_for_symbol(db, sym)
        if not res:
            continue

        sym_elapsed = time.time() - sym_t0
        new_results[sym] = res

        old_meta = old_config.get(sym, {})
        old_model = f"{old_meta.get('model', 'None')} ({old_meta.get('horizon', '?')})"
        old_rate = old_meta.get("recent_30d_hit_rate_pct", 0.0)
        old_mae = old_meta.get("recent_30d_mae_pct", 0.0)

        new_model = f"{res['model']} ({res['horizon']})"
        new_hits = res["recent_30d_hits"]
        new_rate = res["recent_30d_hit_rate_pct"]
        new_mae = res["recent_30d_mae_pct"]
        hit_delta = new_rate - old_rate

        print(
            f"   -> Winner: {new_model:<18} | Hits: {new_hits}/30 ({new_rate:>4.1f}%) | "
            f"MAE: {new_mae:>4.2f}% | SigRate: {res['recent_30d_sig_move_hit_rate_pct']:>4.1f}% | "
            f"Loss: {res['recent_30d_penalty_loss']:>6.2f} ({sym_elapsed:.1f}s)",
            flush=True,
        )

        scorecard_rows.append({
            "Stock": sym,
            "Incumbent": old_model,
            "Old Hit%": f"{old_rate:.1f}%",
            "Old MAE%": f"{old_mae:.2f}%",
            "New Champion": new_model,
            "New Hit%": f"{new_rate:.1f}%",
            "New MAE%": f"{new_mae:.2f}%",
            "Hit% Delta": f"{'+' if hit_delta > 0 else ''}{hit_delta:.1f}%",
            "Sig Move Hit%": f"{res['recent_30d_sig_move_hit_rate_pct']:.1f}%",
            "Loss": f"{res['recent_30d_penalty_loss']:.2f}",
        })

    # Print Final Scorecard Table
    df_sc = pd.DataFrame(scorecard_rows)
    print("\n" + "=" * 125)
    print("MDK TRADING ORACLE — FULL BIST 30 MODEL TOURNAMENT SCORECARD")
    print("=" * 125)
    print(df_sc.to_string(index=False))
    print("=" * 125)

    total_elapsed = time.time() - t_start
    avg_new_hit = sum(r["recent_30d_hit_rate_pct"] for r in new_results.values()) / len(new_results)
    avg_new_mae = sum(r["recent_30d_mae_pct"] for r in new_results.values()) / len(new_results)
    print(f"\nTournament completed in {total_elapsed:.1f}s (Average {total_elapsed/len(eval_symbols):.1f}s per stock)")
    print(f"Overall BIST 30 Directional Hit Rate: {avg_new_hit:.1f}% | Average MAE: {avg_new_mae:.2f}%\n")

    if args.dry_run:
        logger.info("Dry-run requested. Skipped updating YAML and PostgreSQL.")
        return

    # Update YAML configuration
    updated_config = {
        "_metadata": {
            "description": "Crowned Tertip Confluence models and training lookback horizons selected via 3-Criteria Tournament Arena",
            "calibration_date": datetime.now(timezone.utc).strftime("%Y-%m-%d"),
            "eval_window": "30d_walk_forward_multi_horizon",
            "selection_metric": "3_criteria_tournament_loss",
        },
        "symbols": {sym: {k: v for k, v in meta.items() if k != "symbol"} for sym, meta in new_results.items()},
    }

    with open(CONFIG_PATH, "w", encoding="utf-8") as f:
        yaml.dump(updated_config, f, sort_keys=False, indent=2)
    logger.info(f"Successfully saved new crowned champions to {CONFIG_PATH}")

    # Synchronize PostgreSQL gold_tertip_daily_forecasts
    logger.info("Synchronizing PostgreSQL gold_tertip_daily_forecasts...")
    res_date = db.query_pl("SELECT MAX(as_of_date) as max_date FROM gold_tertip_daily_forecasts;")
    max_date = res_date["max_date"][0]
    if max_date is not None:
        update_query = """
        UPDATE gold_tertip_daily_forecasts
        SET 
            ml_champion_type = %(ml_champion_type)s,
            crowned_horizon = %(crowned_horizon)s,
            training_lookback_sessions = %(training_lookback_sessions)s,
            champion_dir_hits = %(champion_dir_hits)s,
            champion_dir_hit_rate_pct = %(champion_dir_hit_rate_pct)s,
            champion_mae_pct = %(champion_mae_pct)s
        WHERE symbol = %(symbol)s AND as_of_date = %(as_of_date)s;
        """
        for sym, meta in new_results.items():
            db.execute(update_query, {
                "symbol": sym,
                "as_of_date": max_date,
                "ml_champion_type": meta["model"],
                "crowned_horizon": meta["horizon"],
                "training_lookback_sessions": meta["training_lookback_sessions"],
                "champion_dir_hits": meta["recent_30d_hits"],
                "champion_dir_hit_rate_pct": meta["recent_30d_hit_rate_pct"],
                "champion_mae_pct": meta["recent_30d_mae_pct"],
            })
        logger.info(f"Synchronized {len(new_results)} equities in gold_tertip_daily_forecasts for as_of_date {max_date}.")

    # Invalidate in-memory caches
    clear_forecast_cache()
    logger.info("Cleared in-memory forecast cache. All frontend endpoints will immediately serve fresh champions.")


if __name__ == "__main__":
    main()
