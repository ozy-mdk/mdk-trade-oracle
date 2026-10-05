"""Automated BIST 30 Full-Constituent Week Start (Monday Day Close) Model Tournament.

Evaluates all 30 BIST 30 equities across 5 model paradigms (LightGBM, XGBoost, Huber,
Ridge, BayesianRidge) and 3 dynamic lookback horizons (6M=26w, 12M=52w, 24M=104w)
over the trailing 20 Monday trading sessions.

Crowning Rule (The 3 Core Champion Selection Criteria):
1. Directional Hit Rate % (±0.25% market consolidation deadband)
2. Hit MAE % (Calibration accuracy on correct calls)
3. Miss Drawdown MAE % (2.5x downside penalty on bad calls)

Composite Tournament Loss:
    Loss = (100.0 - hit_rate_pct) + 1.0 * hit_mae_pct + 2.5 * miss_mae_pct

Outputs:
- Updates config/week_start_crowned_models.yaml
- Updates PostgreSQL gold_week_start_daily_forecasts & backtests
"""

from __future__ import annotations

import argparse
import logging
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import yaml

from mdk_trading_oracle.core.db import PostgresManager
from mdk_trading_oracle.data.gold.week_start_forecasts import sync_week_start_forecasts_from_yaml
from mdk_trading_oracle.data.silver.week_start_forecaster import (
    CANDIDATE_MODELS,
    WEEKLY_HORIZONS,
    extract_week_start_time_series,
    run_weekly_walk_forward_arena,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("week_start_tournament")

CONFIG_PATH = Path("config/week_start_crowned_models.yaml")


def run_tournament_for_symbol(
    db: PostgresManager,
    symbol: str,
    n_weeks: int = 20,
) -> dict[str, Any] | None:
    """Run weekly walk-forward arena across all candidate models and horizons for a single equity."""
    sym = symbol.upper()
    df = extract_week_start_time_series(db, sym)
    if df.empty or len(df) < (n_weeks + 25):
        logger.warning("Skipping %s: Insufficient historical weekly data (%d rows)", sym, len(df))
        return None

    best_config: dict[str, Any] | None = None
    best_key: tuple[float, float, float, float] | None = None

    for h_name, h_lb in WEEKLY_HORIZONS.items():
        _, summary, _ = run_weekly_walk_forward_arena(
            df,
            n_weeks=n_weeks,
            lookback_weeks=h_lb,
        )

        for model_name in CANDIDATE_MODELS:
            prefix = model_name.lower()

            hits = summary.get(f"{prefix}_dir_hits", 0)
            hit_rate = summary.get(f"{prefix}_hit_rate_pct", 0.0)
            mae = summary.get(f"{prefix}_mae_pct", 0.0)
            hit_mae = summary.get(f"{prefix}_hit_mae_pct", 0.0)
            miss_mae = summary.get(f"{prefix}_miss_mae_pct", 0.0)
            sig_hits = summary.get(f"{prefix}_sig_hits", 0)
            sig_total = summary.get(f"{prefix}_sig_total", 0)
            sig_rate = summary.get(f"{prefix}_sig_rate_pct", 0.0)
            loss = summary.get(f"{prefix}_tournament_loss", 999.0)

            cfg = {
                "symbol": sym,
                "model": model_name,
                "horizon": h_name,
                "training_lookback_weeks": h_lb,
                "recent_20w_hits": int(hits),
                "recent_20w_hit_rate_pct": float(round(hit_rate, 1)),
                "recent_20w_mae_pct": float(round(mae, 2)),
                "recent_20w_hit_mae_pct": float(round(hit_mae, 2)),
                "recent_20w_miss_mae_pct": float(round(miss_mae, 2)),
                "recent_20w_sig_hits": int(sig_hits),
                "recent_20w_sig_total": int(sig_total),
                "recent_20w_sig_rate_pct": float(round(sig_rate, 1)),
                "recent_20w_penalty_loss": float(round(loss, 2)),
            }

            key = (loss, -sig_rate, -hit_rate, mae)
            if best_key is None or key < best_key:
                best_key = key
                best_config = cfg

    return best_config


def main() -> None:
    parser = argparse.ArgumentParser(description="Run BIST 30 Full-Constituent Week Start Tournament")
    parser.add_argument("--weeks", type=int, default=20, help="Walk-forward evaluation window in Mondays (default: 20)")
    parser.add_argument("--symbol", type=str, default=None, help="Optional single stock ticker to evaluate")
    args = parser.parse_args()

    db = PostgresManager()

    # Query BIST 30 symbols
    if args.symbol:
        symbols = [args.symbol.upper()]
    else:
        # Load from config/tertip_crowned_models.yaml if exists, or query from DB
        tertip_cfg_path = Path("config/tertip_crowned_models.yaml")
        if tertip_cfg_path.exists():
            with open(tertip_cfg_path, "r", encoding="utf-8") as f:
                t_cfg = yaml.safe_load(f)
            symbols = sorted(list(t_cfg.get("symbols", {}).keys()))
        else:
            rows = db.query_df(
                "SELECT DISTINCT symbol FROM bronze_instruments WHERE is_active = TRUE ORDER BY symbol LIMIT 30;"
            )
            symbols = rows["symbol"].tolist()

    logger.info("Starting Week Start Tournament for %d symbols across %d Mondays...", len(symbols), args.weeks)
    t0 = time.time()

    results: dict[str, Any] = {}
    for idx, sym in enumerate(symbols, 1):
        t_sym = time.time()
        crowned = run_tournament_for_symbol(db, sym, n_weeks=args.weeks)
        dur = time.time() - t_sym

        if crowned:
            results[sym] = crowned
            logger.info(
                "[%d/%d] %s -> CROWNED %s (%s, %dw) | Hit Rate: %.1f%% (%d/%d) | Loss: %.2f (took %.1fs)",
                idx,
                len(symbols),
                sym,
                crowned["model"],
                crowned["horizon"],
                crowned["training_lookback_weeks"],
                crowned["recent_20w_hit_rate_pct"],
                crowned["recent_20w_hits"],
                args.weeks,
                crowned["recent_20w_penalty_loss"],
                dur,
            )
        else:
            logger.warning("[%d/%d] %s -> FAILED / Insufficient data", idx, len(symbols), sym)

    # Save to config
    CONFIG_PATH.parent.mkdir(parents=True, exist_ok=True)
    yaml_payload = {
        "_metadata": {
            "description": "Crowned Week Start (Monday Day Close) models and training lookback horizons selected via 3-Criteria Tournament Arena",
            "calibration_date": datetime.now(timezone.utc).strftime("%Y-%m-%d"),
            "eval_window": f"{args.weeks}w_walk_forward_multi_horizon",
            "selection_metric": "3_criteria_tournament_loss",
        },
        "symbols": results,
    }

    with open(CONFIG_PATH, "w", encoding="utf-8") as f:
        yaml.dump(yaml_payload, f, sort_keys=False, indent=2)

    logger.info("Saved crowned models to %s. Synchronizing Gold layer...", CONFIG_PATH)
    sync_res = sync_week_start_forecasts_from_yaml(db, CONFIG_PATH)
    logger.info("Gold layer sync complete: %s (total elapsed: %.1fs)", sync_res, time.time() - t0)


if __name__ == "__main__":
    main()
