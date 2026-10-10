#!/usr/bin/env python3
"""Multi-Horizon Tournament Arena & Champion Referee.

Evaluates candidate models (LightGBM, XGBoost, Ridge, BayesianRidge, Huber)
across dynamic training lookbacks (3M, 6M, 12M) for each of the 5 forward prediction windows:
    - 3 Days (N=3)
    - 5 Days (N=5)
    - 10 Days (N=10)
    - 15 Days (N=15)
    - 30 Days (N=30)

Crowning is determined strictly by the 3-Criteria Composite Loss:
    Loss = (100.0 - hit_rate_pct) + 1.0 * hit_mae_pct + 2.5 * miss_mae_pct

Usage:
    .venv/bin/python scripts/run_multi_horizon_tournament.py --symbols THYAO,GARAN,ASELS
    .venv/bin/python scripts/run_multi_horizon_tournament.py --all-bist30
"""

from __future__ import annotations

import os

# Prevent OpenMP / BLAS thread contention and deadlocks on macOS
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"

import argparse
from datetime import datetime
from typing import Any

import yaml

from mdk_trading_oracle.core.db import PostgresManager
from mdk_trading_oracle.core.logger import get_logger
from mdk_trading_oracle.data.gold.tertip_multi_horizon import (
    CONFIG_PATH,
    update_gold_tertip_multi_horizon_forecasts,
)
from mdk_trading_oracle.data.silver.tertip_ml_forecaster import extract_3pillar_time_series
from mdk_trading_oracle.data.silver.tertip_multi_horizon_forecaster import (
    HORIZON_MAP,
    evaluate_horizon_walk_forward,
)

logger = get_logger("mdk_oracle.tournament.multi_horizon")

CANDIDATE_MODELS = ["LightGBM", "XGBoost", "BayesianRidge", "Ridge", "Huber"]
LOOKBACKS = [
    ("3m", 63),
    ("6m", 126),
    ("12m", 252),
    ("18m", 378),
    ("24m", 504),
]
HORIZONS = ["3d", "5d", "10d", "15d", "30d"]



def run_tournament_for_symbol(
    db: PostgresManager,
    symbol: str,
    eval_sessions: int = 30,
) -> dict[str, Any]:
    """Run full tournament across 5 horizons, benchmarking 5 models x 3 lookbacks = 15 configs per horizon."""
    sym = symbol.upper()
    logger.info(f"🏆 Running Multi-Horizon Tournament Arena for {sym}...")

    df = extract_3pillar_time_series(db, sym, lookback_days=550)
    if df.empty or len(df) < 80:
        logger.warning(f"Insufficient history for {sym}. Skipping tournament.")
        return {}

    symbol_champions = {}

    for h_key in HORIZONS:
        logger.info(f"  ── Benchmarking Horizon {h_key} ({HORIZON_MAP[h_key]} Days)...")
        best_loss = 999999.0
        best_res: dict[str, Any] = {}

        for model_name in CANDIDATE_MODELS:
            for lb_str, lb_sessions in LOOKBACKS:
                res = evaluate_horizon_walk_forward(
                    df=df,
                    horizon_key=h_key,
                    model_name=model_name,
                    train_lookback_sessions=lb_sessions,
                    n_eval_sessions=eval_sessions,
                )

                loss = res.get("loss", 999.0)
                hit_rate = res.get("hit_rate_pct", 0.0)

                # Prioritize lower composite loss; tie-break on higher hit rate
                if (loss < best_loss) or (abs(loss - best_loss) < 0.05 and hit_rate > best_res.get("hit_rate_pct", 0.0)):
                    best_loss = loss
                    best_res = {
                        "model": model_name,
                        "horizon": lb_str,
                        "training_lookback_sessions": lb_sessions,
                        "hit_rate_pct": hit_rate,
                        "mae_pct": res.get("mae_pct", 0.0),
                        "hit_mae_pct": res.get("hit_mae_pct", 0.0),
                        "miss_mae_pct": res.get("miss_mae_pct", 0.0),
                        "loss": loss,
                        "hits": res.get("hits", 0),
                        "total_evals": res.get("total_evals", 0),
                    }

        logger.info(
            f"     👑 Crowned [{h_key}]: {best_res['model']} ({best_res['horizon']}) "
            f"| Win: {best_res['hit_rate_pct']}% ({best_res['hits']}/{best_res['total_evals']}) "
            f"| Loss: {best_res['loss']:.2f}"
        )
        symbol_champions[h_key] = best_res

    return symbol_champions


def _worker_multi_horizon(args_tuple: tuple[str, int]) -> tuple[str, dict[str, Any]]:
    sym, eval_sessions = args_tuple
    db = PostgresManager()
    try:
        champs = run_tournament_for_symbol(db, sym, eval_sessions=eval_sessions)
        return sym, champs
    finally:
        db.close()


def main() -> None:
    parser = argparse.ArgumentParser(description="Multi-Horizon Tournament Referee")
    parser.add_argument("--symbols", type=str, default="", help="Comma-separated symbols")
    parser.add_argument("--all-bist30", action="store_true", help="Run tournament for all BIST 30 symbols")
    parser.add_argument("--eval-sessions", type=int, default=180, help="Evaluation sessions (e.g. 180 or 30)")
    parser.add_argument("--sync-gold", action="store_true", default=True, help="Sync forecasts to Gold layer")
    args = parser.parse_args()

    db = PostgresManager()

    if args.all_bist30:
        rows = db.query_pl("SELECT symbol FROM bronze_bist30_membership WHERE is_active = true ORDER BY symbol ASC").to_pandas()
        target_symbols = rows["symbol"].tolist()
    elif args.symbols:
        target_symbols = [s.strip().upper() for s in args.symbols.split(",") if s.strip()]
    else:
        target_symbols = ["THYAO", "ASELS", "GARAN"]

    # Load existing config or init
    if CONFIG_PATH.exists():
        with open(CONFIG_PATH, "r", encoding="utf-8") as f:
            cfg = yaml.safe_load(f) or {}
    else:
        cfg = {"_metadata": {}, "symbols": {}}

    if "symbols" not in cfg:
        cfg["symbols"] = {}

    import concurrent.futures

    logger.info(
        f"🚀 Running Multi-Horizon Tournament for {len(target_symbols)} equities "
        f"({args.eval_sessions} sessions) across 4 parallel workers..."
    )

    with concurrent.futures.ProcessPoolExecutor(max_workers=4) as executor:
        futures = {executor.submit(_worker_multi_horizon, (sym, args.eval_sessions)): sym for sym in target_symbols}
        for future in concurrent.futures.as_completed(futures):
            sym = futures[future]
            try:
                sym_res, champs = future.result()
                if champs:
                    cfg["symbols"][sym_res] = champs
                    logger.info(f"✅ Completed Multi-Horizon Tournament for {sym_res}")
            except Exception as e:
                logger.error(f"❌ Error during tournament for {sym}: {e}")

    cfg["_metadata"]["updated_at"] = datetime.now().isoformat()
    with open(CONFIG_PATH, "w", encoding="utf-8") as f:
        yaml.dump(cfg, f, default_flow_style=False, sort_keys=False)
    logger.info(f"💾 Updated crowned configurations in {CONFIG_PATH}")

    if args.sync_gold:
        logger.info("⚡ Synchronizing gold_tertip_multi_horizon_forecasts and backtests...")
        update_gold_tertip_multi_horizon_forecasts(db, symbols=target_symbols)
        logger.info("✅ Gold layer synchronization complete.")


if __name__ == "__main__":
    main()
