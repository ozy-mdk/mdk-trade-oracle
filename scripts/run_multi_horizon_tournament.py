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

FAMILIES = {
    "GBDT": ["LightGBM", "XGBoost"],
    "Regularized Linear": ["Ridge", "BayesianRidge"],
    "Robust M-Estimator": ["Huber"],
}

HORIZON_WEIGHTS = {
    "10d": 0.35,
    "5d": 0.25,
    "3d": 0.20,
    "15d": 0.12,
    "30d": 0.08,
}

LOOKBACKS = [
    ("3m", 63),
    ("6m", 126),
    ("12m", 252),
    ("18m", 378),
    ("24m", 504),
]
HORIZONS = ["3d", "5d", "10d", "15d", "30d"]


def compute_option1_recency_score(ledger: list[dict[str, Any]]) -> tuple[float, float, float, float]:
    """Compute Option 1 tiered recency score: 0.50*b1(0-30d) + 0.30*b2(30-60d) + 0.20*b3(60-90d)."""
    n = len(ledger)
    if n == 0:
        return 0.0, 0.0, 0.0, 0.0

    b1 = ledger[-30:] if n >= 30 else ledger
    h1 = sum(1 for r in b1 if r.get("is_hit"))
    r1 = (h1 / len(b1) * 100.0) if b1 else 0.0

    b2 = ledger[-60:-30] if n >= 60 else (ledger[:-30] if n > 30 else [])
    h2 = sum(1 for r in b2 if r.get("is_hit"))
    r2 = (h2 / len(b2) * 100.0) if b2 else r1

    b3 = ledger[-90:-60] if n >= 90 else (ledger[:-60] if n > 60 else [])
    h3 = sum(1 for r in b3 if r.get("is_hit"))
    r3 = (h3 / len(b3) * 100.0) if b3 else r2

    weighted_score = 0.50 * r1 + 0.30 * r2 + 0.20 * r3
    return weighted_score, r1, r2, r3


def run_tournament_for_symbol(
    db: PostgresManager,
    symbol: str,
    eval_sessions: int = 90,
) -> dict[str, Any]:
    """Run Two-Stage Family-First Tournament with Option 1 Tiered Recency Weighting across all 5 horizons."""
    from mdk_trading_oracle.data.silver.tertip_multi_horizon_forecaster import (
        compute_forward_terminal_return,
        engineer_multi_horizon_features,
    )

    sym = symbol.upper()
    logger.info(f"🏆 Running Multi-Horizon Tournament Arena for {sym}...")

    df_raw = extract_3pillar_time_series(db, sym, lookback_days=1100)
    if df_raw.empty or len(df_raw) < 80:
        logger.warning(f"Insufficient history for {sym}. Skipping tournament.")
        return {}

    df_feat, _ = engineer_multi_horizon_features(df_raw)

    # Pre-compute forward targets on df_feat
    for h in HORIZONS:
        n_days = HORIZON_MAP[h]
        p_term, r_term = compute_forward_terminal_return(df_feat["close_price"], n_days)
        df_feat[f"fwd_avg_price_{h}"] = p_term
        df_feat[f"fwd_avg_ret_{h}"] = r_term

    swing_evals: dict[str, list[dict[str, Any]]] = {h: [] for h in HORIZONS}

    for h_key in HORIZONS:
        for fam_name, models in FAMILIES.items():
            for model_name in models:
                for lb_str, lb_sessions in LOOKBACKS:
                    res = evaluate_horizon_walk_forward(
                        df=df_feat,
                        horizon_key=h_key,
                        model_name=model_name,
                        train_lookback_sessions=lb_sessions,
                        n_eval_sessions=eval_sessions,
                    )

                    ledger = res.get("ledger", [])
                    opt1_score, r1, r2, r3 = compute_option1_recency_score(ledger)

                    swing_evals[h_key].append({
                        "family": fam_name,
                        "model": model_name,
                        "horizon": lb_str,
                        "training_lookback_sessions": lb_sessions,
                        "opt1_score": opt1_score,
                        "r_0_30": r1,
                        "r_30_60": r2,
                        "r_60_90": r3,
                        "hit_rate_pct": res.get("hit_rate_pct", 0.0),
                        "mae_pct": res.get("mae_pct", 0.0),
                        "hit_mae_pct": res.get("hit_mae_pct", 0.0),
                        "miss_mae_pct": res.get("miss_mae_pct", 0.0),
                        "loss": res.get("loss", 999.0),
                        "hits": res.get("hits", 0),
                        "total_evals": res.get("total_evals", 0),
                    })

    # =========================================================================
    # STAGE 1: CROWN MODEL FAMILY USING OPTION 1 RECENCY SCORES
    # Family Score = sum(horizon_weight * max_opt1_score_in_family)
    # Priority: 10d (35%), 5d (25%), 3d (20%), 15d (12%), 30d (8%)
    # =========================================================================
    family_scores: dict[str, float] = {}
    for fam_name in FAMILIES.keys():
        fam_h_scores = {}
        for h in HORIZONS:
            fam_evals = [e for e in swing_evals[h] if e["family"] == fam_name]
            best_e = max(fam_evals, key=lambda x: (x["opt1_score"], x["r_0_30"], -x["mae_pct"]))
            fam_h_scores[h] = best_e["opt1_score"]

        score = sum(HORIZON_WEIGHTS[h] * fam_h_scores[h] for h in HORIZONS)
        family_scores[fam_name] = score

    crowned_family = max(family_scores, key=lambda f: family_scores[f])
    logger.info(
        f"👑 {sym} Crowned Multi-Horizon Family: {crowned_family} "
        f"({family_scores[crowned_family]:.1f}%) "
        f"[Scores: " + " | ".join(f"{f}: {s:.1f}%" for f, s in family_scores.items()) + "]"
    )

    # =========================================================================
    # STAGE 2: CROWN HORIZON SPECIALISTS STRICTLY WITHIN CROWNED FAMILY
    # =========================================================================
    symbol_champions = {}
    for h_key in HORIZONS:
        crowned_fam_evals = [e for e in swing_evals[h_key] if e["family"] == crowned_family]
        best_spec = max(crowned_fam_evals, key=lambda x: (x["opt1_score"], x["r_0_30"], -x["mae_pct"]))
        symbol_champions[h_key] = {
            "model": best_spec["model"],
            "horizon": best_spec["horizon"],
            "training_lookback_sessions": best_spec["training_lookback_sessions"],
            "hit_rate_pct": best_spec["hit_rate_pct"],
            "opt1_score": round(best_spec["opt1_score"], 2),
            "bucket_1_hit_rate": round(best_spec["r_0_30"], 1),
            "bucket_2_hit_rate": round(best_spec["r_30_60"], 1),
            "bucket_3_hit_rate": round(best_spec["r_60_90"], 1),
            "mae_pct": best_spec["mae_pct"],
            "hit_mae_pct": best_spec["hit_mae_pct"],
            "miss_mae_pct": best_spec["miss_mae_pct"],
            "loss": best_spec["loss"],
            "hits": best_spec["hits"],
            "total_evals": best_spec["total_evals"],
            "crowned_family": crowned_family,
        }
        logger.info(
            f"   👑 Crowned [{h_key}]: {best_spec['model']} ({best_spec['horizon']}) "
            f"| Opt1 Score: {best_spec['opt1_score']:.1f}% "
            f"[0-30d: {best_spec['r_0_30']:.1f}% | 30-60d: {best_spec['r_30_60']:.1f}% | 60-90d: {best_spec['r_60_90']:.1f}%] "
            f"| Win: {best_spec['hit_rate_pct']}% ({best_spec['hits']}/{best_spec['total_evals']})"
        )

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
    parser.add_argument("--eval-sessions", type=int, default=90, help="Evaluation sessions (e.g. 90 or 180)")
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
    cfg["_metadata"]["selection_metric"] = "two_stage_family_first_option1_recency_score"
    cfg["_metadata"]["recency_weights"] = {"0-30d": 0.50, "30-60d": 0.30, "60-90d": 0.20}
    cfg["_metadata"]["family_horizon_weights"] = {"10d": 0.35, "5d": 0.25, "3d": 0.20, "15d": 0.12, "30d": 0.08}
    with open(CONFIG_PATH, "w", encoding="utf-8") as f:
        yaml.dump(cfg, f, default_flow_style=False, sort_keys=False)
    logger.info(f"💾 Updated crowned configurations in {CONFIG_PATH}")

    if args.sync_gold:
        logger.info("⚡ Synchronizing gold_tertip_multi_horizon_forecasts and backtests...")
        update_gold_tertip_multi_horizon_forecasts(db, symbols=target_symbols)
        logger.info("✅ Gold layer synchronization complete.")


if __name__ == "__main__":
    main()
