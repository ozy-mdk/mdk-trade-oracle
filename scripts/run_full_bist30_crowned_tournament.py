"""Automated BIST 30 Full-Constituent Walk-Forward Model Tournament.

Evaluates all 30 BIST 30 equities across 5 model paradigms (XGBoost, LightGBM,
BayesianRidge, Huber, Ridge) and 5 dynamic lookback horizons (3M=63, 6M=126, 12M=252, 18M=378, 24M=504)
over retrospective evaluation sessions (default 60 sessions).

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
- Updates PostgreSQL gold_tertip_daily_forecasts & gold_tertip_walk_forward_backtests
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
HORIZONS = {"3m": 63, "6m": 126, "12m": 252, "18m": 378, "24m": 504}
CANDIDATE_MODELS = ["XGBoost", "LightGBM", "BayesianRidge", "Huber", "Ridge"]


def save_ledger_to_database(
    db: PostgresManager,
    symbol: str,
    ledger: list[dict[str, Any]],
    ml_champion: str,
    train_lb: int,
) -> None:
    """Upsert 60-day retrospective walk-forward ledger records into gold_tertip_walk_forward_backtests."""
    if not ledger:
        return

    insert_query = """
    INSERT INTO gold_tertip_walk_forward_backtests (
        symbol, trade_date, actual_price, actual_return_pct, bist30_ret_pct,
        confluence_pred_price, confluence_pred_return_pct, confluence_direction, confluence_err_pct, confluence_is_hit,
        ml_pred_price, ml_pred_return_pct, ml_direction, ml_err_pct, ml_is_hit,
        prophet_pred_price, prophet_pred_return_pct, prophet_direction, prophet_err_pct, prophet_is_hit,
        winner, is_shock_day, shock_type, days_since_pos_shock, days_since_neg_shock,
        mlb_action, mlb_flow_tl, mlb_buy_tl, mlb_sell_tl, mlb_pnl_tl,
        big5_action, big5_flow_tl, big5_buy_tl, big5_sell_tl, big5_pnl_tl,
        kamu_action, kamu_flow_tl, kamu_buy_tl, kamu_sell_tl, kamu_pnl_tl,
        ml_champion_type, training_lookback_sessions, actual_training_sessions, data_sufficiency_status, calculated_at
    ) VALUES (
        %(symbol)s, %(trade_date)s, %(actual_price)s, %(actual_return_pct)s, %(bist30_ret_pct)s,
        %(confluence_pred_price)s, %(confluence_pred_return_pct)s, %(confluence_direction)s, %(confluence_err_pct)s, %(confluence_is_hit)s,
        %(ml_pred_price)s, %(ml_pred_return_pct)s, %(ml_direction)s, %(ml_err_pct)s, %(ml_is_hit)s,
        %(prophet_pred_price)s, %(prophet_pred_return_pct)s, %(prophet_direction)s, %(prophet_err_pct)s, %(prophet_is_hit)s,
        %(winner)s, %(is_shock_day)s, %(shock_type)s, %(days_since_pos_shock)s, %(days_since_neg_shock)s,
        %(mlb_action)s, %(mlb_flow_tl)s, %(mlb_buy_tl)s, %(mlb_sell_tl)s, %(mlb_pnl_tl)s,
        %(big5_action)s, %(big5_flow_tl)s, %(big5_buy_tl)s, %(big5_sell_tl)s, %(big5_pnl_tl)s,
        %(kamu_action)s, %(kamu_flow_tl)s, %(kamu_buy_tl)s, %(kamu_sell_tl)s, %(kamu_pnl_tl)s,
        %(ml_champion_type)s, %(training_lookback_sessions)s, %(actual_training_sessions)s, %(data_sufficiency_status)s, CURRENT_TIMESTAMP
    ) ON CONFLICT (symbol, trade_date) DO UPDATE SET
        actual_price = EXCLUDED.actual_price,
        actual_return_pct = EXCLUDED.actual_return_pct,
        bist30_ret_pct = EXCLUDED.bist30_ret_pct,
        confluence_pred_price = EXCLUDED.confluence_pred_price,
        confluence_pred_return_pct = EXCLUDED.confluence_pred_return_pct,
        confluence_direction = EXCLUDED.confluence_direction,
        confluence_err_pct = EXCLUDED.confluence_err_pct,
        confluence_is_hit = EXCLUDED.confluence_is_hit,
        ml_pred_price = EXCLUDED.ml_pred_price,
        ml_pred_return_pct = EXCLUDED.ml_pred_return_pct,
        ml_direction = EXCLUDED.ml_direction,
        ml_err_pct = EXCLUDED.ml_err_pct,
        ml_is_hit = EXCLUDED.ml_is_hit,
        prophet_pred_price = EXCLUDED.prophet_pred_price,
        prophet_pred_return_pct = EXCLUDED.prophet_pred_return_pct,
        prophet_direction = EXCLUDED.prophet_direction,
        prophet_err_pct = EXCLUDED.prophet_err_pct,
        prophet_is_hit = EXCLUDED.prophet_is_hit,
        winner = EXCLUDED.winner,
        is_shock_day = EXCLUDED.is_shock_day,
        shock_type = EXCLUDED.shock_type,
        days_since_pos_shock = EXCLUDED.days_since_pos_shock,
        days_since_neg_shock = EXCLUDED.days_since_neg_shock,
        mlb_action = EXCLUDED.mlb_action,
        mlb_flow_tl = EXCLUDED.mlb_flow_tl,
        mlb_buy_tl = EXCLUDED.mlb_buy_tl,
        mlb_sell_tl = EXCLUDED.mlb_sell_tl,
        mlb_pnl_tl = EXCLUDED.mlb_pnl_tl,
        big5_action = EXCLUDED.big5_action,
        big5_flow_tl = EXCLUDED.big5_flow_tl,
        big5_buy_tl = EXCLUDED.big5_buy_tl,
        big5_sell_tl = EXCLUDED.big5_sell_tl,
        big5_pnl_tl = EXCLUDED.big5_pnl_tl,
        kamu_action = EXCLUDED.kamu_action,
        kamu_flow_tl = EXCLUDED.kamu_flow_tl,
        kamu_buy_tl = EXCLUDED.kamu_buy_tl,
        kamu_sell_tl = EXCLUDED.kamu_sell_tl,
        kamu_pnl_tl = EXCLUDED.kamu_pnl_tl,
        ml_champion_type = EXCLUDED.ml_champion_type,
        training_lookback_sessions = EXCLUDED.training_lookback_sessions,
        actual_training_sessions = EXCLUDED.actual_training_sessions,
        data_sufficiency_status = EXCLUDED.data_sufficiency_status,
        calculated_at = CURRENT_TIMESTAMP;
    """
    for row in ledger:
        db.execute(
            insert_query,
            {
                "symbol": symbol,
                "trade_date": row["date"],
                "actual_price": row["actual_price"],
                "actual_return_pct": row["actual_return_pct"],
                "bist30_ret_pct": row.get("bist30_ret_pct", 0.0),
                "confluence_pred_price": row.get("confluence_pred_price", row.get("champion_pred_price")),
                "confluence_pred_return_pct": row.get("confluence_pred_return_pct", row.get("champion_pred_return_pct")),
                "confluence_direction": row.get("confluence_direction", row.get("champion_direction")),
                "confluence_err_pct": row.get("confluence_err_pct", row.get("champion_err_pct")),
                "confluence_is_hit": row.get("confluence_is_hit", row.get("champion_is_hit")),
                "ml_pred_price": row["ml_pred_price"],
                "ml_pred_return_pct": row["ml_pred_return_pct"],
                "ml_direction": row["ml_direction"],
                "ml_err_pct": row["ml_err_pct"],
                "ml_is_hit": row["ml_is_hit"],
                "prophet_pred_price": row["prophet_pred_price"],
                "prophet_pred_return_pct": row["prophet_pred_return_pct"],
                "prophet_direction": row["prophet_direction"],
                "prophet_err_pct": row["prophet_err_pct"],
                "prophet_is_hit": row["prophet_is_hit"],
                "winner": row["winner"],
                "is_shock_day": row["is_shock_day"],
                "shock_type": row["shock_type"],
                "days_since_pos_shock": row.get("days_since_pos_shock", 63),
                "days_since_neg_shock": row.get("days_since_neg_shock", 63),
                "mlb_action": row["mlb_action"],
                "mlb_flow_tl": row["mlb_flow_tl"],
                "mlb_buy_tl": row.get("mlb_buy_tl", 0.0),
                "mlb_sell_tl": row.get("mlb_sell_tl", 0.0),
                "mlb_pnl_tl": row.get("mlb_pnl_tl", 0.0),
                "big5_action": row["big5_action"],
                "big5_flow_tl": row["big5_flow_tl"],
                "big5_buy_tl": row.get("big5_buy_tl", 0.0),
                "big5_sell_tl": row.get("big5_sell_tl", 0.0),
                "big5_pnl_tl": row.get("big5_pnl_tl", 0.0),
                "kamu_action": row["kamu_action"],
                "kamu_flow_tl": row["kamu_flow_tl"],
                "kamu_buy_tl": row.get("kamu_buy_tl", 0.0),
                "kamu_sell_tl": row.get("kamu_sell_tl", 0.0),
                "kamu_pnl_tl": row.get("kamu_pnl_tl", 0.0),
                "ml_champion_type": ml_champion,
                "training_lookback_sessions": row.get("training_lookback_sessions", train_lb),
                "actual_training_sessions": row.get("actual_training_sessions", train_lb),
                "data_sufficiency_status": row.get("data_sufficiency_status", "FULL"),
            },
        )


def run_tournament_for_symbol(
    db: PostgresManager,
    symbol: str,
    n_sessions: int = 60,
) -> dict[str, Any] | None:
    """Run walk-forward arena across all candidate models and horizons for a single equity."""
    sym = symbol.upper()
    df = extract_3pillar_time_series(db, sym, lookback_days=1000)
    if df.empty or len(df) < 63:
        logger.warning(f"Skipping {sym}: Insufficient historical data ({len(df)} rows < 63 sessions / 3 months minimum)")
        return None

    df = attach_prophet_rolling_features(df, sym, n_history_needed=n_sessions + 40)

    best_config: dict[str, Any] | None = None
    best_key: tuple[float, float, float, float] | None = None
    best_ledger: list[dict[str, Any]] = []

    for h_name, h_lb in HORIZONS.items():
        ledger, summary, _ = run_30d_walk_forward_arena(
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
                best_ledger = ledger

    if best_config:
        best_config["ledger"] = best_ledger
    return best_config


def main() -> None:
    parser = argparse.ArgumentParser(description="Run Full BIST 30 Tournament")
    parser.add_argument("--symbols", nargs="*", help="Optional subset of symbols to evaluate")
    parser.add_argument("--sessions", type=int, default=60, help="Retrospective evaluation sessions (default: 60)")
    parser.add_argument("--dry-run", action="store_true", help="Print scorecard without writing to disk or DB")
    args = parser.parse_args()

    db = PostgresManager()

    # Load existing configuration to compare deltas
    old_config = {}
    if CONFIG_PATH.exists():
        with open(CONFIG_PATH, "r", encoding="utf-8") as f:
            old_config = yaml.safe_load(f).get("symbols", {})

    # Determine symbols universe: strictly query active BIST 30 membership
    if args.symbols:
        eval_symbols = [s.upper() for s in args.symbols]
    else:
        res = db.query_pl("SELECT symbol FROM bronze_bist30_membership WHERE is_active = true ORDER BY symbol ASC;")
        eval_symbols = res["symbol"].to_list() if not res.is_empty() else sorted(list(old_config.keys()))

    logger.info(f"Starting Full BIST 30 Tournament for {len(eval_symbols)} equities over {args.sessions} retrospective sessions...")
    logger.info("Candidates: XGBoost, LightGBM, BayesianRidge, Huber, Ridge across [3M, 6M, 12M, 18M, 24M]")
    t_start = time.time()

    new_results: dict[str, Any] = {}
    scorecard_rows: list[dict[str, Any]] = []

    for idx, sym in enumerate(eval_symbols, 1):
        sym_t0 = time.time()
        print(f"\n[{idx:02d}/{len(eval_symbols):02d}] Benchmarking {sym} across 25 configurations ({args.sessions}d)...", flush=True)
        res = run_tournament_for_symbol(db, sym, n_sessions=args.sessions)
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
            f"   -> Winner: {new_model:<18} | Hits: {new_hits}/{args.sessions} ({new_rate:>4.1f}%) | "
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
    print(f"MDK TRADING ORACLE — FULL BIST 30 MODEL TOURNAMENT SCORECARD ({args.sessions}-DAY RETROSPECTIVE)")
    print("=" * 125)
    print(df_sc.to_string(index=False))
    print("=" * 125)

    total_elapsed = time.time() - t_start
    avg_new_hit = sum(r["recent_30d_hit_rate_pct"] for r in new_results.values()) / max(1, len(new_results))
    avg_new_mae = sum(r["recent_30d_mae_pct"] for r in new_results.values()) / max(1, len(new_results))
    print(f"\nTournament completed in {total_elapsed:.1f}s (Average {total_elapsed/max(1, len(eval_symbols)):.1f}s per stock)")
    print(f"Overall BIST 30 Directional Hit Rate: {avg_new_hit:.1f}% | Average MAE: {avg_new_mae:.2f}%\n")

    if args.dry_run:
        logger.info("Dry-run requested. Skipped updating YAML and PostgreSQL.")
        return

    # Update YAML configuration
    updated_config = {
        "_metadata": {
            "description": "Crowned Tertip Confluence models and training lookback horizons selected via 3-Criteria Tournament Arena",
            "calibration_date": datetime.now(timezone.utc).strftime("%Y-%m-%d"),
            "eval_window": f"{args.sessions}d_walk_forward_multi_horizon",
            "selection_metric": "3_criteria_tournament_loss",
        },
        "symbols": {sym: {k: v for k, v in meta.items() if k not in ("symbol", "ledger")} for sym, meta in new_results.items()},
    }

    with open(CONFIG_PATH, "w", encoding="utf-8") as f:
        yaml.dump(updated_config, f, sort_keys=False, indent=2)
    logger.info(f"Successfully saved new crowned champions to {CONFIG_PATH}")

    # Persist 60-day retrospective walk-forward backtests into gold_tertip_walk_forward_backtests
    logger.info(f"Persisting {args.sessions}-day retrospective walk-forward backtests to PostgreSQL...")
    saved_ledger_cnt = 0
    for sym, meta in new_results.items():
        if "ledger" in meta and meta["ledger"]:
            save_ledger_to_database(db, sym, meta["ledger"], meta["model"], meta["training_lookback_sessions"])
            saved_ledger_cnt += 1
    logger.info(f"Persisted retrospective backtest ledgers for {saved_ledger_cnt} equities.")

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
