"""Export 180-Day Multi-Horizon Walk-Forward Backtest Track Data for BIST 30 to Database & CSV.

Evaluates 180 out-of-sample trading sessions for all 30 BIST 30 equities.
Supports lookback modes:
- auto (default): multi-horizon tournament crowning the best lookback (3M=63, 6M=126, 12M=252) per stock
- 12m: production baseline 12-month structural carry lookback (252 sessions)
- 6m: semi-annual structural balance lookback (126 sessions)
- 3m: ultra-adaptive fast reaction lookback (63 sessions)

Persists records into PostgreSQL:
- gold_tertip_walk_forward_backtests
- gold_tertip_daily_forecasts

Exports spreadsheet-ready CSVs to:
- /Users/ozkanyildirim/data/mdk_oracle/export_artifacts/tertip_180d_walk_forward_backtests.csv
- /Users/ozkanyildirim/data/mdk_oracle/export_artifacts/tertip_180d_horizon_tournament_summary.csv
"""

from __future__ import annotations

import argparse
import logging
import time
from pathlib import Path
from typing import Any

import pandas as pd

from mdk_trading_oracle.core.db import PostgresManager
from mdk_trading_oracle.data.silver.tertip_ml_forecaster import get_tertip_ml_forecast

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("export_tertip_180d")
logging.getLogger("cmdstanpy").setLevel(logging.ERROR)
logging.getLogger("prophet").setLevel(logging.ERROR)
logging.getLogger("lightgbm").setLevel(logging.ERROR)

EXPORT_DIR = Path.home() / "data" / "mdk_oracle" / "export_artifacts"
EXPORT_DIR.mkdir(parents=True, exist_ok=True)


def ensure_gold_tables(db: PostgresManager) -> None:
    """Ensure gold tables exist with all necessary columns in PostgreSQL."""
    db.execute("""
    CREATE TABLE IF NOT EXISTS gold_tertip_walk_forward_backtests (
        symbol VARCHAR(16) NOT NULL,
        trade_date DATE NOT NULL,
        actual_price DOUBLE PRECISION,
        actual_return_pct DOUBLE PRECISION,
        bist30_ret_pct DOUBLE PRECISION,
        ml_pred_price DOUBLE PRECISION,
        ml_pred_return_pct DOUBLE PRECISION,
        ml_direction VARCHAR(16),
        ml_err_pct DOUBLE PRECISION,
        ml_is_hit BOOLEAN,
        prophet_pred_price DOUBLE PRECISION,
        prophet_pred_return_pct DOUBLE PRECISION,
        prophet_direction VARCHAR(16),
        prophet_err_pct DOUBLE PRECISION,
        prophet_is_hit BOOLEAN,
        winner VARCHAR(32),
        is_shock_day BOOLEAN,
        shock_type VARCHAR(32),
        days_since_pos_shock INTEGER,
        days_since_neg_shock INTEGER,
        mlb_action VARCHAR(32),
        mlb_flow_tl DOUBLE PRECISION,
        mlb_buy_tl DOUBLE PRECISION,
        mlb_sell_tl DOUBLE PRECISION,
        mlb_pnl_tl DOUBLE PRECISION,
        big5_action VARCHAR(32),
        big5_flow_tl DOUBLE PRECISION,
        big5_buy_tl DOUBLE PRECISION,
        big5_sell_tl DOUBLE PRECISION,
        big5_pnl_tl DOUBLE PRECISION,
        kamu_action VARCHAR(32),
        kamu_flow_tl DOUBLE PRECISION,
        kamu_buy_tl DOUBLE PRECISION,
        kamu_sell_tl DOUBLE PRECISION,
        kamu_pnl_tl DOUBLE PRECISION,
        ml_champion_type VARCHAR(32),
        training_lookback_sessions INTEGER DEFAULT 252,
        calculated_at TIMESTAMPTZ DEFAULT CURRENT_TIMESTAMP,
        PRIMARY KEY (symbol, trade_date)
    );

    ALTER TABLE gold_tertip_walk_forward_backtests ADD COLUMN IF NOT EXISTS training_lookback_sessions INTEGER DEFAULT 252;

    CREATE TABLE IF NOT EXISTS gold_tertip_daily_forecasts (
        symbol VARCHAR(16) NOT NULL,
        as_of_date DATE NOT NULL,
        current_price DOUBLE PRECISION,
        target_price DOUBLE PRECISION,
        expected_return_pct DOUBLE PRECISION,
        price_low DOUBLE PRECISION,
        price_high DOUBLE PRECISION,
        stance VARCHAR(32),
        conviction VARCHAR(32),
        playbook VARCHAR(64),
        ml_champion_type VARCHAR(32),
        champion_dir_hits INTEGER,
        champion_dir_hit_rate_pct DOUBLE PRECISION,
        champion_mae_pct DOUBLE PRECISION,
        prophet_target_price DOUBLE PRECISION,
        prophet_expected_return_pct DOUBLE PRECISION,
        training_lookback_sessions INTEGER DEFAULT 252,
        crowned_horizon VARCHAR(16) DEFAULT '12m',
        calculated_at TIMESTAMPTZ DEFAULT CURRENT_TIMESTAMP,
        PRIMARY KEY (symbol, as_of_date)
    );

    ALTER TABLE gold_tertip_daily_forecasts ADD COLUMN IF NOT EXISTS training_lookback_sessions INTEGER DEFAULT 252;
    ALTER TABLE gold_tertip_daily_forecasts ADD COLUMN IF NOT EXISTS crowned_horizon VARCHAR(16) DEFAULT '12m';
    """)


def get_active_bist30_symbols(db: PostgresManager) -> list[str]:
    """Retrieve all active BIST 30 constituent stock symbols."""
    res = db.query_pl("""
        SELECT symbol 
        FROM bronze_bist30_membership 
        WHERE is_active = true 
        ORDER BY symbol
    """)
    if res.is_empty():
        return [
            "AEFES", "AKBNK", "ASELS", "ASTOR", "BIMAS", "DSTKF", "EKGYO", "ENKAI",
            "EREGL", "FROTO", "GARAN", "GUBRF", "ISCTR", "KCHOL", "KRDMD", "MGROS",
            "PETKM", "PGSUS", "SAHOL", "SASA", "SISE", "TAVHL", "TCELL", "THYAO",
            "TOASO", "TRALT", "TTKOM", "TUPRS", "VAKBN", "YKBNK"
        ]
    return res["symbol"].to_list()


def save_ledger_to_database(db: PostgresManager, symbol: str, ledger: list[dict[str, Any]], ml_champion: str, train_lb: int) -> None:
    """Upsert backtest ledger records into gold_tertip_walk_forward_backtests."""
    if not ledger:
        return

    insert_query = """
    INSERT INTO gold_tertip_walk_forward_backtests (
        symbol, trade_date, actual_price, actual_return_pct, bist30_ret_pct,
        ml_pred_price, ml_pred_return_pct, ml_direction, ml_err_pct, ml_is_hit,
        prophet_pred_price, prophet_pred_return_pct, prophet_direction, prophet_err_pct, prophet_is_hit,
        winner, is_shock_day, shock_type, days_since_pos_shock, days_since_neg_shock,
        mlb_action, mlb_flow_tl, mlb_buy_tl, mlb_sell_tl, mlb_pnl_tl,
        big5_action, big5_flow_tl, big5_buy_tl, big5_sell_tl, big5_pnl_tl,
        kamu_action, kamu_flow_tl, kamu_buy_tl, kamu_sell_tl, kamu_pnl_tl,
        ml_champion_type, training_lookback_sessions, calculated_at
    ) VALUES (
        %(symbol)s, %(trade_date)s, %(actual_price)s, %(actual_return_pct)s, %(bist30_ret_pct)s,
        %(ml_pred_price)s, %(ml_pred_return_pct)s, %(ml_direction)s, %(ml_err_pct)s, %(ml_is_hit)s,
        %(prophet_pred_price)s, %(prophet_pred_return_pct)s, %(prophet_direction)s, %(prophet_err_pct)s, %(prophet_is_hit)s,
        %(winner)s, %(is_shock_day)s, %(shock_type)s, %(days_since_pos_shock)s, %(days_since_neg_shock)s,
        %(mlb_action)s, %(mlb_flow_tl)s, %(mlb_buy_tl)s, %(mlb_sell_tl)s, %(mlb_pnl_tl)s,
        %(big5_action)s, %(big5_flow_tl)s, %(big5_buy_tl)s, %(big5_sell_tl)s, %(big5_pnl_tl)s,
        %(kamu_action)s, %(kamu_flow_tl)s, %(kamu_buy_tl)s, %(kamu_sell_tl)s, %(kamu_pnl_tl)s,
        %(ml_champion_type)s, %(training_lookback_sessions)s, CURRENT_TIMESTAMP
    ) ON CONFLICT (symbol, trade_date) DO UPDATE SET
        actual_price = EXCLUDED.actual_price,
        actual_return_pct = EXCLUDED.actual_return_pct,
        bist30_ret_pct = EXCLUDED.bist30_ret_pct,
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
        calculated_at = CURRENT_TIMESTAMP;
    """

    for r in ledger:
        p = {
            "symbol": symbol,
            "trade_date": r["date"],
            "actual_price": r.get("actual_price"),
            "actual_return_pct": r.get("actual_return_pct"),
            "bist30_ret_pct": r.get("bist30_ret_pct", 0.0),
            "ml_pred_price": r.get("ml_pred_price"),
            "ml_pred_return_pct": r.get("ml_pred_return_pct"),
            "ml_direction": r.get("ml_direction"),
            "ml_err_pct": r.get("ml_err_pct"),
            "ml_is_hit": r.get("ml_is_hit"),
            "prophet_pred_price": r.get("prophet_pred_price"),
            "prophet_pred_return_pct": r.get("prophet_pred_return_pct"),
            "prophet_direction": r.get("prophet_direction"),
            "prophet_err_pct": r.get("prophet_err_pct"),
            "prophet_is_hit": r.get("prophet_is_hit"),
            "winner": r.get("winner"),
            "is_shock_day": r.get("is_shock_day", False),
            "shock_type": r.get("shock_type", "NONE"),
            "days_since_pos_shock": r.get("days_since_pos_shock", 63),
            "days_since_neg_shock": r.get("days_since_neg_shock", 63),
            "mlb_action": r.get("mlb_action"),
            "mlb_flow_tl": r.get("mlb_flow_tl"),
            "mlb_buy_tl": r.get("mlb_buy_tl"),
            "mlb_sell_tl": r.get("mlb_sell_tl"),
            "mlb_pnl_tl": r.get("mlb_pnl_tl"),
            "big5_action": r.get("big5_action"),
            "big5_flow_tl": r.get("big5_flow_tl"),
            "big5_buy_tl": r.get("big5_buy_tl"),
            "big5_sell_tl": r.get("big5_sell_tl"),
            "big5_pnl_tl": r.get("big5_pnl_tl"),
            "kamu_action": r.get("kamu_action"),
            "kamu_flow_tl": r.get("kamu_flow_tl"),
            "kamu_buy_tl": r.get("kamu_buy_tl"),
            "kamu_sell_tl": r.get("kamu_sell_tl"),
            "kamu_pnl_tl": r.get("kamu_pnl_tl"),
            "ml_champion_type": ml_champion,
            "training_lookback_sessions": r.get("training_lookback_sessions", train_lb),
        }
        db.execute(insert_query, p)


def save_forecast_to_database(db: PostgresManager, symbol: str, fc: dict[str, Any]) -> None:
    """Upsert live T+1 forecast into gold_tertip_daily_forecasts."""
    t = fc.get("tournament_summary", {})
    insert_query = """
    INSERT INTO gold_tertip_daily_forecasts (
        symbol, as_of_date, current_price, target_price, expected_return_pct,
        price_low, price_high, stance, conviction, playbook,
        ml_champion_type, champion_dir_hits, champion_dir_hit_rate_pct, champion_mae_pct,
        prophet_target_price, prophet_expected_return_pct,
        training_lookback_sessions, crowned_horizon, calculated_at
    ) VALUES (
        %(symbol)s, %(as_of_date)s, %(current_price)s, %(target_price)s, %(expected_return_pct)s,
        %(price_low)s, %(price_high)s, %(stance)s, %(conviction)s, %(playbook)s,
        %(ml_champion_type)s, %(champion_dir_hits)s, %(champion_dir_hit_rate_pct)s, %(champion_mae_pct)s,
        %(prophet_target_price)s, %(prophet_expected_return_pct)s,
        %(training_lookback_sessions)s, %(crowned_horizon)s, CURRENT_TIMESTAMP
    ) ON CONFLICT (symbol, as_of_date) DO UPDATE SET
        current_price = EXCLUDED.current_price,
        target_price = EXCLUDED.target_price,
        expected_return_pct = EXCLUDED.expected_return_pct,
        price_low = EXCLUDED.price_low,
        price_high = EXCLUDED.price_high,
        stance = EXCLUDED.stance,
        conviction = EXCLUDED.conviction,
        playbook = EXCLUDED.playbook,
        ml_champion_type = EXCLUDED.ml_champion_type,
        champion_dir_hits = EXCLUDED.champion_dir_hits,
        champion_dir_hit_rate_pct = EXCLUDED.champion_dir_hit_rate_pct,
        champion_mae_pct = EXCLUDED.champion_mae_pct,
        prophet_target_price = EXCLUDED.prophet_target_price,
        prophet_expected_return_pct = EXCLUDED.prophet_expected_return_pct,
        training_lookback_sessions = EXCLUDED.training_lookback_sessions,
        crowned_horizon = EXCLUDED.crowned_horizon,
        calculated_at = CURRENT_TIMESTAMP;
    """
    latest_price = fc.get("latest_close_price") or fc.get("current_price")
    playbook_val = fc.get("stance_badge") or fc.get("playbook")
    conviction_val = fc.get("stance")

    p = {
        "symbol": symbol,
        "as_of_date": fc.get("as_of_date"),
        "current_price": latest_price,
        "target_price": fc.get("target_price"),
        "expected_return_pct": fc.get("expected_return_pct"),
        "price_low": fc.get("price_low"),
        "price_high": fc.get("price_high"),
        "stance": fc.get("stance"),
        "conviction": conviction_val,
        "playbook": playbook_val,
        "ml_champion_type": t.get("ml_champion_type"),
        "champion_dir_hits": t.get("champion_dir_hits"),
        "champion_dir_hit_rate_pct": t.get("champion_dir_hit_rate_pct"),
        "champion_mae_pct": t.get("champion_mae_pct"),
        "prophet_target_price": fc.get("prophet_target_price"),
        "prophet_expected_return_pct": fc.get("prophet_expected_return_pct"),
        "training_lookback_sessions": t.get("training_lookback_sessions", 252),
        "crowned_horizon": t.get("crowned_horizon", "12m"),
    }
    db.execute(insert_query, p)


def main() -> None:
    parser = argparse.ArgumentParser(description="Export 180-Day Multi-Horizon Backtest Data")
    parser.add_argument("--eval-sessions", type=int, default=180, help="Out-of-sample evaluation sessions (default 180)")
    parser.add_argument("--lookback-mode", type=str, default="auto", choices=["auto", "12m", "6m", "3m"], help="Lookback horizon mode")
    parser.add_argument("--symbols", type=str, default="", help="Comma-separated symbols (default: all 30 BIST 30)")
    args = parser.parse_args()

    t_start = time.time()
    db = PostgresManager()

    logger.info("Ensuring Gold database tables exist...")
    ensure_gold_tables(db)

    if args.symbols:
        symbols = [s.strip().upper() for s in args.symbols.split(",") if s.strip()]
    else:
        symbols = get_active_bist30_symbols(db)

    n_eval = args.eval_sessions
    lb_mode = args.lookback_mode

    logger.info(f"Targeting {len(symbols)} symbols over {n_eval} sessions with lookback_mode='{lb_mode}'")

    all_ledger_rows: list[dict[str, Any]] = []
    summary_rows: list[dict[str, Any]] = []

    success_count = 0
    fail_count = 0

    for idx, sym in enumerate(symbols, 1):
        sym_t0 = time.time()
        logger.info(f"[{idx}/{len(symbols)}] Processing {sym}...")

        try:
            fc = get_tertip_ml_forecast(
                db,
                sym,
                force_refresh=True,
                n_eval_sessions=n_eval,
                lookback_mode=lb_mode,
            )
            if "error" in fc:
                logger.warning(f"Skipping {sym}: {fc['error']}")
                fail_count += 1
                continue

            ledger = fc.get("walk_forward_ledger", [])
            tournament = fc.get("tournament_summary", {})
            ml_champ = tournament.get("ml_champion_type", "XGBoost")
            crowned_h = tournament.get("crowned_horizon", "12m")
            train_lb = tournament.get("training_lookback_sessions", 252)

            # Persist to PostgreSQL
            save_ledger_to_database(db, sym, ledger, ml_champ, train_lb)
            save_forecast_to_database(db, sym, fc)

            # Accumulate for CSV export
            for row in ledger:
                enriched_row = {
                    "symbol": sym,
                    **row,
                    "crowned_horizon": crowned_h,
                    "training_lookback_sessions": row.get("training_lookback_sessions", train_lb),
                    "ml_champion_type": ml_champ,
                }
                all_ledger_rows.append(enriched_row)

            # Extract horizon comparison metrics if present
            h_comp = tournament.get("horizon_comparison", {})
            m3_data = h_comp.get("3m", {})
            m6_data = h_comp.get("6m", {})
            m12_data = h_comp.get("12m", {})

            latest_price = fc.get("latest_close_price") or fc.get("current_price")
            summary_rows.append({
                "symbol": sym,
                "as_of_date": fc.get("as_of_date"),
                "current_price": latest_price,
                "crowned_horizon": crowned_h,
                "training_lookback_sessions": train_lb,
                "ml_champion_type": ml_champ,
                "grand_champion": tournament.get("champion"),
                "champion_30d_hits": tournament.get("champion_30d_hits", tournament.get("champion_dir_hits")),
                "champion_30d_hit_rate_pct": tournament.get("champion_30d_hit_rate_pct", tournament.get("champion_dir_hit_rate_pct")),
                "champion_30d_mae_pct": tournament.get("champion_30d_mae_pct", tournament.get("champion_mae_pct")),
                "champion_full_hits": tournament.get("champion_full_hits"),
                "champion_full_hit_rate_pct": tournament.get("champion_full_hit_rate_pct"),
                "champion_full_mae_pct": tournament.get("champion_full_mae_pct"),
                "champion_30d_penalty_loss": tournament.get("champion_30d_penalty_loss", tournament.get("champion_penalty_loss")),
                "champion_full_penalty_loss": tournament.get("champion_full_penalty_loss"),
                "prophet_30d_hits": tournament.get("prophet_dir_hits"),
                "prophet_30d_hit_rate_pct": tournament.get("prophet_dir_hit_rate_pct"),
                "prophet_30d_penalty_loss": tournament.get("prophet_penalty_loss_30d"),
                "prophet_full_hit_rate_pct": tournament.get("prophet_full_hit_rate_pct"),
                "ridge_30d_hit_rate_pct": tournament.get("ridge_30d_hit_rate_pct"),
                "ridge_30d_penalty_loss": tournament.get("ridge_penalty_loss_30d"),
                "ridge_full_hit_rate_pct": tournament.get("ridge_hit_rate_pct"),
                "xgboost_30d_hit_rate_pct": tournament.get("xgboost_30d_hit_rate_pct"),
                "xgboost_30d_penalty_loss": tournament.get("xgboost_penalty_loss_30d"),
                "xgboost_full_hit_rate_pct": tournament.get("xgboost_hit_rate_pct"),
                "lightgbm_30d_hit_rate_pct": tournament.get("lightgbm_30d_hit_rate_pct"),
                "lightgbm_30d_penalty_loss": tournament.get("lightgbm_penalty_loss_30d"),
                "lightgbm_full_hit_rate_pct": tournament.get("lightgbm_hit_rate_pct"),
                "huber_30d_hit_rate_pct": tournament.get("huber_30d_hit_rate_pct"),
                "huber_30d_penalty_loss": tournament.get("huber_penalty_loss_30d"),
                "huber_full_hit_rate_pct": tournament.get("huber_hit_rate_pct"),
                "bayesian_ridge_30d_hit_rate_pct": tournament.get("bayesian_ridge_30d_hit_rate_pct"),
                "bayesian_ridge_30d_penalty_loss": tournament.get("bayesian_ridge_penalty_loss_30d"),
                "bayesian_ridge_full_hit_rate_pct": tournament.get("bayesian_ridge_hit_rate_pct"),
                "hit_rate_3m_30d_pct": m3_data.get("champion_dir_hit_rate_pct"),
                "penalty_loss_3m_30d": m3_data.get("champion_30d_penalty_loss"),
                "hit_rate_3m_full_pct": m3_data.get("champion_full_hit_rate_pct"),
                "hit_rate_6m_30d_pct": m6_data.get("champion_dir_hit_rate_pct"),
                "penalty_loss_6m_30d": m6_data.get("champion_30d_penalty_loss"),
                "hit_rate_6m_full_pct": m6_data.get("champion_full_hit_rate_pct"),
                "hit_rate_12m_30d_pct": m12_data.get("champion_dir_hit_rate_pct"),
                "penalty_loss_12m_30d": m12_data.get("champion_30d_penalty_loss"),
                "hit_rate_12m_full_pct": m12_data.get("champion_full_hit_rate_pct"),
                "t_plus_1_target_price": fc.get("target_price"),
                "t_plus_1_expected_return_pct": fc.get("expected_return_pct"),
                "t_plus_1_stance": fc.get("stance"),
                "t_plus_1_conviction": fc.get("stance"),
                "t_plus_1_playbook": fc.get("stance_badge") or fc.get("playbook"),
                "total_eval_sessions": len(ledger),
            })

            success_count += 1
            sym_t1 = time.time()
            c_30_hits = tournament.get("champion_30d_hits", tournament.get("champion_dir_hits", 0))
            c_30_pct = tournament.get("champion_30d_hit_rate_pct", tournament.get("champion_dir_hit_rate_pct", 0.0))
            c_30_loss = tournament.get("champion_30d_penalty_loss", tournament.get("champion_penalty_loss", 0.0))
            c_full_pct = tournament.get("champion_full_hit_rate_pct", 0.0)
            logger.info(
                f"[{idx}/{len(symbols)}] {sym} done in {sym_t1 - sym_t0:.1f}s | "
                f"Crowned: {crowned_h} ({train_lb}d) | ML: {ml_champ} | "
                f"30D Loss: {c_30_loss:.2f}% | 30D Hits: {c_30_hits}/30 ({c_30_pct:.1f}%) | 180D: {c_full_pct:.1f}% | "
                f"T+1: {fc.get('stance')} ({fc.get('expected_return_pct', 0.0):+.2f}%)"
            )

        except Exception as e:
            logger.error(f"Failed processing {sym}: {e}", exc_info=True)
            fail_count += 1

    # 1. Export 180-Day Reality Ledger CSV
    if all_ledger_rows:
        df_ledger = pd.DataFrame(all_ledger_rows)
        priority_cols = [
            "symbol", "date", "actual_price", "actual_return_pct",
            "ml_pred_price", "ml_pred_return_pct", "ml_direction", "ml_is_hit", "ml_err_pct",
            "prophet_pred_price", "prophet_pred_return_pct", "prophet_direction", "prophet_is_hit", "prophet_err_pct",
            "winner", "bist30_ret_pct", "is_shock_day", "shock_type",
            "mlb_action", "mlb_flow_tl", "mlb_pnl_tl",
            "big5_action", "big5_flow_tl", "big5_pnl_tl",
            "kamu_action", "kamu_flow_tl", "kamu_pnl_tl",
            "ml_champion_type", "crowned_horizon", "training_lookback_sessions"
        ]
        remaining_cols = [c for c in df_ledger.columns if c not in priority_cols]
        final_cols = [c for c in priority_cols if c in df_ledger.columns] + remaining_cols
        df_ledger = df_ledger[final_cols]

        out_ledger_path = EXPORT_DIR / f"tertip_{n_eval}d_walk_forward_backtests.csv"
        df_ledger.to_csv(out_ledger_path, index=False)
        logger.info(f"Exported {len(df_ledger)} ledger rows across {success_count} symbols to: {out_ledger_path}")

    # 2. Export Multi-Horizon Tournament Summary CSV
    if summary_rows:
        df_summary = pd.DataFrame(summary_rows)
        out_summary_path = EXPORT_DIR / f"tertip_{n_eval}d_horizon_tournament_summary.csv"
        df_summary.to_csv(out_summary_path, index=False)
        logger.info(f"Exported tournament summary for {len(df_summary)} symbols to: {out_summary_path}")

        # 3. Synchronize config/tertip_crowned_models.yaml
        yaml_path = Path(__file__).resolve().parents[1] / "config" / "tertip_crowned_models.yaml"
        sync_crowned_yaml(summary_rows, yaml_path)

    t_total = time.time() - t_start
    logger.info("=" * 80)
    logger.info(f"COMPLETED {n_eval}D BIST 30 EXPORT: {success_count} succeeded, {fail_count} failed in {t_total:.1f}s")
    logger.info(f"Artifacts located at: {EXPORT_DIR}")
    logger.info("=" * 80)


def sync_crowned_yaml(summary_rows: list[dict[str, Any]], yaml_path: Path) -> None:
    """Sync crowned models and horizons to YAML config."""
    import yaml
    current_data: dict[str, Any] = {}
    if yaml_path.exists():
        try:
            with open(yaml_path, "r") as f:
                current_data = yaml.safe_load(f) or {}
        except Exception as e:
            logger.warning(f"Could not read existing YAML: {e}")

    symbols_map = current_data.get("symbols", {})
    for r in summary_rows:
        sym = r["symbol"]
        symbols_map[sym] = {
            "model": r["ml_champion_type"],
            "horizon": r["crowned_horizon"],
            "training_lookback_sessions": r["training_lookback_sessions"],
            "recent_30d_hits": r["champion_30d_hits"],
            "recent_30d_hit_rate_pct": r["champion_30d_hit_rate_pct"],
            "recent_30d_mae_pct": r["champion_30d_mae_pct"],
            "recent_30d_penalty_loss": r.get("champion_30d_penalty_loss"),
        }

    output_data = {
        "_metadata": {
            "description": "Crowned Tertip Machine Learning models and training lookback horizons optimized for lowest 30-day directional penalty loss (Option 2).",
            "calibration_date": str(pd.Timestamp.now().date()),
            "eval_window": "180d_walk_forward",
            "selection_metric": "lowest_30d_directional_penalty_loss",
        },
        "symbols": symbols_map,
    }

    yaml_path.parent.mkdir(parents=True, exist_ok=True)
    with open(yaml_path, "w") as f:
        yaml.dump(output_data, f, sort_keys=False, default_flow_style=False)
    logger.info(f"Successfully synced {len(symbols_map)} crowned model selections to: {yaml_path}")


if __name__ == "__main__":
    main()
