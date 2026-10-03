"""Export 30-Day Walk-Forward Backtest Track Data for BIST 30 to Database & CSV.

Computes the 30-day walk-forward arena for all active BIST 30 constituent equities,
persists the results to PostgreSQL:
- gold_tertip_walk_forward_backtests (session-by-session audited ledger)
- gold_tertip_daily_forecasts (latest T+1 signal & tournament summary)
And exports clean spreadsheet-ready CSV files to:
- /Users/ozkanyildirim/data/mdk_oracle/export_artifacts/tertip_30d_walk_forward_backtests.csv
- /Users/ozkanyildirim/data/mdk_oracle/export_artifacts/tertip_30d_tournament_summary.csv
"""

from __future__ import annotations

import logging
import time
from pathlib import Path
from typing import Any

import pandas as pd

from mdk_trading_oracle.core.db import PostgresManager
from mdk_trading_oracle.data.silver.tertip_ml_forecaster import get_tertip_ml_forecast

# Suppress verbose Prophet and CmdStanPy logs during batch run
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("export_tertip_30d")
logging.getLogger("cmdstanpy").setLevel(logging.ERROR)
logging.getLogger("prophet").setLevel(logging.ERROR)
logging.getLogger("lightgbm").setLevel(logging.ERROR)

EXPORT_DIR = Path.home() / "data" / "mdk_oracle" / "export_artifacts"
EXPORT_DIR.mkdir(parents=True, exist_ok=True)


def ensure_gold_tables(db: PostgresManager) -> None:
    """Ensure gold tables exist in PostgreSQL."""
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
        calculated_at TIMESTAMPTZ DEFAULT CURRENT_TIMESTAMP,
        PRIMARY KEY (symbol, trade_date)
    );

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
        calculated_at TIMESTAMPTZ DEFAULT CURRENT_TIMESTAMP,
        PRIMARY KEY (symbol, as_of_date)
    );
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
        # Fallback to standard BIST 30 list if table empty
        return [
            "AEFES", "AKBNK", "ASELS", "ASTOR", "BIMAS", "DSTKF", "EKGYO", "ENKAI",
            "EREGL", "FROTO", "GARAN", "GUBRF", "ISCTR", "KCHOL", "KRDMD", "MGROS",
            "PETKM", "PGSUS", "SAHOL", "SASA", "SISE", "TAVHL", "TCELL", "THYAO",
            "TOASO", "TRALT", "TTKOM", "TUPRS", "VAKBN", "YKBNK"
        ]
    return res["symbol"].to_list()


def save_ledger_to_database(db: PostgresManager, symbol: str, ledger: list[dict[str, Any]], ml_champion: str) -> None:
    """Upsert 30D backtest ledger records into gold_tertip_walk_forward_backtests."""
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
        ml_champion_type, calculated_at
    ) VALUES (
        %(symbol)s, %(trade_date)s, %(actual_price)s, %(actual_return_pct)s, %(bist30_ret_pct)s,
        %(ml_pred_price)s, %(ml_pred_return_pct)s, %(ml_direction)s, %(ml_err_pct)s, %(ml_is_hit)s,
        %(prophet_pred_price)s, %(prophet_pred_return_pct)s, %(prophet_direction)s, %(prophet_err_pct)s, %(prophet_is_hit)s,
        %(winner)s, %(is_shock_day)s, %(shock_type)s, %(days_since_pos_shock)s, %(days_since_neg_shock)s,
        %(mlb_action)s, %(mlb_flow_tl)s, %(mlb_buy_tl)s, %(mlb_sell_tl)s, %(mlb_pnl_tl)s,
        %(big5_action)s, %(big5_flow_tl)s, %(big5_buy_tl)s, %(big5_sell_tl)s, %(big5_pnl_tl)s,
        %(kamu_action)s, %(kamu_flow_tl)s, %(kamu_buy_tl)s, %(kamu_sell_tl)s, %(kamu_pnl_tl)s,
        %(ml_champion_type)s, CURRENT_TIMESTAMP
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
        prophet_target_price, prophet_expected_return_pct, calculated_at
    ) VALUES (
        %(symbol)s, %(as_of_date)s, %(current_price)s, %(target_price)s, %(expected_return_pct)s,
        %(price_low)s, %(price_high)s, %(stance)s, %(conviction)s, %(playbook)s,
        %(ml_champion_type)s, %(champion_dir_hits)s, %(champion_dir_hit_rate_pct)s, %(champion_mae_pct)s,
        %(prophet_target_price)s, %(prophet_expected_return_pct)s, CURRENT_TIMESTAMP
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
        calculated_at = CURRENT_TIMESTAMP;
    """
    p = {
        "symbol": symbol,
        "as_of_date": fc.get("as_of_date"),
        "current_price": fc.get("latest_close_price") or fc.get("current_price"),
        "target_price": fc.get("target_price"),
        "expected_return_pct": fc.get("expected_return_pct"),
        "price_low": fc.get("price_low"),
        "price_high": fc.get("price_high"),
        "stance": fc.get("stance"),
        "conviction": fc.get("conviction", "MEDIUM"),
        "playbook": fc.get("playbook") or fc.get("stance_badge", "CONSOLIDATION"),
        "ml_champion_type": t.get("grand_champion_key") or t.get("ml_champion_type", "Ridge"),
        "champion_dir_hits": t.get("champion_dir_hits"),
        "champion_dir_hit_rate_pct": t.get("champion_dir_hit_rate_pct"),
        "champion_mae_pct": t.get("champion_mae_pct"),
        "prophet_target_price": fc.get("prophet_target_price"),
        "prophet_expected_return_pct": fc.get("prophet_expected_return_pct"),
    }
    db.execute(insert_query, p)


def main() -> None:
    import argparse
    parser = argparse.ArgumentParser(description="Export 30-Day Walk-Forward Backtest Track Data")
    parser.add_argument("--symbols", type=str, default="", help="Comma-separated symbols (e.g. ASTOR,ASELS)")
    args = parser.parse_args()

    t_start = time.time()
    db = PostgresManager()

    logger.info("Initializing Gold database tables...")
    ensure_gold_tables(db)

    if args.symbols:
        symbols = [s.strip().upper() for s in args.symbols.split(",") if s.strip()]
    else:
        symbols = get_active_bist30_symbols(db)
    logger.info(f"Targeting {len(symbols)} active symbols: {symbols}")

    all_ledger_rows: list[dict[str, Any]] = []
    summary_rows: list[dict[str, Any]] = []

    success_count = 0
    fail_count = 0

    for idx, sym in enumerate(symbols, 1):
        sym_t0 = time.time()
        logger.info(f"[{idx}/{len(symbols)}] Processing {sym}...")

        try:
            fc = get_tertip_ml_forecast(db, sym, force_refresh=True)
            if "error" in fc:
                logger.warning(f"Skipping {sym}: {fc['error']}")
                fail_count += 1
                continue

            ledger = fc.get("walk_forward_ledger", [])
            tournament = fc.get("tournament_summary", {})
            ml_champ = tournament.get("ml_champion_type", "Ridge")
            grand_champ = tournament.get("grand_champion_key") or ml_champ

            # Persist to database
            save_ledger_to_database(db, sym, ledger, grand_champ)
            save_forecast_to_database(db, sym, fc)

            # Accumulate for CSV export
            for row in ledger:
                enriched_row = {"symbol": sym, **row, "ml_champion_type": grand_champ}
                all_ledger_rows.append(enriched_row)

            # Accumulate tournament summary row
            summary_rows.append({
                "symbol": sym,
                "as_of_date": fc.get("as_of_date"),
                "current_price": fc.get("current_price"),
                "t_plus_1_target_price": fc.get("target_price"),
                "t_plus_1_expected_return_pct": fc.get("expected_return_pct"),
                "t_plus_1_stance": fc.get("stance"),
                "t_plus_1_conviction": fc.get("conviction"),
                "t_plus_1_playbook": fc.get("playbook"),
                "ml_champion_type": ml_champ,
                "grand_champion": tournament.get("champion"),
                "champion_dir_hits_30d": tournament.get("champion_dir_hits"),
                "champion_dir_hit_rate_pct": tournament.get("champion_dir_hit_rate_pct"),
                "champion_mae_pct": tournament.get("champion_mae_pct"),
                "ml_dir_hits_30d": tournament.get("ml_dir_hits"),
                "ml_dir_hit_rate_pct": tournament.get("ml_dir_hit_rate_pct"),
                "ml_mae_pct": tournament.get("ml_mae_pct"),
                "prophet_dir_hits_30d": tournament.get("prophet_dir_hits"),
                "prophet_dir_hit_rate_pct": tournament.get("prophet_dir_hit_rate_pct"),
                "prophet_mae_pct": tournament.get("prophet_mae_pct"),
                "ridge_dir_hit_rate_pct": tournament.get("ridge_dir_hit_rate_pct", tournament.get("ridge_hit_rate_pct")),
                "xgboost_dir_hit_rate_pct": tournament.get("xgboost_dir_hit_rate_pct", tournament.get("xgboost_hit_rate_pct")),
                "ml_error_wins": tournament.get("ml_error_wins"),
                "prophet_error_wins": tournament.get("prophet_error_wins"),
                "total_sessions": len(ledger),
            })

            success_count += 1
            sym_t1 = time.time()
            logger.info(
                f"[{idx}/{len(symbols)}] {sym} done in {sym_t1 - sym_t0:.1f}s | "
                f"Champ: {grand_champ} | Hits: {tournament.get('champion_dir_hits')}/30 "
                f"({tournament.get('champion_dir_hit_rate_pct', 0.0):.1f}%) | "
                f"T+1: {fc.get('stance')} ({fc.get('expected_return_pct', 0.0):+.2f}%)"
            )

        except Exception as e:
            logger.error(f"Failed processing {sym}: {e}", exc_info=True)
            fail_count += 1

    # 1. Export 30-Day Walk-Forward Reality Ledger CSV
    if all_ledger_rows:
        df_ledger = pd.DataFrame(all_ledger_rows)
        # Reorder columns for optimal spreadsheet readability
        priority_cols = [
            "symbol", "date", "actual_price", "actual_return_pct",
            "ml_pred_price", "ml_pred_return_pct", "ml_direction", "ml_is_hit", "ml_err_pct",
            "prophet_pred_price", "prophet_pred_return_pct", "prophet_direction", "prophet_is_hit", "prophet_err_pct",
            "winner", "bist30_ret_pct", "is_shock_day", "shock_type",
            "mlb_action", "mlb_flow_tl", "mlb_pnl_tl",
            "big5_action", "big5_flow_tl", "big5_pnl_tl",
            "kamu_action", "kamu_flow_tl", "kamu_pnl_tl",
            "ml_champion_type"
        ]
        remaining_cols = [c for c in df_ledger.columns if c not in priority_cols]
        final_cols = [c for c in priority_cols if c in df_ledger.columns] + remaining_cols
        df_ledger = df_ledger[final_cols]

        out_ledger_path = EXPORT_DIR / "tertip_30d_walk_forward_backtests.csv"
        df_ledger.to_csv(out_ledger_path, index=False)
        logger.info(f"Exported {len(df_ledger)} ledger rows across {success_count} symbols to: {out_ledger_path}")

    # 2. Export Tournament Performance Summary CSV
    if summary_rows:
        df_summary = pd.DataFrame(summary_rows)
        out_summary_path = EXPORT_DIR / "tertip_30d_tournament_summary.csv"
        df_summary.to_csv(out_summary_path, index=False)
        logger.info(f"Exported tournament summary for {len(df_summary)} symbols to: {out_summary_path}")

    t_total = time.time() - t_start
    logger.info("=" * 80)
    logger.info(f"COMPLETED BIST 30 EXPORT: {success_count} succeeded, {fail_count} failed in {t_total:.1f}s")
    logger.info(f"Artifacts located at: {EXPORT_DIR}")
    logger.info("=" * 80)


if __name__ == "__main__":
    main()
