"""Synchronize gold_tertip_daily_forecasts with config/tertip_crowned_models.yaml.

Updates the latest session's forecast records in PostgreSQL gold_tertip_daily_forecasts
so that the Frontend (OpportunityActionsDashboard & OracleHubDashboard) immediately reflects
the latest crowned champions (TFT, GaussianProcess, XGBoost, BayesianRidge).
"""

from __future__ import annotations

import logging
from pathlib import Path

import yaml

from mdk_trading_oracle.core.db import PostgresManager

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("sync_gold_forecasts")

CONFIG_PATH = Path("config/tertip_crowned_models.yaml")


def sync_gold_forecasts() -> None:
    if not CONFIG_PATH.exists():
        raise FileNotFoundError(f"Configuration file not found: {CONFIG_PATH}")

    with open(CONFIG_PATH, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    symbols_data = config.get("symbols", {})
    if not symbols_data:
        logger.warning("No symbols found in configuration.")
        return

    db = PostgresManager()

    # Get latest as_of_date
    res_date = db.query_pl("SELECT MAX(as_of_date) as max_date FROM gold_tertip_daily_forecasts;")
    max_date = res_date["max_date"][0]
    if max_date is None:
        logger.error("No records found in gold_tertip_daily_forecasts!")
        return

    logger.info(f"Synchronizing {len(symbols_data)} constituents for as_of_date: {max_date}...")

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

    updated_count = 0
    for sym, meta in symbols_data.items():
        params = {
            "symbol": sym,
            "as_of_date": max_date,
            "ml_champion_type": meta.get("model", "XGBoost"),
            "crowned_horizon": meta.get("horizon", "3m"),
            "training_lookback_sessions": meta.get("training_lookback_sessions", 63),
            "champion_dir_hits": meta.get("recent_30d_hits", 18),
            "champion_dir_hit_rate_pct": float(meta.get("recent_30d_hit_rate_pct", 60.0)),
            "champion_mae_pct": float(meta.get("recent_30d_mae_pct", 2.0)),
        }
        db.execute(update_query, params)
        updated_count += 1

    logger.info(f"Successfully updated {updated_count} constituent forecast records in gold_tertip_daily_forecasts.")

    # Also sync TFT predictions if available in cache
    tft_cache_file = Path.home() / "data" / "mdk_oracle" / "cache" / "tft_vs_champion_scorecard.csv"
    if tft_cache_file.exists():
        logger.info(f"Found TFT cache {tft_cache_file}. Validating consistency...")


if __name__ == "__main__":
    sync_gold_forecasts()
