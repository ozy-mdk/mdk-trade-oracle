"""30-Stock Cross-Constituent Temporal Fusion Transformer (TFT) Walk-Forward Arena.

Evaluates whether a universal, cross-sectional TFT foundation model trained simultaneously
across all 30 BIST 30 equities can outperform each constituent's incumbent champion
(persisted in gold_tertip_walk_forward_backtests and config/tertip_crowned_models.yaml).

Features:
- Static categoricals: symbol, sector
- Time-varying known categoricals: day_of_week
- Time-varying unknown reals: 3-pillar microstructure features + zero-leakage Prophet baseline
- Out-of-sample evaluation: trailing 30 trading sessions (2026-08-19 to 2026-09-30)
- Metric: Directional Hit Rate % (±0.25% / 25 bps deadband), MAE %, and Tournament Loss.
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
import time
import warnings
from pathlib import Path

import lightning.pytorch as pl
import numpy as np
import pandas as pd
from pytorch_forecasting import TemporalFusionTransformer, TimeSeriesDataSet
from pytorch_forecasting.metrics import QuantileLoss
import torch

from mdk_trading_oracle.core.db import PostgresManager
from mdk_trading_oracle.data.silver.tertip_ml_forecaster import (
    DEADBAND_PCT,
    FEATURE_COLS,
    _check_hit,
    attach_prophet_rolling_features,
    extract_3pillar_time_series,
)

warnings.filterwarnings("ignore")
logging.getLogger("lightning.pytorch").setLevel(logging.ERROR)
logging.getLogger("pytorch_forecasting").setLevel(logging.ERROR)
logging.getLogger("cmdstanpy").setLevel(logging.ERROR)
logging.getLogger("prophet").setLevel(logging.ERROR)

CACHE_DIR = Path.home() / "data" / "mdk_oracle" / "cache"
CACHE_DIR.mkdir(parents=True, exist_ok=True)
PANEL_CACHE_FILE = CACHE_DIR / "bist30_panel_tft.parquet"

ACCELERATOR = "cpu"


def load_bist30_panel(db: PostgresManager, force_reload: bool = False) -> pd.DataFrame:
    """Load or build panel dataset of all 30 BIST 30 stocks with sector and microstructure features."""
    if not force_reload and PANEL_CACHE_FILE.exists():
        print(f"Loading cached BIST 30 panel dataset from {PANEL_CACHE_FILE}...")
        df_panel = pd.read_parquet(PANEL_CACHE_FILE)
        return df_panel

    print("Building BIST 30 multi-constituent panel dataset from PostgreSQL...")

    # 1. Fetch sectors
    df_inst = db.query_pl(
        """
        SELECT symbol, COALESCE(sector, 'Diversified') as sector 
        FROM bronze_instruments 
        WHERE symbol IN (SELECT DISTINCT symbol FROM gold_tertip_walk_forward_backtests);
        """
    )
    sector_map = dict(zip(df_inst["symbol"], df_inst["sector"]))
    symbols = sorted(list(sector_map.keys()))

    all_dfs = []
    t0 = time.time()
    for idx, sym in enumerate(symbols, 1):
        print(f"[{idx}/30] Extracting features for {sym} ({sector_map.get(sym, 'Unknown')})...")
        df_stock = extract_3pillar_time_series(db, sym, lookback_days=500)
        if df_stock.empty or len(df_stock) < 60:
            print(f"Warning: skipping {sym}, insufficient records ({len(df_stock)})")
            continue
        df_stock = attach_prophet_rolling_features(df_stock, sym)
        df_stock["symbol"] = sym
        df_stock["sector"] = sector_map.get(sym, "Diversified")
        all_dfs.append(df_stock)

    print(f"Extracted {len(all_dfs)} stocks in {time.time() - t0:.1f}s.")
    df_all = pd.concat(all_dfs, ignore_index=True)

    # Clean & ensure trade_date is datetime
    df_all["trade_date"] = pd.to_datetime(df_all["trade_date"])
    df_all = df_all.sort_values(["trade_date", "symbol"]).reset_index(drop=True)

    # Build discrete integer time_idx per unique trading session
    unique_dates = sorted(df_all["trade_date"].unique())
    date_to_idx = {d: i for i, d in enumerate(unique_dates)}
    df_all["time_idx"] = df_all["trade_date"].map(date_to_idx)
    df_all["day_of_week"] = df_all["trade_date"].dt.dayofweek.astype(str)

    # Target return: 1-step forward close return
    # We predict (Close_{t} - Close_{t-1}) / Close_{t-1} * 100 for session t
    df_all["daily_return_pct"] = (
        (df_all["close_price"] - df_all.groupby("symbol")["close_price"].shift(1))
        / df_all.groupby("symbol")["close_price"].shift(1)
        * 100.0
    ).fillna(0.0)

    # Save to cache
    df_all.to_parquet(PANEL_CACHE_FILE)
    print(f"Saved panel dataset with {len(df_all)} rows to {PANEL_CACHE_FILE}.")
    return df_all


def run_tft_bist30_arena(n_eval: int = 30) -> pd.DataFrame:
    """Run walk-forward evaluation of Global TFT across all 30 BIST 30 stocks."""
    db = PostgresManager()
    df_panel = load_bist30_panel(db)

    # Determine evaluation date range from database gold_tertip_walk_forward_backtests
    db_eval = db.query_pl(
        f"""
        WITH last_dates AS (
            SELECT DISTINCT trade_date FROM gold_tertip_walk_forward_backtests ORDER BY trade_date DESC LIMIT {n_eval}
        )
        SELECT symbol, trade_date, actual_price, actual_return_pct, ml_pred_return_pct, ml_is_hit, ml_err_pct
        FROM gold_tertip_walk_forward_backtests
        WHERE trade_date IN (SELECT trade_date FROM last_dates)
        ORDER BY trade_date ASC, symbol ASC;
        """
    )
    df_champion = db_eval.to_pandas()
    eval_dates = sorted(df_champion["trade_date"].unique())
    symbols = sorted(df_panel["symbol"].unique())

    print("=" * 115)
    print(f"30-CONSTITUENT UNIVERSAL TFT TOURNAMENT ARENA")
    print(f"Evaluation Window: {len(eval_dates)} Sessions ({eval_dates[0]} to {eval_dates[-1]})")
    print(f"Total Constituents: {len(symbols)} Equities")
    print(f"Device: {ACCELERATOR.upper()}")
    print("=" * 115)

    # Identify numerical feature columns present
    feature_candidates = [
        "feat_prophet_ret_today_pct",
        "feat_mlb_cost_spread_pct",
        "feat_total_inst_net_share_today",
        "feat_bofa_aggression_ratio",
        "feat_tertip_squeeze_delta",
        "feat_ret_today_pct",
        "feat_ret_yesterday_pct",
        "feat_ret_3d_cum_pct",
        "feat_ret_acceleration_pct",
        "feat_intraday_range_pct",
        "feat_volatility_pinch_5d_20d",
        "feat_bollinger_bandwidth_pct",
        "feat_bist30_ret_today_pct",
        "feat_days_since_pos_shock",
        "feat_days_since_neg_shock",
        "feat_kamu_post_shock_defense",
        "feat_mlb_turnover_intensity",
        "feat_big5_turnover_intensity",
        "feat_kamu_turnover_intensity",
    ]
    f_cols = [c for c in feature_candidates if c in df_panel.columns]
    # Fill any NaNs in features
    df_panel[f_cols] = df_panel[f_cols].fillna(0.0)

    # Clip extreme outlier returns to [-10, 10]
    df_panel["daily_return_pct"] = df_panel["daily_return_pct"].clip(-10.0, 10.0)

    # Find the time_idx corresponding to evaluation start
    first_eval_date = pd.to_datetime(eval_dates[0])
    eval_start_idx = df_panel[df_panel["trade_date"] == first_eval_date]["time_idx"].min()
    max_time_idx = df_panel["time_idx"].max()

    tft_predictions: list[dict[str, Any]] = []

    # Sequence parameters
    max_encoder_length = 5  # trailing 5 sessions
    max_prediction_length = 1

    print(f"Encoder Sequence Length: {max_encoder_length} sessions")
    print(f"Training lookback horizon: FULL EXPANDING HISTORY (~33,000 pooled multi-stock samples) - Unweighted Quantile Loss")
    print("-" * 115)

    # Walk-forward loop over evaluation sessions
    for step_num, eval_d in enumerate(eval_dates, 1):
        step_t0 = time.time()
        curr_eval_date = pd.to_datetime(eval_d)
        curr_idx = df_panel[df_panel["trade_date"] == curr_eval_date]["time_idx"].iloc[0]

        # Training data: FULL EXPANDING HISTORY strictly BEFORE curr_idx
        # Zero future leakage: train on all history up to curr_idx - 1
        train_start_idx = 0
        df_train = df_panel[
            (df_panel["time_idx"] >= train_start_idx) & (df_panel["time_idx"] < curr_idx)
        ].copy()

        # Build TimeSeriesDataSet strictly on training slice
        training_dataset = TimeSeriesDataSet(
            df_train,
            time_idx="time_idx",
            target="daily_return_pct",
            group_ids=["symbol"],
            min_encoder_length=max_encoder_length,
            max_encoder_length=max_encoder_length,
            min_prediction_length=max_prediction_length,
            max_prediction_length=max_prediction_length,
            static_categoricals=["symbol", "sector"],
            time_varying_known_categoricals=["day_of_week"],
            time_varying_unknown_reals=["daily_return_pct"] + f_cols,
        )

        train_dataloader = training_dataset.to_dataloader(batch_size=128, shuffle=True, num_workers=0)

        # For predicting curr_idx: slice [curr_idx - max_encoder_length : curr_idx + 1]
        df_predict_slice = df_panel[
            (df_panel["time_idx"] >= curr_idx - max_encoder_length) & (df_panel["time_idx"] <= curr_idx)
        ].copy()

        # Predict dataset using training dataset parameters
        predict_dataset = TimeSeriesDataSet.from_dataset(
            training_dataset,
            df_predict_slice,
            predict=True,
            stop_randomization=True,
        )
        predict_dataloader = predict_dataset.to_dataloader(batch_size=128, shuffle=False, num_workers=0)

        # Instantiate TFT with 4 attention heads for full historical regime discovery
        tft = TemporalFusionTransformer.from_dataset(
            training_dataset,
            learning_rate=0.015,
            hidden_size=32,
            attention_head_size=4,
            dropout=0.15,
            hidden_continuous_size=16,
            loss=QuantileLoss(quantiles=[0.1, 0.5, 0.9]),
            reduce_on_plateau_patience=2,
        )

        # Train model with Lightning
        epochs = 5 if step_num == 1 else 2  # full train on step 1, fast adaptation on subsequent
        trainer = pl.Trainer(
            max_epochs=epochs,
            accelerator="cpu" if ACCELERATOR == "cpu" else "mps",
            enable_model_summary=False,
            enable_checkpointing=False,
            logger=False,
            enable_progress_bar=False,
        )
        trainer.fit(tft, train_dataloaders=train_dataloader)

        # Run inference (predict median quantile index 1)
        raw_preds = tft.predict(
            predict_dataloader, mode="raw", return_x=True, trainer_kwargs=dict(accelerator="cpu")
        )
        # raw_preds.output.prediction shape: [n_stocks, 1, 3] (quantiles: 0.1, 0.5, 0.9)
        median_preds = raw_preds.output.prediction[:, 0, 1].numpy()
        pred_symbols = predict_dataset.decoded_index["symbol"].to_list()

        step_elapsed = time.time() - step_t0
        print(f"Session {step_num:02d}/{n_eval:02d} [{eval_d}] -> Trained & forecasted {len(pred_symbols)} stocks ({step_elapsed:.1f}s)")

        for sym, pred_val in zip(pred_symbols, median_preds):
            tft_predictions.append({
                "symbol": sym,
                "trade_date": eval_d,
                "tft_pred_return_pct": float(np.clip(pred_val, -10.0, 10.0)),
            })

    # -----------------------------------------------------------------------
    # Comparative Scorecard: TFT vs. Incumbent Champion
    # -----------------------------------------------------------------------
    df_tft = pd.DataFrame(tft_predictions)
    # Merge with ground truth and incumbent champion from database
    df_merged = pd.merge(
        df_champion,
        df_tft,
        on=["symbol", "trade_date"],
        how="inner",
    )

    df_merged["tft_is_hit"] = df_merged.apply(
        lambda r: _check_hit(r["tft_pred_return_pct"], r["actual_return_pct"], DEADBAND_PCT),
        axis=1,
    )
    df_merged["tft_err_pct"] = np.abs(df_merged["tft_pred_return_pct"] - df_merged["actual_return_pct"])

    scorecard = []
    tft_wins = 0
    champ_wins = 0
    ties = 0

    for sym in symbols:
        sub = df_merged[df_merged["symbol"] == sym]
        if sub.empty:
            continue
        n_sessions = len(sub)
        
        # Champion metrics
        champ_hits = sub["ml_is_hit"].sum()
        champ_hit_rate = (champ_hits / n_sessions) * 100.0
        champ_mae = sub["ml_err_pct"].mean()
        champ_hit_mae = sub[sub["ml_is_hit"]]["ml_err_pct"].mean() if champ_hits > 0 else 0.0
        champ_miss_mae = sub[~sub["ml_is_hit"]]["ml_err_pct"].mean() if (n_sessions - champ_hits) > 0 else 0.0
        champ_loss = (100.0 - champ_hit_rate) + 1.0 * champ_hit_mae + 2.5 * champ_miss_mae

        # TFT metrics
        tft_hits = sub["tft_is_hit"].sum()
        tft_hit_rate = (tft_hits / n_sessions) * 100.0
        tft_mae = sub["tft_err_pct"].mean()
        tft_hit_mae = sub[sub["tft_is_hit"]]["tft_err_pct"].mean() if tft_hits > 0 else 0.0
        tft_miss_mae = sub[~sub["tft_is_hit"]]["tft_err_pct"].mean() if (n_sessions - tft_hits) > 0 else 0.0
        tft_loss = (100.0 - tft_hit_rate) + 1.0 * tft_hit_mae + 2.5 * tft_miss_mae

        hit_diff = tft_hit_rate - champ_hit_rate
        loss_diff = tft_loss - champ_loss  # lower loss is better

        if hit_diff > 0 or (abs(hit_diff) < 1e-4 and loss_diff < 0):
            winner = "TFT"
            tft_wins += 1
        elif hit_diff < 0 or (abs(hit_diff) < 1e-4 and loss_diff > 0):
            winner = "Champion"
            champ_wins += 1
        else:
            winner = "Tie"
            ties += 1

        scorecard.append({
            "symbol": sym,
            "champ_hits": f"{champ_hits}/{n_sessions}",
            "champ_hit_rate": champ_hit_rate,
            "champ_mae": champ_mae,
            "champ_loss": champ_loss,
            "tft_hits": f"{tft_hits}/{n_sessions}",
            "tft_hit_rate": tft_hit_rate,
            "tft_mae": tft_mae,
            "tft_loss": tft_loss,
            "hit_diff": hit_diff,
            "winner": winner,
        })

    df_score = pd.DataFrame(scorecard)

    print("\n" + "=" * 125)
    print(f"{'Stock':<7} | {'Incumbent Champion':<22} | {'Universal TFT':<22} | {'Hit% Delta':<10} | {'Winner':<10}")
    print(f"{'':<7} | {'Hits':<8} {'Hit%':<6} {'MAE%':<6} | {'Hits':<8} {'Hit%':<6} {'MAE%':<6} | {'':<10} | {'':<10}")
    print("-" * 125)
    for _, r in df_score.iterrows():
        sign = "+" if r["hit_diff"] > 0 else ""
        print(
            f"{r['symbol']:<7} | {r['champ_hits']:<8} {r['champ_hit_rate']:>5.1f}% {r['champ_mae']:>5.2f}% | "
            f"{r['tft_hits']:<8} {r['tft_hit_rate']:>5.1f}% {r['tft_mae']:>5.2f}% | "
            f"{sign}{r['hit_diff']:>5.1f}%    | {r['winner']:<10}"
        )
    print("=" * 125)

    print("\nFINAL TOURNAMENT TALLY:")
    print(f"Universal TFT Wins:        {tft_wins} / {len(scorecard)} stocks")
    print(f"Incumbent Champion Wins:   {champ_wins} / {len(scorecard)} stocks")
    print(f"Ties:                      {ties} / {len(scorecard)} stocks")
    avg_champ_hit = df_score["champ_hit_rate"].mean()
    avg_tft_hit = df_score["tft_hit_rate"].mean()
    print(f"Average Incumbent Hit Rate: {avg_champ_hit:.1f}%")
    print(f"Average TFT Hit Rate:       {avg_tft_hit:.1f}%")
    print("=" * 125)

    # Save summary CSV
    export_csv = CACHE_DIR / "tft_vs_champion_scorecard.csv"
    df_score.to_csv(export_csv, index=False)
    print(f"Exported detailed scorecard to {export_csv}")

    return df_score


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Test Universal TFT against Incumbent Champions on all 30 BIST Stocks")
    parser.add_argument("--eval-sessions", type=int, default=30, help="Number of evaluation sessions")
    args = parser.parse_args()

    run_tft_bist30_arena(n_eval=args.eval_sessions)
