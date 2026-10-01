"""Experiment: Net 3-Pillar Imbalance & Deadband Calibration for ASELS and BIST Portfolio.

Tests:
1. Feature Configurations:
   - V0: Baseline (Current 18 Features)
   - V1: + Macro Shock Distance (20 Features: + feat_days_since_pos_shock, feat_days_since_neg_shock)
   - V2: + Net 3-Pillar Flow Shares (23 Features: + mlb_net_share, big5_net_share, kamu_net_share)
   - V3: Lean Net Imbalance (17 Features: replaces gross buy/sell shares with net shares + shock)
   - V4: Net Imbalance Ratios (23 Features)
   - V5: Aggregate Institutional Net Direction (Base 18 + total_inst_net_share + shock distance)

2. Instant Vectorized Deadband Calibration:
   - 0.02% (Current prod)
   - 0.15% (1-2 ticks)
   - 0.20% (2 ticks)
   - 0.25% (Consolidation band)
   - 0.30%
   - 0.35%
"""

from __future__ import annotations

import logging
import warnings
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from xgboost import XGBRegressor

from mdk_trading_oracle.core.db import PostgresManager
from mdk_trading_oracle.data.silver.tertip_ml_forecaster import (
    FEATURE_COLS,
    TRAIN_LOOKBACK_SESSIONS,
    extract_3pillar_time_series,
)

warnings.filterwarnings("ignore")
logging.getLogger("cmdstanpy").setLevel(logging.ERROR)
logging.getLogger("prophet").setLevel(logging.ERROR)

CACHE_DIR = Path.home() / "data" / "mdk_oracle" / "cache" / "prophet_ablation"


def load_data_with_all_features(db: PostgresManager, symbol: str) -> pd.DataFrame:
    """Load time series and compute candidate feature variants."""
    sym = symbol.upper()
    cache_file = CACHE_DIR / f"{sym}_prophet_rolling_features.parquet"

    if cache_file.exists():
        df = pd.read_parquet(cache_file)
    else:
        df = extract_3pillar_time_series(db, sym, lookback_days=550)

    tt = df["total_turnover_tl"].replace(0, np.nan).fillna(1e6)

    # 1. Net 3-Pillar Flow Shares (% of Total Turnover)
    for p in ["mlb", "big5", "kamu"]:
        df[f"feat_{p}_net_share_today"] = (
            (df[f"{p}_buy_tl"].fillna(0) - df[f"{p}_sell_tl"].fillna(0)) / tt * 100.0
        ).clip(-100.0, 100.0)

        turnover_p = df[f"{p}_buy_tl"].fillna(0) + df[f"{p}_sell_tl"].fillna(0)
        df[f"feat_{p}_imbalance_ratio_today"] = np.where(
            turnover_p > 1000.0,
            (df[f"{p}_buy_tl"].fillna(0) - df[f"{p}_sell_tl"].fillna(0)) / turnover_p,
            0.0,
        ).clip(-1.0, 1.0)

    # Aggregate Institutional Net Flow Share
    df["feat_total_inst_net_share_today"] = (
        df["feat_mlb_net_share_today"] + df["feat_big5_net_share_today"] + df["feat_kamu_net_share_today"]
    ).clip(-100.0, 100.0)

    # 2. Shock distances if not present
    if "feat_days_since_pos_shock" not in df.columns:
        series_idx = pd.Series(range(len(df)), index=df.index)
        pos_shock_idx = series_idx.where(df["daily_return_pct"] >= 0.03).ffill()
        df["feat_days_since_pos_shock"] = (series_idx - pos_shock_idx).fillna(63.0).clip(0.0, 63.0)

        neg_shock_idx = series_idx.where(df["daily_return_pct"] <= -0.03).ffill()
        df["feat_days_since_neg_shock"] = (series_idx - neg_shock_idx).fillna(63.0).clip(0.0, 63.0)

    return df


def define_feature_variants() -> dict[str, list[str]]:
    """Define the feature candidate configurations."""
    base_18 = list(FEATURE_COLS)

    v1_shock = base_18 + ["feat_days_since_pos_shock", "feat_days_since_neg_shock"]

    v2_net_and_gross = base_18 + [
        "feat_days_since_pos_shock",
        "feat_days_since_neg_shock",
        "feat_mlb_net_share_today",
        "feat_big5_net_share_today",
        "feat_kamu_net_share_today",
    ]

    gross_shares = [
        "feat_mlb_buy_share_today", "feat_mlb_sell_share_today",
        "feat_big5_buy_share_today", "feat_big5_sell_share_today",
        "feat_kamu_buy_share_today", "feat_kamu_sell_share_today",
    ]
    v3_lean_net = [f for f in base_18 if f not in gross_shares] + [
        "feat_mlb_net_share_today",
        "feat_big5_net_share_today",
        "feat_kamu_net_share_today",
        "feat_days_since_pos_shock",
        "feat_days_since_neg_shock",
    ]

    v4_ratios = base_18 + [
        "feat_days_since_pos_shock",
        "feat_days_since_neg_shock",
        "feat_mlb_imbalance_ratio_today",
        "feat_big5_imbalance_ratio_today",
        "feat_kamu_imbalance_ratio_today",
    ]

    v5_agg_net = base_18 + [
        "feat_days_since_pos_shock",
        "feat_days_since_neg_shock",
        "feat_total_inst_net_share_today",
    ]

    return {
        "V0_Base18": base_18,
        "V1_Shock20": v1_shock,
        "V2_NetAndGross23": v2_net_and_gross,
        "V3_LeanNet17": v3_lean_net,
        "V4_ImbalanceRatios23": v4_ratios,
        "V5_AggNet21": v5_agg_net,
    }


def precompute_walk_forward_predictions(
    df: pd.DataFrame,
    symbol: str,
    variant_name: str,
    feature_cols: list[str],
    n_sessions: int = 30,
    train_lookback: int = TRAIN_LOOKBACK_SESSIONS,
) -> pd.DataFrame:
    """Run model fits once for 30 sessions and return actual & predicted returns."""
    N = len(df)
    eval_start_idx = N - n_sessions
    rows = []

    for k in range(eval_start_idx, N):
        test_row = df.iloc[k]
        prev_row = df.iloc[k - 1]
        train_data = df.iloc[:k]

        actual_price = float(test_row["close_price"])
        prev_price = float(prev_row["close_price"])
        actual_ret = (actual_price - prev_price) / prev_price * 100.0

        prophet_ret = float(test_row.get("feat_prophet_ret_today_pct", 0.0))
        prophet_price = prev_price * (1.0 + prophet_ret / 100.0)
        prophet_err = abs(prophet_price - actual_price) / actual_price * 100.0

        tr_clean = train_data.iloc[-train_lookback:].dropna(subset=feature_cols).copy()
        y_tr = (tr_clean["close_price"].shift(-1) - tr_clean["close_price"]) / tr_clean["close_price"] * 100.0
        X_tr = tr_clean[feature_cols].iloc[:-1]
        y_tr = y_tr.iloc[:-1]

        # Ridge
        ridge = Ridge(alpha=10.0, random_state=42)
        ridge.fit(X_tr, y_tr)
        pred_ret_ridge = float(ridge.predict(pd.DataFrame([prev_row[feature_cols]]))[0])
        pred_ret_ridge = max(-10.0, min(10.0, pred_ret_ridge))
        ridge_price = prev_price * (1.0 + pred_ret_ridge / 100.0)
        ridge_err = abs(ridge_price - actual_price) / actual_price * 100.0

        # XGBoost
        xgb = XGBRegressor(
            n_estimators=45,
            max_depth=2,
            learning_rate=0.03,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=42,
            n_jobs=1,
        )
        xgb.fit(X_tr, y_tr)
        pred_ret_xgb = float(xgb.predict(pd.DataFrame([prev_row[feature_cols]]))[0])
        pred_ret_xgb = max(-10.0, min(10.0, pred_ret_xgb))
        xgb_price = prev_price * (1.0 + pred_ret_xgb / 100.0)
        xgb_err = abs(xgb_price - actual_price) / actual_price * 100.0

        rows.append({
            "symbol": symbol,
            "variant": variant_name,
            "trade_date": str(test_row["trade_date"]).split(" ")[0],
            "actual_ret": actual_ret,
            "prophet_ret": prophet_ret,
            "pred_ret_ridge": pred_ret_ridge,
            "pred_ret_xgb": pred_ret_xgb,
            "prophet_err": prophet_err,
            "ridge_err": ridge_err,
            "xgb_err": xgb_err,
        })

    return pd.DataFrame(rows)


def score_hits_vectorized(
    preds_df: pd.DataFrame,
    deadband_pct: float,
    mode: str,
) -> dict[str, Any]:
    """Score directional hits across models using vectorized operations."""
    actual = preds_df["actual_ret"].to_numpy()
    ridge = preds_df["pred_ret_ridge"].to_numpy()
    xgb = preds_df["pred_ret_xgb"].to_numpy()
    prophet = preds_df["prophet_ret"].to_numpy()

    def eval_hits(pred_arr: np.ndarray) -> np.ndarray:
        if mode == "current_prod":
            # Current 0.02% threshold logic
            flat_mask = np.abs(actual) <= deadband_pct
            hit_flat = np.abs(pred_arr) <= deadband_pct
            hit_up = (actual > deadband_pct) & (pred_arr > deadband_pct)
            hit_down = (actual < -deadband_pct) & (pred_arr < -deadband_pct)
            return np.where(flat_mask, hit_flat, hit_up | hit_down)

        elif mode == "symmetric":
            # Consolidation classification:
            # If actual is in [-deadband, +deadband], hit if pred is also in [-deadband, +deadband]
            # If actual > deadband, hit if pred > 0
            # If actual < -deadband, hit if pred < 0
            flat_mask = np.abs(actual) <= deadband_pct
            hit_flat = np.abs(pred_arr) <= deadband_pct
            hit_up = (actual > deadband_pct) & (pred_arr > 0.0)
            hit_down = (actual < -deadband_pct) & (pred_arr < 0.0)
            return np.where(flat_mask, hit_flat, hit_up | hit_down)

        elif mode == "tolerant":
            # Tolerant: if flat, hit if predicted flat OR sign matched
            flat_mask = np.abs(actual) <= deadband_pct
            hit_flat = (np.abs(pred_arr) <= deadband_pct) | ((pred_arr * actual) >= 0)
            hit_up = (actual > deadband_pct) & (pred_arr > 0.0)
            hit_down = (actual < -deadband_pct) & (pred_arr < 0.0)
            return np.where(flat_mask, hit_flat, hit_up | hit_down)

        elif mode == "strict_deadband":
            # Both must exceed deadband in same direction, or both in deadband
            flat_mask = np.abs(actual) <= deadband_pct
            hit_flat = np.abs(pred_arr) <= deadband_pct
            hit_up = (actual > deadband_pct) & (pred_arr > deadband_pct)
            hit_down = (actual < -deadband_pct) & (pred_arr < -deadband_pct)
            return np.where(flat_mask, hit_flat, hit_up | hit_down)
        return np.zeros(len(actual), dtype=bool)

    r_hits = eval_hits(ridge)
    x_hits = eval_hits(xgb)
    p_hits = eval_hits(prophet)

    r_hit_count = int(np.sum(r_hits))
    x_hit_count = int(np.sum(x_hits))
    p_hit_count = int(np.sum(p_hits))

    r_mae = float(preds_df["ridge_err"].mean())
    x_mae = float(preds_df["xgb_err"].mean())
    p_mae = float(preds_df["prophet_err"].mean())

    # ML Candidate Selection
    if x_hit_count > r_hit_count:
        ml_model = "XGBoost"
        ml_hits = x_hit_count
        ml_mae = x_mae
    elif r_hit_count > x_hit_count:
        ml_model = "Ridge"
        ml_hits = r_hit_count
        ml_mae = r_mae
    else:
        if x_mae <= r_mae:
            ml_model = "XGBoost"
            ml_hits = x_hit_count
            ml_mae = x_mae
        else:
            ml_model = "Ridge"
            ml_hits = r_hit_count
            ml_mae = r_mae

    return {
        "ml_model": ml_model,
        "ml_hits": ml_hits,
        "ml_hit_rate": (ml_hits / len(preds_df)) * 100.0,
        "ml_mae": ml_mae,
        "ridge_hits": r_hit_count,
        "xgb_hits": x_hit_count,
        "prophet_hits": p_hit_count,
        "prophet_hit_rate": (p_hit_count / len(preds_df)) * 100.0,
        "prophet_mae": p_mae,
    }


def main() -> None:
    db = PostgresManager()
    symbols = ["ASELS", "THYAO", "AKBNK", "EREGL", "TUPRS"]
    feature_variants = define_feature_variants()

    print("Computing 30-day walk-forward predictions for all symbols & variants...")
    all_preds_list: list[pd.DataFrame] = []

    for sym in symbols:
        df = load_data_with_all_features(db, sym)
        for vname, fcols in feature_variants.items():
            preds = precompute_walk_forward_predictions(df, sym, vname, fcols)
            all_preds_list.append(preds)

    full_preds = pd.concat(all_preds_list, ignore_index=True)
    print(f"Generated {len(full_preds)} total prediction rows across {len(symbols)} symbols and {len(feature_variants)} variants.\n")

    # ---------------------------------------------------------
    # PART 1: Compare Feature Variants at Baseline Deadband (0.02%)
    # ---------------------------------------------------------
    print("=" * 85)
    print("PART 1: FEATURE VARIANTS AT CURRENT 0.02% DEADBAND")
    print("=" * 85)

    p1_rows = []
    for (sym, vname), g in full_preds.groupby(["symbol", "variant"]):
        scored = score_hits_vectorized(g, deadband_pct=0.02, mode="current_prod")
        p1_rows.append({
            "symbol": sym,
            "variant": vname,
            **scored,
        })
    df_p1 = pd.DataFrame(p1_rows)

    port_p1 = df_p1.groupby("variant").agg(
        total_ml_hits=("ml_hits", "sum"),
        avg_hit_rate=("ml_hit_rate", "mean"),
        avg_mae=("ml_mae", "mean"),
        prophet_hits=("prophet_hits", "sum"),
    ).reset_index()
    port_p1["portfolio_hit_pct"] = port_p1["total_ml_hits"] / 150.0 * 100.0
    print(port_p1.sort_values(by="portfolio_hit_pct", ascending=False).to_string(index=False))

    print("\nDetailed Per-Symbol Hits (out of 30 sessions):")
    print(df_p1.pivot(index="symbol", columns="variant", values="ml_hits").to_string())

    # ---------------------------------------------------------
    # PART 2: Calibrating Deadband Thresholds & Modes
    # ---------------------------------------------------------
    print("\n" + "=" * 85)
    print("PART 2: DEADBAND THRESHOLD & MODE CALIBRATION (PORTFOLIO WIDE)")
    print("=" * 85)

    deadband_configs = [
        (0.02, "current_prod", "0.02% Current Prod"),
        (0.15, "symmetric", "0.15% Symmetric"),
        (0.20, "symmetric", "0.20% Symmetric"),
        (0.25, "symmetric", "0.25% Symmetric Consolidation"),
        (0.25, "tolerant", "0.25% Tolerant Directional"),
        (0.30, "symmetric", "0.30% Symmetric Consolidation"),
        (0.35, "symmetric", "0.35% Symmetric Consolidation"),
    ]

    p2_rows = []
    for db_pct, db_mode, db_label in deadband_configs:
        for vname in ["V0_Base18", "V1_Shock20", "V2_NetAndGross23", "V3_LeanNet17", "V5_AggNet21"]:
            sub = full_preds[full_preds["variant"] == vname]
            total_hits = 0
            for sym, g in sub.groupby("symbol"):
                scored = score_hits_vectorized(g, deadband_pct=db_pct, mode=db_mode)
                total_hits += scored["ml_hits"]
            p2_rows.append({
                "deadband_label": db_label,
                "variant": vname,
                "total_ml_hits": total_hits,
                "portfolio_hit_rate": (total_hits / 150.0) * 100.0,
            })

    df_p2 = pd.DataFrame(p2_rows)
    p2_pivot = df_p2.pivot(index="deadband_label", columns="variant", values="portfolio_hit_rate")
    print(p2_pivot.to_string())

    # ---------------------------------------------------------
    # PART 3: Deep-Dive on ASELSAN Deadlock Resolution
    # ---------------------------------------------------------
    print("\n" + "=" * 85)
    print("PART 3: ASELSAN DEADLOCK RESOLUTION ANALYSIS")
    print("=" * 85)

    asels_preds = full_preds[full_preds["symbol"] == "ASELS"]
    for vname in ["V0_Base18", "V1_Shock20", "V2_NetAndGross23", "V3_LeanNet17", "V5_AggNet21"]:
        sub_asels = asels_preds[asels_preds["variant"] == vname]
        cur_scored = score_hits_vectorized(sub_asels, deadband_pct=0.02, mode="current_prod")
        sym_scored = score_hits_vectorized(sub_asels, deadband_pct=0.25, mode="symmetric")
        tol_scored = score_hits_vectorized(sub_asels, deadband_pct=0.25, mode="tolerant")

        # Flat vs trending breakdown under 0.25%
        flat_mask = np.abs(sub_asels["actual_ret"]) <= 0.25
        n_flat = int(flat_mask.sum())
        n_trend = int((~flat_mask).sum())

        model_suffix = "xgb" if sym_scored['ml_model'].lower() == "xgboost" else "ridge"
        chosen_col = f"pred_ret_{model_suffix}"
        preds_model = sub_asels[chosen_col].to_numpy()
        acts = sub_asels["actual_ret"].to_numpy()

        flat_hits = int(np.sum((np.abs(acts) <= 0.25) & (np.abs(preds_model) <= 0.25)))
        trend_hits = int(np.sum((acts > 0.25) & (preds_model > 0.0)) + np.sum((acts < -0.25) & (preds_model < 0.0)))

        print(f"ASELSAN [{vname}] (Champion: {sym_scored['ml_model']}):")
        print(f"  At 0.02% Deadband: {cur_scored['ml_hits']}/30 hits ({cur_scored['ml_hit_rate']:.1f}%)")
        print(f"  At 0.25% Symmetric: {sym_scored['ml_hits']}/30 hits ({sym_scored['ml_hit_rate']:.1f}%) -> Flat: {flat_hits}/{n_flat}, Trend: {trend_hits}/{n_trend}")
        print(f"  At 0.25% Tolerant:  {tol_scored['ml_hits']}/30 hits ({tol_scored['ml_hit_rate']:.1f}%)")


if __name__ == "__main__":
    main()
