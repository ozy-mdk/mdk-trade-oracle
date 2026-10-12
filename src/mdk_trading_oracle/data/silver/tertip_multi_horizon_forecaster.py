"""Multi-Horizon Institutional Tertip Machine Learning Forecaster.

Extends the single-day Tertip ML Forecaster across 5 forward market trading windows:
    - 3 Days (N=3)
    - 5 Days (N=5, ~1 market week)
    - 10 Days (N=10, ~2 market weeks)
    - 15 Days (N=15, ~3 market weeks)
    - 30 Days (N=30, ~6 market weeks / 1.5 months)

Mathematical Target:
    For session t and window N (N=3, 5, 10, 15, 30 trading sessions), the target is
    the percentage return from today's closing price P_t to the TERMINAL closing price
    at the end of session t+N:
        P_terminal_{t, N} = Close_{t+N}
        Y_{t, N} = ((P_terminal_{t, N} - Close_t) / Close_t) * 100.0

Zero-Lookahead Temporal Causality:
    At session t, only historical targets where i + N <= t are elapsed and eligible
    for training. Future sessions never leak into training sets.
"""

from __future__ import annotations

import logging
import math
import os
import warnings
from datetime import datetime, timezone
from typing import Any

# Prevent OpenMP / BLAS thread contention and deadlocks on macOS
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.linear_model import BayesianRidge, HuberRegressor, Ridge
from xgboost import XGBRegressor

from mdk_trading_oracle.core.db import PostgresManager
from mdk_trading_oracle.core.logger import get_logger
from mdk_trading_oracle.data.silver.tertip_ml_forecaster import (
    ALL_FIFO_COST_PNL_COLS,
    DEADBAND_PCT,
    YTD_FEATURE_COLS,
    calibrate_expected_return,
    extract_3pillar_time_series,
)

logger = get_logger("mdk_oracle.tertip_multi_horizon")


def filter_multi_horizon_features(
    all_feature_cols: list[str],
    feature_mode: str = "FIFO_CORE",
) -> list[str]:
    """Filter candidate multi-horizon features based on institutional feature mode."""
    mode = str(feature_mode).upper()
    if mode == "FIFO_CORE":
        return [c for c in all_feature_cols if c not in YTD_FEATURE_COLS]
    elif mode == "YTD_SUBSTITUTION":
        return [c for c in all_feature_cols if c not in ALL_FIFO_COST_PNL_COLS]
    elif mode == "FIFO_PLUS_YTD":
        return list(all_feature_cols)
    return [c for c in all_feature_cols if c not in YTD_FEATURE_COLS]

# Suppress verbose warnings from third-party math packages
warnings.filterwarnings("ignore")
logging.getLogger("lightgbm").setLevel(logging.WARNING)

# Horizon Definitions (strictly BIST trading sessions, skipping weekends/holidays)
HORIZON_MAP: dict[str, int] = {
    "3d": 3,
    "5d": 5,
    "10d": 10,
    "15d": 15,
    "30d": 30,
}

LOOKBACK_MAP: dict[str, int] = {
    "3m": 63,
    "6m": 126,
    "12m": 252,
    "18m": 378,
    "24m": 504,
}

# 3 Market Months Hard Stop Threshold (63 trading sessions)
MIN_TRAIN_SESSIONS: int = 63


DEFAULT_CHAMPIONS: dict[str, dict[str, Any]] = {
    "3d": {"model": "LightGBM", "horizon": "6m", "lookback": 126},
    "5d": {"model": "XGBoost", "horizon": "12m", "lookback": 252},
    "10d": {"model": "BayesianRidge", "horizon": "12m", "lookback": 252},
    "15d": {"model": "Ridge", "horizon": "18m", "lookback": 378},
    "30d": {"model": "BayesianRidge", "horizon": "24m", "lookback": 504},
}

# In-memory cache for fast UI serving (TTL 24 hours)
_MULTI_HORIZON_CACHE: dict[str, tuple[float, dict[str, Any]]] = {}
CACHE_TTL_SECONDS = 86400.0


def clear_multi_horizon_cache() -> None:
    """Clear in-memory forecast cache."""
    global _MULTI_HORIZON_CACHE
    _MULTI_HORIZON_CACHE.clear()
    logger.info("Cleared Multi-Horizon Forecaster in-memory cache.")


def get_horizon_deadband(n_days: int) -> float:
    """Calibrate consolidation deadband proportionately with time horizon volatility."""
    return float(round(min(1.50, 0.25 * math.sqrt(n_days)), 2))


def compute_forward_terminal_return(
    close_prices: pd.Series,
    n_days: int,
) -> tuple[pd.Series, pd.Series]:
    """Compute forward N-day terminal price and percentage return (P_{t+N} - P_t) / P_t * 100.

    Parameters
    ----------
    close_prices : pd.Series
        Series of historical close prices sorted chronologically ascending.
    n_days : int
        Number of forward trading sessions (e.g. 3, 5, 10, 15, 30).

    Returns
    -------
    tuple[pd.Series, pd.Series]
        (fwd_terminal_prices, fwd_terminal_returns_pct)
        Values will be NaN for the last n_days rows where future prices do not exist yet.
    """
    if len(close_prices) < n_days:
        nans = pd.Series([np.nan] * len(close_prices), index=close_prices.index)
        return nans, nans

    fwd_terminal_price = close_prices.shift(-n_days)
    fwd_terminal_ret = (fwd_terminal_price - close_prices) / close_prices * 100.0
    return fwd_terminal_price, fwd_terminal_ret


# Backwards compatibility alias
compute_forward_average_return = compute_forward_terminal_return


def engineer_multi_horizon_features(
    df: pd.DataFrame,
    feature_mode: str = "FIFO_CORE",
) -> tuple[pd.DataFrame, list[str]]:
    """Enrich 3-pillar time series with multi-horizon momentum, trend alignment,
    and cumulative institutional flow features (strictly zero data leakage).
    """
    if "feat_close_ret_3d" in df.columns and "feat_mlb_ewma_flow_share_5d" in df.columns:
        feature_cols = [c for c in df.columns if c.startswith("feat_")]
        return df, filter_multi_horizon_features(feature_cols, feature_mode=feature_mode)

    df = df.copy()

    # 1. Multi-Day Trailing Price Momentum (%)
    df["feat_close_ret_3d"] = ((df["close_price"] - df["close_price"].shift(3)) / df["close_price"].shift(3) * 100.0).fillna(0.0)
    df["feat_close_ret_5d"] = ((df["close_price"] - df["close_price"].shift(5)) / df["close_price"].shift(5) * 100.0).fillna(0.0)
    df["feat_close_ret_10d"] = ((df["close_price"] - df["close_price"].shift(10)) / df["close_price"].shift(10) * 100.0).fillna(0.0)
    df["feat_close_ret_20d"] = ((df["close_price"] - df["close_price"].shift(20)) / df["close_price"].shift(20) * 100.0).fillna(0.0)

    # 2. Moving Average Trend Slopes & Distance (%)
    sma10 = df["close_price"].rolling(10, min_periods=5).mean()
    sma20 = df["close_price"].rolling(20, min_periods=10).mean()
    sma50 = df["close_price"].rolling(50, min_periods=20).mean()
    df["feat_price_vs_sma10_pct"] = ((df["close_price"] - sma10) / sma10 * 100.0).fillna(0.0)
    df["feat_price_vs_sma20_pct"] = ((df["close_price"] - sma20) / sma20 * 100.0).fillna(0.0)
    df["feat_price_vs_sma50_pct"] = ((df["close_price"] - sma50) / sma50 * 100.0).fillna(0.0)
    df["feat_sma10_vs_sma20_pct"] = ((sma10 - sma20) / sma20 * 100.0).fillna(0.0)
    df["feat_sma20_vs_sma50_pct"] = ((sma20 - sma50) / sma50 * 100.0).fillna(0.0)

    # 3. Cumulative Multi-Day BofA & Institutional Net Flow Shares (%)
    turnover_3d = df["total_turnover_tl"].rolling(3, min_periods=1).sum().replace(0, np.nan)
    turnover_5d = df["total_turnover_tl"].rolling(5, min_periods=1).sum().replace(0, np.nan)
    turnover_10d = df["total_turnover_tl"].rolling(10, min_periods=1).sum().replace(0, np.nan)

    mlb_3d_flow = df["mlb_flow"].rolling(3, min_periods=1).sum()
    df["feat_mlb_flow_3d_share"] = (mlb_3d_flow / turnover_3d * 100.0).fillna(0.0)

    mlb_5d_flow = df["mlb_flow"].rolling(5, min_periods=1).sum()
    df["feat_mlb_flow_5d_share"] = (mlb_5d_flow / turnover_5d * 100.0).fillna(0.0)

    mlb_10d_flow = df["mlb_flow"].rolling(10, min_periods=1).sum()
    df["feat_mlb_flow_10d_share"] = (mlb_10d_flow / turnover_10d * 100.0).fillna(0.0)

    big5_5d_flow = df["big5_flow"].rolling(5, min_periods=1).sum()
    df["feat_big5_flow_5d_share"] = (big5_5d_flow / turnover_5d * 100.0).fillna(0.0)
    big5_10d_flow = df["big5_flow"].rolling(10, min_periods=1).sum()

    inst_5d_flow = mlb_5d_flow + big5_5d_flow
    df["feat_total_inst_5d_share"] = (inst_5d_flow / turnover_5d * 100.0).fillna(0.0)

    # Multi-Day Institutional Concordance & Flow Z-Scores
    df["feat_inst_concordance_5d"] = (np.sign(mlb_5d_flow) * np.sign(big5_5d_flow)).fillna(0.0)
    df["feat_inst_concordance_10d"] = (np.sign(mlb_10d_flow) * np.sign(big5_10d_flow)).fillna(0.0)

    inst_10d_flow = mlb_10d_flow + big5_10d_flow
    inst_10d_mean = inst_10d_flow.rolling(30, min_periods=10).mean()
    inst_10d_std = inst_10d_flow.rolling(30, min_periods=10).std().replace(0, np.nan)
    df["feat_inst_10d_flow_zscore"] = ((inst_10d_flow - inst_10d_mean) / inst_10d_std).clip(-4.0, 4.0).fillna(0.0)

    # 3b. Flow Divergences & Momentum Rollover (Break Regime Lag at Tops/Bottoms)
    inst_10d_flow_share = (inst_10d_flow / turnover_10d * 100.0).fillna(0.0)
    df["feat_mlb_flow_divergence_3d_10d"] = (df["feat_mlb_flow_3d_share"] - df["feat_mlb_flow_10d_share"]).clip(-100.0, 100.0).fillna(0.0)
    df["feat_inst_flow_divergence_5d_10d"] = (df["feat_total_inst_5d_share"] - inst_10d_flow_share).clip(-100.0, 100.0).fillna(0.0)
    df["feat_ret_rollover_3d_10d"] = (df["feat_close_ret_3d"] - df["feat_close_ret_10d"]).clip(-50.0, 50.0).fillna(0.0)
    df["feat_ret_rollover_5d_20d"] = (df["feat_close_ret_5d"] - df["feat_close_ret_20d"]).clip(-50.0, 50.0).fillna(0.0)

    # 3c. Exponentially-Decayed Institutional Flow Share (EWMA 5-Day)
    ewma_to = df["total_turnover_tl"].ewm(span=5, min_periods=1).mean().replace(0, np.nan)
    ewma_mlb = df["mlb_flow"].ewm(span=5, min_periods=1).mean()
    df["feat_mlb_ewma_flow_share_5d"] = (ewma_mlb / ewma_to * 100.0).clip(-100.0, 100.0).fillna(0.0)

    # 4. Multi-Day Realized Volatility & 14-Day ATR (%)
    df["feat_stock_vol_10d"] = (df["daily_return_pct"].rolling(10, min_periods=5).std() * np.sqrt(252) * 100.0).fillna(0.0)
    df["feat_stock_vol_20d"] = (df["daily_return_pct"].rolling(20, min_periods=10).std() * np.sqrt(252) * 100.0).fillna(0.0)

    if "high_price" in df.columns and "low_price" in df.columns:
        prev_close_series = df["close_price"].shift(1)
        tr_hl = (df["high_price"] - df["low_price"]).abs()
        tr_hpc = (df["high_price"] - prev_close_series).abs()
        tr_lpc = (df["low_price"] - prev_close_series).abs()
        tr_d = pd.concat([tr_hl, tr_hpc, tr_lpc], axis=1).max(axis=1)
        df["atr_14d_pct"] = ((tr_d.rolling(14, min_periods=5).mean() / df["close_price"] * 100.0).clip(0.5, 15.0).fillna(2.2))
    else:
        df["atr_14d_pct"] = 2.2

    feature_cols = [c for c in df.columns if c.startswith("feat_")]
    return df, filter_multi_horizon_features(feature_cols, feature_mode=feature_mode)


def _check_hit(
    pred: float,
    actual: float,
    tol_delta: float = 0.25,
    deadband: float = DEADBAND_PCT,
) -> bool:
    """Evaluate directional correctness against market deadband and tolerance."""
    if abs(pred - actual) <= tol_delta:
        return True
    if abs(actual) <= deadband:
        return (abs(pred) <= deadband) or ((pred * actual) >= 0.0)
    elif actual > deadband:
        return pred > 0.0
    else:
        return pred < 0.0


def _build_model(model_name: str, seed: int = 42) -> Any:
    """Instantiate candidate regression model."""
    if model_name == "Ridge":
        return Ridge(alpha=2.0, random_state=seed)
    elif model_name == "XGBoost":
        return XGBRegressor(
            n_estimators=50,
            max_depth=3,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=seed,
            n_jobs=1,
            nthread=1,
            tree_method="hist",
            max_bin=64,
        )
    elif model_name == "LightGBM":
        return lgb.LGBMRegressor(
            n_estimators=50,
            max_depth=3,
            learning_rate=0.05,
            num_leaves=11,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=seed,
            n_jobs=1,
            verbosity=-1,
            max_bin=64,
        )
    elif model_name == "BayesianRidge":
        return BayesianRidge()
    elif model_name == "Huber":
        return HuberRegressor(max_iter=300, alpha=1.0)
    else:
        return Ridge(alpha=2.0, random_state=seed)


def _compute_sample_weights(y: np.ndarray | pd.Series, deadband: float = 0.50) -> np.ndarray:
    """Piecewise volatility calibration sample weights adapted for multi-day returns."""
    abs_y = np.abs(y)
    return np.where(
        abs_y <= deadband,
        0.50,
        np.where(
            abs_y <= deadband * 2.5,
            3.00,
            5.50,
        ),
    )


def evaluate_horizon_walk_forward(
    df: pd.DataFrame,
    horizon_key: str,
    model_name: str,
    train_lookback_sessions: int = 126,
    n_eval_sessions: int = 30,
    feature_mode: str = "FIFO_CORE",
) -> dict[str, Any]:
    """Run a zero-lookahead walk-forward evaluation for a single horizon and candidate model.

    Temporal Integrity:
        At evaluation session t, candidate training targets are restricted to
        samples i where i + n_days <= t. This ensures zero data leakage.
    """
    n_days = HORIZON_MAP.get(horizon_key, 3)
    df, f_cols = engineer_multi_horizon_features(df, feature_mode=feature_mode)
    deadband = get_horizon_deadband(n_days)

    fwd_avg_price_col = f"fwd_avg_price_{horizon_key}"
    fwd_avg_ret_col = f"fwd_avg_ret_{horizon_key}"

    if fwd_avg_ret_col not in df.columns:
        p_term, r_term = compute_forward_terminal_return(df["close_price"], n_days)
        df[fwd_avg_price_col] = p_term
        df[fwd_avg_ret_col] = r_term

    n_total = len(df)
    max_eval_idx = n_total - n_days - 1
    if max_eval_idx < MIN_TRAIN_SESSIONS:
        return {
            "status": "insufficient_data",
            "model": model_name,
            "horizon": horizon_key,
            "hit_rate_pct": 0.0,
            "hits": 0,
            "total_evals": 0,
            "mae_pct": 999.0,
            "loss": 999.0,
            "sessions_skipped_insufficient": 0,
            "sessions_with_partial_history": 0,
            "sessions_with_full_history": 0,
            "data_sufficiency_pct": 0.0,
            "ledger": [],
        }

    # Pre-extract contiguous NumPy arrays for C-speed model fitting
    X_matrix = np.ascontiguousarray(df[f_cols].to_numpy(dtype=np.float64))
    y_vector = np.ascontiguousarray(df[fwd_avg_ret_col].to_numpy(dtype=np.float64))
    close_vector = np.ascontiguousarray(df["close_price"].to_numpy(dtype=np.float64))
    fwd_price_vector = np.ascontiguousarray(df[fwd_avg_price_col].to_numpy(dtype=np.float64))
    atr_vector = (
        np.ascontiguousarray(df["atr_14d_pct"].to_numpy(dtype=np.float64))
        if "atr_14d_pct" in df.columns
        else np.full(n_total, 2.2, dtype=np.float64)
    )
    trade_dates = [str(d)[:10] for d in df["trade_date"].tolist()]

    # Identify candidate evaluation sessions over requested retrospective window
    target_start_eval_idx = max(0, max_eval_idx - n_eval_sessions + 1)
    candidate_indices = list(range(target_start_eval_idx, max_eval_idx + 1))

    ledger = []
    hits = 0
    hit_errors = []
    miss_errors = []
    all_errors = []
    sessions_skipped_insufficient = 0
    sessions_with_partial_history = 0
    sessions_with_full_history = 0
    actual_sessions_list = []

    for t_idx in candidate_indices:
        trade_date = trade_dates[t_idx]
        current_price = float(close_vector[t_idx])

        target_start_date = trade_dates[t_idx + 1] if t_idx + 1 < n_total else trade_date
        target_end_date = trade_dates[t_idx + n_days] if t_idx + n_days < n_total else trade_date

        actual_avg_price = float(fwd_price_vector[t_idx])
        actual_ret = float(y_vector[t_idx])

        # Strict zero-lookahead: completed historical rows prior to session t_idx
        train_end_idx = t_idx - n_days
        available_history_count = train_end_idx + 1

        # ── 3-MONTH HARD STOP THRESHOLD ───────────────────────────────────────
        if available_history_count < MIN_TRAIN_SESSIONS:
            sessions_skipped_insufficient += 1
            continue

        actual_sessions_used = min(available_history_count, train_lookback_sessions)
        if actual_sessions_used < train_lookback_sessions:
            sufficiency_status = "PARTIAL_HISTORY"
            sessions_with_partial_history += 1
        else:
            sufficiency_status = "FULL"
            sessions_with_full_history += 1

        actual_sessions_list.append(actual_sessions_used)

        train_start_idx = train_end_idx - actual_sessions_used + 1
        X_train = X_matrix[train_start_idx : train_end_idx + 1]
        y_train_raw = y_vector[train_start_idx : train_end_idx + 1]

        valid_mask = ~np.isnan(y_train_raw)
        if not np.all(valid_mask):
            X_train = X_train[valid_mask]
            y_train_raw = y_train_raw[valid_mask]

        if len(y_train_raw) < 15:
            continue

        # Zero-drift target demeaning: prevents trailing bull/bear momentum bias from skewing swing forecasts
        y_train = y_train_raw - float(np.mean(y_train_raw))
        weights = _compute_sample_weights(y_train_raw, deadband=deadband)

        model = _build_model(model_name)
        try:
            if hasattr(model, "fit"):
                if "sample_weight" in model.fit.__code__.co_varnames:
                    model.fit(X_train, y_train, sample_weight=weights)
                else:
                    model.fit(X_train, y_train)

            X_test = X_matrix[t_idx : t_idx + 1]
            raw_pred_ret = float(model.predict(X_test)[0])
            t_atr = float(atr_vector[t_idx]) if not np.isnan(atr_vector[t_idx]) else 2.2
            horizon_atr = t_atr * math.sqrt(n_days / 2.0)
            pred_ret = calibrate_expected_return(raw_pred_ret, horizon_atr, deadband=deadband)
        except Exception as e:
            logger.debug(f"Error training {model_name} at step {t_idx}: {e}")
            pred_ret = 0.0

        # Bound predicted return safely based on horizon scale
        max_bound = min(25.0, 5.0 * math.sqrt(n_days))
        pred_ret = max(-max_bound, min(max_bound, pred_ret))
        pred_avg_price = current_price * (1.0 + pred_ret / 100.0)

        err_pct = abs(pred_avg_price - actual_avg_price) / actual_avg_price * 100.0
        all_errors.append(err_pct)

        is_hit = _check_hit(pred_ret, actual_ret, tol_delta=0.25, deadband=deadband)
        if is_hit:
            hits += 1
            hit_errors.append(err_pct)
        else:
            miss_errors.append(err_pct)

        pred_dir = "BUY" if pred_ret > deadband else ("SELL" if pred_ret < -deadband else "NEUTRAL")
        actual_dir = "BUY" if actual_ret > deadband else ("SELL" if actual_ret < -deadband else "NEUTRAL")

        ledger.append({
            "trade_date": trade_date,
            "target_date_start": target_start_date,
            "target_date_end": target_end_date,
            "current_price": round(current_price, 2),
            "actual_avg_price": round(actual_avg_price, 2),
            "actual_return_pct": round(actual_ret, 2),
            "pred_avg_price": round(pred_avg_price, 2),
            "pred_return_pct": round(pred_ret, 2),
            "pred_direction": pred_dir,
            "actual_direction": actual_dir,
            "err_pct": round(err_pct, 2),
            "is_hit": is_hit,
            "is_pending": False,
            "actual_training_sessions": actual_sessions_used,
            "target_lookback_sessions": train_lookback_sessions,
            "actual_training_months": round(actual_sessions_used / 21.0, 1),
            "data_sufficiency_status": sufficiency_status,
        })

    total_evals = len(ledger)
    if total_evals > 1:
        eval_weights = np.linspace(0.5, 1.5, total_evals)
        eval_weights /= np.mean(eval_weights)
    else:
        eval_weights = np.ones(total_evals)

    hits_arr = np.array([1.0 if row.get("is_hit") else 0.0 for row in ledger])
    errors_arr = np.array([float(row.get("err_pct", 0.0)) for row in ledger])

    hits = int(np.sum(hits_arr))
    hit_rate_pct = (hits / total_evals * 100.0) if total_evals > 0 else 0.0
    weighted_hit_rate_pct = float(np.sum(hits_arr * eval_weights) / np.sum(eval_weights) * 100.0) if total_evals > 0 else 0.0

    hit_mask = hits_arr == 1.0
    miss_mask = hits_arr == 0.0
    mae_pct = float(np.average(errors_arr, weights=eval_weights)) if total_evals > 0 else 0.0
    hit_mae_pct = float(np.average(errors_arr[hit_mask], weights=eval_weights[hit_mask])) if np.any(hit_mask) else mae_pct
    miss_mae_pct = float(np.average(errors_arr[miss_mask], weights=eval_weights[miss_mask])) if np.any(miss_mask) else mae_pct

    # Recency-weighted 3-Criteria Composite Loss: (100 - weighted_hit_rate) + 1.0 * hit_mae + 2.5 * miss_mae
    composite_loss = (100.0 - weighted_hit_rate_pct) + (1.0 * hit_mae_pct) + (2.5 * miss_mae_pct)
    data_sufficiency_pct = round((sessions_with_full_history / total_evals * 100.0), 1) if total_evals > 0 else 0.0

    return {
        "status": "success",
        "model": model_name,
        "horizon": horizon_key,
        "lookback_sessions": train_lookback_sessions,
        "hits": hits,
        "total_evals": total_evals,
        "eval_sessions_requested": n_eval_sessions,
        "sessions_skipped_insufficient": sessions_skipped_insufficient,
        "sessions_with_partial_history": sessions_with_partial_history,
        "sessions_with_full_history": sessions_with_full_history,
        "min_training_sessions_used": min(actual_sessions_list) if actual_sessions_list else 0,
        "max_training_sessions_used": max(actual_sessions_list) if actual_sessions_list else 0,
        "data_sufficiency_pct": data_sufficiency_pct,
        "hit_rate_pct": round(hit_rate_pct, 1),
        "mae_pct": round(mae_pct, 2),
        "hit_mae_pct": round(hit_mae_pct, 2),
        "miss_mae_pct": round(miss_mae_pct, 2),
        "loss": round(composite_loss, 2),
        "feature_mode": feature_mode,
        "ledger": ledger,
    }


def generate_live_horizon_forecast(
    df: pd.DataFrame,
    horizon_key: str,
    champion_model: str,
    train_lookback_sessions: int,
    feature_mode: str = "FIFO_CORE",
) -> dict[str, Any]:
    """Fit crowned champion model on all available history and produce live upcoming forecast.

    Zero-lookahead & 3-Month Hard Stop:
        Training data is strictly limited to completed sessions (rows i where i + n_days <= T).
        If completed sessions are fewer than 3 months (63 sessions), live predictions are hard-stopped.
    """
    n_days = HORIZON_MAP.get(horizon_key, 3)
    df, f_cols = engineer_multi_horizon_features(df, feature_mode=feature_mode)
    deadband = get_horizon_deadband(n_days)

    fwd_avg_price_col = f"fwd_avg_price_{horizon_key}"
    fwd_avg_ret_col = f"fwd_avg_ret_{horizon_key}"

    if fwd_avg_ret_col not in df.columns:
        p_term, r_term = compute_forward_terminal_return(df["close_price"], n_days)
        df[fwd_avg_price_col] = p_term
        df[fwd_avg_ret_col] = r_term

    n_total = len(df)
    latest_row = df.iloc[-1]
    as_of_date = str(latest_row["trade_date"])[:10]
    current_price = float(latest_row["close_price"])

    # Completed historical sessions strictly prior to live horizon
    train_end_idx = n_total - n_days - 1
    available_history_count = train_end_idx + 1

    # ── 3-MONTH HARD STOP THRESHOLD ───────────────────────────────────────────
    if available_history_count < MIN_TRAIN_SESSIONS:
        return {
            "status": "insufficient_history",
            "error": f"Fewer than 3 months of completed trading history ({available_history_count}/63 sessions). Live forecast halted.",
            "horizon": horizon_key,
            "horizon_days": n_days,
            "as_of_date": as_of_date,
            "current_price": round(current_price, 2),
            "target_price": round(current_price, 2),
            "expected_return_pct": 0.0,
            "price_low": round(current_price, 2),
            "price_high": round(current_price, 2),
            "stance": "NEUTRAL",
            "conviction": "NEUTRAL",
            "playbook": "INSUFFICIENT_HISTORY",
            "champion_model": champion_model,
            "training_lookback_sessions": train_lookback_sessions,
            "actual_training_sessions": available_history_count,
            "actual_training_months": round(available_history_count / 21.0, 1),
            "target_training_months": round(train_lookback_sessions / 21.0, 1),
            "actual_window_desc": f"{available_history_count} sessions (<3M minimum threshold)",
            "data_sufficiency_status": "INSUFFICIENT",
        }

    actual_sessions_used = min(available_history_count, train_lookback_sessions)
    actual_months = round(actual_sessions_used / 21.0, 1)
    target_months = round(train_lookback_sessions / 21.0, 1)

    if actual_sessions_used >= train_lookback_sessions:
        sufficiency_status = "FULL"
        actual_window_desc = f"{actual_sessions_used} sessions (~{actual_months}M full lookback)"
    else:
        sufficiency_status = "PARTIAL_ADAPTED"
        actual_window_desc = f"{actual_sessions_used} sessions (~{actual_months}M / {target_months}M target)"

    train_start_idx = train_end_idx - actual_sessions_used + 1
    df_train_sub = df.iloc[train_start_idx : train_end_idx + 1].dropna(
        subset=f_cols + [fwd_avg_ret_col]
    )

    if len(df_train_sub) < 15:
        expected_ret = 0.0
    else:
        X_train = df_train_sub[f_cols]
        y_train_raw = df_train_sub[fwd_avg_ret_col]
        # Zero-drift target demeaning: isolates idiosyncratic institutional alpha without obsolete secular drift
        y_train = y_train_raw - y_train_raw.mean()
        weights = _compute_sample_weights(y_train_raw, deadband=deadband)

        model = _build_model(champion_model)
        try:
            if hasattr(model, "fit"):
                if "sample_weight" in model.fit.__code__.co_varnames:
                    model.fit(X_train, y_train, sample_weight=weights)
                else:
                    model.fit(X_train, y_train)

            X_live = pd.DataFrame([latest_row[f_cols]])
            raw_exp_ret = float(model.predict(X_live)[0])
            latest_atr = float(latest_row.get("atr_14d_pct", 2.2))
            horizon_atr = latest_atr * math.sqrt(n_days / 2.0)
            expected_ret = calibrate_expected_return(raw_exp_ret, horizon_atr, deadband=deadband)
        except Exception as e:
            logger.warning(f"Live forecast fit failed for {champion_model} ({horizon_key}): {e}")
            expected_ret = 0.0

    max_bound = min(25.0, 5.0 * math.sqrt(n_days))
    expected_ret = max(-max_bound, min(max_bound, expected_ret))
    target_price = current_price * (1.0 + expected_ret / 100.0)

    # Price range envelope using 20-day historical return volatility scaled to horizon
    vol_20d = float(df["feat_ret_today_pct"].iloc[-20:].std()) if len(df) >= 20 else 2.0
    horizon_vol = max(1.0, vol_20d * math.sqrt(n_days / 5.0))
    price_low = target_price * (1.0 - (1.645 * horizon_vol / 100.0))
    price_high = target_price * (1.0 + (1.645 * horizon_vol / 100.0))

    # Stance & Conviction
    if expected_ret > deadband:
        stance = "BULLISH"
    elif expected_ret < -deadband:
        stance = "BEARISH"
    else:
        stance = "NEUTRAL"

    if expected_ret >= 3.0:
        conviction = "STRONG_BUY"
    elif expected_ret >= max(1.0, deadband):
        conviction = "BUY"
    elif expected_ret > deadband:
        conviction = "WEAK_BUY"
    elif expected_ret <= -3.0:
        conviction = "STRONG_SELL"
    elif expected_ret <= -max(1.0, deadband):
        conviction = "SELL"
    elif expected_ret < -deadband:
        conviction = "WEAK_SELL"
    else:
        conviction = "NEUTRAL"

    # Multi-Horizon Tactical Playbook Formulation
    if expected_ret >= 2.5:
        playbook = "EXPANSION_ACCUMULATION"
    elif expected_ret >= deadband:
        playbook = "SWING_LONG_MOMENTUM"
    elif expected_ret <= -2.5:
        playbook = "INSTITUTIONAL_DISTRIBUTION_FADE"
    elif expected_ret <= -deadband:
        playbook = "TACTICAL_SHORT_ROTATION"
    else:
        playbook = "RANGE_BOUND_CONSOLIDATION"

    return {
        "horizon": horizon_key,
        "horizon_days": n_days,
        "as_of_date": as_of_date,
        "current_price": round(current_price, 2),
        "target_price": round(target_price, 2),
        "expected_return_pct": round(expected_ret, 2),
        "price_low": round(price_low, 2),
        "price_high": round(price_high, 2),
        "stance": stance,
        "conviction": conviction,
        "playbook": playbook,
        "champion_model": champion_model,
        "training_lookback_sessions": train_lookback_sessions,
        "actual_training_sessions": actual_sessions_used,
        "actual_training_months": actual_months,
        "target_training_months": target_months,
        "actual_window_desc": actual_window_desc,
        "data_sufficiency_status": sufficiency_status,
        "feature_mode": feature_mode,
    }


def _clean_champ_name(name: str | None) -> str:
    if not name:
        return "Auto"
    import re
    m = re.search(r"\(([^)]+)\)", str(name))
    if m:
        return m.group(1).strip()
    return re.sub(r"^(?:Crowned|Grand|ML)?\s*Champion\s*", "", str(name), flags=re.IGNORECASE).strip() or str(name)


def get_multi_horizon_forecaster_payload(
    db: PostgresManager,
    symbol: str,
    selected_composition: dict[str, dict[str, Any]] | None = None,
    n_eval_sessions: int = 180,
    force_recompute: bool = False,
    target_date: str | None = None,
) -> dict[str, Any]:
    """Generate comprehensive multi-horizon forecast payload across all 5 windows.

    Returns live forecasts and retrospective backtest ledgers for 3d, 5d, 10d, 15d, and 30d.
    """
    sym = symbol.upper()
    cache_key = f"{sym}_multi_horizon_{n_eval_sessions}_{target_date or 'latest'}"
    now_ts = datetime.now(timezone.utc).timestamp()

    if not force_recompute and cache_key in _MULTI_HORIZON_CACHE:
        cached_ts, cached_data = _MULTI_HORIZON_CACHE[cache_key]
        if (now_ts - cached_ts) < CACHE_TTL_SECONDS:
            return cached_data

    # Fast-path: Check database precomputed gold tables first
    if db is not None and not force_recompute:
        try:
            date_filter = "as_of_date = %s" if target_date else "as_of_date = (SELECT MAX(as_of_date) FROM gold_tertip_multi_horizon_forecasts WHERE symbol = %s)"
            params = [sym, target_date] if target_date else [sym, sym]
            gold_rows = db.query_pl(
                f"""
                SELECT symbol, as_of_date, horizon, horizon_days, current_price, target_price,
                       expected_return_pct, price_low, price_high, stance, conviction, playbook,
                       ml_champion_type, champion_dir_hits, champion_total_evals,
                       champion_dir_hit_rate_pct, champion_mae_pct, training_lookback_sessions,
                       actual_training_sessions, data_sufficiency_status, crowned_horizon,
                       COALESCE(feature_mode, 'FIFO_CORE') AS feature_mode
                FROM gold_tertip_multi_horizon_forecasts
                WHERE symbol = %s AND {date_filter};
                """,
                params=params,
            ).to_dicts()

            if len(gold_rows) >= 5:
                gold_map = {r["horizon"]: r for r in gold_rows}
                as_of_str = str(gold_rows[0]["as_of_date"])
                curr_p = float(gold_rows[0].get("current_price") or 0.0)
                horizons_result = {}

                for h_key in ["3d", "5d", "10d", "15d", "30d"]:
                    g = gold_map.get(h_key, {})
                    c_model = _clean_champ_name(g.get("ml_champion_type"))
                    lb_sess = int(g.get("training_lookback_sessions") or 126)
                    act_sess = int(g.get("actual_training_sessions") or lb_sess)

                    # Backtest ledger up to as_of_date
                    db_rows = []
                    try:
                        db_res = db.query_pl(
                            """
                            SELECT * FROM (
                                SELECT trade_date, target_date_start, target_date_end, current_price,
                                       actual_avg_price, actual_return_pct, pred_avg_price, pred_return_pct,
                                       pred_direction, actual_direction, err_pct, is_hit, is_pending, champion_model,
                                       actual_training_sessions, data_sufficiency_status
                                FROM gold_tertip_multi_horizon_backtests
                                WHERE symbol = %s AND horizon = %s AND trade_date <= %s
                                ORDER BY trade_date DESC
                                LIMIT %s
                            ) sub
                            ORDER BY trade_date ASC;
                            """,
                            params=[sym, h_key, as_of_str, n_eval_sessions],
                        )
                        if len(db_res) > 0:
                            db_rows = db_res.to_dicts()
                    except Exception:
                        db_rows = []

                    completed = [r for r in db_rows if not r.get("is_pending")]
                    hits = sum(1 for r in completed if r.get("is_hit"))
                    total = len(completed)
                    hit_rate = round(hits / max(1, total) * 100.0, 1) if total > 0 else float(g.get("champion_dir_hit_rate_pct") or 0.0)
                    errs = [float(r.get("err_pct", 0.0)) for r in completed]
                    hit_errs = [float(r.get("err_pct", 0.0)) for r in completed if r.get("is_hit")]
                    miss_errs = [float(r.get("err_pct", 0.0)) for r in completed if not r.get("is_hit")]
                    mae = round(float(np.mean(errs)), 2) if errs else float(g.get("champion_mae_pct") or 0.0)
                    h_mae = round(float(np.mean(hit_errs)), 2) if hit_errs else 0.0
                    m_mae = round(float(np.mean(miss_errs)), 2) if miss_errs else 0.0
                    loss = round((100.0 - hit_rate) + 1.0 * h_mae + 2.5 * m_mae, 2)

                    ledger = [
                        {
                            "trade_date": str(r["trade_date"]),
                            "target_date_start": str(r["target_date_start"]),
                            "target_date_end": str(r["target_date_end"]),
                            "current_price": float(r["current_price"] or 0.0),
                            "actual_avg_price": float(r["actual_avg_price"] or 0.0),
                            "actual_return_pct": float(r["actual_return_pct"] or 0.0),
                            "pred_avg_price": float(r["pred_avg_price"] or 0.0),
                            "pred_return_pct": float(r["pred_return_pct"] or 0.0),
                            "pred_direction": r.get("pred_direction") or "NEUTRAL",
                            "actual_direction": r.get("actual_direction") or "NEUTRAL",
                            "err_pct": float(r["err_pct"] or 0.0),
                            "is_hit": bool(r.get("is_hit", False)),
                            "is_pending": bool(r.get("is_pending", False)),
                            "champion_model": r.get("champion_model") or c_model,
                            "actual_training_sessions": r.get("actual_training_sessions") or act_sess,
                            "target_lookback_sessions": lb_sess,
                            "actual_training_months": round((r.get("actual_training_sessions") or act_sess) / 21.0, 1),
                            "data_sufficiency_status": r.get("data_sufficiency_status") or g.get("data_sufficiency_status") or "FULL",
                        }
                        for r in db_rows
                    ]

                    live_fc = {
                        "horizon": h_key,
                        "horizon_days": HORIZON_MAP[h_key],
                        "as_of_date": as_of_str,
                        "current_price": curr_p,
                        "target_price": float(g.get("target_price") or curr_p),
                        "expected_return_pct": float(g.get("expected_return_pct") or 0.0),
                        "price_low": float(g.get("price_low") or curr_p),
                        "price_high": float(g.get("price_high") or curr_p),
                        "direction": g.get("stance") or "NEUTRAL",
                        "stance": g.get("stance") or "NEUTRAL",
                        "conviction": g.get("conviction") or "",
                        "playbook": g.get("playbook") or "",
                        "champion_model": c_model,
                        "crowned_horizon": g.get("crowned_horizon") or "6m",
                        "training_lookback_sessions": lb_sess,
                        "actual_training_sessions": act_sess,
                        "data_sufficiency_status": g.get("data_sufficiency_status") or "FULL",
                        "feature_mode": g.get("feature_mode") or "FIFO_CORE",
                    }

                    horizons_result[h_key] = {
                        "meta": {
                            "horizon": h_key,
                            "horizon_days": HORIZON_MAP[h_key],
                            "crowned_model": c_model,
                            "crowned_horizon": g.get("crowned_horizon") or "6m",
                            "training_lookback_sessions": lb_sess,
                            "feature_mode": g.get("feature_mode") or "FIFO_CORE",
                            "hit_rate_pct": hit_rate,
                            "hits": hits,
                            "total_evals": total,
                            "eval_sessions_requested": len(db_rows),
                            "mae_pct": mae,
                            "hit_mae_pct": h_mae,
                            "miss_mae_pct": m_mae,
                            "loss": loss,
                        },
                        "live_forecast": live_fc,
                        "backtest_ledger": ledger,
                    }

                payload = {
                    "status": "success",
                    "symbol": sym,
                    "as_of_date": as_of_str,
                    "current_price": curr_p,
                    "calculated_at": datetime.now(timezone.utc).isoformat(),
                    "horizons": horizons_result,
                }
                _MULTI_HORIZON_CACHE[cache_key] = (now_ts, payload)
                return payload
        except Exception as e:
            logger.warning(f"Error reading precomputed gold forecasts for {sym}: {e}")

    # Extract 3-pillar time series with 39 stationary features
    df = extract_3pillar_time_series(db, sym, lookback_days=550)
    if target_date and not df.empty:
        df = df[df["trade_date"] <= target_date]
    if df.empty or len(df) < 60:
        return {
            "status": "error",
            "message": f"Insufficient historical data for {sym}",
            "symbol": sym,
            "horizons": {},
        }

    if selected_composition is None:
        from mdk_trading_oracle.data.gold.tertip_multi_horizon import load_crowned_multi_horizon_config
        cfg_all = load_crowned_multi_horizon_config()
        composition = cfg_all.get("symbols", {}).get(sym, DEFAULT_CHAMPIONS)
    else:
        composition = selected_composition

    horizons_result = {}

    for h_key in ["3d", "5d", "10d", "15d", "30d"]:
        cfg = composition.get(h_key, DEFAULT_CHAMPIONS[h_key])
        champ_model = cfg.get("model", "Ridge")
        lb_sessions = cfg.get("lookback", cfg.get("training_lookback_sessions", 126))
        horizon_lb_str = cfg.get("horizon", "6m")
        f_mode = cfg.get("feature_mode", "FIFO_CORE")

        # Check if pre-computed backtest ledger exists in database
        db_rows = []
        if db is not None and not force_recompute:
            try:
                db_res = db.query_pl(
                    """
                    SELECT * FROM (
                        SELECT trade_date, target_date_start, target_date_end, current_price,
                               actual_avg_price, actual_return_pct, pred_avg_price, pred_return_pct,
                               pred_direction, actual_direction, err_pct, is_hit, is_pending, champion_model,
                               actual_training_sessions, data_sufficiency_status
                        FROM gold_tertip_multi_horizon_backtests
                        WHERE symbol = %s AND horizon = %s AND (champion_model = %s OR champion_model IS NULL)
                        ORDER BY trade_date DESC
                        LIMIT %s
                    ) sub
                    ORDER BY trade_date ASC;
                    """,
                    params=[sym, h_key, champ_model, n_eval_sessions],
                )
                if len(db_res) > 0:
                    db_rows = db_res.to_dicts()
            except Exception as e:
                logger.debug(f"Could not load precomputed backtests from DB: {e}")

        if db_rows:
            # Reconstruct bt_eval from persisted records
            completed = [r for r in db_rows if not r.get("is_pending")]
            hits = sum(1 for r in completed if r.get("is_hit"))
            total = len(completed)
            hit_rate = round(hits / max(1, total) * 100.0, 1)
            errs = [float(r.get("err_pct", 0.0)) for r in completed]
            hit_errs = [float(r.get("err_pct", 0.0)) for r in completed if r.get("is_hit")]
            miss_errs = [float(r.get("err_pct", 0.0)) for r in completed if not r.get("is_hit")]
            mae = round(float(np.mean(errs)), 2) if errs else 0.0
            h_mae = round(float(np.mean(hit_errs)), 2) if hit_errs else 0.0
            m_mae = round(float(np.mean(miss_errs)), 2) if miss_errs else 0.0
            loss = round((100.0 - hit_rate) + 1.0 * h_mae + 2.5 * m_mae, 2)

            partial_cnt = sum(1 for r in db_rows if r.get("data_sufficiency_status") == "PARTIAL_HISTORY")
            full_cnt = sum(1 for r in db_rows if r.get("data_sufficiency_status") != "PARTIAL_HISTORY")
            train_sess_list = [r.get("actual_training_sessions") for r in db_rows if r.get("actual_training_sessions")]

            ledger = []
            for r in db_rows:
                act_s = r.get("actual_training_sessions") or lb_sessions
                ledger.append({
                    "trade_date": str(r["trade_date"]),
                    "target_date_start": str(r["target_date_start"]),
                    "target_date_end": str(r["target_date_end"]),
                    "current_price": float(r["current_price"] or 0.0),
                    "actual_avg_price": float(r["actual_avg_price"] or 0.0),
                    "actual_return_pct": float(r["actual_return_pct"] or 0.0),
                    "pred_avg_price": float(r["pred_avg_price"] or 0.0),
                    "pred_return_pct": float(r["pred_return_pct"] or 0.0),
                    "pred_direction": r.get("pred_direction") or "NEUTRAL",
                    "actual_direction": r.get("actual_direction") or "NEUTRAL",
                    "err_pct": float(r["err_pct"] or 0.0),
                    "is_hit": bool(r.get("is_hit", False)),
                    "is_pending": bool(r.get("is_pending", False)),
                    "champion_model": r.get("champion_model") or champ_model,
                    "actual_training_sessions": act_s,
                    "target_lookback_sessions": lb_sessions,
                    "actual_training_months": round(act_s / 21.0, 1),
                    "data_sufficiency_status": r.get("data_sufficiency_status") or "FULL",
                })

            bt_eval = {
                "hit_rate_pct": hit_rate,
                "hits": hits,
                "total_evals": total,
                "eval_sessions_requested": len(db_rows),
                "sessions_skipped_insufficient": 0,
                "sessions_with_partial_history": partial_cnt,
                "sessions_with_full_history": full_cnt,
                "min_training_sessions_used": min(train_sess_list) if train_sess_list else lb_sessions,
                "max_training_sessions_used": max(train_sess_list) if train_sess_list else lb_sessions,
                "data_sufficiency_pct": round(full_cnt / max(1, (full_cnt + partial_cnt)) * 100.0, 1),
                "target_lookback_months": lb_sessions // 21,
                "mae_pct": mae,
                "hit_mae_pct": h_mae,
                "miss_mae_pct": m_mae,
                "loss": loss,
                "ledger": ledger,
            }
        else:
            # Fall back to dynamic walk-forward evaluation
            bt_eval = evaluate_horizon_walk_forward(
                df=df,
                horizon_key=h_key,
                model_name=champ_model,
                train_lookback_sessions=lb_sessions,
                n_eval_sessions=n_eval_sessions,
                feature_mode=f_mode,
            )

        # 2. Generate Live T+1 Upcoming Forecast
        live_fc = generate_live_horizon_forecast(
            df=df,
            horizon_key=h_key,
            champion_model=champ_model,
            train_lookback_sessions=lb_sessions,
            feature_mode=f_mode,
        )

        horizons_result[h_key] = {
            "meta": {
                "horizon": h_key,
                "horizon_days": HORIZON_MAP[h_key],
                "crowned_model": champ_model,
                "crowned_horizon": horizon_lb_str,
                "training_lookback_sessions": lb_sessions,
                "feature_mode": f_mode,
                "hit_rate_pct": bt_eval.get("hit_rate_pct", 0.0),
                "hits": bt_eval.get("hits", 0),
                "total_evals": bt_eval.get("total_evals", 0),
                "eval_sessions_requested": bt_eval.get("eval_sessions_requested", n_eval_sessions),
                "sessions_skipped_insufficient": bt_eval.get("sessions_skipped_insufficient", 0),
                "sessions_with_partial_history": bt_eval.get("sessions_with_partial_history", 0),
                "sessions_with_full_history": bt_eval.get("sessions_with_full_history", 0),
                "min_training_sessions_used": bt_eval.get("min_training_sessions_used", 0),
                "max_training_sessions_used": bt_eval.get("max_training_sessions_used", 0),
                "data_sufficiency_pct": bt_eval.get("data_sufficiency_pct", 100.0),
                "target_lookback_months": bt_eval.get("target_lookback_months", lb_sessions // 21),
                "target_lookback_sessions": lb_sessions,
                "min_required_train_sessions": MIN_TRAIN_SESSIONS,
                "mae_pct": bt_eval.get("mae_pct", 0.0),
                "hit_mae_pct": bt_eval.get("hit_mae_pct", 0.0),
                "miss_mae_pct": bt_eval.get("miss_mae_pct", 0.0),
                "loss": bt_eval.get("loss", 0.0),
            },
            "live_forecast": live_fc,
            "backtest_ledger": bt_eval.get("ledger", []),
        }

    payload = {
        "status": "success",
        "symbol": sym,
        "as_of_date": str(df.iloc[-1]["trade_date"]),
        "current_price": round(float(df.iloc[-1]["close_price"]), 2),
        "calculated_at": datetime.now(timezone.utc).isoformat(),
        "horizons": horizons_result,
    }

    _MULTI_HORIZON_CACHE[cache_key] = (now_ts, payload)
    return payload
