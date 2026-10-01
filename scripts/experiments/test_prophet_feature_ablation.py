"""Isolated Experiment: Prophet Feature Ablation for Tertip ML Forecaster.

Tests adding Prophet rolling baseline forecasts as features into the Tertip ML Forecaster:
- Variant 0 (Baseline): 17 Lean Microstructure Features (Production)
- Variant 1 (+Prophet Today): 17 Features + feat_prophet_ret_today_pct
- Variant 2 (+Prophet Today & Surprise): 17 Features + feat_prophet_ret_today_pct + feat_prophet_surprise_today_pct
- Variant 3 (+Prophet Tomorrow Prior): 17 Features + feat_prophet_ret_tomorrow_pct
- Variant 4 (Full Hybrid): 17 Features + feat_prophet_ret_today_pct + feat_prophet_surprise_today_pct + feat_prophet_ret_tomorrow_pct

Evaluates each configuration on the last 30 trading sessions across benchmark BIST symbols.
Strict zero-lookahead / zero data leakage protocol.
"""

from __future__ import annotations

import logging
import sys
import time
import warnings
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from prophet import Prophet
from sklearn.linear_model import Ridge
from xgboost import XGBRegressor

from mdk_trading_oracle.core.db import PostgresManager
from mdk_trading_oracle.data.silver.tertip_ml_forecaster import (
    FEATURE_COLS,
    TRAIN_LOOKBACK_SESSIONS,
    extract_3pillar_time_series,
    run_30d_walk_forward_arena,
)

# Suppress verbose warnings and cmdstanpy chain output
warnings.filterwarnings("ignore")
logging.getLogger("cmdstanpy").setLevel(logging.ERROR)
logging.getLogger("cmdstanpy").disabled = True
logging.getLogger("prophet").setLevel(logging.ERROR)
logging.getLogger("lightgbm").setLevel(logging.ERROR)
try:
    import cmdstanpy
    cmdstanpy.utils.get_logger().setLevel(logging.ERROR)
    cmdstanpy.utils.get_logger().disabled = True
except Exception:
    pass

CACHE_DIR = Path.home() / "data" / "mdk_oracle" / "cache" / "prophet_ablation"
CACHE_DIR.mkdir(parents=True, exist_ok=True)


def compute_or_load_prophet_rolling_features(
    df: pd.DataFrame,
    symbol: str,
    train_lookback: int = TRAIN_LOOKBACK_SESSIONS,
    n_eval_sessions: int = 30,
    force_recompute: bool = False,
) -> pd.DataFrame:
    """Precompute zero-leakage rolling 1-step ahead Prophet baseline forecasts.

    For each session k in [N - n_eval_sessions - train_lookback - 5, N]:
    - History [k - train_lookback : k] (ends at session k - 1) is used to fit Prophet.
    - Prophet predicts session k's close price: yhat_k.
    - feat_prophet_ret_today_pct = (yhat_k - P_{k-1}) / P_{k-1} * 100.
    - feat_prophet_surprise_today_pct = (P_k - yhat_k) / P_{k-1} * 100.
    - feat_prophet_ret_tomorrow_pct = (yhat_{k+1} - P_k) / P_k * 100.
    """
    sym = symbol.upper()
    cache_file = CACHE_DIR / f"{sym}_prophet_rolling_features.parquet"

    if not force_recompute and cache_file.exists():
        try:
            cached_df = pd.read_parquet(cache_file)
            if len(cached_df) == len(df) and "feat_prophet_ret_today_pct" in cached_df.columns:
                print(f"[{sym}] Loaded cached Prophet rolling features ({len(cached_df)} rows)")
                return cached_df
        except Exception as e:
            print(f"[{sym}] Cache read failed ({e}), recomputing...")

    df_out = df.copy()
    N = len(df_out)
    start_idx = max(train_lookback + 5, N - n_eval_sessions - train_lookback - 5)

    prophet_ret_today = np.zeros(N, dtype=np.float64)
    prophet_surprise_today = np.zeros(N, dtype=np.float64)
    prophet_ret_tomorrow = np.zeros(N, dtype=np.float64)

    print(f"[{sym}] Computing zero-leakage rolling Prophet fits for {N - start_idx} sessions (start idx {start_idx} of {N})...")
    t0 = time.time()

    # Pre-calculate prices and dates
    close_prices = df_out["close_price"].to_numpy()
    trade_dates = pd.to_datetime(df_out["trade_date"])

    for k in range(start_idx, N):
        hist_dates = trade_dates.iloc[k - train_lookback : k]
        hist_prices = close_prices[k - train_lookback : k]
        p_prev = close_prices[k - 1]
        p_today = close_prices[k]
        target_date = trade_dates.iloc[k]

        p_df = pd.DataFrame({"ds": hist_dates, "y": hist_prices})
        m = Prophet(
            daily_seasonality=False,
            weekly_seasonality=False,
            yearly_seasonality=False,
            changepoint_prior_scale=0.05,
        )
        m.fit(p_df)

        # Forecast session k
        target_df = pd.DataFrame({"ds": [target_date]})
        fc = m.predict(target_df)
        yhat_today = float(fc.iloc[-1]["yhat"])

        ret_today_fc = (yhat_today - p_prev) / p_prev * 100.0 if p_prev > 0 else 0.0
        surprise_today = (p_today - yhat_today) / p_prev * 100.0 if p_prev > 0 else 0.0

        prophet_ret_today[k] = np.clip(ret_today_fc, -25.0, 25.0)
        prophet_surprise_today[k] = np.clip(surprise_today, -25.0, 25.0)

        # Print progress every 50 steps
        steps_done = k - start_idx + 1
        if steps_done % 50 == 0 or k == N - 1:
            elapsed = time.time() - t0
            print(f"  [{sym}] {steps_done}/{N - start_idx} done ({elapsed:.1f}s, {elapsed/steps_done:.3f}s/fit)")

    # Shift prophet_ret_today by -1 to get prophet_ret_tomorrow at session k:
    # At session k, prophet_ret_tomorrow is the forecast of session k+1 made using history up to k.
    # Notice that prophet_ret_today[k+1] was calculated using history up to k!
    for k in range(start_idx, N - 1):
        prophet_ret_tomorrow[k] = prophet_ret_today[k + 1]
    prophet_ret_tomorrow[N - 1] = prophet_ret_today[N - 1]  # edge fallback

    df_out["feat_prophet_ret_today_pct"] = prophet_ret_today
    df_out["feat_prophet_surprise_today_pct"] = prophet_surprise_today
    df_out["feat_prophet_ret_tomorrow_pct"] = prophet_ret_tomorrow

    # Cache to disk
    try:
        df_out.to_parquet(cache_file)
        print(f"[{sym}] Successfully saved Prophet rolling features cache to {cache_file}")
    except Exception as e:
        print(f"[{sym}] Warning: failed to save cache ({e})")

    return df_out


def run_experiment_ablation(
    symbols: list[str],
    n_sessions: int = 30,
) -> dict[str, Any]:
    """Run full ablation across symbols and compare variants."""
    db = PostgresManager()

    variants = {
        "V0_Baseline_17": list(FEATURE_COLS),
        "V1_Prophet_Today": list(FEATURE_COLS) + ["feat_prophet_ret_today_pct"],
        "V2_Prophet_Today_Surprise": list(FEATURE_COLS) + ["feat_prophet_ret_today_pct", "feat_prophet_surprise_today_pct"],
        "V3_Prophet_Tomorrow_Prior": list(FEATURE_COLS) + ["feat_prophet_ret_tomorrow_pct"],
        "V4_Full_Hybrid": list(FEATURE_COLS) + [
            "feat_prophet_ret_today_pct",
            "feat_prophet_surprise_today_pct",
            "feat_prophet_ret_tomorrow_pct",
        ],
    }

    results: dict[str, dict[str, Any]] = {}

    for sym in symbols:
        print("\n=======================================================")
        print(f"  ANALYZING SYMBOL: {sym}")
        print("=======================================================")

        df = extract_3pillar_time_series(db, sym, lookback_days=550)
        if df.empty or len(df) < TRAIN_LOOKBACK_SESSIONS + n_sessions + 10:
            print(f"[{sym}] Insufficient history ({len(df)} rows), skipping.")
            continue

        # Precalculate zero-leakage Prophet rolling features
        df_enriched = compute_or_load_prophet_rolling_features(df, sym, n_eval_sessions=n_sessions)

        sym_results: dict[str, Any] = {}

        for var_name, f_cols in variants.items():
            print(f"  -> Testing {var_name} ({len(f_cols)} features)...")
            ledger, summary, model = run_30d_walk_forward_arena(
                df_enriched,
                n_sessions=n_sessions,
                model_type="auto",
                feature_cols=f_cols,
                train_lookback_sessions=TRAIN_LOOKBACK_SESSIONS,
            )

            sym_results[var_name] = {
                "dir_hits": summary.get("champion_dir_hits", 0),
                "dir_hit_rate_pct": summary.get("champion_dir_hit_rate_pct", 0.0),
                "mae_pct": summary.get("champion_mae_pct", 0.0),
                "ml_champion_type": summary.get("ml_champion_type", "N/A"),
                "ml_dir_hits": summary.get("ml_dir_hits", 0),
                "ml_dir_hit_rate_pct": summary.get("ml_dir_hit_rate_pct", 0.0),
                "ml_mae_pct": summary.get("ml_mae_pct", 0.0),
                "prophet_dir_hits": summary.get("prophet_dir_hits", 0),
                "prophet_dir_hit_rate_pct": summary.get("prophet_dir_hit_rate_pct", 0.0),
                "prophet_mae_pct": summary.get("prophet_mae_pct", 0.0),
                "grand_champion": summary.get("champion", "N/A"),
                "ml_wins": summary.get("ml_wins", 0),
                "prophet_wins": summary.get("prophet_wins", 0),
                "total_sessions": len(ledger),
            }

            # Inspect model weights / importance if available
            if model is not None:
                if isinstance(model, Ridge):
                    coefs = dict(zip(f_cols, model.coef_))
                    prophet_coefs = {k: round(float(v), 4) for k, v in coefs.items() if "prophet" in k}
                    sym_results[var_name]["prophet_weights"] = prophet_coefs
                elif isinstance(model, XGBRegressor):
                    importances = dict(zip(f_cols, model.feature_importances_))
                    prophet_imp = {k: round(float(v), 4) for k, v in importances.items() if "prophet" in k}
                    sym_results[var_name]["prophet_importances"] = prophet_imp

        results[sym] = sym_results

    return results


def print_summary_table(results: dict[str, dict[str, Any]]) -> None:
    """Format and display comparative benchmark tables."""
    print("\n" + "=" * 90)
    print("                     PROPHET FEATURE ABLATION - 30-DAY ARENA RESULTS")
    print("=" * 90)

    symbols = list(results.keys())
    if not symbols:
        print("No results found.")
        return

    variants = list(next(iter(results.values())).keys())

    # 1. Directional Hit Rate %
    print("\n[DIRECTIONAL HIT RATE % (Last 30 Sessions)]")
    header = f"{'Symbol':<10}" + "".join(f"{v:<16}" for v in variants)
    print(header)
    print("-" * len(header))

    avg_hits: dict[str, list[float]] = {v: [] for v in variants}
    avg_mae: dict[str, list[float]] = {v: [] for v in variants}

    for sym, var_res in results.items():
        row = f"{sym:<10}"
        for v in variants:
            res = var_res.get(v, {})
            rate = res.get("ml_dir_hit_rate_pct", 0.0)
            hits = res.get("ml_dir_hits", 0)
            avg_hits[v].append(rate)
            row += f"{hits}/30 ({rate:.1f}%)   "
        print(row)

    print("-" * len(header))
    avg_row = f"{'PORTFOLIO':<10}"
    for v in variants:
        mean_rate = np.mean(avg_hits[v]) if avg_hits[v] else 0.0
        avg_row += f"{mean_rate:.1f}%          "
    print(avg_row)

    # 2. Mean Absolute Error (MAE %)
    print("\n[MEAN ABSOLUTE ERROR (MAE %) (Last 30 Sessions)]")
    print(header)
    print("-" * len(header))

    for sym, var_res in results.items():
        row = f"{sym:<10}"
        for v in variants:
            res = var_res.get(v, {})
            mae = res.get("ml_mae_pct", 0.0)
            avg_mae[v].append(mae)
            row += f"{mae:.2f}%          "
        print(row)

    print("-" * len(header))
    avg_mae_row = f"{'PORTFOLIO':<10}"
    for v in variants:
        mean_mae = np.mean(avg_mae[v]) if avg_mae[v] else 0.0
        avg_mae_row += f"{mean_mae:.2f}%          "
    print(avg_mae_row)

    # 3. Model & Feature Details
    print("\n[MODEL SELECTION & PROPHET FEATURE WEIGHTS]")
    for sym, var_res in results.items():
        print(f"\nSymbol: {sym}")
        for v in variants:
            res = var_res.get(v, {})
            m_type = res.get("ml_champion_type", "N/A")
            g_champ = res.get("grand_champion", "N/A")
            p_imp = res.get("prophet_importances") or res.get("prophet_weights") or {}
            print(f"  {v:<28} | Model: {m_type:<8} | Grand Champion: {g_champ:<22} | Prophet Feature Imp/Weight: {p_imp}")

    print("\n" + "=" * 90)


if __name__ == "__main__":
    test_symbols = ["THYAO", "AKBNK", "ASELS", "EREGL", "TUPRS"]
    if len(sys.argv) > 1:
        test_symbols = [s.strip().upper() for s in sys.argv[1].split(",") if s.strip()]

    print(f"Starting Prophet Feature Ablation on {len(test_symbols)} symbols: {test_symbols}")
    res = run_experiment_ablation(test_symbols, n_sessions=30)
    print_summary_table(res)
