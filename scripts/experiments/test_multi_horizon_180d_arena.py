"""Experiment: Multi-Horizon Training Window & 180-Day Backtest Arena.

Evaluates 180-day out-of-sample walk-forward performance across 3 training lookback horizons:
- 3 Months (63 trading sessions): Ultra-adaptive / fast regime reaction
- 6 Months (126 trading sessions): Semi-annual structural balance
- 12 Months (252 trading sessions): Full structural carry cycle (Production Baseline)

Zero data leakage protocol.
Evaluated on calibrated ±0.25% (25 bps) market consolidation deadband.
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
    DEADBAND_PCT,
    FEATURE_COLS,
    extract_3pillar_time_series,
)

warnings.filterwarnings("ignore")
logging.getLogger("cmdstanpy").setLevel(logging.ERROR)
logging.getLogger("prophet").setLevel(logging.ERROR)
logging.getLogger("lightgbm").setLevel(logging.ERROR)

CACHE_DIR = Path.home() / "data" / "mdk_oracle" / "cache" / "multi_horizon_180d"
CACHE_DIR.mkdir(parents=True, exist_ok=True)

LOOKBACK_HORIZONS = {
    "3M_63d": 63,
    "6M_126d": 126,
    "12M_252d": 252,
    "18M_378d": 378,
    "24M_504d": 504,
}


def compute_or_load_prophet_rolling(
    df: pd.DataFrame,
    symbol: str,
    n_eval_sessions: int = 180,
    train_lookback: int = 252,
    force_recompute: bool = False,
) -> pd.DataFrame:
    """Precompute zero-leakage rolling Prophet 1-step forecast for the evaluation sessions."""
    sym = symbol.upper()
    cache_file = CACHE_DIR / f"{sym}_prophet_180d.parquet"

    if not force_recompute and cache_file.exists():
        try:
            cached_df = pd.read_parquet(cache_file)
            if len(cached_df) == len(df) and "feat_prophet_ret_today_pct" in cached_df.columns:
                print(f"[{sym}] Loaded cached Prophet rolling features ({len(cached_df)} rows)")
                return cached_df
        except Exception:
            pass

    df_out = df.copy()
    N = len(df_out)
    start_idx = max(train_lookback + 5, N - n_eval_sessions)

    prophet_ret_today = np.zeros(N, dtype=np.float64)
    print(f"[{sym}] Precomputing Prophet rolling baseline for {N - start_idx} sessions...")
    t0 = time.time()

    close_prices = df_out["close_price"].to_numpy()
    trade_dates = pd.to_datetime(df_out["trade_date"])

    for k in range(start_idx, N):
        hist_dates = trade_dates.iloc[k - train_lookback : k]
        hist_prices = close_prices[k - train_lookback : k]
        p_prev = close_prices[k - 1]
        target_date = trade_dates.iloc[k]

        p_df = pd.DataFrame({"ds": hist_dates, "y": hist_prices})
        m = Prophet(
            daily_seasonality=False,
            weekly_seasonality=False,
            yearly_seasonality=False,
            changepoint_prior_scale=0.05,
        )
        m.fit(p_df)

        target_df = pd.DataFrame({"ds": [target_date]})
        fc = m.predict(target_df)
        yhat_today = float(fc.iloc[-1]["yhat"])
        ret_today_fc = (yhat_today - p_prev) / p_prev * 100.0 if p_prev > 0 else 0.0
        prophet_ret_today[k] = ret_today_fc

    df_out["feat_prophet_ret_today_pct"] = prophet_ret_today
    df_out.to_parquet(cache_file, index=False)
    print(f"[{sym}] Prophet precomputation done in {time.time() - t0:.1f}s")
    return df_out


def run_horizon_walk_forward(
    df: pd.DataFrame,
    symbol: str,
    lookback_sessions: int,
    horizon_name: str,
    n_eval_sessions: int = 180,
    feature_cols: list[str] | None = None,
    deadband_pct: float = DEADBAND_PCT,
) -> dict[str, Any]:
    """Execute walk-forward evaluation across n_eval_sessions using specified lookback horizon."""
    f_cols = feature_cols if feature_cols is not None else list(FEATURE_COLS)
    N = len(df)
    start_eval_idx = N - n_eval_sessions

    records: list[dict[str, Any]] = []

    ridge_hits = 0
    xgb_hits = 0
    prophet_hits = 0
    total_eval = 0

    ridge_errors: list[float] = []
    xgb_errors: list[float] = []
    prophet_errors: list[float] = []

    for k in range(start_eval_idx, N):
        train_start = max(0, k - lookback_sessions)
        tr = df.iloc[train_start:k].copy()

        # Target: 1-session forward return
        # In tr, close_price.shift(-1) represents tomorrow's close price relative to each row
        y_tr = (tr["close_price"].shift(-1) - tr["close_price"]) / tr["close_price"] * 100.0
        X_tr = tr[f_cols].iloc[:-1]
        y_tr = y_tr.iloc[:-1]

        # Drop NaN
        valid_mask = (~X_tr.isna().any(axis=1)) & (~y_tr.isna())
        X_tr = X_tr[valid_mask]
        y_tr = y_tr[valid_mask]

        if len(X_tr) < 63:
            continue

        # Fit Ridge
        m_ridge = Ridge(alpha=100.0)
        m_ridge.fit(X_tr, y_tr)

        # Fit XGBoost
        m_xgb = XGBRegressor(
            n_estimators=45,
            max_depth=2,
            learning_rate=0.03,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=42,
            n_jobs=1,
        )
        m_xgb.fit(X_tr, y_tr)

        # Test on session k
        test_row = df.iloc[k]
        prev_row = df.iloc[k - 1]
        x_test = df[f_cols].iloc[[k - 1]]  # Prior close features at session k-1 Close

        pred_ret_ridge = float(m_ridge.predict(x_test)[0])
        pred_ret_xgb = float(m_xgb.predict(x_test)[0])
        prophet_ret = float(test_row.get("feat_prophet_ret_today_pct", 0.0))

        actual_price = float(test_row["close_price"])
        prev_price = float(prev_row["close_price"])
        actual_ret = (actual_price - prev_price) / prev_price * 100.0 if prev_price > 0 else 0.0

        # Scoring with deadband
        def is_hit(pred: float, act: float, db: float) -> bool:
            if abs(act) <= db:
                return abs(pred) <= db
            return (act > db and pred > db) or (act < -db and pred < -db)

        ridge_hit = is_hit(pred_ret_ridge, actual_ret, deadband_pct)
        xgb_hit = is_hit(pred_ret_xgb, actual_ret, deadband_pct)
        prophet_hit = is_hit(prophet_ret, actual_ret, deadband_pct)

        err_ridge = abs(pred_ret_ridge - actual_ret)
        err_xgb = abs(pred_ret_xgb - actual_ret)
        err_prophet = abs(prophet_ret - actual_ret)

        if ridge_hit:
            ridge_hits += 1
        if xgb_hit:
            xgb_hits += 1
        if prophet_hit:
            prophet_hits += 1

        ridge_errors.append(err_ridge)
        xgb_errors.append(err_xgb)
        prophet_errors.append(err_prophet)
        total_eval += 1

        records.append({
            "symbol": symbol,
            "horizon": horizon_name,
            "lookback_sessions": lookback_sessions,
            "trade_date": str(test_row["trade_date"]).split(" ")[0],
            "actual_price": actual_price,
            "actual_ret": actual_ret,
            "pred_ret_ridge": pred_ret_ridge,
            "pred_ret_xgb": pred_ret_xgb,
            "prophet_ret": prophet_ret,
            "ridge_hit": ridge_hit,
            "xgb_hit": xgb_hit,
            "prophet_hit": prophet_hit,
            "err_ridge": err_ridge,
            "err_xgb": err_xgb,
            "err_prophet": err_prophet,
        })

    ridge_is_hits = [r["ridge_hit"] for r in records]
    xgb_is_hits = [r["xgb_hit"] for r in records]
    prophet_is_hits = [r["prophet_hit"] for r in records]

    def calc_3crit(hits_cnt: int, err_list: list[float], is_hit_list: list[bool]) -> tuple[float, float, float, float]:
        n = len(err_list)
        if n == 0:
            return 0.0, 0.0, 0.0, 100.0
        hr = (hits_cnt / n) * 100.0
        hit_e = [e for e, h in zip(err_list, is_hit_list) if h]
        miss_e = [e for e, h in zip(err_list, is_hit_list) if not h]
        h_mae = float(np.mean(hit_e)) if hit_e else 0.0
        m_mae = float(np.mean(miss_e)) if miss_e else 0.0
        loss = (100.0 - hr) + 1.0 * h_mae + 2.5 * m_mae
        return round(hr, 1), round(h_mae, 2), round(m_mae, 2), round(loss, 2)

    ridge_hr, ridge_hmae, ridge_mmae, ridge_loss = calc_3crit(ridge_hits, ridge_errors, ridge_is_hits)
    xgb_hr, xgb_hmae, xgb_mmae, xgb_loss = calc_3crit(xgb_hits, xgb_errors, xgb_is_hits)
    prophet_hr, prophet_hmae, prophet_mmae, prophet_loss = calc_3crit(prophet_hits, prophet_errors, prophet_is_hits)

    ridge_mae = float(np.mean(ridge_errors)) if ridge_errors else 0.0
    xgb_mae = float(np.mean(xgb_errors)) if xgb_errors else 0.0
    prophet_mae = float(np.mean(prophet_errors)) if prophet_errors else 0.0

    # Pick ML model champion for this horizon (3-Criteria Tournament Loss primary: sign accuracy + hit calibration error + miss risk)
    if xgb_loss < ridge_loss:
        ml_champ = "XGBoost"
        ml_hits = xgb_hits
        ml_hit_rate = xgb_hr
        ml_mae = xgb_mae
        ml_hmae = xgb_hmae
        ml_mmae = xgb_mmae
        ml_loss = xgb_loss
    else:
        ml_champ = "Ridge"
        ml_hits = ridge_hits
        ml_hit_rate = ridge_hr
        ml_mae = ridge_mae
        ml_hmae = ridge_hmae
        ml_mmae = ridge_mmae
        ml_loss = ridge_loss

    return {
        "symbol": symbol,
        "horizon": horizon_name,
        "lookback_sessions": lookback_sessions,
        "total_eval": total_eval,
        "ml_champion": ml_champ,
        "ml_hits": ml_hits,
        "ml_hit_rate_pct": round(ml_hit_rate, 1),
        "ml_mae_pct": round(ml_mae, 2),
        "ml_hit_mae_pct": ml_hmae,
        "ml_miss_mae_pct": ml_mmae,
        "ml_tournament_loss": ml_loss,
        "ridge_hits": ridge_hits,
        "ridge_hit_rate_pct": ridge_hr,
        "ridge_mae_pct": round(ridge_mae, 2),
        "ridge_hit_mae_pct": ridge_hmae,
        "ridge_miss_mae_pct": ridge_mmae,
        "ridge_tournament_loss": ridge_loss,
        "xgb_hits": xgb_hits,
        "xgb_hit_rate_pct": xgb_hr,
        "xgb_mae_pct": round(xgb_mae, 2),
        "xgb_hit_mae_pct": xgb_hmae,
        "xgb_miss_mae_pct": xgb_mmae,
        "xgb_tournament_loss": xgb_loss,
        "prophet_hits": prophet_hits,
        "prophet_hit_rate_pct": prophet_hr,
        "prophet_mae_pct": round(prophet_mae, 2),
        "prophet_hit_mae_pct": prophet_hmae,
        "prophet_miss_mae_pct": prophet_mmae,
        "prophet_tournament_loss": prophet_loss,
        "records": records,
    }


def run_experiment(symbols: list[str], n_eval_sessions: int = 180) -> dict[str, Any]:
    """Run multi-horizon tournament across symbols."""
    db = PostgresManager()
    all_results: dict[str, dict[str, Any]] = {}
    all_pred_records: list[dict[str, Any]] = []

    for sym in symbols:
        print(f"\n==================== PROCESSING {sym} ({n_eval_sessions} Sessions) ====================")
        sym_t0 = time.time()
        # Need lookback_days >= 800 to cover 180 eval + 252 train + 63 burn-in (~500 trading days)
        df = extract_3pillar_time_series(db, sym, lookback_days=850)
        if len(df) < (n_eval_sessions + 252 + 10):
            print(f"[{sym}] Insufficient historical sessions: {len(df)} (need {n_eval_sessions + 262})")
            continue

        # Precompute Prophet once
        df_prophet = compute_or_load_prophet_rolling(df, sym, n_eval_sessions=n_eval_sessions)

        sym_horizon_results: dict[str, Any] = {}
        for h_name, h_lookback in LOOKBACK_HORIZONS.items():
            h_res = run_horizon_walk_forward(
                df_prophet,
                sym,
                lookback_sessions=h_lookback,
                horizon_name=h_name,
                n_eval_sessions=n_eval_sessions,
            )
            sym_horizon_results[h_name] = h_res
            all_pred_records.extend(h_res["records"])
            print(
                f"  [{sym}] {h_name:<10} | ML Champ: {h_res['ml_champion']:<7} | "
                f"Hits: {h_res['ml_hits']}/{h_res['total_eval']} ({h_res['ml_hit_rate_pct']:.1f}%) | "
                f"MAE: {h_res['ml_mae_pct']:.2f}% | (Prophet: {h_res['prophet_hit_rate_pct']:.1f}%)"
            )

        # Crown Horizon Champion for this symbol (3-Criteria Tournament Loss)
        best_h = min(
            sym_horizon_results.keys(),
            key=lambda k: (
                sym_horizon_results[k]["ml_tournament_loss"],
                -sym_horizon_results[k]["ml_hit_rate_pct"],
                sym_horizon_results[k]["ml_mae_pct"],
            ),
        )
        sym_horizon_results["crowned_horizon"] = best_h
        print(f"  >>> CROWNED LOOKBACK FOR {sym}: {best_h} (Loss: {sym_horizon_results[best_h]['ml_tournament_loss']}, Hits: {sym_horizon_results[best_h]['ml_hit_rate_pct']}%, Hit MAE: {sym_horizon_results[best_h]['ml_hit_mae_pct']}%, Miss MAE: {sym_horizon_results[best_h]['ml_miss_mae_pct']}%) in {time.time() - sym_t0:.1f}s")
        all_results[sym] = sym_horizon_results

    # Save detailed prediction logs
    if all_pred_records:
        df_preds = pd.DataFrame(all_pred_records)
        df_preds.to_parquet(CACHE_DIR / "multi_horizon_180d_predictions.parquet", index=False)

    return all_results


def print_comparison_tables(results: dict[str, dict[str, Any]], n_eval_sessions: int = 180) -> None:
    """Print clean comparison and portfolio summary tables."""
    print("\n" + "=" * 95)
    print("          MULTI-HORIZON TOURNAMENT: 180-DAY WALK-FORWARD OUT-OF-SAMPLE ARENA")
    print(f"          Evaluation Horizon: Last {n_eval_sessions} Sessions | Deadband: ±0.25% (25 bps)")
    print("=" * 95)

    symbols = [s for s in results.keys() if "crowned_horizon" in results[s]]
    if not symbols:
        print("No results to display.")
        return

    horizons = list(LOOKBACK_HORIZONS.keys())

    # 1. Directional Hit Rate % Table
    print("\n[DIRECTIONAL HIT RATE % (Out-of-Sample 180 Sessions)]")
    header = f"{'Symbol':<10}" + "".join(f"{h:<20}" for h in horizons) + f"{'Crowned Lookback':<18}" + f"{'Delta vs 12M':<12}"
    print(header)
    print("-" * len(header))

    portfolio_hits: dict[str, list[float]] = {h: [] for h in horizons}
    portfolio_maes: dict[str, list[float]] = {h: [] for h in horizons}

    for sym in symbols:
        res = results[sym]
        crowned = res["crowned_horizon"]
        row = f"{sym:<10}"
        rate_12m = res.get("12M_252d", {}).get("ml_hit_rate_pct", 0.0)

        for h in horizons:
            h_data = res.get(h, {})
            hits = h_data.get("ml_hits", 0)
            tot = h_data.get("total_eval", n_eval_sessions)
            rate = h_data.get("ml_hit_rate_pct", 0.0)
            model = h_data.get("ml_champion", "ML")
            portfolio_hits[h].append(rate)
            portfolio_maes[h].append(h_data.get("ml_mae_pct", 0.0))
            is_best = " *" if h == crowned else "  "
            row += f"{hits}/{tot} ({rate:.1f}%) {model[:1]}{is_best}  "

        crowned_rate = res.get(crowned, {}).get("ml_hit_rate_pct", 0.0)
        delta_vs_12m = crowned_rate - rate_12m
        delta_str = f"{delta_vs_12m:+.1f}%" if crowned != "12M_252d" else "Baseline"
        row += f"{crowned:<18}{delta_str:<12}"
        print(row)

    print("-" * len(header))
    port_row = f"{'PORTFOLIO':<10}"
    for h in horizons:
        mean_h = np.mean(portfolio_hits[h]) if portfolio_hits[h] else 0.0
        port_row += f"AVG: {mean_h:.1f}%              "
    print(port_row)

    # 2. MAE % Table
    print("\n[MEAN ABSOLUTE ERROR (MAE %) (Out-of-Sample 180 Sessions)]")
    mae_header = f"{'Symbol':<10}" + "".join(f"{h:<20}" for h in horizons)
    print(mae_header)
    print("-" * len(mae_header))

    for sym in symbols:
        res = results[sym]
        row = f"{sym:<10}"
        for h in horizons:
            mae = res.get(h, {}).get("ml_mae_pct", 0.0)
            row += f"{mae:.2f}%              "
        print(row)

    print("-" * len(mae_header))
    port_mae_row = f"{'PORTFOLIO':<10}"
    for h in horizons:
        mean_mae = np.mean(portfolio_maes[h]) if portfolio_maes[h] else 0.0
        port_mae_row += f"AVG: {mean_mae:.2f}%           "
    print(port_mae_row)

    print("\n" + "=" * 95)


def main() -> None:
    symbols = ["THYAO", "AKBNK", "ASELS", "EREGL", "TUPRS"]
    n_eval = 180
    if len(sys.argv) > 1:
        symbols = [s.strip().upper() for s in sys.argv[1].split(",") if s.strip()]

    print(f"Launching Multi-Horizon 180-Day Arena on {len(symbols)} benchmark equities: {symbols}")
    results = run_experiment(symbols, n_eval_sessions=n_eval)
    print_comparison_tables(results, n_eval_sessions=n_eval)


if __name__ == "__main__":
    main()
