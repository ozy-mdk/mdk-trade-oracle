"""Week Start (Monday Day Closing) Institutional Predictive Forecaster.

Pits candidate machine learning paradigms (LightGBM, XGBoost, Ridge, Huber, Bayesian Ridge)
against univariate baselines armed with scarce weekend carry friction and Friday microstructure:
    - Cluster 1: Friday Window 5 (16:00-18:15 TRT) Institutional Flow Shares (MLB, BIG5, KAMU)
    - Cluster 2: Trailing 5-Session (WTD) Cumulative Accumulation & Broker Concordance
    - Cluster 3: Weekend FIFO Tertip Inventory Carrying Posture & Unrealized MTM PnL
    - Cluster 4: 3-Day Weekend Repo Carry Financing Hurdle (TCMB 1-Week Repo Rate)
    - Cluster 5: Weekly Price Momentum, Volatility & Benchmark BIST 30 Shock Distance
"""

from __future__ import annotations

import logging
import warnings
from pathlib import Path
from typing import Any

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.linear_model import BayesianRidge, HuberRegressor, Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from xgboost import XGBRegressor

from mdk_trading_oracle.core.db import PostgresManager
from mdk_trading_oracle.core.logger import get_logger
from mdk_trading_oracle.data.silver.tertip_analytics import (
    BIG_FIVE_BROKERS,
    KAMU_BROKERS,
)

logger = get_logger("mdk_oracle.week_start_forecaster")

# Persistent disk cache for rolling Prophet baseline features on weekly data
WEEKLY_PROPHET_CACHE_DIR = Path.home() / "data" / "mdk_oracle" / "cache" / "prophet_weekly_features"
WEEKLY_PROPHET_CACHE_DIR.mkdir(parents=True, exist_ok=True)

# 22 Zero-Leakage Lean Features for Monday Day Close Prediction (as of Friday 18:15 TRT)
WEEK_START_FEATURES = [
    # 1. Friday Closing Window 5 (16:00-18:15 TRT) Flow Shares % of Turnover
    "feat_fri_w5_mlb_share",
    "feat_fri_w5_big5_share",
    "feat_fri_w5_kamu_share",

    # 2. Friday Close vs VWAP & Intraday Dynamics
    "feat_fri_close_to_vwap_pct",
    "feat_fri_ret_pct",
    "feat_fri_range_pct",

    # 3. Trailing 5-Session Cumulative Accumulation (WTD)
    "feat_wtd_mlb_net_share_turnover",
    "feat_wtd_big5_net_share_turnover",
    "feat_wtd_kamu_net_share_turnover",
    "feat_wtd_inst_flow_concordance",

    # 4. Institutional FIFO Tertip Weekend Carrying Posture
    "feat_tertip_weekend_inventory_cost_spread_pct",
    "feat_tertip_weekend_unrealized_pnl_tl",
    "feat_tertip_weekend_intraday_pnl_5d_tl",

    # 5. Weekly Price Dynamics & Volatility
    "feat_week_ret_pct",
    "feat_fri_close_to_week_range_pct",
    "feat_week_volatility_pct",

    # 6. 3-Day Weekend Repo Carry Financing Hurdle & Macro
    "feat_weekend_carry_cost_bps",
    "feat_macro_repo_rate_pct",

    # 7. Benchmark Index (BIST 30) Momentum & Shock Proximity
    "feat_bist30_week_ret_pct",
    "feat_bist30_trend_vs_20d_sma_pct",
    "feat_days_since_pos_shock",
    "feat_days_since_neg_shock",
]

# Market Consolidation Deadband (+/- 0.25% / 25 bps)
DEADBAND_PCT: float = 0.25

# Minimum training weeks hard stop (3 months = ~13 trading weeks)
MIN_WEEKLY_TRAIN_WEEKS: int = 13

# Multi-Horizon Lookbacks for Weekly Cycles
WEEKLY_HORIZONS = {
    "3m": 13,    # ~3 months (13 trading weeks)
    "6m": 26,    # ~6 months (26 trading weeks)
    "12m": 52,   # ~12 months (52 trading weeks)
    "18m": 78,   # ~18 months (78 trading weeks)
    "24m": 104,  # ~24 months (104 trading weeks)
}

# Candidate Model Implementations
CANDIDATE_MODELS = ["LightGBM", "XGBoost", "Huber", "Ridge", "BayesianRidge"]

warnings.filterwarnings("ignore")
logging.getLogger("cmdstanpy").setLevel(logging.WARNING)
logging.getLogger("prophet").setLevel(logging.WARNING)
logging.getLogger("lightgbm").setLevel(logging.WARNING)


def extract_week_start_time_series(db: PostgresManager, symbol: str) -> pd.DataFrame:
    """Extract historical daily prices and multi-pillar flows, then build weekly transitions."""
    sym = symbol.upper()

    # 1. Stock Prices & Turnover
    df_prices = db.query_pl(
        """
        SELECT trade_date, open_price, high_price, low_price, close_price, daily_return_pct, total_turnover_tl, total_volume
        FROM silver_daily_stock_summary
        WHERE symbol = %s
        ORDER BY trade_date ASC;
        """,
        params=[sym],
    ).to_pandas()

    if len(df_prices) < 35:
        return pd.DataFrame()

    big5_in = ", ".join(f"'{b}'" for b in BIG_FIVE_BROKERS)
    kamu_in = ", ".join(f"'{b}'" for b in KAMU_BROKERS)

    # 2. Broker Daily Flows
    df_flows = db.query_pl(
        """
        SELECT 
            trade_date,
            SUM(CASE WHEN broker_id = 'MLB' THEN net_flow_tl ELSE 0 END) AS mlb_flow,
            SUM(CASE WHEN broker_id = ANY(%s) THEN net_flow_tl ELSE 0 END) AS big5_flow,
            SUM(CASE WHEN broker_id = ANY(%s) THEN net_flow_tl ELSE 0 END) AS kamu_flow,
            SUM(CASE WHEN broker_id = 'MLB' THEN total_turnover_tl ELSE 0 END) AS mlb_turnover,
            SUM(CASE WHEN broker_id = ANY(%s) THEN total_turnover_tl ELSE 0 END) AS big5_turnover,
            SUM(CASE WHEN broker_id = ANY(%s) THEN total_turnover_tl ELSE 0 END) AS kamu_turnover
        FROM silver_daily_broker_summary
        WHERE symbol = %s
        GROUP BY trade_date
        ORDER BY trade_date ASC;
        """,
        params=[
            list(BIG_FIVE_BROKERS),
            list(KAMU_BROKERS),
            list(BIG_FIVE_BROKERS),
            list(KAMU_BROKERS),
            sym,
        ],
    ).to_pandas()

    # 3. W5 Closing Session (16:00-18:15 TRT) Flows
    df_w5 = db.query_pl(
        f"""
        SELECT 
            trade_date,
            SUM(CASE WHEN broker_id = 'MLB' THEN net_flow_tl ELSE 0 END) AS mlb_w5_flow,
            SUM(CASE WHEN broker_id IN ({big5_in}) THEN net_flow_tl ELSE 0 END) AS big5_w5_flow,
            SUM(CASE WHEN broker_id IN ({kamu_in}) THEN net_flow_tl ELSE 0 END) AS kamu_w5_flow
        FROM silver_intraday_broker_window_summary
        WHERE symbol = %s AND window_name = 'closing_session'
        GROUP BY trade_date
        ORDER BY trade_date ASC;
        """,
        params=[sym],
    ).to_pandas()

    # 4. FIFO Inventory & PnL
    df_fifo = db.query_pl(
        """
        SELECT 
            trade_date,
            SUM(CASE WHEN broker_id = 'MLB' THEN open_stock_quantity ELSE 0 END) AS mlb_open_qty,
            MAX(CASE WHEN broker_id = 'MLB' THEN fifo_avg_cost ELSE 0 END) AS mlb_fifo_cost,
            SUM(CASE WHEN broker_id = 'MLB' THEN total_daily_pnl_tl ELSE 0 END) AS mlb_daily_pnl_tl,
            SUM(CASE WHEN broker_id = 'MLB' THEN unrealized_pnl_tl ELSE 0 END) AS mlb_unrealized_pnl_tl,
            SUM(CASE WHEN broker_id = 'MLB' THEN intraday_realized_pnl_tl ELSE 0 END) AS mlb_intraday_pnl_tl
        FROM silver_broker_fifo_daily
        WHERE symbol = %s
        GROUP BY trade_date
        ORDER BY trade_date ASC;
        """,
        params=[sym],
    ).to_pandas()

    # 5. Macro Policy Interest Rates
    df_macro = db.query_pl(
        """
        SELECT trade_date, interest_rate, rate_change, daily_carry_cost_bps, is_rate_change_day, days_since_last_rate_change
        FROM silver_daily_macro_rates
        ORDER BY trade_date ASC;
        """
    ).to_pandas()

    # 6. Benchmark BIST 30 Series
    df_bench = db.query_pl(
        """
        SELECT trade_date, daily_return_pct as bist30_ret_pct, rolling_5d_return_pct as bist30_5d_ret_pct,
               index_trend_vs_20d_sma as bist30_trend_20d_pct, days_since_last_positive_shock, days_since_last_negative_shock
        FROM silver_daily_benchmark_index
        ORDER BY trade_date ASC;
        """
    ).to_pandas()

    # Merge daily data
    df_daily = df_prices.merge(df_flows, on="trade_date", how="left")
    df_daily = df_daily.merge(df_w5, on="trade_date", how="left")
    df_daily = df_daily.merge(df_fifo, on="trade_date", how="left")
    df_daily = df_daily.merge(df_macro, on="trade_date", how="left")
    df_daily = df_daily.merge(df_bench, on="trade_date", how="left")

    fill_zero_cols = [
        "mlb_flow", "big5_flow", "kamu_flow", "mlb_turnover", "big5_turnover", "kamu_turnover",
        "mlb_w5_flow", "big5_w5_flow", "kamu_w5_flow", "mlb_open_qty", "mlb_fifo_cost",
        "mlb_daily_pnl_tl", "mlb_unrealized_pnl_tl", "mlb_intraday_pnl_tl"
    ]
    for c in fill_zero_cols:
        if c in df_daily.columns:
            df_daily[c] = df_daily[c].fillna(0.0)

    df_daily["trade_date"] = pd.to_datetime(df_daily["trade_date"])
    df_daily = df_daily.sort_values("trade_date").reset_index(drop=True)

    # Detect week start transitions (when ISO week changes)
    trade_dates = df_daily["trade_date"]
    week_indices = []

    for i in range(1, len(df_daily)):
        cur = trade_dates.iloc[i]
        prev = trade_dates.iloc[i - 1]
        if cur.isocalendar().week != prev.isocalendar().week or cur.isocalendar().year != prev.isocalendar().year:
            week_indices.append((i - 1, i))

    rows = []
    for fri_idx, mon_idx in week_indices:
        if fri_idx < 4:
            continue

        w5_slice = df_daily.iloc[fri_idx - 4 : fri_idx + 1]
        fri_row = df_daily.iloc[fri_idx]
        mon_row = df_daily.iloc[mon_idx]

        fri_close = float(fri_row["close_price"])
        mon_close = float(mon_row["close_price"])
        mon_open = float(mon_row["open_price"])
        fri_turnover = float(fri_row["total_turnover_tl"]) if fri_row["total_turnover_tl"] > 0 else 1.0

        # Targets
        target_ret_pct = ((mon_close - fri_close) / fri_close) * 100.0
        target_gap_pct = ((mon_open - fri_close) / fri_close) * 100.0

        # Cluster 1: Friday Window 5 (16:00-18:15 TRT)
        feat_fri_w5_mlb_share = (float(fri_row["mlb_w5_flow"]) / fri_turnover) * 100.0
        feat_fri_w5_big5_share = (float(fri_row["big5_w5_flow"]) / fri_turnover) * 100.0
        feat_fri_w5_kamu_share = (float(fri_row["kamu_w5_flow"]) / fri_turnover) * 100.0

        # Cluster 2: Friday Close vs VWAP & Intraday
        fri_vol = float(fri_row["total_volume"])
        fri_vwap = (fri_turnover / fri_vol) if fri_vol > 0 else fri_close
        feat_fri_close_to_vwap_pct = ((fri_close - fri_vwap) / fri_vwap) * 100.0 if fri_vwap > 0 else 0.0
        feat_fri_ret_pct = float(fri_row["daily_return_pct"] or 0.0) * 100.0
        feat_fri_range_pct = ((float(fri_row["high_price"]) - float(fri_row["low_price"])) / fri_close) * 100.0

        # Cluster 3: Trailing 5-Session Cumulative Accumulation (WTD)
        wtd_turnover = float(w5_slice["total_turnover_tl"].sum())
        wtd_turnover = wtd_turnover if wtd_turnover > 0 else 1.0

        wtd_mlb_flow = float(w5_slice["mlb_flow"].sum())
        wtd_big5_flow = float(w5_slice["big5_flow"].sum())
        wtd_kamu_flow = float(w5_slice["kamu_flow"].sum())

        feat_wtd_mlb_net_share_turnover = (wtd_mlb_flow / wtd_turnover) * 100.0
        feat_wtd_big5_net_share_turnover = (wtd_big5_flow / wtd_turnover) * 100.0
        feat_wtd_kamu_net_share_turnover = (wtd_kamu_flow / wtd_turnover) * 100.0

        mlb_sign = np.sign(wtd_mlb_flow)
        big5_sign = np.sign(wtd_big5_flow)
        feat_wtd_inst_flow_concordance = float(mlb_sign * big5_sign)

        # Cluster 4: FIFO Weekend Posture
        mlb_cost = float(fri_row["mlb_fifo_cost"])
        cost_spread = ((fri_close - mlb_cost) / mlb_cost * 100.0) if mlb_cost > 0 else 0.0
        feat_tertip_weekend_inventory_cost_spread_pct = float(np.clip(cost_spread, -50.0, 50.0))
        feat_tertip_weekend_unrealized_pnl_tl = float(fri_row["mlb_unrealized_pnl_tl"])
        feat_tertip_weekend_intraday_pnl_5d_tl = float(w5_slice["mlb_intraday_pnl_tl"].sum())

        # Cluster 5: Weekly Price Dynamics
        start_close = float(w5_slice.iloc[0]["open_price"])
        feat_week_ret_pct = ((fri_close - start_close) / start_close * 100.0) if start_close > 0 else 0.0
        w_high = float(w5_slice["high_price"].max())
        w_low = float(w5_slice["low_price"].min())
        feat_fri_close_to_week_range_pct = ((fri_close - w_low) / (w_high - w_low)) if (w_high - w_low) > 0 else 0.5
        returns_5d = w5_slice["daily_return_pct"].fillna(0.0).values
        feat_week_volatility_pct = float(np.std(returns_5d) * np.sqrt(252) * 100.0)

        # Cluster 6: 3-Day Carry Cost & Macro
        carry_bps = float(fri_row.get("daily_carry_cost_bps") or 10.0)
        feat_weekend_carry_cost_bps = carry_bps * 3.0  # 3 days across weekend
        feat_macro_repo_rate_pct = float(fri_row.get("interest_rate") or 45.0)

        # Cluster 7: Benchmark Index
        feat_bist30_week_ret_pct = float(fri_row.get("bist30_5d_ret_pct") or 0.0) * 100.0
        feat_bist30_trend_vs_20d_sma_pct = float(fri_row.get("bist30_trend_20d_pct") or 0.0) * 100.0
        feat_days_since_pos_shock = float(fri_row.get("days_since_last_positive_shock") or 30.0)
        feat_days_since_neg_shock = float(fri_row.get("days_since_last_negative_shock") or 30.0)

        rows.append({
            "as_of_friday_date": fri_row["trade_date"].strftime("%Y-%m-%d"),
            "target_monday_date": mon_row["trade_date"].strftime("%Y-%m-%d"),
            "friday_close_price": fri_close,
            "monday_close_price": mon_close,
            "monday_open_price": mon_open,
            "target_return_pct": target_ret_pct,
            "target_gap_pct": target_gap_pct,
            "target_bofa_flow_tl": float(mon_row.get("mlb_flow") or 0.0),
            "bist30_ret_pct": float(mon_row.get("bist30_ret_pct") or 0.0) * 100.0,
            "wtd_mlb_net_flow_tl": wtd_mlb_flow,
            "feat_fri_w5_mlb_share": feat_fri_w5_mlb_share,
            "feat_fri_w5_big5_share": feat_fri_w5_big5_share,
            "feat_fri_w5_kamu_share": feat_fri_w5_kamu_share,
            "feat_fri_close_to_vwap_pct": feat_fri_close_to_vwap_pct,
            "feat_fri_ret_pct": feat_fri_ret_pct,
            "feat_fri_range_pct": feat_fri_range_pct,
            "feat_wtd_mlb_net_share_turnover": feat_wtd_mlb_net_share_turnover,
            "feat_wtd_big5_net_share_turnover": feat_wtd_big5_net_share_turnover,
            "feat_wtd_kamu_net_share_turnover": feat_wtd_kamu_net_share_turnover,
            "feat_wtd_inst_flow_concordance": feat_wtd_inst_flow_concordance,
            "feat_tertip_weekend_inventory_cost_spread_pct": feat_tertip_weekend_inventory_cost_spread_pct,
            "feat_tertip_weekend_unrealized_pnl_tl": feat_tertip_weekend_unrealized_pnl_tl,
            "feat_tertip_weekend_intraday_pnl_5d_tl": feat_tertip_weekend_intraday_pnl_5d_tl,
            "feat_week_ret_pct": feat_week_ret_pct,
            "feat_fri_close_to_week_range_pct": feat_fri_close_to_week_range_pct,
            "feat_week_volatility_pct": feat_week_volatility_pct,
            "feat_weekend_carry_cost_bps": feat_weekend_carry_cost_bps,
            "feat_macro_repo_rate_pct": feat_macro_repo_rate_pct,
            "feat_bist30_week_ret_pct": feat_bist30_week_ret_pct,
            "feat_bist30_trend_vs_20d_sma_pct": feat_bist30_trend_vs_20d_sma_pct,
            "feat_days_since_pos_shock": feat_days_since_pos_shock,
            "feat_days_since_neg_shock": feat_days_since_neg_shock,
        })

    return pd.DataFrame(rows)


def build_candidate_model(model_name: str) -> Any:
    """Build initialized model with hyperparameter defaults tuned for weekly order flow."""
    if model_name == "LightGBM":
        return lgb.LGBMRegressor(
            n_estimators=45,
            max_depth=3,
            learning_rate=0.05,
            reg_alpha=1.0,
            reg_lambda=1.0,
            random_state=42,
            verbose=-1,
        )
    elif model_name == "XGBoost":
        return XGBRegressor(
            n_estimators=45,
            max_depth=3,
            learning_rate=0.05,
            reg_alpha=1.0,
            reg_lambda=1.0,
            random_state=42,
        )
    elif model_name == "Huber":
        return Pipeline([
            ("scaler", StandardScaler()),
            ("reg", HuberRegressor(epsilon=1.35, max_iter=200)),
        ])
    elif model_name == "Ridge":
        return Pipeline([
            ("scaler", StandardScaler()),
            ("reg", Ridge(alpha=10.0)),
        ])
    elif model_name == "BayesianRidge":
        return Pipeline([
            ("scaler", StandardScaler()),
            ("reg", BayesianRidge(max_iter=300)),
        ])
    else:
        raise ValueError(f"Unknown candidate model paradigm: {model_name}")


def compute_sample_weights(y_returns: np.ndarray) -> np.ndarray:
    """Calibrate piecewise sample weighting for quiet vs breakout weeks."""
    weights = np.ones(len(y_returns), dtype=float)
    for i, ret in enumerate(y_returns):
        abs_r = abs(ret)
        if abs_r < 1.0:
            weights[i] = 0.75
        elif abs_r < 2.0:
            weights[i] = 2.50
        else:
            weights[i] = 3.00
    return weights


def run_weekly_walk_forward_arena(
    df: pd.DataFrame,
    n_weeks: int = 20,
    lookback_weeks: int = 52,
) -> tuple[pd.DataFrame, dict[str, Any], list[dict[str, Any]]]:
    """Execute walk-forward out-of-sample evaluation across candidate models."""
    if df.empty or len(df) < MIN_WEEKLY_TRAIN_WEEKS:
        return pd.DataFrame(), {}, []

    total_len = len(df)
    eval_indices = list(range(max(0, total_len - n_weeks), total_len))

    model_preds: dict[str, list[float]] = {m: [] for m in CANDIDATE_MODELS}
    actual_returns: list[float] = []
    trade_dates: list[str] = []
    prior_dates: list[str] = []
    actual_prices: list[float] = []
    friday_prices: list[float] = []
    bist30_returns: list[float] = []
    carry_bps_list: list[float] = []
    w5_mlb_shares: list[float] = []
    wtd_mlb_flows: list[float] = []
    actual_training_weeks_list: list[int] = []
    actual_training_months_list: list[float] = []
    data_sufficiency_status_list: list[str] = []

    weeks_skipped_insufficient = 0
    weeks_with_partial_history = 0
    weeks_with_full_history = 0

    for idx in eval_indices:
        # Check available training history before this week
        available_weeks = idx
        if available_weeks < MIN_WEEKLY_TRAIN_WEEKS:
            # HARD STOP: insufficient training data (< 13 weeks = 3 months)
            weeks_skipped_insufficient += 1
            continue

        train_start = max(0, idx - lookback_weeks)
        train_df = df.iloc[train_start:idx]
        actual_train_len = len(train_df)
        if actual_train_len < MIN_WEEKLY_TRAIN_WEEKS:
            weeks_skipped_insufficient += 1
            continue

        if actual_train_len >= lookback_weeks:
            weeks_with_full_history += 1
            suff_status = "FULL"
        else:
            weeks_with_partial_history += 1
            suff_status = "PARTIAL_HISTORY"

        test_row = df.iloc[idx]

        X_train = train_df[WEEK_START_FEATURES].values
        y_train = train_df["target_return_pct"].values
        weights = compute_sample_weights(y_train)

        X_test = test_row[WEEK_START_FEATURES].values.reshape(1, -1)
        y_true = float(test_row["target_return_pct"])

        actual_returns.append(y_true)
        trade_dates.append(str(test_row["target_monday_date"]))
        prior_dates.append(str(test_row["as_of_friday_date"]))
        actual_prices.append(float(test_row["monday_close_price"]))
        friday_prices.append(float(test_row["friday_close_price"]))
        bist30_returns.append(float(test_row.get("bist30_ret_pct") or 0.0))
        carry_bps_list.append(float(test_row.get("feat_weekend_carry_cost_bps") or 0.0))
        w5_mlb_shares.append(float(test_row.get("feat_fri_w5_mlb_share") or 0.0))
        wtd_mlb_flows.append(float(test_row.get("wtd_mlb_net_flow_tl") or 0.0))
        actual_training_weeks_list.append(actual_train_len)
        actual_training_months_list.append(round(actual_train_len / 4.33, 1))
        data_sufficiency_status_list.append(suff_status)

        for m_name in CANDIDATE_MODELS:
            model = build_candidate_model(m_name)
            try:
                if m_name in ("LightGBM", "XGBoost"):
                    model.fit(X_train, y_train, sample_weight=weights)
                else:
                    model.fit(X_train, y_train, reg__sample_weight=weights)
                pred = float(model.predict(X_test)[0])
            except Exception as e:
                logger.warning("Fit error for %s on week %s: %s", m_name, test_row['target_monday_date'], e)
                pred = 0.0
            model_preds[m_name].append(pred)

    valid_eval_count = len(actual_returns)
    if valid_eval_count == 0:
        return df, {}, []

    summary: dict[str, Any] = {}
    ledger_records: list[dict[str, Any]] = []

    for m_name in CANDIDATE_MODELS:
        prefix = m_name.lower()
        preds = model_preds[m_name]
        hits = 0
        hit_errors = []
        miss_errors = []
        all_errors = []
        sig_hits = 0
        sig_total = 0

        for y_true, y_pred in zip(actual_returns, preds):
            if abs(y_true) <= DEADBAND_PCT and abs(y_pred) <= 1.0:
                is_hit = True
            elif y_pred > DEADBAND_PCT and y_true > DEADBAND_PCT:
                is_hit = True
            elif y_pred < -DEADBAND_PCT and y_true < -DEADBAND_PCT:
                is_hit = True
            else:
                is_hit = False

            err = abs(y_pred - y_true)
            all_errors.append(err)
            if is_hit:
                hits += 1
                hit_errors.append(err)
            else:
                miss_errors.append(err)

            if abs(y_true) >= 1.0:
                sig_total += 1
                if is_hit:
                    sig_hits += 1

        hit_rate = (hits / valid_eval_count) * 100.0 if valid_eval_count > 0 else 0.0
        mae = float(np.mean(all_errors)) if all_errors else 0.0
        hit_mae = float(np.mean(hit_errors)) if hit_errors else 0.0
        miss_mae = float(np.mean(miss_errors)) if miss_errors else 0.0
        loss = (100.0 - hit_rate) + 1.0 * hit_mae + 2.5 * miss_mae
        sig_rate = (sig_hits / sig_total * 100.0) if sig_total > 0 else 0.0

        summary[f"{prefix}_dir_hits"] = hits
        summary[f"{prefix}_hit_rate_pct"] = float(round(hit_rate, 1))
        summary[f"{prefix}_mae_pct"] = float(round(mae, 2))
        summary[f"{prefix}_hit_mae_pct"] = float(round(hit_mae, 2))
        summary[f"{prefix}_miss_mae_pct"] = float(round(miss_mae, 2))
        summary[f"{prefix}_sig_hits"] = sig_hits
        summary[f"{prefix}_sig_total"] = sig_total
        summary[f"{prefix}_sig_rate_pct"] = float(round(sig_rate, 1))
        summary[f"{prefix}_tournament_loss"] = float(round(loss, 2))

    data_sufficiency_pct = (weeks_with_full_history / valid_eval_count * 100.0) if valid_eval_count > 0 else 0.0
    summary["weeks_skipped_insufficient"] = weeks_skipped_insufficient
    summary["weeks_with_partial_history"] = weeks_with_partial_history
    summary["weeks_with_full_history"] = weeks_with_full_history
    summary["data_sufficiency_pct"] = float(round(data_sufficiency_pct, 1))
    summary["target_lookback_weeks"] = lookback_weeks

    # Build backtest ledger records for all evaluated sessions
    for i in range(valid_eval_count):
        y_true = actual_returns[i]
        rec = {
            "trade_date": trade_dates[i],
            "prior_date": prior_dates[i],
            "actual_price": actual_prices[i],
            "friday_price": friday_prices[i],
            "actual_return_pct": round(y_true, 2),
            "bist30_ret_pct": round(bist30_returns[i], 2),
            "weekend_carry_cost_bps": round(carry_bps_list[i], 1),
            "fri_w5_mlb_share_pct": round(w5_mlb_shares[i], 2),
            "wtd_mlb_net_flow_tl": round(wtd_mlb_flows[i], 0),
            "actual_training_weeks": actual_training_weeks_list[i],
            "actual_training_months": actual_training_months_list[i],
            "data_sufficiency_status": data_sufficiency_status_list[i],
        }
        for m_name in CANDIDATE_MODELS:
            pred = model_preds[m_name][i]
            if abs(y_true) <= DEADBAND_PCT and abs(pred) <= 1.0:
                is_hit = True
            elif pred > DEADBAND_PCT and y_true > DEADBAND_PCT:
                is_hit = True
            elif pred < -DEADBAND_PCT and y_true < -DEADBAND_PCT:
                is_hit = True
            else:
                is_hit = False

            rec[f"{m_name.lower()}_pred_return_pct"] = round(pred, 2)
            rec[f"{m_name.lower()}_pred_price"] = round(friday_prices[i] * (1.0 + pred / 100.0), 2)
            rec[f"{m_name.lower()}_is_hit"] = is_hit
            rec[f"{m_name.lower()}_err_pct"] = round(abs(pred - y_true), 2)
        ledger_records.append(rec)

    return df, summary, ledger_records


def assign_week_start_playbook(
    expected_return_pct: float,
    fri_ret_pct: float,
    fri_w5_mlb_share: float,
    carry_cost_bps: float,
    wtd_mlb_flow: float,
) -> str:
    """Classify the tactical Monday trade playbook from quantitative microstructure context."""
    if expected_return_pct >= 1.0 and fri_ret_pct < -0.5 and fri_w5_mlb_share > 1.0:
        return "WEEKEND_RISK_ABSORPTION"
    elif expected_return_pct >= 1.5 and fri_ret_pct > 0.5 and fri_w5_mlb_share > 1.5:
        return "FRIDAY_RUNUP_CONTINUATION"
    elif expected_return_pct <= -1.0 and carry_cost_bps > 35.0 and fri_w5_mlb_share < -1.0:
        return "WEEKEND_CARRY_LIQUIDATION"
    elif expected_return_pct >= 1.0 and wtd_mlb_flow > 0:
        return "TACTICAL_WEEK_OPEN_LONG"
    elif expected_return_pct <= -1.0:
        return "TACTICAL_WEEK_OPEN_SHORT"
    elif abs(expected_return_pct) <= DEADBAND_PCT:
        return "NEUTRAL_WEEK_START"
    elif expected_return_pct > 0:
        return "MILD_WEEK_START_LONG"
    else:
        return "MILD_WEEK_START_SHORT"


def predict_live_week_start(
    db: PostgresManager,
    symbol: str,
    champion_model: str = "LightGBM",
    lookback_weeks: int = 52,
    crowned_horizon: str = "12m",
) -> dict[str, Any] | None:
    """Generate live out-of-sample forecast for upcoming Monday session."""
    sym = symbol.upper()
    df = extract_week_start_time_series(db, sym)
    if df.empty or len(df) < MIN_WEEKLY_TRAIN_WEEKS:
        return None

    # Adapt training lookback if total history < lookback_weeks (while >= 13 weeks)
    actual_weeks = min(lookback_weeks, len(df))
    actual_months = round(actual_weeks / 4.33, 1)
    target_months = round(lookback_weeks / 4.33, 1)
    suff_status = "FULL" if len(df) >= lookback_weeks else "PARTIAL_HISTORY"
    window_desc = f"{actual_weeks}W ({actual_months:.1f}M) / Target {lookback_weeks}W"

    # Train model on the latest available weeks
    train_df = df.iloc[-actual_weeks:]
    X_train = train_df[WEEK_START_FEATURES].values
    y_train = train_df["target_return_pct"].values
    weights = compute_sample_weights(y_train)

    model = build_candidate_model(champion_model)
    if champion_model in ("LightGBM", "XGBoost"):
        model.fit(X_train, y_train, sample_weight=weights)
    else:
        model.fit(X_train, y_train, reg__sample_weight=weights)

    # Latest row features (as of latest Friday close)
    latest_row = df.iloc[-1]
    X_live = latest_row[WEEK_START_FEATURES].values.reshape(1, -1)
    exp_return = float(model.predict(X_live)[0])
    base_price = float(latest_row["friday_close_price"])

    target_price = round(base_price * (1.0 + exp_return / 100.0), 2)
    # 1.5x expected volatility band
    vol_band = max(0.8, float(latest_row.get("feat_week_volatility_pct") or 25.0) / 16.0)
    price_low = round(base_price * (1.0 + (exp_return - vol_band) / 100.0), 2)
    price_high = round(base_price * (1.0 + (exp_return + vol_band) / 100.0), 2)

    if exp_return >= 2.0:
        stance = "BULLISH"
        conviction = "HIGH_CONVICTION_LONG"
    elif exp_return <= -2.0:
        stance = "BEARISH"
        conviction = "HIGH_CONVICTION_SHORT"
    elif exp_return >= 1.0:
        stance = "BULLISH"
        conviction = "MODERATE_LONG"
    elif exp_return <= -1.0:
        stance = "BEARISH"
        conviction = "MODERATE_SHORT"
    elif exp_return > DEADBAND_PCT:
        stance = "BULLISH"
        conviction = "MILD_LONG"
    elif exp_return < -DEADBAND_PCT:
        stance = "BEARISH"
        conviction = "MILD_SHORT"
    else:
        stance = "CONSOLIDATION"
        conviction = "CONSOLIDATION"

    playbook = assign_week_start_playbook(
        expected_return_pct=exp_return,
        fri_ret_pct=float(latest_row["feat_fri_ret_pct"]),
        fri_w5_mlb_share=float(latest_row["feat_fri_w5_mlb_share"]),
        carry_cost_bps=float(latest_row["feat_weekend_carry_cost_bps"]),
        wtd_mlb_flow=float(latest_row["wtd_mlb_net_flow_tl"]),
    )

    return {
        "symbol": sym,
        "as_of_date": str(latest_row["as_of_friday_date"]),
        "target_date": str(latest_row["target_monday_date"]),
        "current_price": base_price,
        "target_price": target_price,
        "expected_return_pct": round(exp_return, 2),
        "price_low": price_low,
        "price_high": price_high,
        "stance": stance,
        "conviction": conviction,
        "playbook": playbook,
        "ml_champion_type": champion_model,
        "crowned_horizon": crowned_horizon,
        "training_lookback_weeks": lookback_weeks,
        "actual_training_weeks": actual_weeks,
        "actual_training_months": actual_months,
        "target_training_months": target_months,
        "actual_window_desc": window_desc,
        "data_sufficiency_status": suff_status,
        "weekend_carry_cost_bps": round(float(latest_row["feat_weekend_carry_cost_bps"]), 1),
        "fri_w5_mlb_share_pct": round(float(latest_row["feat_fri_w5_mlb_share"]), 2),
        "wtd_mlb_net_flow_tl": round(float(latest_row["wtd_mlb_net_flow_tl"]), 0),
    }
