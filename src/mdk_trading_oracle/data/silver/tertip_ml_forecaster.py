"""Tertip-Anchored Machine Learning Forecaster & 3-Pillar Tournament Engine.

Pits a univariate time-series base model (Prophet) against trained Machine Learning
challenger models (Bayesian Ridge / LightGBM) armed with scarce 3-pillar institutional
microstructure features:
    - Pillar 1: Bank of America (MLB)
    - Pillar 2: Domestic Big 5 Desks (BIG5: YKR, IYM, AKM, GRM, ZRY)
    - Pillar 3: State Conduits (KAMU: ZRY, VKY, HLY, TRA)

Features zero-lookahead point-in-time training and a 30-day walk-forward reality ledger
recording the exact next-day actions executed by each institutional pillar.
"""

import logging
import warnings
from datetime import datetime, timezone
from typing import Any

import numpy as np
import pandas as pd
from prophet import Prophet
from sklearn.linear_model import Ridge
from xgboost import XGBRegressor

from mdk_trading_oracle.core.db import PostgresManager
from mdk_trading_oracle.core.logger import get_logger
from mdk_trading_oracle.data.silver.tertip_analytics import (
    BIG_FIVE_BROKERS,
    KAMU_BROKERS,
    get_tertip_horizons_analysis,
)

logger = get_logger("mdk_oracle.tertip_ml_forecaster")

# Active 28-Feature Microstructure, Tertip, Execution & BIST 30 Benchmark Suite
FEATURE_COLS = [
    # Core Valuation & Momentum
    "feat_cost_spread_pct",
    "feat_ret_today_pct",
    "feat_ret_yesterday_pct",
    "feat_mlb_w5_share",

    # BIST 30 (XU030) 3-Day Return Dynamics
    "feat_bist30_ret_today_pct",
    "feat_bist30_ret_yesterday_pct",
    "feat_bist30_ret_day_before_pct",

    # 3-Month Tertip Inventory Expansion
    "feat_mlb_tertip_3m_ratio",
    "feat_big5_tertip_3m_ratio",
    "feat_kamu_tertip_3m_ratio",

    # Today's Execution Breakdown (Buy, Sell, Realized PnL)
    "feat_mlb_buy_share_today",
    "feat_mlb_sell_share_today",
    "feat_mlb_pnl_share_today",
    "feat_big5_buy_share_today",
    "feat_big5_sell_share_today",
    "feat_big5_pnl_share_today",
    "feat_kamu_buy_share_today",
    "feat_kamu_sell_share_today",
    "feat_kamu_pnl_share_today",

    # Yesterday's Execution Breakdown (Buy, Sell, Realized PnL)
    "feat_mlb_buy_share_yesterday",
    "feat_mlb_sell_share_yesterday",
    "feat_mlb_pnl_share_yesterday",
    "feat_big5_buy_share_yesterday",
    "feat_big5_sell_share_yesterday",
    "feat_big5_pnl_share_yesterday",
    "feat_kamu_buy_share_yesterday",
    "feat_kamu_sell_share_yesterday",
    "feat_kamu_pnl_share_yesterday",
]

# Suppress verbose warnings from third-party math packages
warnings.filterwarnings("ignore")
logging.getLogger("cmdstanpy").setLevel(logging.WARNING)
logging.getLogger("prophet").setLevel(logging.WARNING)
logging.getLogger("lightgbm").setLevel(logging.WARNING)

# In-memory cache for fast UI serving (TTL 30 seconds to stay fresh)
_FORECAST_CACHE: dict[str, tuple[float, dict[str, Any]]] = {}
CACHE_TTL_SECONDS = 30.0


def clear_forecast_cache() -> None:
    """Clear in-memory forecast cache to ensure fresh computation."""
    global _FORECAST_CACHE
    _FORECAST_CACHE.clear()
    logger.info("Cleared Tertip ML Forecaster in-memory cache.")


def extract_3pillar_time_series(
    db: PostgresManager, symbol: str, lookback_days: int = 400
) -> pd.DataFrame:
    """Extract historical daily prices and 3-pillar institutional order flows."""
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
    )

    if len(df_prices) < 35:
        return pd.DataFrame()

    big5_in = ", ".join(f"'{b}'" for b in BIG_FIVE_BROKERS)
    kamu_in = ", ".join(f"'{b}'" for b in KAMU_BROKERS)

    # 2. 3-Pillar FIFO Inventories, Costs & Execution (Buy, Sell, PnL)
    df_inventory = db.query_pl(
        f"""
        SELECT 
            trade_date,
            SUM(CASE WHEN broker_id = 'MLB' THEN open_stock_quantity ELSE 0 END) AS mlb_open_qty,
            MAX(CASE WHEN broker_id = 'MLB' THEN fifo_avg_cost ELSE 0 END) AS fifo_avg_cost,
            SUM(CASE WHEN broker_id = 'MLB' THEN buy_turnover_tl ELSE 0 END) AS mlb_buy_tl,
            SUM(CASE WHEN broker_id = 'MLB' THEN sell_turnover_tl ELSE 0 END) AS mlb_sell_tl,
            SUM(CASE WHEN broker_id = 'MLB' THEN total_daily_pnl_tl ELSE 0 END) AS mlb_daily_pnl_tl,
            SUM(CASE WHEN broker_id = 'MLB' THEN unrealized_pnl_tl ELSE 0 END) AS mlb_unrealized_pnl_tl,

            SUM(CASE WHEN broker_id IN ({big5_in}) THEN open_stock_quantity ELSE 0 END) AS big5_open_qty,
            SUM(CASE WHEN broker_id IN ({big5_in}) THEN buy_turnover_tl ELSE 0 END) AS big5_buy_tl,
            SUM(CASE WHEN broker_id IN ({big5_in}) THEN sell_turnover_tl ELSE 0 END) AS big5_sell_tl,
            SUM(CASE WHEN broker_id IN ({big5_in}) THEN total_daily_pnl_tl ELSE 0 END) AS big5_daily_pnl_tl,
            SUM(CASE WHEN broker_id IN ({big5_in}) THEN unrealized_pnl_tl ELSE 0 END) AS big5_unrealized_pnl_tl,

            SUM(CASE WHEN broker_id IN ({kamu_in}) THEN open_stock_quantity ELSE 0 END) AS kamu_open_qty,
            SUM(CASE WHEN broker_id IN ({kamu_in}) THEN buy_turnover_tl ELSE 0 END) AS kamu_buy_tl,
            SUM(CASE WHEN broker_id IN ({kamu_in}) THEN sell_turnover_tl ELSE 0 END) AS kamu_sell_tl,
            SUM(CASE WHEN broker_id IN ({kamu_in}) THEN total_daily_pnl_tl ELSE 0 END) AS kamu_daily_pnl_tl,
            SUM(CASE WHEN broker_id IN ({kamu_in}) THEN unrealized_pnl_tl ELSE 0 END) AS kamu_unrealized_pnl_tl
        FROM silver_broker_fifo_daily
        WHERE symbol = %s
        GROUP BY trade_date
        ORDER BY trade_date ASC;
        """,
        params=[sym],
    )

    # 3. 3-Pillar Aggregated Daily Flows
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
    )

    # 4. MLB Closing Session Window 5 Flow
    df_w5 = db.query_pl(
        """
        SELECT trade_date, net_flow_tl as mlb_w5_flow
        FROM silver_intraday_broker_window_summary
        WHERE symbol = %s AND broker_id = 'MLB' AND window_name = 'closing_session'
        ORDER BY trade_date ASC;
        """,
        params=[sym],
    )

    # 5. BIST 30 (XU030) Benchmark Daily Returns
    df_bist30 = db.query_pl(
        """
        SELECT 
            s.trade_date,
            COALESCE(
                b.daily_return_pct * 100.0,
                s.basket_return_pct
            ) AS bist30_return_pct
        FROM (
            SELECT trade_date,
                   (COALESCE(SUM(adj_daily_return_pct * total_turnover_tl) / NULLIF(SUM(total_turnover_tl), 0), 0.0) * 100.0) AS basket_return_pct
            FROM silver_daily_stock_summary
            WHERE index_name = 'BIST30'
            GROUP BY trade_date
        ) s
        LEFT JOIN silver_daily_benchmark_index b USING (trade_date)
        ORDER BY s.trade_date ASC;
        """
    )

    # Join on trade_date
    joined = (
        df_prices.join(df_inventory, on="trade_date", how="left")
        .join(df_flows, on="trade_date", how="left")
        .join(df_w5, on="trade_date", how="left")
        .join(df_bist30, on="trade_date", how="left")
        .to_pandas()
    )

    if joined.empty:
        return pd.DataFrame()

    # Engineer 26 Scarce Microstructure, Tertip, Execution & BIST 30 Features
    joined["total_turnover_tl"] = joined["total_turnover_tl"].replace(0, np.nan).fillna(1e6)
    tt_today = joined["total_turnover_tl"]
    tt_yesterday = joined["total_turnover_tl"].shift(1).replace(0, np.nan).fillna(1e6)

    # 1. BofA Cost Spread %: (Close - BofA FIFO Cost) / BofA FIFO Cost * 100
    joined["feat_cost_spread_pct"] = np.where(
        (joined["fifo_avg_cost"] > 0) & (joined["close_price"] > 0),
        (joined["close_price"] - joined["fifo_avg_cost"]) / joined["fifo_avg_cost"] * 100.0,
        0.0,
    )

    # 2. Today's & Yesterday's Price Returns %
    joined["feat_ret_today_pct"] = (joined["daily_return_pct"].fillna(0) * 100.0).clip(-25.0, 25.0)
    joined["feat_ret_yesterday_pct"] = (joined["daily_return_pct"].shift(1).fillna(0) * 100.0).clip(-25.0, 25.0)

    # 3. MLB W5 Closing Session Flow Share
    joined["feat_mlb_w5_share"] = joined["mlb_w5_flow"].fillna(0) / tt_today

    # 4. 3-Month (63d) Tertip Inventory Expansion %: (Current Qty - 63d EWMA Qty) / 63d EWMA Qty * 100
    ewma_q_mlb_63 = joined["mlb_open_qty"].fillna(0).ewm(span=63, adjust=False).mean()
    joined["feat_mlb_tertip_3m_ratio"] = np.where(
        ewma_q_mlb_63.abs() > 1.0,
        (joined["mlb_open_qty"].fillna(0) - ewma_q_mlb_63) / ewma_q_mlb_63.abs() * 100.0,
        0.0,
    )
    joined["feat_mlb_tertip_3m_ratio"] = joined["feat_mlb_tertip_3m_ratio"].clip(-100.0, 100.0).fillna(0.0)

    ewma_q_big5_63 = joined["big5_open_qty"].fillna(0).ewm(span=63, adjust=False).mean()
    joined["feat_big5_tertip_3m_ratio"] = np.where(
        ewma_q_big5_63.abs() > 1.0,
        (joined["big5_open_qty"].fillna(0) - ewma_q_big5_63) / ewma_q_big5_63.abs() * 100.0,
        0.0,
    )
    joined["feat_big5_tertip_3m_ratio"] = joined["feat_big5_tertip_3m_ratio"].clip(-100.0, 100.0).fillna(0.0)

    ewma_q_kamu_63 = joined["kamu_open_qty"].fillna(0).ewm(span=63, adjust=False).mean()
    joined["feat_kamu_tertip_3m_ratio"] = np.where(
        ewma_q_kamu_63.abs() > 1.0,
        (joined["kamu_open_qty"].fillna(0) - ewma_q_kamu_63) / ewma_q_kamu_63.abs() * 100.0,
        0.0,
    )
    joined["feat_kamu_tertip_3m_ratio"] = joined["feat_kamu_tertip_3m_ratio"].clip(-100.0, 100.0).fillna(0.0)

    # 5. Today's Execution Breakdown (Buy, Sell, Realized PnL) normalized by Today's Total Turnover %
    for p in ["mlb", "big5", "kamu"]:
        joined[f"feat_{p}_buy_share_today"] = (joined[f"{p}_buy_tl"].fillna(0) / tt_today * 100.0).clip(-100.0, 100.0)
        joined[f"feat_{p}_sell_share_today"] = (joined[f"{p}_sell_tl"].fillna(0) / tt_today * 100.0).clip(-100.0, 100.0)
        joined[f"feat_{p}_pnl_share_today"] = (joined[f"{p}_daily_pnl_tl"].fillna(0) / tt_today * 100.0).clip(-100.0, 100.0)

    # 6. Yesterday's Execution Breakdown (Buy, Sell, Realized PnL) normalized by Yesterday's Total Turnover %
    for p in ["mlb", "big5", "kamu"]:
        joined[f"feat_{p}_buy_share_yesterday"] = (joined[f"{p}_buy_tl"].shift(1).fillna(0) / tt_yesterday * 100.0).clip(-100.0, 100.0)
        joined[f"feat_{p}_sell_share_yesterday"] = (joined[f"{p}_sell_tl"].shift(1).fillna(0) / tt_yesterday * 100.0).clip(-100.0, 100.0)
        joined[f"feat_{p}_pnl_share_yesterday"] = (joined[f"{p}_daily_pnl_tl"].shift(1).fillna(0) / tt_yesterday * 100.0).clip(-100.0, 100.0)

    # 7. BIST 30 (XU030) 3-Day Return Dynamics (Today T, Yesterday T-1, Day Before T-2)
    joined["bist30_return_pct"] = joined["bist30_return_pct"].fillna(0.0)
    joined["feat_bist30_ret_today_pct"] = joined["bist30_return_pct"].clip(-15.0, 15.0)
    joined["feat_bist30_ret_yesterday_pct"] = joined["bist30_return_pct"].shift(1).fillna(0.0).clip(-15.0, 15.0)
    joined["feat_bist30_ret_day_before_pct"] = joined["bist30_return_pct"].shift(2).fillna(0.0).clip(-15.0, 15.0)

    # Distance (in sessions) from Last Positive Shock (>= +3%) and Last Negative Shock (<= -3%)
    series_idx = pd.Series(range(len(joined)), index=joined.index)
    pos_shock_idx = series_idx.where(joined["daily_return_pct"] >= 0.03).ffill()
    joined["feat_days_since_pos_shock"] = (series_idx - pos_shock_idx).fillna(63.0).clip(0.0, 63.0)

    neg_shock_idx = series_idx.where(joined["daily_return_pct"] <= -0.03).ffill()
    joined["feat_days_since_neg_shock"] = (series_idx - neg_shock_idx).fillna(63.0).clip(0.0, 63.0)

    return joined


def get_pillar_live_matrix(db: PostgresManager, symbol: str) -> list[dict[str, Any]]:
    """Build the 3-Pillar status scorecard for the live session."""
    sym = symbol.upper()
    pillars_meta = [
        {"key": "MLB", "name": "Bank of America (MLB)", "desc": "Foreign Algo Powerhouse"},
        {"key": "BIG5", "name": "Top 5 Domestic Desks", "desc": "YKR, IYM, AKM, GRM, ZRY"},
        {"key": "KAMU", "name": "State-Backed Conduits", "desc": "ZRY, VKY, HLY, TRA (State Support Floor)"},
    ]

    # Fetch latest stock turnover to compute turnover share
    latest_stock = db.query_pl(
        """
        SELECT total_turnover_tl FROM silver_daily_stock_summary
        WHERE symbol = %s
        ORDER BY trade_date DESC LIMIT 1;
        """,
        params=[sym],
    )
    total_to = latest_stock["total_turnover_tl"][0] if len(latest_stock) > 0 else 1.0

    # Turnovers
    mlb_to_res = db.query_pl(
        "SELECT total_turnover_tl FROM silver_daily_broker_summary WHERE symbol = %s AND broker_id = 'MLB' ORDER BY trade_date DESC LIMIT 1;",
        params=[sym],
    )
    big5_to_res = db.query_pl(
        "SELECT SUM(total_turnover_tl) AS to_tl FROM silver_daily_broker_summary WHERE symbol = %s AND broker_id = ANY(%s) AND trade_date = (SELECT MAX(trade_date) FROM silver_daily_broker_summary WHERE symbol = %s);",
        params=[sym, list(BIG_FIVE_BROKERS), sym],
    )
    kamu_to_res = db.query_pl(
        "SELECT SUM(total_turnover_tl) AS to_tl FROM silver_daily_broker_summary WHERE symbol = %s AND broker_id = ANY(%s) AND trade_date = (SELECT MAX(trade_date) FROM silver_daily_broker_summary WHERE symbol = %s);",
        params=[sym, list(KAMU_BROKERS), sym],
    )

    mlb_to = mlb_to_res["total_turnover_tl"][0] if len(mlb_to_res) > 0 else 0.0
    big5_to = big5_to_res["to_tl"][0] if len(big5_to_res) > 0 and big5_to_res["to_tl"][0] is not None else 0.0
    kamu_to = kamu_to_res["to_tl"][0] if len(kamu_to_res) > 0 and kamu_to_res["to_tl"][0] is not None else 0.0

    pillar_to_map = {
        "MLB": (mlb_to / total_to * 100.0) if total_to > 0 else 0.0,
        "BIG5": (big5_to / total_to * 100.0) if total_to > 0 else 0.0,
        "KAMU": (kamu_to / total_to * 100.0) if total_to > 0 else 0.0,
    }

    matrix = []
    for p in pillars_meta:
        pkey = p["key"]
        t_analysis = get_tertip_horizons_analysis(db, sym, pkey)
        cr = t_analysis.get("confluence_realization_12m") or {}

        stance = cr.get("active_stance") or "NEUTRAL"
        realized_pct = cr.get("realized_pct", 50.0)
        opposite_pct = cr.get("opposite_pct", 50.0)
        avg_flow = cr.get("avg_next_day_flow_tl", 0.0)
        price_up_pct = cr.get("price_up_pct", 50.0)

        # Expected Next-Day Action
        if stance == "BUY":
            if realized_pct >= 52.0:
                expected_action = "ACCUMULATE BUY"
            else:
                expected_action = "FADE / OPPOSITE SELL"
        elif stance == "SELL":
            if realized_pct >= 52.0:
                expected_action = "DISTRIBUTE SELL"
            else:
                expected_action = "SHORT COVER / REBOUND"
        else:
            expected_action = "NEUTRAL WAIT"

        matrix.append({
            "pillar": pkey,
            "name": p["name"],
            "desc": p["desc"],
            "stance": stance,
            "realized_pct": round(realized_pct, 1),
            "opposite_pct": round(opposite_pct, 1),
            "expected_action": expected_action,
            "expected_flow_tl": round(avg_flow, 2),
            "turnover_share_pct": round(pillar_to_map.get(pkey, 0.0), 1),
            "price_up_pct": round(price_up_pct, 1),
        })

    return matrix


def run_30d_walk_forward_arena(
    df: pd.DataFrame, n_sessions: int = 30, model_type: str = "auto"
) -> tuple[list[dict[str, Any]], dict[str, Any], Any]:
    """Execute zero-lookahead walk-forward benchmarking Prophet against ML models.

    Supports dynamic on-the-fly champion selection (Ridge vs XGBoost) when model_type="auto".
    Returns the 30-day reality ledger, summary tournament metrics, and the trained champion ML model.
    """
    feature_cols = FEATURE_COLS

    if len(df) < n_sessions + 30:
        return [], {}, None

    step_records: list[dict[str, Any]] = []

    for idx in range(len(df) - n_sessions, len(df)):
        train_data = df.iloc[:idx].copy()
        test_row = df.iloc[idx]
        prev_row = df.iloc[idx - 1]

        # Ground Truth
        actual_price = float(test_row["close_price"])
        prev_price = float(prev_row["close_price"])
        actual_ret = (actual_price - prev_price) / prev_price * 100.0

        # Actual Realized Flows on session t
        mlb_actual_flow = float(test_row["mlb_flow"]) if pd.notna(test_row["mlb_flow"]) else 0.0
        big5_actual_flow = float(test_row["big5_flow"]) if pd.notna(test_row["big5_flow"]) else 0.0
        kamu_actual_flow = float(test_row["kamu_flow"]) if pd.notna(test_row["kamu_flow"]) else 0.0

        mlb_action = "BUY" if mlb_actual_flow > 0 else "SELL"
        big5_action = "BUY" if big5_actual_flow > 0 else "SELL"
        kamu_action = "BUY" if kamu_actual_flow > 0 else "SELL"

        # 1. Base Model: Prophet (trailing 120 days pure price series)
        p_df = pd.DataFrame({
            "ds": pd.to_datetime(train_data["trade_date"].iloc[-120:]),
            "y": train_data["close_price"].iloc[-120:],
        })
        m_prophet = Prophet(
            daily_seasonality=False,
            weekly_seasonality=False,
            yearly_seasonality=False,
            changepoint_prior_scale=0.05,
        )
        m_prophet.fit(p_df)
        fc_prophet = m_prophet.predict(m_prophet.make_future_dataframe(periods=1))
        prophet_price = float(fc_prophet.iloc[-1]["yhat"])
        prophet_ret = (prophet_price - prev_price) / prev_price * 100.0
        prophet_err = abs(prophet_price - actual_price) / actual_price * 100.0

        if abs(actual_ret) <= 0.02:
            prophet_hit = abs(prophet_ret) <= 0.02
        else:
            prophet_hit = (prophet_ret > 0.02) if actual_ret > 0.02 else (prophet_ret < -0.02)

        # 2. ML Candidate Features
        tr_clean = train_data.iloc[-200:].dropna(subset=feature_cols).copy()
        y_tr = (tr_clean["close_price"].shift(-1) - tr_clean["close_price"]) / tr_clean["close_price"] * 100.0
        X_tr = tr_clean[feature_cols].iloc[:-1]
        y_tr = y_tr.iloc[:-1]

        # Candidate A: Ridge
        ridge = Ridge(alpha=10.0, random_state=42)
        ridge.fit(X_tr, y_tr)
        pred_ret_ridge = float(ridge.predict(pd.DataFrame([prev_row[feature_cols]]))[0])
        pred_ret_ridge = max(-10.0, min(10.0, pred_ret_ridge))
        ridge_price = prev_price * (1.0 + pred_ret_ridge / 100.0)
        ridge_err = abs(ridge_price - actual_price) / actual_price * 100.0

        if abs(actual_ret) <= 0.02:
            ridge_hit = abs(pred_ret_ridge) <= 0.02
        else:
            ridge_hit = (pred_ret_ridge > 0.02) if actual_ret > 0.02 else (pred_ret_ridge < -0.02)

        # Candidate B: XGBoost
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

        if abs(actual_ret) <= 0.02:
            xgb_hit = abs(pred_ret_xgb) <= 0.02
        else:
            xgb_hit = (pred_ret_xgb > 0.02) if actual_ret > 0.02 else (pred_ret_xgb < -0.02)

        # Shock day metrics on actual realization
        is_pos_shock = bool(actual_ret >= 3.0)
        is_neg_shock = bool(actual_ret <= -3.0)
        shock_type = "POSITIVE" if is_pos_shock else ("NEGATIVE" if is_neg_shock else "NONE")

        step_records.append({
            "test_row": test_row,
            "prev_row": prev_row,
            "actual_price": actual_price,
            "actual_ret": actual_ret,
            "prophet_price": prophet_price,
            "prophet_ret": prophet_ret,
            "prophet_err": prophet_err,
            "prophet_hit": prophet_hit,
            "ridge_price": ridge_price,
            "ridge_ret": pred_ret_ridge,
            "ridge_err": ridge_err,
            "ridge_hit": ridge_hit,
            "xgb_price": xgb_price,
            "xgb_ret": pred_ret_xgb,
            "xgb_err": xgb_err,
            "xgb_hit": xgb_hit,
            "is_pos_shock": is_pos_shock,
            "is_neg_shock": is_neg_shock,
            "shock_type": shock_type,
            "mlb_actual_flow": mlb_actual_flow,
            "big5_actual_flow": big5_actual_flow,
            "kamu_actual_flow": kamu_actual_flow,
            "mlb_action": mlb_action,
            "big5_action": big5_action,
            "kamu_action": kamu_action,
            "ridge_model": ridge,
            "xgb_model": xgb,
        })

    n_total = len(step_records)
    ridge_hits = sum(1 for s in step_records if s["ridge_hit"])
    ridge_hit_rate = (ridge_hits / n_total) * 100.0 if n_total > 0 else 0.0
    ridge_mae = sum(s["ridge_err"] for s in step_records) / n_total if n_total > 0 else 0.0

    xgb_hits = sum(1 for s in step_records if s["xgb_hit"])
    xgb_hit_rate = (xgb_hits / n_total) * 100.0 if n_total > 0 else 0.0
    xgb_mae = sum(s["xgb_err"] for s in step_records) / n_total if n_total > 0 else 0.0

    # Dynamic Champion Selection
    m_choice = model_type.lower()
    if m_choice == "ridge":
        ml_champion_type = "Ridge"
    elif m_choice == "xgboost":
        ml_champion_type = "XGBoost"
    else:  # "auto"
        if xgb_hits > ridge_hits:
            ml_champion_type = "XGBoost"
        elif ridge_hits > xgb_hits:
            ml_champion_type = "Ridge"
        else:
            # Tie breaker: lower MAE
            ml_champion_type = "XGBoost" if xgb_mae <= ridge_mae else "Ridge"

    champion_ml_model = (
        step_records[-1]["xgb_model"] if ml_champion_type == "XGBoost" else step_records[-1]["ridge_model"]
    )

    # Build the 30-Day Reality Ledger using the crowned ML champion
    ledger: list[dict[str, Any]] = []
    for s in step_records:
        test_row = s["test_row"]
        prev_row = s["prev_row"]
        actual_price = s["actual_price"]
        actual_ret = s["actual_ret"]
        prophet_price = s["prophet_price"]
        prophet_ret = s["prophet_ret"]
        prophet_err = s["prophet_err"]
        prophet_hit = s["prophet_hit"]

        if ml_champion_type == "XGBoost":
            ml_price = s["xgb_price"]
            pred_ret_ml = s["xgb_ret"]
            ml_err = s["xgb_err"]
            ml_hit = s["xgb_hit"]
        else:
            ml_price = s["ridge_price"]
            pred_ret_ml = s["ridge_ret"]
            ml_err = s["ridge_err"]
            ml_hit = s["ridge_hit"]

        winner = "CHALLENGER" if ml_err < prophet_err else "BASE"
        ml_direction = "UP" if pred_ret_ml > 0.02 else ("DOWN" if pred_ret_ml < -0.02 else "FLAT")
        prophet_direction = "UP" if prophet_ret > 0.02 else ("DOWN" if prophet_ret < -0.02 else "FLAT")

        ledger.append({
            "date": str(test_row["trade_date"]).split(" ")[0],
            "actual_price": round(actual_price, 2),
            "actual_return_pct": round(actual_ret, 2),
            "bist30_ret_pct": round(float(test_row["bist30_return_pct"]) if ("bist30_return_pct" in test_row and pd.notna(test_row["bist30_return_pct"])) else 0.0, 2),
            "ml_pred_price": round(ml_price, 2),
            "ml_pred_return_pct": round(pred_ret_ml, 2),
            "ml_direction": ml_direction,
            "ml_err_pct": round(ml_err, 2),
            "ml_is_hit": bool(ml_hit),
            "prophet_pred_price": round(prophet_price, 2),
            "prophet_pred_return_pct": round(prophet_ret, 2),
            "prophet_direction": prophet_direction,
            "prophet_err_pct": round(prophet_err, 2),
            "prophet_is_hit": bool(prophet_hit),
            "winner": winner,
            "is_shock_day": s["is_pos_shock"] or s["is_neg_shock"],
            "shock_type": s["shock_type"],
            "days_since_pos_shock": int(round(float(prev_row.get("feat_days_since_pos_shock", 63.0)))),
            "days_since_neg_shock": int(round(float(prev_row.get("feat_days_since_neg_shock", 63.0)))),
            "mlb_action": s["mlb_action"],
            "mlb_flow_tl": round(s["mlb_actual_flow"], 1),
            "mlb_buy_tl": round(float(test_row["mlb_buy_tl"]) if ("mlb_buy_tl" in test_row and pd.notna(test_row["mlb_buy_tl"])) else 0.0, 1),
            "mlb_sell_tl": round(float(test_row["mlb_sell_tl"]) if ("mlb_sell_tl" in test_row and pd.notna(test_row["mlb_sell_tl"])) else 0.0, 1),
            "mlb_pnl_tl": round(float(test_row["mlb_daily_pnl_tl"]) if ("mlb_daily_pnl_tl" in test_row and pd.notna(test_row["mlb_daily_pnl_tl"])) else 0.0, 1),
            "big5_action": s["big5_action"],
            "big5_flow_tl": round(s["big5_actual_flow"], 1),
            "big5_buy_tl": round(float(test_row["big5_buy_tl"]) if ("big5_buy_tl" in test_row and pd.notna(test_row["big5_buy_tl"])) else 0.0, 1),
            "big5_sell_tl": round(float(test_row["big5_sell_tl"]) if ("big5_sell_tl" in test_row and pd.notna(test_row["big5_sell_tl"])) else 0.0, 1),
            "big5_pnl_tl": round(float(test_row["big5_daily_pnl_tl"]) if ("big5_daily_pnl_tl" in test_row and pd.notna(test_row["big5_daily_pnl_tl"])) else 0.0, 1),
            "kamu_action": s["kamu_action"],
            "kamu_flow_tl": round(s["kamu_actual_flow"], 1),
            "kamu_buy_tl": round(float(test_row["kamu_buy_tl"]) if ("kamu_buy_tl" in test_row and pd.notna(test_row["kamu_buy_tl"])) else 0.0, 1),
            "kamu_sell_tl": round(float(test_row["kamu_sell_tl"]) if ("kamu_sell_tl" in test_row and pd.notna(test_row["kamu_sell_tl"])) else 0.0, 1),
            "kamu_pnl_tl": round(float(test_row["kamu_daily_pnl_tl"]) if ("kamu_daily_pnl_tl" in test_row and pd.notna(test_row["kamu_daily_pnl_tl"])) else 0.0, 1),
        })

    # Summary Statistics
    res_df = pd.DataFrame(ledger)
    ml_hit_rate = float(res_df["ml_is_hit"].mean() * 100.0)
    ml_mae = float(res_df["ml_err_pct"].mean())
    prophet_hit_rate = float(res_df["prophet_is_hit"].mean() * 100.0)
    prophet_mae = float(res_df["prophet_err_pct"].mean())
    ml_wins = int((res_df["winner"] == "CHALLENGER").sum())
    prophet_wins = int((res_df["winner"] == "BASE").sum())

    # Champion designation
    if ml_wins >= prophet_wins or ml_mae < prophet_mae:
        champion = "TERTIP_ML_CHALLENGER"
        champion_label = f"Tertip ML Challenger ({ml_champion_type})"
    else:
        champion = "PROPHET_BASE"
        champion_label = "Prophet Base Model"

    tournament_summary = {
        "champion": champion,
        "champion_label": champion_label,
        "ml_champion_type": ml_champion_type,
        "ml_hit_rate_pct": round(ml_hit_rate, 1),
        "ml_mae_pct": round(ml_mae, 2),
        "ml_wins": ml_wins,
        "prophet_hit_rate_pct": round(prophet_hit_rate, 1),
        "prophet_mae_pct": round(prophet_mae, 2),
        "prophet_wins": prophet_wins,
        "total_sessions": len(ledger),
        "ridge_hit_rate_pct": round(ridge_hit_rate, 1),
        "ridge_mae_pct": round(ridge_mae, 2),
        "xgboost_hit_rate_pct": round(xgb_hit_rate, 1),
        "xgboost_mae_pct": round(xgb_mae, 2),
    }

    return ledger, tournament_summary, champion_ml_model


def get_tertip_ml_forecast(
    db: PostgresManager,
    symbol: str,
    force_refresh: bool = False,
    model_type: str = "auto",
) -> dict[str, Any]:
    """Generate live upcoming session (T+1) forecast and 30-day walk-forward track.

    Uses zero-lookahead training, compares Prophet Base vs Tertip ML Challenger
    with dynamic on-the-fly champion selection (Ridge vs XGBoost),
    and returns comprehensive institutional intelligence.
    """
    sym = symbol.upper()
    m_type = model_type.lower()
    now_ts = datetime.now(timezone.utc).timestamp()
    cache_key = f"{sym}:{m_type}"

    # Check cache (bypassed if force_refresh is True)
    if not force_refresh and cache_key in _FORECAST_CACHE:
        cached_ts, cached_data = _FORECAST_CACHE[cache_key]
        if (now_ts - cached_ts) < CACHE_TTL_SECONDS:
            return cached_data

    # Extract time series
    df = extract_3pillar_time_series(db, sym, lookback_days=450)
    if df.empty or len(df) < 35:
        return {"error": f"Insufficient historical data for symbol {sym}"}

    # Run 30-Day Walk-Forward Tournament
    ledger, tournament, trained_ml_model = run_30d_walk_forward_arena(
        df, n_sessions=30, model_type=m_type
    )

    # 3-Pillar Matrix
    pillar_matrix = get_pillar_live_matrix(db, sym)

    # Generate Live Forecast for Tomorrow (T+1)
    latest_row = df.iloc[-1]
    latest_price = float(latest_row["close_price"])
    latest_date_str = str(latest_row["trade_date"]).split(" ")[0]

    feature_cols = FEATURE_COLS

    # 1. Base Prophet Prediction for T+1
    p_df = pd.DataFrame({
        "ds": pd.to_datetime(df["trade_date"].iloc[-120:]),
        "y": df["close_price"].iloc[-120:],
    })
    m_prophet = Prophet(
        daily_seasonality=False,
        weekly_seasonality=False,
        yearly_seasonality=False,
        changepoint_prior_scale=0.05,
    )
    m_prophet.fit(p_df)
    fc_prophet = m_prophet.predict(m_prophet.make_future_dataframe(periods=1))
    prophet_target_price = float(fc_prophet.iloc[-1]["yhat"])
    prophet_ret_pct = (prophet_target_price - latest_price) / latest_price * 100.0

    # 2. ML Challenger Prediction for T+1 (fitted on full available history up to session T)
    tr_full = df.iloc[-200:].dropna(subset=feature_cols).copy()
    y_tr_full = (tr_full["close_price"].shift(-1) - tr_full["close_price"]) / tr_full["close_price"] * 100.0
    X_tr_full = tr_full[feature_cols].iloc[:-1]
    y_tr_full = y_tr_full.iloc[:-1]

    ml_champion_type = tournament.get("ml_champion_type", "XGBoost")
    if ml_champion_type == "XGBoost":
        live_ml = XGBRegressor(
            n_estimators=45,
            max_depth=2,
            learning_rate=0.03,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=42,
            n_jobs=1,
        )
    else:
        live_ml = Ridge(alpha=10.0, random_state=42)

    live_ml.fit(X_tr_full, y_tr_full)
    pred_ret_ml = float(live_ml.predict(pd.DataFrame([latest_row[feature_cols]]))[0])
    pred_ret_ml = max(-10.0, min(10.0, pred_ret_ml))

    ml_target_price = latest_price * (1.0 + pred_ret_ml / 100.0)

    # Range envelope (using 20-day historical volatility)
    vol_20d = float(df["daily_return_pct"].iloc[-20:].std() * 100.0) if len(df) >= 20 else 2.5
    price_low = ml_target_price * (1.0 - (1.645 * vol_20d / 100.0))
    price_high = ml_target_price * (1.0 + (1.645 * vol_20d / 100.0))

    # Stance derivation
    if pred_ret_ml >= 2.0:
        stance = "STRONG_BUY"
        stance_badge = "STRONG BUY ACCUMULATION"
        stance_color = "emerald"
    elif pred_ret_ml > 0.02:
        stance = "BUY"
        stance_badge = "BUY ABSORPTION REBOUND"
        stance_color = "teal"
    elif pred_ret_ml <= -2.0:
        stance = "STRONG_SELL"
        stance_badge = "STRONG SELL PRESSURE"
        stance_color = "rose"
    elif pred_ret_ml < -0.02:
        stance = "SELL"
        stance_badge = "DISTRIBUTION FADE"
        stance_color = "orange"
    else:
        stance = "NEUTRAL"
        stance_badge = "NEUTRAL CONSOLIDATION"
        stance_color = "slate"

    # Actionable Blueprint
    playbook_headline = (
        f"Projecting {stance_badge} towards ₺{ml_target_price:.2f} ({pred_ret_ml:+.2f}%)"
    )
    playbook_rationale = (
        f"As of {latest_date_str}, {sym} closed at ₺{latest_price:.2f}. "
        f"The 30-day walk-forward arena crowns {tournament.get('champion_label', 'Tertip ML')} "
        f"with {tournament.get('ml_wins', 0)} of {tournament.get('total_sessions', 30)} daily wins "
        f"(MAE: {tournament.get('ml_mae_pct', 0.0):.2f}% vs Prophet: {tournament.get('prophet_mae_pct', 0.0):.2f}%). "
        f"Institutional synthesis projects next session trading between ₺{price_low:.2f} and ₺{price_high:.2f}."
    )

    # Build 3-Pillar Execution Summary (Today vs Yesterday)
    prev_row = df.iloc[-2] if len(df) >= 2 else latest_row
    prev_date_str = str(prev_row["trade_date"]).split(" ")[0]

    def _safe_float(row: pd.Series, col: str) -> float:
        return float(row[col]) if (col in row and pd.notna(row[col])) else 0.0

    pillar_execution = {
        "today_date": latest_date_str,
        "yesterday_date": prev_date_str,
        "mlb": {
            "today": {
                "buy_tl": round(_safe_float(latest_row, "mlb_buy_tl"), 2),
                "sell_tl": round(_safe_float(latest_row, "mlb_sell_tl"), 2),
                "net_flow_tl": round(_safe_float(latest_row, "mlb_flow"), 2),
                "daily_pnl_tl": round(_safe_float(latest_row, "mlb_daily_pnl_tl"), 2),
                "unrealized_pnl_tl": round(_safe_float(latest_row, "mlb_unrealized_pnl_tl"), 2),
            },
            "yesterday": {
                "buy_tl": round(_safe_float(prev_row, "mlb_buy_tl"), 2),
                "sell_tl": round(_safe_float(prev_row, "mlb_sell_tl"), 2),
                "net_flow_tl": round(_safe_float(prev_row, "mlb_flow"), 2),
                "daily_pnl_tl": round(_safe_float(prev_row, "mlb_daily_pnl_tl"), 2),
                "unrealized_pnl_tl": round(_safe_float(prev_row, "mlb_unrealized_pnl_tl"), 2),
            },
        },
        "big5": {
            "today": {
                "buy_tl": round(_safe_float(latest_row, "big5_buy_tl"), 2),
                "sell_tl": round(_safe_float(latest_row, "big5_sell_tl"), 2),
                "net_flow_tl": round(_safe_float(latest_row, "big5_flow"), 2),
                "daily_pnl_tl": round(_safe_float(latest_row, "big5_daily_pnl_tl"), 2),
                "unrealized_pnl_tl": round(_safe_float(latest_row, "big5_unrealized_pnl_tl"), 2),
            },
            "yesterday": {
                "buy_tl": round(_safe_float(prev_row, "big5_buy_tl"), 2),
                "sell_tl": round(_safe_float(prev_row, "big5_sell_tl"), 2),
                "net_flow_tl": round(_safe_float(prev_row, "big5_flow"), 2),
                "daily_pnl_tl": round(_safe_float(prev_row, "big5_daily_pnl_tl"), 2),
                "unrealized_pnl_tl": round(_safe_float(prev_row, "big5_unrealized_pnl_tl"), 2),
            },
        },
        "kamu": {
            "today": {
                "buy_tl": round(_safe_float(latest_row, "kamu_buy_tl"), 2),
                "sell_tl": round(_safe_float(latest_row, "kamu_sell_tl"), 2),
                "net_flow_tl": round(_safe_float(latest_row, "kamu_flow"), 2),
                "daily_pnl_tl": round(_safe_float(latest_row, "kamu_daily_pnl_tl"), 2),
                "unrealized_pnl_tl": round(_safe_float(latest_row, "kamu_unrealized_pnl_tl"), 2),
            },
            "yesterday": {
                "buy_tl": round(_safe_float(prev_row, "kamu_buy_tl"), 2),
                "sell_tl": round(_safe_float(prev_row, "kamu_sell_tl"), 2),
                "net_flow_tl": round(_safe_float(prev_row, "kamu_flow"), 2),
                "daily_pnl_tl": round(_safe_float(prev_row, "kamu_daily_pnl_tl"), 2),
                "unrealized_pnl_tl": round(_safe_float(prev_row, "kamu_unrealized_pnl_tl"), 2),
            },
        },
    }

    days_pos_shock = int(round(float(latest_row.get("feat_days_since_pos_shock", 63.0))))
    days_neg_shock = int(round(float(latest_row.get("feat_days_since_neg_shock", 63.0))))

    bist30_today = round(float(latest_row.get("feat_bist30_ret_today_pct", 0.0)), 2)
    bist30_yesterday = round(float(latest_row.get("feat_bist30_ret_yesterday_pct", 0.0)), 2)
    bist30_day_before = round(float(latest_row.get("feat_bist30_ret_day_before_pct", 0.0)), 2)

    bist30_trend = {
        "today_pct": bist30_today,
        "yesterday_pct": bist30_yesterday,
        "day_before_pct": bist30_day_before,
    }

    response_data = {
        "symbol": sym,
        "as_of_date": latest_date_str,
        "latest_close_price": round(latest_price, 2),
        "target_price": round(ml_target_price, 2),
        "expected_return_pct": round(pred_ret_ml, 2),
        "price_low": round(price_low, 2),
        "price_high": round(price_high, 2),
        "stance": stance,
        "stance_badge": stance_badge,
        "stance_color": stance_color,
        "prophet_target_price": round(prophet_target_price, 2),
        "prophet_expected_return_pct": round(prophet_ret_pct, 2),
        "days_since_last_positive_shock": days_pos_shock,
        "days_since_last_negative_shock": days_neg_shock,
        "bist30_trend": bist30_trend,
        "playbook_headline": playbook_headline,
        "playbook_rationale": playbook_rationale,
        "tournament_summary": tournament,
        "pillar_matrix": pillar_matrix,
        "pillar_execution": pillar_execution,
        "walk_forward_ledger": ledger,
        "calculated_at": datetime.now(timezone.utc).isoformat(),
    }

    _FORECAST_CACHE[cache_key] = (now_ts, response_data)
    return response_data
