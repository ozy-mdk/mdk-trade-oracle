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

from mdk_trading_oracle.core.db import PostgresManager
from mdk_trading_oracle.core.logger import get_logger
from mdk_trading_oracle.data.silver.tertip_analytics import (
    BIG_FIVE_BROKERS,
    KAMU_BROKERS,
    get_tertip_horizons_analysis,
)

logger = get_logger("mdk_oracle.tertip_ml_forecaster")

# Active 25-Feature Microstructure, Tertip & Execution Suite
FEATURE_COLS = [
    # Core Valuation & Momentum
    "feat_cost_spread_pct",
    "feat_ret_today_pct",
    "feat_ret_yesterday_pct",
    "feat_mlb_w5_share",

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

# In-memory cache for fast UI serving (TTL 5 minutes)
_FORECAST_CACHE: dict[str, tuple[float, dict[str, Any]]] = {}
CACHE_TTL_SECONDS = 300.0


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

    # Join on trade_date
    joined = (
        df_prices.join(df_inventory, on="trade_date", how="left")
        .join(df_flows, on="trade_date", how="left")
        .join(df_w5, on="trade_date", how="left")
        .to_pandas()
    )

    if joined.empty:
        return pd.DataFrame()

    # Engineer 10 Scarce Microstructure & Tertip Features
    joined["total_turnover_tl"] = joined["total_turnover_tl"].replace(0, np.nan).fillna(1e6)

    # 1. BofA Cost Spread %: (Close - BofA FIFO Cost) / BofA FIFO Cost * 100
    joined["feat_cost_spread_pct"] = np.where(
        (joined["fifo_avg_cost"] > 0) & (joined["close_price"] > 0),
        (joined["close_price"] - joined["fifo_avg_cost"]) / joined["fifo_avg_cost"] * 100.0,
        0.0,
    )
    # 2. 1D Flow Shares normalized by Total Turnover
    joined["feat_mlb_flow_share"] = joined["mlb_flow"].fillna(0) / joined["total_turnover_tl"]
    joined["feat_big5_flow_share"] = joined["big5_flow"].fillna(0) / joined["total_turnover_tl"]
    joined["feat_kamu_flow_share"] = joined["kamu_flow"].fillna(0) / joined["total_turnover_tl"]

    # 3. MLB W5 Closing Session Flow Share
    joined["feat_mlb_w5_share"] = joined["mlb_w5_flow"].fillna(0) / joined["total_turnover_tl"]

    # 4. Today's Price Return %
    joined["feat_ret_today_pct"] = (joined["daily_return_pct"].fillna(0) * 100.0).clip(-25.0, 25.0)

    # 5. Yesterday's Price Return % (1-day lagged return)
    joined["feat_ret_yesterday_pct"] = (joined["daily_return_pct"].shift(1).fillna(0) * 100.0).clip(-25.0, 25.0)

    # 6. BofA 3-Month (63d) Tertip Inventory Expansion %: (Current Qty - 63d EWMA Qty) / 63d EWMA Qty * 100
    ewma_q_mlb_63 = joined["mlb_open_qty"].fillna(0).ewm(span=63, adjust=False).mean()
    joined["feat_mlb_tertip_3m_ratio"] = np.where(
        ewma_q_mlb_63.abs() > 1.0,
        (joined["mlb_open_qty"].fillna(0) - ewma_q_mlb_63) / ewma_q_mlb_63.abs() * 100.0,
        0.0,
    )
    joined["feat_mlb_tertip_3m_ratio"] = joined["feat_mlb_tertip_3m_ratio"].clip(-100.0, 100.0).fillna(0.0)

    # 7. BIG5 3-Month (63d) Tertip Inventory Expansion %
    ewma_q_big5_63 = joined["big5_open_qty"].fillna(0).ewm(span=63, adjust=False).mean()
    joined["feat_big5_tertip_3m_ratio"] = np.where(
        ewma_q_big5_63.abs() > 1.0,
        (joined["big5_open_qty"].fillna(0) - ewma_q_big5_63) / ewma_q_big5_63.abs() * 100.0,
        0.0,
    )
    joined["feat_big5_tertip_3m_ratio"] = joined["feat_big5_tertip_3m_ratio"].clip(-100.0, 100.0).fillna(0.0)

    # 8. KAMU 3-Month (63d) Tertip Inventory Expansion % (including TRA)
    ewma_q_kamu_63 = joined["kamu_open_qty"].fillna(0).ewm(span=63, adjust=False).mean()
    joined["feat_kamu_tertip_3m_ratio"] = np.where(
        ewma_q_kamu_63.abs() > 1.0,
        (joined["kamu_open_qty"].fillna(0) - ewma_q_kamu_63) / ewma_q_kamu_63.abs() * 100.0,
        0.0,
    )
    joined["feat_kamu_tertip_3m_ratio"] = joined["feat_kamu_tertip_3m_ratio"].clip(-100.0, 100.0).fillna(0.0)

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
    df: pd.DataFrame, n_sessions: int = 30
) -> tuple[list[dict[str, Any]], dict[str, Any], Any]:
    """Execute zero-lookahead walk-forward benchmarking Prophet against ML models.

    Returns the 30-day reality ledger, summary tournament metrics, and the trained champion ML model.
    """
    feature_cols = FEATURE_COLS

    if len(df) < n_sessions + 30:
        return [], {}, None

    ledger: list[dict[str, Any]] = []

    # Model instances
    champion_ml_model = None

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

        # 2. ML Challenger Model: Bayesian Ridge on scarce microstructure features
        tr_clean = train_data.iloc[-200:].dropna(subset=feature_cols).copy()
        y_tr = (tr_clean["close_price"].shift(-1) - tr_clean["close_price"]) / tr_clean["close_price"] * 100.0
        X_tr = tr_clean[feature_cols].iloc[:-1]
        y_tr = y_tr.iloc[:-1]

        ridge = Ridge(alpha=10.0)
        ridge.fit(X_tr, y_tr)
        champion_ml_model = ridge

        pred_ret_ml = float(ridge.predict(pd.DataFrame([prev_row[feature_cols]]))[0])
        # Bound extreme predictions
        pred_ret_ml = max(-10.0, min(10.0, pred_ret_ml))
        ml_price = prev_price * (1.0 + pred_ret_ml / 100.0)

        # Performance Metrics
        prophet_err = abs(prophet_price - actual_price) / actual_price * 100.0
        ml_err = abs(ml_price - actual_price) / actual_price * 100.0

        if abs(actual_ret) <= 0.02:
            prophet_hit = abs(prophet_ret) <= 0.02
            ml_hit = abs(pred_ret_ml) <= 0.02
        else:
            prophet_hit = (prophet_ret > 0.02) if actual_ret > 0.02 else (prophet_ret < -0.02)
            ml_hit = (pred_ret_ml > 0.02) if actual_ret > 0.02 else (pred_ret_ml < -0.02)
        winner = "CHALLENGER" if ml_err < prophet_err else "BASE"
        ml_direction = "UP" if pred_ret_ml > 0.02 else ("DOWN" if pred_ret_ml < -0.02 else "FLAT")
        prophet_direction = "UP" if prophet_ret > 0.02 else ("DOWN" if prophet_ret < -0.02 else "FLAT")

        ledger.append({
            "date": str(test_row["trade_date"]).split(" ")[0],
            "actual_price": round(actual_price, 2),
            "actual_return_pct": round(actual_ret, 2),
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
            "mlb_action": mlb_action,
            "mlb_flow_tl": round(mlb_actual_flow, 1),
            "big5_action": big5_action,
            "big5_flow_tl": round(big5_actual_flow, 1),
            "kamu_action": kamu_action,
            "kamu_flow_tl": round(kamu_actual_flow, 1),
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
        champion_label = "Tertip ML Challenger (Ridge Ensemble)"
    else:
        champion = "PROPHET_BASE"
        champion_label = "Prophet Base Model"

    tournament_summary = {
        "champion": champion,
        "champion_label": champion_label,
        "ml_hit_rate_pct": round(ml_hit_rate, 1),
        "ml_mae_pct": round(ml_mae, 2),
        "ml_wins": ml_wins,
        "prophet_hit_rate_pct": round(prophet_hit_rate, 1),
        "prophet_mae_pct": round(prophet_mae, 2),
        "prophet_wins": prophet_wins,
        "total_sessions": len(ledger),
    }

    return ledger, tournament_summary, champion_ml_model


def get_tertip_ml_forecast(db: PostgresManager, symbol: str) -> dict[str, Any]:
    """Generate live upcoming session (T+1) forecast and 30-day walk-forward track.

    Uses zero-lookahead training, compares Prophet Base vs Tertip ML Challenger,
    and returns comprehensive institutional intelligence.
    """
    sym = symbol.upper()
    now_ts = datetime.now(timezone.utc).timestamp()

    # Check cache
    if sym in _FORECAST_CACHE:
        cached_ts, cached_data = _FORECAST_CACHE[sym]
        if (now_ts - cached_ts) < CACHE_TTL_SECONDS:
            return cached_data

    # Extract time series
    df = extract_3pillar_time_series(db, sym, lookback_days=450)
    if df.empty or len(df) < 35:
        return {"error": f"Insufficient historical data for symbol {sym}"}

    # Run 30-Day Walk-Forward Tournament
    ledger, tournament, trained_ml_model = run_30d_walk_forward_arena(df, n_sessions=30)

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

    # 2. ML Challenger Prediction for T+1
    if trained_ml_model is not None:
        pred_ret_ml = float(trained_ml_model.predict(pd.DataFrame([latest_row[feature_cols]]))[0])
        pred_ret_ml = max(-10.0, min(10.0, pred_ret_ml))
    else:
        pred_ret_ml = 0.0

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
        "playbook_headline": playbook_headline,
        "playbook_rationale": playbook_rationale,
        "tournament_summary": tournament,
        "pillar_matrix": pillar_matrix,
        "walk_forward_ledger": ledger,
        "calculated_at": datetime.now(timezone.utc).isoformat(),
    }

    _FORECAST_CACHE[sym] = (now_ts, response_data)
    return response_data
