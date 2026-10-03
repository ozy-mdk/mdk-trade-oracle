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

from __future__ import annotations

import logging
import math
import warnings
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import lightgbm as lgb
import numpy as np
import pandas as pd
from prophet import Prophet
from sklearn.linear_model import BayesianRidge, HuberRegressor, Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from xgboost import XGBRegressor

from mdk_trading_oracle.core.db import PostgresManager
from mdk_trading_oracle.core.logger import get_logger
from mdk_trading_oracle.data.silver.tertip_analytics import (
    BIG_FIVE_BROKERS,
    KAMU_BROKERS,
    get_tertip_horizons_analysis,
)

logger = get_logger("mdk_oracle.tertip_ml_forecaster")

# Persistent disk cache for rolling Prophet baseline features
PROPHET_CACHE_DIR = Path.home() / "data" / "mdk_oracle" / "cache" / "prophet_features"
PROPHET_CACHE_DIR.mkdir(parents=True, exist_ok=True)
ABLATION_CACHE_DIR = Path.home() / "data" / "mdk_oracle" / "cache" / "prophet_ablation"

# Lean & Pragmatic Feature Suite (21 Scarce Microstructure, Tertip Inventory, Execution, Shock Distance & Prophet Baseline)
FEATURE_COLS = [
    # Core Valuation & Momentum
    "feat_cost_spread_pct",
    "feat_ret_today_pct",
    "feat_ret_yesterday_pct",
    "feat_mlb_w5_share",

    # Univariate Baseline Momentum (Prophet 1-Step Prior for Session T, Zero-Leakage)
    "feat_prophet_ret_today_pct",

    # BIST 30 (XU030) Today's Return Dynamics
    "feat_bist30_ret_today_pct",

    # Macro Regime Shock Distance (Sessions elapsed since last positive / negative shock)
    "feat_days_since_pos_shock",
    "feat_days_since_neg_shock",

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

    # Aggregate Institutional Net Order Flow (MLB + Big 5 + Kamu Net Flow as % of Turnover)
    "feat_total_inst_net_share_today",
]

# Strict 12-Month Historical Training Lookback (252 BIST Trading Sessions)
TRAIN_LOOKBACK_SESSIONS = 252

# Strict 12-Month Historical Lookback (252 BIST Trading Sessions) for Prophet Baseline
PROPHET_LOOKBACK_SESSIONS = 252

# Neutral Consolidation Deadband % (+/- 0.25% / 25 bps)
# Reflects realistic BIST equity tick size (1-2 ticks) and intraday consolidation range.
DEADBAND_PCT: float = 0.25

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


_PROPHET_HOLIDAYS_CACHE: pd.DataFrame | None = None


def get_prophet_holidays(db: PostgresManager | None = None) -> pd.DataFrame:
    """Return Turkish public holidays + BIST 30 benchmark shock dates for Prophet.

    Includes:
    - Official Turkish national and religious holidays via `holidays.Turkey`
    - Historical BIST 30 positive & negative market shock dates from `silver_daily_benchmark_index`
    """
    global _PROPHET_HOLIDAYS_CACHE
    if _PROPHET_HOLIDAYS_CACHE is not None:
        return _PROPHET_HOLIDAYS_CACHE

    try:
        import holidays

        tr_hols = holidays.Turkey(years=range(2021, 2028))
        hol_rows = [{"ds": pd.to_datetime(dt), "holiday": "TR_HOLIDAY"} for dt in tr_hols.keys()]
    except Exception as e:
        logger.warning(f"Could not load Turkish holidays: {e}")
        hol_rows = []

    shock_rows = []
    active_db = db
    if active_db is None:
        try:
            active_db = PostgresManager()
        except Exception:
            active_db = None

    if active_db is not None:
        try:
            df_shocks = active_db.query_df(
                "SELECT DISTINCT trade_date, shock_type FROM silver_daily_benchmark_index WHERE is_shock_day = TRUE ORDER BY trade_date;"
            )
            shock_rows = [
                {"ds": pd.to_datetime(row["trade_date"]), "holiday": str(row["shock_type"])}
                for _, row in df_shocks.iterrows()
            ]
        except Exception as e:
            logger.warning(f"Could not load BIST shock regimes from DB: {e}")

    all_rows = hol_rows + shock_rows
    if all_rows:
        holidays_df = (
            pd.DataFrame(all_rows)
            .drop_duplicates(subset=["ds", "holiday"])
            .sort_values("ds")
            .reset_index(drop=True)
        )
    else:
        holidays_df = pd.DataFrame(columns=["ds", "holiday"])

    _PROPHET_HOLIDAYS_CACHE = holidays_df
    return _PROPHET_HOLIDAYS_CACHE


def attach_prophet_rolling_features(
    df: pd.DataFrame,
    symbol: str,
    train_lookback_sessions: int = TRAIN_LOOKBACK_SESSIONS,
    n_history_needed: int = 180,
    db: PostgresManager | None = None,
) -> pd.DataFrame:
    """Attach zero-leakage rolling Prophet 1-step baseline returns for each session T.

    Uses persistent disk caching to avoid redundant computation:
    - Fits Prophet strictly on all available historical data up to session k - 1 (iloc[:k]).
    - Incorporates official Turkish public holidays and BIST 30 positive & negative shock dates.
    - Captures weekly and yearly seasonality, evaluating 1-step model drift:
      feat_prophet_ret_today_pct = (yhat_k - yhat_{k-1}) / yhat_{k-1} * 100.
    - Automatically computes any missing/new sessions incrementally and saves to cache.
    """
    if df.empty or len(df) < 35:
        df["feat_prophet_ret_today_pct"] = 0.0
        return df

    # If df already contains clean feat_prophet_ret_today_pct with no trailing NaNs, preserve it
    if "feat_prophet_ret_today_pct" in df.columns:
        if not df["feat_prophet_ret_today_pct"].iloc[-30:].isna().any():
            return df

    sym = symbol.upper()
    cache_file = PROPHET_CACHE_DIR / f"{sym}_prophet_daily.parquet"
    cached_dict: dict[str, float] = {}

    if cache_file.exists():
        try:
            cached_df = pd.read_parquet(cache_file)
            if "trade_date" in cached_df.columns and "feat_prophet_ret_today_pct" in cached_df.columns:
                cached_dict = dict(
                    zip(
                        pd.to_datetime(cached_df["trade_date"]).dt.strftime("%Y-%m-%d"),
                        cached_df["feat_prophet_ret_today_pct"].astype(float),
                    )
                )
        except Exception as e:
            logger.warning("Failed to load Prophet cache for %s: %s", sym, e)

    N = len(df)
    start_idx = max(60, N - n_history_needed)
    close_prices = df["close_price"].to_numpy()
    trade_dates = pd.to_datetime(df["trade_date"])
    date_strs = trade_dates.dt.strftime("%Y-%m-%d").to_numpy()

    holidays_df = get_prophet_holidays(db)
    hols_param = holidays_df if not holidays_df.empty else None

    dirty = False
    for k in range(start_idx, N):
        d_str = date_strs[k]
        if d_str in cached_dict and pd.notna(cached_dict[d_str]):
            continue

        # Strictly point-in-time: trailing 24 months (up to 504 trading sessions) up to session k - 1
        start_hist = max(0, k - PROPHET_LOOKBACK_SESSIONS)
        hist_dates = trade_dates.iloc[start_hist:k]
        hist_prices = close_prices[start_hist:k]
        p_prev = close_prices[k - 1]
        target_date_prev = trade_dates.iloc[k - 1]
        target_date_today = trade_dates.iloc[k]

        if p_prev <= 0:
            cached_dict[d_str] = 0.0
            dirty = True
            continue

        try:
            p_df = pd.DataFrame({"ds": hist_dates, "y": hist_prices})
            m = Prophet(
                holidays=hols_param,
                daily_seasonality=False,
                weekly_seasonality=True,
                yearly_seasonality=True,
                changepoint_range=0.98,
                changepoint_prior_scale=0.15,
                uncertainty_samples=0,
            )
            m.fit(p_df)
            target_df = pd.DataFrame({"ds": [target_date_prev, target_date_today]})
            fc = m.predict(target_df)
            yhat_prev = float(fc.iloc[0]["yhat"])
            yhat_today = float(fc.iloc[1]["yhat"])
            if yhat_prev > 0:
                ret_today = (yhat_today - yhat_prev) / yhat_prev * 100.0
            else:
                ret_today = 0.0
            cached_dict[d_str] = float(np.clip(ret_today, -25.0, 25.0))
            dirty = True
        except Exception as e:
            logger.warning("Prophet fit failed for %s on %s: %s", sym, d_str, e)
            cached_dict[d_str] = 0.0
            dirty = True

    if dirty or (not cache_file.exists() and cached_dict):
        try:
            save_df = pd.DataFrame([
                {"trade_date": d, "feat_prophet_ret_today_pct": v}
                for d, v in cached_dict.items()
            ])
            save_df.to_parquet(cache_file)
        except Exception as e:
            logger.warning("Failed to save Prophet cache for %s: %s", sym, e)

    df["feat_prophet_ret_today_pct"] = [cached_dict.get(d, 0.0) for d in date_strs]
    return df


def extract_3pillar_time_series(
    db: PostgresManager, symbol: str, lookback_days: int = 550
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

    # 6. BIST 30 (XU030) Return Dynamics (Today T)
    joined["bist30_return_pct"] = joined["bist30_return_pct"].fillna(0.0)
    joined["feat_bist30_ret_today_pct"] = joined["bist30_return_pct"].clip(-15.0, 15.0)

    # 6. Distance (in sessions) from Last Positive Shock (>= +3%) and Last Negative Shock (<= -3%)
    series_idx = pd.Series(range(len(joined)), index=joined.index)
    pos_shock_idx = series_idx.where(joined["daily_return_pct"] >= 0.03).ffill()
    joined["feat_days_since_pos_shock"] = (series_idx - pos_shock_idx).fillna(63.0).clip(0.0, 63.0)

    neg_shock_idx = series_idx.where(joined["daily_return_pct"] <= -0.03).ffill()
    joined["feat_days_since_neg_shock"] = (series_idx - neg_shock_idx).fillna(63.0).clip(0.0, 63.0)

    # 7. Aggregate Institutional Net Flow Share % of Total Market Turnover
    joined["feat_total_inst_net_share_today"] = (
        (
            joined["mlb_buy_tl"].fillna(0) - joined["mlb_sell_tl"].fillna(0)
            + joined["big5_buy_tl"].fillna(0) - joined["big5_sell_tl"].fillna(0)
            + joined["kamu_buy_tl"].fillna(0) - joined["kamu_sell_tl"].fillna(0)
        )
        / tt_today * 100.0
    ).clip(-100.0, 100.0)

    # 8. Attach zero-leakage Prophet rolling baseline feature
    joined = attach_prophet_rolling_features(joined, sym, db=db)

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


def compute_convex_softmax_weights(hit_rate_ml: float, hit_rate_p: float, tau: float = 8.0) -> tuple[float, float]:
    """Compute adaptive convex softmax weights between ML and Prophet based on out-of-sample hit rates.

    Using a calibrated temperature parameter (tau=8.0):
    - When ML has a strong performance advantage (e.g. 63% vs 37%), ML receives ~93% dominance.
    - When performance is balanced (e.g. 52% vs 50%), weights remain balanced (~56% ML / 44% Prophet).
    - Bounded to [0.05, 0.95] to ensure continuous synthesis without degeneracy.
    """
    s_ml = max(0.0, min(100.0, float(hit_rate_ml))) / tau
    s_p = max(0.0, min(100.0, float(hit_rate_p))) / tau
    exp_ml = math.exp(s_ml)
    exp_p = math.exp(s_p)
    w_ml = exp_ml / (exp_ml + exp_p)
    w_ml = max(0.05, min(0.95, w_ml))
    w_p = 1.0 - w_ml
    return round(w_ml, 4), round(w_p, 4)


def _check_hit(pred: float, actual: float, tol_delta: float = 0.20, deadband: float = DEADBAND_PCT) -> bool:
    """Directional Hit Evaluation with Tolerance for Minor Consolidation (< 0.20% difference)."""
    if abs(pred - actual) <= tol_delta:
        return True
    if abs(actual) <= deadband:
        return (abs(pred) <= deadband) or ((pred * actual) >= 0.0)
    elif actual > deadband:
        return pred > 0.0
    else:
        return pred < 0.0


def run_30d_walk_forward_arena(
    df: pd.DataFrame,
    n_sessions: int = 30,
    model_type: str = "auto",
    feature_cols: list[str] | None = None,
    train_lookback_sessions: int = TRAIN_LOOKBACK_SESSIONS,
) -> tuple[list[dict[str, Any]], dict[str, Any], Any]:
    """Execute zero-lookahead walk-forward benchmarking Prophet against ML models.

    Strictly anchors historical training data to the last 12 months (252 trading sessions).
    Supports dynamic on-the-fly champion selection (Ridge vs XGBoost) when model_type="auto".
    Returns the 30-day reality ledger, summary tournament metrics, and the trained champion ML model.
    """
    f_cols = feature_cols if feature_cols is not None else FEATURE_COLS

    if len(df) < n_sessions + 30:
        return [], {}, None

    # Fallback safety: ensure all feature columns exist in df (e.g. In synthetic tests or custom inputs)
    for col in f_cols:
        if col not in df.columns:
            df[col] = 0.0

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

        # 1. Base Model: Prophet (strictly trailing lookback pure price series)
        if "feat_prophet_ret_today_pct" in test_row and pd.notna(test_row["feat_prophet_ret_today_pct"]) and float(test_row["feat_prophet_ret_today_pct"]) != 0.0:
            prophet_ret = float(test_row["feat_prophet_ret_today_pct"])
            prophet_price = prev_price * (1.0 + prophet_ret / 100.0)
            prophet_err = abs(prophet_price - actual_price) / actual_price * 100.0
        else:
            start_hist = max(0, len(train_data) - PROPHET_LOOKBACK_SESSIONS)
            p_df = pd.DataFrame({
                "ds": pd.to_datetime(train_data["trade_date"].iloc[start_hist:]),
                "y": train_data["close_price"].iloc[start_hist:],
            })
            holidays_df = get_prophet_holidays()
            m_prophet = Prophet(
                holidays=holidays_df if not holidays_df.empty else None,
                daily_seasonality=False,
                weekly_seasonality=True,
                yearly_seasonality=True,
                changepoint_range=0.98,
                changepoint_prior_scale=0.15,
                uncertainty_samples=0,
            )
            m_prophet.fit(p_df)
            target_df = pd.DataFrame({"ds": [prev_row["trade_date"], test_row["trade_date"]]})
            fc_prophet = m_prophet.predict(target_df)
            yhat_prev = float(fc_prophet.iloc[0]["yhat"])
            yhat_today = float(fc_prophet.iloc[1]["yhat"])
            prophet_ret = (yhat_today - yhat_prev) / yhat_prev * 100.0 if yhat_prev > 0 else 0.0
            prophet_ret = float(np.clip(prophet_ret, -25.0, 25.0))
            prophet_price = prev_price * (1.0 + prophet_ret / 100.0)
            prophet_err = abs(prophet_price - actual_price) / actual_price * 100.0

        # Directional Hit Evaluation with Tolerance for Minor Consolidation (< 0.20% difference)
        def _check_hit(pred: float, actual: float, tol_delta: float = 0.20, deadband: float = DEADBAND_PCT) -> bool:
            # If the difference between predicted return and actual return is <= 0.20%,
            # count as a hit regardless of opposite direction (minor consolidation / flat noise)
            if abs(pred - actual) <= tol_delta:
                return True
            if abs(actual) <= deadband:
                return (abs(pred) <= deadband) or ((pred * actual) >= 0.0)
            elif actual > deadband:
                return pred > 0.0
            else:
                return pred < 0.0

        prophet_hit = _check_hit(prophet_ret, actual_ret)

        # 2. ML Candidate Features (strictly trailing train_lookback_sessions)
        tr_clean = train_data.iloc[-train_lookback_sessions:].dropna(subset=f_cols).copy()
        y_tr = (tr_clean["close_price"].shift(-1) - tr_clean["close_price"]) / tr_clean["close_price"] * 100.0
        X_tr = tr_clean[f_cols].iloc[:-1]
        y_tr = y_tr.iloc[:-1]
        X_prev = pd.DataFrame([prev_row[f_cols]])

        # Candidate A: Ridge
        ridge = Ridge(alpha=10.0, random_state=42)
        ridge.fit(X_tr, y_tr)
        pred_ret_ridge = float(ridge.predict(X_prev)[0])
        pred_ret_ridge = max(-10.0, min(10.0, pred_ret_ridge))
        ridge_price = prev_price * (1.0 + pred_ret_ridge / 100.0)
        ridge_err = abs(ridge_price - actual_price) / actual_price * 100.0
        ridge_hit = _check_hit(pred_ret_ridge, actual_ret)

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
        pred_ret_xgb = float(xgb.predict(X_prev)[0])
        pred_ret_xgb = max(-10.0, min(10.0, pred_ret_xgb))
        xgb_price = prev_price * (1.0 + pred_ret_xgb / 100.0)
        xgb_err = abs(xgb_price - actual_price) / actual_price * 100.0
        xgb_hit = _check_hit(pred_ret_xgb, actual_ret)

        # Candidate C: LightGBM
        lgbm = lgb.LGBMRegressor(
            n_estimators=45,
            max_depth=3,
            learning_rate=0.03,
            num_leaves=7,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=42,
            n_jobs=1,
            verbose=-1,
        )
        lgbm.fit(X_tr, y_tr)
        pred_ret_lgbm = float(lgbm.predict(X_prev)[0])
        pred_ret_lgbm = max(-10.0, min(10.0, pred_ret_lgbm))
        lgbm_price = prev_price * (1.0 + pred_ret_lgbm / 100.0)
        lgbm_err = abs(lgbm_price - actual_price) / actual_price * 100.0
        lgbm_hit = _check_hit(pred_ret_lgbm, actual_ret)

        # Candidate D: Huber Regressor (L1/L2 Hybrid with Scaler)
        huber = Pipeline([
            ("scaler", StandardScaler()),
            ("regressor", HuberRegressor(epsilon=1.35, alpha=10.0, max_iter=300)),
        ])
        huber.fit(X_tr, y_tr)
        pred_ret_huber = float(huber.predict(X_prev)[0])
        pred_ret_huber = max(-10.0, min(10.0, pred_ret_huber))
        huber_price = prev_price * (1.0 + pred_ret_huber / 100.0)
        huber_err = abs(huber_price - actual_price) / actual_price * 100.0
        huber_hit = _check_hit(pred_ret_huber, actual_ret)

        # Candidate E: Bayesian Ridge (Automatic Relevance / Shrinkage with Scaler)
        bayes = Pipeline([
            ("scaler", StandardScaler()),
            ("regressor", BayesianRidge(max_iter=300)),
        ])
        bayes.fit(X_tr, y_tr)
        pred_ret_bayes = float(bayes.predict(X_prev)[0])
        pred_ret_bayes = max(-10.0, min(10.0, pred_ret_bayes))
        bayes_price = prev_price * (1.0 + pred_ret_bayes / 100.0)
        bayes_err = abs(bayes_price - actual_price) / actual_price * 100.0
        bayes_hit = _check_hit(pred_ret_bayes, actual_ret)

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
            "lgbm_price": lgbm_price,
            "lgbm_ret": pred_ret_lgbm,
            "lgbm_err": lgbm_err,
            "lgbm_hit": lgbm_hit,
            "huber_price": huber_price,
            "huber_ret": pred_ret_huber,
            "huber_err": huber_err,
            "huber_hit": huber_hit,
            "bayes_price": bayes_price,
            "bayes_ret": pred_ret_bayes,
            "bayes_err": bayes_err,
            "bayes_hit": bayes_hit,
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
            "lgbm_model": lgbm,
            "huber_model": huber,
            "bayes_model": bayes,
        })

    n_total = len(step_records)
    recent_window = min(30, n_total)
    recent_slice = step_records[-recent_window:]

    # Option 2: Simple Linear Composite Score / Loss
    # Direct linear combination prioritizing Directional Hit Rate:
    # Score = Hit Rate % - (2.0 * MAE %) (higher score is better)
    # Loss = (100.0 - Hit Rate %) + 2.0 * MAE % (lower loss is better, directly mirrors Score)
    def _calc_score(records: list[dict[str, Any]], err_key: str, hit_key: str) -> float:
        if not records:
            return 0.0
        n = len(records)
        hits = sum(1 for s in records if s[hit_key])
        hit_rate = (hits / n) * 100.0
        mae = sum(s[err_key] for s in records) / n
        return round((100.0 - hit_rate) + 2.0 * mae, 2)

    # Prophet metrics: recent 30-day window (for selection) and full-window
    prophet_30d_hits = sum(1 for s in recent_slice if s["prophet_hit"])
    prophet_30d_hit_rate = (prophet_30d_hits / recent_window) * 100.0 if recent_window > 0 else 0.0
    prophet_30d_mae = sum(s["prophet_err"] for s in recent_slice) / recent_window if recent_window > 0 else 0.0
    prophet_30d_penalty_loss = _calc_score(recent_slice, "prophet_err", "prophet_hit")

    prophet_hits = sum(1 for s in step_records if s["prophet_hit"])
    prophet_hit_rate = (prophet_hits / n_total) * 100.0 if n_total > 0 else 0.0
    prophet_mae = sum(s["prophet_err"] for s in step_records) / n_total if n_total > 0 else 0.0
    prophet_penalty_loss = _calc_score(step_records, "prophet_err", "prophet_hit")

    candidates_meta = {
        "Ridge": {
            "hits_30d": sum(1 for s in recent_slice if s["ridge_hit"]),
            "mae_30d": sum(s["ridge_err"] for s in recent_slice) / recent_window if recent_window > 0 else 0.0,
            "penalty_loss_30d": _calc_score(recent_slice, "ridge_err", "ridge_hit"),
            "hits": sum(1 for s in step_records if s["ridge_hit"]),
            "mae": sum(s["ridge_err"] for s in step_records) / n_total if n_total > 0 else 0.0,
            "penalty_loss": _calc_score(step_records, "ridge_err", "ridge_hit"),
            "model_key": "ridge_model",
            "price_key": "ridge_price",
            "ret_key": "ridge_ret",
            "err_key": "ridge_err",
            "hit_key": "ridge_hit",
        },
        "XGBoost": {
            "hits_30d": sum(1 for s in recent_slice if s["xgb_hit"]),
            "mae_30d": sum(s["xgb_err"] for s in recent_slice) / recent_window if recent_window > 0 else 0.0,
            "penalty_loss_30d": _calc_score(recent_slice, "xgb_err", "xgb_hit"),
            "hits": sum(1 for s in step_records if s["xgb_hit"]),
            "mae": sum(s["xgb_err"] for s in step_records) / n_total if n_total > 0 else 0.0,
            "penalty_loss": _calc_score(step_records, "xgb_err", "xgb_hit"),
            "model_key": "xgb_model",
            "price_key": "xgb_price",
            "ret_key": "xgb_ret",
            "err_key": "xgb_err",
            "hit_key": "xgb_hit",
        },
        "LightGBM": {
            "hits_30d": sum(1 for s in recent_slice if s["lgbm_hit"]),
            "mae_30d": sum(s["lgbm_err"] for s in recent_slice) / recent_window if recent_window > 0 else 0.0,
            "penalty_loss_30d": _calc_score(recent_slice, "lgbm_err", "lgbm_hit"),
            "hits": sum(1 for s in step_records if s["lgbm_hit"]),
            "mae": sum(s["lgbm_err"] for s in step_records) / n_total if n_total > 0 else 0.0,
            "penalty_loss": _calc_score(step_records, "lgbm_err", "lgbm_hit"),
            "model_key": "lgbm_model",
            "price_key": "lgbm_price",
            "ret_key": "lgbm_ret",
            "err_key": "lgbm_err",
            "hit_key": "lgbm_hit",
        },
        "Huber": {
            "hits_30d": sum(1 for s in recent_slice if s["huber_hit"]),
            "mae_30d": sum(s["huber_err"] for s in recent_slice) / recent_window if recent_window > 0 else 0.0,
            "penalty_loss_30d": _calc_score(recent_slice, "huber_err", "huber_hit"),
            "hits": sum(1 for s in step_records if s["huber_hit"]),
            "mae": sum(s["huber_err"] for s in step_records) / n_total if n_total > 0 else 0.0,
            "penalty_loss": _calc_score(step_records, "huber_err", "huber_hit"),
            "model_key": "huber_model",
            "price_key": "huber_price",
            "ret_key": "huber_ret",
            "err_key": "huber_err",
            "hit_key": "huber_hit",
        },
        "BayesianRidge": {
            "hits_30d": sum(1 for s in recent_slice if s["bayes_hit"]),
            "mae_30d": sum(s["bayes_err"] for s in recent_slice) / recent_window if recent_window > 0 else 0.0,
            "penalty_loss_30d": _calc_score(recent_slice, "bayes_err", "bayes_hit"),
            "hits": sum(1 for s in step_records if s["bayes_hit"]),
            "mae": sum(s["bayes_err"] for s in step_records) / n_total if n_total > 0 else 0.0,
            "penalty_loss": _calc_score(step_records, "bayes_err", "bayes_hit"),
            "model_key": "bayes_model",
            "price_key": "bayes_price",
            "ret_key": "bayes_ret",
            "err_key": "bayes_err",
            "hit_key": "bayes_hit",
        },
    }

    for m_info in candidates_meta.values():
        m_info["hit_rate_30d"] = (m_info["hits_30d"] / recent_window) * 100.0 if recent_window > 0 else 0.0
        m_info["hit_rate"] = (m_info["hits"] / n_total) * 100.0 if n_total > 0 else 0.0

    # Dynamic ML Challenger Selection: Strictly focuses on the LAST 30 DAYS (recent_slice)
    # Blended Criterion: Lowest 30-Day Directional Penalty Loss ((100 - HitRate) + 2.0 * MAE)
    # Tie-breakers: Higher 30-Day Hit Rate %, Lower 30-Day MAE %
    m_choice = model_type.lower()
    if m_choice in ("ridge",):
        ml_champion_type = "Ridge"
    elif m_choice in ("xgboost", "xgb"):
        ml_champion_type = "XGBoost"
    elif m_choice in ("lightgbm", "lgb", "lgbm"):
        ml_champion_type = "LightGBM"
    elif m_choice in ("huber", "huberregressor"):
        ml_champion_type = "Huber"
    elif m_choice in ("bayesianridge", "bayesian_ridge", "bayes"):
        ml_champion_type = "BayesianRidge"
    else:  # "auto": ranked by lowest recent 30-day penalty_loss, then hits descending, then mae ascending
        sorted_candidates = sorted(
            candidates_meta.keys(),
            key=lambda k: (
                candidates_meta[k]["penalty_loss_30d"],
                -candidates_meta[k]["hits_30d"],
                candidates_meta[k]["mae_30d"],
            ),
        )
        ml_champion_type = sorted_candidates[0]

    champ_meta = candidates_meta[ml_champion_type]
    champion_ml_model = step_records[-1][champ_meta["model_key"]]
    ml_candidate_30d_hits = champ_meta["hits_30d"]
    ml_candidate_30d_hit_rate = champ_meta["hit_rate_30d"]
    ml_candidate_30d_mae = champ_meta["mae_30d"]
    ml_candidate_30d_penalty = champ_meta["penalty_loss_30d"]
    ml_candidate_hits = champ_meta["hits"]
    ml_candidate_hit_rate = champ_meta["hit_rate"]
    ml_candidate_mae = champ_meta["mae"]
    ml_candidate_penalty = champ_meta["penalty_loss"]

    # Tournament Grand Champion Selection (Blended Score of Directional Hit Rate and MAE):
    # Evaluates all models: ML candidates and Prophet Base
    all_tournament_models = dict(candidates_meta)
    all_tournament_models["Prophet"] = {
        "hits_30d": prophet_30d_hits,
        "hit_rate_30d": prophet_30d_hit_rate,
        "mae_30d": prophet_30d_mae,
        "penalty_loss_30d": prophet_30d_penalty_loss,
        "hits": prophet_hits,
        "hit_rate": prophet_hit_rate,
        "mae": prophet_mae,
        "penalty_loss": prophet_penalty_loss,
        "price_key": "prophet_price",
        "ret_key": "prophet_ret",
        "err_key": "prophet_err",
        "hit_key": "prophet_hit",
    }

    sorted_grand = sorted(
        all_tournament_models.keys(),
        key=lambda k: (
            all_tournament_models[k]["penalty_loss_30d"],
            -all_tournament_models[k]["hits_30d"],
            all_tournament_models[k]["mae_30d"],
        ),
    )
    grand_champion_key = sorted_grand[0]
    grand_meta = all_tournament_models[grand_champion_key]

    if grand_champion_key == "Prophet":
        champion = "PROPHET_BASE"
        champion_label = "Prophet Base (12M Drift)"
    else:
        champion = "TERTIP_ML_CHALLENGER"
        champion_label = f"ML Champion ({grand_champion_key})"

    champion_dir_hits = grand_meta["hits_30d"]
    champion_dir_hit_rate_pct = grand_meta["hit_rate_30d"]
    champion_mae_pct = grand_meta["mae_30d"]
    champion_penalty_loss = grand_meta["penalty_loss_30d"]
    runner_up_key = sorted_grand[1]
    runner_up_dir_hit_rate_pct = all_tournament_models[runner_up_key]["hit_rate_30d"]

    # Telemetry weights for transparency
    convex_weight_ml, convex_weight_prophet = compute_convex_softmax_weights(
        ml_candidate_30d_hit_rate, prophet_30d_hit_rate, tau=8.0
    )

    # Build the Walk-Forward Reality Ledger using the Crowned Champion's pure prediction
    ledger: list[dict[str, Any]] = []
    ml_error_wins = 0
    prophet_error_wins = 0

    for s in step_records:
        test_row = s["test_row"]
        prev_row = s["prev_row"]
        actual_price = s["actual_price"]
        actual_ret = s["actual_ret"]
        prophet_price = s["prophet_price"]
        prophet_ret = s["prophet_ret"]
        prophet_err = s["prophet_err"]
        prophet_hit = s["prophet_hit"]

        ml_price = s[champ_meta["price_key"]]
        pred_ret_ml = s[champ_meta["ret_key"]]
        ml_err = s[champ_meta["err_key"]]
        ml_hit = s[champ_meta["hit_key"]]

        # Pure Crowned Champion prediction (no convex dilution)
        champ_price = s[grand_meta["price_key"]]
        champ_ret = s[grand_meta["ret_key"]]
        champ_err = s[grand_meta["err_key"]]
        champ_hit = s[grand_meta["hit_key"]]
        champ_dir = "UP" if champ_ret > DEADBAND_PCT else ("DOWN" if champ_ret < -DEADBAND_PCT else "FLAT")

        s["confluence_price"] = champ_price
        s["confluence_ret"] = champ_ret
        s["confluence_err"] = champ_err
        s["confluence_hit"] = champ_hit

        # Track error wins
        if ml_err <= prophet_err:
            ml_error_wins += 1
        else:
            prophet_error_wins += 1

        # Daily Session Winner: Directional Hit first!
        if ml_hit and not prophet_hit:
            winner = "CHALLENGER"
        elif prophet_hit and not ml_hit:
            winner = "BASE"
        else:
            winner = "CHALLENGER" if ml_err <= prophet_err else "BASE"

        ml_direction = "UP" if pred_ret_ml > DEADBAND_PCT else ("DOWN" if pred_ret_ml < -DEADBAND_PCT else "FLAT")
        prophet_direction = "UP" if prophet_ret > DEADBAND_PCT else ("DOWN" if prophet_ret < -DEADBAND_PCT else "FLAT")

        ledger.append({
            "date": str(test_row["trade_date"]).split(" ")[0],
            "actual_price": round(actual_price, 2),
            "actual_return_pct": round(actual_ret, 2),
            "bist30_ret_pct": round(float(test_row["bist30_return_pct"]) if ("bist30_return_pct" in test_row and pd.notna(test_row["bist30_return_pct"])) else 0.0, 2),
            "confluence_pred_price": round(champ_price, 2),
            "confluence_pred_return_pct": round(champ_ret, 2),
            "confluence_direction": champ_dir,
            "confluence_err_pct": round(champ_err, 2),
            "confluence_is_hit": bool(champ_hit),
            "champion_pred_price": round(champ_price, 2),
            "champion_pred_return_pct": round(champ_ret, 2),
            "champion_direction": champ_dir,
            "champion_err_pct": round(champ_err, 2),
            "champion_is_hit": bool(champ_hit),
            "convex_weight_ml": convex_weight_ml,
            "convex_weight_prophet": convex_weight_prophet,
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
            "training_lookback_sessions": train_lookback_sessions,
        })

    # Confluence Metrics (30-day and full-history)
    confluence_30d_hits = sum(1 for s in recent_slice if s["confluence_hit"])
    confluence_30d_hit_rate = (confluence_30d_hits / recent_window) * 100.0 if recent_window > 0 else 0.0
    confluence_30d_mae = sum(s["confluence_err"] for s in recent_slice) / recent_window if recent_window > 0 else 0.0
    confluence_30d_penalty_loss = _calc_score(recent_slice, "confluence_err", "confluence_hit")

    confluence_hits = sum(1 for s in step_records if s["confluence_hit"])
    confluence_hit_rate = (confluence_hits / n_total) * 100.0 if n_total > 0 else 0.0
    confluence_mae = sum(s["confluence_err"] for s in step_records) / n_total if n_total > 0 else 0.0
    confluence_penalty_loss = _calc_score(step_records, "confluence_err", "confluence_hit")

    # Operational Grand Champion: Crowned Champion (blended score winner)
    tournament_summary = {
        "champion": champion,
        "champion_label": champion_label,
        "grand_champion_key": grand_champion_key,
        "ml_champion_type": ml_champion_type,
        "convex_weight_ml": convex_weight_ml,
        "convex_weight_prophet": convex_weight_prophet,
        "selection_window_sessions": recent_window,
        "champion_dir_hits": champion_dir_hits,
        "champion_dir_hit_rate_pct": round(champion_dir_hit_rate_pct, 1),
        "runner_up_dir_hit_rate_pct": round(runner_up_dir_hit_rate_pct, 1),
        "champion_mae_pct": round(champion_mae_pct, 2),
        "champion_30d_hits": champion_dir_hits,
        "champion_30d_hit_rate_pct": round(champion_dir_hit_rate_pct, 1),
        "champion_30d_mae_pct": round(champion_mae_pct, 2),
        "champion_30d_penalty_loss": round(champion_penalty_loss, 2),
        "champion_full_hits": confluence_hits,
        "champion_full_hit_rate_pct": round(confluence_hit_rate, 1),
        "champion_full_mae_pct": round(confluence_mae, 2),
        "champion_full_penalty_loss": round(confluence_penalty_loss, 2),
        "confluence_30d_hits": confluence_30d_hits,
        "confluence_30d_hit_rate_pct": round(confluence_30d_hit_rate, 1),
        "confluence_30d_mae_pct": round(confluence_30d_mae, 2),
        "confluence_30d_penalty_loss": round(confluence_30d_penalty_loss, 2),
        "ml_dir_hits": ml_candidate_30d_hits,
        "ml_dir_hit_rate_pct": round(ml_candidate_30d_hit_rate, 1),
        "ml_hit_rate_pct": round(ml_candidate_30d_hit_rate, 1),
        "ml_penalty_loss_30d": round(ml_candidate_30d_penalty, 2),
        "ml_full_hits": ml_candidate_hits,
        "ml_full_hit_rate_pct": round(ml_candidate_hit_rate, 1),
        "ml_full_mae_pct": round(ml_candidate_mae, 2),
        "ml_full_penalty_loss": round(ml_candidate_penalty, 2),
        "prophet_dir_hits": prophet_30d_hits,
        "prophet_dir_hit_rate_pct": round(prophet_30d_hit_rate, 1),
        "prophet_hit_rate_pct": round(prophet_30d_hit_rate, 1),
        "prophet_penalty_loss_30d": round(prophet_30d_penalty_loss, 2),
        "prophet_full_hits": prophet_hits,
        "prophet_full_hit_rate_pct": round(prophet_hit_rate, 1),
        "prophet_full_mae_pct": round(prophet_mae, 2),
        "prophet_full_penalty_loss": round(prophet_penalty_loss, 2),
        "ridge_30d_hit_rate_pct": round(candidates_meta["Ridge"]["hit_rate_30d"], 1),
        "ridge_dir_hits": candidates_meta["Ridge"]["hits_30d"],
        "ridge_dir_hit_rate_pct": round(candidates_meta["Ridge"]["hit_rate_30d"], 1),
        "ridge_hit_rate_pct": round(candidates_meta["Ridge"]["hit_rate"], 1),
        "ridge_mae_pct": round(candidates_meta["Ridge"]["mae_30d"], 2),
        "ridge_penalty_loss_30d": round(candidates_meta["Ridge"]["penalty_loss_30d"], 2),
        "xgboost_30d_hit_rate_pct": round(candidates_meta["XGBoost"]["hit_rate_30d"], 1),
        "xgboost_dir_hits": candidates_meta["XGBoost"]["hits_30d"],
        "xgboost_dir_hit_rate_pct": round(candidates_meta["XGBoost"]["hit_rate_30d"], 1),
        "xgboost_hit_rate_pct": round(candidates_meta["XGBoost"]["hit_rate"], 1),
        "xgboost_mae_pct": round(candidates_meta["XGBoost"]["mae_30d"], 2),
        "xgboost_penalty_loss_30d": round(candidates_meta["XGBoost"]["penalty_loss_30d"], 2),
        "lightgbm_30d_hit_rate_pct": round(candidates_meta["LightGBM"]["hit_rate_30d"], 1),
        "lightgbm_dir_hits": candidates_meta["LightGBM"]["hits_30d"],
        "lightgbm_dir_hit_rate_pct": round(candidates_meta["LightGBM"]["hit_rate_30d"], 1),
        "lightgbm_hit_rate_pct": round(candidates_meta["LightGBM"]["hit_rate"], 1),
        "lightgbm_mae_pct": round(candidates_meta["LightGBM"]["mae_30d"], 2),
        "lightgbm_penalty_loss_30d": round(candidates_meta["LightGBM"]["penalty_loss_30d"], 2),
        "huber_30d_hit_rate_pct": round(candidates_meta["Huber"]["hit_rate_30d"], 1),
        "huber_dir_hits": candidates_meta["Huber"]["hits_30d"],
        "huber_dir_hit_rate_pct": round(candidates_meta["Huber"]["hit_rate_30d"], 1),
        "huber_hit_rate_pct": round(candidates_meta["Huber"]["hit_rate"], 1),
        "huber_mae_pct": round(candidates_meta["Huber"]["mae_30d"], 2),
        "huber_penalty_loss_30d": round(candidates_meta["Huber"]["penalty_loss_30d"], 2),
        "bayesian_ridge_30d_hit_rate_pct": round(candidates_meta["BayesianRidge"]["hit_rate_30d"], 1),
        "bayesian_ridge_dir_hits": candidates_meta["BayesianRidge"]["hits_30d"],
        "bayesian_ridge_dir_hit_rate_pct": round(candidates_meta["BayesianRidge"]["hit_rate_30d"], 1),
        "bayesian_ridge_hit_rate_pct": round(candidates_meta["BayesianRidge"]["hit_rate"], 1),
        "bayesian_ridge_mae_pct": round(candidates_meta["BayesianRidge"]["mae_30d"], 2),
        "bayesian_ridge_penalty_loss_30d": round(candidates_meta["BayesianRidge"]["penalty_loss_30d"], 2),
        "ml_mae_pct": round(ml_candidate_30d_mae, 2),
        "prophet_mae_pct": round(prophet_30d_mae, 2),
        "ml_error_wins": ml_error_wins,
        "prophet_error_wins": prophet_error_wins,
        "ml_wins": ml_error_wins,
        "prophet_wins": prophet_error_wins,
        "training_lookback_sessions": train_lookback_sessions,
        "total_sessions": len(ledger),
    }

    return ledger, tournament_summary, champion_ml_model


def get_calibrated_model_selection(db: PostgresManager | None, symbol: str) -> dict[str, Any] | None:
    """Fetch latest crowned (model, training_window) from PostgreSQL or YAML config.

    Ensures production pipelines and the frontend automatically use the optimal
    champion configuration crowned during the periodic calibration competition.
    """
    sym = symbol.upper()
    # 1. Try PostgreSQL gold_tertip_daily_forecasts
    if db is not None:
        try:
            query = """
            SELECT ml_champion_type, crowned_horizon, training_lookback_sessions,
                   champion_dir_hits, champion_dir_hit_rate_pct, champion_mae_pct
            FROM gold_tertip_daily_forecasts
            WHERE symbol = %(symbol)s
            ORDER BY as_of_date DESC, calculated_at DESC
            LIMIT 1;
            """
            row = db.execute(query, {"symbol": sym}).fetchone()
            if row and row[0]:
                return {
                    "ml_champion_type": str(row[0]),
                    "crowned_horizon": str(row[1] or "12m"),
                    "training_lookback_sessions": int(row[2] or 252),
                    "champion_dir_hits": int(row[3]) if row[3] is not None else None,
                    "champion_dir_hit_rate_pct": float(row[4]) if row[4] is not None else None,
                    "champion_mae_pct": float(row[5]) if row[5] is not None else None,
                    "source": "database",
                }
        except Exception as e:
            logger.debug(f"DB lookup for calibrated model {sym} failed: {e}")

    # 2. Fall back to config/tertip_crowned_models.yaml
    yaml_path = Path(__file__).resolve().parents[3] / "config" / "tertip_crowned_models.yaml"
    if yaml_path.exists():
        try:
            import yaml
            with open(yaml_path, "r") as f:
                data = yaml.safe_load(f) or {}
            symbols_meta = data.get("symbols", {})
            if sym in symbols_meta:
                m_info = symbols_meta[sym]
                return {
                    "ml_champion_type": str(m_info.get("model", "Ridge")),
                    "crowned_horizon": str(m_info.get("horizon", "12m")),
                    "training_lookback_sessions": int(m_info.get("training_lookback_sessions", 252)),
                    "champion_dir_hits": m_info.get("recent_30d_hits"),
                    "champion_dir_hit_rate_pct": m_info.get("recent_30d_hit_rate_pct"),
                    "champion_mae_pct": m_info.get("recent_30d_mae_pct"),
                    "champion_penalty_loss": m_info.get("recent_30d_penalty_loss"),
                    "champion_30d_penalty_loss": m_info.get("recent_30d_penalty_loss"),
                    "source": "yaml",
                }
        except Exception as e:
            logger.debug(f"YAML lookup for calibrated model {sym} failed: {e}")

    return None


def get_tertip_ml_forecast(
    db: PostgresManager,
    symbol: str,
    force_refresh: bool = False,
    model_type: str = "auto",
    feature_cols: list[str] | None = None,
    train_lookback_sessions: int = TRAIN_LOOKBACK_SESSIONS,
    n_eval_sessions: int = 30,
    lookback_mode: str = "default",
    selected_composition: str = "champion",
    **kwargs: Any,
) -> dict[str, Any]:
    """Generate live upcoming session (T+1) forecast and walk-forward track.

    Uses strictly zero-lookahead historical training lookback,
    compares Prophet Base vs Tertip ML Challenger with dynamic on-the-fly champion selection (Ridge vs XGBoost),
    and utilizes the lean 21-feature microstructure suite.

    Supports lookback_mode:
    - 'default': Automatically applies the crowned champion configuration (model + training window)
      calibrated from the periodic competition, providing sub-second live inference.
    - '12m': Trailing 12 months (252 sessions)
    - '6m': Trailing 6 months (126 sessions)
    - '3m': Trailing 3 months (63 sessions)
    - 'auto': Runs a multi-horizon tournament across [3M, 6M, 12M] and crowns the best horizon.
    """
    sym = symbol.upper()
    m_type = model_type.lower()
    lb_mode = str(lookback_mode).lower()
    now_ts = datetime.now(timezone.utc).timestamp()
    cache_key = f"{sym}:{m_type}:{train_lookback_sessions}:{n_eval_sessions}:{lb_mode}"

    # Check cache (bypassed if force_refresh is True)
    if not force_refresh and cache_key in _FORECAST_CACHE:
        cached_ts, cached_data = _FORECAST_CACHE[cache_key]
        if (now_ts - cached_ts) < CACHE_TTL_SECONDS:
            return cached_data

    # Resolve active feature set (strictly lean 21 features)
    active_features = list(feature_cols) if feature_cols is not None else list(FEATURE_COLS)

    # Determine needed calendar days to cover eval sessions + training lookback + burn-in
    max_train_lb = 252
    needed_calendar_days = max(550, int((n_eval_sessions + max_train_lb + 75) * 1.5))
    df = extract_3pillar_time_series(db, sym, lookback_days=needed_calendar_days)
    if df.empty or len(df) < (n_eval_sessions + 35):
        return {"error": f"Insufficient historical data for symbol {sym}"}

    # Ensure rolling Prophet baseline is attached
    df = attach_prophet_rolling_features(df, sym, n_history_needed=n_eval_sessions + 40, db=db)

    # Multi-Horizon Tournament or Static Horizon Selection
    crowned_horizon = "12m"
    horizon_comparison: dict[str, Any] = {}

    if lb_mode == "auto":
        candidate_horizons = {"3m": 63, "6m": 126, "12m": 252}
        horizon_results: dict[str, Any] = {}
        for h_key, h_lb in candidate_horizons.items():
            h_ledger, h_summary, _ = run_30d_walk_forward_arena(
                df,
                n_sessions=n_eval_sessions,
                model_type=m_type,
                feature_cols=active_features,
                train_lookback_sessions=h_lb,
            )
            horizon_results[h_key] = {
                "lookback_sessions": h_lb,
                "ml_champion_type": h_summary.get("ml_champion_type"),
                "champion_dir_hits": h_summary.get("champion_dir_hits"),
                "champion_dir_hit_rate_pct": h_summary.get("champion_dir_hit_rate_pct"),
                "champion_mae_pct": h_summary.get("champion_mae_pct"),
                "champion_30d_penalty_loss": h_summary.get("champion_30d_penalty_loss"),
                "champion_penalty_loss": h_summary.get("champion_penalty_loss"),
                "ml_dir_hit_rate_pct": h_summary.get("ml_dir_hit_rate_pct"),
                "ml_mae_pct": h_summary.get("ml_mae_pct"),
                "ml_penalty_loss_30d": h_summary.get("ml_penalty_loss_30d"),
                "prophet_dir_hit_rate_pct": h_summary.get("prophet_dir_hit_rate_pct"),
                "prophet_penalty_loss_30d": h_summary.get("prophet_penalty_loss_30d"),
                "ledger": h_ledger,
                "summary": h_summary,
            }

        # Crown best horizon based on Option 2: lowest 30-day directional penalty loss from today
        best_h_key = min(
            candidate_horizons.keys(),
            key=lambda k: (
                horizon_results[k]["summary"].get("champion_30d_penalty_loss", 999.0),
                -(horizon_results[k]["summary"].get("champion_30d_hit_rate_pct", 0.0)),
                horizon_results[k]["summary"].get("champion_30d_mae_pct", 999.0),
            ),
        )
        crowned_horizon = best_h_key
        train_lookback_sessions = candidate_horizons[best_h_key]
        ledger = horizon_results[best_h_key]["ledger"]
        tournament = horizon_results[best_h_key]["summary"]
        horizon_comparison = {k: {sk: sv for sk, sv in v.items() if sk != "ledger"} for k, v in horizon_results.items()}
    elif lb_mode in ("3m", "63"):
        train_lookback_sessions = 63
        crowned_horizon = "3m"
        ledger, tournament, _ = run_30d_walk_forward_arena(
            df,
            n_sessions=n_eval_sessions,
            model_type=m_type,
            feature_cols=active_features,
            train_lookback_sessions=train_lookback_sessions,
        )
    elif lb_mode in ("6m", "126"):
        train_lookback_sessions = 126
        crowned_horizon = "6m"
        ledger, tournament, _ = run_30d_walk_forward_arena(
            df,
            n_sessions=n_eval_sessions,
            model_type=m_type,
            feature_cols=active_features,
            train_lookback_sessions=train_lookback_sessions,
        )
    elif lb_mode in ("12m", "252"):
        train_lookback_sessions = 252
        crowned_horizon = "12m"
        ledger, tournament, _ = run_30d_walk_forward_arena(
            df,
            n_sessions=n_eval_sessions,
            model_type=m_type,
            feature_cols=active_features,
            train_lookback_sessions=train_lookback_sessions,
        )
    else:
        # 'default': Automatically apply pre-calibrated champion configuration (model + horizon)
        calibrated = get_calibrated_model_selection(db, sym)
        if calibrated:
            train_lookback_sessions = calibrated["training_lookback_sessions"]
            crowned_horizon = calibrated["crowned_horizon"]
        else:
            train_lookback_sessions = 252
            crowned_horizon = "12m"

        ledger, tournament, _ = run_30d_walk_forward_arena(
            df,
            n_sessions=n_eval_sessions,
            model_type=m_type,
            feature_cols=active_features,
            train_lookback_sessions=train_lookback_sessions,
        )

    tournament["crowned_horizon"] = crowned_horizon
    tournament["training_lookback_sessions"] = train_lookback_sessions
    if horizon_comparison:
        tournament["horizon_comparison"] = horizon_comparison

    # Hydrate full multi-month ledger (up to 180 sessions) if available in gold_tertip_walk_forward_backtests
    try:
        cur_res = db.execute("""
            SELECT trade_date, actual_price, actual_return_pct, bist30_ret_pct,
                   ml_pred_price, ml_pred_return_pct, ml_direction, ml_err_pct, ml_is_hit,
                   prophet_pred_price, prophet_pred_return_pct, prophet_direction, prophet_err_pct, prophet_is_hit,
                   winner, is_shock_day, shock_type, days_since_pos_shock, days_since_neg_shock,
                   mlb_action, mlb_flow_tl, mlb_buy_tl, mlb_sell_tl, mlb_pnl_tl,
                   big5_action, big5_flow_tl, big5_buy_tl, big5_sell_tl, big5_pnl_tl,
                   kamu_action, kamu_flow_tl, kamu_buy_tl, kamu_sell_tl, kamu_pnl_tl,
                   training_lookback_sessions
            FROM gold_tertip_walk_forward_backtests
            WHERE symbol = %s
            ORDER BY trade_date ASC
        """, (sym,))
        db_rows = cur_res.fetchall()
        if db_rows and len(db_rows) > len(ledger):
                c_w_ml = float(tournament.get("convex_weight_ml", 0.70))
                c_w_p = float(tournament.get("convex_weight_prophet", 0.30))
                full_ledger = []
                for r in db_rows:
                    act_price = float(r[1])
                    act_ret = float(r[2])
                    ml_ret_row = float(r[5] or 0.0)
                    p_ret_row = float(r[10] or 0.0)
                    c_ret_row = c_w_ml * ml_ret_row + c_w_p * p_ret_row
                    prev_p_row = act_price / (1.0 + act_ret / 100.0) if (1.0 + act_ret / 100.0) != 0 else act_price
                    c_price_row = prev_p_row * (1.0 + c_ret_row / 100.0)
                    c_err_row = abs(c_price_row - act_price) / act_price * 100.0 if act_price > 0 else 0.0
                    c_hit_row = _check_hit(c_ret_row, act_ret)
                    c_dir_row = "UP" if c_ret_row > DEADBAND_PCT else ("DOWN" if c_ret_row < -DEADBAND_PCT else "FLAT")

                    full_ledger.append({
                        "date": str(r[0]).split(" ")[0],
                        "actual_price": round(act_price, 2),
                        "actual_return_pct": round(act_ret, 2),
                        "bist30_ret_pct": round(float(r[3] or 0.0), 2),
                        "confluence_pred_price": round(c_price_row, 2),
                        "confluence_pred_return_pct": round(c_ret_row, 2),
                        "confluence_direction": c_dir_row,
                        "confluence_err_pct": round(c_err_row, 2),
                        "confluence_is_hit": bool(c_hit_row),
                        "convex_weight_ml": c_w_ml,
                        "convex_weight_prophet": c_w_p,
                        "ml_pred_price": round(float(r[4] or 0.0), 2),
                        "ml_pred_return_pct": round(ml_ret_row, 2),
                        "ml_direction": r[6],
                        "ml_err_pct": round(float(r[7] or 0.0), 2),
                        "ml_is_hit": bool(r[8]),
                        "prophet_pred_price": round(float(r[9] or 0.0), 2),
                        "prophet_pred_return_pct": round(p_ret_row, 2),
                        "prophet_direction": r[11],
                        "prophet_err_pct": round(float(r[12] or 0.0), 2),
                        "prophet_is_hit": bool(r[13]),
                        "winner": r[14],
                        "is_shock_day": bool(r[15]),
                        "shock_type": r[16],
                        "days_since_pos_shock": int(r[17] or 63),
                        "days_since_neg_shock": int(r[18] or 63),
                        "mlb_action": r[19],
                        "mlb_flow_tl": round(float(r[20] or 0.0), 1),
                        "mlb_buy_tl": round(float(r[21] or 0.0), 1),
                        "mlb_sell_tl": round(float(r[22] or 0.0), 1),
                        "mlb_pnl_tl": round(float(r[23] or 0.0), 1),
                        "big5_action": r[24],
                        "big5_flow_tl": round(float(r[25] or 0.0), 1),
                        "big5_buy_tl": round(float(r[26] or 0.0), 1),
                        "big5_sell_tl": round(float(r[27] or 0.0), 1),
                        "big5_pnl_tl": round(float(r[28] or 0.0), 1),
                        "kamu_action": r[29],
                        "kamu_flow_tl": round(float(r[30] or 0.0), 1),
                        "kamu_buy_tl": round(float(r[31] or 0.0), 1),
                        "kamu_sell_tl": round(float(r[32] or 0.0), 1),
                        "kamu_pnl_tl": round(float(r[33] or 0.0), 1),
                        "training_lookback_sessions": int(r[34] or train_lookback_sessions),
                    })
                ledger = full_ledger
    except Exception as e:
        logger.warning(f"Could not load full historical backtests from DB for {sym}: {e}")

    # 3-Pillar Matrix
    pillar_matrix = get_pillar_live_matrix(db, sym)

    # Generate Live Forecast for Tomorrow (T+1)
    latest_row = df.iloc[-1]
    latest_price = float(latest_row["close_price"])
    latest_date_str = str(latest_row["trade_date"]).split(" ")[0]

    # 1. Base Prophet Prediction for T+1 (fitted on trailing 24 months / 504 sessions + holidays + shocks)
    start_hist = max(0, len(df) - PROPHET_LOOKBACK_SESSIONS)
    p_df = pd.DataFrame({
        "ds": pd.to_datetime(df["trade_date"].iloc[start_hist:]),
        "y": df["close_price"].iloc[start_hist:],
    })
    holidays_df = get_prophet_holidays(db)
    m_prophet = Prophet(
        holidays=holidays_df if not holidays_df.empty else None,
        daily_seasonality=False,
        weekly_seasonality=True,
        yearly_seasonality=True,
        changepoint_range=0.98,
        changepoint_prior_scale=0.15,
        uncertainty_samples=0,
    )
    m_prophet.fit(p_df)
    fut = m_prophet.make_future_dataframe(periods=1)
    fc_prophet = m_prophet.predict(fut.iloc[-2:])
    yhat_T = float(fc_prophet.iloc[0]["yhat"])
    yhat_T1 = float(fc_prophet.iloc[1]["yhat"])
    prophet_drift_ret = (yhat_T1 - yhat_T) / yhat_T * 100.0 if yhat_T > 0 else 0.0
    prophet_ret_pct = float(np.clip(prophet_drift_ret, -25.0, 25.0))
    prophet_target_price = latest_price * (1.0 + prophet_ret_pct / 100.0)

    # 2. ML Challenger Prediction for T+1 (fitted on trailing 12 months / 252 sessions up to session T)
    tr_full = df.iloc[-train_lookback_sessions:].dropna(subset=active_features).copy()
    y_tr_full = (tr_full["close_price"].shift(-1) - tr_full["close_price"]) / tr_full["close_price"] * 100.0
    X_tr_full = tr_full[active_features].iloc[:-1]
    y_tr_full = y_tr_full.iloc[:-1]

    ml_champion_type = tournament.get("ml_champion_type", "Ridge")
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
    elif ml_champion_type == "LightGBM":
        live_ml = lgb.LGBMRegressor(
            n_estimators=45,
            max_depth=3,
            learning_rate=0.03,
            num_leaves=7,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=42,
            n_jobs=1,
            verbose=-1,
        )
    elif ml_champion_type == "Huber":
        live_ml = Pipeline([
            ("scaler", StandardScaler()),
            ("regressor", HuberRegressor(epsilon=1.35, alpha=10.0, max_iter=300)),
        ])
    elif ml_champion_type == "BayesianRidge":
        live_ml = Pipeline([
            ("scaler", StandardScaler()),
            ("regressor", BayesianRidge(max_iter=300)),
        ])
    else:
        live_ml = Ridge(alpha=10.0, random_state=42)

    live_ml.fit(X_tr_full, y_tr_full)
    pred_ret_ml = float(live_ml.predict(pd.DataFrame([latest_row[active_features]]))[0])
    pred_ret_ml = max(-10.0, min(10.0, pred_ret_ml))

    ml_target_price = latest_price * (1.0 + pred_ret_ml / 100.0)

    # Headline Operational Forecast is the Pure Crowned Champion (blended score winner, zero convex dilution)
    grand_champ = tournament.get("champion", "TERTIP_ML_CHALLENGER")
    if grand_champ == "PROPHET_BASE":
        champion_target_price = round(prophet_target_price, 2)
        champion_pred_ret = round(prophet_ret_pct, 2)
    else:
        champion_target_price = round(ml_target_price, 2)
        champion_pred_ret = round(pred_ret_ml, 2)

    champion = grand_champ
    champion_label = tournament.get("champion_label", f"Champion ({grand_champ})")
    confluence_target_price = champion_target_price
    confluence_pred_ret = champion_pred_ret
    convex_w_ml = float(tournament.get("convex_weight_ml", 0.70))
    convex_w_p = float(tournament.get("convex_weight_prophet", 0.30))

    # Range envelope (using 20-day historical volatility)
    vol_20d = float(df["daily_return_pct"].iloc[-20:].std() * 100.0) if len(df) >= 20 else 2.5
    price_low = champion_target_price * (1.0 - (1.645 * vol_20d / 100.0))
    price_high = champion_target_price * (1.0 + (1.645 * vol_20d / 100.0))

    def _derive_stance(ret_val: float) -> tuple[str, str, str]:
        if ret_val >= 2.0:
            return "STRONG_BUY", "STRONG BUY ACCUMULATION", "emerald"
        elif ret_val > DEADBAND_PCT:
            return "BUY", "BUY ABSORPTION REBOUND", "teal"
        elif ret_val <= -2.0:
            return "STRONG_SELL", "STRONG SELL PRESSURE", "rose"
        elif ret_val < -DEADBAND_PCT:
            return "SELL", "DISTRIBUTION FADE", "orange"
        else:
            return "NEUTRAL", "NEUTRAL CONSOLIDATION", "slate"

    stance, stance_badge, stance_color = _derive_stance(champion_pred_ret)
    ml_stance, ml_stance_badge, ml_stance_color = _derive_stance(pred_ret_ml)
    prophet_stance, prophet_stance_badge, prophet_stance_color = _derive_stance(prophet_ret_pct)

    # Composition selection (user override):
    if selected_composition == "pure_ml":
        target_price = round(ml_target_price, 2)
        expected_ret = round(pred_ret_ml, 2)
        stance, stance_badge, stance_color = ml_stance, ml_stance_badge, ml_stance_color
    elif selected_composition == "pure_prophet":
        target_price = round(prophet_target_price, 2)
        expected_ret = round(prophet_ret_pct, 2)
        stance, stance_badge, stance_color = prophet_stance, prophet_stance_badge, prophet_stance_color
    else:  # "champion", "auto", "convex"
        target_price = champion_target_price
        expected_ret = champion_pred_ret

    # Actionable Blueprint
    playbook_headline = (
        f"Projecting {stance_badge} towards ₺{target_price:.2f} ({expected_ret:+.2f}%)"
    )
    playbook_rationale = (
        f"As of {latest_date_str}, {sym} closed at ₺{latest_price:.2f}. "
        f"The 30-day walk-forward arena crowns {champion_label} "
        f"based on superior directional accuracy ({tournament.get('champion_dir_hits', 0)} of {tournament.get('total_sessions', 30)} sessions directionally correct, "
        f"{tournament.get('champion_dir_hit_rate_pct', 0.0):.1f}% vs runner-up {tournament.get('runner_up_dir_hit_rate_pct', 0.0):.1f}%, MAE: {tournament.get('champion_mae_pct', 0.0):.2f}%). "
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
        "champion": champion,
        "as_of_date": latest_date_str,
        "latest_close_price": round(latest_price, 2),
        "target_price": round(target_price, 2),
        "expected_return_pct": round(expected_ret, 2),
        "champion_target_price": round(champion_target_price, 2),
        "champion_expected_return_pct": round(champion_pred_ret, 2),
        "confluence_target_price": round(confluence_target_price, 2),
        "confluence_expected_return_pct": round(confluence_pred_ret, 2),
        "convex_weight_ml": convex_w_ml,
        "convex_weight_prophet": convex_w_p,
        "ml_target_price": round(ml_target_price, 2),
        "ml_expected_return_pct": round(pred_ret_ml, 2),
        "ml_stance": ml_stance,
        "ml_stance_badge": ml_stance_badge,
        "ml_stance_color": ml_stance_color,
        "price_low": round(price_low, 2),
        "price_high": round(price_high, 2),
        "stance": stance,
        "stance_badge": stance_badge,
        "stance_color": stance_color,
        "prophet_target_price": round(prophet_target_price, 2),
        "prophet_expected_return_pct": round(prophet_ret_pct, 2),
        "prophet_stance": prophet_stance,
        "prophet_stance_badge": prophet_stance_badge,
        "prophet_stance_color": prophet_stance_color,
        "days_since_last_positive_shock": days_pos_shock,
        "days_since_last_negative_shock": days_neg_shock,
        "bist30_trend": bist30_trend,
        "playbook_headline": playbook_headline,
        "playbook_rationale": playbook_rationale,
        "features_mode": "lean",
        "active_features_count": len(active_features),
        "train_lookback_sessions": train_lookback_sessions,
        "active_features": active_features,
        "tournament_summary": tournament,
        "pillar_matrix": pillar_matrix,
        "pillar_execution": pillar_execution,
        "walk_forward_ledger": ledger,
        "calculated_at": datetime.now(timezone.utc).isoformat(),
    }

    _FORECAST_CACHE[cache_key] = (now_ts, response_data)
    return response_data
