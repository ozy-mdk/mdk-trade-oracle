"""Tertip Intelligence & Action Forensics Analytical Engine.

Connects an institutional desk's single-day trading execution with their multi-horizon
underlying Tertip (inventory / custody posture), Exponentially Weighted Moving Average (EWMA)
positioning ribbons, and cost-basis PnL pressure.
"""

from datetime import date, datetime, timezone
from typing import Any, Optional

import polars as pl

from mdk_trading_oracle.core.db import PostgresManager
from mdk_trading_oracle.core.ewma import (
    add_multi_horizon_ewma_polars,
    add_multi_horizon_size_weighted_ewma_polars,
    calculate_bounded_ewma,
    calculate_size_weighted_bounded_ewma,
)
from mdk_trading_oracle.core.logger import get_logger

logger = get_logger("mdk_oracle.tertip_analytics")

# Big Five Institutional Bundle (Next 5 largest players excluding Bank of America)
BIG_FIVE_BROKERS = ("YKR", "IYM", "AKM", "GRM", "ZRY")

# State-Backed Institutional Bundle (Kamu Bankaları / TVF Conduits)
KAMU_BROKERS = ("ZRY", "VKY", "HLY")

# Registry of Institutional Bundles
INSTITUTIONAL_BUNDLES = {
    "BIG5": BIG_FIVE_BROKERS,
    "KAMU": KAMU_BROKERS,
}

# Canonical 6 Horizons + 1D Today
HORIZON_DEFINITIONS = [
    {"code": "1W", "label": "1 Week", "days": 5, "desc": "Tactical Inventory Impulse"},
    {"code": "2W", "label": "2 Weeks", "days": 10, "desc": "Bi-Weekly Swing Posture"},
    {"code": "1M", "label": "1 Month", "days": 21, "desc": "Monthly Rebalancing Cycle"},
    {"code": "3M", "label": "3 Months", "days": 63, "desc": "Quarterly Cycle"},
    {"code": "6M", "label": "6 Months", "days": 126, "desc": "Semi-Annual Strategic Allocation"},
    {"code": "12M", "label": "12 Months", "days": 252, "desc": "Core Structural Custody"},
]


def classify_action_diagnostic(
    day_net_flow: float,
    open_qty: float,
    close_price: float,
    fifo_avg_cost: float,
    ewma_5d_qty: float,
    ewma_21d_qty: float,
    ewma_63d_qty: float,
    ewma_126d_qty: float,
    ewma_cost_63d: float,
    unrealized_pnl_pct: float,
    saturation_pct: float,
    matched_volume_pct: float,
) -> dict[str, Any]:
    """Diagnose the institutional rationale driving the trader's session action.

    Evaluates:
        1. Profit Taking / Harvest: Selling into large positive PnL.
        2. Cost Defense: Buying into price testing the average cost basis.
        3. Momentum Expansion: Buying with Golden EWMA alignment and capacity headroom.
        4. De-Grossing / Unwind: Persistent multi-week position reduction.
        5. Liquidity Fade / Scalp: High matched volume with low directional residual.
        6. Capitulation Dump: Selling at deep losses breaking support.
        7. Stable Core Hold: Minimal net flow near strategic baseline.
    """
    cost_spread_pct = ((close_price - fifo_avg_cost) / fifo_avg_cost * 100.0) if fifo_avg_cost > 0 else 0.0
    cost_63d_spread_pct = ((close_price - ewma_cost_63d) / ewma_cost_63d * 100.0) if ewma_cost_63d > 0 else 0.0

    # 1. Profit Harvest: Net seller into gains, 5d EWMA dropping below 21d EWMA
    if day_net_flow <= -25_000_000 and (unrealized_pnl_pct >= 8.0 or cost_spread_pct >= 8.0 or cost_63d_spread_pct >= 8.0) and open_qty > 0:
        headline = f"Locking in profits into +{max(unrealized_pnl_pct, cost_spread_pct, cost_63d_spread_pct):.1f}% gain"
        rationale = (
            f"The desk is sitting on substantial unrealized gains (+{unrealized_pnl_pct:.1f}%). "
            f"With 5-day EWMA inventory ({ewma_5d_qty:,.0f} sh) contracting below medium-term levels, "
            f"today's net selling (₺{day_net_flow / 1e6:.1f}M) represents an orderly profit-taking program."
        )
        return {
            "diagnostic_badge": "PROFIT_HARVEST",
            "badge_color": "emerald",
            "conviction_pct": 88,
            "headline": headline,
            "rationale": rationale,
        }

    # 2. Defense Support: Net buyer when price is near average unit cost (+- 3%)
    if day_net_flow >= 25_000_000 and abs(cost_spread_pct) <= 3.5 and open_qty > 0:
        headline = f"Defending institutional unit cost at ₺{fifo_avg_cost:.2f}"
        rationale = (
            f"Price pulled back directly into the desk's average FIFO cost basis (₺{fifo_avg_cost:.2f}, spread: {cost_spread_pct:+.1f}%). "
            f"The desk actively absorbed selling pressure with ₺{day_net_flow / 1e6:.1f}M net buying to protect their inventory."
        )
        return {
            "diagnostic_badge": "DEFENSE_SUPPORT",
            "badge_color": "cyan",
            "conviction_pct": 85,
            "headline": headline,
            "rationale": rationale,
        }

    # 3. Momentum Expansion: Strong buyer, fast EWMA > slow EWMA, room in saturation
    if day_net_flow >= 40_000_000 and ewma_5d_qty >= ewma_21d_qty and saturation_pct < 85.0:
        headline = "Aggressive momentum expansion with available capacity"
        rationale = (
            f"The desk shows Golden EWMA alignment (5d: {ewma_5d_qty:,.0f} > 21d: {ewma_21d_qty:,.0f} sh) with "
            f"{saturation_pct:.0f}% saturation. Today's ₺{day_net_flow / 1e6:.1f}M net buy flow is pressing momentum into continuation."
        )
        return {
            "diagnostic_badge": "MOMENTUM_EXPANSION",
            "badge_color": "indigo",
            "conviction_pct": 84,
            "headline": headline,
            "rationale": rationale,
        }

    # 4. De-Grossing / Unwind: Persistent selling, EWMA 5d < 21d < 63d
    if day_net_flow <= -20_000_000 and ewma_5d_qty < ewma_21d_qty <= ewma_63d_qty:
        headline = "Systematic multi-week inventory de-grossing"
        rationale = (
            f"Inventory EWMA ribbons indicate a sustained unwinding trajectory (5d < 21d < 63d). "
            f"Today's net selling of ₺{day_net_flow / 1e6:.1f}M is part of a broader risk-reduction program."
        )
        return {
            "diagnostic_badge": "DE_GROSSING_UNWIND",
            "badge_color": "amber",
            "conviction_pct": 82,
            "headline": headline,
            "rationale": rationale,
        }

    # 5. Capitulation Dump: Heavy selling at deep losses (< -8%)
    if day_net_flow <= -40_000_000 and cost_spread_pct <= -8.0:
        headline = f"Forced de-leveraging / stop-loss dump at {cost_spread_pct:.1f}% loss"
        rationale = (
            f"The position is severely underwater ({cost_spread_pct:.1f}% vs cost basis). "
            f"Heavy net selling (₺{day_net_flow / 1e6:.1f}M) indicates institutional stop-loss liquidation."
        )
        return {
            "diagnostic_badge": "CAPITULATION_DUMP",
            "badge_color": "rose",
            "conviction_pct": 89,
            "headline": headline,
            "rationale": rationale,
        }

    # 6. Liquidity Fade / Scalp: High matched volume, small net flow
    if matched_volume_pct >= 55.0 and abs(day_net_flow) < 30_000_000:
        headline = f"Algorithmic two-sided market making ({matched_volume_pct:.0f}% matched)"
        rationale = (
            f"Trading activity was dominated by intraday churn ({matched_volume_pct:.0f}% matched volume) "
            f"with low residual directional flow (₺{day_net_flow / 1e6:.1f}M), reflecting rebate capture and market making."
        )
        return {
            "diagnostic_badge": "LIQUIDITY_FADE",
            "badge_color": "slate",
            "conviction_pct": 75,
            "headline": headline,
            "rationale": rationale,
        }

    # 7. Stable Core Hold
    headline = "Passive custody holding near strategic baseline"
    rationale = (
        f"Inventory remains stable relative to the 3-month strategic baseline ({ewma_63d_qty:,.0f} sh) "
        f"with balanced daily flow (₺{day_net_flow / 1e6:.1f}M)."
    )
    return {
        "diagnostic_badge": "STABLE_CORE_HOLD",
        "badge_color": "slate",
        "conviction_pct": 70,
        "headline": headline,
        "rationale": rationale,
    }


def compute_12m_horizon_realizations(
    df: pl.DataFrame,
    horizons_data: list[dict[str, Any]],
    close_p: float,
    fifo_cost: float,
) -> tuple[dict[str, dict[str, Any]], dict[str, Any], dict[str, Any]]:
    """Compute empirical 12-month rolling realization metrics for horizons and confluence.

    Evaluates how many times each horizon's stance (BUY/SELL/NEUTRAL) and overall confluence
    occurred over the prior 252 sessions (last 12M), and whether the broker followed through
    on session T+1 (realized same direction) or executed in the opposite direction.
    """
    empty_res: dict[str, Any] = {
        "active_stance": "NEUTRAL",
        "total_occurrences": 0,
        "realized_count": 0,
        "opposite_count": 0,
        "neutral_count": 0,
        "realized_pct": 0.0,
        "opposite_pct": 0.0,
        "neutral_pct": 0.0,
        "next_day_buy_count": 0,
        "next_day_sell_count": 0,
        "next_day_buy_pct": 0.0,
        "next_day_sell_pct": 0.0,
        "avg_next_day_flow_tl": 0.0,
        "price_up_count": 0,
        "price_down_count": 0,
        "price_up_pct": 0.0,
    }

    if len(df) < 3 or close_p <= 0:
        return {h["code"]: dict(empty_res) for h in horizons_data}, dict(empty_res), dict(empty_res)

    # Enrich df with multi-horizon size-weighted EWMA costs if not present
    if "ewma_cost_1m_21d" not in df.columns:
        df_ewma = add_multi_horizon_size_weighted_ewma_polars(
            df, value_col="entry_price", size_col="entry_size", prefix="ewma_cost"
        )
    else:
        df_ewma = df

    # Evaluate up to the last 253 sessions (252 evaluation days + 1 for next-day lead)
    lookback = min(len(df_ewma), 253)
    df_slice = df_ewma.tail(lookback).clone()
    df_slice = df_slice.with_columns([
        pl.col("net_flow_tl").shift(-1).alias("next_day_net_flow_tl"),
        pl.col("market_close_price").shift(-1).alias("next_day_close_price"),
    ])
    eval_df = df_slice.slice(0, len(df_slice) - 1)
    if eval_df.is_empty():
        return {h["code"]: dict(empty_res) for h in horizons_data}, dict(empty_res), dict(empty_res)

    eval_df = eval_df.with_columns([
        (((pl.col("next_day_close_price") - pl.col("market_close_price")) / pl.col("market_close_price")) * 100.0).alias("next_day_return_pct")
    ])

    horizon_col_map = {
        "1D": "fifo_avg_cost",
        "1W": "ewma_cost_1w_5d",
        "2W": "ewma_cost_2w_10d",
        "1M": "ewma_cost_1m_21d",
        "3M": "ewma_cost_3m_63d",
        "6M": "ewma_cost_6m_126d",
        "12M": "ewma_cost_12m_252d",
        "FIFO": "fifo_avg_cost",
    }

    def _calc_stats(matches_df: pl.DataFrame, active_dir: str) -> dict[str, Any]:
        tot = len(matches_df)
        if tot == 0:
            res = dict(empty_res)
            res["active_stance"] = active_dir
            return res

        next_buys = len(matches_df.filter(pl.col("next_day_net_flow_tl") > 0))
        next_sells = len(matches_df.filter(pl.col("next_day_net_flow_tl") < 0))
        buy_pct = round((next_buys / tot * 100.0), 1)
        sell_pct = round((next_sells / tot * 100.0), 1)

        if active_dir == "BUY":
            real_cnt, real_pct = next_buys, buy_pct
            opp_cnt, opp_pct = next_sells, sell_pct
        elif active_dir == "SELL":
            real_cnt, real_pct = next_sells, sell_pct
            opp_cnt, opp_pct = next_buys, buy_pct
        else:
            real_cnt, real_pct = 0, 0.0
            opp_cnt, opp_pct = 0, 0.0

        neutral_cnt = tot - next_buys - next_sells
        neutral_pct = round((neutral_cnt / tot * 100.0), 1)
        avg_flow = round(float(matches_df["next_day_net_flow_tl"].mean() or 0.0), 2)
        price_ups = len(matches_df.filter(pl.col("next_day_return_pct") > 0))
        price_downs = len(matches_df.filter(pl.col("next_day_return_pct") < 0))
        price_up_pct = round((price_ups / tot * 100.0), 1)

        return {
            "active_stance": active_dir,
            "total_occurrences": tot,
            "realized_count": real_cnt,
            "opposite_count": opp_cnt,
            "neutral_count": neutral_cnt,
            "realized_pct": real_pct,
            "opposite_pct": opp_pct,
            "neutral_pct": neutral_pct,
            "next_day_buy_count": next_buys,
            "next_day_sell_count": next_sells,
            "next_day_buy_pct": buy_pct,
            "next_day_sell_pct": sell_pct,
            "avg_next_day_flow_tl": avg_flow,
            "price_up_count": price_ups,
            "price_down_count": price_downs,
            "price_up_pct": price_up_pct,
        }

    horizons_res = {}
    for h in horizons_data:
        code = h["code"]
        col_name = horizon_col_map.get(code, "fifo_avg_cost")
        current_cost = float(h.get("ewma_unit_cost") or 0.0)
        spread_c = ((close_p - current_cost) / current_cost * 100.0) if current_cost > 0 else 0.0
        active_dir = "BUY" if spread_c <= -3.0 else ("SELL" if spread_c >= 3.0 else "NEUTRAL")

        if col_name in eval_df.columns:
            sub = eval_df.with_columns([
                (((pl.col("market_close_price") - pl.col(col_name)) / pl.col(col_name)) * 100.0).alias("_h_spread")
            ]).with_columns([
                pl.when(pl.col("_h_spread") <= -3.0).then(pl.lit("BUY"))
                  .when(pl.col("_h_spread") >= 3.0).then(pl.lit("SELL"))
                  .otherwise(pl.lit("NEUTRAL")).alias("_h_dir")
            ])
            matches = sub.filter(pl.col("_h_dir") == active_dir)
            horizons_res[code] = _calc_stats(matches, active_dir)
        else:
            horizons_res[code] = _calc_stats(eval_df.filter(pl.lit(False)), active_dir)

    # Core FIFO realization
    fifo_spread = ((close_p - fifo_cost) / fifo_cost * 100.0) if fifo_cost > 0 else 0.0
    fifo_active_dir = "BUY" if fifo_spread <= -3.0 else ("SELL" if fifo_spread >= 3.0 else "NEUTRAL")
    if "fifo_avg_cost" in eval_df.columns:
        sub_fifo = eval_df.with_columns([
            (((pl.col("market_close_price") - pl.col("fifo_avg_cost")) / pl.col("fifo_avg_cost")) * 100.0).alias("_fifo_spread")
        ]).with_columns([
            pl.when(pl.col("_fifo_spread") <= -3.0).then(pl.lit("BUY"))
              .when(pl.col("_fifo_spread") >= 3.0).then(pl.lit("SELL"))
              .otherwise(pl.lit("NEUTRAL")).alias("_fifo_dir")
        ])
        fifo_res = _calc_stats(sub_fifo.filter(pl.col("_fifo_dir") == fifo_active_dir), fifo_active_dir)
    else:
        fifo_res = _calc_stats(eval_df.filter(pl.lit(False)), fifo_active_dir)

    # Confluence realization across standard multi-horizon suite
    key_h_cols = ["ewma_cost_1w_5d", "ewma_cost_2w_10d", "ewma_cost_1m_21d", "ewma_cost_3m_63d", "ewma_cost_6m_126d", "fifo_avg_cost"]
    avail_cols = [c for c in key_h_cols if c in eval_df.columns]
    if avail_cols:
        buy_exprs = [(((pl.col("market_close_price") - pl.col(c)) / pl.col(c)) * 100.0 <= -3.0).cast(pl.Int32) for c in avail_cols]
        sell_exprs = [(((pl.col("market_close_price") - pl.col(c)) / pl.col(c)) * 100.0 >= 3.0).cast(pl.Int32) for c in avail_cols]
        conf_eval = eval_df.with_columns([
            sum(buy_exprs).alias("_buy_cnt"),
            sum(sell_exprs).alias("_sell_cnt"),
        ]).with_columns([
            pl.when(pl.col("_buy_cnt") >= 3).then(pl.lit("BUY"))
              .when(pl.col("_sell_cnt") >= 3).then(pl.lit("SELL"))
              .otherwise(pl.lit("NEUTRAL")).alias("_conf_dir")
        ])

        latest_buys = sum([1 for c in avail_cols if ((close_p - float(df_ewma[c][-1] or close_p)) / float(df_ewma[c][-1] or close_p) * 100.0) <= -3.0])
        latest_sells = sum([1 for c in avail_cols if ((close_p - float(df_ewma[c][-1] or close_p)) / float(df_ewma[c][-1] or close_p) * 100.0) >= 3.0])
        cur_conf_dir = "BUY" if latest_buys >= 3 and latest_buys > latest_sells else ("SELL" if latest_sells >= 3 and latest_sells > latest_buys else "NEUTRAL")
        confluence_res = _calc_stats(conf_eval.filter(pl.col("_conf_dir") == cur_conf_dir), cur_conf_dir)
    else:
        confluence_res = _calc_stats(eval_df.filter(pl.lit(False)), "NEUTRAL")

    return horizons_res, fifo_res, confluence_res


def get_tertip_horizons_analysis(
    db: PostgresManager,
    symbol: str,
    broker_id: str = "MLB",
    trade_date: Optional[str] = None,
) -> dict[str, Any]:
    """Calculate multi-horizon EWMA metrics and institutional action forensics.

    Returns the session action, the 6-horizon matrix (1W, 2W, 1M, 3M, 6M, 12M + 1D),
    and the automated action diagnostic.
    """
    sym = symbol.upper()
    bid = broker_id.upper()

    # Determine latest available trade date if not specified
    if not trade_date:
        if sym == "XU030":
            d_row = db.execute("SELECT MAX(trade_date) FROM silver_daily_benchmark_index;").fetchone()
        elif bid in INSTITUTIONAL_BUNDLES:
            d_row = db.execute(
                "SELECT MAX(trade_date) FROM silver_broker_fifo_daily WHERE symbol = %s AND broker_id = ANY(%s);",
                [sym, list(INSTITUTIONAL_BUNDLES[bid])],
            ).fetchone()
        else:
            d_row = db.execute(
                "SELECT MAX(trade_date) FROM silver_broker_fifo_daily WHERE symbol = %s AND broker_id = %s;",
                [sym, bid],
            ).fetchone()
        if not d_row or not d_row[0]:
            trade_date = "2026-09-16"
        else:
            trade_date = str(d_row[0])

    # Fetch chronological history up to trade_date
    if sym == "XU030":
        if bid in INSTITUTIONAL_BUNDLES:
            bundle_list = list(INSTITUTIONAL_BUNDLES[bid])
            query = """
                SELECT 
                    f.trade_date,
                    'XU030' AS symbol,
                    %s AS broker_id,
                    SUM(f.buy_volume) AS buy_volume,
                    SUM(f.buy_turnover_tl) AS buy_turnover_tl,
                    SUM(f.sell_volume) AS sell_volume,
                    SUM(f.sell_turnover_tl) AS sell_turnover_tl,
                    SUM(f.buy_turnover_tl) - SUM(f.sell_turnover_tl) AS net_flow_tl,
                    SUM(f.buy_volume) - SUM(f.sell_volume) AS net_volume,
                    SUM(f.buy_turnover_tl) / NULLIF(SUM(f.buy_volume), 0) AS buy_vwap,
                    SUM(f.matched_volume) AS matched_volume,
                    SUM(f.intraday_realized_pnl_tl) AS intraday_realized_pnl_tl,
                    SUM(f.carry_fifo_realized_pnl_tl) AS carry_fifo_realized_pnl_tl,
                    SUM(f.daily_realized_pnl_tl) AS daily_realized_pnl_tl,
                    SUM(f.cumulative_realized_pnl_tl) AS cumulative_realized_pnl_tl,
                    CASE WHEN SUM(f.open_stock_quantity) > 0 THEN 'LONG'
                         WHEN SUM(f.open_stock_quantity) < 0 THEN 'SHORT'
                         ELSE 'FLAT' END AS position_side,
                    SUM(f.open_stock_quantity) AS open_stock_quantity,
                    0.0 AS fifo_avg_cost,
                    MAX(COALESCE(b.close_price, 10000.0)) AS market_close_price,
                    SUM(f.market_value_tl) AS market_value_tl,
                    SUM(f.unrealized_pnl_tl) AS unrealized_pnl_tl
                FROM silver_broker_fifo_daily f
                JOIN silver_daily_stock_summary s ON f.trade_date = s.trade_date AND f.symbol = s.symbol
                LEFT JOIN silver_daily_benchmark_index b ON f.trade_date = b.trade_date
                WHERE s.index_name = 'BIST30' AND f.broker_id = ANY(%s) AND f.trade_date <= %s
                GROUP BY f.trade_date
                ORDER BY f.trade_date ASC;
            """
            df = db.query_pl(query, params=[bid, bundle_list, trade_date])
        else:
            query = """
                SELECT 
                    f.trade_date,
                    'XU030' AS symbol,
                    %s AS broker_id,
                    SUM(f.buy_volume) AS buy_volume,
                    SUM(f.buy_turnover_tl) AS buy_turnover_tl,
                    SUM(f.sell_volume) AS sell_volume,
                    SUM(f.sell_turnover_tl) AS sell_turnover_tl,
                    SUM(f.buy_turnover_tl) - SUM(f.sell_turnover_tl) AS net_flow_tl,
                    SUM(f.buy_volume) - SUM(f.sell_volume) AS net_volume,
                    SUM(f.buy_turnover_tl) / NULLIF(SUM(f.buy_volume), 0) AS buy_vwap,
                    SUM(f.matched_volume) AS matched_volume,
                    SUM(f.intraday_realized_pnl_tl) AS intraday_realized_pnl_tl,
                    SUM(f.carry_fifo_realized_pnl_tl) AS carry_fifo_realized_pnl_tl,
                    SUM(f.daily_realized_pnl_tl) AS daily_realized_pnl_tl,
                    SUM(f.cumulative_realized_pnl_tl) AS cumulative_realized_pnl_tl,
                    CASE WHEN SUM(f.open_stock_quantity) > 0 THEN 'LONG'
                         WHEN SUM(f.open_stock_quantity) < 0 THEN 'SHORT'
                         ELSE 'FLAT' END AS position_side,
                    SUM(f.open_stock_quantity) AS open_stock_quantity,
                    0.0 AS fifo_avg_cost,
                    MAX(COALESCE(b.close_price, 10000.0)) AS market_close_price,
                    SUM(f.market_value_tl) AS market_value_tl,
                    SUM(f.unrealized_pnl_tl) AS unrealized_pnl_tl
                FROM silver_broker_fifo_daily f
                JOIN silver_daily_stock_summary s ON f.trade_date = s.trade_date AND f.symbol = s.symbol
                LEFT JOIN silver_daily_benchmark_index b ON f.trade_date = b.trade_date
                WHERE s.index_name = 'BIST30' AND f.broker_id = %s AND f.trade_date <= %s
                GROUP BY f.trade_date
                ORDER BY f.trade_date ASC;
            """
            df = db.query_pl(query, params=[bid, bid, trade_date])
    elif bid in INSTITUTIONAL_BUNDLES:
        bundle_list = list(INSTITUTIONAL_BUNDLES[bid])
        query = f"""
            SELECT 
                trade_date,
                symbol,
                '{bid}' AS broker_id,
                SUM(buy_volume) AS buy_volume,
                SUM(buy_turnover_tl) AS buy_turnover_tl,
                SUM(sell_volume) AS sell_volume,
                SUM(sell_turnover_tl) AS sell_turnover_tl,
                SUM(buy_turnover_tl) - SUM(sell_turnover_tl) AS net_flow_tl,
                SUM(buy_volume) - SUM(sell_volume) AS net_volume,
                SUM(buy_turnover_tl) / NULLIF(SUM(buy_volume), 0) AS buy_vwap,
                SUM(matched_volume) AS matched_volume,
                SUM(intraday_realized_pnl_tl) AS intraday_realized_pnl_tl,
                SUM(carry_fifo_realized_pnl_tl) AS carry_fifo_realized_pnl_tl,
                SUM(daily_realized_pnl_tl) AS daily_realized_pnl_tl,
                SUM(cumulative_realized_pnl_tl) AS cumulative_realized_pnl_tl,
                CASE WHEN SUM(open_stock_quantity) > 0 THEN 'LONG'
                     WHEN SUM(open_stock_quantity) < 0 THEN 'SHORT'
                     ELSE 'FLAT' END AS position_side,
                SUM(open_stock_quantity) AS open_stock_quantity,
                SUM(open_fifo_cost_tl) / NULLIF(SUM(open_stock_quantity), 0) AS fifo_avg_cost,
                MAX(market_close_price) AS market_close_price,
                SUM(market_value_tl) AS market_value_tl,
                SUM(unrealized_pnl_tl) AS unrealized_pnl_tl
            FROM silver_broker_fifo_daily
            WHERE symbol = %s AND broker_id = ANY(%s) AND trade_date <= %s
            GROUP BY trade_date, symbol
            ORDER BY trade_date ASC;
        """
        df = db.query_pl(query, params=[sym, bundle_list, trade_date])
    else:
        query = """
            SELECT 
                trade_date,
                symbol,
                broker_id,
                buy_volume,
                buy_turnover_tl,
                sell_volume,
                sell_turnover_tl,
                buy_turnover_tl - sell_turnover_tl AS net_flow_tl,
                buy_volume - sell_volume AS net_volume,
                buy_vwap,
                matched_volume,
                intraday_realized_pnl_tl,
                carry_fifo_realized_pnl_tl,
                daily_realized_pnl_tl,
                cumulative_realized_pnl_tl,
                position_side,
                open_stock_quantity,
                fifo_avg_cost,
                market_close_price,
                market_value_tl,
                unrealized_pnl_tl
            FROM silver_broker_fifo_daily
            WHERE symbol = %s AND broker_id = %s AND trade_date <= %s
            ORDER BY trade_date ASC;
        """
        df = db.query_pl(query, params=[sym, bid, trade_date])

    if df.is_empty():
        return {
            "symbol": sym,
            "broker_id": bid,
            "trade_date": trade_date,
            "error": "No historical FIFO data found",
            "horizons": [],
        }

    # Extract latest session row
    latest_row = df.tail(1).to_dicts()[0]

    close_p = float(latest_row["market_close_price"] or 0.0)
    open_qty = float(latest_row["open_stock_quantity"] or 0.0)
    fifo_cost = float(latest_row["fifo_avg_cost"] or 0.0)
    mtm_val = float(latest_row["market_value_tl"] or 0.0)
    unreal_pnl = float(latest_row["unrealized_pnl_tl"] or 0.0)
    unreal_pct = ((close_p - fifo_cost) / fifo_cost * 100.0) if fifo_cost > 0 else 0.0
    day_net_flow = float(latest_row["net_flow_tl"] or 0.0)
    day_buy_tl = float(latest_row["buy_turnover_tl"] or 0.0)
    day_sell_tl = float(latest_row["sell_turnover_tl"] or 0.0)
    matched_vol = float(latest_row["matched_volume"] or 0.0)
    total_vol = float(latest_row["buy_volume"] or 0.0) + float(latest_row["sell_volume"] or 0.0)
    matched_pct = (matched_vol * 2.0 / total_vol * 100.0) if total_vol > 0 else 0.0

    # Prepare daily execution entry price (buy_vwap or close) and size (turnover)
    df = df.with_columns([
        pl.coalesce(pl.col("buy_vwap"), pl.col("market_close_price")).alias("entry_price"),
        pl.coalesce(pl.col("buy_turnover_tl"), pl.lit(1e6)).alias("entry_size"),
    ])

    # Extract historical vectors for bounded EWMA & rolling metrics
    net_flows = df["net_flow_tl"].to_list()
    net_vols = df["net_volume"].to_list()
    quantities = df["open_stock_quantity"].to_list()
    entry_prices = df["entry_price"].to_list()
    entry_sizes = df["entry_size"].to_list()

    n_sessions = len(quantities)

    # 12-Month Rolling Min-Max for Inventory Saturation
    lookback_252 = min(n_sessions, 252)
    qty_slice_252 = quantities[-lookback_252:]
    min_qty_252 = min(qty_slice_252) if qty_slice_252 else 0.0
    max_qty_252 = max(qty_slice_252) if qty_slice_252 else 1.0
    qty_range_252 = max(max_qty_252 - min_qty_252, 1.0)
    global_sat_pct = max(0.0, min(100.0, (open_qty - min_qty_252) / qty_range_252 * 100.0))

    # Compute EWMA for key diagnostic horizons (Size & Recency weighted)
    ewma_5d = calculate_bounded_ewma(quantities, window=5, span=5)
    ewma_21d = calculate_bounded_ewma(quantities, window=21, span=21)
    ewma_63d = calculate_bounded_ewma(quantities, window=63, span=63)
    ewma_126d = calculate_bounded_ewma(quantities, window=126, span=126)
    eff_63 = min(n_sessions, 63)
    ewma_cost_63d = calculate_size_weighted_bounded_ewma(
        entry_prices[-eff_63:], sizes=entry_sizes[-eff_63:], window=63, span=63
    )

    # Calculate Horizon Matrix
    horizons_data = []

    # Prepend 1D (Today)
    horizons_data.append({
        "code": "1D",
        "label": "Today (1D)",
        "lookback_days": 1,
        "cum_net_flow_tl": round(day_net_flow, 2),
        "cum_net_shares": round(float(latest_row["net_volume"] or 0.0), 0),
        "ewma_inventory_qty": round(open_qty, 0),
        "ewma_unit_cost": round(fifo_cost, 2),
        "cost_spread_pct": round(unreal_pct, 2),
        "saturation_pct": round(global_sat_pct, 1),
        "stance": "BUYING" if day_net_flow > 10_000_000 else ("SELLING" if day_net_flow < -10_000_000 else "NEUTRAL"),
        "description": "Current Session Execution",
    })

    # Add 1W, 2W, 1M, 3M, 6M, 12M
    for h in HORIZON_DEFINITIONS:
        w = h["days"]
        eff_w = min(n_sessions, w)

        flow_slice = net_flows[-eff_w:]
        vol_slice = net_vols[-eff_w:]
        q_slice = quantities[-eff_w:]

        cum_flow = sum(flow_slice) if flow_slice else 0.0
        cum_shares = sum(vol_slice) if vol_slice else 0.0

        # EWMA inventory quantity (recency weighted)
        ewma_q = calculate_bounded_ewma(quantities, window=w, span=float(w))
        # EWMA acquisition cost on actual execution entry prices (compounding recency decay WITH position magnitude taken)
        ewma_c = calculate_size_weighted_bounded_ewma(
            entry_prices[-eff_w:], sizes=entry_sizes[-eff_w:], window=w, span=float(w)
        )

        spread_c = ((close_p - ewma_c) / ewma_c * 100.0) if ewma_c > 0 else 0.0

        min_q = min(q_slice) if q_slice else 0.0
        max_q = max(q_slice) if q_slice else 1.0
        r_q = max(max_q - min_q, 1.0)
        sat_p = max(0.0, min(100.0, (open_qty - min_q) / r_q * 100.0))

        # Classify Horizon Stance
        if cum_flow >= 50_000_000:
            st = "ACCUMULATION"
        elif cum_flow <= -50_000_000:
            st = "DISTRIBUTION"
        elif ewma_q > open_qty * 1.05:
            st = "UNWINDING"
        elif ewma_q < open_qty * 0.95:
            st = "EXPANDING"
        else:
            st = "CORE_HOLD"

        horizons_data.append({
            "code": h["code"],
            "label": h["label"],
            "lookback_days": w,
            "cum_net_flow_tl": round(cum_flow, 2),
            "cum_net_shares": round(cum_shares, 0),
            "ewma_inventory_qty": round(ewma_q, 0),
            "ewma_unit_cost": round(ewma_c, 2),
            "cost_spread_pct": round(spread_c, 2),
            "saturation_pct": round(sat_p, 1),
            "stance": st,
            "description": h["desc"],
        })

    # Calculate 12-Month Realization Metrics (Horizon-Specific & Confluence)
    horizons_res, fifo_res, confluence_res = compute_12m_horizon_realizations(
        df=df,
        horizons_data=horizons_data,
        close_p=close_p,
        fifo_cost=fifo_cost,
    )

    # Attach realization stats to each horizon item
    for h in horizons_data:
        h["realization_12m"] = horizons_res.get(h["code"])

    # Run Action Diagnostic
    diagnostic = classify_action_diagnostic(
        day_net_flow=day_net_flow,
        open_qty=open_qty,
        close_price=close_p,
        fifo_avg_cost=fifo_cost,
        ewma_5d_qty=ewma_5d,
        ewma_21d_qty=ewma_21d,
        ewma_63d_qty=ewma_63d,
        ewma_126d_qty=ewma_126d,
        ewma_cost_63d=ewma_cost_63d,
        unrealized_pnl_pct=unreal_pct,
        saturation_pct=global_sat_pct,
        matched_volume_pct=matched_pct,
    )

    # EWMA Ribbon Trend Summary
    if ewma_5d > ewma_21d > ewma_63d:
        ribbon_status = f"Golden Accumulation (+{((ewma_5d - ewma_21d) / ewma_21d * 100.0):.1f}% vs 1M EWMA)"
    elif ewma_5d < ewma_21d < ewma_63d:
        ribbon_status = f"Tactical De-Grossing ({((ewma_5d - ewma_21d) / ewma_21d * 100.0):.1f}% vs 1M EWMA)"
    else:
        ribbon_status = f"Mixed Realignment (5d: {ewma_5d:,.0f} | 1M: {ewma_21d:,.0f} sh)"

    return {
        "symbol": sym,
        "broker_id": bid,
        "trade_date": trade_date,
        "market_close_price": round(close_p, 2),
        "day_net_flow_tl": round(day_net_flow, 2),
        "day_buy_turnover_tl": round(day_buy_tl, 2),
        "day_sell_turnover_tl": round(day_sell_tl, 2),
        "open_stock_quantity": round(open_qty, 0),
        "market_value_tl": round(mtm_val, 2),
        "fifo_avg_cost": round(fifo_cost, 2),
        "unrealized_pnl_tl": round(unreal_pnl, 2),
        "unrealized_pnl_pct": round(unreal_pct, 2),
        "matched_volume_pct": round(matched_pct, 1),
        "global_saturation_pct": round(global_sat_pct, 1),
        "ribbon_status": ribbon_status,
        "diagnostic": diagnostic,
        "horizons": horizons_data,
        "fifo_realization_12m": fifo_res,
        "confluence_realization_12m": confluence_res,
    }


def get_tertip_timeseries_chart(
    db: PostgresManager,
    symbol: str,
    broker_id: str = "MLB",
    limit_days: int = 1500,
) -> list[dict[str, Any]]:
    """Return chronological time series with EWMA ribbons and cost basis for charting."""
    sym = symbol.upper()
    bid = broker_id.upper()

    if sym == "XU030":
        if bid in INSTITUTIONAL_BUNDLES:
            bundle_list = list(INSTITUTIONAL_BUNDLES[bid])
            query = """
                SELECT 
                    f.trade_date,
                    'XU030' AS symbol,
                    %s AS broker_id,
                    SUM(f.open_stock_quantity) AS open_stock_quantity,
                    0.0 AS fifo_avg_cost,
                    MAX(COALESCE(b.close_price, 10000.0)) AS market_close_price,
                    SUM(f.market_value_tl) AS market_value_tl,
                    SUM(f.unrealized_pnl_tl) AS unrealized_pnl_tl,
                    SUM(f.buy_turnover_tl) - SUM(f.sell_turnover_tl) AS net_flow_tl,
                    SUM(f.buy_turnover_tl) AS buy_turnover_tl,
                    SUM(f.sell_turnover_tl) AS sell_turnover_tl,
                    MAX(COALESCE(b.close_price, 10000.0)) AS buy_vwap
                FROM silver_broker_fifo_daily f
                JOIN silver_daily_stock_summary s ON f.trade_date = s.trade_date AND f.symbol = s.symbol
                LEFT JOIN silver_daily_benchmark_index b ON f.trade_date = b.trade_date
                WHERE s.index_name = 'BIST30' AND f.broker_id = ANY(%s)
                GROUP BY f.trade_date
                ORDER BY f.trade_date ASC;
            """
            df = db.query_pl(query, params=[bid, bundle_list])
        else:
            query = """
                SELECT 
                    f.trade_date,
                    'XU030' AS symbol,
                    %s AS broker_id,
                    SUM(f.open_stock_quantity) AS open_stock_quantity,
                    0.0 AS fifo_avg_cost,
                    MAX(COALESCE(b.close_price, 10000.0)) AS market_close_price,
                    SUM(f.market_value_tl) AS market_value_tl,
                    SUM(f.unrealized_pnl_tl) AS unrealized_pnl_tl,
                    SUM(f.buy_turnover_tl) - SUM(f.sell_turnover_tl) AS net_flow_tl,
                    SUM(f.buy_turnover_tl) AS buy_turnover_tl,
                    SUM(f.sell_turnover_tl) AS sell_turnover_tl,
                    MAX(COALESCE(b.close_price, 10000.0)) AS buy_vwap
                FROM silver_broker_fifo_daily f
                JOIN silver_daily_stock_summary s ON f.trade_date = s.trade_date AND f.symbol = s.symbol
                LEFT JOIN silver_daily_benchmark_index b ON f.trade_date = b.trade_date
                WHERE s.index_name = 'BIST30' AND f.broker_id = %s
                GROUP BY f.trade_date
                ORDER BY f.trade_date ASC;
            """
            df = db.query_pl(query, params=[bid, bid])
    elif bid in INSTITUTIONAL_BUNDLES:
        bundle_list = list(INSTITUTIONAL_BUNDLES[bid])
        query = f"""
            SELECT 
                trade_date,
                symbol,
                '{bid}' AS broker_id,
                SUM(open_stock_quantity) AS open_stock_quantity,
                SUM(open_fifo_cost_tl) / NULLIF(SUM(open_stock_quantity), 0) AS fifo_avg_cost,
                MAX(market_close_price) AS market_close_price,
                SUM(market_value_tl) AS market_value_tl,
                SUM(unrealized_pnl_tl) AS unrealized_pnl_tl,
                SUM(buy_turnover_tl) - SUM(sell_turnover_tl) AS net_flow_tl,
                SUM(buy_turnover_tl) AS buy_turnover_tl,
                SUM(sell_turnover_tl) AS sell_turnover_tl,
                SUM(buy_turnover_tl) / NULLIF(SUM(buy_volume), 0) AS buy_vwap
            FROM silver_broker_fifo_daily
            WHERE symbol = %s AND broker_id = ANY(%s)
            GROUP BY trade_date, symbol
            ORDER BY trade_date ASC;
        """
        df = db.query_pl(query, params=[sym, bundle_list])
    else:
        query = """
            SELECT 
                trade_date,
                symbol,
                broker_id,
                open_stock_quantity,
                fifo_avg_cost,
                market_close_price,
                market_value_tl,
                unrealized_pnl_tl,
                buy_turnover_tl - sell_turnover_tl AS net_flow_tl,
                buy_turnover_tl,
                sell_turnover_tl,
                buy_vwap
            FROM silver_broker_fifo_daily
            WHERE symbol = %s AND broker_id = %s
            ORDER BY trade_date ASC;
        """
        df = db.query_pl(query, params=[sym, bid])

    if df.is_empty():
        return []

    # Prepare daily execution entry price (buy_vwap or close) and size (turnover)
    df = df.with_columns([
        pl.coalesce(pl.col("buy_vwap"), pl.col("market_close_price")).alias("entry_price"),
        pl.coalesce(pl.col("buy_turnover_tl"), pl.lit(1e6)).alias("entry_size"),
    ])

    # Enrich with multi-horizon EWMAs
    df = add_multi_horizon_ewma_polars(df, value_col="open_stock_quantity", prefix="ewma_qty")
    df = add_multi_horizon_size_weighted_ewma_polars(
        df, value_col="entry_price", size_col="entry_size", prefix="ewma_cost"
    )

    # Take the last limit_days sessions
    if len(df) > limit_days:
        df = df.tail(limit_days)

    rows = df.to_dicts()
    result = []

    for r in rows:
        td = r["trade_date"]
        if hasattr(td, "timestamp"):
            epoch_sec = int(td.replace(tzinfo=timezone.utc).timestamp()) if getattr(td, "tzinfo", None) is None else int(td.timestamp())
        elif isinstance(td, date):
            epoch_sec = int(datetime.combine(td, datetime.min.time(), tzinfo=timezone.utc).timestamp())
        else:
            d_obj = date.fromisoformat(str(td)[:10])
            epoch_sec = int(datetime.combine(d_obj, datetime.min.time(), tzinfo=timezone.utc).timestamp())

        result.append({
            "time": epoch_sec,
            "trade_date": str(td),
            "close_price": round(float(r["market_close_price"] or 0.0), 2),
            "fifo_avg_cost": round(float(r["fifo_avg_cost"] or 0.0), 2),
            "ewma_cost_5d": round(float(r.get("ewma_cost_1w_5d") or r["fifo_avg_cost"] or 0.0), 2),
            "ewma_cost_10d": round(float(r.get("ewma_cost_2w_10d") or r["fifo_avg_cost"] or 0.0), 2),
            "ewma_cost_21d": round(float(r.get("ewma_cost_1m_21d") or r["fifo_avg_cost"] or 0.0), 2),
            "ewma_cost_63d": round(float(r.get("ewma_cost_3m_63d") or r["fifo_avg_cost"] or 0.0), 2),
            "ewma_cost_126d": round(float(r.get("ewma_cost_6m_126d") or r["fifo_avg_cost"] or 0.0), 2),
            "ewma_cost_252d": round(float(r.get("ewma_cost_12m_252d") or r["fifo_avg_cost"] or 0.0), 2),
            "open_quantity": round(float(r["open_stock_quantity"] or 0.0), 0),
            "ewma_qty_5d": round(float(r.get("ewma_qty_1w_5d") or r["open_stock_quantity"] or 0.0), 0),
            "ewma_qty_10d": round(float(r.get("ewma_qty_2w_10d") or r["open_stock_quantity"] or 0.0), 0),
            "ewma_qty_21d": round(float(r.get("ewma_qty_1m_21d") or r["open_stock_quantity"] or 0.0), 0),
            "ewma_qty_63d": round(float(r.get("ewma_qty_3m_63d") or r["open_stock_quantity"] or 0.0), 0),
            "ewma_qty_126d": round(float(r.get("ewma_qty_6m_126d") or r["open_stock_quantity"] or 0.0), 0),
            "ewma_qty_252d": round(float(r.get("ewma_qty_12m_252d") or r["open_stock_quantity"] or 0.0), 0),
            "net_flow_tl": round(float(r["net_flow_tl"] or 0.0), 2),
            "unrealized_pnl_tl": round(float(r["unrealized_pnl_tl"] or 0.0), 2),
        })

    return result
