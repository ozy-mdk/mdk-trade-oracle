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
        d_row = db.execute(
            "SELECT MAX(trade_date) FROM silver_broker_fifo_daily WHERE symbol = %s AND broker_id = %s;",
            [sym, bid],
        ).fetchone()
        if not d_row or not d_row[0]:
            trade_date = "2026-09-16"
        else:
            trade_date = str(d_row[0])

    # Fetch chronological history up to trade_date
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
