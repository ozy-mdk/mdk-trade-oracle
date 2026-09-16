"""Strawberry GraphQL types for market candles and institutional order flows."""

from typing import Optional

import strawberry


@strawberry.type
class Instrument:
    symbol: str
    name: str
    sector: str


@strawberry.type
class Broker:
    broker_id: str
    broker_name: str
    category: str


@strawberry.type
class CandleWithBroker:
    timeframe: str
    bucket_start: str
    bucket_end: str
    symbol: str
    open: float
    high: float
    low: float
    close: float
    volume: float
    turnover_tl: float
    vwap: float
    trade_count: int
    broker_id: str
    buy_volume: float
    buy_turnover_tl: float
    buy_vwap: Optional[float]
    sell_volume: float
    sell_turnover_tl: float
    sell_vwap: Optional[float]
    net_volume: float
    net_flow_tl: float
    matched_volume: float
    realized_pnl_tl: float
    broker_share_pct: float


@strawberry.type
class BrokerSummary:
    broker_id: str
    trade_date: str
    total_buy_volume: float
    total_buy_turnover_tl: float
    total_sell_volume: float
    total_sell_turnover_tl: float
    net_flow_tl: float
    matched_volume: float
    intraday_pnl_tl: float
    carry_fifo_pnl_tl: float
    realized_pnl_tl: float
    position_bias: str


@strawberry.type
class DailyFifoRecord:
    trade_date: str
    symbol: str
    symbol_name: str
    sector: str
    broker_id: str
    buy_volume: float
    buy_turnover_tl: float
    buy_vwap: Optional[float]
    sell_volume: float
    sell_turnover_tl: float
    sell_vwap: Optional[float]
    matched_volume: float
    intraday_realized_pnl_tl: float
    carry_fifo_realized_pnl_tl: float
    daily_realized_pnl_tl: float
    position_side: str
    open_stock_quantity: float
    open_fifo_cost_tl: float
    fifo_avg_cost: Optional[float]
    market_close_price: Optional[float]
    market_value_tl: float
    unrealized_pnl_tl: float
    total_daily_pnl_tl: float
    cumulative_realized_pnl_tl: float


@strawberry.type
class TertipLot:
    lot_id: str
    broker_id: str
    symbol: str
    direction: str
    open_date: str
    opened_quantity: float
    opened_value_tl: float
    opened_unit_cost: float
    status: str
    closed_date: Optional[str]
    total_quantity_closed: float
    total_closing_value_tl: float
    total_realized_pnl_tl: float
    remaining_quantity: float
    remaining_value_tl: float


@strawberry.type
class TertipExecutiveSummary:
    broker_id: str
    trade_date: str
    cumulative_realized_pnl_tl: float
    daily_realized_pnl_tl: float
    intraday_realized_pnl_tl: float
    carry_fifo_realized_pnl_tl: float
    open_inventory_cost_tl: float
    open_inventory_value_tl: float
    unrealized_pnl_tl: float
    total_daily_pnl_tl: float
    total_open_lots_count: int
