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
    symbol: Optional[str] = None
    net_volume: Optional[float] = None
    open_stock_quantity: Optional[float] = None


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


@strawberry.type
class ForwardReturnPoint:
    day_offset: int
    date: Optional[str]
    prev_close_price: Optional[float]
    close_price: Optional[float]
    return_pct: Optional[float]  # Independent daily return: (P_{T+k} - P_{T+k-1}) / P_{T+k-1} * 100
    daily_return_pct: Optional[float]
    cumulative_return_pct: Optional[float]  # Cumulative return from T: (P_{T+k} - P_T) / P_T * 100


@strawberry.type
class ForwardHorizonStat:
    day_offset: int
    avg_return_pct: float  # Independent daily average return at T+k
    win_rate_pct: float  # % of days with positive daily return
    median_return_pct: float  # Median daily return
    max_gain_pct: float
    max_loss_pct: float
    cumul_avg_return_pct: Optional[float]  # Cumulative average return from T
    cumul_win_rate_pct: Optional[float]
    sample_count: int


@strawberry.type
class EventStudyOccurrence:
    event_date: str
    prev_close_price: Optional[float]
    open_price: Optional[float]
    close_price: float
    price_change_tl: Optional[float]
    price_change_pct: Optional[float]
    movement_value: float
    bofa_net_flow_tl: float
    total_turnover_tl: float
    forward_returns: list[ForwardReturnPoint]


@strawberry.type
class EventStudyResult:
    symbol: str
    condition_type: str
    min_value: Optional[float]
    max_value: Optional[float]
    direction: Optional[str]
    forward_days: int
    total_occurrences: int
    horizon_stats: list[ForwardHorizonStat]
    occurrences: list[EventStudyOccurrence]


@strawberry.type
class TimeWindowAuctionDetail:
    auction_type: str  # "OPENING_AUCTION" (09:55) or "CLOSING_AUCTION" (18:05)
    match_time: str
    match_price: float
    total_volume: float
    total_turnover_tl: float
    broker_buy_volume: float
    broker_sell_volume: float
    broker_net_volume: float
    broker_net_flow_tl: float
    broker_share_pct: float


@strawberry.type
class TimeWindowCandle:
    bucket_start: str
    bucket_end: str
    open: float
    high: float
    low: float
    close: float
    volume: float
    turnover_tl: float
    vwap: float
    trades_count: int
    broker_buy_volume: float
    broker_sell_volume: float
    broker_net_volume: float
    broker_net_flow_tl: float
    broker_realized_pnl_tl: float
    broker_share_pct: float


@strawberry.type
class TimeWindowAnalysisResult:
    symbol: str
    symbol_name: str
    broker_id: str
    broker_name: str
    start_datetime: str
    end_datetime: str
    timeframe: str
    # Window price stats
    window_open_price: Optional[float]
    window_close_price: Optional[float]
    price_change_tl: Optional[float]
    price_change_pct: Optional[float]
    window_high_price: Optional[float]
    window_low_price: Optional[float]
    price_range_pct: Optional[float]
    total_volume: float
    total_turnover_tl: float
    total_trades_count: int
    # Broker stats in this window
    broker_buy_volume: float
    broker_buy_turnover_tl: float
    broker_buy_vwap: Optional[float]
    broker_sell_volume: float
    broker_sell_turnover_tl: float
    broker_sell_vwap: Optional[float]
    broker_net_volume: float
    broker_net_flow_tl: float
    broker_realized_pnl_tl: float
    broker_market_share_pct: float
    # Auction match details
    opening_auction: Optional[TimeWindowAuctionDetail]
    closing_auction: Optional[TimeWindowAuctionDetail]
    # Candle series
    candles: list[TimeWindowCandle]


