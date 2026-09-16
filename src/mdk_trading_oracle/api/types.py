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
    realized_pnl_tl: float
    position_bias: str
