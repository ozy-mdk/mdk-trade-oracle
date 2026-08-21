"""Pydantic domain schemas and data transfer objects."""

from datetime import date, datetime
from enum import Enum
from typing import List, Optional

from pydantic import BaseModel, Field

from mdk_trading_oracle.core.time import now_turkey


class TradeSide(str, Enum):
    BUY = "BUY"
    SELL = "SELL"


class SignalType(str, Enum):
    STRONG_BUY = "STRONG_BUY"
    BUY = "BUY"
    NEUTRAL = "NEUTRAL"
    SELL = "SELL"
    STRONG_SELL = "STRONG_SELL"


class RawTradeRecord(BaseModel):
    """Bronze layer: Raw trade line model."""
    trade_id: str
    timestamp: datetime
    symbol: str
    price: float
    volume: float
    buyer_broker_id: str
    seller_broker_id: str
    raw_source: str


class SilverBrokerTransaction(BaseModel):
    """Silver layer: Normalized single-sided broker transaction."""
    tx_id: str
    timestamp: datetime
    date_val: date
    symbol: str
    broker_id: str
    side: TradeSide
    price: float
    volume: float
    amount_tl: float
    counterparty_broker_id: Optional[str] = None


class SilverDailyBrokerSummary(BaseModel):
    """Silver layer: Daily aggregate per symbol & broker."""
    date_val: date
    symbol: str
    broker_id: str
    total_buy_volume: float
    total_sell_volume: float
    net_volume: float
    total_buy_tl: float
    total_sell_tl: float
    net_tl: float
    vwap_buy: Optional[float] = None
    vwap_sell: Optional[float] = None


class GoldBofAFlowMetrics(BaseModel):
    """Gold layer: Engineered BofA order flow features."""
    date_val: date
    symbol: str
    close_price: float
    total_symbol_volume: float
    total_symbol_tl: float

    # BofA specific metrics
    bofa_buy_tl: float
    bofa_sell_tl: float
    bofa_net_tl: float
    bofa_volume_share: float  # (buy + sell) / total
    bofa_net_share: float     # net / total

    # Rolling window metrics
    bofa_net_tl_roll_3d: float
    bofa_net_tl_roll_5d: float
    bofa_net_tl_roll_10d: float
    bofa_cum_net_tl_20d: float
    bofa_flow_acceleration_5d: float
    bofa_flow_zscore_20d: float


class OracleDecisionSignal(BaseModel):
    """Oracle decision layer: Final actionable decision output."""
    signal_id: str
    date_val: date
    symbol: str
    signal: SignalType
    confidence: float = Field(ge=0.0, le=1.0)
    bofa_net_tl: float
    bofa_net_share: float
    bofa_flow_zscore: float
    summary: str
    reasons: List[str]
    created_at: datetime = Field(default_factory=now_turkey)
