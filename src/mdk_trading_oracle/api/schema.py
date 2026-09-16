"""Strawberry GraphQL Schema definition for MDK Trading Oracle."""

from typing import List, Optional
import strawberry

from mdk_trading_oracle.api.types import (
    Broker,
    BrokerSummary,
    CandleWithBroker,
    Instrument,
)
from mdk_trading_oracle.api.resolvers import (
    get_available_dates,
    get_brokers,
    get_broker_summary,
    get_candles,
    get_instruments,
)


@strawberry.type
class Query:
    @strawberry.field(description="List of all liquid tracked BIST equities.")
    def instruments(self) -> List[Instrument]:
        return get_instruments()

    @strawberry.field(description="List of all active brokerages.")
    def brokers(self) -> List[Broker]:
        return get_brokers()

    @strawberry.field(description="List of distinct available trading dates.")
    def available_dates(self) -> List[str]:
        return get_available_dates()

    @strawberry.field(description="Candles with broker flow and realized PnL for given symbol, timeframe, and date.")
    def candles(
        self,
        symbol: str,
        timeframe: str = "60m",
        date: Optional[str] = None,
        broker_id: str = "MLB",
    ) -> List[CandleWithBroker]:
        return get_candles(symbol=symbol, timeframe=timeframe, date=date, broker_id=broker_id)

    @strawberry.field(description="Macro daily summary of institutional broker execution across all stocks.")
    def broker_summary(
        self,
        date: Optional[str] = None,
        broker_id: str = "MLB",
    ) -> BrokerSummary:
        return get_broker_summary(date=date, broker_id=broker_id)


schema = strawberry.Schema(query=Query)
