"""Strawberry GraphQL Schema definition for MDK Trading Oracle."""

from typing import List, Optional

import strawberry

from mdk_trading_oracle.api.resolvers import (
    get_available_dates,
    get_broker_summary,
    get_brokers,
    get_candles,
    get_daily_fifo,
    get_event_study,
    get_instruments,
    get_tertip_lots,
    get_tertip_summary,
)
from mdk_trading_oracle.api.types import (
    Broker,
    BrokerSummary,
    CandleWithBroker,
    DailyFifoRecord,
    EventStudyResult,
    Instrument,
    TertipExecutiveSummary,
    TertipLot,
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

    @strawberry.field(description="Historical daily FIFO ledger rows from 2022 to present.")
    def daily_fifo(
        self,
        broker_id: str = "MLB",
        symbol: Optional[str] = None,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
        limit: int = 100,
    ) -> List[DailyFifoRecord]:
        return get_daily_fifo(
            broker_id=broker_id,
            symbol=symbol,
            start_date=start_date,
            end_date=end_date,
            limit=limit,
        )

    @strawberry.field(description="Historical individual tertip lot lifecycle records from 2022 to present.")
    def tertip_lots(
        self,
        broker_id: str = "MLB",
        symbol: Optional[str] = None,
        status: Optional[str] = None,
        limit: int = 100,
    ) -> List[TertipLot]:
        return get_tertip_lots(
            broker_id=broker_id,
            symbol=symbol,
            status=status,
            limit=limit,
        )

    @strawberry.field(description="Executive tertip inventory and PnL summary for selected session.")
    def tertip_summary(
        self,
        broker_id: str = "MLB",
        date: Optional[str] = None,
    ) -> TertipExecutiveSummary:
        return get_tertip_summary(broker_id=broker_id, date=date)

    @strawberry.field(description="Event study: Analyze historical stock movements and forward reaction returns.")
    def event_study(
        self,
        symbol: str,
        condition_type: str = "DAILY_RETURN",
        min_value: Optional[float] = None,
        max_value: Optional[float] = None,
        direction: Optional[str] = None,
        forward_days: int = 5,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
        limit: int = 100,
    ) -> EventStudyResult:
        return get_event_study(
            symbol=symbol,
            condition_type=condition_type,
            min_value=min_value,
            max_value=max_value,
            direction=direction,
            forward_days=forward_days,
            start_date=start_date,
            end_date=end_date,
            limit=limit,
        )


schema = strawberry.Schema(query=Query)
