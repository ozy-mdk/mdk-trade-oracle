"""Tertip Mechanism (INTRADAY_MATCHED_FIFO_V1) & Immutable Lot Ledger Engine.

Implements institutional FIFO inventory tracking, intraday matching vs. carry FIFO PnL,
immutable lot entries, partial/full closures, and mark-to-market valuations.
"""

import datetime as dt
from collections import deque
from dataclasses import dataclass
from decimal import ROUND_HALF_UP, Decimal
from typing import Any, Dict, List, Optional, Tuple

# Fixed financial precision helpers
QUANT_QTY = Decimal("0.000001")
QUANT_MONEY = Decimal("0.000001")
ZERO = Decimal("0")


def qquantity(v: Any) -> Decimal:
    """Quantize quantity to 6 decimal places."""
    if isinstance(v, Decimal):
        return v.quantize(QUANT_QTY, rounding=ROUND_HALF_UP)
    return Decimal(str(v)).quantize(QUANT_QTY, rounding=ROUND_HALF_UP)


def qmoney(v: Any) -> Decimal:
    """Quantize currency value to 6 decimal places."""
    if isinstance(v, Decimal):
        return v.quantize(QUANT_MONEY, rounding=ROUND_HALF_UP)
    return Decimal(str(v)).quantize(QUANT_MONEY, rounding=ROUND_HALF_UP)


def allocate_value(value: Decimal, available: Decimal, taken: Decimal) -> Decimal:
    """Allocate proportional monetary value when consuming partial quantity."""
    if available == ZERO:
        return ZERO
    if taken == available:
        return value
    return qmoney(value * taken / available)


@dataclass
class Lot:
    """Active FIFO lot tracked in memory."""
    lot_id: str
    broker_id: str
    symbol: str
    direction: str  # 'LONG' or 'SHORT'
    open_date: dt.date
    quantity: Decimal
    value: Decimal
    unit_cost: Decimal

    @property
    def is_empty(self) -> bool:
        return self.quantity == ZERO


@dataclass
class LotEntryRecord:
    """Immutable entry record written once when a lot is created."""
    lot_id: str
    broker_id: str
    symbol: str
    direction: str
    open_date: dt.date
    opened_quantity: Decimal
    opened_value: Decimal
    opened_unit_cost: Decimal


@dataclass
class LotRealizationRecord:
    """Audited realization event created for each partial or full lot closure."""
    realization_id: str
    lot_id: str
    broker_id: str
    symbol: str
    close_date: dt.date
    direction: str
    quantity_closed: Decimal
    entry_value_closed: Decimal
    closing_value: Decimal
    realized_pnl: Decimal
    remaining_quantity_after: Decimal
    is_final: bool


@dataclass
class DailyFlowResult:
    """Result of intraday matching and residual FIFO execution for a single day."""
    trade_date: dt.date
    broker_id: str
    symbol: str
    buy_quantity: Decimal
    buy_turnover: Decimal
    buy_vwap: Optional[Decimal]
    sell_quantity: Decimal
    sell_turnover: Decimal
    sell_vwap: Optional[Decimal]
    matched_quantity: Decimal
    matched_buy_value: Decimal
    matched_sell_value: Decimal
    intraday_realized_pnl: Decimal
    residual_quantity: Decimal
    residual_value: Decimal
    residual_unit_cost: Optional[Decimal]
    carry_fifo_realized_pnl: Decimal
    daily_realized_pnl: Decimal
    position_side: str
    open_stock_quantity: Decimal
    open_fifo_cost_tl: Decimal
    fifo_avg_cost: Optional[Decimal]
    market_close_price: Optional[Decimal]
    market_value_tl: Decimal
    unrealized_pnl_tl: Decimal
    total_daily_pnl_tl: Decimal
    cumulative_realized_pnl_tl: Decimal


class LotChanges:
    """Collects state mutations during FIFO execution."""

    def __init__(self):
        self.entries: Dict[str, LotEntryRecord] = {}
        self.realizations: List[LotRealizationRecord] = []
        self._realization_seq: int = 0

    def next_realization_id(self, lot_id: str) -> str:
        self._realization_seq += 1
        return f"{lot_id}_R{self._realization_seq:04d}"

    def record_entry(self, entry: LotEntryRecord) -> None:
        if entry.lot_id not in self.entries:
            self.entries[entry.lot_id] = entry

    def record_realization(
        self,
        lot: Lot,
        close_date: dt.date,
        quantity_closed: Decimal,
        entry_value_closed: Decimal,
        closing_value: Decimal,
        realized_pnl: Decimal,
    ) -> LotRealizationRecord:
        rec = LotRealizationRecord(
            realization_id=self.next_realization_id(lot.lot_id),
            lot_id=lot.lot_id,
            broker_id=lot.broker_id,
            symbol=lot.symbol,
            close_date=close_date,
            direction=lot.direction,
            quantity_closed=quantity_closed,
            entry_value_closed=entry_value_closed,
            closing_value=closing_value,
            realized_pnl=realized_pnl,
            remaining_quantity_after=lot.quantity,
            is_final=(lot.quantity == ZERO),
        )
        self.realizations.append(rec)
        return rec


class PositionState:
    """Manages the FIFO queues and cumulative accounting for a single broker-symbol pair."""

    def __init__(self, broker_id: str, symbol: str):
        self.broker_id = broker_id
        self.symbol = symbol
        self.long_lots: deque[Lot] = deque()
        self.short_lots: deque[Lot] = deque()
        self.cumulative_realized: Decimal = ZERO
        self.previous_unrealized_pnl: Decimal = ZERO
        self._lot_seq: int = 0

    def next_lot_id(self, direction: str, day: dt.date) -> str:
        self._lot_seq += 1
        date_str = day.strftime("%Y%m%d")
        return f"LOT_{self.broker_id}_{self.symbol}_{direction}_{date_str}_{self._lot_seq:04d}"

    @property
    def position_side(self) -> str:
        if self.long_lots:
            return "LONG"
        if self.short_lots:
            return "SHORT"
        return "FLAT"

    @property
    def total_quantity(self) -> Decimal:
        if self.long_lots:
            return sum((lot.quantity for lot in self.long_lots), ZERO)
        if self.short_lots:
            return sum((lot.quantity for lot in self.short_lots), ZERO)
        return ZERO

    @property
    def total_value(self) -> Decimal:
        if self.long_lots:
            return sum((lot.value for lot in self.long_lots), ZERO)
        if self.short_lots:
            return sum((lot.value for lot in self.short_lots), ZERO)
        return ZERO

    @property
    def average_unit_cost(self) -> Optional[Decimal]:
        qty = self.total_quantity
        if qty == ZERO:
            return None
        return self.total_value / qty


def _consume_lot(
    queue: deque[Lot],
    quantity: Decimal,
) -> Tuple[Lot, Decimal, Decimal]:
    """Consume up to `quantity` from the front lot in FIFO queue."""
    lot = queue[0]
    taken = min(quantity, lot.quantity)
    allocated_value = allocate_value(lot.value, lot.quantity, taken)
    lot.quantity = qquantity(lot.quantity - taken)
    lot.value = qmoney(lot.value - allocated_value)
    if lot.quantity == ZERO:
        queue.popleft()
    return lot, taken, allocated_value


def _new_lot(
    state: PositionState,
    direction: str,
    day: dt.date,
    quantity: Decimal,
    value: Decimal,
    changes: LotChanges,
) -> Optional[Lot]:
    """Create and append a new immutable FIFO lot."""
    if quantity <= ZERO:
        return None
    unit_cost = value / quantity
    lot_id = state.next_lot_id(direction, day)
    lot = Lot(
        lot_id=lot_id,
        broker_id=state.broker_id,
        symbol=state.symbol,
        direction=direction,
        open_date=day,
        quantity=quantity,
        value=value,
        unit_cost=unit_cost,
    )
    if direction == "LONG":
        state.long_lots.append(lot)
    else:
        state.short_lots.append(lot)

    changes.record_entry(
        LotEntryRecord(
            lot_id=lot_id,
            broker_id=state.broker_id,
            symbol=state.symbol,
            direction=direction,
            open_date=day,
            opened_quantity=quantity,
            opened_value=value,
            opened_unit_cost=unit_cost,
        )
    )
    return lot


def apply_sell(
    state: PositionState,
    quantity: Decimal,
    turnover: Decimal,
    day: dt.date,
    changes: LotChanges,
) -> Decimal:
    """Apply residual sell flow to FIFO state: closes LONG lots first, opens SHORT if excess."""
    remaining_quantity = qquantity(quantity)
    remaining_value = qmoney(turnover)
    realized = ZERO

    while remaining_quantity > ZERO and state.long_lots:
        lot, taken, entry_value = _consume_lot(state.long_lots, remaining_quantity)
        exit_value = allocate_value(remaining_value, remaining_quantity, taken)
        lot_realized = qmoney(exit_value - entry_value)
        realized = qmoney(realized + lot_realized)
        changes.record_realization(
            lot=lot,
            close_date=day,
            quantity_closed=taken,
            entry_value_closed=entry_value,
            closing_value=exit_value,
            realized_pnl=lot_realized,
        )
        remaining_quantity = qquantity(remaining_quantity - taken)
        remaining_value = qmoney(remaining_value - exit_value)

    if remaining_quantity > ZERO:
        _new_lot(state, "SHORT", day, remaining_quantity, remaining_value, changes)

    state.cumulative_realized = qmoney(state.cumulative_realized + realized)
    return realized


def apply_buy(
    state: PositionState,
    quantity: Decimal,
    turnover: Decimal,
    day: dt.date,
    changes: LotChanges,
) -> Decimal:
    """Apply residual buy flow to FIFO state: closes SHORT lots first, opens LONG if excess."""
    remaining_quantity = qquantity(quantity)
    remaining_value = qmoney(turnover)
    realized = ZERO

    while remaining_quantity > ZERO and state.short_lots:
        lot, taken, entry_proceeds = _consume_lot(state.short_lots, remaining_quantity)
        cover_value = allocate_value(remaining_value, remaining_quantity, taken)
        lot_realized = qmoney(entry_proceeds - cover_value)
        realized = qmoney(realized + lot_realized)
        changes.record_realization(
            lot=lot,
            close_date=day,
            quantity_closed=taken,
            entry_value_closed=entry_proceeds,
            closing_value=cover_value,
            realized_pnl=lot_realized,
        )
        remaining_quantity = qquantity(remaining_quantity - taken)
        remaining_value = qmoney(remaining_value - cover_value)

    if remaining_quantity > ZERO:
        _new_lot(state, "LONG", day, remaining_quantity, remaining_value, changes)

    state.cumulative_realized = qmoney(state.cumulative_realized + realized)
    return realized


def match_daily_flow(
    buy_volume: float,
    buy_turnover: float,
    sell_volume: float,
    sell_turnover: float,
) -> Tuple[Decimal, Decimal, Decimal, Decimal, Decimal, Decimal, Optional[Decimal]]:
    """Match intraday common volume and compute residual flow."""
    buy_quantity = qquantity(buy_volume)
    sell_quantity = qquantity(sell_volume)
    buy_val = qmoney(buy_turnover)
    sell_val = qmoney(sell_turnover)

    matched_quantity = min(buy_quantity, sell_quantity)
    matched_buy_value = (
        ZERO if matched_quantity == ZERO else allocate_value(buy_val, buy_quantity, matched_quantity)
    )
    matched_sell_value = (
        ZERO if matched_quantity == ZERO else allocate_value(sell_val, sell_quantity, matched_quantity)
    )
    intraday_realized = qmoney(matched_sell_value - matched_buy_value)

    if buy_quantity > sell_quantity:
        residual_quantity = qquantity(buy_quantity - sell_quantity)
        residual_value = qmoney(buy_val - matched_buy_value)
    elif sell_quantity > buy_quantity:
        residual_quantity = qquantity(-(sell_quantity - buy_quantity))
        residual_value = qmoney(sell_val - matched_sell_value)
    else:
        residual_quantity = ZERO
        residual_value = ZERO

    residual_unit_cost = (
        None if residual_quantity == ZERO else (residual_value / abs(residual_quantity))
    )

    return (
        matched_quantity,
        matched_buy_value,
        matched_sell_value,
        intraday_realized,
        residual_quantity,
        residual_value,
        residual_unit_cost,
    )


def execute_daily_flow_step(
    state: PositionState,
    day: dt.date,
    buy_volume: float,
    buy_turnover: float,
    sell_volume: float,
    sell_turnover: float,
    market_close_price: Optional[float],
    changes: LotChanges,
) -> DailyFlowResult:
    """Execute complete daily intraday matching, FIFO queue update, and mark-to-market valuation."""
    buy_qty = qquantity(buy_volume)
    sell_qty = qquantity(sell_volume)
    buy_to = qmoney(buy_turnover)
    sell_to = qmoney(sell_turnover)

    buy_vwap = (buy_to / buy_qty) if buy_qty > ZERO else None
    sell_vwap = (sell_to / sell_qty) if sell_qty > ZERO else None

    (
        matched_quantity,
        matched_buy_value,
        matched_sell_value,
        intraday_realized,
        residual_quantity,
        residual_value,
        residual_unit_cost,
    ) = match_daily_flow(buy_volume, buy_turnover, sell_volume, sell_turnover)

    fifo_realized = ZERO
    if residual_quantity > ZERO:
        fifo_realized = apply_buy(state, residual_quantity, residual_value, day, changes)
    elif residual_quantity < ZERO:
        fifo_realized = apply_sell(state, -residual_quantity, residual_value, day, changes)

    state.cumulative_realized = qmoney(state.cumulative_realized + intraday_realized)
    daily_realized_pnl = qmoney(intraday_realized + fifo_realized)

    # Position side & Inventory
    pos_side = state.position_side
    open_qty = state.total_quantity
    open_val = state.total_value
    avg_cost = state.average_unit_cost

    # Mark to market
    close_px = Decimal(str(market_close_price)) if market_close_price is not None else None
    market_val = ZERO
    unrealized_pnl = ZERO

    if close_px is not None and open_qty > ZERO:
        market_val = qmoney(open_qty * close_px)
        if pos_side == "LONG":
            unrealized_pnl = qmoney(market_val - open_val)
        elif pos_side == "SHORT":
            unrealized_pnl = qmoney(open_val - market_val)

    unrealized_delta = qmoney(unrealized_pnl - state.previous_unrealized_pnl)
    total_daily_pnl = qmoney(daily_realized_pnl + unrealized_delta)
    state.previous_unrealized_pnl = unrealized_pnl

    return DailyFlowResult(
        trade_date=day,
        broker_id=state.broker_id,
        symbol=state.symbol,
        buy_quantity=buy_qty,
        buy_turnover=buy_to,
        buy_vwap=buy_vwap,
        sell_quantity=sell_qty,
        sell_turnover=sell_to,
        sell_vwap=sell_vwap,
        matched_quantity=matched_quantity,
        matched_buy_value=matched_buy_value,
        matched_sell_value=matched_sell_value,
        intraday_realized_pnl=intraday_realized,
        residual_quantity=residual_quantity,
        residual_value=residual_value,
        residual_unit_cost=residual_unit_cost,
        carry_fifo_realized_pnl=fifo_realized,
        daily_realized_pnl=daily_realized_pnl,
        position_side=pos_side,
        open_stock_quantity=open_qty,
        open_fifo_cost_tl=open_val,
        fifo_avg_cost=avg_cost,
        market_close_price=close_px,
        market_value_tl=market_val,
        unrealized_pnl_tl=unrealized_pnl,
        total_daily_pnl_tl=total_daily_pnl,
        cumulative_realized_pnl_tl=state.cumulative_realized,
    )
