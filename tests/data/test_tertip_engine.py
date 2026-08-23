"""Unit tests for the Tertip Mechanism (INTRADAY_MATCHED_FIFO_V1) and Immutable Lot Ledger."""

import datetime as dt
from decimal import Decimal

from mdk_trading_oracle.data.silver.tertip_engine import (
    LotChanges,
    PositionState,
    apply_buy,
    apply_sell,
    execute_daily_flow_step,
)


def test_intraday_match_then_residual_sell_consumes_existing_long_fifo():
    """Verify Ek C Test 1: Intraday match + residual sell consumes existing LONG FIFO lot."""
    state = PositionState(broker_id="MLB", symbol="AKBNK")
    changes = LotChanges()

    # Day 1: Initial position Long 100 @ 50 TL (5000 TL)
    apply_buy(state, Decimal("100"), Decimal("5000"), dt.date(2026, 1, 1), changes)

    # Day 2: Buy 40 @ 10 TL (400 TL), Sell 100 @ 20 TL (2000 TL)
    result = execute_daily_flow_step(
        state=state,
        day=dt.date(2026, 1, 2),
        buy_volume=40.0,
        buy_turnover=400.0,
        sell_volume=100.0,
        sell_turnover=2000.0,
        market_close_price=20.0,
        changes=changes,
    )

    assert result.matched_quantity == Decimal("40.000000")
    assert result.matched_buy_value == Decimal("400.000000")
    assert result.matched_sell_value == Decimal("800.000000")
    assert result.intraday_realized_pnl == Decimal("400.000000")
    assert result.residual_quantity == Decimal("-60.000000")
    assert result.residual_value == Decimal("1200.000000")
    assert result.residual_unit_cost == Decimal("20")
    assert result.carry_fifo_realized_pnl == Decimal("-1800.000000")
    assert result.daily_realized_pnl == Decimal("-1400.000000")
    assert result.position_side == "LONG"
    assert result.open_stock_quantity == Decimal("40.000000")
    assert result.open_fifo_cost_tl == Decimal("2000.000000")
    assert result.fifo_avg_cost == Decimal("50")


def test_lot_entry_cost_stays_immutable_and_closures_are_audited():
    """Verify Ek C Test 2: Lot entry record stays immutable while realizations are recorded."""
    state = PositionState(broker_id="MLB", symbol="AKBNK")
    changes = LotChanges()

    # Day 1: Open LONG 100 @ 10 TL (1000 TL)
    apply_buy(state, Decimal("100"), Decimal("1000"), dt.date(2026, 1, 2), changes)
    entry = next(iter(changes.entries.values()))

    # Day 2: Sell 40 for 480 TL (12 TL/unit)
    first_realized = apply_sell(state, Decimal("40"), Decimal("480"), dt.date(2026, 1, 3), changes)
    # Day 3: Sell 60 for 660 TL (11 TL/unit)
    second_realized = apply_sell(state, Decimal("60"), Decimal("660"), dt.date(2026, 1, 4), changes)

    # Initial entry record remains unchanged
    assert entry.opened_quantity == Decimal("100.000000")
    assert entry.opened_value == Decimal("1000.000000")
    assert entry.opened_unit_cost == Decimal("10")

    # Realized PnLs:
    # 40 * 12 - 40 * 10 = +80 TL
    assert first_realized == Decimal("80.000000")
    # 60 * 11 - 60 * 10 = +60 TL
    assert second_realized == Decimal("60.000000")

    assert len(changes.realizations) == 2
    assert not changes.realizations[0].is_final
    assert changes.realizations[0].remaining_quantity_after == Decimal("60.000000")
    assert changes.realizations[1].is_final
    assert changes.realizations[1].remaining_quantity_after == Decimal("0.000000")
    assert sum(r.realized_pnl for r in changes.realizations) == Decimal("140.000000")


def test_three_day_full_numerical_scenario():
    """Verify the 3-day scenario from Section 5 in the specification: LONG -> SHORT -> FLAT."""
    state = PositionState(broker_id="MLB", symbol="GARAN")
    changes = LotChanges()

    # Day 0 / Start: LONG 100 @ 50 TL (5000 TL)
    apply_buy(state, Decimal("100"), Decimal("5000"), dt.date(2026, 1, 1), changes)

    # Day 1: Buy 40 @ 10 TL, Sell 100 @ 20 TL
    res1 = execute_daily_flow_step(
        state, dt.date(2026, 1, 2), 40.0, 400.0, 100.0, 2000.0, 20.0, changes
    )
    assert res1.intraday_realized_pnl == Decimal("400.000000")
    assert res1.carry_fifo_realized_pnl == Decimal("-1800.000000")
    assert res1.daily_realized_pnl == Decimal("-1400.000000")
    assert res1.position_side == "LONG"
    assert res1.open_stock_quantity == Decimal("40.000000")
    assert res1.open_fifo_cost_tl == Decimal("2000.000000")

    # Day 2: Net 50 sell @ 60 TL (3000 TL turnover, 0 buy)
    res2 = execute_daily_flow_step(
        state, dt.date(2026, 1, 3), 0.0, 0.0, 50.0, 3000.0, 60.0, changes
    )
    assert res2.intraday_realized_pnl == Decimal("0.000000")
    # 40 long @ 50 closed at 60 TL -> 40 * (60 - 50) = +400 TL
    assert res2.carry_fifo_realized_pnl == Decimal("400.000000")
    assert res2.daily_realized_pnl == Decimal("400.000000")
    # Remaining 10 sell opens SHORT 10 @ 60 TL (600 TL)
    assert res2.position_side == "SHORT"
    assert res2.open_stock_quantity == Decimal("10.000000")
    assert res2.open_fifo_cost_tl == Decimal("600.000000")
    assert res2.fifo_avg_cost == Decimal("60")

    # Day 3: Net 10 buy @ 55 TL (550 TL turnover, 0 sell)
    res3 = execute_daily_flow_step(
        state, dt.date(2026, 1, 4), 10.0, 550.0, 0.0, 0.0, 55.0, changes
    )
    assert res3.intraday_realized_pnl == Decimal("0.000000")
    # Short 10 @ 60 covered at 55 TL -> 600 - 550 = +50 TL
    assert res3.carry_fifo_realized_pnl == Decimal("50.000000")
    assert res3.daily_realized_pnl == Decimal("50.000000")
    assert res3.position_side == "FLAT"
    assert res3.open_stock_quantity == Decimal("0.000000")
    assert res3.open_fifo_cost_tl == Decimal("0.000000")
    assert res3.cumulative_realized_pnl_tl == Decimal("-950.000000")  # 400 - 1800 + 400 + 50 = -950
