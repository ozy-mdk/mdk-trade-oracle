"""Unit tests for Tertip Analytics & Action Diagnostic Engine."""

from mdk_trading_oracle.data.silver.tertip_analytics import classify_action_diagnostic


def test_classify_profit_harvest():
    diag = classify_action_diagnostic(
        day_net_flow=-50_000_000,
        open_qty=10_000_000,
        close_price=330.0,
        fifo_avg_cost=290.0,
        ewma_5d_qty=9_000_000,
        ewma_21d_qty=10_500_000,
        ewma_63d_qty=11_000_000,
        ewma_126d_qty=11_000_000,
        ewma_cost_63d=280.0,
        unrealized_pnl_pct=13.8,
        saturation_pct=70.0,
        matched_volume_pct=20.0,
    )
    assert diag["diagnostic_badge"] == "PROFIT_HARVEST"
    assert "profit" in diag["headline"].lower()


def test_classify_defense_support():
    diag = classify_action_diagnostic(
        day_net_flow=60_000_000,
        open_qty=15_000_000,
        close_price=301.0,
        fifo_avg_cost=300.0,
        ewma_5d_qty=15_000_000,
        ewma_21d_qty=14_500_000,
        ewma_63d_qty=14_000_000,
        ewma_126d_qty=13_000_000,
        ewma_cost_63d=295.0,
        unrealized_pnl_pct=0.33,
        saturation_pct=60.0,
        matched_volume_pct=15.0,
    )
    assert diag["diagnostic_badge"] == "DEFENSE_SUPPORT"
    assert "defending" in diag["headline"].lower()


def test_classify_momentum_expansion():
    diag = classify_action_diagnostic(
        day_net_flow=80_000_000,
        open_qty=12_000_000,
        close_price=350.0,
        fifo_avg_cost=310.0,
        ewma_5d_qty=11_500_000,
        ewma_21d_qty=10_000_000,
        ewma_63d_qty=9_000_000,
        ewma_126d_qty=8_000_000,
        ewma_cost_63d=300.0,
        unrealized_pnl_pct=12.9,
        saturation_pct=72.0,
        matched_volume_pct=25.0,
    )
    assert diag["diagnostic_badge"] == "MOMENTUM_EXPANSION"


def test_classify_de_grossing():
    diag = classify_action_diagnostic(
        day_net_flow=-30_000_000,
        open_qty=8_000_000,
        close_price=270.0,
        fifo_avg_cost=290.0,
        ewma_5d_qty=8_500_000,
        ewma_21d_qty=9_500_000,
        ewma_63d_qty=10_500_000,
        ewma_126d_qty=11_000_000,
        ewma_cost_63d=290.0,
        unrealized_pnl_pct=-6.9,
        saturation_pct=50.0,
        matched_volume_pct=20.0,
    )
    assert diag["diagnostic_badge"] == "DE_GROSSING_UNWIND"
