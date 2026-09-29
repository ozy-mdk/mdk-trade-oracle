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


def test_compute_12m_horizon_realizations():
    import polars as pl

    from mdk_trading_oracle.data.silver.tertip_analytics import compute_12m_horizon_realizations

    # Create synthetic DataFrame with 10 sessions
    dates = [f"2026-01-{i+1:02d}" for i in range(10)]
    closes = [100.0, 102.0, 95.0, 93.0, 90.0, 94.0, 96.0, 98.0, 105.0, 108.0]
    flows = [10e6, -5e6, -15e6, 20e6, 25e6, -10e6, 30e6, 15e6, -20e6, 5e6]
    fifo_costs = [100.0] * 10

    df = pl.DataFrame({
        "trade_date": dates,
        "market_close_price": closes,
        "net_flow_tl": flows,
        "fifo_avg_cost": fifo_costs,
        "buy_vwap": closes,
        "buy_turnover_tl": [50e6] * 10,
        "entry_price": closes,
        "entry_size": [50e6] * 10,
    })

    horizons_data = [
        {"code": "1D", "label": "Today", "ewma_unit_cost": 100.0},
        {"code": "1W", "label": "1 Week", "ewma_unit_cost": 100.0},
    ]

    horizons_res, fifo_res, confluence_res = compute_12m_horizon_realizations(
        df=df,
        horizons_data=horizons_data,
        close_p=108.0,
        fifo_cost=100.0,
    )

    assert "1D" in horizons_res
    assert "1W" in horizons_res
    assert "total_occurrences" in fifo_res
    assert "realized_pct" in confluence_res
    assert isinstance(confluence_res["realized_count"], int)
    assert isinstance(confluence_res["opposite_count"], int)

