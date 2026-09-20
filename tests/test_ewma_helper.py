"""Unit tests for the Core EWMA (Exponentially Weighted Moving Average) Helper."""

from datetime import date, timedelta

import numpy as np
import polars as pl
import pytest

from mdk_trading_oracle.core.ewma import (
    HORIZON_MAP,
    add_multi_horizon_ewma_polars,
    alpha_from_span,
    calculate_bounded_ewma,
    calculate_size_weighted_bounded_ewma,
    compute_ewma_series_numpy,
    compute_ewma_weights,
    span_from_half_life,
)


def test_alpha_from_span():
    assert alpha_from_span(1) == 1.0
    assert alpha_from_span(3) == 0.5
    assert alpha_from_span(5) == 2.0 / 6.0
    assert alpha_from_span(21) == 2.0 / 22.0

    with pytest.raises(ValueError):
        alpha_from_span(0)


def test_span_from_half_life():
    span = span_from_half_life(5.0)
    assert span > 0.0
    with pytest.raises(ValueError):
        span_from_half_life(-1.0)


def test_compute_ewma_weights_decay():
    """Verify weights decay strictly as distance/lag increases and sum to 1."""
    weights = compute_ewma_weights(window=10, span=5)
    assert len(weights) == 10
    assert np.isclose(np.sum(weights), 1.0)

    # Strictly decreasing
    for i in range(len(weights) - 1):
        assert weights[i] > weights[i + 1]


def test_calculate_bounded_ewma():
    # If all values are constant, EWMA must equal that constant
    const_series = [100.0] * 50
    ewma_val = calculate_bounded_ewma(const_series, window=21, span=21)
    assert np.isclose(ewma_val, 100.0)

    # Empty sequence returns 0
    assert calculate_bounded_ewma([], window=5) == 0.0

    # Step increase: EWMA must be between baseline and peak, closer to peak
    step_series = [10.0] * 20 + [50.0] * 5
    ewma_step = calculate_bounded_ewma(step_series, window=10, span=5)
    assert 10.0 < ewma_step < 50.0
    assert ewma_step > 30.0  # recent values dominate


def test_compute_ewma_series_numpy():
    series = [10.0, 20.0, 30.0, 40.0, 50.0]
    ewma_arr = compute_ewma_series_numpy(series, span=3.0)
    assert len(ewma_arr) == 5
    assert ewma_arr[0] == 10.0
    # Recursive check: alpha = 2/4 = 0.5
    assert ewma_arr[1] == 0.5 * 20.0 + 0.5 * 10.0  # 15.0
    assert ewma_arr[2] == 0.5 * 30.0 + 0.5 * 15.0  # 22.5


def test_add_multi_horizon_ewma_polars():
    n = 300
    df = pl.DataFrame({
        "trade_date": pl.date_range(date(2025, 1, 1), date(2025, 1, 1) + timedelta(days=n - 1), eager=True),
        "open_qty": [1000.0 + float(i) * 10.0 for i in range(n)],
    })

    enriched = add_multi_horizon_ewma_polars(df, value_col="open_qty", prefix="ewma_qty")

    # Verify all 6 horizons are generated
    for tag, span in HORIZON_MAP.items():
        col_name = f"ewma_qty_{tag.lower()}_{span}d"
        assert col_name in enriched.columns
        # Check non-null values
        assert enriched[col_name].null_count() == 0


def test_calculate_size_weighted_bounded_ewma():
    # When sizes are equal, size-weighted EWMA equals pure bounded EWMA
    costs = [100.0, 110.0, 120.0, 130.0, 140.0]
    equal_sizes = [1000.0] * 5
    v1 = calculate_bounded_ewma(costs, window=5, span=5)
    v2 = calculate_size_weighted_bounded_ewma(costs, sizes=equal_sizes, window=5, span=5)
    assert np.isclose(v1, v2)

    # When an earlier day has a huge position size, it should dominate despite time decay
    # Day 0 (5 days ago): huge trade of 1,000,000 shares at 100.0
    # Days 1-4 (recent): tiny trades of 10 shares at 200.0
    unbalanced_sizes = [1_000_000.0, 10.0, 10.0, 10.0, 10.0]
    unbalanced_costs = [100.0, 200.0, 200.0, 200.0, 200.0]
    v_huge = calculate_size_weighted_bounded_ewma(unbalanced_costs, sizes=unbalanced_sizes, window=5, span=5)
    # The huge trade at 100.0 must dominate, resulting in a value close to 100
    assert v_huge < 105.0

