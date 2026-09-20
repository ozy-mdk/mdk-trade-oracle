"""Core EWMA (Exponentially Weighted Moving Average) Helper Module.

Provides institutional-grade mathematical utilities for calculating exponentially weighted
moving averages where older observations are penalised with exponential decay based on their
temporal distance from the calculation point. Supports both unbounded recursive formulations
and strictly bounded historical lookback windows.

Mathematical Formulation:
    For observation sequence x_0, x_1, ..., x_{T-1} at session T:
    Distance / Lag k in {0, 1, ..., W-1} (where k=0 is yesterday T-1).
    Penalisation weight:
        w_k = (1 - alpha)^k = exp(-lambda * k)
    where:
        alpha = 2 / (span + 1)
        lambda = ln(2) / half_life

    Bounded EWMA over window W:
        ewma_T(x, W, alpha) = sum_{k=0}^{W-1} (w_k * x_{T-1-k}) / sum_{k=0}^{W-1} w_k
"""

import math
from typing import Optional, Sequence

import numpy as np
import polars as pl

# Standard institutional horizons in trading days
HORIZON_MAP = {
    "1W": 5,     # 1 Week (5 trading days) - Tactical Impulse
    "2W": 10,    # 2 Weeks (10 trading days) - Bi-Weekly Swing
    "1M": 21,    # 1 Month (21 trading days) - Monthly Rebalancing
    "3M": 63,    # 3 Months (63 trading days) - Quarterly Cycle
    "6M": 126,   # 6 Months (126 trading days) - Semi-Annual Allocation
    "12M": 252,  # 12 Months (252 trading days) - Core Custody Anchor
}


def alpha_from_span(span: float) -> float:
    """Calculate smoothing factor alpha from span N: alpha = 2 / (N + 1)."""
    if span <= 0:
        raise ValueError(f"Span must be positive, got {span}")
    return 2.0 / (span + 1.0)


def span_from_half_life(half_life: float) -> float:
    """Convert decay half-life in days to equivalent span N: alpha = 1 - 2^(-1/half_life)."""
    if half_life <= 0:
        raise ValueError(f"Half-life must be positive, got {half_life}")
    alpha = 1.0 - math.pow(2.0, -1.0 / half_life)
    return (2.0 / alpha) - 1.0


def compute_ewma_weights(
    window: int,
    span: Optional[float] = None,
    half_life: Optional[float] = None,
    alpha: Optional[float] = None,
) -> np.ndarray:
    """Generate normalized exponential decay weights penalising distance from current point.

    The first element (index 0) corresponds to the most recent observation (lag 0),
    and subsequent elements decay exponentially. Weights sum strictly to 1.0.

    Args:
        window: Number of lookback periods (W).
        span: Span parameter N (alpha = 2 / (N + 1)).
        half_life: Decay half-life in periods (alpha = 1 - 2^(-1/half_life)).
        alpha: Direct smoothing parameter in (0, 1].

    Returns:
        1D numpy array of length `window` summing to 1.0.
    """
    if window <= 0:
        raise ValueError(f"Window must be positive, got {window}")

    if alpha is not None:
        a = float(alpha)
    elif span is not None:
        a = alpha_from_span(span)
    elif half_life is not None:
        a = 1.0 - math.pow(2.0, -1.0 / half_life)
    else:
        # Default span equals window
        a = alpha_from_span(float(window))

    if not (0.0 < a <= 1.0):
        raise ValueError(f"Smoothing parameter alpha must be in (0, 1], got {a}")

    # Lags: 0, 1, 2, ..., window - 1
    lags = np.arange(window, dtype=np.float64)
    # Weights decay with distance: w_k = (1 - alpha)^k
    raw_weights = np.power(1.0 - a, lags)
    weight_sum = np.sum(raw_weights)

    if weight_sum == 0.0:
        return np.ones(window, dtype=np.float64) / float(window)

    return raw_weights / weight_sum


def calculate_bounded_ewma(
    values: Sequence[float],
    window: int,
    span: Optional[float] = None,
    half_life: Optional[float] = None,
) -> float:
    """Compute bounded EWMA for the most recent observation given trailing history.

    Values are assumed to be chronological with the latest observation at index -1.
    Only the last `window` observations are considered, with exponential penalisation
    applied to earlier values.

    Args:
        values: Historical sequence of numerical values.
        window: Maximum lookback cutoff W.
        span: Exponential span parameter (defaults to window).
        half_life: Optional half-life parameter.

    Returns:
        Weighted scalar average.
    """
    if not values:
        return 0.0

    arr = np.asarray(values, dtype=np.float64)
    n = len(arr)
    effective_w = min(n, window)

    if effective_w == 0:
        return 0.0

    # Slice the last effective_w elements and reverse so lag 0 is first
    recent_slice = arr[-effective_w:][::-1]
    weights = compute_ewma_weights(effective_w, span=span, half_life=half_life)

    return float(np.dot(recent_slice, weights))


def calculate_size_weighted_bounded_ewma(
    values: Sequence[float],
    sizes: Sequence[float],
    window: int,
    span: Optional[float] = None,
    half_life: Optional[float] = None,
) -> float:
    """Compute Bounded EWMA combining exponential recency decay with position size magnitude.

    Compound Weight Formulation:
        For lag k in {0, ..., W-1}:
            Time Decay Weight: w_time_k = (1 - alpha)^k
            Position Magnitude: s_k = max(|Size_k|, epsilon)
            Combined Weight: Omega_k = w_time_k * s_k

        Result:
            EWMA = sum(Omega_k * Value_k) / sum(Omega_k)

    Args:
        values: Numerical series to smooth (e.g. costs, inventory levels).
        sizes: Position magnitude / volume taken on each session (e.g. |net_flow_tl| or |net_volume|).
        window: Maximum lookback cutoff W.
        span: Exponential span parameter (defaults to window).
        half_life: Optional half-life parameter.

    Returns:
        Recency-and-Size weighted average scalar.
    """
    if not values:
        return 0.0

    arr_val = np.asarray(values, dtype=np.float64)
    arr_size = np.asarray(sizes, dtype=np.float64)
    n = len(arr_val)
    effective_w = min(n, window, len(arr_size))

    if effective_w == 0:
        return 0.0

    recent_val = arr_val[-effective_w:][::-1]
    recent_sizes = np.abs(arr_size[-effective_w:][::-1])

    # Time recency weights (sum to 1)
    time_weights = compute_ewma_weights(effective_w, span=span, half_life=half_life)

    # Compound weights: Time Recency * Position Size
    epsilon = 1e-6
    compound_weights = time_weights * (recent_sizes + epsilon)
    sum_weights = np.sum(compound_weights)

    if sum_weights == 0.0:
        return float(np.mean(recent_val))

    return float(np.dot(recent_val, compound_weights) / sum_weights)


def compute_ewma_series_numpy(
    values: Sequence[float],
    span: float,
) -> np.ndarray:
    """Compute full recursive continuous EWMA series using standard vectorized exponential decay.

    Formula:
        ewma_0 = x_0
        ewma_t = alpha * x_t + (1 - alpha) * ewma_{t-1}
    """
    arr = np.asarray(values, dtype=np.float64)
    n = len(arr)
    if n == 0:
        return np.array([], dtype=np.float64)

    alpha = alpha_from_span(span)
    result = np.empty(n, dtype=np.float64)
    result[0] = arr[0]

    for t in range(1, n):
        result[t] = alpha * arr[t] + (1.0 - alpha) * result[t - 1]

    return result


def add_multi_horizon_ewma_polars(
    df: pl.DataFrame,
    value_col: str,
    prefix: str = "ewma",
    horizons: Optional[dict[str, int]] = None,
) -> pl.DataFrame:
    """Add multi-horizon EWMA columns to a sorted Polars DataFrame.

    Computes:
        - 1W: span = 5
        - 2W: span = 10
        - 1M: span = 21
        - 3M: span = 63
        - 6M: span = 126
        - 12M: span = 252

    Args:
        df: Polars DataFrame sorted chronologically.
        value_col: Column name to smooth.
        prefix: Column prefix (e.g. 'ewma_qty' or 'ewma_cost').
        horizons: Mapping of horizon tag to span (defaults to HORIZON_MAP).

    Returns:
        Enriched DataFrame with new EWMA columns.
    """
    h_map = horizons or HORIZON_MAP
    exprs = []

    for tag, span in h_map.items():
        col_name = f"{prefix}_{tag.lower()}_{span}d"
        exprs.append(
            pl.col(value_col).ewm_mean(span=span, adjust=False).alias(col_name)
        )

    return df.with_columns(exprs)
