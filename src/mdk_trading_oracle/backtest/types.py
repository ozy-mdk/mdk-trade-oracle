"""Data transfer objects and type definitions for unified backtest performance analysis."""

from dataclasses import dataclass, field
from datetime import date
from enum import Enum
from typing import Any, Dict, List, Optional


class TargetUnit(str, Enum):
    """Target unit for scaling and formatting."""

    TL = "TL"
    PERCENTAGE = "%"


class ModelScope(str, Enum):
    """Model scope classification."""

    DAY_START_MACRO = "DAY_START_MACRO"
    SECTOR_DAY_START = "SECTOR_DAY_START"
    STOCK_REACTION = "STOCK_REACTION"


@dataclass
class RegressionMetrics:
    """Continuous regression accuracy metrics."""

    mae: float
    rmse: float
    pearson_r: float
    r2: float
    mean_bias: float
    normalized_mae: float
    max_error: float


@dataclass
class DirectionalMetrics:
    """Directional hit rate and conviction tier metrics."""

    hit_rate_pct: float
    total_predictions: int
    hits: int
    misses: int
    conviction_hit_rates: Dict[str, float] = field(default_factory=dict)
    conviction_counts: Dict[str, int] = field(default_factory=dict)
    monotonicity_score: float = 0.0  # Correlation between conviction level and hit rate


@dataclass
class ProbabilisticMetrics:
    """Credible interval coverage and calibration metrics."""

    picp_90_pct: float  # Prediction Interval Coverage Probability (nominal: 90%)
    mpiw: float  # Mean Prediction Interval Width (sharpness)
    coverage_bias_pct: float  # picp_90_pct - 90.0%
    interval_score: float = 0.0  # Winkler score for 90% interval


@dataclass
class TradingUtilityMetrics:
    """Simulated trading utility and directional capture proxy."""

    directional_capture_total: float  # sum of actual * sign(predicted)
    win_loss_ratio: float
    profit_factor: float
    cumulative_directional_pnl: List[float] = field(default_factory=list)


@dataclass
class SliceMetrics:
    """Metrics aggregated across a specific slice/sub-segment."""

    slice_type: str  # "sector", "symbol", "window", "day_of_week", "playbook"
    slice_key: str  # e.g., "XBANK", "THYAO", "W2", "Monday", "SQUEEZE_LONG"
    sample_count: int
    hit_rate_pct: float
    mae: float
    rmse: float
    picp_90_pct: float
    directional_capture: float
    extra_attributes: Dict[str, Any] = field(default_factory=dict)


@dataclass
class BacktestSummary:
    """Unified container summarizing complete backtest performance."""

    model_name: str
    target_name: str
    target_unit: TargetUnit
    total_samples: int
    start_date: Optional[date]
    end_date: Optional[date]
    regression: RegressionMetrics
    directional: DirectionalMetrics
    probabilistic: ProbabilisticMetrics
    trading: TradingUtilityMetrics
    slices: Dict[str, List[SliceMetrics]] = field(default_factory=dict)
    calculated_at: Optional[str] = None
