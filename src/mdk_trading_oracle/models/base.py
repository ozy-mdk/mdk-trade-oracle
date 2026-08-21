"""Abstract Base Classes and Data Transfer Objects for Gold Layer Predictive Models."""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import date, datetime
from typing import Any, Dict, List, Optional

import pandas as pd
import polars as pl

from mdk_trading_oracle.core.time import now_turkey_naive


class ForecastDirection(str):
    """Institutional directional posture for opening session."""
    STRONG_BUY = "STRONG_BUY"
    BUY = "BUY"
    WEAK_BUY = "WEAK_BUY"
    NEUTRAL = "NEUTRAL"
    WEAK_SELL = "WEAK_SELL"
    SELL = "SELL"
    STRONG_SELL = "STRONG_SELL"

    # Backward compatibility aliases
    STRONG_ACCUMULATE = "STRONG_BUY"
    ACCUMULATE = "BUY"
    DISTRIBUTE = "SELL"
    STRONG_DISTRIBUTE = "STRONG_SELL"

    @classmethod
    def all_valid(cls) -> List[str]:
        """Return list of all valid directional classification values."""
        return [
            cls.STRONG_BUY,
            cls.BUY,
            cls.WEAK_BUY,
            cls.NEUTRAL,
            cls.WEAK_SELL,
            cls.SELL,
            cls.STRONG_SELL,
            "STRONG_ACCUMULATE",
            "ACCUMULATE",
            "DISTRIBUTE",
            "STRONG_DISTRIBUTE",
        ]


@dataclass
class FlowThresholdProfile:
    """Empirical percentile distribution thresholds for institutional net flow classification."""
    buy_p25_tl: float = 10e6
    buy_p50_tl: float = 30e6
    buy_p85_tl: float = 75e6
    sell_p25_tl: float = 10e6
    sell_p50_tl: float = 30e6
    sell_p85_tl: float = 75e6
    buy_count: int = 0
    sell_count: int = 0
    total_sessions: int = 0


class FlowThresholdClassifier:
    """Classifies predicted net flows into statistical conviction levels based on empirical percentiles."""

    @staticmethod
    def classify(net_flow_tl: float, thresholds: Optional[FlowThresholdProfile] = None) -> str:
        """Classify continuous predicted net flow (TL) into dynamic percentile direction."""
        th = thresholds or FlowThresholdProfile()
        if net_flow_tl > 0:
            if net_flow_tl >= th.buy_p85_tl:
                return ForecastDirection.STRONG_BUY
            elif net_flow_tl >= th.buy_p50_tl:
                return ForecastDirection.BUY
            elif net_flow_tl >= th.buy_p25_tl:
                return ForecastDirection.WEAK_BUY
            else:
                return ForecastDirection.NEUTRAL
        elif net_flow_tl < 0:
            abs_flow = abs(net_flow_tl)
            if abs_flow >= th.sell_p85_tl:
                return ForecastDirection.STRONG_SELL
            elif abs_flow >= th.sell_p50_tl:
                return ForecastDirection.SELL
            elif abs_flow >= th.sell_p25_tl:
                return ForecastDirection.WEAK_SELL
            else:
                return ForecastDirection.NEUTRAL
        return ForecastDirection.NEUTRAL


class OpeningPlaybook(str):
    """Actionable institutional execution playbook classification."""
    SQUEEZE_LONG = "SQUEEZE_LONG"          # Exploiting competitor short deficit / panic chasing
    MOMENTUM_EXPANSION = "MOMENTUM_EXPANSION"  # Coordinated institutional sweep of the offer
    LIQUIDITY_FADE = "LIQUIDITY_FADE"      # Selling into domestic morning retail euphoria
    SECTOR_ROTATION = "SECTOR_ROTATION"    # Shifting capital into uncrowded / lagging sectors
    DEFENSE_SUPPORT = "DEFENSE_SUPPORT"    # Bidding at 20-day cost basis support level
    NEUTRAL_WAIT = "NEUTRAL_WAIT"          # Low conviction / range-bound tape


@dataclass
class ForecastResult:
    """Standardized output container for probabilistic model forecasts."""
    forecast_date: date
    target_broker_id: str
    predicted_net_flow_tl: float
    predicted_flow_lower_90: float
    predicted_flow_upper_90: float
    predicted_direction: str
    direction_confidence: float
    predicted_playbook: str
    top_predicted_buy_sector: str
    top_predicted_sell_sector: str
    model_name: str
    model_version: str
    features_used: Dict[str, Any] = field(default_factory=dict)
    sector_forecasts: List[Dict[str, Any]] = field(default_factory=list)
    generated_at: datetime = field(default_factory=now_turkey_naive)


class BaseFeatureExtractor(ABC):
    """Abstract feature extractor for preparing model-ready datasets from Silver fact tables."""

    @abstractmethod
    def extract_features(self, start_date: Optional[date] = None, end_date: Optional[date] = None) -> pl.DataFrame:
        """Extract multi-cluster feature matrix from DuckDB Silver tables.
        
        Returns:
            pl.DataFrame: Clean tabular dataset with features and target columns.
        """
        pass


class BaseForecaster(ABC):
    """Abstract base class for all predictive models in MDK Trading Oracle.
    
    Designed to scale to 10+ distinct models (e.g. Day Start, Intraday Expansion, 
    Weekly Bias, Sector Rotations, Institutional Divergence).
    """

    def __init__(self, model_name: str, model_version: str = "1.0.0"):
        self.model_name = model_name
        self.model_version = model_version
        self.is_fitted = False

    @abstractmethod
    def fit(self, X: pd.DataFrame, y: pd.Series) -> "BaseForecaster":
        """Train the model on historical features and targets."""
        pass

    @abstractmethod
    def predict(self, X: pd.DataFrame) -> ForecastResult:
        """Generate point and probabilistic interval predictions for a single inference instance."""
        pass

    @abstractmethod
    def evaluate(self, X_test: pd.DataFrame, y_test: pd.Series) -> Dict[str, float]:
        """Evaluate model against test data and return metrics (MAE, Hit Rate %, PICP)."""
        pass
