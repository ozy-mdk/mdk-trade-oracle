"""Abstract Base Classes and Data Transfer Objects for Gold Layer Predictive Models."""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import date, datetime
from typing import Any, Dict, List, Optional

import pandas as pd
import polars as pl


class ForecastDirection(str):
    """Institutional directional posture for opening session."""
    STRONG_ACCUMULATE = "STRONG_ACCUMULATE"
    ACCUMULATE = "ACCUMULATE"
    NEUTRAL = "NEUTRAL"
    DISTRIBUTE = "DISTRIBUTE"
    STRONG_DISTRIBUTE = "STRONG_DISTRIBUTE"


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
    predicted_open_market_share: float
    predicted_playbook: str
    top_predicted_buy_sector: str
    top_predicted_sell_sector: str
    model_name: str
    model_version: str
    features_used: Dict[str, Any] = field(default_factory=dict)
    sector_forecasts: List[Dict[str, Any]] = field(default_factory=list)
    generated_at: datetime = field(default_factory=datetime.now)


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
