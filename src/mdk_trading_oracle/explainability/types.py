"""Explainability Data Transfer Objects (DTOs) and Result Containers.

Supports local instance-level waterfall attribution (T+1 predictions),
global cross-session feature rankings, and automated feature selection audits.
"""

from dataclasses import asdict, dataclass, field
from datetime import date
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd


@dataclass
class FeatureAttribution:
    """Individual feature contribution for a specific inference instance."""

    feature_name: str
    cluster_name: str
    feature_value: float
    attribution: float  # Signed additive contribution (TL or %)
    attribution_abs: float = 0.0
    direction: str = "POSITIVE"  # 'POSITIVE' (catalyst) or 'NEGATIVE' (headwind)

    def __post_init__(self):
        self.attribution_abs = abs(self.attribution)
        self.direction = "POSITIVE" if self.attribution >= 0 else "NEGATIVE"


@dataclass
class ClusterAttribution:
    """Aggregated semantic cluster contribution for an inference instance."""

    cluster_name: str
    total_attribution: float  # Net signed sum of feature attributions
    total_abs_attribution: float  # Sum of absolute feature attributions
    percentage_share: float  # Relative share of absolute attribution (0 to 100%)
    feature_count: int
    top_feature: str = ""


@dataclass
class LocalExplanation:
    """Instance-level explanation for upcoming session T+1 forecast."""

    model_name: str
    model_version: str
    target_broker_or_symbol: str
    base_value: float  # E[y] baseline expected value
    predicted_value: float  # y_hat prediction
    unit: str = "TL"  # 'TL' or '%'
    target_date: Optional[date] = None
    feature_attributions: List[FeatureAttribution] = field(default_factory=list)
    cluster_attributions: List[ClusterAttribution] = field(default_factory=list)
    top_positive_drivers: List[FeatureAttribution] = field(default_factory=list)
    top_negative_drivers: List[FeatureAttribution] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        """Convert local explanation to a serializable dictionary for logging and persistence."""
        return {
            "model_name": self.model_name,
            "model_version": self.model_version,
            "target_broker_or_symbol": self.target_broker_or_symbol,
            "target_date": self.target_date.isoformat() if self.target_date else None,
            "unit": self.unit,
            "base_value": float(self.base_value),
            "predicted_value": float(self.predicted_value),
            "top_positive_drivers": [asdict(d) for d in self.top_positive_drivers],
            "top_negative_drivers": [asdict(d) for d in self.top_negative_drivers],
            "cluster_breakdown": {
                c.cluster_name: {
                    "total_attribution": float(c.total_attribution),
                    "percentage_share": float(c.percentage_share),
                    "top_feature": c.top_feature,
                }
                for c in self.cluster_attributions
            },
        }


@dataclass
class GlobalExplanation:
    """Global feature importance summary across multiple historical sessions."""

    model_name: str
    model_version: str
    unit: str
    feature_importance_df: pd.DataFrame
    cluster_importance_df: pd.DataFrame
    top_features: List[Tuple[str, float]] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        """Convert global explanation to serializable summary."""
        return {
            "model_name": self.model_name,
            "model_version": self.model_version,
            "unit": self.unit,
            "top_features": [
                {"feature": f, "importance": float(imp)} for f, imp in self.top_features
            ],
            "cluster_importance": (
                self.cluster_importance_df.to_dict(orient="records")
                if not self.cluster_importance_df.empty
                else []
            ),
        }


@dataclass
class FeatureAuditReport:
    """Diagnostics and recommendations for feature selection and configuration."""

    model_name: str
    evaluated_sessions: int
    total_features: int
    prune_candidates: List[Dict[str, Any]] = field(default_factory=list)
    collinear_pairs: List[Dict[str, Any]] = field(default_factory=list)
    top_drivers: List[Dict[str, Any]] = field(default_factory=list)
    recommended_features_yaml: str = ""
