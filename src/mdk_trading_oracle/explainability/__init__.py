"""MDK Trading Oracle — Model Explainability, SHAP Attribution & Feature Selection Engine."""

from mdk_trading_oracle.explainability.explainer import ModelExplainer
from mdk_trading_oracle.explainability.selection import FeatureAuditor
from mdk_trading_oracle.explainability.types import (
    ClusterAttribution,
    FeatureAttribution,
    FeatureAuditReport,
    GlobalExplanation,
    LocalExplanation,
)
from mdk_trading_oracle.explainability.visualizer import (
    format_markdown_card,
    plot_cluster_donut,
    plot_waterfall,
)

__all__ = [
    "ModelExplainer",
    "FeatureAuditor",
    "FeatureAttribution",
    "ClusterAttribution",
    "LocalExplanation",
    "GlobalExplanation",
    "FeatureAuditReport",
    "plot_waterfall",
    "plot_cluster_donut",
    "format_markdown_card",
]
