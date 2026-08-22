"""Feature Catalog & Selection Engine for Predictive Gold Models."""

from typing import Any, Dict, List, Optional, Set, Union

import pandas as pd
import polars as pl

from mdk_trading_oracle.core.config import get_settings
from mdk_trading_oracle.core.logger import get_logger

logger = get_logger("mdk_oracle.models.features_config")


class FeatureSelector:
    """Manages declarative feature selection, cluster resolution, and granular column filtering."""

    def __init__(
        self,
        model_name: str = "day_start",
        config: Optional[Dict[str, Any]] = None,
        disabled_clusters: Optional[List[str]] = None,
        enabled_clusters: Optional[List[str]] = None,
        include_features: Optional[List[str]] = None,
        exclude_features: Optional[List[str]] = None,
    ):
        """Initialize FeatureSelector for a specific model.

        Args:
            model_name: Target model identifier ('day_start' or 'sector_day_start').
            config: Optional raw dictionary config override (defaults to loading from features.yaml).
            disabled_clusters: Optional list of cluster names to deactivate at runtime.
            enabled_clusters: Optional explicit list of cluster names to activate (deactivates others).
            include_features: Optional list of specific feature column names to force include.
            exclude_features: Optional list of specific feature column names to exclude.
        """
        self.model_name = model_name
        settings = get_settings()
        self.raw_config: Dict[str, Any] = config or settings.get_features_config(model_name)

        self.cluster_catalog: Dict[str, Dict[str, Any]] = self.raw_config.get("clusters", {})
        self.default_include: List[str] = self.raw_config.get("include_features", []) or []
        self.default_exclude: List[str] = self.raw_config.get("exclude_features", []) or []

        # Runtime overrides
        self.override_disabled_clusters = set(disabled_clusters or [])
        self.override_enabled_clusters = set(enabled_clusters) if enabled_clusters is not None else None
        self.override_include_features = set(include_features or [])
        self.override_exclude_features = set(exclude_features or [])

        # Validate and compute resolved active features
        self.active_features: List[str] = self._resolve_active_features()

    def _resolve_active_features(self) -> List[str]:
        """Compute the ordered list of active feature columns based on clusters and override rules."""
        active_set: Set[str] = set()

        for cluster_name, cluster_info in self.cluster_catalog.items():
            is_enabled_default = cluster_info.get("enabled", True)
            
            # Check cluster activation
            if self.override_enabled_clusters is not None:
                cluster_active = cluster_name in self.override_enabled_clusters
            else:
                cluster_active = is_enabled_default and (cluster_name not in self.override_disabled_clusters)

            if cluster_active:
                for feat in cluster_info.get("features", []):
                    active_set.add(feat)

        # Apply include overrides (force add)
        for feat in self.default_include:
            active_set.add(feat)
        for feat in self.override_include_features:
            active_set.add(feat)

        # Apply exclude overrides (force remove)
        for feat in self.default_exclude:
            active_set.discard(feat)
        for feat in self.override_exclude_features:
            active_set.discard(feat)

        # Preserve canonical catalog ordering
        all_canonical_features = self.get_all_features()
        ordered_active = [f for f in all_canonical_features if f in active_set]

        # Add any explicitly included features not in catalog
        for feat in active_set:
            if feat not in ordered_active:
                ordered_active.append(feat)

        if not ordered_active:
            logger.warning(
                f"FeatureSelector for '{self.model_name}' resulted in 0 active features! "
                "Check cluster enable flags and exclusion lists."
            )

        logger.debug(
            f"FeatureSelector resolved {len(ordered_active)} active feature(s) for '{self.model_name}' "
            f"(out of {len(all_canonical_features)} total available)."
        )
        return ordered_active

    def get_all_features(self) -> List[str]:
        """Return the complete canonical list of all feature columns across all clusters."""
        features: List[str] = []
        for cluster_info in self.cluster_catalog.values():
            for feat in cluster_info.get("features", []):
                if feat not in features:
                    features.append(feat)
        return features

    def get_available_clusters(self) -> List[str]:
        """Return list of all available cluster names."""
        return list(self.cluster_catalog.keys())

    def get_active_clusters(self) -> List[str]:
        """Return list of cluster names that contribute at least one active feature."""
        active_set = set(self.active_features)
        active_clusters = []
        for cluster_name, cluster_info in self.cluster_catalog.items():
            cluster_feats = set(cluster_info.get("features", []))
            if cluster_feats.intersection(active_set):
                active_clusters.append(cluster_name)
        return active_clusters

    def get_active_features(self) -> List[str]:
        """Return the list of currently active feature column names."""
        return list(self.active_features)

    def get_feature_cluster_map(self) -> Dict[str, str]:
        """Return mapping of feature column name to its parent cluster name."""
        mapping: Dict[str, str] = {}
        for cluster_name, cluster_info in self.cluster_catalog.items():
            for feat in cluster_info.get("features", []):
                mapping[feat] = cluster_name
        return mapping

    def filter_dataframe(self, df: Union[pd.DataFrame, pl.DataFrame]) -> Union[pd.DataFrame, pl.DataFrame]:
        """Filter a DataFrame to retain only non-feature metadata/target columns PLUS active features.

        Args:
            df: Input pandas or polars DataFrame containing raw extracted features and targets.

        Returns:
            Filtered DataFrame with deactivated feature columns dropped.
        """
        all_catalog_features = set(self.get_all_features())
        active_set = set(self.active_features)
        features_to_drop = all_catalog_features - active_set

        if isinstance(df, pl.DataFrame):
            existing_drop_cols = [col for col in features_to_drop if col in df.columns]
            if existing_drop_cols:
                return df.drop(existing_drop_cols)
            return df
        elif isinstance(df, pd.DataFrame):
            existing_drop_cols = [col for col in features_to_drop if col in df.columns]
            if existing_drop_cols:
                return df.drop(columns=existing_drop_cols)
            return df
        return df

    def summary(self) -> Dict[str, Any]:
        """Return a structured summary of active vs disabled clusters and features."""
        all_feats = self.get_all_features()
        return {
            "model_name": self.model_name,
            "total_available_features": len(all_feats),
            "active_feature_count": len(self.active_features),
            "active_features": self.active_features,
            "active_clusters": self.get_active_clusters(),
            "disabled_clusters": [c for c in self.get_available_clusters() if c not in self.get_active_clusters()],
            "excluded_features": [f for f in all_feats if f not in self.active_features],
        }
