"""Unified Model Explainer with TreeSHAP, Linear Attribution, and Cluster Aggregations."""

from datetime import date
from typing import Any, Dict, List, Optional, Union

import numpy as np
import pandas as pd

from mdk_trading_oracle.core.logger import get_logger
from mdk_trading_oracle.explainability.types import (
    ClusterAttribution,
    FeatureAttribution,
    GlobalExplanation,
    LocalExplanation,
)

logger = get_logger("mdk_oracle.explainability.explainer")

# Check for SHAP availability
try:
    import shap
    HAS_SHAP = True
except ImportError:
    HAS_SHAP = False


class ModelExplainer:
    """Unified explainability engine supporting tree models, linear/Bayesian models, and ensembles."""

    def __init__(
        self,
        model: Any,
        feature_names: List[str],
        cluster_map: Optional[Dict[str, str]] = None,
        background_data: Optional[pd.DataFrame] = None,
        model_name: str = "Model",
        model_version: str = "1.0.0",
        unit: str = "TL",
    ):
        """Initialize ModelExplainer.

        Args:
            model: Trained estimator or forecaster instance.
            feature_names: List of feature column names in model input order.
            cluster_map: Mapping of feature column name to its parent semantic cluster.
            background_data: Historical feature matrix used as baseline reference.
            model_name: Name of the underlying forecaster.
            model_version: Version string.
            unit: Units of the target variable ('TL' or '%').
        """
        self.raw_model = model
        self.feature_names = list(feature_names)
        self.cluster_map = cluster_map or {}
        self.model_name = model_name
        self.model_version = model_version
        self.unit = unit

        # Extract underlying estimator if wrapped
        self.estimator = self._extract_estimator(model)
        self.background_data = background_data
        self.shap_explainer = None

        self._init_explainer()

    def _extract_estimator(self, model: Any) -> Any:
        """Unwrap estimator from forecaster class if needed."""
        if hasattr(model, "model") and getattr(model, "model") is not None:
            return getattr(model, "model")
        if hasattr(model, "regressor") and getattr(model, "regressor") is not None:
            return getattr(model, "regressor")
        if hasattr(model, "fitted_model_") and getattr(model, "fitted_model_") is not None:
            return getattr(model, "fitted_model_")
        return model

    def _init_explainer(self) -> None:
        """Initialize SHAP explainer backend based on estimator type."""
        if not HAS_SHAP:
            logger.info("SHAP package not available; using analytical and model-native explainability.")
            return

        est_type = type(self.estimator).__name__

        try:
            # Tree-based models (LightGBM, XGBoost)
            if "LGBM" in est_type or "Booster" in est_type or "XGB" in est_type:
                self.shap_explainer = shap.TreeExplainer(self.estimator)
                logger.debug(f"Initialized TreeSHAP explainer for {est_type}")
            # Linear and Bayesian Ridge models
            elif hasattr(self.estimator, "coef_") and self.background_data is not None:
                # Use a small representative sample of background data for speed
                bg_sample = self.background_data[self.feature_names].dropna()
                if len(bg_sample) > 50:
                    bg_sample = bg_sample.sample(50, random_state=42)
                if not bg_sample.empty:
                    self.shap_explainer = shap.LinearExplainer(self.estimator, bg_sample)
                    logger.debug(f"Initialized LinearExplainer for {est_type}")
        except Exception as e:
            logger.warning(f"Failed to initialize SHAP explainer ({e}); falling back to analytical attribution.")
            self.shap_explainer = None

    def explain_instance(
        self,
        X_row: Union[pd.DataFrame, pd.Series, Dict[str, Any]],
        target_broker_or_symbol: str = "MLB",
        target_date: Optional[date] = None,
        top_n: int = 3,
    ) -> LocalExplanation:
        """Generate additive feature attribution decomposition for a single prediction.

        Args:
            X_row: Feature vector for the target session (DataFrame or Series).
            target_broker_or_symbol: Ticker symbol or clearing broker code.
            target_date: Date of target forecast.
            top_n: Number of top positive/negative drivers to extract.

        Returns:
            LocalExplanation containing base value, prediction, and driver breakdowns.
        """
        # Ensure row is a 1-row DataFrame aligned to feature_names
        df_row = self._prepare_row(X_row)

        attributions: List[FeatureAttribution] = []
        base_val = 0.0
        pred_val = 0.0

        if self.shap_explainer is not None:
            try:
                shap_res = self.shap_explainer(df_row)
                raw_values = shap_res.values
                if len(raw_values.shape) == 2:
                    raw_values = raw_values[0]
                elif len(raw_values.shape) == 3:
                    raw_values = raw_values[0, :, 0]

                raw_base = shap_res.base_values
                if isinstance(raw_base, (np.ndarray, list)):
                    base_val = float(raw_base[0])
                else:
                    base_val = float(raw_base)

                for feat, val, attr in zip(self.feature_names, df_row.iloc[0], raw_values):
                    cluster = self.cluster_map.get(feat, "other")
                    attributions.append(
                        FeatureAttribution(
                            feature_name=feat,
                            cluster_name=cluster,
                            feature_value=float(val),
                            attribution=float(attr),
                        )
                    )
                pred_val = base_val + sum(a.attribution for a in attributions)
            except Exception as e:
                logger.warning(f"TreeSHAP calculation failed ({e}); falling back to analytical attribution.")
                attributions, base_val, pred_val = self._analytical_attribution(df_row)
        else:
            attributions, base_val, pred_val = self._analytical_attribution(df_row)

        # Microstructure Cluster Rollup
        cluster_attributions = self._aggregate_clusters(attributions)

        # Top Positive (Catalysts) and Negative (Headwinds)
        positive_sorted = sorted(
            [a for a in attributions if a.attribution > 0],
            key=lambda x: x.attribution,
            reverse=True,
        )
        negative_sorted = sorted(
            [a for a in attributions if a.attribution < 0],
            key=lambda x: x.attribution,
        )

        return LocalExplanation(
            model_name=self.model_name,
            model_version=self.model_version,
            target_broker_or_symbol=target_broker_or_symbol,
            target_date=target_date,
            unit=self.unit,
            base_value=base_val,
            predicted_value=pred_val,
            feature_attributions=attributions,
            cluster_attributions=cluster_attributions,
            top_positive_drivers=positive_sorted[:top_n],
            top_negative_drivers=negative_sorted[:top_n],
        )

    def explain_global(self, X: pd.DataFrame) -> GlobalExplanation:
        """Compute dataset-wide feature importances and semantic cluster shares.

        Args:
            X: Feature matrix across historical sessions.

        Returns:
            GlobalExplanation containing feature and cluster rankings.
        """
        df_clean = X[self.feature_names].dropna()
        if df_clean.empty:
            logger.warning("No valid samples provided for global explanation.")
            return GlobalExplanation(
                model_name=self.model_name,
                model_version=self.model_version,
                unit=self.unit,
                feature_importance_df=pd.DataFrame(),
                cluster_importance_df=pd.DataFrame(),
            )

        mean_abs_attributions: Dict[str, float] = {}

        if self.shap_explainer is not None and HAS_SHAP:
            try:
                # Subsample if large to keep computation sub-second
                eval_sample = df_clean.sample(min(150, len(df_clean)), random_state=42)
                shap_vals = self.shap_explainer(eval_sample).values
                if len(shap_vals.shape) == 3:
                    shap_vals = shap_vals[:, :, 0]
                mean_vals = np.mean(np.abs(shap_vals), axis=0)
                for feat, val in zip(self.feature_names, mean_vals):
                    mean_abs_attributions[feat] = float(val)
            except Exception as e:
                logger.warning(f"Global SHAP calculation failed ({e}); falling back to model-intrinsic importance.")
                mean_abs_attributions = self._model_intrinsic_importance(df_clean)
        else:
            mean_abs_attributions = self._model_intrinsic_importance(df_clean)

        total_imp = sum(mean_abs_attributions.values()) or 1.0

        # Build feature DataFrame
        records = []
        for feat, imp in mean_abs_attributions.items():
            records.append(
                {
                    "feature_name": feat,
                    "cluster_name": self.cluster_map.get(feat, "other"),
                    "mean_abs_attribution": imp,
                    "relative_importance_pct": (imp / total_imp) * 100.0,
                }
            )
        feat_df = pd.DataFrame(records).sort_values("mean_abs_attribution", ascending=False).reset_index(drop=True)
        feat_df["rank"] = range(1, len(feat_df) + 1)

        # Build cluster DataFrame
        cluster_records = []
        for cluster_name, group in feat_df.groupby("cluster_name"):
            c_imp = group["mean_abs_attribution"].sum()
            cluster_records.append(
                {
                    "cluster_name": cluster_name,
                    "total_abs_attribution": c_imp,
                    "relative_importance_pct": (c_imp / total_imp) * 100.0,
                    "feature_count": len(group),
                    "top_feature": group.iloc[0]["feature_name"],
                }
            )
        cluster_df = (
            pd.DataFrame(cluster_records).sort_values("total_abs_attribution", ascending=False).reset_index(drop=True)
        )
        cluster_df["rank"] = range(1, len(cluster_df) + 1)

        top_features = [
            (row["feature_name"], row["mean_abs_attribution"])
            for _, row in feat_df.head(10).iterrows()
        ]

        return GlobalExplanation(
            model_name=self.model_name,
            model_version=self.model_version,
            unit=self.unit,
            feature_importance_df=feat_df,
            cluster_importance_df=cluster_df,
            top_features=top_features,
        )

    def _prepare_row(self, X_row: Union[pd.DataFrame, pd.Series, Dict[str, Any]]) -> pd.DataFrame:
        """Standardize input row to a 1-row DataFrame strictly containing active features."""
        if isinstance(X_row, pd.Series):
            df_row = pd.DataFrame([X_row.to_dict()])
        elif isinstance(X_row, dict):
            df_row = pd.DataFrame([X_row])
        elif isinstance(X_row, pd.DataFrame):
            df_row = X_row.iloc[[0]].copy()
        else:
            raise ValueError(f"Unsupported row format: {type(X_row)}")

        # Ensure all feature columns exist, coalescing missing to 0.0
        for feat in self.feature_names:
            if feat not in df_row.columns:
                df_row[feat] = 0.0

        return df_row[self.feature_names].astype(float)

    def _analytical_attribution(
        self, df_row: pd.DataFrame
    ) -> (List[FeatureAttribution], float, float):
        """Analytical attribution fallback for linear models and generic estimators."""
        attributions = []
        row_vals = df_row.iloc[0]

        # Case A: Linear / Bayesian Ridge with coefficients
        if hasattr(self.estimator, "coef_"):
            coefs = np.asarray(self.estimator.coef_).flatten()
            intercept = float(getattr(self.estimator, "intercept_", 0.0))

            # Reference baseline means
            if self.background_data is not None and not self.background_data.empty:
                means = self.background_data[self.feature_names].mean()
            else:
                means = pd.Series(0.0, index=self.feature_names)

            base_val = intercept + float(np.dot(coefs, means.fillna(0.0).values))

            for feat, coef in zip(self.feature_names, coefs):
                x_val = float(row_vals.get(feat, 0.0))
                x_mean = float(means.get(feat, 0.0))
                attr = coef * (x_val - x_mean)
                cluster = self.cluster_map.get(feat, "other")
                attributions.append(
                    FeatureAttribution(
                        feature_name=feat,
                        cluster_name=cluster,
                        feature_value=x_val,
                        attribution=float(attr),
                    )
                )
            pred_val = base_val + sum(a.attribution for a in attributions)
            return attributions, base_val, pred_val

        # Case B: Tree model without SHAP (proportional to feature_importances_)
        if hasattr(self.estimator, "feature_importances_"):
            importances = np.asarray(self.estimator.feature_importances_).flatten()
            norm_imp = importances / (np.sum(importances) or 1.0)

            try:
                pred_val = float(self.estimator.predict(df_row)[0])
            except Exception:
                pred_val = 0.0

            base_val = 0.0
            for feat, val, imp in zip(self.feature_names, row_vals, norm_imp):
                cluster = self.cluster_map.get(feat, "other")
                # Scale relative to predicted value
                attr = float(pred_val * imp)
                attributions.append(
                    FeatureAttribution(
                        feature_name=feat,
                        cluster_name=cluster,
                        feature_value=float(val),
                        attribution=attr,
                    )
                )
            return attributions, base_val, pred_val

        # Case C: Baseline or constant prediction
        try:
            pred_val = float(self.raw_model.predict(df_row).predicted_net_flow_tl)
        except Exception:
            pred_val = 0.0

        for feat in self.feature_names:
            attributions.append(
                FeatureAttribution(
                    feature_name=feat,
                    cluster_name=self.cluster_map.get(feat, "other"),
                    feature_value=float(row_vals.get(feat, 0.0)),
                    attribution=0.0,
                )
            )
        return attributions, pred_val, pred_val

    def _model_intrinsic_importance(self, df: pd.DataFrame) -> Dict[str, float]:
        """Extract intrinsic feature importance from model when SHAP is unavailable."""
        if hasattr(self.estimator, "feature_importances_"):
            return {
                feat: float(imp)
                for feat, imp in zip(self.feature_names, self.estimator.feature_importances_)
            }
        elif hasattr(self.estimator, "coef_"):
            coefs = np.abs(np.asarray(self.estimator.coef_).flatten())
            # Scale by feature standard deviation for standardized beta importance
            stds = df[self.feature_names].std().fillna(1.0).values
            standardized = coefs * stds
            return {feat: float(imp) for feat, imp in zip(self.feature_names, standardized)}

        return {feat: 1.0 / len(self.feature_names) for feat in self.feature_names}

    def _aggregate_clusters(
        self, attributions: List[FeatureAttribution]
    ) -> List[ClusterAttribution]:
        """Aggregate feature attributions up to semantic microstructure clusters."""
        cluster_groups: Dict[str, List[FeatureAttribution]] = {}
        for attr in attributions:
            cluster_groups.setdefault(attr.cluster_name, []).append(attr)

        total_abs_all = sum(a.attribution_abs for a in attributions) or 1.0

        clusters: List[ClusterAttribution] = []
        for cluster_name, attrs in cluster_groups.items():
            net_signed = sum(a.attribution for a in attrs)
            sum_abs = sum(a.attribution_abs for a in attrs)
            top_feat = max(attrs, key=lambda x: x.attribution_abs).feature_name if attrs else ""

            clusters.append(
                ClusterAttribution(
                    cluster_name=cluster_name,
                    total_attribution=net_signed,
                    total_abs_attribution=sum_abs,
                    percentage_share=(sum_abs / total_abs_all) * 100.0,
                    feature_count=len(attrs),
                    top_feature=top_feat,
                )
            )

        # Sort clusters by absolute attribution descending
        return sorted(clusters, key=lambda c: c.total_abs_attribution, reverse=True)
