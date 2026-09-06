"""Feature Auditor: Out-of-sample permutation testing and collinearity screening for data-driven feature pruning."""

from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.inspection import permutation_importance
from sklearn.metrics import mean_squared_error

from mdk_trading_oracle.core.logger import get_logger
from mdk_trading_oracle.explainability.explainer import ModelExplainer
from mdk_trading_oracle.explainability.types import FeatureAuditReport

logger = get_logger("mdk_oracle.explainability.selection")


class FeatureAuditor:
    """Evaluates out-of-sample feature alpha, collinear redundancy, and generates configuration recommendations."""

    def __init__(
        self,
        feature_names: List[str],
        cluster_map: Optional[Dict[str, str]] = None,
        collinearity_threshold: float = 0.85,
        n_repeats: int = 5,
        random_state: int = 42,
    ):
        """Initialize FeatureAuditor.

        Args:
            feature_names: List of active feature names.
            cluster_map: Mapping of feature column name to cluster name.
            collinearity_threshold: Absolute Pearson correlation threshold to flag redundancy (|r| >= threshold).
            n_repeats: Number of permutation iterations per feature.
            random_state: Random seed for permutations.
        """
        self.feature_names = list(feature_names)
        self.cluster_map = cluster_map or {}
        self.collinearity_threshold = collinearity_threshold
        self.n_repeats = n_repeats
        self.random_state = random_state

    def audit(
        self,
        estimator: Any,
        X_val: pd.DataFrame,
        y_val: pd.Series,
        model_name: str = "Model",
    ) -> FeatureAuditReport:
        """Run comprehensive feature audit across validation data.

        Args:
            estimator: Fitted scikit-learn compatible estimator or forecaster with a .predict() method.
            X_val: Out-of-sample validation feature matrix.
            y_val: Realized target series.
            model_name: Name of the evaluated model.

        Returns:
            FeatureAuditReport containing prune candidates and collinearity alerts.
        """
        X_clean = X_val[self.feature_names].dropna()
        common_idx = X_clean.index.intersection(y_val.index)
        X_eval = X_clean.loc[common_idx]
        y_eval = y_val.loc[common_idx]

        if len(X_eval) < 5:
            logger.warning(f"Insufficient validation samples ({len(X_eval)}) for feature audit.")
            return FeatureAuditReport(
                model_name=model_name,
                evaluated_sessions=len(X_eval),
                total_features=len(self.feature_names),
            )

        # 1. Collinearity Screening
        collinear_pairs = self._detect_collinearity(X_eval)

        # 2. Permutation Importance
        perm_results = self._compute_permutation_importance(estimator, X_eval, y_eval)

        # 3. ModelExplainer for mean absolute SHAP / attribution
        explainer = ModelExplainer(
            model=estimator,
            feature_names=self.feature_names,
            cluster_map=self.cluster_map,
            background_data=X_eval,
            model_name=model_name,
        )
        global_exp = explainer.explain_global(X_eval)
        shap_importances = dict(global_exp.top_features)

        # 4. Synthesize Prune Candidates
        prune_candidates: List[Dict[str, Any]] = []
        top_drivers: List[Dict[str, Any]] = []

        for feat in self.feature_names:
            perm_drop = perm_results.get(feat, 0.0)
            cluster = self.cluster_map.get(feat, "other")

            # Check if this feature is a redundant collinear partner
            is_redundant_collinear = any(
                p["feature_b"] == feat for p in collinear_pairs
            )

            # Prune candidate condition: Shuffling does not degrade error (drop <= 0) or redundant collinear
            if perm_drop <= 0.0 or is_redundant_collinear:
                reason = "Negative/Zero out-of-sample alpha contribution"
                if is_redundant_collinear:
                    matched = [p["feature_a"] for p in collinear_pairs if p["feature_b"] == feat]
                    reason = f"Collinear redundant with {matched[0]} (r >= {self.collinearity_threshold})"

                prune_candidates.append(
                    {
                        "feature_name": feat,
                        "cluster_name": cluster,
                        "permutation_score_drop": float(perm_drop),
                        "reason": reason,
                    }
                )
            else:
                top_drivers.append(
                    {
                        "feature_name": feat,
                        "cluster_name": cluster,
                        "permutation_score_drop": float(perm_drop),
                    }
                )

        # Sort top drivers descending
        top_drivers = sorted(top_drivers, key=lambda x: x["permutation_score_drop"], reverse=True)

        # 5. Generate YAML Snippet
        yaml_snippet = self._generate_yaml_snippet(model_name, prune_candidates)

        return FeatureAuditReport(
            model_name=model_name,
            evaluated_sessions=len(X_eval),
            total_features=len(self.feature_names),
            prune_candidates=prune_candidates,
            collinear_pairs=collinear_pairs,
            top_drivers=top_drivers[:10],
            recommended_features_yaml=yaml_snippet,
        )

    def _detect_collinearity(self, X: pd.DataFrame) -> List[Dict[str, Any]]:
        """Identify pairs of features exceeding collinearity threshold."""
        corr_matrix = X[self.feature_names].corr().abs()
        pairs = []

        for i in range(len(self.feature_names)):
            for j in range(i + 1, len(self.feature_names)):
                f1 = self.feature_names[i]
                f2 = self.feature_names[j]
                r_val = float(corr_matrix.loc[f1, f2])

                if r_val >= self.collinearity_threshold and not np.isnan(r_val):
                    pairs.append(
                        {
                            "feature_a": f1,
                            "feature_b": f2,
                            "correlation": round(r_val, 4),
                            "cluster_a": self.cluster_map.get(f1, "other"),
                            "cluster_b": self.cluster_map.get(f2, "other"),
                        }
                    )

        return sorted(pairs, key=lambda x: x["correlation"], reverse=True)

    def _predict_numeric(self, est: Any, X: pd.DataFrame) -> np.ndarray:
        """Extract numeric 1D numpy array of predictions regardless of forecaster wrapper."""
        raw_est = est
        if hasattr(est, "regressor") and getattr(est, "regressor") is not None:
            raw_est = getattr(est, "regressor")
        elif hasattr(est, "model") and getattr(est, "model") is not None:
            raw_est = getattr(est, "model")
        elif hasattr(est, "fitted_model_") and getattr(est, "fitted_model_") is not None:
            raw_est = getattr(est, "fitted_model_")

        try:
            res = raw_est.predict(X[self.feature_names])
            if isinstance(res, tuple):
                res = res[0]  # e.g., BayesianRidge return_std=True
            if isinstance(res, np.ndarray):
                return np.asarray(res).flatten()
            if hasattr(res, "predicted_net_flow_tl"):
                return np.array([res.predicted_net_flow_tl])
            if hasattr(res, "predicted_return_pct"):
                return np.array([res.predicted_return_pct])
        except Exception:
            pass

        # Fallback: row-by-row predict if forecaster wrapper
        vals = []
        for i in range(len(X)):
            row = X.iloc[[i]]
            pred = est.predict(row)
            if hasattr(pred, "predicted_net_flow_tl"):
                vals.append(pred.predicted_net_flow_tl)
            elif hasattr(pred, "predicted_return_pct"):
                vals.append(pred.predicted_return_pct)
            elif isinstance(pred, (int, float)):
                vals.append(float(pred))
            elif isinstance(pred, (np.ndarray, list)):
                vals.append(float(pred[0]))
            else:
                vals.append(0.0)
        return np.array(vals)

    def _compute_permutation_importance(
        self, estimator: Any, X: pd.DataFrame, y: pd.Series
    ) -> Dict[str, float]:
        """Compute out-of-sample RMSE degradation when each feature is randomly permuted."""
        # Unwrap underlying estimator if forecaster
        est = getattr(estimator, "regressor", getattr(estimator, "model", estimator))
        if hasattr(est, "fitted_model_"):
            est = est.fitted_model_

        try:
            r = permutation_importance(
                est,
                X[self.feature_names],
                y,
                scoring="neg_root_mean_squared_error",
                n_repeats=self.n_repeats,
                random_state=self.random_state,
            )
            return {
                feat: float(drop)
                for feat, drop in zip(self.feature_names, r.importances_mean)
            }
        except Exception as e:
            logger.debug(f"sklearn permutation_importance skipped ({e}); running optimized permutation loop.")
            return self._manual_permutation(estimator, X, y)

    def _manual_permutation(
        self, est: Any, X: pd.DataFrame, y: pd.Series
    ) -> Dict[str, float]:
        """Manual permutation fallback for custom forecasters."""
        baseline_preds = self._predict_numeric(est, X)
        baseline_rmse = np.sqrt(mean_squared_error(y, baseline_preds))

        drops = {}
        rng = np.random.default_rng(self.random_state)

        for feat in self.feature_names:
            X_perm = X[self.feature_names].copy()
            X_perm[feat] = rng.permutation(X_perm[feat].values)
            perm_preds = self._predict_numeric(est, X_perm)
            perm_rmse = np.sqrt(mean_squared_error(y, perm_preds))
            # Positive drop means permuted RMSE is worse (higher) -> feature was helpful
            drops[feat] = float(perm_rmse - baseline_rmse)

        return drops

    def _generate_yaml_snippet(
        self, model_name: str, prune_candidates: List[Dict[str, Any]]
    ) -> str:
        """Format recommended exclude_features snippet for config/features.yaml."""
        if not prune_candidates:
            return f"# Model: {model_name}\n# No features recommended for exclusion. Feature set is optimal."

        lines = [
            f"# Recommended updates for config/features.yaml ({model_name}):",
            f"{model_name}:",
            "  exclude_features:",
        ]
        for c in prune_candidates:
            lines.append(f"    - {c['feature_name']}  # {c['reason']}")

        return "\n".join(lines)
