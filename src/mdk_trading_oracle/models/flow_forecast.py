"""Predictive model for institutional flow direction based on historical patterns."""

from typing import Dict, List, Optional, Tuple
import numpy as np
import polars as pl
from sklearn.ensemble import GradientBoostingClassifier
from mdk_trading_oracle.core.db import DuckDBManager
from mdk_trading_oracle.core.logger import get_logger

logger = get_logger("mdk_oracle.models")


class InstitutionalFlowModel:
    """Predictive model trained on Gold flow metrics to forecast short-term momentum."""

    def __init__(self, db_manager: DuckDBManager):
        self.db = db_manager
        self.model = GradientBoostingClassifier(
            n_estimators=50,
            learning_rate=0.05,
            max_depth=3,
            random_state=42,
        )
        self.is_trained: bool = False
        self.feature_columns: List[str] = [
            "bofa_volume_share",
            "bofa_net_share",
            "bofa_net_tl_roll_3d",
            "bofa_net_tl_roll_5d",
            "bofa_flow_acceleration_5d",
            "bofa_flow_zscore_20d",
        ]

    def prepare_dataset(self) -> Tuple[np.ndarray, np.ndarray]:
        """Prepare X, y dataset from Gold table for direction prediction."""
        query = """
            WITH target_calc AS (
                SELECT 
                    *,
                    LEAD(close_price, 1) OVER (PARTITION BY symbol ORDER BY date_val) - close_price AS next_day_diff
                FROM gold_bofa_flow_metrics
            )
            SELECT *
            FROM target_calc
            WHERE next_day_diff IS NOT NULL;
        """
        df = self.db.query_pl(query)
        if df.is_empty():
            return np.empty((0, len(self.feature_columns))), np.empty((0,))

        # Fill nulls with 0
        df_clean = df.fill_nan(0.0).fill_null(0.0)
        X = df_clean.select(self.feature_columns).to_numpy()
        # Binary target: 1 if next day up, 0 if down
        y = (df_clean["next_day_diff"].to_numpy() > 0).astype(int)
        return X, y

    def train(self) -> Dict[str, float]:
        """Train classifier on existing Gold metrics."""
        X, y = self.prepare_dataset()
        if len(X) < 10:
            logger.warning("Not enough samples in Gold layer to train model (< 10 samples).")
            return {"accuracy": 0.0, "samples": len(X)}

        self.model.fit(X, y)
        self.is_trained = True
        accuracy = float(self.model.score(X, y))
        logger.info(f"Model trained on {len(X)} samples with training accuracy: {accuracy:.2%}")
        return {"accuracy": accuracy, "samples": len(X)}

    def predict_probability(self, features: Dict[str, float]) -> float:
        """Predict probability of upward price continuation given flow metrics."""
        if not self.is_trained:
            # Neutral baseline prior
            return 0.50

        x_vec = np.array([[features.get(c, 0.0) for c in self.feature_columns]])
        probs = self.model.predict_proba(x_vec)
        # Probability of class 1 (UP)
        return float(probs[0][1])
