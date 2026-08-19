"""Production Day-Start Forecaster Orchestrator."""

from datetime import date
from typing import List, Optional

from mdk_trading_oracle.core.db import DuckDBManager
from mdk_trading_oracle.core.logger import get_logger
from mdk_trading_oracle.models.base import BaseForecaster, ForecastResult
from mdk_trading_oracle.models.day_start.features import DayStartFeatureExtractor
from mdk_trading_oracle.models.day_start.models import (
    DayStartBayesianModel,
    DayStartLightGBMModel,
    DayStartNaivePersistenceModel,
)
from mdk_trading_oracle.models.registry import ModelRegistry

logger = get_logger("mdk_oracle.models.day_start.forecaster")


@ModelRegistry.register("day_start_forecaster")
class DayStartForecaster:
    """Production Forecaster for Model 1: 'How Will BofA Start the Day?'
    
    Orchestrates end-to-end:
        1. Feature extraction across all 7 Feature Clusters from DuckDB Silver tables
        2. Probabilistic model training and cross-validation
        3. Backtesting and generating daily forecasts
        4. Persisting forecasts directly into DuckDB Gold layer tables
    """

    def __init__(self, db: Optional[DuckDBManager] = None, model_type: str = "bayesian"):
        self.db = db or DuckDBManager()
        self.target_broker = "MLB"
        self.feature_extractor = DayStartFeatureExtractor(self.db, target_broker_id=self.target_broker)
        
        if model_type == "lightgbm":
            self.model: BaseForecaster = DayStartLightGBMModel()
        elif model_type == "baseline":
            self.model = DayStartNaivePersistenceModel()
        else:
            self.model = DayStartBayesianModel()

    def train_and_forecast_all(self) -> List[ForecastResult]:
        """Extract features, fit model on historical data, and generate forecasts for all sessions."""
        df_pl = self.feature_extractor.extract_features()
        if df_pl.height == 0:
            logger.warning("No feature records found to train DayStartForecaster.")
            return []

        df_pd = df_pl.to_pandas()
        X = df_pd.drop(columns=["target_open_net_flow_tl", "target_open_direction"], errors="ignore")
        y = df_pd["target_open_net_flow_tl"]

        # Train model
        logger.info(f"Training {self.model.model_name} on {len(df_pd)} historical daily sessions...")
        self.model.fit(X, y)

        # Generate walk-forward / backtest predictions for each session
        results: List[ForecastResult] = []
        for idx in range(len(df_pd)):
            row = X.iloc[[idx]]
            res = self.model.predict(row)
            
            # Predict top buy/sell sector based on sector features
            banking_flow = float(row["feat_bofa_banking_flow_prev_day"].iloc[0]) if "feat_bofa_banking_flow_prev_day" in row.columns else 0.0
            transport_flow = float(row["feat_bofa_transport_flow_prev_day"].iloc[0]) if "feat_bofa_transport_flow_prev_day" in row.columns else 0.0
            res.top_predicted_buy_sector = "Banking" if banking_flow > transport_flow else "Transportation"
            res.top_predicted_sell_sector = "Holding" if res.predicted_net_flow_tl > 0 else "Energy & Refining"
            
            results.append(res)

        logger.info(f"Generated {len(results)} day-start forecasts.")
        return results

    def save_forecasts_to_gold(self, forecasts: List[ForecastResult]) -> int:
        """Persist generated forecasts into DuckDB Gold tables (`gold_bofa_day_start_forecasts`)."""
        if not forecasts:
            return 0

        conn = self.db.get_connection()
        logger.info(f"Persisting {len(forecasts)} forecasts to `gold_bofa_day_start_forecasts`...")

        # Ensure schema exists
        conn.execute("""
            CREATE TABLE IF NOT EXISTS gold_bofa_day_start_forecasts (
                forecast_date DATE PRIMARY KEY,
                day_of_week INTEGER,
                is_monday BOOLEAN,
                predicted_open_net_flow_tl DOUBLE,
                predicted_open_flow_lower_90 DOUBLE,
                predicted_open_flow_upper_90 DOUBLE,
                predicted_direction VARCHAR,
                direction_confidence DOUBLE,
                predicted_open_market_share DOUBLE,
                predicted_playbook VARCHAR,
                top_predicted_buy_sector VARCHAR,
                top_predicted_sell_sector VARCHAR,
                model_name VARCHAR,
                model_version VARCHAR,
                generated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            );
        """)

        # Insert or replace predictions
        for f in forecasts:
            d = f.forecast_date
            dow = d.weekday() + 1 if isinstance(d, date) else 1
            is_mon = (dow == 1)
            conn.execute("""
                INSERT OR REPLACE INTO gold_bofa_day_start_forecasts (
                    forecast_date, day_of_week, is_monday,
                    predicted_open_net_flow_tl, predicted_open_flow_lower_90, predicted_open_flow_upper_90,
                    predicted_direction, direction_confidence, predicted_open_market_share,
                    predicted_playbook, top_predicted_buy_sector, top_predicted_sell_sector,
                    model_name, model_version, generated_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?);
            """, [
                f.forecast_date,
                dow,
                is_mon,
                f.predicted_net_flow_tl,
                f.predicted_flow_lower_90,
                f.predicted_flow_upper_90,
                f.predicted_direction,
                f.direction_confidence,
                f.predicted_open_market_share,
                f.predicted_playbook,
                f.top_predicted_buy_sector,
                f.top_predicted_sell_sector,
                f.model_name,
                f.model_version,
                f.generated_at,
            ])

        saved_count = conn.execute("SELECT COUNT(*) FROM gold_bofa_day_start_forecasts;").fetchone()[0]
        logger.info(f"Successfully updated `gold_bofa_day_start_forecasts`: {saved_count:,} total forecasts.")
        return saved_count
