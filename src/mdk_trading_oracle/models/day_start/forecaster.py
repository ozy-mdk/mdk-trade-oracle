"""Production Day-Start Forecaster Orchestrator & Auto-Champion Model Arena."""

from datetime import date
from typing import Dict, List, Optional, Tuple

import pandas as pd

from mdk_trading_oracle.core.db import DuckDBManager
from mdk_trading_oracle.core.logger import get_logger
from mdk_trading_oracle.models.base import BaseForecaster, ForecastResult
from mdk_trading_oracle.models.day_start.features import DayStartFeatureExtractor
from mdk_trading_oracle.models.day_start.models import (
    DayStartBayesianModel,
    DayStartLightGBMModel,
    DayStartNaivePersistenceModel,
    DayStartPyMCModel,
    DayStartRollingMeanModel,
)
from mdk_trading_oracle.models.registry import ModelRegistry

logger = get_logger("mdk_oracle.models.day_start.forecaster")


@ModelRegistry.register("day_start_model_arena")
class DayStartModelArena:
    """Evaluates all candidate models using expanding-window walk-forward validation and crowns the champion."""

    def __init__(self):
        self.candidates: Dict[str, BaseForecaster] = {
            "Baseline 0: Naive W4 Persistence": DayStartNaivePersistenceModel(),
            "Baseline 1: 5-Day Historical Mean": DayStartRollingMeanModel(),
            "LightGBM Non-Linear Ensemble": DayStartLightGBMModel(),
            "Bayesian Ridge Probabilistic": DayStartBayesianModel(),
            "PyMC Bayesian GLM (MAP)": DayStartPyMCModel(use_map=True),
        }

    def run_tournament(
        self, X: pd.DataFrame, y: pd.Series, min_train_samples: int = 5
    ) -> Tuple[pd.DataFrame, BaseForecaster]:
        """Execute walk-forward out-of-sample tournament across all candidate models."""
        scoreboard = []
        for name, model in self.candidates.items():
            metrics = model.walk_forward_evaluate(X, y, min_train_samples=min_train_samples)
            scoreboard.append({
                "Model": name,
                "hit_rate_pct": metrics["hit_rate_pct"],
                "picp_90_pct": metrics["picp_90_pct"],
                "mae_million_tl": metrics["mae_million_tl"],
                "rmse_million_tl": metrics["rmse_million_tl"],
                "sample_size": metrics["sample_size"],
                "_model_instance": model,
            })

        df_scores = pd.DataFrame(scoreboard).sort_values(
            by=["hit_rate_pct", "picp_90_pct", "rmse_million_tl"],
            ascending=[False, False, True],
        ).reset_index(drop=True)

        champion_row = df_scores.iloc[0]
        champion_model: BaseForecaster = champion_row["_model_instance"]
        champion_name = str(champion_row["Model"])

        logger.info(
            f"🏆 Model Arena Champion Crowned: '{champion_name}' "
            f"(Out-of-Sample Hit Rate: {champion_row['hit_rate_pct']:.1f}%, "
            f"90% PICP: {champion_row['picp_90_pct']:.1f}%, "
            f"RMSE: {champion_row['rmse_million_tl']:.2f}M TL)"
        )

        display_df = df_scores.drop(columns=["_model_instance"])
        return display_df, champion_model


@ModelRegistry.register("day_start_forecaster")
class DayStartForecaster:
    """Production Forecaster for Model 1: 'How Will BofA Start the Day?'
    
    Orchestrates end-to-end:
        1. Feature extraction across all 7 Feature Clusters from DuckDB Silver tables
        2. Automated Model Arena tournament selection or configured model type
        3. Probabilistic model training and walk-forward validation
        4. Persisting forecasts directly into DuckDB Gold layer tables
    """

    def __init__(self, db: Optional[DuckDBManager] = None, model_type: str = "auto"):
        self.db = db or DuckDBManager()
        self.target_broker = "MLB"
        self.feature_extractor = DayStartFeatureExtractor(self.db, target_broker_id=self.target_broker)
        self.model_type = model_type
        self.arena = DayStartModelArena()
        self.champion_name: Optional[str] = None
        
        if model_type == "lightgbm":
            self.model: Optional[BaseForecaster] = DayStartLightGBMModel()
        elif model_type == "baseline":
            self.model = DayStartNaivePersistenceModel()
        elif model_type == "pymc":
            self.model = DayStartPyMCModel(use_map=True)
        elif model_type == "bayesian":
            self.model = DayStartBayesianModel()
        else:  # "auto"
            self.model = None

    def train_and_forecast_all(self) -> List[ForecastResult]:
        """Extract features, select/fit champion model on historical data, and generate forecasts."""
        df_pl = self.feature_extractor.extract_features()
        if df_pl.height == 0:
            logger.warning("No feature records found to train DayStartForecaster.")
            return []

        df_pd = df_pl.to_pandas()
        X = df_pd.drop(columns=["target_open_net_flow_tl", "target_open_direction"], errors="ignore")
        y = df_pd["target_open_net_flow_tl"]

        # Automatic Champion Selection on the fly if model_type == "auto"
        if self.model_type == "auto" or self.model is None:
            logger.info("Running DayStartModelArena Walk-Forward Tournament for Auto-Selection...")
            min_burn_in = min(5, max(2, len(df_pd) - 1))
            _, champion_model = self.arena.run_tournament(X, y, min_train_samples=min_burn_in)
            self.model = champion_model
            self.champion_name = champion_model.model_name

        # Train champion model on full historical dataset
        logger.info(f"Fitting Champion Model '{self.model.model_name}' on {len(df_pd)} historical daily sessions...")
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

        logger.info(f"Generated {len(results)} day-start forecasts using Champion '{self.model.model_name}'.")
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
