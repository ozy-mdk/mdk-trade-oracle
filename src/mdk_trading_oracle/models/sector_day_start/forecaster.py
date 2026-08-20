"""Production Sector Day-Start Forecaster Orchestrator & Auto-Champion Model Arena."""

from datetime import date
from typing import Dict, List, Optional, Tuple

import pandas as pd

from mdk_trading_oracle.core.db import DuckDBManager
from mdk_trading_oracle.core.logger import get_logger
from mdk_trading_oracle.models.base import ForecastResult
from mdk_trading_oracle.models.registry import ModelRegistry
from mdk_trading_oracle.models.sector_day_start.features import SectorDayStartFeatureExtractor
from mdk_trading_oracle.models.sector_day_start.models import (
    BaseSectorDayStartModel,
    SectorDayStartBayesianModel,
    SectorDayStartLightGBMModel,
    SectorDayStartNaivePersistenceModel,
    SectorDayStartPyMCModel,
    SectorDayStartRollingMeanModel,
)

logger = get_logger("mdk_oracle.models.sector_day_start.forecaster")


@ModelRegistry.register("sector_day_start_model_arena")
class SectorDayStartModelArena:
    """Evaluates candidate sector models using expanding-window walk-forward validation and crowns the champion."""

    def __init__(self, include_pymc: bool = False):
        self.candidates: Dict[str, BaseSectorDayStartModel] = {
            "Baseline 0: Naive W4 Sector Persistence": SectorDayStartNaivePersistenceModel(),
            "Baseline 1: 5-Day Historical Sector Mean": SectorDayStartRollingMeanModel(),
            "LightGBM Non-Linear Sector Ensemble": SectorDayStartLightGBMModel(),
            "Bayesian Ridge Probabilistic": SectorDayStartBayesianModel(),
        }
        if include_pymc:
            self.candidates["PyMC Bayesian GLM (MAP)"] = SectorDayStartPyMCModel(use_map=True)

    def run_tournament(
        self, X: pd.DataFrame, y: pd.Series, min_train_samples: int = 5
    ) -> Tuple[pd.DataFrame, BaseSectorDayStartModel]:
        """Execute walk-forward out-of-sample tournament across all candidate sector models."""
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
        champion_model: BaseSectorDayStartModel = champion_row["_model_instance"]
        champion_name = str(champion_row["Model"])

        logger.info(
            f"🏆 Sector Model Arena Champion Crowned: '{champion_name}' "
            f"(Out-of-Sample Hit Rate: {champion_row['hit_rate_pct']:.1f}%, "
            f"90% PICP: {champion_row['picp_90_pct']:.1f}%, "
            f"RMSE: {champion_row['rmse_million_tl']:.2f}M TL)"
        )

        display_df = df_scores.drop(columns=["_model_instance"])
        return display_df, champion_model


@ModelRegistry.register("sector_day_start_forecaster")
class SectorDayStartForecaster:
    """Production Forecaster for Model 2: 'How Will BofA Allocate Across Sectors at the Open?'
    
    Orchestrates end-to-end:
        1. Multi-sector feature extraction from DuckDB Silver tables at T-1 Close
        2. Automated Model Arena tournament selection across sectors
        3. Probabilistic model training and walk-forward validation
        4. Persisting forecasts directly into DuckDB Gold table `gold_bofa_sector_day_start_forecasts`
    """

    def __init__(self, db: Optional[DuckDBManager] = None, model_type: str = "auto", include_pymc_arena: bool = False):
        self.db = db or DuckDBManager()
        self.target_broker = "MLB"
        self.feature_extractor = SectorDayStartFeatureExtractor(self.db, target_broker_id=self.target_broker)
        self.model_type = model_type
        self.arena = SectorDayStartModelArena(include_pymc=include_pymc_arena)
        self.champion_name: Optional[str] = None

    def train_and_forecast_all(self, sectors: Optional[List[str]] = None) -> List[ForecastResult]:
        """Extract sector features, run arena to select champion, and generate forecasts for each sector."""
        tracked_sectors = sectors or self.feature_extractor.get_tracked_sectors(min_session_count=10)
        if not tracked_sectors:
            logger.warning("No tracked sectors found for SectorDayStartForecaster.")
            return []

        # First, if auto, crown champion on the primary benchmark sector (Banking)
        if self.model_type == "auto":
            benchmark_sector = "Banking" if "Banking" in tracked_sectors else tracked_sectors[0]
            df_bm = self.feature_extractor.extract_features(sector=benchmark_sector).to_pandas()
            if len(df_bm) > 5:
                X_bm = df_bm.drop(columns=["target_sector_open_net_flow_tl", "target_sector_open_direction"], errors="ignore")
                y_bm = df_bm["target_sector_open_net_flow_tl"]
                min_burn_in = min(5, max(2, len(df_bm) - 1))
                _, champion_model = self.arena.run_tournament(X_bm, y_bm, min_train_samples=min_burn_in)
                self.champion_name = champion_model.model_name
            else:
                self.champion_name = "sector_day_start_bayesian_ridge"
        else:
            self.champion_name = self.model_type

        all_forecasts: List[ForecastResult] = []

        for sector in tracked_sectors:
            df_pl = self.feature_extractor.extract_features(sector=sector)
            if df_pl.height == 0:
                continue

            df_pd = df_pl.to_pandas()
            X = df_pd.drop(columns=["target_sector_open_net_flow_tl", "target_sector_open_direction"], errors="ignore")
            y = df_pd["target_sector_open_net_flow_tl"]

            # Instantiate model for this sector
            if self.champion_name in ["sector_day_start_bayesian_ridge", "bayesian"]:
                model: BaseSectorDayStartModel = SectorDayStartBayesianModel()
            elif self.champion_name in ["sector_day_start_lightgbm", "lightgbm"]:
                model = SectorDayStartLightGBMModel()
            elif self.champion_name in ["sector_day_start_pymc", "pymc"]:
                model = SectorDayStartPyMCModel(use_map=True)
            elif self.champion_name in ["sector_day_start_rolling_mean", "rolling_mean"]:
                model = SectorDayStartRollingMeanModel()
            else:
                model = SectorDayStartNaivePersistenceModel()

            # Train on full history
            model.fit(X, y)

            # Generate forecasts
            for idx in range(len(df_pd)):
                row = X.iloc[[idx]]
                res = model.predict(row)
                res.top_predicted_buy_sector = sector
                all_forecasts.append(res)

        logger.info(f"Generated {len(all_forecasts)} sector day-start forecasts across {len(tracked_sectors)} sectors using Champion '{self.champion_name}'.")
        return all_forecasts

    def save_forecasts_to_gold(self, forecasts: List[ForecastResult]) -> int:
        """Persist sector forecasts into DuckDB Gold table `gold_bofa_sector_day_start_forecasts`."""
        if not forecasts:
            return 0

        conn = self.db.get_connection()
        logger.info(f"Persisting {len(forecasts)} forecasts to `gold_bofa_sector_day_start_forecasts`...")

        # Ensure schema exists and has all current columns
        existing_tables = [r[0] for r in conn.execute("SHOW TABLES;").fetchall()]
        if "gold_bofa_sector_day_start_forecasts" in existing_tables:
            cols = [r[1] for r in conn.execute("PRAGMA table_info('gold_bofa_sector_day_start_forecasts');").fetchall()]
            if "day_of_week" not in cols:
                conn.execute("DROP TABLE gold_bofa_sector_day_start_forecasts;")

        conn.execute("""
            CREATE TABLE IF NOT EXISTS gold_bofa_sector_day_start_forecasts (
                forecast_date DATE,
                sector VARCHAR,
                day_of_week INTEGER,
                is_monday BOOLEAN,
                predicted_open_net_flow_tl DOUBLE,
                predicted_open_flow_lower_90 DOUBLE,
                predicted_open_flow_upper_90 DOUBLE,
                predicted_direction VARCHAR,
                direction_confidence DOUBLE,
                predicted_playbook VARCHAR,
                model_name VARCHAR,
                model_version VARCHAR,
                generated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                PRIMARY KEY (forecast_date, sector)
            );
        """)

        # Insert or replace predictions
        for f in forecasts:
            d = f.forecast_date
            dow = d.weekday() + 1 if isinstance(d, date) else 1
            is_mon = (dow == 1)
            sector_name = f.top_predicted_buy_sector or "General"
            conn.execute("""
                INSERT OR REPLACE INTO gold_bofa_sector_day_start_forecasts (
                    forecast_date, sector, day_of_week, is_monday,
                    predicted_open_net_flow_tl, predicted_open_flow_lower_90, predicted_open_flow_upper_90,
                    predicted_direction, direction_confidence,
                    predicted_playbook, model_name, model_version, generated_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?);
            """, [
                f.forecast_date,
                sector_name,
                dow,
                is_mon,
                f.predicted_net_flow_tl,
                f.predicted_flow_lower_90,
                f.predicted_flow_upper_90,
                f.predicted_direction,
                f.direction_confidence,
                f.predicted_playbook,
                f.model_name,
                f.model_version,
                f.generated_at,
            ])

        saved_count = conn.execute("SELECT COUNT(*) FROM gold_bofa_sector_day_start_forecasts;").fetchone()[0]
        logger.info(f"Successfully updated `gold_bofa_sector_day_start_forecasts`: {saved_count:,} total forecasts.")
        return saved_count
