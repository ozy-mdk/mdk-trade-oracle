"""Production Sector Day-Start Forecaster Orchestrator & Auto-Champion Model Arena."""

from datetime import date
from typing import Dict, List, Optional, Tuple, Union

import pandas as pd

from mdk_trading_oracle.core.config import get_settings
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
        self,
        X: pd.DataFrame,
        y: pd.Series,
        min_train_samples: int = 5,
        eval_window_days: Optional[int] = None,
    ) -> Tuple[pd.DataFrame, BaseSectorDayStartModel]:
        """Execute walk-forward out-of-sample tournament across all candidate sector models."""
        scoreboard = []
        for name, model in self.candidates.items():
            metrics = model.walk_forward_evaluate(
                X, y, min_train_samples=min_train_samples, eval_window_days=eval_window_days
            )
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
        4. Live next-day sector forecasting (T+1) and historical backtest evaluation
        5. Persisting forecasts directly into DuckDB Gold table `gold_bofa_sector_day_start_forecasts`
    """

    def __init__(
        self,
        db: Optional[DuckDBManager] = None,
        model_type: Optional[str] = None,
        lookback_months: Optional[int] = None,
        eval_window_days: Optional[int] = None,
        min_burn_in_days: Optional[int] = None,
        include_pymc_arena: Optional[bool] = None,
    ):
        self.db = db or DuckDBManager()
        self.target_broker = "MLB"
        self.settings = get_settings()
        cfg = self.settings.get_model_config("sector_day_start")

        self.lookback_months = lookback_months if lookback_months is not None else cfg.get("lookback_months", 12)
        self.eval_window_days = eval_window_days if eval_window_days is not None else cfg.get("eval_window_days", 20)
        self.min_burn_in_days = min_burn_in_days if min_burn_in_days is not None else cfg.get("min_burn_in_days", 5)
        self.model_type = model_type or cfg.get("model_type", "auto")
        include_pymc = include_pymc_arena if include_pymc_arena is not None else cfg.get("include_pymc_arena", False)

        self.feature_extractor = SectorDayStartFeatureExtractor(
            self.db, target_broker_id=self.target_broker, lookback_months=self.lookback_months
        )
        self.arena = SectorDayStartModelArena(include_pymc=include_pymc)
        self.champion_name: Optional[str] = None

    def _ensure_champion_selected(self, tracked_sectors: List[str]) -> str:
        """Select champion model dynamically on the fly if model_type == 'auto'."""
        if self.model_type == "auto" or self.champion_name is None:
            benchmark_sector = "Banking" if "Banking" in tracked_sectors else tracked_sectors[0]
            df_bm = self.feature_extractor.extract_features(sector=benchmark_sector).to_pandas()
            if len(df_bm) > 5:
                X_bm = df_bm.drop(columns=["target_sector_open_net_flow_tl", "target_sector_open_direction"], errors="ignore")
                y_bm = df_bm["target_sector_open_net_flow_tl"]
                min_burn_in = min(self.min_burn_in_days, max(2, len(df_bm) - 1))
                _, champion_model = self.arena.run_tournament(
                    X_bm, y_bm, min_train_samples=min_burn_in, eval_window_days=self.eval_window_days
                )
                self.champion_name = champion_model.model_name
            else:
                self.champion_name = "sector_day_start_bayesian_ridge"
        else:
            self.champion_name = self.model_type
        return self.champion_name

    def _create_sector_model(self) -> BaseSectorDayStartModel:
        """Instantiate a fresh model instance of the crowned champion paradigm."""
        if self.champion_name in ["sector_day_start_bayesian_ridge", "bayesian"]:
            return SectorDayStartBayesianModel()
        elif self.champion_name in ["sector_day_start_lightgbm", "lightgbm"]:
            return SectorDayStartLightGBMModel()
        elif self.champion_name in ["sector_day_start_pymc", "pymc"]:
            return SectorDayStartPyMCModel(use_map=True)
        elif self.champion_name in ["sector_day_start_rolling_mean", "rolling_mean"]:
            return SectorDayStartRollingMeanModel()
        else:
            return SectorDayStartNaivePersistenceModel()

    def forecast_next_day(self, sectors: Optional[List[str]] = None) -> List[ForecastResult]:
        """Generate live sector forecasts for the upcoming trading session (T_next)."""
        tracked_sectors = sectors or self.feature_extractor.get_tracked_sectors(min_session_count=10)
        if not tracked_sectors:
            logger.warning("No tracked sectors found for SectorDayStartForecaster.")
            return []

        self._ensure_champion_selected(tracked_sectors)

        # Extract next-day feature rows across tracked sectors
        df_next_pl = self.feature_extractor.extract_next_day_features(sectors=tracked_sectors)
        if df_next_pl.height == 0:
            logger.warning("No next-day sector feature records extracted.")
            return []

        df_next_pd = df_next_pl.to_pandas()
        live_forecasts: List[ForecastResult] = []

        for sector in tracked_sectors:
            sec_next_row = df_next_pd[df_next_pd["sector"] == sector]
            if sec_next_row.empty:
                continue

            # Train on historical sector data
            df_hist_pl = self.feature_extractor.extract_features(sector=sector)
            if df_hist_pl.height == 0:
                continue

            df_hist_pd = df_hist_pl.to_pandas()
            X_hist = df_hist_pd.drop(columns=["target_sector_open_net_flow_tl", "target_sector_open_direction"], errors="ignore")
            y_hist = df_hist_pd["target_sector_open_net_flow_tl"]

            model = self._create_sector_model()
            model.fit(X_hist, y_hist)

            res = model.predict(sec_next_row)
            res.top_predicted_buy_sector = sector
            live_forecasts.append(res)

        logger.info(
            f"🎯 Generated {len(live_forecasts)} Live Next-Day Sector Forecasts across sectors "
            f"for {live_forecasts[0].forecast_date if live_forecasts else 'N/A'} (Champion: '{self.champion_name}')."
        )
        return live_forecasts

    def backtest_all_history(self, sectors: Optional[List[str]] = None) -> List[ForecastResult]:
        """Generate historical in-sample / backtest predictions across all historical training sessions."""
        tracked_sectors = sectors or self.feature_extractor.get_tracked_sectors(min_session_count=10)
        if not tracked_sectors:
            return []

        self._ensure_champion_selected(tracked_sectors)
        all_backtests: List[ForecastResult] = []

        for sector in tracked_sectors:
            df_pl = self.feature_extractor.extract_features(sector=sector)
            if df_pl.height == 0:
                continue

            df_pd = df_pl.to_pandas()
            X = df_pd.drop(columns=["target_sector_open_net_flow_tl", "target_sector_open_direction"], errors="ignore")
            y = df_pd["target_sector_open_net_flow_tl"]

            model = self._create_sector_model()
            model.fit(X, y)

            for idx in range(len(df_pd)):
                row = X.iloc[[idx]]
                res = model.predict(row)
                res.top_predicted_buy_sector = sector
                all_backtests.append(res)

        logger.info(f"Generated {len(all_backtests)} historical backtest sector forecasts across {len(tracked_sectors)} sectors.")
        return all_backtests

    def train_and_forecast_all(
        self,
        sectors: Optional[List[str]] = None,
        include_history: bool = False,
        include_next_day: bool = True,
    ) -> List[ForecastResult]:
        """Extract sector features, fit champion model, and generate forecasts.
        
        Args:
            sectors: Optional list of sectors to forecast.
            include_history: If True, includes historical backtest forecasts.
            include_next_day: If True (default), includes the live forecast for upcoming session T_next.
        """
        results: List[ForecastResult] = []
        if include_history:
            results.extend(self.backtest_all_history(sectors=sectors))
        if include_next_day:
            results.extend(self.forecast_next_day(sectors=sectors))
        return results

    def save_forecasts_to_gold(self, forecasts: Union[ForecastResult, List[ForecastResult]]) -> int:
        """Persist sector forecasts into DuckDB Gold table `gold_bofa_sector_day_start_forecasts`."""
        if isinstance(forecasts, ForecastResult):
            forecast_list = [forecasts]
        else:
            forecast_list = forecasts

        if not forecast_list:
            return 0

        conn = self.db.get_connection()
        logger.info(f"Persisting {len(forecast_list)} forecast(s) to `gold_bofa_sector_day_start_forecasts`...")

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
        for f in forecast_list:
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

