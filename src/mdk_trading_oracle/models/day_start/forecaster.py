"""Production Day-Start Forecaster Orchestrator & Auto-Champion Model Arena."""

from datetime import date, datetime
from typing import Dict, List, Optional, Tuple, Union

import pandas as pd

from mdk_trading_oracle.core.config import get_settings
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
        self,
        X: pd.DataFrame,
        y: pd.Series,
        min_train_samples: int = 5,
        eval_window_days: Optional[int] = None,
    ) -> Tuple[pd.DataFrame, BaseForecaster]:
        """Execute walk-forward out-of-sample tournament across all candidate models."""
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
        4. Live next-day forecasting (T+1) and historical backtest evaluation
        5. Persisting forecasts directly into DuckDB Gold layer tables
    """

    def __init__(
        self,
        db: Optional[DuckDBManager] = None,
        model_type: Optional[str] = None,
        lookback_months: Optional[int] = None,
        eval_window_days: Optional[int] = None,
        min_burn_in_days: Optional[int] = None,
    ):
        self.db = db or DuckDBManager()
        self.target_broker = "MLB"
        self.settings = get_settings()
        cfg = self.settings.get_model_config("day_start")

        self.lookback_months = lookback_months if lookback_months is not None else cfg.get("lookback_months", 12)
        self.eval_window_days = eval_window_days if eval_window_days is not None else cfg.get("eval_window_days", 20)
        self.min_burn_in_days = min_burn_in_days if min_burn_in_days is not None else cfg.get("min_burn_in_days", 5)
        self.model_type = model_type or cfg.get("model_type", "auto")

        self.feature_extractor = DayStartFeatureExtractor(
            self.db, target_broker_id=self.target_broker, lookback_months=self.lookback_months
        )
        self.arena = DayStartModelArena()
        self.champion_name: Optional[str] = None
        
        if self.model_type == "lightgbm":
            self.model: Optional[BaseForecaster] = DayStartLightGBMModel()
        elif self.model_type == "baseline":
            self.model = DayStartNaivePersistenceModel()
        elif self.model_type == "pymc":
            self.model = DayStartPyMCModel(use_map=True)
        elif self.model_type == "bayesian":
            self.model = DayStartBayesianModel()
        else:  # "auto"
            self.model = None

    def _ensure_champion_fitted(self) -> Tuple[pd.DataFrame, pd.Series]:
        """Extract historical features, select champion model dynamically if auto, and fit champion model."""
        df_pl = self.feature_extractor.extract_features()
        if df_pl.height == 0:
            raise ValueError("No historical feature records found to train DayStartForecaster.")

        df_pd = df_pl.to_pandas()
        X = df_pd.drop(columns=["target_open_net_flow_tl", "target_open_direction"], errors="ignore")
        y = df_pd["target_open_net_flow_tl"]

        # Automatic Champion Selection on the fly if model_type == "auto" or model is not instantiated
        if self.model_type == "auto" or self.model is None:
            logger.info(
                f"Running DayStartModelArena Walk-Forward Tournament "
                f"(eval_window_days={self.eval_window_days}, min_burn_in={self.min_burn_in_days})..."
            )
            min_burn_in = min(self.min_burn_in_days, max(2, len(df_pd) - 1))
            _, champion_model = self.arena.run_tournament(
                X, y, min_train_samples=min_burn_in, eval_window_days=self.eval_window_days
            )
            self.model = champion_model
            self.champion_name = champion_model.model_name
        else:
            self.champion_name = self.model.model_name

        # Train champion model on full historical dataset
        logger.info(f"Fitting Champion Model '{self.model.model_name}' on {len(df_pd)} historical daily sessions...")
        self.model.fit(X, y)
        return X, y

    def forecast_next_day(self) -> ForecastResult:
        """Generate the live prediction for the upcoming trading morning (T_next) based on latest T_close."""
        self._ensure_champion_fitted()
        df_next_pl = self.feature_extractor.extract_next_day_features()
        if df_next_pl.height == 0:
            raise ValueError("Failed to extract next-day feature vector.")

        df_next_pd = df_next_pl.to_pandas()
        res = self.model.predict(df_next_pd)

        # Predict top buy/sell sector based on latest session flows
        banking_flow = float(df_next_pd["feat_bofa_banking_flow_prev_day"].iloc[0]) if "feat_bofa_banking_flow_prev_day" in df_next_pd.columns else 0.0
        transport_flow = float(df_next_pd["feat_bofa_transport_flow_prev_day"].iloc[0]) if "feat_bofa_transport_flow_prev_day" in df_next_pd.columns else 0.0
        res.top_predicted_buy_sector = "Banking" if banking_flow > transport_flow else "Transportation"
        res.top_predicted_sell_sector = "Holding" if res.predicted_net_flow_tl > 0 else "Energy & Refining"

        logger.info(
            f"🎯 Generated Live Next-Day Forecast for {res.forecast_date}: "
            f"Predicted Flow = {res.predicted_net_flow_tl / 1e6:+.2f}M TL, "
            f"Direction = {res.predicted_direction} ({res.direction_confidence*100:.1f}%), "
            f"Playbook = {res.predicted_playbook} (Champion: '{self.champion_name}')."
        )
        return res

    def backtest_all_history(self) -> List[ForecastResult]:
        """Generate historical in-sample / backtest predictions across all historical training sessions."""
        X, _ = self._ensure_champion_fitted()
        results: List[ForecastResult] = []
        for idx in range(len(X)):
            row = X.iloc[[idx]]
            res = self.model.predict(row)
            banking_flow = float(row["feat_bofa_banking_flow_prev_day"].iloc[0]) if "feat_bofa_banking_flow_prev_day" in row.columns else 0.0
            transport_flow = float(row["feat_bofa_transport_flow_prev_day"].iloc[0]) if "feat_bofa_transport_flow_prev_day" in row.columns else 0.0
            res.top_predicted_buy_sector = "Banking" if banking_flow > transport_flow else "Transportation"
            res.top_predicted_sell_sector = "Holding" if res.predicted_net_flow_tl > 0 else "Energy & Refining"
            results.append(res)
        logger.info(f"Generated {len(results)} historical backtest forecasts using Champion '{self.champion_name}'.")
        return results

    def train_and_forecast_all(
        self,
        include_history: bool = False,
        include_next_day: bool = True,
    ) -> List[ForecastResult]:
        """Extract features, fit champion model, and generate forecasts.
        
        Args:
            include_history: If True, includes historical backtest forecasts for all training days.
            include_next_day: If True (default), includes the live forecast for upcoming session T_next.
        """
        results: List[ForecastResult] = []
        if include_history:
            results.extend(self.backtest_all_history())
        if include_next_day:
            next_forecast = self.forecast_next_day()
            results.append(next_forecast)
        return results

    def save_forecasts_to_gold(self, forecasts: Union[ForecastResult, List[ForecastResult]]) -> int:
        """Persist generated forecast(s) into DuckDB Gold table (`gold_bofa_day_start_forecasts`)."""
        if isinstance(forecasts, ForecastResult):
            forecast_list = [forecasts]
        else:
            forecast_list = forecasts

        if not forecast_list:
            return 0

        conn = self.db.get_connection()
        logger.info(f"Persisting {len(forecast_list)} forecast(s) to `gold_bofa_day_start_forecasts`...")

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
                predicted_playbook VARCHAR,
                top_predicted_buy_sector VARCHAR,
                top_predicted_sell_sector VARCHAR,
                model_name VARCHAR,
                model_version VARCHAR,
                generated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            );
        """)

        # Insert or replace predictions
        for f in forecast_list:
            d = f.forecast_date
            dow = d.weekday() + 1 if isinstance(d, date) else 1
            is_mon = (dow == 1)
            conn.execute("""
                INSERT OR REPLACE INTO gold_bofa_day_start_forecasts (
                    forecast_date, day_of_week, is_monday,
                    predicted_open_net_flow_tl, predicted_open_flow_lower_90, predicted_open_flow_upper_90,
                    predicted_direction, direction_confidence,
                    predicted_playbook, top_predicted_buy_sector, top_predicted_sell_sector,
                    model_name, model_version, generated_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?);
            """, [
                f.forecast_date,
                dow,
                is_mon,
                f.predicted_net_flow_tl,
                f.predicted_flow_lower_90,
                f.predicted_flow_upper_90,
                f.predicted_direction,
                f.direction_confidence,
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

    def save_backtests_to_gold(self, backtest_results: Optional[List[ForecastResult]] = None) -> int:
        """Persist historical backtest results joined with actuals into `gold_bofa_day_start_backtests`."""
        if backtest_results is None:
            backtest_results = self.backtest_all_history()

        if not backtest_results:
            return 0

        conn = self.db.get_connection()
        logger.info(f"Persisting {len(backtest_results)} backtest record(s) to `gold_bofa_day_start_backtests`...")

        # Ensure schema exists
        conn.execute("""
            CREATE TABLE IF NOT EXISTS gold_bofa_day_start_backtests (
                trade_date DATE PRIMARY KEY,
                day_of_week INTEGER,
                is_monday BOOLEAN,
                predicted_open_net_flow_tl DOUBLE,
                predicted_open_flow_lower_90 DOUBLE,
                predicted_open_flow_upper_90 DOUBLE,
                actual_open_net_flow_tl DOUBLE,
                error_open_net_flow_tl DOUBLE,
                predicted_direction VARCHAR,
                actual_direction VARCHAR,
                is_direction_hit BOOLEAN,
                is_inside_90_ci BOOLEAN,
                direction_confidence DOUBLE,
                predicted_playbook VARCHAR,
                top_predicted_buy_sector VARCHAR,
                top_predicted_sell_sector VARCHAR,
                model_name VARCHAR,
                model_version VARCHAR,
                calculated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            );
        """)

        # Fetch actual Window 1 flows from silver_intraday_broker_window_summary
        actuals_df = conn.execute(f"""
            SELECT 
                trade_date,
                SUM(net_flow_tl) AS actual_w1_net_flow_tl
            FROM silver_intraday_broker_window_summary
            WHERE broker_id = '{self.target_broker}' AND window_name = 'day_start'
            GROUP BY trade_date;
        """).df()
        actuals_map = dict(zip(actuals_df["trade_date"].astype(str).str.slice(0, 10), actuals_df["actual_w1_net_flow_tl"]))

        for f in backtest_results:
            d = f.forecast_date
            d_str = str(d)[:10]
            dow = d.weekday() + 1 if isinstance(d, date) else 1
            is_mon = (dow == 1)
            actual_flow = float(actuals_map.get(d_str, 0.0))
            pred_flow = float(f.predicted_net_flow_tl)
            error_flow = actual_flow - pred_flow
            actual_dir = "BUY" if actual_flow > 0 else ("SELL" if actual_flow < 0 else "NEUTRAL")
            is_hit = bool((pred_flow > 0 and actual_flow > 0) or (pred_flow <= 0 and actual_flow <= 0))
            is_in_ci = bool(f.predicted_flow_lower_90 <= actual_flow <= f.predicted_flow_upper_90)

            conn.execute("""
                INSERT OR REPLACE INTO gold_bofa_day_start_backtests (
                    trade_date, day_of_week, is_monday,
                    predicted_open_net_flow_tl, predicted_open_flow_lower_90, predicted_open_flow_upper_90,
                    actual_open_net_flow_tl, error_open_net_flow_tl,
                    predicted_direction, actual_direction,
                    is_direction_hit, is_inside_90_ci,
                    direction_confidence, predicted_playbook,
                    top_predicted_buy_sector, top_predicted_sell_sector,
                    model_name, model_version, calculated_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?);
            """, [
                f.forecast_date,
                dow,
                is_mon,
                pred_flow,
                f.predicted_flow_lower_90,
                f.predicted_flow_upper_90,
                actual_flow,
                error_flow,
                f.predicted_direction,
                actual_dir,
                is_hit,
                is_in_ci,
                f.direction_confidence,
                f.predicted_playbook,
                f.top_predicted_buy_sector,
                f.top_predicted_sell_sector,
                f.model_name,
                f.model_version,
                datetime.now(),
            ])

        saved_count = conn.execute("SELECT COUNT(*) FROM gold_bofa_day_start_backtests;").fetchone()[0]
        logger.info(f"Successfully updated `gold_bofa_day_start_backtests`: {saved_count:,} total records.")
        return saved_count


