"""Production Day-Start Forecaster Orchestrator & Auto-Champion Model Arena."""

from datetime import date, datetime, timedelta
from typing import Dict, List, Optional, Tuple, Union

import pandas as pd
from dateutil.relativedelta import relativedelta

from mdk_trading_oracle.core.config import get_settings
from mdk_trading_oracle.core.db import DuckDBManager
from mdk_trading_oracle.core.logger import get_logger
from mdk_trading_oracle.core.time import now_turkey_naive
from mdk_trading_oracle.models.base import BaseForecaster, FlowThresholdProfile, ForecastResult
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

    def __init__(self, thresholds: Optional[FlowThresholdProfile] = None):
        self.thresholds = thresholds or FlowThresholdProfile()
        self.candidates: Dict[str, BaseForecaster] = {
            "Baseline 0: Naive W4 Persistence": DayStartNaivePersistenceModel(thresholds=self.thresholds),
            "Baseline 1: 5-Day Historical Mean": DayStartRollingMeanModel(thresholds=self.thresholds),
            "LightGBM Non-Linear Ensemble": DayStartLightGBMModel(thresholds=self.thresholds),
            "Bayesian Ridge Probabilistic": DayStartBayesianModel(thresholds=self.thresholds),
            "PyMC Bayesian GLM (MAP)": DayStartPyMCModel(use_map=True, thresholds=self.thresholds),
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
        1. Feature extraction across all 8 Feature Clusters from DuckDB Silver tables
        2. Dynamic empirical flow percentile threshold integration
        3. Automated Model Arena tournament selection or configured model type
        4. Probabilistic model training and walk-forward validation
        5. Live next-day forecasting (T+1) and historical backtest evaluation
        6. Persisting forecasts directly into DuckDB Gold layer tables
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
        self.threshold_profile = self._load_threshold_profile()
        self.arena = DayStartModelArena(thresholds=self.threshold_profile)
        self.champion_name: Optional[str] = None
        
        if self.model_type == "lightgbm":
            self.model: Optional[BaseForecaster] = DayStartLightGBMModel(thresholds=self.threshold_profile)
        elif self.model_type == "baseline":
            self.model = DayStartNaivePersistenceModel(thresholds=self.threshold_profile)
        elif self.model_type == "pymc":
            self.model = DayStartPyMCModel(use_map=True, thresholds=self.threshold_profile)
        elif self.model_type == "bayesian":
            self.model = DayStartBayesianModel(thresholds=self.threshold_profile)
        else:  # "auto"
            self.model = None

    def _load_threshold_profile(self) -> FlowThresholdProfile:
        """Load empirical flow percentile thresholds from silver_bofa_historical_flow_thresholds."""
        conn = self.db.get_connection()
        try:
            row = conn.execute("""
                SELECT buy_p25_tl, buy_p50_tl, buy_p85_tl, sell_p25_tl, sell_p50_tl, sell_p85_tl,
                       buy_count, sell_count, total_sessions
                FROM silver_bofa_historical_flow_thresholds
                WHERE scope_type = 'MACRO' AND scope_name = 'ALL' AND broker_id = 'MLB' AND window_name = 'day_start'
                LIMIT 1;
            """).fetchone()
            if row:
                return FlowThresholdProfile(
                    buy_p25_tl=float(row[0]),
                    buy_p50_tl=float(row[1]),
                    buy_p85_tl=float(row[2]),
                    sell_p25_tl=float(row[3]),
                    sell_p50_tl=float(row[4]),
                    sell_p85_tl=float(row[5]),
                    buy_count=int(row[6]),
                    sell_count=int(row[7]),
                    total_sessions=int(row[8]),
                )
        except Exception as e:
            logger.debug(f"Could not load macro flow thresholds from silver table: {e}. Using fallback defaults.")
        return FlowThresholdProfile()

    def _ensure_champion_fitted(self, as_of_date: Optional[date] = None) -> Tuple[pd.DataFrame, pd.Series]:
        """Extract historical features up to as_of_date, select champion model dynamically if auto, and fit champion model."""
        df_pl = self.feature_extractor.extract_features(end_date=as_of_date)
        if df_pl.height == 0:
            if self.model is not None:
                return pd.DataFrame(), pd.Series()
            self.model = DayStartNaivePersistenceModel()
            self.champion_name = self.model.model_name
            return pd.DataFrame(), pd.Series()

        df_pd = df_pl.to_pandas()
        X = df_pd.drop(columns=["target_open_net_flow_tl", "target_open_direction"], errors="ignore")
        y = df_pd["target_open_net_flow_tl"]

        # Automatic Champion Selection on the fly if model_type == "auto" or model is not instantiated
        if (self.model_type == "auto" or self.model is None) and len(df_pd) >= 2:
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
        elif self.model is None:
            self.model = DayStartNaivePersistenceModel()
            self.champion_name = self.model.model_name
        else:
            self.champion_name = self.model.model_name

        # Train champion model on historical dataset
        logger.info(f"Fitting Champion Model '{self.model.model_name}' on {len(df_pd)} historical daily sessions...")
        self.model.fit(X, y)
        return X, y

    def forecast_next_day(self, as_of_date: Optional[Union[str, date]] = None) -> ForecastResult:
        """Generate the live prediction for the upcoming trading morning (T_next) based on T_close.
        
        Args:
            as_of_date: Optional reference date for point-in-time historical inference (hiding subsequent data).
        """
        if isinstance(as_of_date, str):
            as_of_date = datetime.strptime(as_of_date[:10], "%Y-%m-%d").date()

        self._ensure_champion_fitted(as_of_date=as_of_date)
        df_next_pl = self.feature_extractor.extract_next_day_features(as_of_date=as_of_date)
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
            f"🎯 Generated Forecast for {res.forecast_date}: "
            f"Predicted Flow = {res.predicted_net_flow_tl / 1e6:+.2f}M TL, "
            f"Direction = {res.predicted_direction} ({res.direction_confidence*100:.1f}%), "
            f"Playbook = {res.predicted_playbook} (Champion: '{self.champion_name}')."
        )
        return res

    def backtest_all_history(self) -> List[ForecastResult]:
        """Generate historical in-sample / backtest predictions across all historical training sessions."""
        X, _ = self._ensure_champion_fitted()
        if X.empty:
            return []
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

    def save_forecasts_to_gold(
        self,
        forecasts: Union[ForecastResult, List[ForecastResult]],
        replace_active: bool = True,
    ) -> int:
        """Persist active live forecast(s) into DuckDB Gold table (`gold_bofa_day_start_forecasts`).
        
        Args:
            forecasts: The active forecast(s) to save.
            replace_active: If True (default), cleans out previous forecasts so the table strictly
                            holds only the active upcoming trading session (T+1).
        """
        if isinstance(forecasts, ForecastResult):
            forecast_list = [forecasts]
        else:
            forecast_list = forecasts

        if not forecast_list:
            return 0

        conn = self.db.get_connection()
        logger.info(f"Persisting {len(forecast_list)} live forecast(s) to `gold_bofa_day_start_forecasts` (replace_active={replace_active})...")

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

        if replace_active:
            conn.execute("DELETE FROM gold_bofa_day_start_forecasts;")

        # Insert predictions
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
        logger.info(f"Successfully updated `gold_bofa_day_start_forecasts`: {saved_count:,} active forecast(s).")
        return saved_count

    def reconcile_and_update_performance_ledger(
        self,
        forecasts: Optional[List[ForecastResult]] = None,
    ) -> int:
        """Reconcile completed session forecasts against actual Silver Window 1 market data and upsert into `gold_bofa_day_start_performance`.
        
        Args:
            forecasts: Optional list of previously generated ForecastResult objects. If None, backtest results
                       are used to reconcile all completed historical dates.
        """
        if forecasts is None:
            forecasts = self.backtest_all_history()

        if not forecasts:
            return 0

        conn = self.db.get_connection()
        logger.info(f"Reconciling {len(forecasts)} session(s) into `gold_bofa_day_start_performance`...")

        # Ensure schema exists
        conn.execute("""
            CREATE TABLE IF NOT EXISTS gold_bofa_day_start_performance (
                trade_date DATE PRIMARY KEY,
                day_of_week INTEGER,
                is_monday BOOLEAN,
                predicted_open_net_flow_tl DOUBLE,
                predicted_open_flow_lower_90 DOUBLE,
                predicted_open_flow_upper_90 DOUBLE,
                actual_open_net_flow_tl DOUBLE,
                error_open_net_flow_tl DOUBLE,
                absolute_error_tl DOUBLE,
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
                forecast_generated_at TIMESTAMP,
                realized_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
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

        reconciled_count = 0
        for f in forecasts:
            d = f.forecast_date
            d_str = str(d)[:10]
            if d_str not in actuals_map:
                # Session has not been completed yet (e.g. T+1 future session), skip
                continue

            dow = d.weekday() + 1 if isinstance(d, date) else 1
            is_mon = (dow == 1)
            actual_flow = float(actuals_map[d_str])
            pred_flow = float(f.predicted_net_flow_tl)
            error_flow = actual_flow - pred_flow
            abs_error = abs(error_flow)
            actual_dir = "BUY" if actual_flow > 0 else ("SELL" if actual_flow < 0 else "NEUTRAL")
            is_hit = bool((pred_flow > 0 and actual_flow > 0) or (pred_flow <= 0 and actual_flow <= 0))
            is_in_ci = bool(f.predicted_flow_lower_90 <= actual_flow <= f.predicted_flow_upper_90)

            conn.execute("""
                INSERT OR REPLACE INTO gold_bofa_day_start_performance (
                    trade_date, day_of_week, is_monday,
                    predicted_open_net_flow_tl, predicted_open_flow_lower_90, predicted_open_flow_upper_90,
                    actual_open_net_flow_tl, error_open_net_flow_tl, absolute_error_tl,
                    predicted_direction, actual_direction,
                    is_direction_hit, is_inside_90_ci,
                    direction_confidence, predicted_playbook,
                    top_predicted_buy_sector, top_predicted_sell_sector,
                    model_name, model_version, forecast_generated_at, realized_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?);
            """, [
                f.forecast_date,
                dow,
                is_mon,
                pred_flow,
                f.predicted_flow_lower_90,
                f.predicted_flow_upper_90,
                actual_flow,
                error_flow,
                abs_error,
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
                f.generated_at,
                now_turkey_naive(),
            ])
            reconciled_count += 1

        total_perf = conn.execute("SELECT COUNT(*) FROM gold_bofa_day_start_performance;").fetchone()[0]
        logger.info(f"Successfully updated `gold_bofa_day_start_performance`: {total_perf:,} total recorded sessions ({reconciled_count} reconciled in this run).")
        return total_perf

    def backfill_historical_performance(
        self,
        target_dates: Optional[List[Union[str, date]]] = None,
        all_missing: bool = False,
        lookback_months: Optional[int] = None,
        lookback_days: Optional[int] = None,
    ) -> int:
        """Perform zero-lookahead point-in-time forecasting for past missed trading days and record into `gold_bofa_day_start_performance`.
        
        Args:
            target_dates: Specific list of past trading dates (e.g. ['2026-03-10', '2026-03-18']) to backfill.
            all_missing: If True and target_dates is None, auto-discovers dates in Silver within the configured lookback
                         window that are currently missing from the performance ledger.
            lookback_months: Number of trailing months to look back for missing dates (default: from config, usually 2 months).
            lookback_days: Optional number of trailing days to look back (overrides lookback_months if provided).
        """
        conn = self.db.get_connection()
        all_silver_dates = [
            r[0] for r in conn.execute(
                f"SELECT DISTINCT trade_date FROM silver_intraday_broker_window_summary WHERE broker_id = '{self.target_broker}' AND window_name = 'day_start' ORDER BY trade_date ASC;"
            ).fetchall()
        ]

        dates_to_backfill: List[date] = []
        if target_dates:
            for d in target_dates:
                if isinstance(d, str):
                    d_obj = datetime.strptime(d[:10], "%Y-%m-%d").date()
                else:
                    d_obj = d
                dates_to_backfill.append(d_obj)
        elif all_missing:
            backfill_cfg = self.settings.get_backfill_config()
            eff_months = lookback_months if lookback_months is not None else backfill_cfg.get("default_lookback_months", 2)
            eff_days = lookback_days if lookback_days is not None else backfill_cfg.get("default_lookback_days", None)

            existing_tables = [r[0] for r in conn.execute("SHOW TABLES;").fetchall()]
            existing_perf_dates = set()
            if "gold_bofa_day_start_performance" in existing_tables:
                existing_perf_dates = set(r[0] for r in conn.execute("SELECT trade_date FROM gold_bofa_day_start_performance;").fetchall())

            if all_silver_dates:
                latest_date = max(all_silver_dates)
                if eff_days is not None:
                    cutoff_date = latest_date - timedelta(days=eff_days)
                else:
                    cutoff_date = latest_date - relativedelta(months=eff_months)

                candidate_dates = [d for d in all_silver_dates if d >= cutoff_date]
                dates_to_backfill = [d for d in candidate_dates if d not in existing_perf_dates]
                logger.info(
                    f"Auto-discovering missing dates within trailing window (>= {cutoff_date}, lookback_months={eff_months}, lookback_days={eff_days}): "
                    f"{len(dates_to_backfill)} missing of {len(candidate_dates)} eligible sessions."
                )
            else:
                dates_to_backfill = []

        if not dates_to_backfill:
            logger.info("No dates to backfill for Day-Start Macro model.")
            return 0

        logger.info(f"Starting zero-lookahead point-in-time backfill for {len(dates_to_backfill)} date(s): {dates_to_backfill}...")
        backfilled_forecasts: List[ForecastResult] = []
        for target_d in dates_to_backfill:
            prior_dates = [d for d in all_silver_dates if d < target_d]
            if not prior_dates:
                logger.warning(f"Skipping backfill for {target_d}: No prior historical session available for feature computation.")
                continue
            as_of_d = max(prior_dates)
            logger.info(f"Backfilling {target_d} with strict zero-leakage cutoff as_of_date={as_of_d}...")
            res = self.forecast_next_day(as_of_date=as_of_d)
            res.forecast_date = target_d
            backfilled_forecasts.append(res)

        saved_count = self.reconcile_and_update_performance_ledger(backfilled_forecasts)
        logger.info(f"Point-in-time backfill complete. Processed {len(backfilled_forecasts)} date(s).")
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
                now_turkey_naive(),
            ])

        saved_count = conn.execute("SELECT COUNT(*) FROM gold_bofa_day_start_backtests;").fetchone()[0]
        logger.info(f"Successfully updated `gold_bofa_day_start_backtests`: {saved_count:,} total records.")
        return saved_count



