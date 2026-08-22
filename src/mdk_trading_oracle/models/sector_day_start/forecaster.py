"""Production Sector Day-Start Forecaster Orchestrator & Auto-Champion Model Arena."""

from datetime import date, datetime, timedelta
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
from dateutil.relativedelta import relativedelta

from mdk_trading_oracle.core.config import get_settings
from mdk_trading_oracle.core.db import DuckDBManager
from mdk_trading_oracle.core.logger import get_logger
from mdk_trading_oracle.core.time import now_turkey_naive
from mdk_trading_oracle.models.base import FlowThresholdProfile, ForecastResult
from mdk_trading_oracle.models.features_config import FeatureSelector
from mdk_trading_oracle.models.registry import ModelRegistry
from mdk_trading_oracle.models.sector_day_start.features import SectorDayStartFeatureExtractor
from mdk_trading_oracle.models.sector_day_start.models import (
    BaseSectorDayStartModel,
    SectorDayStartBayesianModel,
    SectorDayStartLightGBMModel,
    SectorDayStartNaivePersistenceModel,
    SectorDayStartPyMCModel,
    SectorDayStartRollingMeanModel,
    SectorDayStartXGBoostModel,
)

logger = get_logger("mdk_oracle.models.sector_day_start.forecaster")


@ModelRegistry.register("sector_day_start_model_arena")
class SectorDayStartModelArena:
    """Evaluates candidate sector models using expanding-window walk-forward validation and crowns the champion."""

    def __init__(
        self,
        include_pymc: bool = False,
        sector_thresholds: Optional[Dict[str, FlowThresholdProfile]] = None,
    ):
        self.sector_thresholds = sector_thresholds or {}
        self.candidates: Dict[str, BaseSectorDayStartModel] = {
            "Baseline 0: Naive W4 Sector Persistence": SectorDayStartNaivePersistenceModel(sector_thresholds=self.sector_thresholds),
            "Baseline 1: 5-Day Historical Sector Mean": SectorDayStartRollingMeanModel(sector_thresholds=self.sector_thresholds),
            "LightGBM Non-Linear Sector Ensemble": SectorDayStartLightGBMModel(sector_thresholds=self.sector_thresholds),
            "XGBoost Non-Linear Sector Ensemble": SectorDayStartXGBoostModel(sector_thresholds=self.sector_thresholds),
            "Bayesian Ridge Probabilistic": SectorDayStartBayesianModel(sector_thresholds=self.sector_thresholds),
        }
        if include_pymc:
            self.candidates["PyMC Bayesian GLM (MAP)"] = SectorDayStartPyMCModel(
                use_map=True, sector_thresholds=self.sector_thresholds
            )

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
        2. Dynamic empirical sector percentile threshold integration
        3. Automated Model Arena tournament selection across sectors
        4. Probabilistic model training and walk-forward validation
        5. Live next-day sector forecasting (T+1) and historical backtest evaluation
        6. Persisting forecasts directly into DuckDB Gold table `gold_bofa_sector_day_start_forecasts`
    """

    def __init__(
        self,
        db: Optional[DuckDBManager] = None,
        model_type: Optional[str] = None,
        lookback_months: Optional[int] = None,
        eval_window_days: Optional[int] = None,
        min_burn_in_days: Optional[int] = None,
        include_pymc_arena: Optional[bool] = None,
        feature_selector: Optional[FeatureSelector] = None,
        disabled_clusters: Optional[List[str]] = None,
        enabled_clusters: Optional[List[str]] = None,
        include_features: Optional[List[str]] = None,
        exclude_features: Optional[List[str]] = None,
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

        self.feature_selector = feature_selector or FeatureSelector(
            model_name="sector_day_start",
            disabled_clusters=disabled_clusters,
            enabled_clusters=enabled_clusters,
            include_features=include_features,
            exclude_features=exclude_features,
        )

        self.feature_extractor = SectorDayStartFeatureExtractor(
            self.db, target_broker_id=self.target_broker, lookback_months=self.lookback_months
        )
        self.sector_thresholds = self._load_sector_thresholds()
        self.arena = SectorDayStartModelArena(include_pymc=include_pymc, sector_thresholds=self.sector_thresholds)
        self.champion_name: Optional[str] = None
        if self.model_type != "auto":
            self.champion_name = self.model_type

    def _load_sector_thresholds(self) -> Dict[str, FlowThresholdProfile]:
        """Load empirical flow percentile thresholds for all sectors from silver_bofa_historical_flow_thresholds."""
        conn = self.db.get_connection()
        profiles: Dict[str, FlowThresholdProfile] = {}
        try:
            rows = conn.execute("""
                SELECT scope_name, buy_p25_tl, buy_p50_tl, buy_p85_tl, sell_p25_tl, sell_p50_tl, sell_p85_tl,
                       buy_count, sell_count, total_sessions
                FROM silver_bofa_historical_flow_thresholds
                WHERE scope_type = 'SECTOR' AND broker_id = 'MLB' AND window_name = 'day_start';
            """).fetchall()
            for row in rows:
                sec_name = str(row[0])
                profiles[sec_name] = FlowThresholdProfile(
                    buy_p25_tl=float(row[1]),
                    buy_p50_tl=float(row[2]),
                    buy_p85_tl=float(row[3]),
                    sell_p25_tl=float(row[4]),
                    sell_p50_tl=float(row[5]),
                    sell_p85_tl=float(row[6]),
                    buy_count=int(row[7]),
                    sell_count=int(row[8]),
                    total_sessions=int(row[9]),
                )
        except Exception as e:
            logger.debug(f"Could not load sector flow thresholds from silver table: {e}. Using fallback defaults.")
        return profiles

    def _ensure_champion_selected(
        self,
        tracked_sectors: List[str],
        as_of_date: Optional[date] = None,
    ) -> None:
        """Run arena tournament across tracked sectors if model_type is auto."""
        if self.model_type != "auto" and self.champion_name is not None:
            return

        scoreboard = []
        for name, candidate_model in self.arena.candidates.items():
            hit_rates = []
            picps = []
            rmses = []
            for sector in tracked_sectors[:5]:
                df_pl = self.feature_extractor.extract_features(sector=sector, end_date=as_of_date)
                if df_pl.height < 5:
                    continue
                df_filtered_pl = self.feature_selector.filter_dataframe(df_pl)
                df_pd = df_filtered_pl.to_pandas()
                X = df_pd.drop(columns=["target_sector_open_net_flow_tl", "target_sector_open_direction"], errors="ignore")
                y = df_pd["target_sector_open_net_flow_tl"]
                min_burn = min(self.min_burn_in_days, max(2, len(df_pd) - 1))
                metrics = candidate_model.walk_forward_evaluate(
                    X, y, min_train_samples=min_burn, eval_window_days=self.eval_window_days
                )
                hit_rates.append(metrics["hit_rate_pct"])
                picps.append(metrics["picp_90_pct"])
                rmses.append(metrics["rmse_million_tl"])

            if hit_rates:
                scoreboard.append({
                    "model_name": name,
                    "avg_hit_rate": float(np.mean(hit_rates)),
                    "avg_picp": float(np.mean(picps)),
                    "avg_rmse": float(np.mean(rmses)),
                })

        if scoreboard:
            df_scores = pd.DataFrame(scoreboard).sort_values(
                by=["avg_hit_rate", "avg_picp", "avg_rmse"],
                ascending=[False, False, True],
            )
            self.champion_name = df_scores.iloc[0]["model_name"]
        else:
            self.champion_name = "sector_day_start_naive_persistence"

        logger.info(
            f"🏆 Sector Model Arena Champion Selected: '{self.champion_name}' "
            f"(with {len(self.feature_selector.active_features)} active feature(s))"
        )

    def _create_sector_model(self) -> BaseSectorDayStartModel:
        """Instantiate a fresh model instance of the crowned champion paradigm."""
        if self.champion_name in ["sector_day_start_bayesian_ridge", "bayesian"]:
            return SectorDayStartBayesianModel(sector_thresholds=self.sector_thresholds)
        elif self.champion_name in ["sector_day_start_lightgbm", "lightgbm"]:
            return SectorDayStartLightGBMModel(sector_thresholds=self.sector_thresholds)
        elif self.champion_name in ["sector_day_start_xgboost", "xgboost"]:
            return SectorDayStartXGBoostModel(sector_thresholds=self.sector_thresholds)
        elif self.champion_name in ["sector_day_start_pymc", "pymc"]:
            return SectorDayStartPyMCModel(use_map=True, sector_thresholds=self.sector_thresholds)
        elif self.champion_name in ["sector_day_start_rolling_mean", "rolling_mean"]:
            return SectorDayStartRollingMeanModel(sector_thresholds=self.sector_thresholds)
        else:
            return SectorDayStartNaivePersistenceModel(sector_thresholds=self.sector_thresholds)

    def forecast_next_day(
        self,
        sectors: Optional[List[str]] = None,
        sector: Optional[str] = None,
        as_of_date: Optional[Union[str, date]] = None,
    ) -> List[ForecastResult]:
        """Generate live sector forecasts for the upcoming trading session (T_next) based on T_close.
        
        Args:
            sectors: Optional list of sectors to forecast.
            sector: Optional single sector filter.
            as_of_date: Optional reference date for point-in-time historical inference (hiding subsequent data).
        """
        if isinstance(as_of_date, str):
            as_of_date = datetime.strptime(as_of_date[:10], "%Y-%m-%d").date()

        if sector:
            tracked_sectors = [sector]
        elif sectors:
            tracked_sectors = sectors
        else:
            tracked_sectors = self.feature_extractor.get_tracked_sectors(min_session_count=10)

        if not tracked_sectors:
            logger.warning("No tracked sectors found for SectorDayStartForecaster.")
            return []

        self._ensure_champion_selected(tracked_sectors, as_of_date=as_of_date)

        # Extract next-day feature rows across tracked sectors
        df_next_pl = self.feature_extractor.extract_next_day_features(sectors=tracked_sectors, as_of_date=as_of_date)
        if df_next_pl.height == 0:
            logger.warning("No next-day sector feature records extracted.")
            return []

        df_next_filtered_pl = self.feature_selector.filter_dataframe(df_next_pl)
        df_next_pd = df_next_filtered_pl.to_pandas()
        live_forecasts: List[ForecastResult] = []

        for sec in tracked_sectors:
            sec_next_row = df_next_pd[df_next_pd["sector"] == sec]
            if sec_next_row.empty:
                continue

            # Train on historical sector data up to as_of_date
            df_hist_pl = self.feature_extractor.extract_features(sector=sec, end_date=as_of_date)
            if df_hist_pl.height == 0:
                continue

            df_hist_filtered_pl = self.feature_selector.filter_dataframe(df_hist_pl)
            df_hist_pd = df_hist_filtered_pl.to_pandas()
            X_hist = df_hist_pd.drop(columns=["target_sector_open_net_flow_tl", "target_sector_open_direction"], errors="ignore")
            y_hist = df_hist_pd["target_sector_open_net_flow_tl"]

            model = self._create_sector_model()
            model.fit(X_hist, y_hist)

            res = model.predict(sec_next_row)
            res.top_predicted_buy_sector = sec
            live_forecasts.append(res)

        logger.info(
            f"🎯 Generated {len(live_forecasts)} Sector Forecasts across sectors "
            f"for {live_forecasts[0].forecast_date if live_forecasts else 'N/A'} (Champion: '{self.champion_name}')."
        )
        return live_forecasts

    def run_ablation_study(self, sectors: Optional[List[str]] = None) -> pd.DataFrame:
        """Run an automated Leave-One-Cluster-Out (LOCO) ablation tournament across sector feature clusters.

        Returns:
            pd.DataFrame: Scoreboard comparing Hit Rate %, 90% PICP %, and RMSE when each cluster is removed.
        """
        tracked_sectors = sectors or self.feature_extractor.get_tracked_sectors(min_session_count=10)
        if not tracked_sectors:
            return pd.DataFrame()

        clusters = self.feature_selector.get_available_clusters()
        ablation_results = []

        # 1. Baseline: All Features
        sel_all = FeatureSelector(model_name="sector_day_start")
        hit_rates_all, picps_all, rmses_all = [], [], []
        for sec in tracked_sectors[:5]:
            df_pl = self.feature_extractor.extract_features(sector=sec)
            if df_pl.height < 5:
                continue
            df_pd = sel_all.filter_dataframe(df_pl).to_pandas()
            X = df_pd.drop(columns=["target_sector_open_net_flow_tl", "target_sector_open_direction"], errors="ignore")
            y = df_pd["target_sector_open_net_flow_tl"]
            min_burn = min(self.min_burn_in_days, max(2, len(df_pd) - 1))
            scores, champ = self.arena.run_tournament(X, y, min_train_samples=min_burn, eval_window_days=self.eval_window_days)
            hit_rates_all.append(scores.iloc[0]["hit_rate_pct"])
            picps_all.append(scores.iloc[0]["picp_90_pct"])
            rmses_all.append(scores.iloc[0]["rmse_million_tl"])

        if hit_rates_all:
            ablation_results.append({
                "Experiment": "Baseline (All Sector Features)",
                "Active_Features": len(sel_all.active_features),
                "Removed_Cluster": "None",
                "Avg_Hit_Rate_Pct": float(np.mean(hit_rates_all)),
                "Avg_PICP_90_Pct": float(np.mean(picps_all)),
                "Avg_RMSE_Million_TL": float(np.mean(rmses_all)),
            })

        # 2. Leave out each cluster
        for cl in clusters:
            sel_loco = FeatureSelector(model_name="sector_day_start", disabled_clusters=[cl])
            hit_rates_loco, picps_loco, rmses_loco = [], [], []
            for sec in tracked_sectors[:5]:
                df_pl = self.feature_extractor.extract_features(sector=sec)
                if df_pl.height < 5:
                    continue
                df_pd = sel_loco.filter_dataframe(df_pl).to_pandas()
                X = df_pd.drop(columns=["target_sector_open_net_flow_tl", "target_sector_open_direction"], errors="ignore")
                y = df_pd["target_sector_open_net_flow_tl"]
                min_burn = min(self.min_burn_in_days, max(2, len(df_pd) - 1))
                scores, champ = self.arena.run_tournament(X, y, min_train_samples=min_burn, eval_window_days=self.eval_window_days)
                hit_rates_loco.append(scores.iloc[0]["hit_rate_pct"])
                picps_loco.append(scores.iloc[0]["picp_90_pct"])
                rmses_loco.append(scores.iloc[0]["rmse_million_tl"])

            if hit_rates_loco:
                ablation_results.append({
                    "Experiment": f"Without '{cl}'",
                    "Active_Features": len(sel_loco.active_features),
                    "Removed_Cluster": cl,
                    "Avg_Hit_Rate_Pct": float(np.mean(hit_rates_loco)),
                    "Avg_PICP_90_Pct": float(np.mean(picps_loco)),
                    "Avg_RMSE_Million_TL": float(np.mean(rmses_loco)),
                })

        df_res = pd.DataFrame(ablation_results).sort_values(by=["Avg_Hit_Rate_Pct", "Avg_PICP_90_Pct", "Avg_RMSE_Million_TL"], ascending=[False, False, True]).reset_index(drop=True)
        return df_res

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

            df_filtered_pl = self.feature_selector.filter_dataframe(df_pl)
            df_pd = df_filtered_pl.to_pandas()
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

    def save_forecasts_to_gold(
        self,
        forecasts: Union[ForecastResult, List[ForecastResult]],
        replace_active: bool = True,
    ) -> int:
        """Persist active live sector forecasts into DuckDB Gold table `gold_bofa_sector_day_start_forecasts`.
        
        Args:
            forecasts: The active sector forecast(s) to save.
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
        logger.info(f"Persisting {len(forecast_list)} live sector forecast(s) to `gold_bofa_sector_day_start_forecasts` (replace_active={replace_active})...")

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

        if replace_active:
            conn.execute("DELETE FROM gold_bofa_sector_day_start_forecasts;")

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
        logger.info(f"Successfully updated `gold_bofa_sector_day_start_forecasts`: {saved_count:,} active sector forecast(s).")
        return saved_count

    def reconcile_and_update_performance_ledger(
        self,
        forecasts: Optional[List[ForecastResult]] = None,
        sectors: Optional[List[str]] = None,
    ) -> int:
        """Reconcile completed sector forecasts against actual Silver Window 1 market data and upsert into `gold_bofa_sector_day_start_performance`."""
        if forecasts is None:
            forecasts = self.backtest_all_history(sectors=sectors)

        if not forecasts:
            return 0

        conn = self.db.get_connection()
        logger.info(f"Reconciling {len(forecasts)} sector session(s) into `gold_bofa_sector_day_start_performance`...")

        # Ensure schema exists
        conn.execute("""
            CREATE TABLE IF NOT EXISTS gold_bofa_sector_day_start_performance (
                trade_date DATE,
                sector VARCHAR,
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
                model_name VARCHAR,
                model_version VARCHAR,
                forecast_generated_at TIMESTAMP,
                realized_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                PRIMARY KEY (trade_date, sector)
            );
        """)

        # Fetch actual sector Window 1 flows from silver_intraday_sector_window_summary
        actuals_df = conn.execute(f"""
            SELECT 
                trade_date,
                sector,
                net_flow_tl AS actual_w1_net_flow_tl
            FROM silver_intraday_sector_window_summary
            WHERE broker_id = '{self.target_broker}' AND window_name = 'day_start';
        """).df()
        actuals_df["trade_date_str"] = actuals_df["trade_date"].astype(str).str.slice(0, 10)
        actuals_map = dict(zip(zip(actuals_df["trade_date_str"], actuals_df["sector"]), actuals_df["actual_w1_net_flow_tl"]))

        reconciled_count = 0
        for f in forecasts:
            d = f.forecast_date
            d_str = str(d)[:10]
            sector_name = f.top_predicted_buy_sector or "General"
            if (d_str, sector_name) not in actuals_map:
                continue

            dow = d.weekday() + 1 if isinstance(d, date) else 1
            is_mon = (dow == 1)
            actual_flow = float(actuals_map[(d_str, sector_name)])
            pred_flow = float(f.predicted_net_flow_tl)
            error_flow = actual_flow - pred_flow
            abs_error = abs(error_flow)
            actual_dir = "BUY" if actual_flow > 0 else ("SELL" if actual_flow < 0 else "NEUTRAL")
            is_hit = bool((pred_flow > 0 and actual_flow > 0) or (pred_flow <= 0 and actual_flow <= 0))
            is_in_ci = bool(f.predicted_flow_lower_90 <= actual_flow <= f.predicted_flow_upper_90)

            conn.execute("""
                INSERT OR REPLACE INTO gold_bofa_sector_day_start_performance (
                    trade_date, sector, day_of_week, is_monday,
                    predicted_open_net_flow_tl, predicted_open_flow_lower_90, predicted_open_flow_upper_90,
                    actual_open_net_flow_tl, error_open_net_flow_tl, absolute_error_tl,
                    predicted_direction, actual_direction,
                    is_direction_hit, is_inside_90_ci,
                    direction_confidence, predicted_playbook,
                    model_name, model_version, forecast_generated_at, realized_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?);
            """, [
                f.forecast_date,
                sector_name,
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
                f.model_name,
                f.model_version,
                f.generated_at,
                now_turkey_naive(),
            ])
            reconciled_count += 1

        total_perf = conn.execute("SELECT COUNT(*) FROM gold_bofa_sector_day_start_performance;").fetchone()[0]
        logger.info(f"Successfully updated `gold_bofa_sector_day_start_performance`: {total_perf:,} total recorded sector sessions ({reconciled_count} reconciled in this run).")
        return total_perf

    def backfill_historical_performance(
        self,
        target_dates: Optional[List[Union[str, date]]] = None,
        sectors: Optional[List[str]] = None,
        all_missing: bool = False,
        lookback_months: Optional[int] = None,
        lookback_days: Optional[int] = None,
    ) -> int:
        """Perform zero-lookahead point-in-time forecasting for past missed trading days across sectors and record into `gold_bofa_sector_day_start_performance`.
        
        Args:
            target_dates: Specific list of past trading dates (e.g. ['2026-03-10', '2026-03-18']) to backfill.
            sectors: Optional list of sectors to evaluate (default: all tracked sectors).
            all_missing: If True and target_dates is None, auto-discovers dates in Silver within the configured lookback
                         window that are currently missing from the sector performance ledger.
            lookback_months: Number of trailing months to look back for missing dates (default: from config, usually 2 months).
            lookback_days: Optional number of trailing days to look back (overrides lookback_months if provided).
        """
        conn = self.db.get_connection()
        all_silver_dates = [
            r[0] for r in conn.execute(
                f"SELECT DISTINCT trade_date FROM silver_intraday_sector_window_summary WHERE broker_id = '{self.target_broker}' AND window_name = 'day_start' ORDER BY trade_date ASC;"
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
            if "gold_bofa_sector_day_start_performance" in existing_tables:
                existing_perf_dates = set(r[0] for r in conn.execute("SELECT DISTINCT trade_date FROM gold_bofa_sector_day_start_performance;").fetchall())

            if all_silver_dates:
                latest_date = max(all_silver_dates)
                if eff_days is not None:
                    cutoff_date = latest_date - timedelta(days=eff_days)
                else:
                    cutoff_date = latest_date - relativedelta(months=eff_months)

                candidate_dates = [d for d in all_silver_dates if d >= cutoff_date]
                dates_to_backfill = [d for d in candidate_dates if d not in existing_perf_dates]
                logger.info(
                    f"Auto-discovering missing sector dates within trailing window (>= {cutoff_date}, lookback_months={eff_months}, lookback_days={eff_days}): "
                    f"{len(dates_to_backfill)} missing of {len(candidate_dates)} eligible sessions."
                )
            else:
                dates_to_backfill = []

        if not dates_to_backfill:
            logger.info("No dates to backfill for Sector Day-Start model.")
            return 0

        logger.info(f"Starting zero-lookahead point-in-time sector backfill for {len(dates_to_backfill)} date(s): {dates_to_backfill}...")
        backfilled_forecasts: List[ForecastResult] = []
        for target_d in dates_to_backfill:
            prior_dates = [d for d in all_silver_dates if d < target_d]
            if not prior_dates:
                logger.warning(f"Skipping sector backfill for {target_d}: No prior historical session available.")
                continue
            as_of_d = max(prior_dates)
            logger.info(f"Backfilling sectors for {target_d} with strict zero-leakage cutoff as_of_date={as_of_d}...")
            sec_results = self.forecast_next_day(sectors=sectors, as_of_date=as_of_d)
            for res in sec_results:
                res.forecast_date = target_d
                backfilled_forecasts.append(res)

        saved_count = self.reconcile_and_update_performance_ledger(backfilled_forecasts, sectors=sectors)
        logger.info(f"Point-in-time sector backfill complete. Processed {len(backfilled_forecasts)} sector forecast(s).")
        return saved_count

    def save_backtests_to_gold(
        self,
        backtest_results: Optional[List[ForecastResult]] = None,
        sectors: Optional[List[str]] = None,
    ) -> int:
        """Persist historical sector backtest results joined with actuals into `gold_bofa_sector_day_start_backtests`."""
        if backtest_results is None:
            backtest_results = self.backtest_all_history(sectors=sectors)

        if not backtest_results:
            return 0

        conn = self.db.get_connection()
        logger.info(f"Persisting {len(backtest_results)} sector backtest record(s) to `gold_bofa_sector_day_start_backtests`...")

        # Ensure schema exists
        conn.execute("""
            CREATE TABLE IF NOT EXISTS gold_bofa_sector_day_start_backtests (
                trade_date DATE,
                sector VARCHAR,
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
                model_name VARCHAR,
                model_version VARCHAR,
                calculated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                PRIMARY KEY (trade_date, sector)
            );
        """)

        # Fetch actual sector Window 1 flows from silver_intraday_sector_window_summary
        actuals_df = conn.execute(f"""
            SELECT 
                trade_date,
                sector,
                net_flow_tl AS actual_w1_net_flow_tl
            FROM silver_intraday_sector_window_summary
            WHERE broker_id = '{self.target_broker}' AND window_name = 'day_start';
        """).df()
        actuals_df["trade_date_str"] = actuals_df["trade_date"].astype(str).str.slice(0, 10)
        actuals_map = dict(zip(zip(actuals_df["trade_date_str"], actuals_df["sector"]), actuals_df["actual_w1_net_flow_tl"]))

        for f in backtest_results:
            d = f.forecast_date
            d_str = str(d)[:10]
            sector_name = f.top_predicted_buy_sector or "General"
            dow = d.weekday() + 1 if isinstance(d, date) else 1
            is_mon = (dow == 1)
            actual_flow = float(actuals_map.get((d_str, sector_name), 0.0))
            pred_flow = float(f.predicted_net_flow_tl)
            error_flow = actual_flow - pred_flow
            actual_dir = "BUY" if actual_flow > 0 else ("SELL" if actual_flow < 0 else "NEUTRAL")
            is_hit = bool((pred_flow > 0 and actual_flow > 0) or (pred_flow <= 0 and actual_flow <= 0))
            is_in_ci = bool(f.predicted_flow_lower_90 <= actual_flow <= f.predicted_flow_upper_90)

            conn.execute("""
                INSERT OR REPLACE INTO gold_bofa_sector_day_start_backtests (
                    trade_date, sector, day_of_week, is_monday,
                    predicted_open_net_flow_tl, predicted_open_flow_lower_90, predicted_open_flow_upper_90,
                    actual_open_net_flow_tl, error_open_net_flow_tl,
                    predicted_direction, actual_direction,
                    is_direction_hit, is_inside_90_ci,
                    direction_confidence, predicted_playbook,
                    model_name, model_version, calculated_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?);
            """, [
                f.forecast_date,
                sector_name,
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
                f.model_name,
                f.model_version,
                now_turkey_naive(),
            ])

        saved_count = conn.execute("SELECT COUNT(*) FROM gold_bofa_sector_day_start_backtests;").fetchone()[0]
        logger.info(f"Successfully updated `gold_bofa_sector_day_start_backtests`: {saved_count:,} total records.")
        return saved_count
