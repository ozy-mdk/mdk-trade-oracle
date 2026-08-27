"""StockReactionForecaster: Per-symbol per-window champion model arena and persistence engine.

Registered as Model 3 in the MDK Trading Oracle Gold Layer pipeline.

For each (symbol, window) pair:
  1. forecast_next_window()         -> live T+1 prediction -> gold_bofa_stock_reaction_<w>_forecasts
  2. reconcile_performance_ledger() -> fill actuals       -> gold_bofa_stock_reaction_<w>_performance
  3. backtest_all_history()         -> walk-forward OOS   -> gold_bofa_stock_reaction_<w>_backtests
"""

from datetime import date
from typing import Dict, Optional, Tuple

import pandas as pd
from dateutil.relativedelta import relativedelta

from mdk_trading_oracle.core.config import get_settings
from mdk_trading_oracle.core.db import DuckDBManager
from mdk_trading_oracle.core.logger import get_logger
from mdk_trading_oracle.core.time import now_turkey_naive
from mdk_trading_oracle.models.registry import ModelRegistry
from mdk_trading_oracle.models.stock_reaction.features import StockReactionFeatureExtractor
from mdk_trading_oracle.models.stock_reaction.models import (
    BaseStockReactionModel,
    ReturnThresholdProfile,
    StockReactionAlwaysLongModel,
    StockReactionAlwaysShortModel,
    StockReactionBayesianModel,
    StockReactionForecastResult,
    StockReactionLightGBMModel,
    StockReactionNaivePersistenceModel,
    StockReactionPyMCModel,
    StockReactionRollingMeanModel,
    StockReactionXGBoostModel,
)

logger = get_logger("mdk_oracle.models.stock_reaction.forecaster")

# Map short window key to Gold table suffix and full window name
WINDOW_MAP = {
    "w2": ("w2", "first_reaction"),
    "w3": ("w3", "midday_followup"),
    "w5": ("w5", "closing_session"),
    "first_reaction": ("w2", "first_reaction"),
    "midday_followup": ("w3", "midday_followup"),
    "closing_session": ("w5", "closing_session"),
}


@ModelRegistry.register("stock_reaction_arena")
class StockReactionModelArena:
    """Tournament across active candidate models and benchmark hurdle baselines for a single (symbol, window) pair."""

    HURDLE_PREFIXES = ("Baseline", "Hurdle")

    def __init__(
        self,
        symbol: str,
        window: str,
        thresholds: Optional[ReturnThresholdProfile] = None,
        include_pymc: bool = False,
    ):
        self.symbol = symbol
        self.window = window
        th = thresholds or ReturnThresholdProfile()
        self.candidates: Dict[str, BaseStockReactionModel] = {
            # Benchmark Hurdles (Audit checkpoints — not eligible for live champion deployment)
            "Hurdle 0: Naive Persistence": StockReactionNaivePersistenceModel(symbol, window, th),
            "Hurdle 1: 5-Day Rolling Mean": StockReactionRollingMeanModel(symbol, window, th),
            "Hurdle 2: Always Long (+1)": StockReactionAlwaysLongModel(symbol, window, th),
            "Hurdle 3: Always Short (-1)": StockReactionAlwaysShortModel(symbol, window, th),
            # Active Microstructure Predictive Models (Eligible Champions)
            "Bayesian Ridge Probabilistic": StockReactionBayesianModel(symbol, window, th),
            "LightGBM Non-Linear Ensemble": StockReactionLightGBMModel(symbol, window, th),
            "XGBoost Non-Linear Ensemble": StockReactionXGBoostModel(symbol, window, th),
        }
        if include_pymc:
            self.candidates["PyMC Bayesian GLM (MAP)"] = StockReactionPyMCModel(symbol, window, th, use_map=True)

    def run_tournament(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        min_train_samples: int = 5,
        eval_window_days: Optional[int] = None,
    ) -> Tuple[pd.DataFrame, BaseStockReactionModel]:
        """Walk-forward tournament. Returns (scoreboard_df, champion_model)."""
        scoreboard = []
        for name, model in self.candidates.items():
            try:
                metrics = model.walk_forward_evaluate(
                    X, y,
                    min_train_samples=min_train_samples,
                    eval_window_days=eval_window_days,
                )
                is_hurdle = any(name.startswith(p) for p in self.HURDLE_PREFIXES)
                scoreboard.append({
                    "Model": name,
                    "hit_rate_pct": metrics["hit_rate_pct"],
                    "picp_90_pct": metrics["picp_90_pct"],
                    "mae_pct": metrics["mae_pct"],
                    "rmse_pct": metrics["rmse_pct"],
                    "sample_size": metrics["sample_size"],
                    "is_active": not is_hurdle,
                    "_model_instance": model,
                })
            except Exception as exc:
                logger.warning(f"[{self.symbol}/{self.window}] Model '{name}' failed tournament: {exc}")

        if not scoreboard:
            logger.warning(f"[{self.symbol}/{self.window}] All models failed — using Bayesian Ridge.")
            fallback = StockReactionBayesianModel(self.symbol, self.window)
            fallback.fit(X, y)
            return pd.DataFrame(), fallback

        # Calculate hurdle majority benchmark
        hurdle_hit_rates = [r["hit_rate_pct"] for r in scoreboard if not r["is_active"]]
        majority_hit_rate = max(hurdle_hit_rates) if hurdle_hit_rates else 50.0

        for r in scoreboard:
            r["directional_alpha"] = r["hit_rate_pct"] - majority_hit_rate

        df_scores = pd.DataFrame(scoreboard)
        active_scores = df_scores[df_scores["is_active"]]

        if not active_scores.empty:
            # Crown champion strictly among active models, prioritizing Directional Alpha & Hit Rate
            sorted_active = active_scores.sort_values(
                by=["directional_alpha", "hit_rate_pct", "picp_90_pct", "rmse_pct"],
                ascending=[False, False, False, True],
            ).reset_index(drop=True)
            champion_row = sorted_active.iloc[0]
        else:
            champion_row = df_scores.sort_values(by=["hit_rate_pct"], ascending=False).iloc[0]

        champion: BaseStockReactionModel = champion_row["_model_instance"]
        champion.fit(X, y)  # Refit champion on full training set
        beats_hurdle = bool(champion_row.get("directional_alpha", 0.0) >= 0.0)
        setattr(champion, "_beats_hurdle", beats_hurdle)
        setattr(champion, "_majority_hit_rate", majority_hit_rate)
        setattr(champion, "_directional_alpha", float(champion_row.get("directional_alpha", 0.0)))

        logger.info(
            f"[{self.symbol}/{self.window}] Champion: '{champion_row['Model']}' "
            f"(Hit: {champion_row['hit_rate_pct']:.1f}% | Alpha vs Hurdle: {champion_row.get('directional_alpha', 0.0):+.1f}% "
            f"| PICP90: {champion_row['picp_90_pct']:.1f}% | MAE: {champion_row['mae_pct']:.3f}% | n={champion_row['sample_size']})"
        )
        return df_scores, champion


@ModelRegistry.register("stock_reaction_forecaster")
class StockReactionForecaster:
    """Production forecaster for a single (symbol, window) combination.

    Manages feature extraction, model arena, and three-table Gold persistence:
      - gold_bofa_stock_reaction_<window>_forecasts    (live T+1)
      - gold_bofa_stock_reaction_<window>_performance  (audited historical)
      - gold_bofa_stock_reaction_<window>_backtests    (walk-forward simulation)
    """

    def __init__(
        self,
        symbol: str,
        window: str,
        db: Optional[DuckDBManager] = None,
        lookback_months: Optional[int] = None,
        model_type: str = "auto",
        include_pymc: bool = False,
        filter_weak_regimes: Optional[bool] = None,
    ):
        self.symbol = symbol.upper()
        self.window = window
        self._table_key, self._window_name = WINDOW_MAP.get(window, ("w2", "first_reaction"))
        self.db = db or DuckDBManager()
        self.settings = get_settings()
        cfg = self.settings.get_model_config("stock_reaction") or {}
        self.lookback_months = lookback_months or cfg.get("lookback_months", 12)
        self.model_type = model_type or cfg.get("model_type", "auto")
        self.include_pymc = include_pymc or cfg.get("include_pymc_arena", False)
        self.filter_weak_regimes = (
            filter_weak_regimes if filter_weak_regimes is not None else cfg.get("filter_weak_regimes", True)
        )
        self.extractor = StockReactionFeatureExtractor(
            symbol=self.symbol,
            db=self.db,
            lookback_months=self.lookback_months,
        )
        self._thresholds: Optional[ReturnThresholdProfile] = None

    def _load_thresholds(self) -> ReturnThresholdProfile:
        """Load empirical return percentile thresholds from Silver for this (symbol, window)."""
        if self._thresholds is not None:
            return self._thresholds
        try:
            conn = self.db.get_connection()
            row = conn.execute(f"""
                SELECT up_p25_pct, up_p50_pct, up_p85_pct,
                       down_p25_pct, down_p50_pct, down_p85_pct,
                       up_session_count, down_session_count, total_sessions
                FROM silver_stock_reaction_thresholds
                WHERE symbol = '{self.symbol}' AND window_name = '{self._window_name}'
                LIMIT 1;
            """).fetchone()
            if row:
                self._thresholds = ReturnThresholdProfile(
                    up_p25_pct=row[0], up_p50_pct=row[1], up_p85_pct=row[2],
                    down_p25_pct=row[3], down_p50_pct=row[4], down_p85_pct=row[5],
                    up_session_count=row[6], down_session_count=row[7], total_sessions=row[8],
                )
                logger.debug(
                    f"[{self.symbol}/{self._table_key}] Loaded thresholds: "
                    f"UP P85={row[2]:.3f}% | DOWN P85={row[5]:.3f}%"
                )
                return self._thresholds
        except Exception as exc:
            logger.warning(f"[{self.symbol}] Could not load thresholds: {exc} — using defaults.")
        self._thresholds = ReturnThresholdProfile()
        return self._thresholds

    def _prepare_training_data(
        self,
        as_of_date: Optional[date] = None,
        lookback_months: Optional[int] = None,
        filter_weak_regimes: Optional[bool] = None,
    ) -> Tuple[pd.DataFrame, pd.Series]:
        """Fetch feature matrix and target series for training, strictly ending at as_of_date.

        If filter_weak_regimes is True, eliminates noise sessions where neither BofA nor
        the domestic Big Players took a decisive day-start positioning (|strength| < 2.0).
        """
        lb = lookback_months or self.lookback_months
        end_date = as_of_date
        start_date = (end_date - relativedelta(months=lb)) if end_date else None
        should_filter = self.filter_weak_regimes if filter_weak_regimes is None else filter_weak_regimes

        df = self.extractor.extract_features(start_date=start_date, end_date=end_date)
        target_col = self.extractor.get_target_column(self.window)
        feat_cols = self.extractor.get_feature_columns()

        # Drop rows with missing target
        df_pd = df.to_pandas()
        df_clean = df_pd.dropna(subset=[target_col]).copy()

        # Filter weak/noise sessions if enabled
        if should_filter and "is_institutional_active_day" in df_clean.columns:
            active_mask = df_clean["is_institutional_active_day"].astype(bool)
            if active_mask.sum() >= 5:
                kept_count = active_mask.sum()
                dropped_count = len(df_clean) - kept_count
                df_clean = df_clean[active_mask].copy()
                logger.debug(
                    f"[{self.symbol}/{self._table_key}] Filtered weak regimes: "
                    f"kept {kept_count} active sessions, dropped {dropped_count} noise sessions."
                )
            else:
                logger.warning(
                    f"[{self.symbol}/{self._table_key}] Insufficient active sessions ({active_mask.sum()}) "
                    f"for filtering — using full {len(df_clean)} sessions."
                )

        # Fill feature NaNs with 0.0 (coalesce)
        for col in feat_cols:
            if col in df_clean.columns:
                df_clean[col] = df_clean[col].fillna(0.0)

        available_feats = [c for c in feat_cols if c in df_clean.columns]
        X = df_clean[["trade_date"] + available_feats].copy()
        y = df_clean[target_col].astype(float)
        return X, y

    def _run_arena(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        as_of_date: Optional[date] = None,
    ) -> BaseStockReactionModel:
        """Run model tournament and return the champion (fitted on full training data)."""
        thresholds = self._load_thresholds()
        cfg = self.settings.get_model_config("stock_reaction") or {}
        eval_window_days = cfg.get("eval_window_days", 10)
        min_burn_in = cfg.get("min_burn_in_days", 5)

        if self.model_type == "auto":
            arena = StockReactionModelArena(
                symbol=self.symbol, window=self.window,
                thresholds=thresholds, include_pymc=self.include_pymc,
            )
            _, champion = arena.run_tournament(
                X, y,
                min_train_samples=min_burn_in,
                eval_window_days=eval_window_days,
            )
            return champion
        else:
            # Specific model type requested — train directly
            model_map = {
                "lightgbm": StockReactionLightGBMModel,
                "xgboost": StockReactionXGBoostModel,
                "bayesian": StockReactionBayesianModel,
                "naive": StockReactionNaivePersistenceModel,
                "rolling_mean": StockReactionRollingMeanModel,
            }
            ModelClass = model_map.get(self.model_type.lower(), StockReactionBayesianModel)
            model = ModelClass(symbol=self.symbol, window=self.window, thresholds=thresholds)
            model.fit(X, y)
            return model

    def forecast_next_window(
        self,
        forecast_date: Optional[date] = None,
    ) -> Optional[StockReactionForecastResult]:
        """Generate live T+1 forecast for the upcoming session.

        Uses all data up to and including yesterday (T-1) as training data.
        The W1 (day_start) features for forecast_date are used as inference input.
        """
        conn = self.db.get_connection()

        # Determine latest settled date
        latest_date = conn.execute("""
            SELECT MAX(trade_date) FROM silver_intraday_broker_window_summary
            WHERE window_name = 'day_start' AND broker_id = 'MLB';
        """).fetchone()[0]

        if latest_date is None:
            logger.warning(f"[{self.symbol}] No settled W1 data found — cannot forecast.")
            return None

        if isinstance(latest_date, str):
            latest_date = date.fromisoformat(latest_date)

        target_date = forecast_date or latest_date

        logger.info(
            f"[{self.symbol}/{self._table_key}] Forecasting for {target_date} "
            f"(W1 data settled through {latest_date})..."
        )

        # Training data: all historical rows with target available
        X_train, y_train = self._prepare_training_data(as_of_date=latest_date)
        if len(X_train) < 5:
            logger.warning(f"[{self.symbol}] Insufficient training data ({len(X_train)} rows) — skipping.")
            return None

        champion = self._run_arena(X_train, y_train)

        # Inference data: today's W1 features (target cols will be NULL)
        df_inference = self.extractor.extract_features(
            start_date=target_date, end_date=target_date
        )
        if df_inference.is_empty():
            logger.warning(f"[{self.symbol}] No W1 features found for {target_date} — skipping.")
            return None

        X_infer = df_inference.to_pandas()
        feat_cols = self.extractor.get_feature_columns()
        available_feats = [c for c in feat_cols if c in X_infer.columns]
        result = champion.predict(X_infer[["trade_date"] + available_feats])

        # Check day-start institutional conviction
        is_today_active = True
        if "is_institutional_active_day" in X_infer.columns:
            is_today_active = bool(X_infer["is_institutional_active_day"].iloc[0])

        # If day-start positioning is weak (neither BofA nor Big Players pushed size),
        # or if champion active model could not beat the baseline hurdle on the trailing evaluation window,
        # flag the execution playbook as NEUTRAL_WAIT to protect trader capital
        if not is_today_active:
            result.predicted_playbook = "NEUTRAL_WAIT"
            result.direction_confidence = round(min(result.direction_confidence * 0.5, 0.35), 4)
            logger.info(
                f"[{self.symbol}/{self._table_key}] Day-start institutional flow is weak — "
                f"assigned NEUTRAL_WAIT playbook."
            )
        elif not getattr(champion, "_beats_hurdle", True):
            result.predicted_playbook = "NEUTRAL_WAIT"
            result.direction_confidence = round(min(result.direction_confidence * 0.5, 0.35), 4)

        # Persist to forecasts table (replace all rows = always <= 30 rows)
        now_ts = now_turkey_naive()
        conn.execute(f"""
            DELETE FROM gold_bofa_stock_reaction_{self._table_key}_forecasts
            WHERE symbol = '{self.symbol}';
        """)
        conn.execute(f"""
            INSERT INTO gold_bofa_stock_reaction_{self._table_key}_forecasts
            (forecast_date, symbol, window_name,
             predicted_return_pct, predicted_return_lower_90, predicted_return_upper_90,
             predicted_direction, direction_confidence, predicted_playbook,
             bofa_w1_direction, bofa_w1_net_flow_tl, bofa_w1_volume_share,
             model_name, model_version, generated_at)
            VALUES (
                '{result.forecast_date}', '{result.symbol}', '{result.window_name}',
                {result.predicted_return_pct}, {result.predicted_return_lower_90},
                {result.predicted_return_upper_90}, '{result.predicted_direction}',
                {result.direction_confidence}, '{result.predicted_playbook}',
                '{result.bofa_w1_direction}', {result.bofa_w1_net_flow_tl},
                {result.bofa_w1_volume_share}, '{result.model_name}', '{result.model_version}',
                '{now_ts}'
            );
        """)
        logger.info(
            f"[{self.symbol}/{self._table_key}] Forecast saved: {result.predicted_direction} "
            f"({result.predicted_return_pct:+.2f}%) | Playbook: {result.predicted_playbook} | Model: {result.model_name}"
        )
        return result

    def reconcile_performance_ledger(self) -> int:
        """Reconcile past forecasts with actual market data and update the performance table.

        Returns: number of reconciled rows.
        """
        conn = self.db.get_connection()
        now_ts = now_turkey_naive()

        # Find forecasts that have not yet been reconciled
        unreconciled = conn.execute(f"""
            SELECT forecast_date, symbol, predicted_return_pct,
                   predicted_return_lower_90, predicted_return_upper_90,
                   predicted_direction, direction_confidence, predicted_playbook,
                   bofa_w1_direction, bofa_w1_net_flow_tl, bofa_w1_volume_share,
                   model_name, model_version
            FROM gold_bofa_stock_reaction_{self._table_key}_forecasts
            WHERE symbol = '{self.symbol}';
        """).fetchall()

        if not unreconciled:
            return 0

        # Upsert unreconciled rows into performance table
        for row in unreconciled:
            f_date = row[0]
            conn.execute(f"""
                INSERT OR IGNORE INTO gold_bofa_stock_reaction_{self._table_key}_performance
                (trade_date, symbol, window_name,
                 predicted_return_pct, predicted_return_lower_90, predicted_return_upper_90,
                 predicted_direction, direction_confidence, predicted_playbook,
                 bofa_w1_direction, bofa_w1_net_flow_tl, bofa_w1_volume_share,
                 model_name, model_version, created_at)
                VALUES (
                    '{f_date}', '{row[1]}', '{self._window_name}',
                    {row[2]}, {row[3]}, {row[4]},
                    '{row[5]}', {row[6]}, '{row[7]}',
                    '{row[8]}', {row[9]}, {row[10]},
                    '{row[11]}', '{row[12]}', '{now_ts}'
                );
            """)

        # Fetch actual returns from Silver layer for rows with NULL actual_return_pct
        pending = conn.execute(f"""
            SELECT trade_date, symbol, predicted_return_pct,
                   predicted_return_lower_90, predicted_return_upper_90
            FROM gold_bofa_stock_reaction_{self._table_key}_performance
            WHERE symbol = '{self.symbol}' AND actual_return_pct IS NULL;
        """).fetchall()

        reconciled = 0
        target_col = self.extractor.get_target_column(self.window)

        for row in pending:
            trade_date = row[0]
            try:
                # Query actual return from Silver
                df_act = self.extractor.extract_features(start_date=trade_date, end_date=trade_date)
                if df_act.is_empty() or target_col not in df_act.columns:
                    continue

                act_series = df_act[target_col].drop_nulls()
                if len(act_series) == 0:
                    continue

                actual_return = float(act_series[0])
                pred_return = float(row[2])
                error = pred_return - actual_return
                abs_error = abs(error)
                direction_hit = (pred_return > 0) == (actual_return > 0)
                inside_ci = float(row[3]) <= actual_return <= float(row[4])
                actual_dir = "RALLY" if actual_return > 0 else ("DECLINE" if actual_return < 0 else "NEUTRAL")

                conn.execute(f"""
                    UPDATE gold_bofa_stock_reaction_{self._table_key}_performance
                    SET
                        actual_return_pct = {actual_return},
                        actual_direction = '{actual_dir}',
                        error_return_pct = {error},
                        absolute_error_pct = {abs_error},
                        is_direction_hit = {str(direction_hit).upper()},
                        is_inside_90_ci = {str(inside_ci).upper()},
                        realized_at = '{now_turkey_naive()}'
                    WHERE trade_date = '{trade_date}' AND symbol = '{self.symbol}';
                """)
                reconciled += 1
            except Exception as exc:
                logger.warning(f"[{self.symbol}/{self._table_key}] Reconcile error for {trade_date}: {exc}")

        if reconciled > 0:
            logger.info(f"[{self.symbol}/{self._table_key}] Reconciled {reconciled} performance rows.")
        return reconciled

    def backtest_all_history(self, lookback_months: Optional[int] = None) -> int:
        """Walk-forward OOS simulation across all available history.

        For each date T in the historical window (expanding window):
          - Train on all data strictly before T (filtering weak regimes if enabled)
          - Predict T's window return
          - Compare against realized actual
          - Persist to backtests table

        Returns: number of backtest rows written.
        """
        conn = self.db.get_connection()
        # Fetch full un-filtered historical matrix so all test dates are evaluated
        X_all, y_all = self._prepare_training_data(lookback_months=lookback_months, filter_weak_regimes=False)

        if len(X_all) < 10:
            logger.warning(f"[{self.symbol}/{self._table_key}] Insufficient history for backtest — skipping.")
            return 0

        cfg = self.settings.get_model_config("stock_reaction") or {}
        min_burn_in = cfg.get("min_burn_in_days", 5)
        thresholds = self._load_thresholds()

        # Extract features matrix to check active regime flags
        df_raw = self.extractor.extract_features()
        df_raw_pd = df_raw.to_pandas()
        active_map = {}
        if "is_institutional_active_day" in df_raw_pd.columns:
            active_map = dict(zip(df_raw_pd["trade_date"].astype(str), df_raw_pd["is_institutional_active_day"]))

        backtest_rows = []
        feat_cols = [c for c in X_all.columns if c.startswith("feat_")]

        for i in range(min_burn_in, len(X_all)):
            X_train_full = X_all.iloc[:i]
            y_train_full = y_all.iloc[:i]
            X_test = X_all.iloc[i:i+1]
            y_true = float(y_all.iloc[i])
            trade_date_val = X_test.iloc[0].get("trade_date") or X_all.index[i]
            trade_date_str = str(trade_date_val)

            # Apply weak regime filtering to training data slice
            if self.filter_weak_regimes and active_map:
                train_dates = X_train_full["trade_date"].astype(str)
                is_active_train = train_dates.map(lambda d: active_map.get(d, True))
                if is_active_train.sum() >= min_burn_in:
                    X_train = X_train_full[is_active_train]
                    y_train = y_train_full.loc[X_train.index]
                else:
                    X_train = X_train_full
                    y_train = y_train_full
            else:
                X_train = X_train_full
                y_train = y_train_full

            try:
                arena = StockReactionModelArena(
                    symbol=self.symbol, window=self.window,
                    thresholds=thresholds, include_pymc=False,
                )
                _, champion = arena.run_tournament(X_train, y_train, min_train_samples=min_burn_in)
                result = champion.predict(X_test[["trade_date"] + feat_cols] if "trade_date" in X_test.columns else X_test)

                is_test_day_active = active_map.get(trade_date_str, True)
                if not is_test_day_active:
                    result.predicted_playbook = "NEUTRAL_WAIT"
                    result.direction_confidence = round(min(result.direction_confidence * 0.5, 0.35), 4)
                elif not getattr(champion, "_beats_hurdle", True):
                    result.predicted_playbook = "NEUTRAL_WAIT"
                    result.direction_confidence = round(min(result.direction_confidence * 0.5, 0.35), 4)

                pred = result.predicted_return_pct
                low90 = result.predicted_return_lower_90
                high90 = result.predicted_return_upper_90
                error = pred - y_true
                dir_hit = (pred > 0) == (y_true > 0)
                inside_ci = low90 <= y_true <= high90
                actual_dir = "RALLY" if y_true > 0 else ("DECLINE" if y_true < 0 else "NEUTRAL")

                backtest_rows.append({
                    "trade_date": trade_date_str,
                    "symbol": self.symbol,
                    "window_name": self._window_name,
                    "predicted_return_pct": round(pred, 4),
                    "predicted_return_lower_90": round(low90, 4),
                    "predicted_return_upper_90": round(high90, 4),
                    "predicted_direction": result.predicted_direction,
                    "direction_confidence": round(result.direction_confidence, 4),
                    "predicted_playbook": result.predicted_playbook,
                    "actual_return_pct": round(y_true, 4),
                    "actual_direction": actual_dir,
                    "error_return_pct": round(error, 4),
                    "absolute_error_pct": round(abs(error), 4),
                    "is_direction_hit": dir_hit,
                    "is_inside_90_ci": inside_ci,
                    "training_start_date": str(X_train.iloc[0].get("trade_date", "")),
                    "training_end_date": str(X_train.iloc[-1].get("trade_date", "")),
                    "training_samples": len(X_train),
                    "model_name": result.model_name,
                    "model_version": result.model_version,
                })
            except Exception as exc:
                logger.debug(f"[{self.symbol}/{self._table_key}] Backtest step {i} failed: {exc}")
                continue

        if not backtest_rows:
            return 0

        import polars as pl
        pl_backtest = pl.DataFrame(backtest_rows)
        conn.register("df_stock_rxn_backtest_temp", pl_backtest)
        conn.execute(f"""
            INSERT OR REPLACE INTO gold_bofa_stock_reaction_{self._table_key}_backtests
            SELECT *, CURRENT_TIMESTAMP AS calculated_at
            FROM df_stock_rxn_backtest_temp;
        """)
        conn.unregister("df_stock_rxn_backtest_temp")

        logger.info(
            f"[{self.symbol}/{self._table_key}] Backtest complete: "
            f"{len(backtest_rows)} rows | "
            f"Hit Rate: {sum(r['is_direction_hit'] for r in backtest_rows)/len(backtest_rows)*100:.1f}%"
        )
        return len(backtest_rows)
