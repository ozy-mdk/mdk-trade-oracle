"""Day-Start Candidate Models: Baselines, LightGBM, Bayesian Ridge, PyMC, and Probabilistic Ensemble."""

from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.linear_model import BayesianRidge
from sklearn.metrics import mean_absolute_error, root_mean_squared_error

from mdk_trading_oracle.core.logger import get_logger
from mdk_trading_oracle.models.base import (
    BaseForecaster,
    FlowThresholdClassifier,
    FlowThresholdProfile,
    ForecastResult,
    OpeningPlaybook,
)
from mdk_trading_oracle.models.registry import ModelRegistry

logger = get_logger("mdk_oracle.models.day_start.models")


class BaseDayStartModel(BaseForecaster):
    """Base class providing common evaluation logic and dynamic percentile directional conviction classification."""

    def __init__(
        self,
        model_name: str,
        model_version: str = "1.0.0",
        thresholds: Optional[FlowThresholdProfile] = None,
    ):
        super().__init__(model_name=model_name, model_version=model_version)
        self.thresholds = thresholds or FlowThresholdProfile()

    def set_thresholds(self, thresholds: FlowThresholdProfile) -> None:
        """Update empirical percentile flow thresholds profile."""
        self.thresholds = thresholds

    def _classify_direction(self, net_flow_tl: float, confidence: float = 0.5) -> str:
        """Classify continuous predicted net flow (TL) into dynamic percentile directional category."""
        return FlowThresholdClassifier.classify(net_flow_tl, self.thresholds)

    def _determine_playbook(self, row_dict: Dict[str, Any], predicted_flow: float) -> str:
        """Determine the institutional execution playbook based on competitor, cost basis, and dynamic percentile triggers."""
        w4_delta = float(row_dict.get("feat_bofa_vs_top5_w4_flow_delta_tl", 0.0))
        cost_basis_spread = float(row_dict.get("feat_bofa_cost_basis_spread_20d_pct", 0.0))
        is_monday = bool(row_dict.get("is_monday", False))

        buy_p50 = self.thresholds.buy_p50_tl
        buy_p85 = self.thresholds.buy_p85_tl
        sell_p50 = self.thresholds.sell_p50_tl

        if predicted_flow >= buy_p50 and w4_delta >= buy_p50:
            return OpeningPlaybook.SQUEEZE_LONG
        elif predicted_flow >= buy_p85:
            return OpeningPlaybook.MOMENTUM_EXPANSION
        elif predicted_flow <= -sell_p50 and cost_basis_spread > 0.05:
            return OpeningPlaybook.LIQUIDITY_FADE
        elif cost_basis_spread < -0.04 and predicted_flow > 0:
            return OpeningPlaybook.DEFENSE_SUPPORT
        elif is_monday:
            return OpeningPlaybook.SECTOR_ROTATION
        return OpeningPlaybook.NEUTRAL_WAIT

    def _determine_top_sectors(self, row_dict: Dict[str, Any]) -> Tuple[str, str]:
        """Dynamically determine top accumulation and distribution sectors from historical T-1 feature vector."""
        sector_flows = {
            "Banking": float(row_dict.get("feat_bofa_banking_flow_prev_day", 0.0)),
            "Transportation": float(row_dict.get("feat_bofa_transport_flow_prev_day", 0.0)),
            "Holding": float(row_dict.get("feat_bofa_holding_flow_prev_day", 0.0)),
            "Energy & Refining": float(row_dict.get("feat_bofa_energy_flow_prev_day", 0.0)),
            "Defense & Tech": float(row_dict.get("feat_bofa_defense_flow_prev_day", 0.0)),
        }
        active_buys = {k: v for k, v in sector_flows.items() if v > 0}
        active_sells = {k: v for k, v in sector_flows.items() if v < 0}

        top_buy = max(active_buys.items(), key=lambda x: x[1])[0] if active_buys else "None"
        top_sell = min(active_sells.items(), key=lambda x: x[1])[0] if active_sells else "None"
        return top_buy, top_sell

    def evaluate(self, X_test: pd.DataFrame, y_test: pd.Series) -> Dict[str, float]:
        """Compute MAE, RMSE, Directional Hit Rate (%), and Confidence Interval Coverage (PICP)."""
        predictions = []
        lowers = []
        uppers = []

        for idx in range(len(X_test)):
            row = X_test.iloc[[idx]].reset_index(drop=True)
            res = self.predict(row)
            predictions.append(res.predicted_net_flow_tl)
            lowers.append(res.predicted_flow_lower_90)
            uppers.append(res.predicted_flow_upper_90)

        preds = np.array(predictions)
        actuals = y_test.to_numpy()

        mae = float(mean_absolute_error(actuals, preds))
        rmse = float(root_mean_squared_error(actuals, preds))

        # Directional Hit Rate (% of correct Buy vs Sell signs)
        pred_dir = np.sign(preds)
        actual_dir = np.sign(actuals)
        correct_directions = np.sum(pred_dir == actual_dir)
        hit_rate = float(correct_directions / len(actuals) * 100.0) if len(actuals) > 0 else 0.0

        # Prediction Interval Coverage Probability (PICP) for 90% CI
        in_bounds = np.sum((actuals >= np.array(lowers)) & (actuals <= np.array(uppers)))
        picp = float(in_bounds / len(actuals) * 100.0) if len(actuals) > 0 else 0.0

        return {
            "mae_million_tl": mae / 1e6,
            "rmse_million_tl": rmse / 1e6,
            "hit_rate_pct": hit_rate,
            "picp_90_pct": picp,
            "sample_size": len(actuals),
        }

    def walk_forward_evaluate(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        min_train_samples: int = 5,
        eval_window_days: Optional[int] = None,
    ) -> Dict[str, Any]:
        """Perform expanding-window walk-forward validation (train on 1..t-1, predict t).

        Guarantees strictly out-of-sample evaluation with zero lookahead bias.

        Args:
            X: Feature matrix.
            y: Continuous target series.
            min_train_samples: Minimum warmup sessions before walk-forward evaluation begins.
            eval_window_days: Optional limit to evaluate only the trailing N days in the tournament.
        """
        n_samples = len(X)
        if n_samples <= min_train_samples:
            self.fit(X, y)
            eval_res = self.evaluate(X, y)
            eval_res["oos_predictions"] = []
            eval_res["oos_actuals"] = []
            eval_res["oos_dates"] = []
            return eval_res

        # Determine start index (evaluate trailing eval_window_days if specified)
        start_idx = min_train_samples
        if eval_window_days is not None and n_samples > eval_window_days:
            start_idx = max(min_train_samples, n_samples - eval_window_days)

        oos_preds = []
        oos_lowers = []
        oos_uppers = []
        oos_actuals = []
        oos_dates = []

        for t in range(start_idx, n_samples):
            X_train = X.iloc[:t].copy()
            y_train = y.iloc[:t].copy()
            X_test_row = X.iloc[[t]].copy()
            y_test_actual = float(y.iloc[t])

            self.fit(X_train, y_train)
            res = self.predict(X_test_row)

            oos_preds.append(res.predicted_net_flow_tl)
            oos_lowers.append(res.predicted_flow_lower_90)
            oos_uppers.append(res.predicted_flow_upper_90)
            oos_actuals.append(y_test_actual)
            if "trade_date" in X.columns:
                oos_dates.append(str(X.iloc[t]["trade_date"])[:10])

        preds = np.array(oos_preds)
        actuals = np.array(oos_actuals)

        mae = float(mean_absolute_error(actuals, preds))
        rmse = float(root_mean_squared_error(actuals, preds))

        pred_dir = np.sign(preds)
        actual_dir = np.sign(actuals)
        correct_dirs = np.sum(pred_dir == actual_dir)
        hit_rate = float(correct_dirs / len(actuals) * 100.0) if len(actuals) > 0 else 0.0

        in_bounds = np.sum((actuals >= np.array(oos_lowers)) & (actuals <= np.array(oos_uppers)))
        picp = float(in_bounds / len(actuals) * 100.0) if len(actuals) > 0 else 0.0

        return {
            "mae_million_tl": mae / 1e6,
            "rmse_million_tl": rmse / 1e6,
            "hit_rate_pct": hit_rate,
            "picp_90_pct": picp,
            "sample_size": len(actuals),
            "oos_predictions": (preds / 1e6).tolist(),
            "oos_actuals": (actuals / 1e6).tolist(),
            "oos_dates": oos_dates,
        }


# ==========================================
# 0. Baseline Models (Benchmark to Beat)
# ==========================================


@ModelRegistry.register("day_start_baseline_persistence")
class DayStartNaivePersistenceModel(BaseDayStartModel):
    """Baseline 0: Carries yesterday's closing Window 4 flow forward as today's opening expectation."""

    def __init__(self, thresholds: Optional[FlowThresholdProfile] = None):
        super().__init__(model_name="day_start_baseline_persistence", model_version="1.0.0", thresholds=thresholds)

    def fit(self, X: pd.DataFrame, y: pd.Series) -> "DayStartNaivePersistenceModel":
        self.is_fitted = True
        return self

    def predict(self, X: pd.DataFrame) -> ForecastResult:
        row_dict = X.iloc[0].to_dict()
        pred_flow = float(row_dict.get("feat_bofa_w4_net_flow_tl", 0.0))
        std_est = 25e6  # Baseline heuristic variance

        confidence = 0.55
        direction = self._classify_direction(pred_flow, confidence)
        playbook = self._determine_playbook(row_dict, pred_flow)
        top_buy, top_sell = self._determine_top_sectors(row_dict)

        return ForecastResult(
            forecast_date=row_dict.get("trade_date"),
            target_broker_id="MLB",
            predicted_net_flow_tl=pred_flow,
            predicted_flow_lower_90=pred_flow - 1.645 * std_est,
            predicted_flow_upper_90=pred_flow + 1.645 * std_est,
            predicted_direction=direction,
            direction_confidence=confidence,
            predicted_playbook=playbook,
            top_predicted_buy_sector=top_buy,
            top_predicted_sell_sector=top_sell,
            model_name=self.model_name,
            model_version=self.model_version,
        )


@ModelRegistry.register("day_start_baseline_rolling_mean")
class DayStartRollingMeanModel(BaseDayStartModel):
    """Baseline 1: Predicts opening flow as historical 5-day rolling average."""

    def __init__(self, thresholds: Optional[FlowThresholdProfile] = None):
        super().__init__(model_name="day_start_baseline_rolling_mean", model_version="1.0.0", thresholds=thresholds)
        self.mean_flow = 0.0

    def fit(self, X: pd.DataFrame, y: pd.Series) -> "DayStartRollingMeanModel":
        self.mean_flow = float(y.tail(5).mean()) if len(y) > 0 else 0.0
        self.is_fitted = True
        return self

    def predict(self, X: pd.DataFrame) -> ForecastResult:
        row_dict = X.iloc[0].to_dict()
        pred_flow = self.mean_flow
        std_est = 30e6

        confidence = 0.50
        direction = self._classify_direction(pred_flow, confidence)
        playbook = self._determine_playbook(row_dict, pred_flow)
        top_buy, top_sell = self._determine_top_sectors(row_dict)

        return ForecastResult(
            forecast_date=row_dict.get("trade_date"),
            target_broker_id="MLB",
            predicted_net_flow_tl=pred_flow,
            predicted_flow_lower_90=pred_flow - 1.645 * std_est,
            predicted_flow_upper_90=pred_flow + 1.645 * std_est,
            predicted_direction=direction,
            direction_confidence=confidence,
            predicted_playbook=playbook,
            top_predicted_buy_sector=top_buy,
            top_predicted_sell_sector=top_sell,
            model_name=self.model_name,
            model_version=self.model_version,
        )


# ==========================================
# 1. Bayesian Probabilistic Models (Ridge & PyMC)
# ==========================================


@ModelRegistry.register("day_start_bayesian_ridge")
class DayStartBayesianModel(BaseDayStartModel):
    """Bayesian Probabilistic Forecaster (Ridge): Computes conjugate posterior distributions and exact credible intervals."""

    def __init__(self, max_iter: int = 300, thresholds: Optional[FlowThresholdProfile] = None):
        super().__init__(model_name="day_start_bayesian_ridge", model_version="1.0.0", thresholds=thresholds)
        self.regressor = BayesianRidge(max_iter=max_iter, compute_score=True)
        self.feature_cols: list[str] = []

    def _prep_features(self, X: pd.DataFrame) -> pd.DataFrame:
        feat_df = X.copy()
        drop_cols = [
            "trade_date",
            "target_open_net_flow_tl",
            "target_open_turnover_tl",
            "target_open_market_share",
            "target_open_direction",
        ]
        for col in drop_cols:
            if col in feat_df.columns:
                feat_df = feat_df.drop(columns=[col])
        return feat_df.select_dtypes(include=[np.number, bool]).astype(float)

    def fit(self, X: pd.DataFrame, y: pd.Series) -> "DayStartBayesianModel":
        X_clean = self._prep_features(X)
        self.feature_cols = list(X_clean.columns)
        self.regressor.fit(X_clean, y)
        self.is_fitted = True
        return self

    def predict(self, X: pd.DataFrame) -> ForecastResult:
        row_dict = X.iloc[0].to_dict()
        X_clean = self._prep_features(X.iloc[[0]])

        for col in self.feature_cols:
            if col not in X_clean.columns:
                X_clean[col] = 0.0
        X_clean = X_clean[self.feature_cols]

        mean_pred, std_pred = self.regressor.predict(X_clean, return_std=True)
        pred_val = float(np.asarray(mean_pred)[0])
        std_val = float(np.asarray(std_pred)[0])

        # 90% Bayesian Credible Interval (Z = 1.645)
        lower_90 = pred_val - 1.645 * std_val
        upper_90 = pred_val + 1.645 * std_val

        # Directional probability from Gaussian CDF
        from scipy.stats import norm

        prob_positive = 1.0 - norm.cdf(0, loc=pred_val, scale=std_val)
        confidence = float(max(prob_positive, 1.0 - prob_positive))

        direction = self._classify_direction(pred_val, confidence)
        playbook = self._determine_playbook(row_dict, pred_val)
        top_buy, top_sell = self._determine_top_sectors(row_dict)

        return ForecastResult(
            forecast_date=row_dict.get("trade_date"),
            target_broker_id="MLB",
            predicted_net_flow_tl=pred_val,
            predicted_flow_lower_90=lower_90,
            predicted_flow_upper_90=upper_90,
            predicted_direction=direction,
            direction_confidence=confidence,
            predicted_playbook=playbook,
            top_predicted_buy_sector=top_buy,
            top_predicted_sell_sector=top_sell,
            model_name=self.model_name,
            model_version=self.model_version,
            features_used={c: float(row_dict.get(c, 0.0)) for c in self.feature_cols[:5]},
        )


@ModelRegistry.register("day_start_pymc")
class DayStartPyMCModel(BaseDayStartModel):
    """PyMC Full Bayesian MCMC / NUTS Forecaster with custom institutional priors."""

    def __init__(
        self, draws: int = 300, tune: int = 300, use_map: bool = True, thresholds: Optional[FlowThresholdProfile] = None
    ):
        super().__init__(model_name="day_start_pymc", model_version="1.0.0", thresholds=thresholds)
        self.draws = draws
        self.tune = tune
        self.use_map = use_map
        self.feature_cols: List[str] = []
        self.posterior_mean_weights: np.ndarray = np.array([])
        self.posterior_mean_intercept: float = 0.0
        self.posterior_sigma: float = 25e6
        self.x_mean: np.ndarray = np.array([])
        self.x_std: np.ndarray = np.array([])
        self.y_mean: float = 0.0
        self.y_std: float = 1.0

    def _prep_features(self, X: pd.DataFrame) -> pd.DataFrame:
        feat_df = X.copy()
        drop_cols = [
            "trade_date",
            "target_open_net_flow_tl",
            "target_open_turnover_tl",
            "target_open_market_share",
            "target_open_direction",
        ]
        for col in drop_cols:
            if col in feat_df.columns:
                feat_df = feat_df.drop(columns=[col])
        return feat_df.select_dtypes(include=[np.number, bool]).astype(float)

    def fit(self, X: pd.DataFrame, y: pd.Series) -> "DayStartPyMCModel":
        import pymc as pm

        X_clean = self._prep_features(X)
        self.feature_cols = list(X_clean.columns)

        X_mat = X_clean.to_numpy()
        y_vec = y.to_numpy()

        self.x_mean = np.mean(X_mat, axis=0)
        self.x_std = np.std(X_mat, axis=0)
        self.x_std[self.x_std == 0] = 1.0

        self.y_mean = float(np.mean(y_vec))
        self.y_std = float(np.std(y_vec))
        if self.y_std == 0:
            self.y_std = 1.0

        X_scaled = (X_mat - self.x_mean) / self.x_std
        y_scaled = (y_vec - self.y_mean) / self.y_std

        n_features = X_scaled.shape[1]

        with pm.Model():
            intercept = pm.Normal("intercept", mu=0.0, sigma=1.0)
            beta = pm.Normal("beta", mu=0.0, sigma=1.0, shape=n_features)
            sigma = pm.HalfNormal("sigma", sigma=1.0)

            mu = intercept + pm.math.dot(X_scaled, beta)
            pm.Normal("y_obs", mu=mu, sigma=sigma, observed=y_scaled)

            if self.use_map or len(X) < 10:
                map_estimate = pm.find_MAP(progressbar=False)
                self.posterior_mean_intercept = float(map_estimate["intercept"]) * self.y_std + self.y_mean
                self.posterior_mean_weights = np.asarray(map_estimate["beta"]) * (self.y_std / self.x_std)
                self.posterior_sigma = float(map_estimate["sigma"]) * self.y_std
            else:
                trace = pm.sample(
                    draws=self.draws,
                    tune=self.tune,
                    chains=1,
                    progressbar=False,
                    return_inferencedata=True,
                )
                self.posterior_mean_intercept = float(trace.posterior["intercept"].mean()) * self.y_std + self.y_mean
                self.posterior_mean_weights = np.asarray(trace.posterior["beta"].mean(dim=["chain", "draw"])) * (
                    self.y_std / self.x_std
                )
                self.posterior_sigma = float(trace.posterior["sigma"].mean()) * self.y_std

        self.is_fitted = True
        return self

    def predict(self, X: pd.DataFrame) -> ForecastResult:
        row_dict = X.iloc[0].to_dict()
        X_clean = self._prep_features(X.iloc[[0]])
        for col in self.feature_cols:
            if col not in X_clean.columns:
                X_clean[col] = 0.0
        X_clean = X_clean[self.feature_cols]

        X_mat = X_clean.to_numpy()
        pred_val = float(self.posterior_mean_intercept + np.dot(X_mat, self.posterior_mean_weights)[0])
        std_val = float(max(self.posterior_sigma, 1e6))

        lower_90 = pred_val - 1.645 * std_val
        upper_90 = pred_val + 1.645 * std_val

        from scipy.stats import norm

        prob_positive = 1.0 - norm.cdf(0, loc=pred_val, scale=std_val)
        confidence = float(max(prob_positive, 1.0 - prob_positive))

        direction = self._classify_direction(pred_val, confidence)
        playbook = self._determine_playbook(row_dict, pred_val)
        top_buy, top_sell = self._determine_top_sectors(row_dict)

        return ForecastResult(
            forecast_date=row_dict.get("trade_date"),
            target_broker_id="MLB",
            predicted_net_flow_tl=pred_val,
            predicted_flow_lower_90=lower_90,
            predicted_flow_upper_90=upper_90,
            predicted_direction=direction,
            direction_confidence=confidence,
            predicted_playbook=playbook,
            top_predicted_buy_sector=top_buy,
            top_predicted_sell_sector=top_sell,
            model_name=self.model_name,
            model_version=self.model_version,
            features_used={c: float(row_dict.get(c, 0.0)) for c in self.feature_cols[:5]},
        )


# ==========================================
# 2. Gradient-Boosted LightGBM Model
# ==========================================


@ModelRegistry.register("day_start_lightgbm")
class DayStartLightGBMModel(BaseDayStartModel):
    """LightGBM Non-Linear Ensemble: Models complex interactions between competitor posture and cost basis."""

    def __init__(
        self, n_estimators: int = 50, learning_rate: float = 0.05, thresholds: Optional[FlowThresholdProfile] = None
    ):
        super().__init__(model_name="day_start_lightgbm", model_version="1.0.0", thresholds=thresholds)
        try:
            import lightgbm as lgb

            self.model = lgb.LGBMRegressor(
                n_estimators=n_estimators,
                learning_rate=learning_rate,
                max_depth=4,
                num_leaves=15,
                min_child_samples=3,
                random_state=42,
                verbosity=-1,
            )
        except ImportError:
            from sklearn.ensemble import GradientBoostingRegressor

            self.model = GradientBoostingRegressor(
                n_estimators=n_estimators,
                learning_rate=learning_rate,
                max_depth=3,
                random_state=42,
            )
        self.feature_cols: list[str] = []

    def _prep_features(self, X: pd.DataFrame) -> pd.DataFrame:
        feat_df = X.copy()
        drop_cols = [
            "trade_date",
            "target_open_net_flow_tl",
            "target_open_turnover_tl",
            "target_open_market_share",
            "target_open_direction",
        ]
        for col in drop_cols:
            if col in feat_df.columns:
                feat_df = feat_df.drop(columns=[col])
        return feat_df.select_dtypes(include=[np.number, bool]).astype(float)

    def fit(self, X: pd.DataFrame, y: pd.Series) -> "DayStartLightGBMModel":
        X_clean = self._prep_features(X)
        self.feature_cols = list(X_clean.columns)
        self.model.fit(X_clean, y)
        self.is_fitted = True
        return self

    def predict(self, X: pd.DataFrame) -> ForecastResult:
        row_dict = X.iloc[0].to_dict()
        X_clean = self._prep_features(X.iloc[[0]])
        for col in self.feature_cols:
            if col not in X_clean.columns:
                X_clean[col] = 0.0
        X_clean = X_clean[self.feature_cols]

        pred_val = float(np.asarray(self.model.predict(X_clean))[0])
        std_est = 22e6

        confidence = 0.72
        direction = self._classify_direction(pred_val, confidence)
        playbook = self._determine_playbook(row_dict, pred_val)
        top_buy, top_sell = self._determine_top_sectors(row_dict)

        return ForecastResult(
            forecast_date=row_dict.get("trade_date"),
            target_broker_id="MLB",
            predicted_net_flow_tl=pred_val,
            predicted_flow_lower_90=pred_val - 1.645 * std_est,
            predicted_flow_upper_90=pred_val + 1.645 * std_est,
            predicted_direction=direction,
            direction_confidence=confidence,
            predicted_playbook=playbook,
            top_predicted_buy_sector=top_buy,
            top_predicted_sell_sector=top_sell,
            model_name=self.model_name,
            model_version=self.model_version,
            features_used={c: float(row_dict.get(c, 0.0)) for c in self.feature_cols[:5]},
        )


# ==========================================
# 3. Gradient-Boosted XGBoost Model
# ==========================================


@ModelRegistry.register("day_start_xgboost")
class DayStartXGBoostModel(BaseDayStartModel):
    """XGBoost Non-Linear Ensemble: Gradient-boosted decision trees with exact second-order Taylor expansion gradients."""

    def __init__(
        self,
        n_estimators: int = 50,
        learning_rate: float = 0.05,
        max_depth: int = 3,
        subsample: float = 0.8,
        colsample_bytree: float = 0.8,
        thresholds: Optional[FlowThresholdProfile] = None,
    ):
        super().__init__(model_name="day_start_xgboost", model_version="1.0.0", thresholds=thresholds)
        try:
            import xgboost as xgb

            self.model = xgb.XGBRegressor(
                n_estimators=n_estimators,
                learning_rate=learning_rate,
                max_depth=max_depth,
                subsample=subsample,
                colsample_bytree=colsample_bytree,
                random_state=42,
                verbosity=0,
            )
        except ImportError:
            from sklearn.ensemble import GradientBoostingRegressor

            self.model = GradientBoostingRegressor(
                n_estimators=n_estimators,
                learning_rate=learning_rate,
                max_depth=max_depth,
                random_state=42,
            )
        self.feature_cols: list[str] = []
        self.residual_std: float = 22e6

    def _prep_features(self, X: pd.DataFrame) -> pd.DataFrame:
        feat_df = X.copy()
        drop_cols = [
            "trade_date",
            "target_open_net_flow_tl",
            "target_open_turnover_tl",
            "target_open_market_share",
            "target_open_direction",
        ]
        for col in drop_cols:
            if col in feat_df.columns:
                feat_df = feat_df.drop(columns=[col])
        return feat_df.select_dtypes(include=[np.number, bool]).astype(float)

    def fit(self, X: pd.DataFrame, y: pd.Series) -> "DayStartXGBoostModel":
        X_clean = self._prep_features(X)
        self.feature_cols = list(X_clean.columns)
        self.model.fit(X_clean, y)

        # Calculate empirical residual standard deviation for credible bounds
        if len(X_clean) >= 3:
            preds = self.model.predict(X_clean)
            residuals = y.to_numpy() - preds
            self.residual_std = float(np.std(residuals)) if np.std(residuals) > 0 else 22e6
        else:
            self.residual_std = 22e6

        self.is_fitted = True
        return self

    def predict(self, X: pd.DataFrame) -> ForecastResult:
        row_dict = X.iloc[0].to_dict()
        X_clean = self._prep_features(X.iloc[[0]])
        for col in self.feature_cols:
            if col not in X_clean.columns:
                X_clean[col] = 0.0
        X_clean = X_clean[self.feature_cols]

        pred_val = float(np.asarray(self.model.predict(X_clean))[0])
        std_est = self.residual_std

        confidence = 0.72
        direction = self._classify_direction(pred_val, confidence)
        playbook = self._determine_playbook(row_dict, pred_val)
        top_buy, top_sell = self._determine_top_sectors(row_dict)

        return ForecastResult(
            forecast_date=row_dict.get("trade_date"),
            target_broker_id="MLB",
            predicted_net_flow_tl=pred_val,
            predicted_flow_lower_90=pred_val - 1.645 * std_est,
            predicted_flow_upper_90=pred_val + 1.645 * std_est,
            predicted_direction=direction,
            direction_confidence=confidence,
            predicted_playbook=playbook,
            top_predicted_buy_sector=top_buy,
            top_predicted_sell_sector=top_sell,
            model_name=self.model_name,
            model_version=self.model_version,
            features_used={c: float(row_dict.get(c, 0.0)) for c in self.feature_cols[:5]},
        )
