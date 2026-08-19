"""Day-Start Candidate Models: Baselines, LightGBM, Bayesian Ridge, PyMC, and Probabilistic Ensemble."""

from typing import Any, Dict, List

import numpy as np
import pandas as pd
from sklearn.linear_model import BayesianRidge
from sklearn.metrics import mean_absolute_error, root_mean_squared_error

from mdk_trading_oracle.core.logger import get_logger
from mdk_trading_oracle.models.base import (
    BaseForecaster,
    ForecastDirection,
    ForecastResult,
    OpeningPlaybook,
)
from mdk_trading_oracle.models.registry import ModelRegistry

logger = get_logger("mdk_oracle.models.day_start.models")


class BaseDayStartModel(BaseForecaster):
    """Base class providing common evaluation logic and directional conviction classification."""

    def _classify_direction(self, net_flow_tl: float, confidence: float) -> str:
        """Classify continuous predicted net flow (TL) into directional category."""
        million_flow = net_flow_tl / 1e6
        if million_flow > 50.0 and confidence >= 0.70:
            return ForecastDirection.STRONG_ACCUMULATE
        elif million_flow > 10.0:
            return ForecastDirection.ACCUMULATE
        elif million_flow < -50.0 and confidence >= 0.70:
            return ForecastDirection.STRONG_DISTRIBUTE
        elif million_flow < -10.0:
            return ForecastDirection.DISTRIBUTE
        return ForecastDirection.NEUTRAL

    def _determine_playbook(self, row_dict: Dict[str, Any], predicted_flow: float) -> str:
        """Determine the institutional execution playbook based on competitor and cost basis dynamics."""
        w4_delta = float(row_dict.get("feat_bofa_vs_top5_w4_flow_delta_tl", 0.0))
        cost_basis_spread = float(row_dict.get("feat_bofa_cost_basis_spread_20d_pct", 0.0))
        is_monday = bool(row_dict.get("is_monday", False))

        if predicted_flow > 20e6 and w4_delta > 30e6:
            return OpeningPlaybook.SQUEEZE_LONG
        elif predicted_flow > 40e6:
            return OpeningPlaybook.MOMENTUM_EXPANSION
        elif predicted_flow < -20e6 and cost_basis_spread > 0.05:
            return OpeningPlaybook.LIQUIDITY_FADE
        elif cost_basis_spread < -0.04 and predicted_flow > 0:
            return OpeningPlaybook.DEFENSE_SUPPORT
        elif is_monday:
            return OpeningPlaybook.SECTOR_ROTATION
        return OpeningPlaybook.NEUTRAL_WAIT

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


# ==========================================
# 0. Baseline Models (Benchmark to Beat)
# ==========================================

@ModelRegistry.register("day_start_baseline_persistence")
class DayStartNaivePersistenceModel(BaseDayStartModel):
    """Baseline 0: Carries yesterday's closing Window 4 flow forward as today's opening expectation."""

    def __init__(self):
        super().__init__(model_name="day_start_baseline_persistence", model_version="1.0.0")

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

        return ForecastResult(
            forecast_date=row_dict.get("trade_date"),
            target_broker_id="MLB",
            predicted_net_flow_tl=pred_flow,
            predicted_flow_lower_90=pred_flow - 1.645 * std_est,
            predicted_flow_upper_90=pred_flow + 1.645 * std_est,
            predicted_direction=direction,
            direction_confidence=confidence,
            predicted_open_market_share=float(row_dict.get("feat_bofa_prev_day_market_share", 0.15)),
            predicted_playbook=playbook,
            top_predicted_buy_sector="Banking" if pred_flow > 0 else "None",
            top_predicted_sell_sector="Transportation" if pred_flow < 0 else "None",
            model_name=self.model_name,
            model_version=self.model_version,
        )


@ModelRegistry.register("day_start_baseline_rolling_mean")
class DayStartRollingMeanModel(BaseDayStartModel):
    """Baseline 1: Predicts opening flow as historical 5-day rolling average."""

    def __init__(self):
        super().__init__(model_name="day_start_baseline_rolling_mean", model_version="1.0.0")
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

        return ForecastResult(
            forecast_date=row_dict.get("trade_date"),
            target_broker_id="MLB",
            predicted_net_flow_tl=pred_flow,
            predicted_flow_lower_90=pred_flow - 1.645 * std_est,
            predicted_flow_upper_90=pred_flow + 1.645 * std_est,
            predicted_direction=direction,
            direction_confidence=confidence,
            predicted_open_market_share=0.15,
            predicted_playbook=playbook,
            top_predicted_buy_sector="None",
            top_predicted_sell_sector="None",
            model_name=self.model_name,
            model_version=self.model_version,
        )


# ==========================================
# 1. Bayesian Probabilistic Models (Ridge & PyMC)
# ==========================================

@ModelRegistry.register("day_start_bayesian_ridge")
class DayStartBayesianModel(BaseDayStartModel):
    """Bayesian Probabilistic Forecaster (Ridge): Computes conjugate posterior distributions and exact credible intervals."""

    def __init__(self, max_iter: int = 300):
        super().__init__(model_name="day_start_bayesian_ridge", model_version="1.0.0")
        self.regressor = BayesianRidge(max_iter=max_iter, compute_score=True)
        self.feature_cols: list[str] = []

    def _prep_features(self, X: pd.DataFrame) -> pd.DataFrame:
        feat_df = X.copy()
        drop_cols = ["trade_date", "target_open_net_flow_tl", "target_open_turnover_tl", 
                     "target_open_market_share", "target_open_direction"]
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

        return ForecastResult(
            forecast_date=row_dict.get("trade_date"),
            target_broker_id="MLB",
            predicted_net_flow_tl=pred_val,
            predicted_flow_lower_90=lower_90,
            predicted_flow_upper_90=upper_90,
            predicted_direction=direction,
            direction_confidence=confidence,
            predicted_open_market_share=float(row_dict.get("feat_bofa_prev_day_market_share", 0.15)),
            predicted_playbook=playbook,
            top_predicted_buy_sector="Banking" if pred_val > 0 else "None",
            top_predicted_sell_sector="Transportation" if pred_val < 0 else "None",
            model_name=self.model_name,
            model_version=self.model_version,
            features_used={c: float(row_dict.get(c, 0.0)) for c in self.feature_cols[:5]},
        )


@ModelRegistry.register("day_start_pymc")
class DayStartPyMCModel(BaseDayStartModel):
    """PyMC Full Bayesian MCMC / NUTS Forecaster with custom institutional priors."""

    def __init__(self, draws: int = 500, tune: int = 500):
        super().__init__(model_name="day_start_pymc", model_version="1.0.0")
        self.draws = draws
        self.tune = tune
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
        drop_cols = ["trade_date", "target_open_net_flow_tl", "target_open_turnover_tl", 
                     "target_open_market_share", "target_open_direction"]
        for col in drop_cols:
            if col in feat_df.columns:
                feat_df = feat_df.drop(columns=[col])
        return feat_df.select_dtypes(include=[np.number, bool]).astype(float)

    def fit(self, X: pd.DataFrame, y: pd.Series) -> "DayStartPyMCModel":
        import pymc as pm

        X_clean = self._prep_features(X)
        self.feature_cols = list(X_clean.columns)

        X_mat = X_clean.to_numpy()
        y_vec = y.to_numpy().astype(float)

        # Standardize for MCMC stability
        self.x_mean = np.mean(X_mat, axis=0)
        self.x_std = np.std(X_mat, axis=0) + 1e-8
        X_scaled = (X_mat - self.x_mean) / self.x_std

        self.y_mean = float(np.mean(y_vec))
        self.y_std = float(np.std(y_vec)) + 1e-8
        y_scaled = (y_vec - self.y_mean) / self.y_std

        n_features = X_scaled.shape[1]

        logger.info(f"Fitting PyMC Bayesian Model on {len(X_scaled)} samples with {n_features} features...")

        with pm.Model() as model:
            # Informative / Regularizing Priors
            intercept = pm.Normal("intercept", mu=0.0, sigma=1.0)
            beta = pm.Normal("beta", mu=0.0, sigma=0.5, shape=n_features)
            sigma = pm.HalfNormal("sigma", sigma=1.0)

            # Likelihood
            mu = intercept + pm.math.dot(X_scaled, beta)
            pm.Normal("y_obs", mu=mu, sigma=sigma, observed=y_scaled)

            # Fast MAP optimization + sampling
            try:
                idata = pm.sample(
                    draws=self.draws,
                    tune=self.tune,
                    chains=2,
                    random_seed=42,
                    progressbar=False,
                    compute_convergence_checks=False,
                )
                self.posterior_mean_weights = idata.posterior["beta"].mean(dim=["chain", "draw"]).values
                self.posterior_mean_intercept = float(idata.posterior["intercept"].mean().values)
                self.posterior_sigma = float(idata.posterior["sigma"].mean().values) * self.y_std
            except Exception as e:
                logger.warning(f"PyMC sampling fallback to find_MAP: {e}")
                map_est = pm.find_MAP()
                self.posterior_mean_weights = np.asarray(map_est["beta"])
                self.posterior_mean_intercept = float(map_est["intercept"])
                self.posterior_sigma = float(map_est["sigma"]) * self.y_std

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
        X_scaled = (X_mat - self.x_mean) / self.x_std

        # Scaled prediction back to original TL units
        pred_scaled = self.posterior_mean_intercept + np.dot(X_scaled, self.posterior_mean_weights)
        pred_val = float(pred_scaled[0] * self.y_std + self.y_mean)
        std_val = float(self.posterior_sigma)

        # 90% Bayesian Credible Interval
        lower_90 = pred_val - 1.645 * std_val
        upper_90 = pred_val + 1.645 * std_val

        from scipy.stats import norm
        prob_positive = 1.0 - norm.cdf(0, loc=pred_val, scale=std_val)
        confidence = float(max(prob_positive, 1.0 - prob_positive))

        direction = self._classify_direction(pred_val, confidence)
        playbook = self._determine_playbook(row_dict, pred_val)

        return ForecastResult(
            forecast_date=row_dict.get("trade_date"),
            target_broker_id="MLB",
            predicted_net_flow_tl=pred_val,
            predicted_flow_lower_90=lower_90,
            predicted_flow_upper_90=upper_90,
            predicted_direction=direction,
            direction_confidence=confidence,
            predicted_open_market_share=float(row_dict.get("feat_bofa_prev_day_market_share", 0.15)),
            predicted_playbook=playbook,
            top_predicted_buy_sector="Banking" if pred_val > 0 else "None",
            top_predicted_sell_sector="Transportation" if pred_val < 0 else "None",
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

    def __init__(self, n_estimators: int = 50, learning_rate: float = 0.05):
        super().__init__(model_name="day_start_lightgbm", model_version="1.0.0")
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
        drop_cols = ["trade_date", "target_open_net_flow_tl", "target_open_turnover_tl", 
                     "target_open_market_share", "target_open_direction"]
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

        return ForecastResult(
            forecast_date=row_dict.get("trade_date"),
            target_broker_id="MLB",
            predicted_net_flow_tl=pred_val,
            predicted_flow_lower_90=pred_val - 1.645 * std_est,
            predicted_flow_upper_90=pred_val + 1.645 * std_est,
            predicted_direction=direction,
            direction_confidence=confidence,
            predicted_open_market_share=float(row_dict.get("feat_bofa_prev_day_market_share", 0.15)),
            predicted_playbook=playbook,
            top_predicted_buy_sector="Banking" if pred_val > 0 else "None",
            top_predicted_sell_sector="Transportation" if pred_val < 0 else "None",
            model_name=self.model_name,
            model_version=self.model_version,
            features_used={c: float(row_dict.get(c, 0.0)) for c in self.feature_cols[:5]},
        )
