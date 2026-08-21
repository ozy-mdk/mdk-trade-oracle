"""Sector Day-Start Candidate Models: Baselines, LightGBM, Bayesian Ridge, and PyMC."""

from typing import Any, Dict, List, Optional

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

logger = get_logger("mdk_oracle.models.sector_day_start.models")


class BaseSectorDayStartModel(BaseForecaster):
    """Base class providing common evaluation logic and dynamic percentile directional conviction classification for sectors."""

    def __init__(
        self,
        model_name: str,
        model_version: str = "1.0.0",
        sector_thresholds: Optional[Dict[str, FlowThresholdProfile]] = None,
    ):
        super().__init__(model_name=model_name, model_version=model_version)
        self.sector_thresholds: Dict[str, FlowThresholdProfile] = sector_thresholds or {}

    def set_sector_thresholds(self, sector_thresholds: Dict[str, FlowThresholdProfile]) -> None:
        """Update sector-specific flow thresholds map."""
        self.sector_thresholds = sector_thresholds

    def _get_thresholds_for_sector(self, sector_name: Optional[str]) -> FlowThresholdProfile:
        """Retrieve threshold profile for specific sector with safe default fallback."""
        if sector_name and sector_name in self.sector_thresholds:
            return self.sector_thresholds[sector_name]
        return FlowThresholdProfile(
            buy_p25_tl=2e6, buy_p50_tl=5e6, buy_p85_tl=15e6,
            sell_p25_tl=2e6, sell_p50_tl=5e6, sell_p85_tl=15e6,
        )

    def _classify_direction(self, net_flow_tl: float, confidence: float = 0.5, sector: Optional[str] = None) -> str:
        """Classify continuous predicted sector net flow (TL) into dynamic percentile directional category."""
        th = self._get_thresholds_for_sector(sector)
        return FlowThresholdClassifier.classify(net_flow_tl, th)

    def _determine_playbook(self, row_dict: Dict[str, Any], predicted_flow: float) -> str:
        """Determine the sector execution playbook based on competitor deltas, wallet share, and dynamic percentiles."""
        w4_delta = float(row_dict.get("feat_sector_bofa_vs_top5_w4_delta_tl", 0.0))
        is_monday = bool(row_dict.get("is_monday", False))
        sector = row_dict.get("sector")
        th = self._get_thresholds_for_sector(sector)

        buy_p50 = th.buy_p50_tl
        buy_p85 = th.buy_p85_tl
        sell_p50 = th.sell_p50_tl

        if predicted_flow >= buy_p50 and w4_delta >= buy_p50:
            return OpeningPlaybook.SQUEEZE_LONG
        elif predicted_flow >= buy_p85:
            return OpeningPlaybook.MOMENTUM_EXPANSION
        elif predicted_flow <= -sell_p50:
            return OpeningPlaybook.LIQUIDITY_FADE
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

    def walk_forward_evaluate(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        min_train_samples: int = 5,
        eval_window_days: Optional[int] = None,
    ) -> Dict[str, Any]:
        """Perform expanding-window walk-forward validation for sector series."""
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
# 0. Baselines
# ==========================================

@ModelRegistry.register("sector_day_start_naive_persistence")
class SectorDayStartNaivePersistenceModel(BaseSectorDayStartModel):
    """Baseline 0: Carries yesterday's closing Window 4 sector flow forward."""

    def __init__(self, sector_thresholds: Optional[Dict[str, FlowThresholdProfile]] = None):
        super().__init__(
            model_name="sector_day_start_naive_persistence",
            model_version="1.0.0",
            sector_thresholds=sector_thresholds,
        )

    def fit(self, X: pd.DataFrame, y: pd.Series) -> "SectorDayStartNaivePersistenceModel":
        self.is_fitted = True
        return self

    def predict(self, X: pd.DataFrame) -> ForecastResult:
        row_dict = X.iloc[0].to_dict()
        pred_flow = float(row_dict.get("feat_sector_bofa_w4_net_flow_tl", 0.0))
        std_est = 15e6
        sector = str(row_dict.get("sector", "Sector"))

        confidence = 0.55
        direction = self._classify_direction(pred_flow, confidence, sector=sector)
        playbook = self._determine_playbook(row_dict, pred_flow)

        return ForecastResult(
            forecast_date=row_dict.get("trade_date"),
            target_broker_id="MLB",
            predicted_net_flow_tl=pred_flow,
            predicted_flow_lower_90=pred_flow - 1.645 * std_est,
            predicted_flow_upper_90=pred_flow + 1.645 * std_est,
            predicted_direction=direction,
            direction_confidence=confidence,
            predicted_playbook=playbook,
            top_predicted_buy_sector=sector,
            top_predicted_sell_sector="None",
            model_name=self.model_name,
            model_version=self.model_version,
        )


@ModelRegistry.register("sector_day_start_rolling_mean")
class SectorDayStartRollingMeanModel(BaseSectorDayStartModel):
    """Baseline 1: Predicts opening sector flow as historical 5-day rolling average."""

    def __init__(self, sector_thresholds: Optional[Dict[str, FlowThresholdProfile]] = None):
        super().__init__(
            model_name="sector_day_start_rolling_mean",
            model_version="1.0.0",
            sector_thresholds=sector_thresholds,
        )
        self.mean_flow = 0.0

    def fit(self, X: pd.DataFrame, y: pd.Series) -> "SectorDayStartRollingMeanModel":
        self.mean_flow = float(y.tail(5).mean()) if len(y) > 0 else 0.0
        self.is_fitted = True
        return self

    def predict(self, X: pd.DataFrame) -> ForecastResult:
        row_dict = X.iloc[0].to_dict()
        pred_flow = self.mean_flow
        std_est = 18e6
        sector = str(row_dict.get("sector", "Sector"))

        confidence = 0.50
        direction = self._classify_direction(pred_flow, confidence, sector=sector)
        playbook = self._determine_playbook(row_dict, pred_flow)

        return ForecastResult(
            forecast_date=row_dict.get("trade_date"),
            target_broker_id="MLB",
            predicted_net_flow_tl=pred_flow,
            predicted_flow_lower_90=pred_flow - 1.645 * std_est,
            predicted_flow_upper_90=pred_flow + 1.645 * std_est,
            predicted_direction=direction,
            direction_confidence=confidence,
            predicted_playbook=playbook,
            top_predicted_buy_sector=sector,
            top_predicted_sell_sector="None",
            model_name=self.model_name,
            model_version=self.model_version,
        )


# ==========================================
# 1. Bayesian Probabilistic Forecaster
# ==========================================

@ModelRegistry.register("sector_day_start_bayesian_ridge")
class SectorDayStartBayesianModel(BaseSectorDayStartModel):
    """Bayesian Ridge Probabilistic Forecaster for Sectors."""

    def __init__(
        self,
        max_iter: int = 300,
        sector_thresholds: Optional[Dict[str, FlowThresholdProfile]] = None,
    ):
        super().__init__(
            model_name="sector_day_start_bayesian_ridge",
            model_version="1.0.0",
            sector_thresholds=sector_thresholds,
        )
        self.regressor = BayesianRidge(max_iter=max_iter, compute_score=True)
        self.feature_cols: list[str] = []

    def _prep_features(self, X: pd.DataFrame) -> pd.DataFrame:
        feat_df = X.copy()
        drop_cols = ["trade_date", "sector", "target_sector_open_net_flow_tl", "target_sector_open_direction"]
        for col in drop_cols:
            if col in feat_df.columns:
                feat_df = feat_df.drop(columns=[col])
        return feat_df.select_dtypes(include=[np.number, bool]).astype(float)

    def fit(self, X: pd.DataFrame, y: pd.Series) -> "SectorDayStartBayesianModel":
        X_clean = self._prep_features(X)
        self.feature_cols = list(X_clean.columns)
        self.regressor.fit(X_clean, y)
        self.is_fitted = True
        return self

    def predict(self, X: pd.DataFrame) -> ForecastResult:
        row_dict = X.iloc[0].to_dict()
        sector = str(row_dict.get("sector", "Sector"))
        X_clean = self._prep_features(X.iloc[[0]])
        for col in self.feature_cols:
            if col not in X_clean.columns:
                X_clean[col] = 0.0
        X_clean = X_clean[self.feature_cols]

        mean_pred, std_pred = self.regressor.predict(X_clean, return_std=True)
        pred_val = float(np.asarray(mean_pred)[0])
        std_val = float(np.asarray(std_pred)[0])

        lower_90 = pred_val - 1.645 * std_val
        upper_90 = pred_val + 1.645 * std_val

        from scipy.stats import norm
        prob_positive = 1.0 - norm.cdf(0, loc=pred_val, scale=std_val)
        confidence = float(max(prob_positive, 1.0 - prob_positive))

        direction = self._classify_direction(pred_val, confidence, sector=sector)
        playbook = self._determine_playbook(row_dict, pred_val)

        return ForecastResult(
            forecast_date=row_dict.get("trade_date"),
            target_broker_id="MLB",
            predicted_net_flow_tl=pred_val,
            predicted_flow_lower_90=lower_90,
            predicted_flow_upper_90=upper_90,
            predicted_direction=direction,
            direction_confidence=confidence,
            predicted_playbook=playbook,
            top_predicted_buy_sector=sector,
            top_predicted_sell_sector="None",
            model_name=self.model_name,
            model_version=self.model_version,
            features_used={c: float(row_dict.get(c, 0.0)) for c in self.feature_cols[:5]},
        )


# ==========================================
# 2. PyMC GLM Model
# ==========================================

@ModelRegistry.register("sector_day_start_pymc")
class SectorDayStartPyMCModel(BaseSectorDayStartModel):
    """PyMC Full Bayesian GLM Forecaster for Sectors with informative shrinkage priors."""

    def __init__(
        self,
        draws: int = 300,
        tune: int = 300,
        use_map: bool = True,
        sector_thresholds: Optional[Dict[str, FlowThresholdProfile]] = None,
    ):
        super().__init__(
            model_name="sector_day_start_pymc",
            model_version="1.0.0",
            sector_thresholds=sector_thresholds,
        )
        self.draws = draws
        self.tune = tune
        self.use_map = use_map
        self.feature_cols: List[str] = []
        self.posterior_mean_weights: np.ndarray = np.array([])
        self.posterior_mean_intercept: float = 0.0
        self.posterior_sigma: float = 15e6
        self.x_mean: np.ndarray = np.array([])
        self.x_std: np.ndarray = np.array([])
        self.y_mean: float = 0.0
        self.y_std: float = 1.0

    def _prep_features(self, X: pd.DataFrame) -> pd.DataFrame:
        feat_df = X.copy()
        drop_cols = ["trade_date", "sector", "target_sector_open_net_flow_tl", "target_sector_open_direction"]
        for col in drop_cols:
            if col in feat_df.columns:
                feat_df = feat_df.drop(columns=[col])
        return feat_df.select_dtypes(include=[np.number, bool]).astype(float)

    def fit(self, X: pd.DataFrame, y: pd.Series) -> "SectorDayStartPyMCModel":
        import pymc as pm

        X_clean = self._prep_features(X)
        self.feature_cols = list(X_clean.columns)

        X_mat = X_clean.to_numpy()
        y_vec = y.to_numpy().astype(float)

        self.x_mean = np.mean(X_mat, axis=0)
        self.x_std = np.std(X_mat, axis=0) + 1e-8
        X_scaled = (X_mat - self.x_mean) / self.x_std

        self.y_mean = float(np.mean(y_vec))
        self.y_std = float(np.std(y_vec)) + 1e-8
        y_scaled = (y_vec - self.y_mean) / self.y_std

        n_features = X_scaled.shape[1]

        with pm.Model():
            intercept = pm.Normal("intercept", mu=0.0, sigma=1.0)
            beta = pm.Normal("beta", mu=0.0, sigma=0.5, shape=n_features)
            sigma = pm.HalfNormal("sigma", sigma=1.0)

            mu = intercept + pm.math.dot(X_scaled, beta)
            pm.Normal("y_obs", mu=mu, sigma=sigma, observed=y_scaled)

            map_est = pm.find_MAP(progressbar=False)
            self.posterior_mean_weights = np.asarray(map_est["beta"])
            self.posterior_mean_intercept = float(map_est["intercept"])
            self.posterior_sigma = float(map_est["sigma"]) * self.y_std

        self.is_fitted = True
        return self

    def predict(self, X: pd.DataFrame) -> ForecastResult:
        row_dict = X.iloc[0].to_dict()
        sector = str(row_dict.get("sector", "Sector"))
        X_clean = self._prep_features(X.iloc[[0]])
        for col in self.feature_cols:
            if col not in X_clean.columns:
                X_clean[col] = 0.0
        X_clean = X_clean[self.feature_cols]

        X_mat = X_clean.to_numpy()
        X_scaled = (X_mat - self.x_mean) / self.x_std

        pred_scaled = self.posterior_mean_intercept + np.dot(X_scaled, self.posterior_mean_weights)
        pred_val = float(pred_scaled[0] * self.y_std + self.y_mean)
        std_val = float(self.posterior_sigma)

        lower_90 = pred_val - 1.645 * std_val
        upper_90 = pred_val + 1.645 * std_val

        from scipy.stats import norm
        prob_positive = 1.0 - norm.cdf(0, loc=pred_val, scale=std_val)
        confidence = float(max(prob_positive, 1.0 - prob_positive))

        direction = self._classify_direction(pred_val, confidence, sector=sector)
        playbook = self._determine_playbook(row_dict, pred_val)

        return ForecastResult(
            forecast_date=row_dict.get("trade_date"),
            target_broker_id="MLB",
            predicted_net_flow_tl=pred_val,
            predicted_flow_lower_90=lower_90,
            predicted_flow_upper_90=upper_90,
            predicted_direction=direction,
            direction_confidence=confidence,
            predicted_playbook=playbook,
            top_predicted_buy_sector=sector,
            top_predicted_sell_sector="None",
            model_name=self.model_name,
            model_version=self.model_version,
            features_used={c: float(row_dict.get(c, 0.0)) for c in self.feature_cols[:5]},
        )


# ==========================================
# 3. LightGBM Model
# ==========================================

@ModelRegistry.register("sector_day_start_lightgbm")
class SectorDayStartLightGBMModel(BaseSectorDayStartModel):
    """LightGBM Non-Linear Regressor for Sector Day-Start Flow."""

    def __init__(
        self,
        n_estimators: int = 50,
        learning_rate: float = 0.05,
        sector_thresholds: Optional[Dict[str, FlowThresholdProfile]] = None,
    ):
        super().__init__(
            model_name="sector_day_start_lightgbm",
            model_version="1.0.0",
            sector_thresholds=sector_thresholds,
        )
        try:
            import lightgbm as lgb
            self.model = lgb.LGBMRegressor(
                n_estimators=n_estimators,
                learning_rate=learning_rate,
                max_depth=3,
                num_leaves=10,
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
        drop_cols = ["trade_date", "sector", "target_sector_open_net_flow_tl", "target_sector_open_direction"]
        for col in drop_cols:
            if col in feat_df.columns:
                feat_df = feat_df.drop(columns=[col])
        return feat_df.select_dtypes(include=[np.number, bool]).astype(float)

    def fit(self, X: pd.DataFrame, y: pd.Series) -> "SectorDayStartLightGBMModel":
        X_clean = self._prep_features(X)
        self.feature_cols = list(X_clean.columns)
        self.model.fit(X_clean, y)
        self.is_fitted = True
        return self

    def predict(self, X: pd.DataFrame) -> ForecastResult:
        row_dict = X.iloc[0].to_dict()
        sector = str(row_dict.get("sector", "Sector"))
        X_clean = self._prep_features(X.iloc[[0]])
        for col in self.feature_cols:
            if col not in X_clean.columns:
                X_clean[col] = 0.0
        X_clean = X_clean[self.feature_cols]

        pred_val = float(np.asarray(self.model.predict(X_clean))[0])
        std_est = 15e6

        confidence = 0.70
        direction = self._classify_direction(pred_val, confidence, sector=sector)
        playbook = self._determine_playbook(row_dict, pred_val)

        return ForecastResult(
            forecast_date=row_dict.get("trade_date"),
            target_broker_id="MLB",
            predicted_net_flow_tl=pred_val,
            predicted_flow_lower_90=pred_val - 1.645 * std_est,
            predicted_flow_upper_90=pred_val + 1.645 * std_est,
            predicted_direction=direction,
            direction_confidence=confidence,
            predicted_playbook=playbook,
            top_predicted_buy_sector=sector,
            top_predicted_sell_sector="None",
            model_name=self.model_name,
            model_version=self.model_version,
            features_used={c: float(row_dict.get(c, 0.0)) for c in self.feature_cols[:5]},
        )
