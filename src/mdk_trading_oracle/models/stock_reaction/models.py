"""Stock Reaction Candidate Models: Baselines, LightGBM, XGBoost, Bayesian Ridge, PyMC.

Predicts intraday stock return % (W2, W3, or W5) from W1 execution + T-1 inventory features.
Target variable: execution-aware return% = (window_vwap - w1_ref_price) / w1_ref_price * 100
"""

from dataclasses import dataclass, field
from datetime import date
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
from sklearn.linear_model import BayesianRidge
from sklearn.metrics import mean_absolute_error, root_mean_squared_error
from sklearn.preprocessing import StandardScaler

from mdk_trading_oracle.core.logger import get_logger
from mdk_trading_oracle.models.base import BaseForecaster, OpeningPlaybook
from mdk_trading_oracle.models.registry import ModelRegistry

logger = get_logger("mdk_oracle.models.stock_reaction.models")


# ─── Return threshold profile (per-stock per-window, loaded from silver_stock_reaction_thresholds)
@dataclass
class ReturnThresholdProfile:
    """Empirical percentile return thresholds for stock direction classification."""
    up_p25_pct: float = 0.20
    up_p50_pct: float = 0.50
    up_p85_pct: float = 1.50
    down_p25_pct: float = 0.20
    down_p50_pct: float = 0.50
    down_p85_pct: float = 1.50
    up_session_count: int = 0
    down_session_count: int = 0
    total_sessions: int = 0


class ReturnDirectionClassifier:
    """Classifies predicted return% into trader-facing conviction levels."""

    STRONG_RALLY = "STRONG_RALLY"
    RALLY = "RALLY"
    WEAK_RALLY = "WEAK_RALLY"
    NEUTRAL = "NEUTRAL"
    WEAK_DECLINE = "WEAK_DECLINE"
    DECLINE = "DECLINE"
    STRONG_DECLINE = "STRONG_DECLINE"

    @classmethod
    def classify(cls, return_pct: float, thresholds: ReturnThresholdProfile) -> str:
        if return_pct >= thresholds.up_p85_pct:
            return cls.STRONG_RALLY
        elif return_pct >= thresholds.up_p50_pct:
            return cls.RALLY
        elif return_pct >= thresholds.up_p25_pct:
            return cls.WEAK_RALLY
        elif return_pct <= -thresholds.down_p85_pct:
            return cls.STRONG_DECLINE
        elif return_pct <= -thresholds.down_p50_pct:
            return cls.DECLINE
        elif return_pct <= -thresholds.down_p25_pct:
            return cls.WEAK_DECLINE
        else:
            return cls.NEUTRAL


@dataclass
class StockReactionForecastResult:
    """Structured forecast output for a single (symbol, window) prediction."""
    forecast_date: date
    symbol: str
    window_name: str         # 'first_reaction' / 'midday_followup' / 'closing_session'
    predicted_return_pct: float
    predicted_return_lower_90: float
    predicted_return_upper_90: float
    predicted_direction: str
    direction_confidence: float
    predicted_playbook: str
    bofa_w1_direction: str
    bofa_w1_net_flow_tl: float
    bofa_w1_volume_share: float
    model_name: str
    model_version: str
    features_used: Dict[str, Any] = field(default_factory=dict)


def _playbook(return_pct: float, thresholds: ReturnThresholdProfile,
              row: Optional[Dict[str, Any]] = None) -> str:
    """Determine institutional execution playbook from predicted return and feature context."""
    row = row or {}
    bofa_net_flow = float(row.get("feat_bofa_w1_net_flow_tl", 0.0))
    tra_contra = float(row.get("feat_w1_bofa_tra_contra_signal", 0.0))
    cost_spread = float(row.get("feat_bofa_t1_cost_spread_pct", 0.0))
    comp_aligned = float(row.get("feat_w1_bofa_comp_alignment", 0.0))

    if return_pct >= thresholds.up_p85_pct and tra_contra > 0:
        return OpeningPlaybook.SQUEEZE_LONG
    elif return_pct >= thresholds.up_p85_pct and comp_aligned > 0:
        return OpeningPlaybook.MOMENTUM_EXPANSION
    elif return_pct >= thresholds.up_p25_pct and cost_spread < -0.5:
        return OpeningPlaybook.DEFENSE_SUPPORT
    elif return_pct <= -thresholds.down_p50_pct and bofa_net_flow < 0:
        return OpeningPlaybook.LIQUIDITY_FADE
    elif abs(return_pct) < thresholds.up_p25_pct:
        return OpeningPlaybook.NEUTRAL_WAIT
    return OpeningPlaybook.SECTOR_ROTATION


# ─── Base stock reaction model ────────────────────────────────────────────────────────────────────

class BaseStockReactionModel(BaseForecaster):
    """Common evaluation logic for all stock reaction candidate models."""

    CORE_FEATURES: List[str] = [
        "feat_bofa_w1_direction_strength",
        "feat_bofa_w1_vol_share",
        "feat_w1_bofa_comp_alignment",
        "feat_w1_bofa_tra_contra_signal",
        "feat_stock_dist_sma20_t1",
        "feat_stock_ret_t1_1d",
        "feat_bofa_t1_cost_spread_pct",
        "feat_bofa_t1_unrealized_pnl_tl",
        "feat_bofa_flow_zscore_t1",
        "feat_peer_spread_t1",
        "feat_macro_carry_t1",
        "feat_macro_rate_shock_decay_t1",
        "feat_is_monday",
        "feat_is_friday",
    ]

    def __init__(
        self,
        model_name: str,
        symbol: str,
        window: str,
        thresholds: Optional[ReturnThresholdProfile] = None,
        model_version: str = "1.0.0",
        use_core_features: bool = True,
    ):
        super().__init__(model_name=model_name, model_version=model_version)
        self.symbol = symbol
        self.window = window
        self.thresholds = thresholds or ReturnThresholdProfile()
        self.use_core_features = use_core_features

    def set_thresholds(self, thresholds: ReturnThresholdProfile) -> None:
        self.thresholds = thresholds

    def _get_X(self, df: pd.DataFrame) -> pd.DataFrame:
        """Extract feature matrix, defaulting to the 12 core microstructure features if present."""
        if self.use_core_features:
            available_core = [c for c in self.CORE_FEATURES if c in df.columns]
            if len(available_core) >= 8:
                return df[available_core].fillna(0.0)
        feat_cols = [c for c in df.columns if c.startswith("feat_")]
        return df[feat_cols].fillna(0.0)

    def evaluate(self, X_test: pd.DataFrame, y_test: pd.Series) -> Dict[str, float]:
        """Evaluate against test data; returns MAE, RMSE, hit-rate, PICP."""
        if not self.is_fitted or len(X_test) == 0:
            return {"hit_rate_pct": 0.0, "picp_90_pct": 0.0, "mae_pct": 999.0,
                    "rmse_pct": 999.0, "sample_size": 0}
        preds, lows, highs = [], [], []
        for _, row in X_test.iterrows():
            result = self.predict(row.to_frame().T)
            preds.append(result.predicted_return_pct)
            lows.append(result.predicted_return_lower_90)
            highs.append(result.predicted_return_upper_90)

        preds_arr = np.array(preds)
        actuals_arr = y_test.values
        lows_arr = np.array(lows)
        highs_arr = np.array(highs)

        direction_hits = np.sign(preds_arr) == np.sign(actuals_arr)
        inside_ci = (actuals_arr >= lows_arr) & (actuals_arr <= highs_arr)

        return {
            "hit_rate_pct": float(direction_hits.mean() * 100),
            "picp_90_pct": float(inside_ci.mean() * 100),
            "mae_pct": float(mean_absolute_error(actuals_arr, preds_arr)),
            "rmse_pct": float(root_mean_squared_error(actuals_arr, preds_arr)),
            "sample_size": len(preds_arr),
        }

    def walk_forward_evaluate(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        min_train_samples: int = 5,
        eval_window_days: Optional[int] = None,
    ) -> Dict[str, float]:
        """Expanding-window walk-forward OOS evaluation."""
        n = len(X)
        if n < min_train_samples + 1:
            return {"hit_rate_pct": 0.0, "picp_90_pct": 0.0, "mae_pct": 999.0,
                    "rmse_pct": 999.0, "sample_size": 0}

        eval_start = n - eval_window_days if eval_window_days and eval_window_days < n else min_train_samples
        preds, lows, highs, actuals = [], [], [], []

        for i in range(eval_start, n):
            X_train = X.iloc[:i]
            y_train = y.iloc[:i]
            X_test = X.iloc[i:i+1]
            y_true = y.iloc[i]

            try:
                self.fit(X_train, y_train)
                res = self.predict(X_test)
                preds.append(res.predicted_return_pct)
                lows.append(res.predicted_return_lower_90)
                highs.append(res.predicted_return_upper_90)
                actuals.append(y_true)
            except Exception as exc:
                logger.debug(f"Walk-forward step {i} failed for {self.symbol}/{self.window}: {exc}")
                continue

        if not preds:
            return {"hit_rate_pct": 0.0, "picp_90_pct": 0.0, "mae_pct": 999.0,
                    "rmse_pct": 999.0, "sample_size": 0}

        preds_arr = np.array(preds)
        actuals_arr = np.array(actuals)
        lows_arr = np.array(lows)
        highs_arr = np.array(highs)

        direction_hits = np.sign(preds_arr) == np.sign(actuals_arr)
        inside_ci = (actuals_arr >= lows_arr) & (actuals_arr <= highs_arr)

        return {
            "hit_rate_pct": float(direction_hits.mean() * 100),
            "picp_90_pct": float(inside_ci.mean() * 100),
            "mae_pct": float(mean_absolute_error(actuals_arr, preds_arr)),
            "rmse_pct": float(root_mean_squared_error(actuals_arr, preds_arr)),
            "sample_size": len(preds_arr),
        }

    def _build_result(
        self,
        forecast_date: date,
        predicted_return_pct: float,
        predicted_return_lower_90: float,
        predicted_return_upper_90: float,
        row: Optional[Dict[str, Any]] = None,
    ) -> StockReactionForecastResult:
        row = row or {}
        direction = ReturnDirectionClassifier.classify(predicted_return_pct, self.thresholds)
        confidence = min(abs(predicted_return_pct) / max(self.thresholds.up_p85_pct, 0.01), 1.0)
        play = _playbook(predicted_return_pct, self.thresholds, row)
        bofa_sign = float(row.get("feat_bofa_w1_direction_sign", 0.0))
        bofa_dir = "BUY" if bofa_sign > 0 else ("SELL" if bofa_sign < 0 else "NEUTRAL")
        window_map = {"w2": "first_reaction", "w3": "midday_followup", "w5": "closing_session",
                      "first_reaction": "first_reaction", "midday_followup": "midday_followup",
                      "closing_session": "closing_session"}
        return StockReactionForecastResult(
            forecast_date=forecast_date,
            symbol=self.symbol,
            window_name=window_map.get(self.window, self.window),
            predicted_return_pct=round(predicted_return_pct, 4),
            predicted_return_lower_90=round(predicted_return_lower_90, 4),
            predicted_return_upper_90=round(predicted_return_upper_90, 4),
            predicted_direction=direction,
            direction_confidence=round(min(confidence, 1.0), 4),
            predicted_playbook=play,
            bofa_w1_direction=bofa_dir,
            bofa_w1_net_flow_tl=float(row.get("feat_bofa_w1_net_flow_tl", 0.0)),
            bofa_w1_volume_share=float(row.get("feat_bofa_w1_vol_share", 0.0)),
            model_name=self.model_name,
            model_version=self.model_version,
            features_used={k: v for k, v in row.items() if k.startswith("feat_")},
        )


# ─── Baseline Hurdles (Used for Benchmark Checkpoints) ──────────────────────────────────────────

@ModelRegistry.register("stock_reaction_naive_persistence")
class StockReactionNaivePersistenceModel(BaseStockReactionModel):
    """Hurdle Baseline 0: Prior session's return in the same window."""

    def __init__(self, symbol: str = "AKBNK", window: str = "w2",
                 thresholds: Optional[ReturnThresholdProfile] = None):
        super().__init__("Baseline 0: Naive Persistence", symbol, window, thresholds)
        self._last_return: float = 0.0
        self._std: float = 0.5

    def fit(self, X: pd.DataFrame, y: pd.Series) -> "StockReactionNaivePersistenceModel":
        if len(y) > 0:
            self._last_return = float(y.iloc[-1])
            self._std = float(y.std()) if len(y) > 1 else 0.5
        self.is_fitted = True
        return self

    def predict(self, X: pd.DataFrame) -> StockReactionForecastResult:
        row = X.iloc[0].to_dict() if len(X) > 0 else {}
        pred = self._last_return
        ci_half = 1.645 * max(self._std, 0.1)
        forecast_date = row.get("trade_date") or date.today()
        return self._build_result(
            forecast_date=forecast_date,
            predicted_return_pct=pred,
            predicted_return_lower_90=pred - ci_half,
            predicted_return_upper_90=pred + ci_half,
            row=row,
        )


@ModelRegistry.register("stock_reaction_rolling_mean")
class StockReactionRollingMeanModel(BaseStockReactionModel):
    """Hurdle Baseline 1: 5-day rolling mean of historical window returns."""

    def __init__(self, symbol: str = "AKBNK", window: str = "w2",
                 thresholds: Optional[ReturnThresholdProfile] = None):
        super().__init__("Baseline 1: 5-Day Rolling Mean", symbol, window, thresholds)
        self._mean: float = 0.0
        self._std: float = 0.5

    def fit(self, X: pd.DataFrame, y: pd.Series) -> "StockReactionRollingMeanModel":
        if len(y) >= 5:
            self._mean = float(y.tail(5).mean())
            self._std = float(y.std()) if len(y) > 1 else 0.5
        elif len(y) > 0:
            self._mean = float(y.mean())
            self._std = float(y.std()) if len(y) > 1 else 0.5
        self.is_fitted = True
        return self

    def predict(self, X: pd.DataFrame) -> StockReactionForecastResult:
        row = X.iloc[0].to_dict() if len(X) > 0 else {}
        ci_half = 1.645 * max(self._std, 0.1)
        forecast_date = row.get("trade_date") or date.today()
        return self._build_result(
            forecast_date=forecast_date,
            predicted_return_pct=self._mean,
            predicted_return_lower_90=self._mean - ci_half,
            predicted_return_upper_90=self._mean + ci_half,
            row=row,
        )


@ModelRegistry.register("stock_reaction_always_long")
class StockReactionAlwaysLongModel(BaseStockReactionModel):
    """Hurdle Baseline 2: Unconditional Long / Always Buy (+1)."""

    def __init__(self, symbol: str = "AKBNK", window: str = "w2",
                 thresholds: Optional[ReturnThresholdProfile] = None):
        super().__init__("Hurdle 2: Always Long (+1)", symbol, window, thresholds)
        self._mean_pos: float = 0.5
        self._std: float = 0.5

    def fit(self, X: pd.DataFrame, y: pd.Series) -> "StockReactionAlwaysLongModel":
        pos_y = y[y > 0]
        self._mean_pos = float(pos_y.mean()) if len(pos_y) > 0 else 0.5
        self._std = float(y.std()) if len(y) > 1 else 0.5
        self.is_fitted = True
        return self

    def predict(self, X: pd.DataFrame) -> StockReactionForecastResult:
        row = X.iloc[0].to_dict() if len(X) > 0 else {}
        ci_half = 1.645 * max(self._std, 0.1)
        forecast_date = row.get("trade_date") or date.today()
        return self._build_result(
            forecast_date=forecast_date,
            predicted_return_pct=self._mean_pos,
            predicted_return_lower_90=self._mean_pos - ci_half,
            predicted_return_upper_90=self._mean_pos + ci_half,
            row=row,
        )


@ModelRegistry.register("stock_reaction_always_short")
class StockReactionAlwaysShortModel(BaseStockReactionModel):
    """Hurdle Baseline 3: Unconditional Short / Always Sell (-1)."""

    def __init__(self, symbol: str = "AKBNK", window: str = "w2",
                 thresholds: Optional[ReturnThresholdProfile] = None):
        super().__init__("Hurdle 3: Always Short (-1)", symbol, window, thresholds)
        self._mean_neg: float = -0.5
        self._std: float = 0.5

    def fit(self, X: pd.DataFrame, y: pd.Series) -> "StockReactionAlwaysShortModel":
        neg_y = y[y < 0]
        self._mean_neg = float(neg_y.mean()) if len(neg_y) > 0 else -0.5
        self._std = float(y.std()) if len(y) > 1 else 0.5
        self.is_fitted = True
        return self

    def predict(self, X: pd.DataFrame) -> StockReactionForecastResult:
        row = X.iloc[0].to_dict() if len(X) > 0 else {}
        ci_half = 1.645 * max(self._std, 0.1)
        forecast_date = row.get("trade_date") or date.today()
        return self._build_result(
            forecast_date=forecast_date,
            predicted_return_pct=self._mean_neg,
            predicted_return_lower_90=self._mean_neg - ci_half,
            predicted_return_upper_90=self._mean_neg + ci_half,
            row=row,
        )


# ─── Candidate Model 1: Bayesian Ridge (Primary Active Probabilistic Model) ──────────────────────

@ModelRegistry.register("stock_reaction_bayesian_ridge")
class StockReactionBayesianModel(BaseStockReactionModel):
    """Analytical Bayesian Ridge Regression with shrinkage priors and closed-form 90% credible intervals."""

    def __init__(self, symbol: str = "AKBNK", window: str = "w2",
                 thresholds: Optional[ReturnThresholdProfile] = None):
        super().__init__("Bayesian Ridge Probabilistic", symbol, window, thresholds)
        self._model = BayesianRidge(
            alpha_1=1e-2,
            alpha_2=1e-2,
            lambda_1=1e-1,
            lambda_2=1e-1,
            compute_score=True,
        )
        self._scaler = StandardScaler()
        self._feature_cols: List[str] = []

    def fit(self, X: pd.DataFrame, y: pd.Series) -> "StockReactionBayesianModel":
        X_filled = self._get_X(X)
        self._feature_cols = list(X_filled.columns)
        X_scaled = self._scaler.fit_transform(X_filled)
        self._model.fit(X_scaled, y)
        self.is_fitted = True
        return self

    def predict(self, X: pd.DataFrame) -> StockReactionForecastResult:
        X_filled = self._get_X(X)
        for col in self._feature_cols:
            if col not in X_filled.columns:
                X_filled[col] = 0.0
        X_scaled = self._scaler.transform(X_filled[self._feature_cols])
        pred, pred_std = self._model.predict(X_scaled, return_std=True)
        pred_val = float(pred[0])
        ci_half = 1.645 * max(float(pred_std[0]), 0.05)
        row = X.iloc[0].to_dict()
        return self._build_result(
            forecast_date=row.get("trade_date") or date.today(),
            predicted_return_pct=pred_val,
            predicted_return_lower_90=pred_val - ci_half,
            predicted_return_upper_90=pred_val + ci_half,
            row=row,
        )


# ─── Candidate Model 2: LightGBM (Active Non-Linear Ensemble) ────────────────────────────────────

@ModelRegistry.register("stock_reaction_lightgbm")
class StockReactionLightGBMModel(BaseStockReactionModel):
    """LightGBM gradient-boosted trees for non-linear stock reaction prediction."""

    def __init__(self, symbol: str = "AKBNK", window: str = "w2",
                 thresholds: Optional[ReturnThresholdProfile] = None):
        super().__init__("LightGBM Non-Linear Ensemble", symbol, window, thresholds)
        self._model = None
        self._residual_std: float = 0.5
        self._feature_cols: List[str] = []

    def fit(self, X: pd.DataFrame, y: pd.Series) -> "StockReactionLightGBMModel":
        try:
            import lightgbm as lgb
        except ImportError:
            logger.warning("LightGBM not installed — falling back to linear model.")
            from sklearn.linear_model import Ridge
            X_filled = self._get_X(X)
            self._feature_cols = list(X_filled.columns)
            self._model = Ridge(alpha=1.0).fit(X_filled, y)
            preds = self._model.predict(X_filled)
            self._residual_std = float(np.std(y.values - preds)) if len(preds) > 1 else 0.5
            self.is_fitted = True
            return self

        X_filled = self._get_X(X)
        self._feature_cols = list(X_filled.columns)
        params = {
            "objective": "regression",
            "metric": "rmse",
            "n_estimators": 100,
            "learning_rate": 0.03,
            "max_depth": 3,
            "num_leaves": 7,
            "min_child_samples": 3,
            "subsample": 0.85,
            "colsample_bytree": 0.85,
            "reg_alpha": 0.1,
            "reg_lambda": 2.0,
            "verbose": -1,
        }
        self._model = lgb.LGBMRegressor(**params)
        self._model.fit(X_filled, y)
        preds = self._model.predict(X_filled)
        self._residual_std = float(np.std(y.values - preds)) if len(preds) > 1 else 0.5
        self.is_fitted = True
        return self

    def predict(self, X: pd.DataFrame) -> StockReactionForecastResult:
        X_filled = self._get_X(X)
        for col in self._feature_cols:
            if col not in X_filled.columns:
                X_filled[col] = 0.0
        X_filled = X_filled[self._feature_cols]
        pred = float(self._model.predict(X_filled)[0])
        ci_half = 1.645 * max(self._residual_std, 0.05)
        row = X.iloc[0].to_dict()
        return self._build_result(
            forecast_date=row.get("trade_date") or date.today(),
            predicted_return_pct=pred,
            predicted_return_lower_90=pred - ci_half,
            predicted_return_upper_90=pred + ci_half,
            row=row,
        )


# ─── Candidate Model 3: XGBoost (Active Non-Linear Ensemble) ─────────────────────────────────────

@ModelRegistry.register("stock_reaction_xgboost")
class StockReactionXGBoostModel(BaseStockReactionModel):
    """XGBoost gradient-boosted trees for stock reaction prediction."""

    def __init__(self, symbol: str = "AKBNK", window: str = "w2",
                 thresholds: Optional[ReturnThresholdProfile] = None):
        super().__init__("XGBoost Non-Linear Ensemble", symbol, window, thresholds)
        self._model = None
        self._residual_std: float = 0.5
        self._feature_cols: List[str] = []

    def fit(self, X: pd.DataFrame, y: pd.Series) -> "StockReactionXGBoostModel":
        try:
            from xgboost import XGBRegressor
        except ImportError:
            logger.warning("XGBoost not installed — skipping.")
            self.is_fitted = False
            return self
        X_filled = self._get_X(X)
        self._feature_cols = list(X_filled.columns)
        self._model = XGBRegressor(
            n_estimators=100, learning_rate=0.03, max_depth=3,
            subsample=0.85, colsample_bytree=0.85, reg_alpha=0.1, reg_lambda=2.0,
            verbosity=0, eval_metric="rmse",
        )
        self._model.fit(X_filled, y, verbose=False)
        preds = self._model.predict(X_filled)
        self._residual_std = float(np.std(y.values - preds)) if len(preds) > 1 else 0.5
        self.is_fitted = True
        return self

    def predict(self, X: pd.DataFrame) -> StockReactionForecastResult:
        if not self.is_fitted or self._model is None:
            row = X.iloc[0].to_dict()
            return self._build_result(row.get("trade_date") or date.today(), 0.0, -1.5, 1.5, row)
        X_filled = self._get_X(X)
        for col in self._feature_cols:
            if col not in X_filled.columns:
                X_filled[col] = 0.0
        pred = float(self._model.predict(X_filled[self._feature_cols])[0])
        ci_half = 1.645 * max(self._residual_std, 0.05)
        row = X.iloc[0].to_dict()
        return self._build_result(
            forecast_date=row.get("trade_date") or date.today(),
            predicted_return_pct=pred,
            predicted_return_lower_90=pred - ci_half,
            predicted_return_upper_90=pred + ci_half,
            row=row,
        )


# ─── Candidate Model 5: PyMC Bayesian GLM (optional) ─────────────────────────────────────────────

@ModelRegistry.register("stock_reaction_pymc")
class StockReactionPyMCModel(BaseStockReactionModel):
    """Full Bayesian GLM via PyMC with MCMC / MAP inference."""

    def __init__(self, symbol: str = "AKBNK", window: str = "w2",
                 thresholds: Optional[ReturnThresholdProfile] = None,
                 use_map: bool = True):
        super().__init__("PyMC Bayesian GLM (MAP)", symbol, window, thresholds)
        self.use_map = use_map
        self._alpha: float = 0.0
        self._betas: Optional[np.ndarray] = None
        self._sigma: float = 0.5
        self._scaler = StandardScaler()
        self._feature_cols: List[str] = []

    def _get_X(self, df: pd.DataFrame) -> pd.DataFrame:
        feat_cols = [c for c in df.columns if c.startswith("feat_")]
        return df[feat_cols].fillna(0.0)

    def fit(self, X: pd.DataFrame, y: pd.Series) -> "StockReactionPyMCModel":
        try:
            import pymc as pm
            import pytensor.tensor as pt
        except ImportError:
            logger.warning("PyMC not available — falling back to Bayesian Ridge.")
            fallback = StockReactionBayesianModel(self.symbol, self.window, self.thresholds)
            fallback.fit(X, y)
            self._alpha = 0.0
            self._feature_cols = fallback._feature_cols
            self.is_fitted = True
            return self

        X_filled = self._get_X(X)
        self._feature_cols = list(X_filled.columns)
        X_scaled = self._scaler.fit_transform(X_filled)
        y_vals = y.values.astype(float)

        with pm.Model() as _:
            alpha = pm.Normal("alpha", mu=0, sigma=1)
            betas = pm.Normal("betas", mu=0, sigma=1, shape=X_scaled.shape[1])
            sigma = pm.HalfNormal("sigma", sigma=1)
            mu = alpha + pt.dot(X_scaled, betas)
            pm.Normal("y_obs", mu=mu, sigma=sigma, observed=y_vals)
            if self.use_map:
                map_est = pm.find_MAP(progressbar=False)
                self._alpha = float(map_est["alpha"])
                self._betas = map_est["betas"]
                self._sigma = float(map_est["sigma"])

        self.is_fitted = True
        return self

    def predict(self, X: pd.DataFrame) -> StockReactionForecastResult:
        X_filled = self._get_X(X)
        for col in self._feature_cols:
            if col not in X_filled.columns:
                X_filled[col] = 0.0
        X_scaled = self._scaler.transform(X_filled[self._feature_cols])
        if self._betas is not None:
            pred_val = float(self._alpha + X_scaled @ self._betas)
        else:
            pred_val = self._alpha
        ci_half = 1.645 * max(self._sigma, 0.05)
        row = X.iloc[0].to_dict()
        return self._build_result(
            forecast_date=row.get("trade_date") or date.today(),
            predicted_return_pct=pred_val,
            predicted_return_lower_90=pred_val - ci_half,
            predicted_return_upper_90=pred_val + ci_half,
            row=row,
        )
