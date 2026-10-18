"""Point-in-time T+1 direction model shared by the live gold forecast and the walk-forward ledger rebuild.

One definition of "what was knowable at the close of session T":

    training row  = features of session d  ->  label r(d -> d+1), realized at the close of d+1
    forecast origin = close of session T (row T); the forecast is for session T+1

A training row is usable at origin T only if its label is realized by then, i.e. d + 1 <= T. Row T itself (label
r(T -> T+1)) is never in the training set. `fit_t1_model` enforces this and records `train_label_end_date`, the
session whose close realized the last training label; it is always <= the origin date.

The ledger row for session T+1 grades the forecast made at origin T. Re-creating that forecast with a model fitted at
a later origin puts r(T -> T+1), the graded outcome, into the training set (the leak documented in
docs/RESEARCH_HANDOFF_2026-10.md §4.5).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

MIN_TRAIN_SESSIONS = 63
RAW_CLIP_PCT = 10.0
# Hyperparameter version of the production candidates below; bump when any of them changes.
T1_SPEC_VERSION = "v1"


@dataclass(frozen=True)
class T1Config:
    """A per-stock T+1 model configuration (model family, training window in sessions, feature mode)."""

    model: str
    lookback: int
    feature_mode: str = "FIFO_CORE"
    label: str = "crowned"
    selected_on: str | None = None  # date the configuration was selected (crowning date); None = never selected

    @property
    def spec(self) -> str:
        return f"{self.model}/{self.lookback}/{self.feature_mode}/{T1_SPEC_VERSION}"

    @classmethod
    def from_crowned(cls, meta: dict[str, Any], selected_on: str | None = None) -> "T1Config":
        return cls(
            model=str(meta.get("model", "XGBoost")),
            lookback=int(meta.get("training_lookback_sessions", 63)),
            feature_mode=str(meta.get("feature_mode") or "FIFO_CORE"),
            label="crowned",
            selected_on=selected_on,
        )


@dataclass
class T1Fit:
    model: Any
    origin_idx: int
    origin_date: str
    train_start_date: str
    train_label_end_date: str
    n_train: int
    data_sufficiency_status: str


def build_t1_model(name: str) -> tuple[Any, str]:
    """Production T+1 candidates; returns (unfitted model, sample-weight fit keyword)."""
    import lightgbm as lgb
    from sklearn.linear_model import BayesianRidge, HuberRegressor, Ridge
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler
    from xgboost import XGBRegressor

    if name == "XGBoost":
        model = XGBRegressor(n_estimators=45, max_depth=2, learning_rate=0.03, subsample=0.8, colsample_bytree=0.8, random_state=42, n_jobs=1)
        return model, "sample_weight"
    if name == "LightGBM":
        model = lgb.LGBMRegressor(
            n_estimators=45, max_depth=3, learning_rate=0.03, num_leaves=7, subsample=0.8, colsample_bytree=0.8, random_state=42, n_jobs=1, verbose=-1
        )
        return model, "sample_weight"
    if name == "BayesianRidge":
        return Pipeline([("scaler", StandardScaler()), ("regressor", BayesianRidge(max_iter=300))]), "regressor__sample_weight"
    if name == "Huber":
        return Pipeline([("scaler", StandardScaler()), ("regressor", HuberRegressor(epsilon=1.35, alpha=1.0, max_iter=300))]), "regressor__sample_weight"
    return Ridge(alpha=2.0, random_state=42), "sample_weight"


def t1_sample_weights(y: pd.Series | np.ndarray) -> np.ndarray:
    a = np.abs(np.asarray(y, dtype=float))
    return np.where(a >= 2.0, 6.0, np.where(a >= 1.0, 3.5, 0.5))


def t1_labels(df: pd.DataFrame) -> pd.Series:
    """Label of row d = split-adjusted close-to-close return d -> d+1 in % (NaN on the last row)."""
    from mdk_trading_oracle.data.silver.tertip_ml_forecaster import adjusted_price

    px = adjusted_price(df, "close")
    return (px.shift(-1) / px - 1.0) * 100.0


def _date(df: pd.DataFrame, idx: int) -> str:
    return str(df["trade_date"].iloc[idx])[:10]


def fit_t1_model(df: pd.DataFrame, origin_idx: int, cfg: T1Config, f_cols: list[str], labels: pd.Series | None = None) -> T1Fit | None:
    """Fit the T+1 model as it would have been fitted at the close of row `origin_idx`.

    `df` must have a 0..n-1 RangeIndex sorted by trade_date. Training rows: the last `cfg.lookback - 1` rows before
    the origin with complete features and a label realized by the origin close (row index <= origin_idx - 1).
    Returns None with fewer than min(MIN_TRAIN_SESSIONS, lookback - 1) usable rows (a 63-session window leaves 62
    training rows, as in production).
    """
    if labels is None:
        labels = t1_labels(df)
    lo = max(0, origin_idx - cfg.lookback + 1)
    cand = df.iloc[lo:origin_idx]  # rows lo .. origin_idx-1; their labels are realized by the origin close
    y = labels.iloc[lo:origin_idx]
    ok = cand[f_cols].notna().all(axis=1) & y.notna()
    X, y = cand.loc[ok, f_cols], y[ok]
    if len(X) < min(MIN_TRAIN_SESSIONS, cfg.lookback - 1):
        return None

    last_train_idx = int(X.index[-1])
    if last_train_idx + 1 > origin_idx:  # the label of the last row is realized at the close of row last_train_idx+1
        raise AssertionError(f"T+1 training label realized after the forecast origin ({_date(df, origin_idx)})")

    model, key = build_t1_model(cfg.model)
    model.fit(X, y, **{key: t1_sample_weights(y)})
    return T1Fit(
        model=model,
        origin_idx=origin_idx,
        origin_date=_date(df, origin_idx),
        train_start_date=_date(df, int(X.index[0])),
        train_label_end_date=_date(df, last_train_idx + 1),
        n_train=len(X),
        data_sufficiency_status="FULL" if len(X) >= cfg.lookback - 1 else "PARTIAL_HISTORY",
    )


def predict_t1_row(fit: T1Fit, df: pd.DataFrame, row_idx: int, f_cols: list[str]) -> dict[str, float] | None:
    """Raw (clipped) and ATR-calibrated T+1 return from the features of row `row_idx`."""
    from mdk_trading_oracle.data.silver.tertip_ml_forecaster import DEADBAND_PCT, calibrate_expected_return

    x = df.loc[[row_idx], f_cols]
    if x.isna().any(axis=None):
        return None
    raw = float(np.clip(fit.model.predict(x)[0], -RAW_CLIP_PCT, RAW_CLIP_PCT))
    atr = df["atr_14d_pct"].iloc[row_idx] if "atr_14d_pct" in df.columns else np.nan
    atr = float(atr) if pd.notna(atr) else 2.5
    return {"raw_return_pct": raw, "model_return_pct": calibrate_expected_return(raw, atr, deadband=DEADBAND_PCT)}


def forecast_t1(df: pd.DataFrame, origin_idx: int, cfg: T1Config, f_cols: list[str], labels: pd.Series | None = None) -> dict[str, Any] | None:
    """Point-in-time T+1 forecast at the close of row `origin_idx` (fit at that origin, predict from that row)."""
    fit = fit_t1_model(df, origin_idx, cfg, f_cols, labels)
    if fit is None:
        return None
    pred = predict_t1_row(fit, df, origin_idx, f_cols)
    if pred is None:
        return None
    return {
        **pred,
        "origin_date": fit.origin_date,
        "train_start_date": fit.train_start_date,
        "train_label_end_date": fit.train_label_end_date,
        "n_train": fit.n_train,
        "data_sufficiency_status": fit.data_sufficiency_status,
        "model_spec": cfg.spec,
    }


def prepare_feature_frame(df: pd.DataFrame, f_cols: list[str]) -> pd.DataFrame:
    """Sorted 0..n-1 frame with every feature column present (missing ones as 0.0, as in production)."""
    out = df.sort_values("trade_date").reset_index(drop=True).copy()
    for c in f_cols:
        if c not in out.columns:
            out[c] = 0.0
    return out


def direction(ret_pct: float | None, deadband: float) -> str:
    if ret_pct is None or not np.isfinite(ret_pct):
        return "FLAT"
    return "UP" if ret_pct > deadband else ("DOWN" if ret_pct < -deadband else "FLAT")


def strict_hit(pred_pct: float, actual_pct: float, deadband: float) -> bool:
    """Called a direction beyond the deadband and the session moved beyond the deadband the same way."""
    return (pred_pct > deadband and actual_pct > deadband) or (pred_pct < -deadband and actual_pct < -deadband)
