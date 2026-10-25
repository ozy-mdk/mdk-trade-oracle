"""Point-in-time N-day swing forecast (3D-30D) with an explicit drift treatment.

One definition of "what was knowable at the close of session t" for an N-session horizon:

    training row     = features of session i  ->  label r(i -> i+N), realized at the close of i+N
    forecast origin  = close of session t; usable training rows satisfy i + N <= t

This reproduces production `evaluate_horizon_walk_forward` (same models, sample weights, ATR calibration and bounds)
but returns the demeaned model output and the trailing label mean separately, so drift treatments can be compared on
one fit:

    pred(kappa) = model_on_demeaned_target + kappa * mean(trailing labels)

kappa = 0 is production's zero-drift demeaning, kappa = 1 is training on raw targets (exact for models with a free,
unpenalised intercept fitted by weighted least squares: Ridge, BayesianRidge, LightGBM with L2 loss), and the
drift-only baseline is kappa = 1 with no features.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
import pandas as pd

MIN_TRAIN_SESSIONS = 63
MIN_TRAIN_ROWS = 15

# Production swing forecast (docs/RESEARCH_HANDOFF_2026-10.md §0.8): no model beat a constant UP call out of sample,
# and no feature set carried stable information beyond the stock's drift. The published N-day forecast is the
# trailing drift (mean realized N-day return over the last DRIFT_LOOKBACK sessions whose outcome is known), shrunk by
# DRIFT_SHRINKAGE = the out-of-sample through-origin slope of realized on drift (2024: 0.55-0.62, 2025-26: 0.29-0.48).
DRIFT_LOOKBACK = 252
DRIFT_SHRINKAGE = 0.5
DRIFT_MODEL_NAME = "Drift"


@dataclass
class HorizonFit:
    origin_idx: int
    demeaned_pred: float  # model output on the demeaned target (production)
    train_mean: float  # mean label over the training window (trailing drift)
    n_train: int
    train_label_end_idx: int  # row whose close realized the last training label (always <= origin)


def horizon_labels(df: pd.DataFrame, n_days: int) -> pd.Series:
    """Label of row i = split-adjusted close-to-close return i -> i+N in % (NaN for the last N rows)."""
    from mdk_trading_oracle.data.silver.tertip_ml_forecaster import adjusted_price

    px = adjusted_price(df, "close")
    return (px.shift(-n_days) - px) / px * 100.0


def fit_horizon(
    X: np.ndarray,
    y: np.ndarray,
    origin_idx: int,
    n_days: int,
    lookback: int,
    model_name: str,
    deadband: float,
) -> HorizonFit | None:
    """Fit at the close of row `origin_idx` on rows i <= origin_idx - N (last `lookback` of them) and predict the origin row."""
    from mdk_trading_oracle.data.silver.tertip_multi_horizon_forecaster import (
        _build_model,
        _compute_sample_weights,
        _fit_model,
    )

    train_end = origin_idx - n_days
    if train_end + 1 < MIN_TRAIN_SESSIONS:
        return None
    lo = train_end - min(train_end + 1, lookback) + 1
    Xt, yt = X[lo : train_end + 1], y[lo : train_end + 1]
    ok = ~np.isnan(yt)
    Xt, yt = Xt[ok], yt[ok]
    if len(yt) < MIN_TRAIN_ROWS:
        return None
    last_label_idx = lo + int(np.flatnonzero(ok)[-1]) + n_days
    if last_label_idx > origin_idx:
        raise AssertionError("training label realized after the forecast origin")

    mean_y = float(np.mean(yt))
    w = _compute_sample_weights(yt, deadband=deadband)
    model = _build_model(model_name)
    _fit_model(model, Xt, yt - mean_y, w)
    pred = float(model.predict(X[origin_idx : origin_idx + 1])[0])
    return HorizonFit(origin_idx, pred, mean_y, len(yt), last_label_idx)


def calibrate_horizon(raw_pct: float, atr_pct: float, n_days: int, deadband: float) -> float:
    """Production ATR calibration and horizon bound of a raw N-day model output."""
    from mdk_trading_oracle.data.silver.tertip_ml_forecaster import calibrate_expected_return

    atr = atr_pct if np.isfinite(atr_pct) else 2.2
    pred = calibrate_expected_return(raw_pct, atr * math.sqrt(n_days / 2.0), deadband=deadband)
    bound = min(25.0, 5.0 * math.sqrt(n_days))
    return max(-bound, min(bound, pred))


@dataclass
class DriftForecast:
    origin_idx: int
    drift_pct: float  # mean realized N-day return over the trailing window (labels realized by the origin close)
    expected_return_pct: float  # DRIFT_SHRINKAGE * drift, bounded to the horizon cap
    n_train: int
    train_label_end_idx: int


def drift_forecast(
    y: np.ndarray,
    origin_idx: int,
    n_days: int,
    lookback: int = DRIFT_LOOKBACK,
    shrinkage: float = DRIFT_SHRINKAGE,
) -> DriftForecast | None:
    """N-day drift forecast at the close of row `origin_idx` from labels r(i -> i+N) with i + N <= origin_idx."""
    train_end = origin_idx - n_days
    if train_end + 1 < MIN_TRAIN_SESSIONS:
        return None
    lo = train_end - min(train_end + 1, lookback) + 1
    window = y[lo : train_end + 1]
    ok = np.isfinite(window)
    if ok.sum() < MIN_TRAIN_ROWS:
        return None
    last_label_idx = lo + int(np.flatnonzero(ok)[-1]) + n_days
    if last_label_idx > origin_idx:
        raise AssertionError("drift label realized after the forecast origin")
    drift = float(window[ok].mean())
    bound = min(25.0, 5.0 * math.sqrt(n_days))
    expected = max(-bound, min(bound, shrinkage * drift))
    return DriftForecast(origin_idx, drift, expected, int(ok.sum()), last_label_idx)
