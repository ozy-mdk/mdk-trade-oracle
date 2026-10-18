"""Guard tests: a T+1 forecast made at the close of session T must not depend on anything after T.

Synthetic data only (no database). The future-invariance test scrambles every price, flow and feature after the
forecast origin and requires an identical forecast (to float rounding); the leak-sensitivity test shows the same check catches the
§4.5 ledger leak (model fitted one session later, predicting the origin row).
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from mdk_trading_oracle.models.t1_forecast import (
    T1Config,
    fit_t1_model,
    forecast_t1,
    predict_t1_row,
    t1_labels,
)

F_COLS = [f"feat_{i}" for i in range(6)]


def synthetic_frame(n: int = 260, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    feats = rng.normal(size=(n, len(F_COLS)))
    # next-day return depends on today's features, so models learn a non-trivial fit
    ret = 0.8 * feats[:, 0] - 0.5 * feats[:, 1] + rng.normal(scale=1.0, size=n)
    close = 100.0 * np.exp(np.cumsum(np.r_[0.0, ret[:-1]]) / 100.0)
    df = pd.DataFrame(feats, columns=F_COLS)
    df["trade_date"] = pd.bdate_range("2024-01-02", periods=n)
    df["close_price"] = close
    df["adj_close_price"] = close
    df["atr_14d_pct"] = 2.0
    return df


def scramble_after(df: pd.DataFrame, origin_idx: int, seed: int = 99) -> pd.DataFrame:
    out = df.copy()
    rng = np.random.default_rng(seed)
    future = out.index > origin_idx
    k = int(future.sum())
    for c in F_COLS + ["close_price", "adj_close_price", "atr_14d_pct"]:
        out.loc[future, c] = rng.uniform(1.0, 500.0, size=k)
    return out


@pytest.mark.parametrize("origin_idx", [130, 180, 258])
def test_training_labels_end_by_origin(origin_idx: int) -> None:
    df = synthetic_frame()
    fit = fit_t1_model(df, origin_idx, T1Config("Ridge", 126), F_COLS)
    assert fit is not None
    assert fit.train_label_end_date <= fit.origin_date
    assert fit.n_train == 125  # lookback window ending at the origin, origin row excluded


def test_ledger_grades_forecast_from_previous_origin() -> None:
    df = synthetic_frame()
    labels = t1_labels(df)
    for origin_idx in range(70, len(df) - 1):
        fc = forecast_t1(df, origin_idx, T1Config("Ridge", 63), F_COLS, labels)
        graded_session = str(df["trade_date"].iloc[origin_idx + 1])[:10]
        assert fc["train_label_end_date"] <= fc["origin_date"] < graded_session


@pytest.mark.parametrize("model", ["Ridge", "BayesianRidge", "Huber", "XGBoost", "LightGBM"])
def test_forecast_invariant_to_future_data(model: str) -> None:
    df = synthetic_frame()
    origin_idx = 200
    cfg = T1Config(model, 126)
    base = forecast_t1(df, origin_idx, cfg, F_COLS)
    scrambled = forecast_t1(scramble_after(df, origin_idx), origin_idx, cfg, F_COLS)
    assert base is not None and scrambled is not None
    assert scrambled["raw_return_pct"] == pytest.approx(base["raw_return_pct"], rel=1e-9, abs=1e-12)
    assert scrambled["model_return_pct"] == base["model_return_pct"]


def test_invariance_check_catches_the_reconstructed_ledger_leak() -> None:
    """The §4.5 pattern (fit at origin T+1, predict row T) changes when the graded close is altered."""
    df = synthetic_frame()
    origin_idx = 200
    cfg = T1Config("Ridge", 126)

    def leaky(frame: pd.DataFrame) -> float:
        fit = fit_t1_model(frame, origin_idx + 1, cfg, F_COLS)
        return predict_t1_row(fit, frame, origin_idx, F_COLS)["raw_return_pct"]

    moved = df.copy()
    moved.loc[origin_idx + 1, ["close_price", "adj_close_price"]] *= 1.08  # the outcome being graded
    assert abs(leaky(df) - leaky(moved)) > 1e-3
    clean = forecast_t1(df, origin_idx, cfg, F_COLS)["raw_return_pct"]
    assert forecast_t1(moved, origin_idx, cfg, F_COLS)["raw_return_pct"] == pytest.approx(clean, rel=1e-9, abs=1e-12)


def test_insufficient_history_returns_none() -> None:
    assert forecast_t1(synthetic_frame(), 40, T1Config("Ridge", 252), F_COLS) is None


@pytest.mark.parametrize("model_type", ["ridge", "xgboost"])
def test_arena_predictions_invariant_to_graded_session(model_type: str) -> None:
    """Arena ledger row for session t: its ML forecast may not depend on session t or later (fixed model, no selection)."""
    from mdk_trading_oracle.data.silver.tertip_ml_forecaster import run_30d_walk_forward_arena

    df = synthetic_frame(200)
    for c in ["mlb_flow", "big5_flow", "kamu_flow"]:
        df[c] = 1.0
    df["daily_return_pct"] = df["close_price"].pct_change().fillna(0.0)
    first_graded = len(df) - 3

    def ml_preds(frame: pd.DataFrame) -> list[float]:
        ledger, _, _ = run_30d_walk_forward_arena(frame.copy(), n_sessions=3, model_type=model_type, feature_cols=F_COLS, train_lookback_sessions=126)
        return [r["ml_pred_return_pct"] for r in ledger]

    base = ml_preds(df)
    scrambled = ml_preds(scramble_after(df, first_graded - 1))
    assert scrambled[0] == pytest.approx(base[0], abs=1e-9)
