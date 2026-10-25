"""Guard tests for the 3D-30D swing forecaster: per-session forecasts and crowning may not see unrealized outcomes.

Synthetic data only (no database). The walk-forward test scrambles every price, flow and feature after the forecast
origin and requires an identical forecast; the crowning tests alter outcomes realized on/after the crowning date and
require the same champions, and check the selection rule against the production tournament script.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from mdk_trading_oracle.data.silver.tertip_multi_horizon_forecaster import evaluate_horizon_walk_forward
from mdk_trading_oracle.models.horizon_crowning import (
    FAMILIES,
    HORIZONS,
    candidate_configs,
    crown,
    option1_recency_score,
    simulate_crowning,
)

F_COLS = [f"feat_x{i}" for i in range(5)]
RAW_COLS = ["close_price", "adj_close_price", "high_price", "low_price", "total_turnover_tl", "mlb_flow", "big5_flow", "daily_return_pct"]


def synthetic_frame(n: int = 320, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    feats = rng.normal(size=(n, len(F_COLS)))
    ret = 0.6 * feats[:, 0] - 0.4 * feats[:, 1] + rng.normal(scale=1.2, size=n)
    close = 100.0 * np.exp(np.cumsum(np.r_[0.0, ret[:-1]]) / 100.0)
    df = pd.DataFrame(feats, columns=F_COLS)
    df["trade_date"] = pd.bdate_range("2024-01-02", periods=n)
    df["close_price"] = close
    df["adj_close_price"] = close
    df["high_price"] = close * 1.01
    df["low_price"] = close * 0.99
    df["total_turnover_tl"] = rng.uniform(1e8, 2e8, size=n)
    df["mlb_flow"] = rng.normal(scale=1e7, size=n)
    df["big5_flow"] = rng.normal(scale=1e7, size=n)
    df["daily_return_pct"] = pd.Series(close).pct_change().fillna(0.0).to_numpy()
    return df


def scramble_after(df: pd.DataFrame, origin_idx: int, seed: int = 99) -> pd.DataFrame:
    out = df.copy()
    rng = np.random.default_rng(seed)
    future = out.index > origin_idx
    k = int(future.sum())
    for c in F_COLS + RAW_COLS:
        out.loc[future, c] = rng.uniform(1.0, 500.0, size=k)
    return out


def _pred_at(df: pd.DataFrame, horizon: str, model: str, trade_date: str) -> float:
    res = evaluate_horizon_walk_forward(df.copy(), horizon, model, train_lookback_sessions=126, n_eval_sessions=len(df))
    row = next(r for r in res["ledger"] if r["trade_date"] == trade_date)
    return row["pred_return_pct"]


@pytest.mark.parametrize("model", ["Ridge", "BayesianRidge", "Huber", "LightGBM", "XGBoost"])
@pytest.mark.parametrize("horizon", ["3d", "30d"])
def test_walk_forward_forecast_invariant_to_future_data(model: str, horizon: str) -> None:
    df = synthetic_frame()
    origin_idx = 220
    d = str(df["trade_date"].iloc[origin_idx])[:10]
    base = _pred_at(df, horizon, model, d)
    assert _pred_at(scramble_after(df, origin_idx), horizon, model, d) == pytest.approx(base, abs=1e-9)


def test_walk_forward_detects_label_from_graded_window() -> None:
    """Sanity check of the invariance test: changing a close inside (t, t+n] must not move the forecast at t,
    while changing a close that a training label used (<= t) does."""
    df = synthetic_frame()
    origin_idx, n = 220, 10
    d = str(df["trade_date"].iloc[origin_idx])[:10]
    base = _pred_at(df, "10d", "Ridge", d)
    graded = df.copy()
    graded.loc[origin_idx + n, ["close_price", "adj_close_price"]] *= 1.15  # the graded outcome of origin t
    assert _pred_at(graded, "10d", "Ridge", d) == pytest.approx(base, abs=1e-9)
    known = df.copy()
    known.loc[origin_idx - 1, ["close_price", "adj_close_price"]] *= 1.15  # a realized training label
    assert abs(_pred_at(known, "10d", "Ridge", d) - base) > 1e-6


def synthetic_ledger(seed: int = 0, n: int = 260) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range("2025-01-02", periods=n + 40)
    rows = []
    for h in HORIZONS:
        k = int(h[:-1])
        for j, c in enumerate(candidate_configs()):
            p = 0.35 + 0.3 * rng.random()
            for i in range(n):
                rows.append({
                    "horizon": h, "config_key": c.key, "trade_date": dates[i].strftime("%Y-%m-%d"),
                    "target_date_end": dates[i + k].strftime("%Y-%m-%d"),
                    "is_hit": bool(rng.random() < p), "err_pct": float(rng.uniform(0.5, 6.0)),
                })
    return pd.DataFrame(rows)


def test_crowning_invariant_to_unrealized_outcomes() -> None:
    led = synthetic_ledger()
    crown_date = "2025-08-01"
    base = crown(led, crown_date)
    assert base
    altered = led.copy()
    future = altered["target_date_end"] >= crown_date
    altered.loc[future, "is_hit"] = ~altered.loc[future, "is_hit"]
    altered.loc[future, "err_pct"] = 0.0
    again = crown(altered, crown_date)
    assert {h: v["config"] for h, v in again.items()} == {h: v["config"] for h, v in base.items()}
    assert all(v["window_label_end_date"] < crown_date for v in base.values())


def test_simulated_track_grades_only_after_each_crowning() -> None:
    led = synthetic_ledger()
    track = simulate_crowning(led, ["2025-07-01", "2025-08-01", "2025-09-01"])
    assert not track.empty
    assert (track["trade_date"] >= track["crown_date"]).all()
    # one champion per horizon and crowning period
    assert (track.groupby(["crown_date", "horizon"])["config_key"].nunique() == 1).all()


def _production_tournament_module():
    path = Path(__file__).resolve().parents[1] / "scripts" / "run_multi_horizon_tournament.py"
    spec = importlib.util.spec_from_file_location("run_multi_horizon_tournament", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_selection_rule_matches_production_tournament() -> None:
    prod = _production_tournament_module()
    rng = np.random.default_rng(3)
    for n in (20, 45, 75, 90):
        hits = rng.random(n) < 0.55
        ledger = [{"is_hit": bool(h)} for h in hits]
        assert option1_recency_score(hits) == pytest.approx(prod.compute_option1_recency_score(ledger))
    assert {k: tuple(v) for k, v in prod.FAMILIES.items()} == FAMILIES
    assert tuple(prod.HORIZONS) == HORIZONS
    expected = [
        (fam, m, lb, mode)
        for fam, models in prod.FAMILIES.items()
        for m in models
        for mode in prod.FEATURE_MODES
        for _, lb in prod.LOOKBACKS
    ]
    assert [(c.family, c.model, c.lookback, c.feature_mode) for c in candidate_configs()] == expected


# ── models/horizon_forecast.py: point-in-time fit with explicit drift treatment ──────────────────────────────────────


def _arrays(df: pd.DataFrame, n_days: int):
    from mdk_trading_oracle.data.silver.tertip_multi_horizon_forecaster import engineer_multi_horizon_features
    from mdk_trading_oracle.models.horizon_forecast import horizon_labels

    feat, cols = engineer_multi_horizon_features(df.copy())
    return feat, feat[cols].to_numpy(float), horizon_labels(feat, n_days).to_numpy(float)


@pytest.mark.parametrize("model", ["Ridge", "LightGBM", "Huber"])
def test_horizon_fit_reproduces_production_walk_forward(model: str) -> None:
    from mdk_trading_oracle.data.silver.tertip_multi_horizon_forecaster import get_horizon_deadband
    from mdk_trading_oracle.models.horizon_forecast import calibrate_horizon, fit_horizon

    df = synthetic_frame()
    feat, X, y = _arrays(df, 10)
    db = get_horizon_deadband(10)
    for t in (150, 220, 300):
        d = str(df["trade_date"].iloc[t])[:10]
        fit = fit_horizon(X, y, t, 10, 126, model, db)
        ours = round(calibrate_horizon(fit.demeaned_pred, float(feat["atr_14d_pct"].iloc[t]), 10, db), 2)
        assert ours == pytest.approx(_pred_at(df, "10d", model, d), abs=1e-9)
        assert fit.train_label_end_idx <= t


@pytest.mark.parametrize("model", ["Ridge", "BayesianRidge", "LightGBM"])
def test_raw_target_equals_demeaned_plus_trailing_mean(model: str) -> None:
    """Training on raw labels = demeaned model + mean(label) for models with a free intercept (kappa = 1)."""
    from mdk_trading_oracle.data.silver.tertip_multi_horizon_forecaster import _build_model, _compute_sample_weights
    from mdk_trading_oracle.models.horizon_forecast import fit_horizon

    df = synthetic_frame()
    _, X, y = _arrays(df, 5)
    t, n, lb = 240, 5, 126
    fit = fit_horizon(X, y, t, n, lb, model, 0.56)
    Xt, yt = X[t - n - lb + 1 : t - n + 1], y[t - n - lb + 1 : t - n + 1]
    m = _build_model(model)
    m.fit(Xt, yt, sample_weight=_compute_sample_weights(yt, deadband=0.56))
    assert float(m.predict(X[t : t + 1])[0]) == pytest.approx(fit.demeaned_pred + fit.train_mean, abs=1e-6)


@pytest.mark.parametrize("n_days", [3, 15])
def test_horizon_fit_invariant_to_future_data(n_days: int) -> None:
    from mdk_trading_oracle.models.horizon_forecast import fit_horizon

    df = synthetic_frame()
    t = 230
    _, X, y = _arrays(df, n_days)
    _, Xs, ys = _arrays(scramble_after(df, t), n_days)
    a, b = fit_horizon(X, y, t, n_days, 252, "Ridge", 0.5), fit_horizon(Xs, ys, t, n_days, 252, "Ridge", 0.5)
    assert (a.demeaned_pred, a.train_mean) == pytest.approx((b.demeaned_pred, b.train_mean), abs=1e-9)
