"""Unit tests for the GJR-GARCH-t / filtered historical simulation risk engine."""

import math

import numpy as np
import pandas as pd
import pytest

from mdk_trading_oracle.models import stochastic_risk as sr


def _simulate_gjr(n: int, omega=0.05, alpha=0.06, gamma=0.08, beta=0.86, nu=6.0, seed=7) -> np.ndarray:
    rng = np.random.default_rng(seed)
    z = rng.standard_t(nu, size=n) / math.sqrt(nu / (nu - 2.0))
    r = np.empty(n)
    s2 = omega / (1 - alpha - 0.5 * gamma - beta)
    for t in range(n):
        r[t] = math.sqrt(s2) * z[t]
        s2 = omega + (alpha + gamma * (r[t] < 0)) * r[t] ** 2 + beta * s2
    return r


@pytest.fixture(autouse=True)
def _no_calibration_file(monkeypatch, tmp_path):
    monkeypatch.setattr(sr, "CALIBRATION_PATH", tmp_path / "missing.yaml")
    sr.clear_volatility_cache()
    yield
    sr.clear_volatility_cache()


def test_gjr_garch_fit_recovers_persistence():
    r = _simulate_gjr(3000)
    p = sr.fit_garch_params(r)
    assert p.method == "GJR_GARCH_T"
    assert abs(p.persistence - 0.96) < 0.04
    assert p.gamma > 0  # leverage effect detected
    assert p.long_run_var == pytest.approx(0.05 / 0.04, rel=0.5)


def test_short_history_falls_back_to_ewma():
    p = sr.fit_garch_params(np.random.default_rng(1).normal(size=100))
    assert p.method == "EWMA"
    assert p.long_run_var is None


def test_term_structure_mean_reverts_and_scales_with_sqrt_time():
    p = sr.GarchParams(omega=0.1, alpha=0.05, gamma=0.0, beta=0.90, nu=None, method="GJR_GARCH_T")
    lr = p.long_run_var
    high = sr.variance_term_structure(4 * lr, p, 60)
    assert high[0] == pytest.approx(4 * lr)
    assert high[-1] < high[0] and high[-1] > lr

    state = sr.VolatilityState(params=p, sigma2_next=lr, std_resid=np.array([-1.0, 1.0]), n_obs=500, as_of_date="x")
    assert sr.horizon_sigma_pct(state, 10) == pytest.approx(math.sqrt(10 * lr))
    # EWMA has a flat term structure
    assert np.allclose(sr.variance_term_structure(2.0, sr.EWMA_PARAMS, 5), 2.0)


def test_corporate_action_jumps_are_dropped_from_unadjusted_prices():
    prices = pd.Series([100.0, 101.0, 102.0, 51.0, 52.0])  # 2:1 split on day 4
    r = sr.compute_log_returns_pct(prices, drop_corporate_actions=True)
    assert len(r) == 3
    assert r.abs().max() < 5.0


def test_forecast_distribution_is_ito_consistent_and_coherent():
    state = sr.fit_volatility_state(pd.Series(_simulate_gjr(1500)))
    out = sr.build_forecast_distribution(
        100.0, 2.0, state, n_days=10, deadband_pct=0.5, band_scale=1.0, center_shrinkage=1.0, seed_key="t"
    )
    assert out["vol_method"] == "GJR_GARCH_T"
    assert out["price_low"] < out["median_price"] < out["price_high"]
    assert out["p_up_pct"] + out["p_down_pct"] + out["p_flat_pct"] == pytest.approx(100.0, abs=0.2)
    assert out["position_side"] == "LONG"
    assert out["stop_price"] < 100.0
    assert out["p_target_before_stop_pct"] + out["p_stop_before_target_pct"] <= 100.0
    assert out["expected_shortfall_95_pct"] > 0

    again = sr.build_forecast_distribution(
        100.0, 2.0, state, n_days=10, deadband_pct=0.5, band_scale=1.0, center_shrinkage=1.0, seed_key="t"
    )
    assert again == out  # deterministic per (seed_key, as_of_date, horizon)


def test_ito_centering_mean_equals_target_and_median_carries_drag():
    p = sr.GarchParams(omega=0.0, alpha=0.0, gamma=0.0, beta=1.0, nu=None, method="EWMA")  # constant 3% daily vol
    pool = np.random.default_rng(2).normal(size=20000)
    state = sr.VolatilityState(params=p, sigma2_next=9.0, std_resid=(pool - pool.mean()) / pool.std(), n_obs=1, as_of_date="x")
    paths = sr.simulate_log_paths(state, 20, 40000, np.random.default_rng(4))
    gross = sr._center_gross_paths(paths, 2.0, 1.0)
    assert gross[:, -1].mean() == pytest.approx(1.02, rel=1e-9)
    # Median sits below the mean by ~sigma_h^2 / 2 = (3% * sqrt(20))^2 / 2 = 0.9%
    drag = (gross[:, -1].mean() - np.median(gross[:, -1])) * 100.0
    assert drag == pytest.approx(0.9, abs=0.3)


def test_center_shrinkage_and_flat_side():
    state = sr.fit_volatility_state(pd.Series(_simulate_gjr(1500)))
    out = sr.build_forecast_distribution(100.0, 5.0, state, 5, 0.5, band_scale=1.0, center_shrinkage=0.0)
    assert out["risk_center_return_pct"] == 0.0
    assert out["target_price"] == 105.0
    assert out["p_up_pct"] == pytest.approx(out["p_down_pct"], abs=6.0)

    flat = sr.build_forecast_distribution(100.0, 0.1, state, 5, 0.5, band_scale=1.0, center_shrinkage=1.0)
    assert flat["position_side"] == "FLAT"
    assert flat["stop_price"] is None and flat["p_target_before_stop_pct"] is None


def test_in_model_band_coverage_is_nominal():
    """Bands built from the true process should cover ~90% of realized outcomes."""
    p = sr.GarchParams(omega=0.05, alpha=0.06, gamma=0.08, beta=0.86, nu=6.0, method="GJR_GARCH_T")
    rng = np.random.default_rng(3)
    pool = rng.standard_t(6.0, size=5000) / math.sqrt(1.5)
    state = sr.VolatilityState(params=p, sigma2_next=1.5, std_resid=pool, n_obs=5000, as_of_date="x")
    out = sr.build_forecast_distribution(
        100.0, 0.0, state, 5, 0.5, band_scale=1.0, center_shrinkage=1.0, n_paths=20000, seed_key="a"
    )
    realized = sr.simulate_log_paths(state, 5, 20000, np.random.default_rng(99))[:, -1]
    prices = 100.0 * np.exp(realized / 100.0)
    coverage = np.mean((prices >= out["price_low"]) & (prices <= out["price_high"]))
    assert coverage == pytest.approx(0.90, abs=0.02)


def test_conformal_scale_recovers_dispersion_mismatch():
    rng = np.random.default_rng(5)
    n = 4000
    base = rng.normal(size=(n, 2000))
    q = np.quantile(base, sr.QUANTILE_GRID, axis=1).T  # unit-sigma model, in percent
    realized = rng.normal(size=n) * 1.3  # truth is 30% wider than the model
    incl = sr.inclusion_matrix(q, np.zeros(n), realized, 0.90)
    assert sr.fit_scale_for_coverage(incl, 0.90) == pytest.approx(1.3, abs=0.05)


def test_center_shrinkage_estimate_is_bounded():
    rng = np.random.default_rng(11)
    pred = rng.normal(scale=5.0, size=5000)
    realized = 0.2 * pred + rng.normal(size=5000)
    assert sr.fit_center_shrinkage(pred, realized) == pytest.approx(0.2, abs=0.02)
    assert sr.fit_center_shrinkage(pred, -pred) == 0.0
    assert sr.fit_center_shrinkage(pred, 3 * pred) == 1.0


def test_uncalibrated_defaults_and_written_calibration(tmp_path, monkeypatch):
    assert sr.get_band_scale(5) == 1.0
    assert sr.get_center_shrinkage(5) == 1.0
    path = tmp_path / "cal.yaml"
    monkeypatch.setattr(sr, "CALIBRATION_PATH", path)
    sr.write_band_calibration({"coverage": 0.9, "horizons": {"5d": {"scale": 1.2, "center_shrinkage": 0.1}}})
    assert sr.get_band_scale(5) == 1.2
    assert sr.get_center_shrinkage(5) == 0.1
    assert sr.get_band_scale(10) == 1.0


def test_size_forecast_direction_scale_and_t1_cap(tmp_path, monkeypatch):
    state = sr.fit_volatility_state(pd.Series(_simulate_gjr(1500)))
    up = sr.size_forecast(9.0, state, 1, seed_key="a")
    dn = sr.size_forecast(-0.4, state, 5, seed_key="a")
    assert up["expected_return_pct"] > 0 and dn["expected_return_pct"] < 0
    assert up["expected_return_pct"] <= sr.T1_CEILING  # T+1 never near the +/-10% limit
    assert up["model_expected_return_pct"] == 9.0
    assert sr.size_forecast(0.0, state, 1) is None and sr.size_forecast(1.0, None, 1) is None
    # Not lazy: even a tiny model output gets a realistic directional size
    tiny = sr.size_forecast(0.01, state, 5, seed_key="a")
    assert abs(tiny["expected_return_pct"]) > 0.5
    # Uncalibrated confidence defaults to 0.5; a reference makes larger signal-to-noise -> larger confidence
    assert up["sizing_confidence"] == 0.5
    path = tmp_path / "cal.yaml"
    monkeypatch.setattr(sr, "CALIBRATION_PATH", path)
    sr.write_sizing_reference({"horizons": {"1d": {"snr_quantiles": [0.0, 0.5, 1.0, 2.0, 4.0]}}})
    sr.write_band_calibration({"coverage": 0.9, "horizons": {}})  # must preserve the sizing section
    assert sr.sizing_confidence(4.0, 1) == 1.0 and sr.sizing_confidence(0.5, 1) == 0.25
    assert sr.sizing_confidence(1.0, 5) == 0.5  # horizon without a reference


def test_compress_t1_and_conviction_labels():
    assert sr.compress_t1_size(3.0) == 3.0
    assert 5.0 < sr.compress_t1_size(6.0) < sr.compress_t1_size(20.0) < sr.T1_CEILING
    assert sr.direction_conviction(-1.0) == "LOW" and sr.direction_conviction(5.0) == "MODERATE"
    assert sr.direction_conviction(15.0) == "HIGH"


def test_distribution_target_override_keeps_center():
    state = sr.fit_volatility_state(pd.Series(_simulate_gjr(1500)))
    base = sr.build_forecast_distribution(100.0, 3.0, state, 5, 0.5, band_scale=1.0, center_shrinkage=0.1, seed_key="t")
    sized = sr.build_forecast_distribution(100.0, 3.0, state, 5, 0.5, band_scale=1.0, center_shrinkage=0.1, seed_key="t", target_return_pct=1.5)
    assert sized["price_low"] == base["price_low"] and sized["p_up_pct"] == base["p_up_pct"]  # same calibrated distribution
    assert sized["target_price"] == 101.5 and base["target_price"] == 103.0
    assert sized["conviction"] in ("LOW", "MODERATE", "HIGH")
