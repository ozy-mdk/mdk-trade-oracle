"""Guard tests for the Intraday Session Forecaster: no look-ahead, frozen signals, grading reads the published grid."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from mdk_trading_oracle.models import intraday_session as m

NAMES = [m.RULE, "LEAN_MLB|Ridge|63", "LEAN_MLB|XGBoost|126", "LEAN_INST|LightGBM|63", "OLD_39|Huber|126"]


def synthetic_panel(n: int = 420, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    cols = sorted({c for fs in m.feature_sets().values() for c in fs})
    p = pd.DataFrame(rng.normal(size=(n, len(cols))), columns=cols)
    p["trade_date"] = pd.bdate_range("2023-01-02", periods=n)
    p["prev_date"] = p["trade_date"].shift(1)
    p["morning_ret"] = rng.normal(scale=1.2, size=n)
    p["target"] = 0.3 * p["feat_mlb_turnover_intensity"] + 0.2 * p["morning_ret"] + rng.normal(scale=1.0, size=n)
    p["sigma_pct"] = 2.0
    p["usable"] = True
    p["symbol"] = "TEST"
    return p.dropna(subset=["prev_date"]).reset_index(drop=True)


def scramble_from(p: pd.DataFrame, idx: int, seed: int = 7) -> pd.DataFrame:
    """Everything unknown when the evening job publishes row `idx`: its morning move / target, and all later rows."""
    q = p.copy()
    rng = np.random.default_rng(seed)
    q.loc[idx:, ["morning_ret", "target"]] = rng.uniform(-9, 9, size=(len(q) - idx, 2))
    later = q.index > idx
    feats = [c for c in q.columns if c.startswith("feat_") or c == "mlb_late_share"]
    q.loc[later, feats] = rng.uniform(-50, 50, size=(int(later.sum()), len(feats)))
    return q


def test_grids_ignore_the_forecast_day_and_the_future() -> None:
    p = synthetic_panel()
    idx = 380
    base = m.candidate_grids(p, idx, names=NAMES)
    q = scramble_from(p, idx)
    other = m.candidate_grids(q, idx, names=NAMES)
    assert set(base) == set(other) and len(base) == len(NAMES)
    for name in base:
        np.testing.assert_allclose(other[name], base[name], rtol=1e-9, atol=1e-12)


def test_rule_state_uses_only_earlier_rows() -> None:
    p = synthetic_panel()
    idx = 380
    a = m.rule_state(p).loc[idx]
    b = m.rule_state(scramble_from(p, idx)).loc[idx]
    keep = ["mu_x", "sd_x", "mu_m", "sd_m", "hi", "lo", "size", "z_x"]
    np.testing.assert_allclose(b[keep].astype(float), a[keep].astype(float), rtol=1e-9, atol=1e-12)


def test_crowning_ignores_results_from_the_crowning_month() -> None:
    rng = np.random.default_rng(3)
    days = pd.bdate_range("2024-01-01", "2024-08-30")
    rows = [{"candidate": c, "session_date": d, "forecast_pct": rng.normal(), "actual_pct": rng.normal(), "sigma_pct": 2.0, "traded": True}
            for d in days for c in NAMES]
    r = pd.DataFrame(rows)
    month = pd.Timestamp("2024-07-01")
    a = m.crown(r, month)
    r2 = r.copy()
    late = r2["session_date"] >= month
    r2.loc[late, "actual_pct"] = np.sign(r2.loc[late, "forecast_pct"]) * 5  # a perfect future for everyone
    b = m.crown(r2, month)
    assert (a.name, a.hit_rate_pct, a.trades) == (b.name, b.hit_rate_pct, b.trades)


def test_grading_reads_the_published_grid() -> None:
    from mdk_trading_oracle.data.gold.intraday_session import grade

    grid = m.GRID.astype(float).copy()  # forecast = morning move: 1.23% clears the regression trade size
    sig = {"symbol": "TEST", "session_date": pd.Timestamp("2026-10-19").date(), "source": "LIVE", "candidate": "LEAN_MLB|Ridge|63",
           "grid_forecast_pct": list(grid)}
    row = pd.Series({"usable": True, "morning_ret": 1.23, "target": 0.4, "open_price": 100.0, "price_1130": 101.23, "close_price": 101.63})
    g = grade(sig, row)
    assert g["forecast_pct"] == pytest.approx(m.forecast_at(grid, 1.23))
    assert g["call"] == 1 and g["hit"] is True


def test_levels_match_where_the_grid_trades() -> None:
    grid = np.where(m.GRID >= 1.0, 1.5, np.where(m.GRID <= -0.8, -1.5, 0.5))  # 0.5% is below the regression trade size
    up, dn = m.levels(grid, "LEAN_MLB|Ridge|63")
    assert up == pytest.approx(1.0) and dn == pytest.approx(-0.8)


def _results(hits: dict[str, float], n: int = 100) -> pd.DataFrame:
    rows = []
    days = pd.bdate_range("2024-01-01", periods=n)
    for c, rate in hits.items():
        for i, d in enumerate(days):
            right = i < rate * n
            rows.append({"candidate": c, "session_date": d, "forecast_pct": 1.5, "actual_pct": 1.0 if right else -1.0, "sigma_pct": 2.0})
    return pd.DataFrame(rows)


def test_rule_stays_champion_unless_a_challenger_clearly_wins() -> None:
    month = pd.Timestamp("2024-07-01")
    assert m.crown(_results({m.RULE: 0.55, "LEAN_MLB|Ridge|63": 0.56}), month).name == m.RULE  # +1 pp is not enough
    assert m.crown(_results({m.RULE: 0.55, "LEAN_MLB|Ridge|63": 0.60}), month).name == "LEAN_MLB|Ridge|63"


def test_published_signals_are_frozen() -> None:
    from mdk_trading_oracle.core.db import PostgresManager
    from mdk_trading_oracle.data.gold import intraday_session as g

    db = PostgresManager(in_memory=True)
    try:
        g.ensure_tables(db)
        base = {c: None for c in g.SIGNAL_COLS}
        base.update({"symbol": "TEST", "session_date": pd.Timestamp("2026-10-19").date(), "source": "LIVE", "candidate": m.RULE, "buy_level_pct": 1.0})
        g.insert_signals(db, [base], overwrite=False)
        g.insert_signals(db, [{**base, "buy_level_pct": 9.9}], overwrite=False)
        assert db.query_df("SELECT buy_level_pct FROM gold_intraday_session_signals")["buy_level_pct"].iloc[0] == pytest.approx(1.0)
        g.insert_signals(db, [{**base, "buy_level_pct": 2.0}], overwrite=True)  # explicit restatement
        assert db.query_df("SELECT buy_level_pct FROM gold_intraday_session_signals")["buy_level_pct"].iloc[0] == pytest.approx(2.0)
    finally:
        db.close()
