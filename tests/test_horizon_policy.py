"""Guard tests for the staged monthly tournament (feature sets as stage 2) and its feature sets.

Synthetic data only. Stage 1 must reproduce the production rule; stage 2 must pick the better feature set for the
stage-1 model and window and may only see outcomes realized before the crown date; event and regime features at row
t may not change when everything after t is removed.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from mdk_trading_oracle.data.gold.tertip_multi_horizon_crowning import crown_month, policy_ledger
from mdk_trading_oracle.data.silver.event_regime_features import (
    EVENT_COLS,
    TOURNAMENT_FEATURE_SETS,
    add_event_features,
    add_regime_features,
    policy_key,
    split_key,
    tournament_policy,
)
from mdk_trading_oracle.models.horizon_crowning import (
    HORIZONS,
    candidate_configs,
    crown,
    crown_policy,
    crown_single,
    crown_staged,
)

ALL_H = ("1d",) + HORIZONS


def synthetic_ledger(seed: int = 0, n: int = 220, boost: dict[str, float] | None = None) -> pd.DataFrame:
    """Candidates for every feature set; `boost` raises one feature set's hit probability."""
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range("2025-01-02", periods=n + 40)
    rows = []
    for h in ALL_H:
        k = 1 if h == "1d" else int(h[:-1])
        for c in candidate_configs():
            base_p = 0.35 + 0.3 * rng.random()
            for fs in TOURNAMENT_FEATURE_SETS:
                p = min(0.95, base_p + (boost or {}).get(fs, 0.0))
                for i in range(n):
                    rows.append({"horizon": h, "config_key": policy_key(fs, c.key), "trade_date": dates[i].strftime("%Y-%m-%d"),
                                 "target_date_end": dates[i + k].strftime("%Y-%m-%d"),
                                 "is_hit": bool(rng.random() < p), "err_pct": float(rng.uniform(0.5, 6.0))})
    return pd.DataFrame(rows)


@pytest.fixture(scope="module")
def ledger() -> pd.DataFrame:
    return synthetic_ledger(boost={"events": 0.25})


def test_stage1_is_the_production_rule(ledger: pd.DataFrame) -> None:
    prod = ledger[~ledger["config_key"].str.contains("|", regex=False)]
    picks = crown_staged(ledger, "2025-09-01", TOURNAMENT_FEATURE_SETS)
    stage1 = {h: v["config"].key for h, v in crown(prod[prod["horizon"] != "1d"], "2025-09-01").items()}
    stage1["1d"] = crown_single(prod, "2025-09-01", "1d")["config"].key
    for h in ALL_H:
        model, lookback, _ = stage1[h].split("/")
        assert split_key(picks[h])[1].split("/")[:2] == [model, lookback]


def test_stage2_picks_the_better_feature_set(ledger: pd.DataFrame) -> None:
    picks = crown_staged(ledger, "2025-09-01", TOURNAMENT_FEATURE_SETS)
    assert sum(split_key(k)[0] == "events" for k in picks.values()) >= 5  # events boosted by +25 pp hit probability


def test_staged_crowning_ignores_unrealized_outcomes(ledger: pd.DataFrame) -> None:
    crown_date = "2025-09-01"
    base = crown_staged(ledger, crown_date, TOURNAMENT_FEATURE_SETS)
    altered = ledger.copy()
    future = altered["target_date_end"] >= crown_date
    altered.loc[future, "is_hit"] = ~altered.loc[future, "is_hit"]
    altered.loc[future, "err_pct"] = 0.0
    assert crown_staged(altered, crown_date, TOURNAMENT_FEATURE_SETS) == base


def test_policy_key_roundtrip() -> None:
    assert split_key(policy_key("events", "Ridge/252/FIFO_CORE")) == ("events", "Ridge/252/FIFO_CORE")
    assert split_key(policy_key("production", "Ridge/252/FIFO_CORE")) == ("production", "Ridge/252/FIFO_CORE")


def synthetic_frame(n: int = 160) -> tuple[pd.DataFrame, dict]:
    rng = np.random.default_rng(4)
    dates = pd.bdate_range("2025-01-02", periods=n)
    df = pd.DataFrame({"trade_date": dates, "close_price": 100 * np.exp(np.cumsum(rng.normal(0, 0.01, n)))})
    for c in ("feat_price_vs_sma20_pct", "feat_price_vs_sma50_pct", "feat_mlb_tertip_share"):
        df[c] = rng.normal(size=n)
    d = [x.strftime("%Y-%m-%d") for x in dates]
    ev = {
        "mpc": [d[20], d[60], d[100], d[150]],
        "rates": pd.DataFrame({"d": d, "rate_change": [(-1.0 if i == 61 else 0.0) for i in range(n)]}),
        "div": pd.DataFrame({"symbol": ["X", "X"], "d": [d[70], d[140]], "gross_dividend_tl": [2.0, 3.0]}),
        "rebal": pd.DataFrame({"d": [d[65], d[130]], "ins": ["X", ""], "outs": ["", "X"]}),
        "bench": pd.DataFrame({"trade_date": dates, "close_price": 1000 * np.exp(np.cumsum(rng.normal(0, 0.01, n)))}),
    }
    return df, ev


@pytest.mark.parametrize("t", [55, 64, 99, 125])
def test_event_and_regime_features_are_point_in_time(t: int) -> None:
    df, ev = synthetic_frame()
    full_e, cut_e = add_event_features(df, "X", ev), add_event_features(df.iloc[: t + 1], "X", ev)
    assert np.allclose(full_e.loc[t, EVENT_COLS].to_numpy(float), cut_e.loc[t, EVENT_COLS].to_numpy(float))
    full_r, cut_r = add_regime_features(df, ev), add_regime_features(df.iloc[: t + 1], ev)
    cols = [c for c in full_r.columns if c.startswith("feat_regime") or c.endswith("_x_down")]
    assert np.allclose(full_r.loc[t, cols].to_numpy(float), cut_r.loc[t, cols].to_numpy(float), equal_nan=True)
    assert "feat_price_vs_sma50_pct" not in full_r.columns


def test_policy_crowning_reproduces_production_rule_on_production_pools() -> None:
    led = synthetic_ledger()
    prod = led[~led["config_key"].str.contains("|", regex=False)]
    pol = crown_policy(prod, "2025-09-01")
    ref = {h: v["config"].key for h, v in crown(prod[prod["horizon"] != "1d"], "2025-09-01").items()}
    assert {h: pol[h] for h in HORIZONS} == ref
    assert pol["1d"] == crown_single(prod, "2025-09-01", "1d")["config"].key


def test_tournament_policy_config_routes_priority_and_default_stocks(ledger: pd.DataFrame, monkeypatch) -> None:
    import mdk_trading_oracle.data.silver.event_regime_features as erf

    cfg = dict(tournament_policy())
    assert set(cfg["feature_sets"]) == set(TOURNAMENT_FEATURE_SETS)
    cfg["priority_stocks"] = ["PRIO"]  # the shipped config lists none; route one synthetic stock through stage 2
    monkeypatch.setattr(erf, "_POLICY", cfg)
    priority = "PRIO"
    picks_p = crown_month(priority, ledger, "2025-09-01")
    for h, key in picks_p.items():
        assert split_key(key)[0] in cfg["stage2_allowed"][h]
    picks_d = crown_month("ZZZZ", ledger, "2025-09-01")  # not a priority stock -> default policy
    for h, spec in cfg["default_policy"].items():
        assert split_key(picks_d[h])[0] == spec["feature_set"]
        if spec.get("exclude_lookback") is not None:
            assert split_key(picks_d[h])[1].split("/")[1] != str(spec["exclude_lookback"])
    pool = policy_ledger(ledger)
    assert set(pool["horizon"]) == set(cfg["default_policy"])
