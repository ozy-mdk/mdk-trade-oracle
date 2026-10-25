"""Guard tests for the per-horizon production policy (event / regime feature sets, policy crowning).

Synthetic data only. Policy crowning must reproduce the production rule when every horizon uses the production pool;
event and regime features at row t may not change when everything after t is removed; the policy ledger must pick
each horizon's pool and its lookback exclusion.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from mdk_trading_oracle.data.gold.tertip_multi_horizon_crowning import policy_ledger
from mdk_trading_oracle.data.silver.event_regime_features import (
    EVENT_COLS,
    HORIZON_POLICY,
    add_event_features,
    add_regime_features,
    policy_key,
    split_key,
)
from mdk_trading_oracle.models.horizon_crowning import HORIZONS, candidate_configs, crown, crown_policy, crown_single


def synthetic_ledger(seed: int = 0, n: int = 220, prefix: dict[str, str] | None = None) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range("2025-01-02", periods=n + 40)
    rows = []
    for h in ("1d",) + HORIZONS:
        k = 1 if h == "1d" else int(h[:-1])
        for c in candidate_configs():
            p = 0.35 + 0.3 * rng.random()
            key = policy_key((prefix or {}).get(h, "production"), c.key)
            for i in range(n):
                rows.append({"horizon": h, "config_key": key, "trade_date": dates[i].strftime("%Y-%m-%d"),
                             "target_date_end": dates[i + k].strftime("%Y-%m-%d"),
                             "is_hit": bool(rng.random() < p), "err_pct": float(rng.uniform(0.5, 6.0))})
    return pd.DataFrame(rows)


@pytest.mark.parametrize("crown_date", ["2025-07-01", "2025-09-01"])
def test_policy_crowning_reproduces_production_rule(crown_date: str) -> None:
    led = synthetic_ledger()
    pol = crown_policy(led, crown_date)
    prod = crown(led[led["horizon"] != "1d"], crown_date)
    one = crown_single(led, crown_date, "1d")
    assert {h: v["config"].key for h, v in prod.items()} == {h: pol[h] for h in HORIZONS}
    assert pol["1d"] == one["config"].key


def test_policy_crowning_handles_prefixed_pools_and_exclusions() -> None:
    led = synthetic_ledger(prefix={"1d": "events", "3d": "events", "5d": "regime_no_sma50", "10d": "regime_no_sma50"})
    pol = policy_ledger(led)
    for h, (fs, excl, _) in HORIZON_POLICY.items():
        keys = pol.loc[pol["horizon"] == h, "config_key"].unique()
        assert len(keys) and all(split_key(k)[0] == fs for k in keys)
        if excl is not None:
            assert not any(split_key(k)[1].split("/")[1] == str(excl) for k in keys)
    champs = crown_policy(pol, "2025-09-01")
    assert set(champs) == set(HORIZON_POLICY)
    assert split_key(champs["1d"])[0] == "events" and split_key(champs["10d"])[0] == "regime_no_sma50"


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
