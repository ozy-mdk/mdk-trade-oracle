"""Point-in-time crowning for the multi-horizon (3D-30D) swing forecaster.

The per-session walk-forward in `evaluate_horizon_walk_forward` is clean: at origin t it trains on rows i <= t - n,
whose labels r(i -> i+n) are realized by the close of t. The leak in the displayed 3D-30D track records is the
*selection*: the former tournament script picked, per stock and horizon, the best of 75 configurations
(5 models x 5 lookbacks x 3 feature modes, family first) on the most recent 90 graded sessions, and the hub then showed
a 180-session ledger of that champion, half of which was the crowning window itself.

This module reproduces the production selection rule but lets it see only ledger rows whose outcome is realized
before the crowning date (target_date_end < crown_date). Crowning repeatedly (e.g. monthly) and grading each champion
only on sessions after its crowning date gives an honest track record of the whole procedure, selection included.

Ledger frame columns used here: horizon, config_key, trade_date, target_date_end, is_hit, err_pct.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

HORIZONS: tuple[str, ...] = ("3d", "5d", "10d", "15d", "30d")
FAMILIES: dict[str, tuple[str, ...]] = {
    "GBDT": ("LightGBM", "XGBoost"),
    "Regularized Linear": ("Ridge", "BayesianRidge"),
    "Robust M-Estimator": ("Huber",),
}
HORIZON_WEIGHTS: dict[str, float] = {"10d": 0.35, "5d": 0.25, "3d": 0.20, "15d": 0.12, "30d": 0.08}
LOOKBACKS: tuple[tuple[str, int], ...] = (("3m", 63), ("6m", 126), ("12m", 252), ("18m", 378), ("24m", 504))
FEATURE_MODES: tuple[str, ...] = ("FIFO_CORE", "FIFO_PLUS_YTD", "YTD_SUBSTITUTION")
CROWN_EVAL_SESSIONS = 90


@dataclass(frozen=True)
class HorizonConfig:
    family: str
    model: str
    lookback_label: str
    lookback: int
    feature_mode: str

    @property
    def key(self) -> str:
        return f"{self.model}/{self.lookback}/{self.feature_mode}"


def candidate_configs() -> list[HorizonConfig]:
    """All tournament candidates in production iteration order (family -> model -> feature mode -> lookback).

    The order matters: production crowns with `max()`, which keeps the first of tied candidates.
    """
    return [
        HorizonConfig(fam, model, lb_label, lb, mode)
        for fam, models in FAMILIES.items()
        for model in models
        for mode in FEATURE_MODES
        for lb_label, lb in LOOKBACKS
    ]


def option1_recency_score(hits: np.ndarray) -> tuple[float, float, float, float]:
    """0.50 * hit% (last 30) + 0.30 * hit% (30-60) + 0.20 * hit% (60-90), as in the production tournament."""
    hits = np.asarray(hits, dtype=float)
    n = len(hits)
    if n == 0:
        return 0.0, 0.0, 0.0, 0.0
    b1 = hits[-30:]
    r1 = b1.mean() * 100.0
    b2 = hits[-60:-30] if n >= 60 else (hits[:-30] if n > 30 else hits[:0])
    r2 = b2.mean() * 100.0 if len(b2) else r1
    b3 = hits[-90:-60] if n >= 90 else (hits[:-60] if n > 60 else hits[:0])
    r3 = b3.mean() * 100.0 if len(b3) else r2
    return 0.50 * r1 + 0.30 * r2 + 0.20 * r3, r1, r2, r3


def recency_weighted_mae(err: np.ndarray) -> float:
    """MAE with linspace(0.5, 1.5) recency weights (tie-break in production crowning), rounded as stored."""
    err = np.asarray(err, dtype=float)
    if len(err) == 0:
        return 0.0
    w = np.linspace(0.5, 1.5, len(err)) if len(err) > 1 else np.ones(1)
    return round(float(np.average(err, weights=w / w.mean())), 2)


def score_window(ledger: pd.DataFrame, crown_date: str, eval_sessions: int = CROWN_EVAL_SESSIONS) -> pd.DataFrame:
    """Per (horizon, config_key) tournament scores on the last `eval_sessions` rows realized before `crown_date`."""
    known = ledger[ledger["target_date_end"].astype(str) < crown_date]
    rows = []
    for (h, key), g in known.groupby(["horizon", "config_key"], sort=False):
        g = g.sort_values("trade_date").iloc[-eval_sessions:]
        opt1, r1, r2, r3 = option1_recency_score(g["is_hit"].to_numpy())
        rows.append({
            "horizon": h, "config_key": key, "opt1_score": opt1, "r_0_30": r1, "r_30_60": r2, "r_60_90": r3,
            "mae_pct": recency_weighted_mae(g["err_pct"].to_numpy()), "n_window": len(g),
            "window_end_date": str(g["trade_date"].iloc[-1])[:10],
            "window_label_end_date": str(g["target_date_end"].max())[:10],
        })
    return pd.DataFrame(rows)


def crown(
    ledger: pd.DataFrame,
    crown_date: str,
    configs: list[HorizonConfig] | None = None,
    eval_sessions: int = CROWN_EVAL_SESSIONS,
    min_window: int = 60,
) -> dict[str, dict]:
    """Two-stage family-first crowning (production rule) using only outcomes realized before `crown_date`.

    Returns {horizon: {"config": HorizonConfig, "family": str, "opt1_score": float, "window_label_end_date": str}};
    empty when any horizon has fewer than `min_window` realized rows for some candidate.
    """
    configs = configs or candidate_configs()
    scores = score_window(ledger, crown_date, eval_sessions)
    if scores.empty:
        return {}
    by_key = {(r.horizon, r.config_key): r for r in scores.itertuples(index=False)}
    for h in HORIZONS:
        for c in configs:
            r = by_key.get((h, c.key))
            if r is None or r.n_window < min_window:
                return {}

    def best(h: str, cands: list[HorizonConfig]) -> tuple[HorizonConfig, object]:
        # max() keeps the first maximal element, matching the production tie-break
        c = max(cands, key=lambda c: (by_key[(h, c.key)].opt1_score, by_key[(h, c.key)].r_0_30, -by_key[(h, c.key)].mae_pct))
        return c, by_key[(h, c.key)]

    family_scores = {
        fam: sum(HORIZON_WEIGHTS[h] * best(h, [c for c in configs if c.family == fam])[1].opt1_score for h in HORIZONS)
        for fam in FAMILIES
        if any(c.family == fam for c in configs)
    }
    crowned_family = max(family_scores, key=lambda f: family_scores[f])
    out = {}
    for h in HORIZONS:
        c, r = best(h, [c for c in configs if c.family == crowned_family])
        assert r.window_label_end_date < crown_date, "crowning used an outcome realized on/after the crowning date"
        out[h] = {"config": c, "family": crowned_family, "opt1_score": r.opt1_score, "window_label_end_date": r.window_label_end_date}
    return out


def simulate_crowning(
    ledger: pd.DataFrame,
    crown_dates: list[str],
    configs: list[HorizonConfig] | None = None,
    eval_sessions: int = CROWN_EVAL_SESSIONS,
) -> pd.DataFrame:
    """Track record of repeated crowning: each champion graded only on origins in [crown_date, next crown_date).

    `ledger` holds one stock's rows for every candidate (column config_key). Returns the champion's ledger rows with
    columns crown_date and crowned_family added.
    """
    configs = configs or candidate_configs()
    parts = []
    bounds = list(crown_dates) + ["9999-12-31"]
    td = ledger["trade_date"].astype(str)
    for start, end in zip(bounds[:-1], bounds[1:]):
        champs = crown(ledger, start, configs, eval_sessions)
        if not champs:
            continue
        for h, ch in champs.items():
            sel = ledger[(ledger["horizon"] == h) & (ledger["config_key"] == ch["config"].key) & (td >= start) & (td < end)]
            parts.append(sel.assign(crown_date=start, crowned_family=ch["family"]))
    return pd.concat(parts, ignore_index=True) if parts else ledger.iloc[:0].assign(crown_date=None, crowned_family=None)


def month_starts(trade_dates: pd.Series, first: str) -> list[str]:
    """First trading date of every month on or after `first`."""
    d = pd.Series(pd.to_datetime(trade_dates).sort_values().unique())
    d = d[d >= pd.Timestamp(first)]
    return [x.strftime("%Y-%m-%d") for x in d.groupby(d.dt.to_period("M")).min()]


def crown_single(
    ledger: pd.DataFrame,
    crown_date: str,
    horizon: str,
    configs: list[HorizonConfig] | None = None,
    eval_sessions: int = CROWN_EVAL_SESSIONS,
    min_window: int = 60,
) -> dict | None:
    """Crown one horizon on its own (used for 1D, which has no family stage): best option-1 recency score, then the
    last-30 hit rate, then the recency-weighted MAE, over rows realized before `crown_date` only."""
    configs = configs or candidate_configs()
    scores = score_window(ledger[ledger["horizon"] == horizon], crown_date, eval_sessions)
    if scores.empty:
        return None
    by_key = {r.config_key: r for r in scores.itertuples(index=False)}
    if any(by_key.get(c.key) is None or by_key[c.key].n_window < min_window for c in configs):
        return None
    best = max(configs, key=lambda c: (by_key[c.key].opt1_score, by_key[c.key].r_0_30, -by_key[c.key].mae_pct))
    r = by_key[best.key]
    assert r.window_label_end_date < crown_date, "crowning used an outcome realized on/after the crowning date"
    return {"config": best, "family": best.family, "opt1_score": r.opt1_score, "window_label_end_date": r.window_label_end_date}


def crown_policy(ledger: pd.DataFrame, crown_date: str, eval_sessions: int = CROWN_EVAL_SESSIONS, min_window: int = 60) -> dict[str, str]:
    """Production crowning rule over per-horizon candidate pools (keys may carry a feature-set prefix `set|key`).

    3D-30D: family-first (horizon-weighted best option-1 score per family), then the best candidate of the crowned
    family per horizon; 1D on its own. Ranking: option-1 recency score, last-30 hit rate, recency MAE, then production
    candidate order. Only rows realized before `crown_date` are used. Returns {horizon: candidate key}.
    """
    configs = candidate_configs()
    fam_of = {c.key: c.family for c in configs}
    order = {c.key: i for i, c in enumerate(configs)}
    sc = score_window(ledger, crown_date, eval_sessions)
    if sc.empty:
        return {}
    sc = sc[sc["n_window"] >= min_window].copy()
    assert (sc["window_label_end_date"] < crown_date).all(), "crowning used an outcome realized on/after the crowning date"
    base = sc["config_key"].str.split("|").str[-1]
    sc["family"] = base.map(fam_of)
    sc["order"] = base.map(order)

    def best(g: pd.DataFrame) -> str | None:
        if g.empty:
            return None
        return g.sort_values(["opt1_score", "r_0_30", "mae_pct", "order"], ascending=[False, False, True, True])["config_key"].iloc[0]

    out: dict[str, str] = {}
    one = best(sc[sc["horizon"] == "1d"])
    if one:
        out["1d"] = one
    swing = sc[sc["horizon"].isin(HORIZONS)]
    if swing["horizon"].nunique() == len(HORIZONS):
        def fam_score(fam: str) -> float:
            tot = 0.0
            for h in HORIZONS:
                g = swing[(swing["horizon"] == h) & (swing["family"] == fam)]
                tot += HORIZON_WEIGHTS[h] * (float(g["opt1_score"].max()) if len(g) else 0.0)
            return tot

        fam = max(FAMILIES, key=fam_score)
        for h in HORIZONS:
            k = best(swing[(swing["horizon"] == h) & (swing["family"] == fam)])
            if k:
                out[h] = k
    return out
