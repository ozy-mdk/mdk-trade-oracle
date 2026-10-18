"""Intraday Session Forecaster: % change from the 11:30 price to the closing auction, published after the previous close.

Decision rule for the trader: at 11:30 compare the price with the day's opening price (the "morning move" m). Everything
else is known at the close of the previous session, so each evening every candidate publishes a *grid*: its forecast
f(m) of the 11:30 -> close % change for morning moves m in [-6%, +6%] (0.1% steps). BUY / SELL levels are read off
the grid, and grading reads the same frozen grid at the realized m, so what is published is exactly what is graded.

Row = session D. Features are yesterday's values (shifted one session); the morning move and target of D are only used
for rows before the forecast session (training) or for grading.

Candidates
    RULE                     z(yesterday MLB intensity) + z(morning move) beyond the trailing 80/20 percentile of past
                             scores (trailing 252 sessions); size = mean outcome of its last 60 trades (>= 20)
    <features>|<model>|<w>   regression on yesterday's features + morning move, fitted on the last w sessions before D;
                             trades only when |f(m)| >= REGRESSION_TRADE_PCT
Tournament (incumbent rule): on the first session of each month, per stock, RULE stays champion unless the best
challenger over the trailing CROWN_SESSIONS sessions of out-of-sample trades (>= MIN_CROWN_TRADES trades; ranked by hit
rate, then mean gain after costs) beat RULE's hit rate by CHALLENGER_MARGIN with a positive mean gain after costs.
Chosen over "best hit rate wins" (which crowned lucky regressions and lost money) on 2023-24 results and confirmed on
2025-26: see docs/RESEARCH_HANDOFF_2026-10.md.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

GRID = np.round(np.arange(-6.0, 6.0001, 0.1), 1)
COST_PCT = 0.14  # 0.10% round-trip commission + ~0.04% entry half-spread (exit in the closing auction)
REGRESSION_TRADE_PCT = 1.0  # a regression candidate trades only on forecasts of at least this size
CHALLENGER_MARGIN = 0.02  # a challenger must beat RULE's hit rate by 2 percentage points to take over
WINDOWS = (63, 126, 252, 504)
MODELS = ("Ridge", "BayesianRidge", "Huber", "XGBoost", "LightGBM")
RULE = "RULE"
RULE_WINDOW, RULE_MIN_P, RULE_Q, Z_CLIP = 252, 126, 0.8, 5.0
SIZE_CALLS, MIN_SIZE_CALLS = 60, 20
CROWN_SESSIONS, MIN_CROWN_TRADES = 120, 20
RECORD_TRADES, MIN_RECORD_TRADES = 60, 20
EWMA_LAMBDA = 0.94
# Trading days without raw data: the next session's "yesterday" would skip a day, so it is neither traded nor graded.
MISSING_RAW_SESSIONS = pd.to_datetime(["2026-09-25", "2026-10-01", "2026-10-02", "2026-10-05"])

LEAN_MLB = ["feat_mlb_buy_share_today", "feat_mlb_turnover_intensity", "feat_total_inst_net_share_today", "mlb_late_share"]
LEAN_INST = LEAN_MLB + ["feat_kamu_turnover_intensity", "feat_big5_turnover_intensity", "feat_kamu_w5_share", "feat_big5_w5_share"]


def feature_sets() -> dict[str, list[str]]:
    from mdk_trading_oracle.data.silver.tertip_ml_forecaster import FEATURE_COLS

    return {"LEAN_MLB": LEAN_MLB, "LEAN_INST": LEAN_INST, "OLD_39": list(FEATURE_COLS)}


def candidate_names() -> list[str]:
    return [RULE] + [f"{f}|{m}|{w}" for f in feature_sets() for m in MODELS for w in WINDOWS]


def parse_candidate(name: str) -> dict[str, Any]:
    if name == RULE:
        return {"features": "MLB+MORNING", "model": RULE, "window": RULE_WINDOW}
    f, m, w = name.split("|")
    return {"features": f, "model": m, "window": int(w)}


# ----------------------------------------------------------------------------------------------------------------------
# Panel


def build_panel(db: Any, symbol: str, next_date: pd.Timestamp | None = None) -> pd.DataFrame:
    """One row per session of `symbol` with yesterday's features, today's session prices, morning move and target.

    With `next_date`, an empty session is appended before the one-session shift, so its row carries the last close's
    features (the forecast row published that evening); its prices, morning move and target stay empty.
    """
    from mdk_trading_oracle.data.silver.tertip_ml_forecaster import extract_3pillar_time_series

    sym = symbol.upper()
    df = extract_3pillar_time_series(db, sym).sort_values("trade_date").reset_index(drop=True)
    if df.empty:
        return pd.DataFrame()
    df["trade_date"] = pd.to_datetime(df["trade_date"])
    late = db.query_df(
        "SELECT trade_date, SUM(net_flow_tl) AS mlb_late_net FROM silver_intraday_broker_window_summary "
        "WHERE symbol = %s AND broker_id = 'MLB' AND window_name IN ('afternoon_reaction', 'closing_session') GROUP BY 1",
        (sym,),
    )
    late["trade_date"] = pd.to_datetime(late["trade_date"])
    df = df.merge(late, on="trade_date", how="left")
    if next_date is not None and pd.Timestamp(next_date) > df["trade_date"].max():
        df = pd.concat([df, pd.DataFrame({"trade_date": [pd.Timestamp(next_date)]})], ignore_index=True)
    df["mlb_late_share"] = (df["mlb_late_net"].fillna(0.0) / df["total_turnover_tl"].replace(0, np.nan) * 100).clip(-100, 100)
    cols = sorted({c for fs in feature_sets().values() for c in fs})
    out = df[["trade_date"]].copy()
    for c in cols:
        out[c] = df[c].shift(1)  # known at the previous close
    out["prev_date"] = df["trade_date"].shift(1)
    r = np.log(df["adj_close_price"].astype(float)).diff() * 100
    sigma = np.sqrt((r ** 2).ewm(alpha=1 - EWMA_LAMBDA, adjust=False, ignore_na=True).mean())
    out["sigma_pct"] = sigma.shift(1)  # through the previous close
    sp = db.query_df(
        "SELECT trade_date, open_price, price_1130, close_price, is_half_day FROM silver_daily_session_prices WHERE symbol = %s",
        (sym,),
    )
    sp["trade_date"] = pd.to_datetime(sp["trade_date"])
    out = out.merge(sp, on="trade_date", how="left")
    out["morning_ret"] = (out["price_1130"] / out["open_price"] - 1) * 100
    out["target"] = (out["close_price"] / out["price_1130"] - 1) * 100
    out["gap"] = [pd.notna(a) and bool(((MISSING_RAW_SESSIONS > a) & (MISSING_RAW_SESSIONS < b)).any()) for a, b in zip(out["prev_date"], out["trade_date"])]
    out["usable"] = out["target"].notna() & out["morning_ret"].notna() & ~out["is_half_day"].fillna(False).astype(bool) & ~out["gap"]
    out["symbol"] = sym
    return out.dropna(subset=["prev_date"]).reset_index(drop=True)


# ----------------------------------------------------------------------------------------------------------------------
# Candidates


def _make_model(name: str):
    import lightgbm as lgb
    from sklearn.linear_model import BayesianRidge, HuberRegressor, Ridge
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler
    from xgboost import XGBRegressor

    if name == "XGBoost":
        return XGBRegressor(n_estimators=45, max_depth=2, learning_rate=0.03, subsample=0.8, colsample_bytree=0.8, random_state=42, n_jobs=1), "sample_weight"
    if name == "LightGBM":
        return lgb.LGBMRegressor(n_estimators=45, max_depth=3, learning_rate=0.03, num_leaves=7, subsample=0.8, colsample_bytree=0.8,
                                 random_state=42, n_jobs=1, verbose=-1), "sample_weight"
    if name == "BayesianRidge":
        return Pipeline([("sc", StandardScaler()), ("m", BayesianRidge(max_iter=300))]), "m__sample_weight"
    if name == "Huber":
        return Pipeline([("sc", StandardScaler()), ("m", HuberRegressor(epsilon=1.35, alpha=1.0, max_iter=300))]), "m__sample_weight"
    return Pipeline([("sc", StandardScaler()), ("m", Ridge(alpha=2.0))]), "m__sample_weight"


def _weights(y: np.ndarray) -> np.ndarray:
    a = np.abs(y)
    return np.where(a >= 2.0, 6.0, np.where(a >= 1.0, 3.5, 0.5))


def rule_state(panel: pd.DataFrame) -> pd.DataFrame:
    """Trailing statistics of the rule, per row, from rows strictly before it (vectorized; row values never used)."""
    s = pd.DataFrame(index=panel.index)
    x = panel["feat_mlb_turnover_intensity"].astype(float)
    m = panel["morning_ret"].where(panel["usable"]).astype(float)
    roll = lambda v: v.shift(1).rolling(RULE_WINDOW, min_periods=RULE_MIN_P)  # noqa: E731
    s["mu_x"], s["sd_x"] = roll(x).mean(), roll(x).std()
    s["mu_m"], s["sd_m"] = roll(m).mean(), roll(m).std()
    s["z_x"] = ((x - s["mu_x"]) / s["sd_x"]).clip(-Z_CLIP, Z_CLIP)
    score = s["z_x"] + ((m - s["mu_m"]) / s["sd_m"]).clip(-Z_CLIP, Z_CLIP)
    s["hi"], s["lo"] = roll(score).quantile(RULE_Q), roll(score).quantile(1 - RULE_Q)
    call = np.where(score > s["hi"], 1, np.where(score < s["lo"], -1, 0))
    outcome = pd.Series(np.where(call != 0, call * panel["target"], np.nan), index=panel.index).where(panel["usable"])
    s["size"] = outcome.shift(1).rolling(10_000, min_periods=1).apply(
        lambda a: np.mean(a[~np.isnan(a)][-SIZE_CALLS:]) if (~np.isnan(a)).sum() >= MIN_SIZE_CALLS else np.nan, raw=True)
    return s


def rule_grid(st: pd.Series) -> np.ndarray:
    """Rule forecast for every morning move on the grid (0 where it does not trade)."""
    if st[["mu_x", "sd_x", "mu_m", "sd_m", "hi", "lo", "size"]].isna().any() or st["size"] <= 0:
        return np.zeros_like(GRID)
    score = st["z_x"] + np.clip((GRID - st["mu_m"]) / st["sd_m"], -Z_CLIP, Z_CLIP)
    return np.where(score > st["hi"], st["size"], np.where(score < st["lo"], -st["size"], 0.0))


def candidate_grids(
    panel: pd.DataFrame,
    idx: int,
    rule: pd.DataFrame | None = None,
    names: list[str] | None = None,
    cache: dict[str, tuple[int, Any]] | None = None,
    refit_every: int = 1,
) -> dict[str, np.ndarray]:
    """Forecast grids of every candidate for session `idx`, each fitted on usable sessions strictly before it.

    Only yesterday's features of row `idx` are read; its morning move, prices and target are never used.
    With a `cache` and `refit_every` > 1 (historical simulation), a regression candidate fitted at row j (on rows < j)
    is reused for rows j .. j + refit_every - 1: still fitted strictly before each forecast row. Production refits daily.
    """
    names = names or candidate_names()
    rule = rule if rule is not None else rule_state(panel)
    out: dict[str, np.ndarray] = {}
    sets = feature_sets()
    day = panel.loc[idx, "trade_date"]
    for name in names:
        if name == RULE:
            out[name] = rule_grid(rule.loc[idx])
            continue
        c = parse_candidate(name)
        cols = sets[c["features"]] + ["morning_ret"]
        hist = panel.iloc[max(0, idx - c["window"]):idx]
        hist = hist[hist["usable"]].dropna(subset=cols + ["target"])
        assert (hist["trade_date"] < day).all()
        x_row = panel.loc[[idx], sets[c["features"]]]
        if len(hist) < min(40, c["window"] - 5) or x_row.isna().any(axis=None):
            continue
        cached = cache.get(name) if cache is not None else None
        if cached is not None and 0 < idx - cached[0] < refit_every:
            model = cached[1]
        else:
            model, key = _make_model(c["model"])
            y = hist["target"].to_numpy(dtype=float)
            model.fit(hist[cols], y, **{key: _weights(y)})
            if cache is not None:
                cache[name] = (idx, model)
        X = pd.concat([x_row] * len(GRID), ignore_index=True)
        X["morning_ret"] = GRID
        out[name] = np.clip(model.predict(X[cols]), -10.0, 10.0)
    return out


def forecast_at(grid: np.ndarray, morning_ret: float) -> float:
    """Forecast for a realized morning move, read from the published grid (clipped to the grid's range)."""
    return float(np.interp(np.clip(morning_ret, GRID[0], GRID[-1]), GRID, grid))


def trades(forecast_pct: float, name: str) -> bool:
    return forecast_pct != 0.0 if name == RULE else abs(forecast_pct) >= REGRESSION_TRADE_PCT


def traded_mask(results: pd.DataFrame) -> pd.Series:
    """Vectorized `trades` for a results frame (recomputed from the forecast, never trusted from storage)."""
    f = results["forecast_pct"].astype(float)
    return pd.Series(np.where(results["candidate"] == RULE, f != 0.0, f.abs() >= REGRESSION_TRADE_PCT), index=results.index)


def levels(grid: np.ndarray, name: str) -> tuple[float | None, float | None]:
    """BUY level = smallest m >= 0 from which the forecast is a trade upwards; SELL level = largest m <= 0 downwards."""
    up = [m for m, f in zip(GRID, grid) if m >= 0 and trades(f, name) and f > 0]
    dn = [m for m, f in zip(GRID, grid) if m <= 0 and trades(f, name) and f < 0]
    return (float(min(up)) + 0.0 if up else None, float(max(dn)) + 0.0 if dn else None)


# ----------------------------------------------------------------------------------------------------------------------
# Tournament and trader numbers


@dataclass
class Champion:
    name: str
    crowned_on: pd.Timestamp
    hit_rate_pct: float | None
    trades: int
    mean_after_cost_pct: float | None


def standings(results: pd.DataFrame, month_start: pd.Timestamp) -> pd.DataFrame:
    """Every candidate's trades over the trailing CROWN_SESSIONS sessions before `month_start`, ranked by hit rate."""
    r = results[results["session_date"] < month_start]
    sessions = sorted(r["session_date"].unique())[-CROWN_SESSIONS:]
    r = r[r["session_date"].isin(sessions)]
    r = r[traded_mask(r)]
    if r.empty:
        return pd.DataFrame(columns=["candidate", "trades", "hit_rate", "gain", "eligible"])
    r = r.assign(hit=np.sign(r["forecast_pct"]) == np.sign(r["actual_pct"]),
                 gain=np.sign(r["forecast_pct"]) * r["actual_pct"] - COST_PCT)
    s = r.groupby("candidate").agg(trades=("hit", "size"), hit_rate=("hit", "mean"), gain=("gain", "mean")).reset_index()
    s["eligible"] = s["trades"] >= MIN_CROWN_TRADES
    return s.sort_values(["eligible", "hit_rate", "gain"], ascending=[False, False, False]).reset_index(drop=True)


def crown(results: pd.DataFrame, month_start: pd.Timestamp) -> Champion:
    """Incumbent-rule crowning for one stock: a challenger replaces RULE only with a clear, profitable edge."""
    s = standings(results, month_start)
    s = s[s["eligible"]] if not s.empty else s
    rule = s[s["candidate"] == RULE] if not s.empty else s
    rule_hit = float(rule["hit_rate"].iloc[0]) if len(rule) else 0.5
    if not s.empty:
        b = s.iloc[0]
        if b["candidate"] != RULE and b["hit_rate"] >= rule_hit + CHALLENGER_MARGIN and b["gain"] > 0:
            return Champion(str(b["candidate"]), month_start, round(float(b["hit_rate"]) * 100, 1), int(b["trades"]), round(float(b["gain"]), 3))
    if len(rule):
        r = rule.iloc[0]
        return Champion(RULE, month_start, round(float(r["hit_rate"]) * 100, 1), int(r["trades"]), round(float(r["gain"]), 3))
    return Champion(RULE, month_start, None, 0, None)


def trader_numbers(results: pd.DataFrame, candidate: str, before: pd.Timestamp, sigma_next: float) -> dict[str, float | int | None]:
    """Potential / risk / record from the candidate's last RECORD_TRADES trades on this stock before `before`."""
    r = results[(results["candidate"] == candidate) & (results["session_date"] < before)]
    r = r[traded_mask(r)].sort_values("session_date").tail(RECORD_TRADES)
    if len(r) < MIN_RECORD_TRADES or not np.isfinite(sigma_next):
        return {"record_trades": len(r)}
    fav = np.sign(r["forecast_pct"]) * r["actual_pct"]
    u = fav / r["sigma_pct"]
    win_u, loss_u = u[fav > 0], -u[fav <= 0]
    return {
        "record_trades": len(r),
        "record_hit_rate_pct": round(float((fav > 0).mean() * 100), 1),
        "record_avg_after_cost_pct": round(float(fav.mean() - COST_PCT), 3),
        "win_typ_pct": round(float(win_u.median() * sigma_next), 3) if len(win_u) else None,
        "win_q75_pct": round(float(np.percentile(win_u, 75) * sigma_next), 3) if len(win_u) else None,
        "loss_typ_pct": round(float(loss_u.median() * sigma_next), 3) if len(loss_u) else None,
    }
