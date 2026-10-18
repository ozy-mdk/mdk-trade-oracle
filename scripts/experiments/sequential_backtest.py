#!/usr/bin/env python
"""Day-by-day replay of the last N sessions with the future physically removed (isolated experiment, read-only).

For each session D (oldest first):
    1. truncate the tick panel to sessions strictly before D (nothing from D onward exists),
    2. add an empty row for D and recompute everything from scratch (as the evening job after the D-1 close would):
       pre-open end-of-day call, 11:30 trigger levels, forecast sizes (from past calls only),
    3. only then read D's realized prices: open, last print before 11:30 (the morning move the trader sees), close,
    4. record forecast vs reality.
Finally the replayed forecasts are compared with the vectorized computation: any difference would indicate leakage.

Output: ~/data/mdk_oracle/experiments/last60/sequential_ledger.csv

Forecast per stock per day = call direction x expected move (mean outcome of the stock's last 60 calls, >= 20, past only);
0% on days without a call. A stock whose recent calls averaged <= 0 gets no call. Error = |forecast - actual| in pp.

Usage:
    .venv/bin/python scripts/experiments/sequential_backtest.py --sessions 60
"""

from __future__ import annotations

import argparse
import sys
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

sys.path.insert(0, str(Path(__file__).parent))
from intraday_flow_experiment import DB, OUT_DIR, engineer  # noqa: E402
from session_triggers import add_state, trigger_call  # noqa: E402

OUT = Path.home() / "data" / "mdk_oracle" / "experiments" / "last60"
MISSING_RAW_DAYS = pd.to_datetime(["2026-09-25", "2026-10-01", "2026-10-02", "2026-10-05"])
MIN_PAST_CALLS = 20
SIZE_CALLS = 60


def forecast_for(raw: pd.DataFrame, day: pd.Timestamp) -> pd.DataFrame:
    """Everything known after the close of the session before `day` (rows >= day are removed first)."""
    hist = raw[raw["trade_date"] < day].copy()
    assert hist["trade_date"].max() < day
    placeholder = pd.DataFrame({"symbol": sorted(hist["symbol"].unique()), "trade_date": day,
                                "last_time": pd.Timestamp("2000-01-01 18:09:59").time()})
    p, _ = engineer(pd.concat([hist, placeholder], ignore_index=True))
    s = add_state(p)
    s["oc"] = (s["close_price"] / s["open_price"] - 1) * 100
    s = s.sort_values(["symbol", "trade_date"])
    sizes = {}
    for name, call, y in (("session", "call", "rest_1130"), ("eod", "eod_call", "oc")):
        # expected move of a call = mean outcome of the stock's last SIZE_CALLS calls before `day` (>= MIN_PAST_CALLS)
        past = s[(s[call] != 0) & (s["trade_date"] < day) & s[y].notna()]
        out = (past[call] * past[y]).groupby(past["symbol"]).apply(lambda x: x.tail(SIZE_CALLS).mean() if len(x) >= MIN_PAST_CALLS else np.nan)
        sizes[name] = out
    f = s[s["trade_date"] == day][["symbol", "trade_date", "z_mlb_net_yday_intensity", "up_trigger_pct", "down_trigger_pct", "eod_call"]].copy()
    f["session_size"] = f["symbol"].map(sizes["session"])
    f["eod_size"] = f["symbol"].map(sizes["eod"])
    # a stock whose recent calls lost money on average gets no call (no "BUY with 0% expected")
    f["eod_call"] = np.where(f["eod_size"] > 0, f["eod_call"], 0)
    f["as_of_close"] = hist["trade_date"].max()
    return f


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--sessions", type=int, default=60)
    a = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    raw = pd.read_parquet(OUT_DIR / "tick_panel.parquet")
    raw = raw.sort_values(["symbol", "trade_date"])
    days = sorted(raw["trade_date"].unique())[-a.sessions:]
    rows = []
    for i, day in enumerate(days, 1):
        day = pd.Timestamp(day)
        f = forecast_for(raw, day)
        real = raw[raw["trade_date"] == day][["symbol", "open_price", "p_1130", "close_price"]]
        r = f.merge(real, on="symbol", how="left")
        r["morning_move_pct"] = (r["p_1130"] / r["open_price"] - 1) * 100
        r["session_call"] = trigger_call(r["morning_move_pct"], r["up_trigger_pct"], r["down_trigger_pct"])
        r["session_call"] = np.where(r["session_size"] > 0, r["session_call"], 0)
        r["rest_of_session_pct"] = (r["close_price"] / r["p_1130"] - 1) * 100
        r["open_to_close_pct"] = (r["close_price"] / r["open_price"] - 1) * 100
        r["data_gap"] = [bool(((MISSING_RAW_DAYS > r_["as_of_close"]) & (MISSING_RAW_DAYS < day)).any()) for _, r_ in r.iterrows()]
        rows.append(r)
        print(f"[{i}/{len(days)}] {day.date()} forecast from close of {r['as_of_close'].iloc[0].date()}", file=sys.stderr)
    L = pd.concat(rows, ignore_index=True)
    for name, call, y in (("session", "session_call", "rest_of_session_pct"), ("eod", "eod_call", "open_to_close_pct")):
        L[f"{name}_forecast_pct"] = np.where(L[call] != 0, L[call] * L[f"{name}_size"], 0.0)
        L[f"{name}_abs_error_pp"] = (L[f"{name}_forecast_pct"] - L[y]).abs()
        L[f"{name}_result"] = np.where(L[call] == 0, "no call", np.where(np.sign(L[call]) == np.sign(L[y]), "HIT", "MISS"))
        L[f"{name}_strict"] = np.where(L[call] == 0, np.nan, (L[call] * L[y] > DB).astype(float))
    L["session_forecast_close"] = np.where(L["session_call"] != 0, L["p_1130"] * (1 + L["session_forecast_pct"] / 100), np.nan)
    L["eod_forecast_close"] = np.where(L["eod_call"] != 0, L["open_price"] * (1 + L["eod_forecast_pct"] / 100), np.nan)
    L.to_csv(OUT / "sequential_ledger.csv", index=False)

    # leakage check: replay vs vectorized computation on the same sessions
    v, _ = engineer(raw)
    v = add_state(v.dropna(subset=["rest_1130"]))
    m = L.merge(v[["symbol", "trade_date", "up_trigger_pct", "down_trigger_pct"]], on=["symbol", "trade_date"], suffixes=("", "_vec"))
    same = np.isclose(m["up_trigger_pct"], m["up_trigger_pct_vec"], equal_nan=True) & np.isclose(m["down_trigger_pct"], m["down_trigger_pct_vec"], equal_nan=True)
    print(f"\nleakage check: replay (future removed) == vectorized on {same.mean() * 100:.1f}% of {len(m)} stock-sessions")
    ok = L[~L["data_gap"]]
    pd.set_option("display.width", 220)
    for name in ("session", "eod"):
        k = ok[ok[f"{name}_result"] != "no call"]
        t = k.groupby("symbol").agg(calls=(f"{name}_result", "size"), hits=(f"{name}_result", lambda s: (s == "HIT").sum()))
        t["hit_rate_%"] = t["hits"] / t["calls"] * 100
        t.loc["ALL6"] = [len(k), (k[f"{name}_result"] == "HIT").sum(), (k[f"{name}_result"] == "HIT").mean() * 100]
        y = "rest_of_session_pct" if name == "session" else "open_to_close_pct"
        t["mae_pp_all_days"] = ok.groupby("symbol")[f"{name}_abs_error_pp"].mean()
        t.loc["ALL6", "mae_pp_all_days"] = ok[f"{name}_abs_error_pp"].mean()
        t["mae_pp_if_always_0"] = ok.groupby("symbol")[y].apply(lambda v: v.abs().mean())
        t.loc["ALL6", "mae_pp_if_always_0"] = ok[y].abs().mean()
        print(f"\n{name}: hits on called sessions (data-gap sessions excluded)")
        print(t.round(1).to_string())


if __name__ == "__main__":
    main()
