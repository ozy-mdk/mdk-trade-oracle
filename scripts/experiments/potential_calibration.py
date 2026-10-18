#!/usr/bin/env python
"""Are the "potential / risk / track record" numbers we would show a trader calibrated? (isolated, read-only)

For every 11:30 session call on day D, using only data before D:
    hit rate        share of the stock's last 60 calls that moved in the called direction
    potential       if right, typically +X%: median of past winning moves divided by that day's volatility, times
                    today's volatility (EWMA lambda 0.94 of close-to-close log returns through D-1);
                    upside "1 in 4": 75th percentile of the same
    risk            if wrong, typically -Z%: same from past losing moves
    expected value  hit_rate * mean win - (1 - hit_rate) * mean loss - round-trip costs
Then compared with what happened on D:
    share of winning calls beating the shown typical win (target ~50%), the "1 in 4" level (~25%),
    share of losing calls losing more than the shown typical loss (~50%), shown vs realized hit rate,
    realized return by shown expected-value bucket.
Also writes the card numbers as of the last close for the trader-screen mockup.

Usage:
    .venv/bin/python scripts/experiments/potential_calibration.py
"""

from __future__ import annotations

import json
import sys
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

sys.path.insert(0, str(Path(__file__).parent))
from intraday_flow_experiment import DISCOVERY_END, OUT_DIR, engineer  # noqa: E402
from session_triggers import add_state  # noqa: E402

OUT = Path.home() / "data" / "mdk_oracle" / "experiments" / "potential"
TRACK_CALLS, MIN_CALLS = 60, 20
COST_RT_PCT = 0.14  # 0.10% round-trip commission + ~0.04% entry half-spread (exit in the closing auction)
EWMA_LAMBDA = 0.94


def ewma_sigma(px: pd.Series) -> pd.Series:
    """Daily volatility (%) known at the close of each row: EWMA of squared close-to-close log returns."""
    r = np.log(px).diff() * 100
    return np.sqrt((r ** 2).ewm(alpha=1 - EWMA_LAMBDA, adjust=False).mean())


def per_stock(g: pd.DataFrame) -> pd.DataFrame:
    g = g.sort_values("trade_date").copy()
    g["sigma"] = ewma_sigma(g["adj_close_price"]).shift(1)  # through D-1
    calls = g[g["call"] != 0].copy()
    calls["fav"] = calls["call"] * calls["rest_1130"]
    calls["hit"] = (calls["fav"] > 0).astype(float)
    calls["u"] = calls["fav"] / calls["sigma"]  # move in volatility units
    roll = lambda s, f: s.shift(1).rolling(TRACK_CALLS, min_periods=MIN_CALLS).apply(f, raw=True)  # noqa: E731
    calls["hit_rate"] = roll(calls["hit"], np.mean)
    win_u = calls["u"].where(calls["fav"] > 0)
    loss_u = (-calls["u"]).where(calls["fav"] <= 0)
    calls["win_med_u"] = roll(win_u, np.nanmedian)
    calls["win_q75_u"] = roll(win_u, lambda a: np.nanpercentile(a, 75))
    calls["win_mean_u"] = roll(win_u, np.nanmean)
    calls["loss_med_u"] = roll(loss_u, np.nanmedian)
    calls["loss_mean_u"] = roll(loss_u, np.nanmean)
    for c in ("win_med", "win_q75", "win_mean", "loss_med", "loss_mean"):
        calls[f"{c}_pct"] = calls[f"{c}_u"] * calls["sigma"]
    calls["ev_pct"] = calls["hit_rate"] * calls["win_mean_pct"] - (1 - calls["hit_rate"]) * calls["loss_mean_pct"] - COST_RT_PCT
    return calls


def card_numbers(p: pd.DataFrame, calls: pd.DataFrame) -> list[dict]:
    """Numbers as of the last close for the next session (all stocks), for the mockup."""
    out = []
    for sym, g in calls.groupby("symbol"):
        last_px = p[p["symbol"] == sym].sort_values("trade_date")
        sigma_next = float(ewma_sigma(last_px["adj_close_price"]).iloc[-1])
        h = g.dropna(subset=["fav"]).tail(TRACK_CALLS)
        win_u, loss_u = h.loc[h["fav"] > 0, "u"], -h.loc[h["fav"] <= 0, "u"]
        hr = float((h["fav"] > 0).mean())
        ev = hr * win_u.mean() * sigma_next - (1 - hr) * loss_u.mean() * sigma_next - COST_RT_PCT
        out.append({"symbol": sym, "as_of_close": str(last_px["trade_date"].max().date()), "calls_in_record": len(h),
                    "hit_rate_pct": round(hr * 100, 1), "win_typ_pct": round(float(win_u.median() * sigma_next), 2),
                    "win_q75_pct": round(float(np.percentile(win_u, 75) * sigma_next), 2), "loss_typ_pct": round(float(loss_u.median() * sigma_next), 2),
                    "ev_after_cost_pct": round(float(ev), 2), "sigma_next_pct": round(sigma_next, 2),
                    "record_mean_pct": round(float(h["fav"].mean() - COST_RT_PCT), 2)})
    return out


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    pd.set_option("display.width", 220)
    p, _ = engineer(pd.read_parquet(OUT_DIR / "tick_panel.parquet"))
    p = add_state(p.dropna(subset=["rest_1130"]))
    calls = pd.concat([per_stock(g) for _, g in p.groupby("symbol")], ignore_index=True)
    ev = calls[(calls["trade_date"] >= "2023-01-01")].dropna(subset=["hit_rate", "win_med_pct", "loss_med_pct", "ev_pct"])
    ev = ev.assign(period=np.where(ev["trade_date"] <= DISCOVERY_END, "2023-24", "2025-26"))
    rows = []
    for per, g in list(ev.groupby("period")) + [("all", ev)]:
        w, l_ = g[g["fav"] > 0], g[g["fav"] <= 0]
        rows.append({"period": per, "calls": len(g),
                     "shown_hit_rate": g["hit_rate"].mean() * 100, "realized_hit_rate": (g["fav"] > 0).mean() * 100,
                     "wins_beating_shown_typical_% (target 50)": (w["fav"] > w["win_med_pct"]).mean() * 100,
                     "wins_beating_shown_1in4_% (target 25)": (w["fav"] > w["win_q75_pct"]).mean() * 100,
                     "losses_worse_than_shown_typical_% (target 50)": ((-l_["fav"]) > l_["loss_med_pct"]).mean() * 100,
                     "shown_typical_win": w["win_med_pct"].median(), "realized_median_win": w["fav"].median(),
                     "shown_typical_loss": l_["loss_med_pct"].median(), "realized_median_loss": (-l_["fav"]).median(),
                     "shown_ev_after_cost": g["ev_pct"].mean(), "realized_mean_after_cost": (g["fav"] - COST_RT_PCT).mean()})
    print("=== calibration of the numbers shown before each 11:30 call ===")
    print(pd.DataFrame(rows).set_index("period").round(2).T.to_string())

    ev["ev_bucket"] = pd.qcut(ev["ev_pct"], 4, labels=["lowest 25%", "2nd", "3rd", "highest 25%"])
    b = ev.groupby("ev_bucket").agg(calls=("fav", "size"), shown_ev=("ev_pct", "mean"), realized_after_cost=("fav", lambda s: (s - COST_RT_PCT).mean()),
                                    realized_hit=("fav", lambda s: (s > 0).mean() * 100))
    print("\n=== does a higher shown expected value earn more? (calls split into quarters by shown EV) ===")
    print(b.round(2).to_string())
    for per, g in ev.groupby("period"):
        pos = g[g["ev_pct"] > 0]
        print(f"{per}: calls with shown EV > 0 after costs: {len(pos)}/{len(g)}, realized {(pos['fav'] - COST_RT_PCT).mean():+.2f}% per call, hit {(pos['fav'] > 0).mean() * 100:.1f}%"
              f" | shown EV <= 0: realized {(g.loc[g['ev_pct'] <= 0, 'fav'] - COST_RT_PCT).mean():+.2f}%")
    cards = card_numbers(p, calls)
    (OUT / "cards_next_session.json").write_text(json.dumps(cards, indent=1))
    print("\n=== card numbers as of the last close ===")
    print(pd.DataFrame(cards).to_string(index=False))


if __name__ == "__main__":
    main()
