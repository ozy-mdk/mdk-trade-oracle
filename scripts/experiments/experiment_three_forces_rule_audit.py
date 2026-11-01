#!/usr/bin/env python
"""Out-of-sample audit of the Three-Forces flow rules on every stock and all available history.

The rules were written while studying the 6 focus stocks in 2026. This grades each rule call directly against the
realized close-to-close return, separately on data the rules were not designed on: the other stocks, and the sessions
before 2026. A rule has an edge only if it beats always-UP and always-DOWN on the same sessions. Overlapping windows
overstate the sample, so results are also shown for independent calls (one per horizon-length window per stock).

Writes ~/data/mdk_oracle/experiments/timestral/three_forces_rule_audit.json
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from mdk_trading_oracle.data.gold.tertip_multi_horizon_crowning import constant_call_hit_rates
from mdk_trading_oracle.models.three_forces_gate import GATED_HORIZONS, gate_direction
from mdk_trading_oracle.models.timestral_forecaster import HORIZON_DAYS, check_horizon_hit

BASE = Path.home() / "data" / "mdk_oracle" / "experiments"
FOCUS = {"AKBNK", "ASELS", "GARAN", "ISCTR", "THYAO", "YKBNK"}
DESIGN_START = "2026-01-01"


def load_events() -> pd.DataFrame:
    clash = pd.read_parquet(BASE / "timestral" / "three_forces_daily_clash.parquet")
    clash["trade_date"] = clash["trade_date"].astype(str).str[:10]
    rows: list[dict] = []
    for sym, g in clash.groupby("symbol"):
        frame = pd.read_parquet(BASE / "feature_importance" / "frames" / f"{sym}.parquet", columns=["trade_date", "adj_close_price"])
        frame["trade_date"] = frame["trade_date"].astype(str).str[:10]
        frame = frame.sort_values("trade_date").reset_index(drop=True)
        px = frame["adj_close_price"].to_numpy(float)
        recs = g.merge(frame, on="trade_date").sort_values("trade_date").to_dict("records")
        dates = [r["trade_date"] for r in recs]
        px = np.array([r["adj_close_price"] for r in recs], dtype=float)
        for i, r in enumerate(recs):
            direction = gate_direction(r, "flow_only")
            for h in GATED_HORIZONS:
                n = HORIZON_DAYS[h]
                if i + n >= len(px) or px[i] <= 0:
                    continue
                rows.append({
                    "symbol": sym,
                    "trade_date": dates[i],
                    "idx": i,
                    "horizon": h,
                    "direction": direction,
                    "actual": (px[i + n] / px[i] - 1.0) * 100.0,
                })
    return pd.DataFrame(rows)


def independent_calls(g: pd.DataFrame, n: int) -> pd.DataFrame:
    keep: list[int] = []
    for _, s in g.sort_values(["symbol", "idx"]).groupby(["symbol", "direction"]):
        last = -(10**9)
        for ix, i in zip(s.index, s["idx"]):
            if i - last >= n:
                keep.append(ix)
                last = i
    return g.loc[keep]


def grade(g: pd.DataFrame, h: str) -> dict:
    if g.empty:
        return {"n": 0, "n_independent": 0}
    n_days = HORIZON_DAYS[h]
    call = np.where(g["direction"] == "BEARISH", -5.0, 5.0)
    hits = np.array([check_horizon_hit(p, a, h) for p, a in zip(call, g["actual"])], dtype=float)
    up, down = constant_call_hit_rates(g["actual"].to_numpy(), n_days)
    ind = independent_calls(g, n_days)
    ind_call = np.where(ind["direction"] == "BEARISH", -5.0, 5.0)
    ind_hits = np.array([check_horizon_hit(p, a, h) for p, a in zip(ind_call, ind["actual"])], dtype=float)
    return {
        "n": int(len(g)),
        "n_independent": int(len(ind)),
        "episodes": int(g[["symbol"]].assign(m=g["trade_date"].str[:7]).drop_duplicates().shape[0]),
        "hit_pct": round(float(hits.mean() * 100), 1),
        "hit_independent_pct": round(float(ind_hits.mean() * 100), 1),
        "always_up_pct": round(float(up), 1),
        "always_down_pct": round(float(down), 1),
        "beats_both": bool(hits.mean() * 100 > max(up, down)),
        "mean_return_pct": round(float(g["actual"].mean()), 2),
    }


def main() -> None:
    ev = load_events()
    fired = ev[ev["direction"].notna()].copy()
    print(f"stock-sessions graded: {len(ev)}, firing: {len(fired)}, stocks: {ev['symbol'].nunique()}, "
          f"{ev['trade_date'].min()} to {ev['trade_date'].max()}")
    splits = {
        "other_stocks_all_dates": ~fired["symbol"].isin(FOCUS),
        "focus_before_design_period": fired["symbol"].isin(FOCUS) & (fired["trade_date"] < DESIGN_START),
        "all_stocks_before_design_period": fired["trade_date"] < DESIGN_START,
        "design_period_focus_stocks": fired["symbol"].isin(FOCUS) & (fired["trade_date"] >= DESIGN_START),
    }
    out: dict = {"splits": {}, "design_start": DESIGN_START, "total_sessions": int(len(ev))}
    for name, mask in splits.items():
        out["splits"][name] = {}
        print(f"\n=== {name} ===")
        for direction in ("BEARISH", "BULLISH"):
            for h in GATED_HORIZONS:
                g = fired[mask & (fired["direction"] == direction) & (fired["horizon"] == h)]
                res = grade(g, h)
                out["splits"][name][f"{direction}_{h}"] = res
                if res["n"]:
                    print(f"  {direction:<7} {h:<3} n={res['n']:<4} independent={res['n_independent']:<3} "
                          f"hit={res['hit_pct']:5.1f}% ind_hit={res['hit_independent_pct']:5.1f}% "
                          f"up={res['always_up_pct']:5.1f}% down={res['always_down_pct']:5.1f}% "
                          f"beats_both={res['beats_both']} mean_ret={res['mean_return_pct']:+.2f}%")
                else:
                    print(f"  {direction:<7} {h:<3} n=0")
    path = BASE / "timestral" / "three_forces_rule_audit.json"
    path.write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(f"\nSaved {path}")


if __name__ == "__main__":
    main()
