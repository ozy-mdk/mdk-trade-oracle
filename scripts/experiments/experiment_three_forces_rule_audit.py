#!/usr/bin/env python
"""Out-of-sample audit of the Three-Forces flow rules on every stock and all available history.

The rules were written while studying the 6 focus stocks in 2026. This grades each rule call directly against the
realized close-to-close return, separately on data the rules were not designed on: the other stocks, and the sessions
before 2026. A rule makes one constant call, so on its own firing sessions it always equals always-UP or always-DOWN.
The bar is therefore lift: its hit rate minus the same call's hit rate on all sessions of the same stocks, weighted by
how often each stock fired. Overlapping windows overstate the sample, so independent calls (one per horizon-length
window per stock) are shown as well.

Writes ~/data/mdk_oracle/experiments/timestral/three_forces_rule_audit.json
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from mdk_trading_oracle.models.three_forces_gate import GATED_HORIZONS, gate_direction
from mdk_trading_oracle.models.timestral_forecaster import HORIZON_DAYS, check_horizon_hit

BASE = Path.home() / "data" / "mdk_oracle" / "experiments"
FOCUS = {"AKBNK", "ASELS", "GARAN", "ISCTR", "THYAO", "YKBNK"}
DESIGN_START = "2026-01-01"
CALL_BY_DIRECTION = {"BEARISH": -5.0, "BULLISH": 5.0}


def load_events() -> pd.DataFrame:
    clash = pd.read_parquet(BASE / "timestral" / "three_forces_daily_clash.parquet")
    clash["trade_date"] = clash["trade_date"].astype(str).str[:10]
    rows: list[dict] = []
    for sym, g in clash.groupby("symbol"):
        frame = pd.read_parquet(BASE / "feature_importance" / "frames" / f"{sym}.parquet", columns=["trade_date", "adj_close_price"])
        frame["trade_date"] = frame["trade_date"].astype(str).str[:10]
        recs = g.merge(frame, on="trade_date").sort_values("trade_date").to_dict("records")
        px = np.array([r["adj_close_price"] for r in recs], dtype=float)
        for i, r in enumerate(recs):
            direction = gate_direction(r, "flow_only")
            for h in GATED_HORIZONS:
                n = HORIZON_DAYS[h]
                if i + n >= len(px) or px[i] <= 0:
                    continue
                rows.append({
                    "symbol": sym,
                    "trade_date": r["trade_date"],
                    "idx": i,
                    "horizon": h,
                    "direction": direction,
                    "actual": (px[i + n] / px[i] - 1.0) * 100.0,
                })
    return pd.DataFrame(rows)


def independent_calls(g: pd.DataFrame, n: int) -> pd.DataFrame:
    keep: list[int] = []
    for _, s in g.sort_values(["symbol", "idx"]).groupby("symbol"):
        last = -(10**9)
        for ix, i in zip(s.index, s["idx"]):
            if i - last >= n:
                keep.append(ix)
                last = i
    return g.loc[keep]


def build_baselines(ev: pd.DataFrame) -> dict[tuple[str, str, float], tuple[float, float]]:
    """(symbol, horizon, call) -> hit rate and mean return of that constant call on all of the stock's sessions."""
    out: dict[tuple[str, str, float], tuple[float, float]] = {}
    for (sym, h), g in ev.groupby(["symbol", "horizon"]):
        actual = g["actual"].to_numpy(float)
        for call in CALL_BY_DIRECTION.values():
            out[(sym, h, call)] = (float(np.mean([check_horizon_hit(call, a, h) for a in actual]) * 100.0), float(actual.mean()))
    return out


def grade(g: pd.DataFrame, h: str, direction: str, baselines: dict[tuple[str, str, float], tuple[float, float]]) -> dict:
    if g.empty:
        return {"n": 0, "n_independent": 0}
    call = CALL_BY_DIRECTION[direction]
    hits = np.array([check_horizon_hit(call, a, h) for a in g["actual"]], dtype=float)
    ind = independent_calls(g, HORIZON_DAYS[h])
    ind_hits = np.array([check_horizon_hit(call, a, h) for a in ind["actual"]], dtype=float)
    weights = g["symbol"].value_counts()
    base = [baselines[(sym, h, call)] for sym in weights.index]
    base_hit = float(np.average([b[0] for b in base], weights=weights.to_numpy(float)))
    base_ret = float(np.average([b[1] for b in base], weights=weights.to_numpy(float)))
    return {
        "n": int(len(g)),
        "n_independent": int(len(ind)),
        "stock_months": int(g.assign(m=g["trade_date"].str[:7])[["symbol", "m"]].drop_duplicates().shape[0]),
        "hit_pct": round(float(hits.mean() * 100), 1),
        "hit_independent_pct": round(float(ind_hits.mean() * 100), 1),
        "baseline_hit_pct": round(base_hit, 1),
        "lift_pp": round(float(hits.mean() * 100) - base_hit, 1),
        "lift_independent_pp": round(float(ind_hits.mean() * 100) - base_hit, 1),
        "mean_return_pct": round(float(g["actual"].mean()), 2),
        "baseline_mean_return_pct": round(base_ret, 2),
    }


def main() -> None:
    ev = load_events()
    fired = ev[ev["direction"].notna()].copy()
    baselines = build_baselines(ev)
    print(f"stock-horizon sessions graded: {len(ev)}, firing: {len(fired)}, stocks: {ev['symbol'].nunique()}, "
          f"{ev['trade_date'].min()} to {ev['trade_date'].max()}")
    splits = {
        "other_stocks_all_dates": ~fired["symbol"].isin(FOCUS),
        "focus_stocks_before_2026": fired["symbol"].isin(FOCUS) & (fired["trade_date"] < DESIGN_START),
        "focus_stocks_2026_design_period": fired["symbol"].isin(FOCUS) & (fired["trade_date"] >= DESIGN_START),
    }
    out: dict = {"splits": {}, "design_start": DESIGN_START, "total_rows": int(len(ev))}
    for name, mask in splits.items():
        out["splits"][name] = {}
        print(f"\n=== {name} ===")
        for direction in CALL_BY_DIRECTION:
            for h in GATED_HORIZONS:
                res = grade(fired[mask & (fired["direction"] == direction) & (fired["horizon"] == h)], h, direction, baselines)
                out["splits"][name][f"{direction}_{h}"] = res
                if res["n"]:
                    print(f"  {direction:<7} {h:<3} n={res['n']:<4} indep={res['n_independent']:<3} "
                          f"hit={res['hit_pct']:5.1f}% base={res['baseline_hit_pct']:5.1f}% lift={res['lift_pp']:+5.1f}pp "
                          f"indep_lift={res['lift_independent_pp']:+5.1f}pp "
                          f"ret={res['mean_return_pct']:+6.2f}% base_ret={res['baseline_mean_return_pct']:+6.2f}%")
                else:
                    print(f"  {direction:<7} {h:<3} n=0")
    path = BASE / "timestral" / "three_forces_rule_audit.json"
    path.write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(f"\nSaved {path}")


if __name__ == "__main__":
    main()
