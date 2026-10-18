#!/usr/bin/env python
"""Robustness, costs and MAE of the 11:30 session signal (isolated experiment, read-only).

Signal (fixed in intraday_flow_experiment.py, chosen on 2023-2024): score = z(yesterday MLB intensity) + z(morning move),
trailing per-stock z-scores; call in the score's direction when it is beyond its trailing 80/20 percentile.
Entry 11:30 (last print, or VWAP 11:30-11:35 / 11:30-11:45), exit at the official closing auction.

Reports:
    1. stability    per stock and per half-year: strict hit vs base, sign accuracy, mean return per call, t-stat
    2. costs        entry half-spread from BIST tick sizes (exit in the auction pays no spread) + round-trip
                    commission scenarios; net mean return per call and per-stock t-stats
    3. slippage     last print vs 11:30-11:35 / 11:30-11:45 VWAP entry
    4. MAE          size forecast = call direction x trailing median |rest-of-session move| vs zero forecast and
                    vs random-sign forecast of the same size (same target: entry -> auction close)
    5. recency      z-score / threshold window 63 / 126 / 252 sessions (choose on discovery, read holdout)

Usage:
    .venv/bin/python scripts/experiments/session_signal_robustness.py
"""

from __future__ import annotations

import sys
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

sys.path.insert(0, str(Path(__file__).parent))
from intraday_flow_experiment import DB, DISCOVERY_END, EVAL_START, OUT_DIR, engineer  # noqa: E402

COMMISSIONS_RT = (0.0, 0.05, 0.10, 0.20)  # % round trip
TICKS = [(20, 0.01), (50, 0.02), (100, 0.05), (250, 0.10), (500, 0.25), (1000, 0.50), (2500, 1.00), (np.inf, 2.50)]


def tick_size(price: float) -> float:
    for upper, tick in TICKS:
        if price < upper:
            return tick
    return TICKS[-1][1]


def signal(p: pd.DataFrame, window: int = 252, q: float = 0.8) -> pd.DataFrame:
    c = p.sort_values(["symbol", "trade_date"]).copy()
    mp = max(40, window // 2)
    for f in ("mlb_net_yday_intensity", "morning_ret"):
        g = c.groupby("symbol")[f]
        mu = g.transform(lambda x: x.shift(1).rolling(window, min_periods=mp).mean())
        sd = g.transform(lambda x: x.shift(1).rolling(window, min_periods=mp).std())
        c[f"z_{f}"] = ((c[f] - mu) / sd).clip(-5, 5)
    c["score"] = c["z_mlb_net_yday_intensity"] + c["z_morning_ret"]
    gs = c.groupby("symbol")["score"]
    hi = gs.transform(lambda x: x.shift(1).rolling(window, min_periods=mp).quantile(q))
    lo = gs.transform(lambda x: x.shift(1).rolling(window, min_periods=mp).quantile(1 - q))
    c["call"] = np.where(c["score"] > hi, 1, np.where(c["score"] < lo, -1, 0))
    c["period"] = np.where(c["trade_date"] <= pd.Timestamp(DISCOVERY_END), "discovery", "holdout")
    # MAE-optimal size: trailing mean of the stock's own past directional outcomes on called sessions (strictly before d)
    past = (c["call"] * c["rest_1130"]).where(c["call"] != 0)
    c["cond_mean_size"] = past.groupby(c["symbol"]).transform(lambda x: x.shift(1).expanding(min_periods=30).mean())
    c["half"] = c["trade_date"].dt.year.astype(str) + "H" + np.where(c["trade_date"].dt.month <= 6, "1", "2")
    # entry variants and costs
    c["ret_last"] = (c["close_price"] / c["p_1130"] - 1) * 100
    c["ret_vwap5"] = (c["close_price"] / c["vwap_1130_1135"] - 1) * 100
    c["ret_vwap15"] = (c["close_price"] / c["vwap_1130_1145"] - 1) * 100
    c["half_spread_pct"] = c["p_1130"].map(tick_size) / 2 / c["p_1130"] * 100
    c["med_abs_move"] = c.groupby("symbol")["ret_last"].transform(lambda x: x.abs().shift(1).rolling(window, min_periods=mp).median())
    return c[c["trade_date"] >= EVAL_START]


def call_stats(g: pd.DataFrame, ret_col: str = "ret_last", cost: pd.Series | float = 0.0) -> dict[str, float]:
    k = g["call"] != 0
    y = g[ret_col]
    pnl = g.loc[k, "call"] * y[k] - (cost[k] if isinstance(cost, pd.Series) else cost)
    base_up, base_dn = (y > DB).mean(), (y < -DB).mean()
    base = np.where(g.loc[k, "call"] > 0, base_up, base_dn)
    hit = (g.loc[k, "call"] * y[k] > DB)
    return {"n_calls": int(k.sum()), "coverage_pct": k.mean() * 100, "strict_hit_pct": hit.mean() * 100, "base_pct": base.mean() * 100,
            "edge_pp": (hit.mean() - base.mean()) * 100, "sign_acc_pct": (np.sign(g.loc[k, "call"]) == np.sign(y[k])).mean() * 100,
            "mean_ret_pct": pnl.mean(), "t_stat": pnl.mean() / (pnl.std(ddof=1) / np.sqrt(len(pnl))) if len(pnl) > 2 else np.nan}


def main() -> None:
    pd.set_option("display.width", 250)
    pd.set_option("display.max_rows", 100)
    panel, _ = engineer(pd.read_parquet(OUT_DIR / "tick_panel.parquet"))
    c = signal(panel.dropna(subset=["rest_1130", "vwap_1130_1135", "vwap_1130_1145"]))
    c = c[c["call"].notna()]

    print("=== 1. stability (entry = last print before 11:30, no costs) ===")
    rows = [{"group": f"{per} {sym}", **call_stats(g)} for (per, sym), g in c.groupby(["period", "symbol"])]
    rows += [{"group": f"{per} ALL6", **call_stats(g)} for per, g in c.groupby("period")]
    print(pd.DataFrame(rows).set_index("group").round(2).to_string())
    print("\nper half-year (ALL6)")
    print(pd.DataFrame([{"half": h, **call_stats(g)} for h, g in c.groupby("half")]).set_index("half").round(2).to_string())

    print("\n=== 2+3. costs and slippage, holdout (net mean return per call, %) ===")
    h = c[c["period"] == "holdout"]
    print(f"median entry half-spread from tick size: {h['half_spread_pct'].median():.3f}% (per stock: "
          + ", ".join(f"{s} {v:.3f}" for s, v in h.groupby('symbol')['half_spread_pct'].median().items()) + ")")
    rows = []
    for entry, col, spread in (("last print + half-spread", "ret_last", True), ("VWAP 11:30-11:35", "ret_vwap5", False), ("VWAP 11:30-11:45", "ret_vwap15", False)):
        for comm in COMMISSIONS_RT:
            cost = (h["half_spread_pct"] if spread else 0.0) + comm
            all6 = call_stats(h, col, cost)
            per_stock = [call_stats(g, col, (g["half_spread_pct"] if spread else 0.0) + comm) for _, g in h.groupby("symbol")]
            rows.append({"entry": entry, "commission_rt_pct": comm, "net_mean_ret_pct": all6["mean_ret_pct"], "t_stat": all6["t_stat"],
                         "stocks_net_positive": sum(s["mean_ret_pct"] > 0 for s in per_stock), "sign_acc_pct": all6["sign_acc_pct"]})
    print(pd.DataFrame(rows).round(3).to_string(index=False))

    print("\n=== 4. MAE on called sessions (target = last print 11:30 -> auction close, %) ===")
    rng = np.random.default_rng(1)
    rows = []
    for per, g in c.groupby("period"):
        k = (g["call"] != 0) & g["med_abs_move"].notna() & g["cond_mean_size"].notna()
        y = g.loc[k, "ret_last"]
        for size_name in ("med_abs_move", "cond_mean_size"):
            size = g.loc[k, size_name].clip(lower=0.0)
            pred = g.loc[k, "call"] * size
            rand = np.mean([np.abs(rng.choice([-1, 1], size=k.sum()) * size - y).mean() for _ in range(300)])
            hitm = np.sign(pred) == np.sign(y)
            rows.append({"period": per, "size": size_name, "n": int(k.sum()), "median_size": size.median(), "mae_signal": (pred - y).abs().mean(),
                         "mae_random_sign_same_size": rand, "mae_zero_forecast": y.abs().mean(),
                         "hit_mae": (pred - y)[hitm].abs().mean(), "miss_mae": (pred - y)[~hitm].abs().mean()})
    print(pd.DataFrame(rows).set_index(["period", "size"]).round(3).to_string())

    print("\n=== 5. recency: z-score / threshold window (choose on discovery, read holdout) ===")
    base = panel.dropna(subset=["rest_1130", "vwap_1130_1135", "vwap_1130_1145"])
    rows = []
    for w in (63, 126, 252):
        cw = signal(base, window=w)
        for per, g in cw.groupby("period"):
            st = call_stats(g)
            per_stock = [call_stats(gs)["edge_pp"] for _, gs in g.groupby("symbol")]
            rows.append({"window": w, "period": per, **{k: st[k] for k in ("n_calls", "coverage_pct", "strict_hit_pct", "base_pct", "edge_pp", "mean_ret_pct", "t_stat")},
                         "stocks_pos": sum(e > 0 for e in per_stock)})
    print(pd.DataFrame(rows).round(2).to_string(index=False))


if __name__ == "__main__":
    main()
