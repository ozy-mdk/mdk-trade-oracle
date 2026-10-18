#!/usr/bin/env python
"""Multi-day institutional accumulation -> multi-day returns (isolated experiment, read-only; end-of-day data only).

Hypothesis: an information advantage (KAMU in particular) shows up as accumulation over several sessions that leads
multi-day moves, not next-day moves.
    features   cumulative net-flow intensity (net TL / trailing 20d median turnover, %) over the last L in {5, 10, 20}
               sessions ending at the close of T, for KAMU / BIG5 / MLB; stock level (focus 6) and BIST30 level
    targets    forward close-to-close return T -> T+h, h in {5, 10, 20} (adjusted prices)
    controls   past L-session return (KAMU buys falls: cumulative flow may only be reversal in disguise) -> partial IC;
               placebo = cumulative iid noise
    overlap    IC averaged over the h non-overlapping sub-samples (offsets); t uses the independent count n / h
Discovery 2023-2024 decides; holdout 2025-2026 read afterwards.

Usage:
    .venv/bin/python scripts/experiments/horizon_flow_experiment.py
"""

from __future__ import annotations

import warnings
from pathlib import Path

warnings.filterwarnings("ignore")

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from scipy.stats import rankdata  # noqa: E402

FOCUS = ["ASELS", "GARAN", "AKBNK", "YKBNK", "ISCTR", "TTKOM"]
OUT = Path.home() / "data" / "mdk_oracle" / "experiments" / "horizon_flow"
EVAL_START, DISCOVERY_END = "2023-01-01", "2024-12-31"
LOOKBACKS, HORIZONS = (5, 10, 20), (5, 10, 20)
GROUPS = ("kamu", "big5", "mlb")


def load_stocks() -> pd.DataFrame:
    from mdk_trading_oracle.core.db import PostgresManager
    from mdk_trading_oracle.data.silver.tertip_analytics import BIG_FIVE_BROKERS, KAMU_BROKERS

    db = PostgresManager(read_only=True)
    fl = db.query_df(
        "SELECT symbol, trade_date, broker_id, net_flow_tl FROM silver_daily_broker_summary WHERE symbol = ANY(%s) AND broker_id = ANY(%s)",
        (FOCUS, ["MLB", *BIG_FIVE_BROKERS, *KAMU_BROKERS]),
    )
    fl["grp"] = np.where(fl["broker_id"] == "MLB", "mlb", np.where(fl["broker_id"].isin(BIG_FIVE_BROKERS), "big5", "kamu"))
    s = db.query_df("SELECT symbol, trade_date, adj_close_price, total_turnover_tl FROM silver_daily_stock_summary WHERE symbol = ANY(%s)", (FOCUS,))
    d = s.merge(fl.pivot_table(index=["symbol", "trade_date"], columns="grp", values="net_flow_tl", aggfunc="sum").reset_index(),
                on=["symbol", "trade_date"], how="left").fillna({g: 0.0 for g in GROUPS})
    d["trade_date"] = pd.to_datetime(d["trade_date"])
    return d.sort_values(["symbol", "trade_date"]).reset_index(drop=True)


def load_index() -> pd.DataFrame:
    ix = pd.read_parquet(Path.home() / "data" / "mdk_oracle" / "experiments" / "kamu_defense" / "index_flows.parquet")
    from mdk_trading_oracle.core.db import PostgresManager

    px = PostgresManager(read_only=True).query_df(
        "SELECT trade_date, close_price AS adj_close_price FROM silver_daily_benchmark_index WHERE index_code = 'XU030' AND NOT COALESCE(is_forward_filled, FALSE)")
    px["trade_date"] = pd.to_datetime(px["trade_date"])
    d = px.merge(ix[["trade_date", "bist30_tl", *[f"{g}_day_net" for g in GROUPS]]], on="trade_date", how="inner")
    d = d.rename(columns={"bist30_tl": "total_turnover_tl", **{f"{g}_day_net": g for g in GROUPS}})
    d["symbol"] = "BIST30"
    return d.sort_values("trade_date").reset_index(drop=True)


def build(d: pd.DataFrame) -> pd.DataFrame:
    g = d.groupby("symbol")
    med = g["total_turnover_tl"].transform(lambda x: x.rolling(20, min_periods=5).median())
    rng = np.random.default_rng(13)
    d["noise"] = rng.normal(size=len(d))
    for grp in (*GROUPS, "noise"):
        inten = d[grp] / med * 100 if grp != "noise" else d["noise"]
        for L in LOOKBACKS:
            d[f"{grp}_cum{L}"] = inten.groupby(d["symbol"]).transform(lambda x: x.rolling(L, min_periods=L).sum())
    for L in LOOKBACKS:
        d[f"past_ret{L}"] = (d["adj_close_price"] / g["adj_close_price"].shift(L) - 1) * 100
    for h in HORIZONS:
        d[f"fwd{h}"] = (g["adj_close_price"].shift(-h) / d["adj_close_price"] - 1) * 100
    d["period"] = np.where(d["trade_date"] < EVAL_START, None, np.where(d["trade_date"] <= DISCOVERY_END, "discovery", "holdout"))
    return d[d["period"].notna()]


def spearman(x: np.ndarray, y: np.ndarray) -> float:
    return float(np.corrcoef(rankdata(x), rankdata(y))[0, 1])


def partial_spearman(x: np.ndarray, y: np.ndarray, z: np.ndarray) -> float:
    rx, ry, rz = rankdata(x), rankdata(y), rankdata(z)
    bx, by = np.polyfit(rz, rx, 1), np.polyfit(rz, ry, 1)
    return float(np.corrcoef(rx - np.polyval(bx, rz), ry - np.polyval(by, rz))[0, 1])


def nonoverlap_ic(g: pd.DataFrame, f: str, t: str, h: int, control: str | None) -> tuple[float, int]:
    ics, n = [], 0
    g = g.dropna(subset=[f, t] + ([control] if control else [])).reset_index(drop=True)
    for off in range(h):
        s = g.iloc[off::h]
        if len(s) < 15:
            continue
        ics.append(partial_spearman(s[f].to_numpy(), s[t].to_numpy(), s[control].to_numpy()) if control else spearman(s[f].to_numpy(), s[t].to_numpy()))
        n = len(s)
    return (float(np.mean(ics)) if ics else np.nan), n


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    pd.set_option("display.width", 250)
    pd.set_option("display.max_rows", 200)
    stocks, index = build(load_stocks()), build(load_index())
    rows = []
    for level, d in (("stock (mean of 6)", stocks), ("BIST30 index", index)):
        for grp in (*GROUPS, "noise"):
            for L in LOOKBACKS:
                for h in HORIZONS:
                    f, t = f"{grp}_cum{L}", f"fwd{h}"
                    r = {"level": level, "feature": f, "horizon": h}
                    for per in ("discovery", "holdout"):
                        ic, pic, ns, pos = [], [], [], 0
                        for _, g in d[d["period"] == per].groupby("symbol"):
                            a, n = nonoverlap_ic(g, f, t, h, None)
                            b, _ = nonoverlap_ic(g, f, t, h, f"past_ret{L}")
                            ic.append(a)
                            pic.append(b)
                            ns.append(n)
                            pos += int(a > 0)
                        m = float(np.nanmean(ic))
                        n_ind = int(np.sum(ns))
                        r[f"ic_{per[:4]}"] = m
                        r[f"partial_ic_{per[:4]}"] = float(np.nanmean(pic))
                        r[f"t_{per[:4]}"] = m * np.sqrt(max(n_ind - 2, 1)) / np.sqrt(max(1 - m * m, 1e-9))
                        r[f"n_ind_{per[:4]}"] = n_ind
                        r[f"pos_{per[:4]}"] = pos
                    r["same_sign"] = np.sign(r["ic_disc"]) == np.sign(r["ic_hold"])
                    rows.append(r)
    res = pd.DataFrame(rows)
    res.to_csv(OUT / "horizon_ic.csv", index=False)
    cols = ["level", "feature", "horizon", "ic_disc", "partial_ic_disc", "t_disc", "pos_disc", "ic_hold", "partial_ic_hold", "t_hold", "pos_hold", "n_ind_disc", "n_ind_hold", "same_sign"]
    print("=== cumulative flow (last L sessions) vs forward h-session return; partial = controlling for past L-session return ===")
    print(res[cols].round(3).to_string(index=False))
    strong = res[(res["t_disc"].abs() >= 2) & ~res["feature"].str.startswith("noise")]
    print("\n=== |t| >= 2 in discovery (candidates), with holdout ===")
    print(strong[cols].round(3).to_string(index=False) if len(strong) else "none")


if __name__ == "__main__":
    main()
