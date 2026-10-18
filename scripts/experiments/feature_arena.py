#!/usr/bin/env python
"""Incremental feature arena: add one (or a pair of) feature(s) at a time to a base signal (isolated, read-only).

Bases (row = session N; inputs known at the close of N-1 unless stated):
    session_v1   z(yesterday MLB intensity) + z(morning move, trader-observed live)  -> 11:30 entry, close exit
    eod_v1       z(yesterday MLB intensity)                                          -> open entry, close exit
Each candidate x is z-scored on its trailing 252 sessions per stock; its sign s is the sign of its DISCOVERY IC with
the base's target; score = base + s * z(x) (pairs: + s1 z1 + s2 z2). Calls beyond the trailing 80/20 percentile of
past scores (as the base).

Pre-declared acceptance (DISCOVERY 2023-2024 only): sign accuracy +1.0 pp AND strict edge +1.0 pp vs the base AND
sign accuracy improved on >= 4 of 6 stocks. Holdout (2025-2026) is printed for transparency, never used to decide.
Every trial is appended to trial_registry.csv (the number of trials matters when judging any winner).
Sessions right after missing raw-data days (no "yesterday") are excluded.

Usage:
    .venv/bin/python scripts/experiments/feature_arena.py                       # default candidate batch
    .venv/bin/python scripts/experiments/feature_arena.py --only kamu_yday,big5_yday
"""

from __future__ import annotations

import argparse
import sys
import warnings
from datetime import datetime
from pathlib import Path

warnings.filterwarnings("ignore")

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

sys.path.insert(0, str(Path(__file__).parent))
from intraday_flow_experiment import DB, DISCOVERY_END, EVAL_START, OUT_DIR, engineer  # noqa: E402

OUT = Path.home() / "data" / "mdk_oracle" / "experiments" / "feature_arena"
WINDOW, MIN_P, Q = 252, 126, 0.8
MISSING_RAW_DAYS = ["2026-09-25", "2026-10-01", "2026-10-02", "2026-10-05"]  # trading days without raw files
ACCEPT = {"sign_acc_pp": 1.0, "edge_pp": 1.0, "stocks_improved": 4}

HYPOTHESES = {
    "kamu_yday": "KAMU net intensity in the stock yesterday (state desks' positioning carries into the next session)",
    "big5_yday": "BIG5 net intensity in the stock yesterday (domestic bank-broker / retail-prop positioning)",
    "kamu_w5_yday": "KAMU closing-session net share yesterday (closing-level support fades or persists)",
    "kamu_minus_mlb_yday": "KAMU absorbing MLB (KAMU z - MLB z) in the stock yesterday",
    "idx_kamu_yday": "index-level KAMU net share across BIST30 yesterday (index defense)",
    "idx_big5_yday": "index-level BIG5 net share across BIST30 yesterday",
    "defended_selloff_yday": "K4 flag: BIST30 <= -2% and KAMU buying while MLB selling yesterday (fades next session)",
    "bist30_ret_yday": "BIST30 close-to-close return yesterday (index momentum / reversal)",
    "days_since_neg_shock": "sessions since the last BIST30 <= -3% day, capped at 10 (post-shock regime)",
}
PAIRS = {"kamu+big5": ("kamu_yday", "big5_yday"), "kamu+idx_kamu": ("kamu_yday", "idx_kamu_yday")}


def load() -> pd.DataFrame:
    from mdk_trading_oracle.core.db import PostgresManager
    from mdk_trading_oracle.data.silver.tertip_analytics import BIG_FIVE_BROKERS, KAMU_BROKERS

    p, _ = engineer(pd.read_parquet(OUT_DIR / "tick_panel.parquet"))
    p = p.dropna(subset=["rest_1130"]).sort_values(["symbol", "trade_date"]).reset_index(drop=True)
    p["oc"] = (p["close_price"] / p["open_price"] - 1) * 100
    db = PostgresManager(read_only=True)
    syms = sorted(p["symbol"].unique())
    fl = db.query_df(
        "SELECT symbol, trade_date, broker_id, net_flow_tl FROM silver_daily_broker_summary WHERE symbol = ANY(%s) AND broker_id = ANY(%s)",
        (syms, ["MLB", *BIG_FIVE_BROKERS, *KAMU_BROKERS]),
    )
    fl["grp"] = np.where(fl["broker_id"] == "MLB", "mlb", np.where(fl["broker_id"].isin(BIG_FIVE_BROKERS), "big5", "kamu"))
    w5 = db.query_df(
        "SELECT symbol, trade_date, net_flow_tl FROM silver_intraday_broker_window_summary WHERE symbol = ANY(%s) AND broker_id = ANY(%s) AND window_name = 'closing_session'",
        (syms, list(KAMU_BROKERS)),
    ).groupby(["symbol", "trade_date"])["net_flow_tl"].sum().rename("kamu_w5_net")
    s = db.query_df("SELECT symbol, trade_date, total_turnover_tl FROM silver_daily_stock_summary WHERE symbol = ANY(%s)", (syms,))
    daily = s.merge(fl.pivot_table(index=["symbol", "trade_date"], columns="grp", values="net_flow_tl", aggfunc="sum").reset_index(), on=["symbol", "trade_date"], how="left")
    daily = daily.merge(w5.reset_index(), on=["symbol", "trade_date"], how="left").fillna({"kamu": 0.0, "big5": 0.0, "mlb": 0.0, "kamu_w5_net": 0.0})
    daily["trade_date"] = pd.to_datetime(daily["trade_date"])
    daily = daily.sort_values(["symbol", "trade_date"])
    med = daily.groupby("symbol")["total_turnover_tl"].transform(lambda x: x.rolling(20, min_periods=5).median())
    for g in ("kamu", "big5", "mlb"):
        daily[f"{g}_int"] = daily[g] / med * 100
    daily["kamu_w5_sh"] = daily["kamu_w5_net"] / daily["total_turnover_tl"] * 100
    # yesterday's values aligned to the next session of the same stock (silver calendar)
    y = daily[["symbol", "trade_date"]].copy()
    for src, dst in (("kamu_int", "kamu_yday"), ("big5_int", "big5_yday"), ("kamu_w5_sh", "kamu_w5_yday"), ("mlb_int", "_mlb_yday")):
        y[dst] = daily.groupby("symbol")[src].shift(1)
    y["prev_date"] = daily.groupby("symbol")["trade_date"].shift(1)
    p = p.merge(y, on=["symbol", "trade_date"], how="left")

    ix = pd.read_parquet(Path.home() / "data" / "mdk_oracle" / "experiments" / "kamu_defense" / "index_flows.parquet").sort_values("trade_date")
    for c in ("kamu_day_share", "mlb_day_share"):
        mu, sd = ix[c].shift(1).rolling(WINDOW, min_periods=MIN_P).mean(), ix[c].shift(1).rolling(WINDOW, min_periods=MIN_P).std()
        ix[f"z_{c}"] = (ix[c] - mu) / sd
    ix["defended_selloff"] = ((ix["bist30_ret"] <= -2) & (ix["z_kamu_day_share"] > 0) & (ix["z_mlb_day_share"] < 0)).astype(float)
    pos = pd.Series(np.arange(len(ix)), index=ix.index)
    last_shock = pos.where(ix["bist30_ret"] <= -3).ffill()
    ix["days_since_neg_shock_t"] = (pos - last_shock).clip(upper=10).fillna(10)
    ix_prev = ix[["trade_date", "kamu_day_share", "big5_day_share", "defended_selloff", "bist30_ret", "days_since_neg_shock_t"]].rename(
        columns={"trade_date": "prev_date", "kamu_day_share": "idx_kamu_yday", "big5_day_share": "idx_big5_yday",
                 "defended_selloff": "defended_selloff_yday", "bist30_ret": "bist30_ret_yday", "days_since_neg_shock_t": "days_since_neg_shock"})
    p = p.merge(ix_prev, on="prev_date", how="left")
    p["kamu_minus_mlb_yday"] = p["kamu_yday"] - p["_mlb_yday"]  # z-scored below like every candidate

    # drop sessions whose "yesterday" crosses a missing raw-data day
    gap = pd.to_datetime(MISSING_RAW_DAYS)
    bad = p.apply(lambda r: pd.notna(r["prev_date"]) and any((gap > r["prev_date"]) & (gap < r["trade_date"])), axis=1)
    p = p[~bad].copy()
    p["period"] = np.where(p["trade_date"] < EVAL_START, None, np.where(p["trade_date"] <= DISCOVERY_END, "discovery", "holdout"))
    return p.sort_values(["symbol", "trade_date"]).reset_index(drop=True)


def z(p: pd.DataFrame, f: str) -> pd.Series:
    g = p.groupby("symbol")[f]
    mu = g.transform(lambda x: x.shift(1).rolling(WINDOW, min_periods=MIN_P).mean())
    sd = g.transform(lambda x: x.shift(1).rolling(WINDOW, min_periods=MIN_P).std()).replace(0, np.nan)
    return ((p[f] - mu) / sd).clip(-5, 5)


def calls(p: pd.DataFrame, score: pd.Series) -> pd.Series:
    g = score.groupby(p["symbol"])
    hi = g.transform(lambda x: x.shift(1).rolling(WINDOW, min_periods=MIN_P).quantile(Q))
    lo = g.transform(lambda x: x.shift(1).rolling(WINDOW, min_periods=MIN_P).quantile(1 - Q))
    return pd.Series(np.where(score > hi, 1, np.where(score < lo, -1, 0)), index=p.index)


def evaluate(p: pd.DataFrame, call: pd.Series, target: str, period: str) -> dict:
    m = (p["period"] == period) & p[target].notna()
    k = m & (call != 0)
    y = p.loc[k, target]
    ok = np.sign(call[k]) == np.sign(y)
    base = np.where(call[k] > 0, (p.loc[m, target] > DB).mean(), (p.loc[m, target] < -DB).mean())
    pnl = call[k] * y
    return {"coverage_pct": k.sum() / m.sum() * 100, "sign_acc_pct": ok.mean() * 100, "edge_pp": ((pnl > DB).mean() - base.mean()) * 100,
            "mean_ret_pct": pnl.mean(), "t_stat": pnl.mean() / pnl.std(ddof=1) * np.sqrt(len(pnl)),
            "_per_stock_acc": pd.Series(ok.to_numpy(), index=p.loc[k, "symbol"].to_numpy()).groupby(level=0).mean()}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", default="", help="comma-separated candidate / pair names (default: all)")
    a = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    pd.set_option("display.width", 250)
    p = load()
    zs = {f: z(p, f) for f in ["mlb_net_yday_intensity", "morning_ret", *HYPOTHESES]}
    bases = {"session_v1": (zs["mlb_net_yday_intensity"] + zs["morning_ret"], "rest_1130"),
             "eod_v1": (zs["mlb_net_yday_intensity"], "oc")}
    wanted = [w for w in a.only.split(",") if w] or list(HYPOTHESES) + list(PAIRS)
    disc = p["period"] == "discovery"
    registry, rows = [], []
    for bname, (bscore, target) in bases.items():
        bcall = calls(p, bscore)
        bd, bh = evaluate(p, bcall, target, "discovery"), evaluate(p, bcall, target, "holdout")
        rows.append({"base": bname, "candidate": "(base)", "decision": "", **{f"d_{k}": v for k, v in bd.items() if not k.startswith("_")},
                     **{f"h_{k}": v for k, v in bh.items() if not k.startswith("_")}})
        for cand in wanted:
            feats = PAIRS.get(cand, (cand,))
            signs = {f: float(np.sign(p.loc[disc, f].corr(p.loc[disc, target], method="spearman")) or 1.0) for f in feats}
            score = bscore + sum(signs[f] * zs[f].fillna(0.0) for f in feats)
            c = calls(p, score)
            d, h = evaluate(p, c, target, "discovery"), evaluate(p, c, target, "holdout")
            improved = int((d["_per_stock_acc"] > bd["_per_stock_acc"].reindex(d["_per_stock_acc"].index)).sum())
            dacc, dedge = d["sign_acc_pct"] - bd["sign_acc_pct"], d["edge_pp"] - bd["edge_pp"]
            accepted = dacc >= ACCEPT["sign_acc_pp"] and dedge >= ACCEPT["edge_pp"] and improved >= ACCEPT["stocks_improved"]
            row = {"base": bname, "candidate": cand, "signs": ",".join(f"{f}:{'+' if s > 0 else '-'}" for f, s in signs.items()),
                   "decision": "ACCEPT" if accepted else "reject", "d_delta_sign_acc_pp": dacc, "d_delta_edge_pp": dedge, "d_stocks_improved": improved,
                   **{f"d_{k}": v for k, v in d.items() if not k.startswith("_")}, **{f"h_{k}": v for k, v in h.items() if not k.startswith("_")},
                   "h_delta_sign_acc_pp": h["sign_acc_pct"] - bh["sign_acc_pct"]}
            rows.append(row)
            registry.append({"run_at": datetime.now().isoformat(timespec="seconds"), "hypothesis": "; ".join(HYPOTHESES.get(f, f) for f in feats), **row})
    res = pd.DataFrame(rows)
    reg_path = OUT / "trial_registry.csv"
    pd.DataFrame(registry).to_csv(reg_path, mode="a", header=not reg_path.exists(), index=False)
    cols = ["base", "candidate", "signs", "decision", "d_delta_sign_acc_pp", "d_delta_edge_pp", "d_stocks_improved", "d_sign_acc_pct", "d_edge_pp",
            "h_sign_acc_pct", "h_edge_pp", "h_delta_sign_acc_pp", "h_mean_ret_pct"]
    print(res[[c for c in cols if c in res.columns]].round(2).to_string(index=False))
    n_trials = len(pd.read_csv(reg_path))
    print(f"\ntrial registry: {n_trials} trials so far ({reg_path})")


if __name__ == "__main__":
    main()
