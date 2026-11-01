#!/usr/bin/env python
"""Compare the Three-Forces gate modes (off, flow_only, flow_news) on the Timestral + Policy C forecasts.

Covers the 6 stocks in `direction_experiment_data.parquet`, on PRE-90 and on the last 90 sessions, and persists the
last 90 sessions to `gold_agentic_gate_audit` for the API. The last 90 sessions are NOT out-of-sample for this gate:
the rules and keywords were written while studying them (docs/AGENTIC_DECISIONS.md, D-002). Use
`experiment_three_forces_rule_audit.py` for the out-of-sample test.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from mdk_trading_oracle.core.db import PostgresManager
from mdk_trading_oracle.data.bronze.news import ensure_news_tables
from mdk_trading_oracle.models.three_forces_gate import GATED_HORIZONS
from mdk_trading_oracle.models.timestral_forecaster import (
    RAW_CACHE_DIR,
    blend_policy_c_with_timestral,
    check_horizon_hit,
)

MODES = ("off", "flow_only", "flow_news")
REGIME_COLS = [
    "symbol", "trade_date", "mlb_5d_share_pct", "mlb_10d_share_pct", "kamu_5d_share_pct", "big5_5d_share_pct",
    "big5_cost_spread_pct", "is_passive_absorption_trap", "three_forces_clash_score", "fon_krizi_ewm5",
    "bofa_sell_ewm5", "bullish_catalyst_ewm5",
]


def load_rows() -> pd.DataFrame:
    df = pd.read_parquet(RAW_CACHE_DIR / "direction_experiment_data.parquet")
    grid = pd.read_parquet(RAW_CACHE_DIR / "timestral_cross_asset_grid.parquet")
    reg = pd.read_parquet(RAW_CACHE_DIR / "agentic_news_regimes.parquet")
    return (
        df[df["horizon"].isin(GATED_HORIZONS)]
        .merge(
            grid[["symbol", "trade_date", "horizon", "s10_med", "m6_10_med", "p10_med", "mkt6_ts128_mean"]],
            on=["symbol", "trade_date", "horizon"],
            how="left",
        )
        .merge(reg[REGIME_COLS], on=["symbol", "trade_date"], how="left")
        .sort_values(["symbol", "horizon", "trade_date"])
        .reset_index(drop=True)
    )


def predict(r: dict, mode: str) -> float:
    p_out, _ = blend_policy_c_with_timestral(
        r["polc_crowned"],
        r["ts128_med"],
        r["ts128_q25"],
        r["ts128_q75"],
        r["horizon"],
        bist30_med=r["bist_l128"],
        bist30_ret_1d=r["bist30_return_pct"],
        days_since_neg_shock=r["feat_days_since_neg_shock"],
        days_since_pos_shock=r["feat_days_since_pos_shock"],
        bist30_vs_sma50=r["mkt_bist30_vs_sma50"],
        bist30_ret_20d=r["mkt_bist30_ret_20d"],
        ts_row=r,
        gate_mode=mode,
    )
    return float(p_out)


def score_frame(sub: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for r in sub.to_dict("records"):
        out = {"symbol": r["symbol"], "horizon": r["horizon"], "trade_date": r["trade_date"], "actual": float(r["actual"])}
        for mode in MODES:
            pred = predict(r, mode)
            out[f"pred_{mode}"] = pred
            out[f"hit_{mode}"] = bool(check_horizon_hit(pred, out["actual"], r["horizon"]))
        rows.append(out)
    return pd.DataFrame(rows)


def report(name: str, scored: pd.DataFrame) -> None:
    print(f"\n=== {name} (n={len(scored)}) ===")
    for h in GATED_HORIZONS:
        g = scored[scored["horizon"] == h]
        changed_flow = int(np.sum(np.sign(g["pred_flow_only"]) != np.sign(g["pred_off"])))
        changed_news = int(np.sum(np.sign(g["pred_flow_news"]) != np.sign(g["pred_off"])))
        print(
            f"  {h:<4} off={g['hit_off'].mean() * 100:5.2f}% flow_only={g['hit_flow_only'].mean() * 100:5.2f}% "
            f"flow_news={g['hit_flow_news'].mean() * 100:5.2f}% | flips flow={changed_flow} news={changed_news}"
        )


def persist_audit(db: PostgresManager, scored: pd.DataFrame) -> None:
    ensure_news_tables(db)
    conn = db.get_connection()
    with conn.cursor() as cur:
        cur.execute("TRUNCATE gold_agentic_gate_audit")
        cur.executemany(
            """
            INSERT INTO gold_agentic_gate_audit (symbol, horizon, trade_date, actual_return_pct, pred_off_pct,
                pred_flow_pct, pred_news_pct, hit_off, hit_flow, hit_news)
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
            """,
            [
                (r.symbol, r.horizon, r.trade_date, r.actual, r.pred_off, r.pred_flow_only, r.pred_flow_news,
                 bool(r.hit_off), bool(r.hit_flow_only), bool(r.hit_flow_news))
                for r in scored.itertuples(index=False)
            ],
        )


def main() -> None:
    rows = load_rows()
    eval90 = rows.groupby(["symbol", "horizon"], as_index=False).tail(90).copy()
    min_eval = eval90.groupby(["symbol", "horizon"])["trade_date"].min().to_dict()
    pre = rows[[r.target_date_end < min_eval[(r.symbol, r.horizon)] for r in rows.itertuples(index=False)]].copy()

    report("PRE-90", score_frame(pre))
    scored_eval = score_frame(eval90)
    report("LAST-90 (rules were written on this period)", scored_eval)

    persist_audit(PostgresManager(), scored_eval)
    print(f"\nWrote {len(scored_eval)} rows to gold_agentic_gate_audit")


if __name__ == "__main__":
    main()
