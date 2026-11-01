#!/usr/bin/env python
"""Zero-Leakage Walk-Forward Experiment: Conditioning Timestral-209M + Policy C on
the Agentic 3-Forces Clash & Point-in-Time News Regime (`agentic_news_regimes.parquet`).

Evaluates strictly across:
1. `PRE-90` Calibration Split (`target_date_end < min_eval_trade_date`, N = 6,792 rows)
2. `EVAL-90` Untouched 90-Session Backtest Split (`N = 3,240` rows)
3. Deep-Dive on `ASELS 10D` during the Sept-Oct 2026 'Fon Krizi' window (`2026-09-14` to `2026-10-09`).
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from mdk_trading_oracle.data.gold.tertip_multi_horizon_crowning import constant_call_hit_rates
from mdk_trading_oracle.models.timestral_forecaster import (
    HORIZON_DAYS,
    HORIZONS_ALL,
    RAW_CACHE_DIR,
    blend_policy_c_with_timestral,
    check_horizon_hit,
)


def main() -> None:
    df = pd.read_parquet(RAW_CACHE_DIR / "direction_experiment_data.parquet")
    grid = pd.read_parquet(RAW_CACHE_DIR / "timestral_cross_asset_grid.parquet")
    reg = pd.read_parquet(RAW_CACHE_DIR / "agentic_news_regimes.parquet")

    reg_cols = [
        "symbol", "trade_date",
        "mlb_5d_share_pct", "mlb_10d_share_pct", "kamu_5d_share_pct", "big5_5d_share_pct",
        "mlb_w5_5d_share_pct", "mlb_cost_spread_pct", "big5_cost_spread_pct", "kamu_cost_spread_pct",
        "is_passive_absorption_trap", "three_forces_clash_score",
        "mkt6_absorption_trap_breadth", "mkt6_three_forces_clash_mean",
        "fon_krizi_news_intensity", "bofa_sell_news_intensity", "kamu_floor_news_intensity", "bullish_catalyst_intensity",
        "fon_krizi_ewm5", "bofa_sell_ewm5", "kamu_floor_ewm5", "bullish_catalyst_ewm5",
        "fon_krizi_ewm10", "bofa_sell_ewm10", "kamu_floor_ewm10", "bullish_catalyst_ewm10",
        "agentic_net_news_sentiment", "agentic_3forces_news_composite", "agentic_news_regime",
    ]
    m = (
        df.merge(
            grid[["symbol", "trade_date", "horizon", "s10_med", "m6_10_med", "p10_med", "mkt6_ts128_mean"]],
            on=["symbol", "trade_date", "horizon"],
            how="left",
        )
        .merge(reg[reg_cols], on=["symbol", "trade_date"], how="left")
        .sort_values(["symbol", "horizon", "trade_date"])
        .reset_index(drop=True)
    )

    eval90 = m.groupby(["symbol", "horizon"], as_index=False).tail(90).copy()
    min_eval_td = eval90.groupby(["symbol", "horizon"])["trade_date"].min().to_dict()
    pre_df = m[[r.target_date_end < min_eval_td[(r.symbol, r.horizon)] for r in m.itertuples(index=False)]].copy()

    def eval_row(r: dict, h: str, use_agentic_3forces: bool) -> float:
        r_in = dict(r)
        if not use_agentic_3forces:
            r_in["is_passive_absorption_trap"] = False
            r_in["mlb_5d_share_pct"] = 0.0
        p_out, _ = blend_policy_c_with_timestral(
            r_in["polc_crowned"],
            r_in["ts128_med"],
            r_in["ts128_q25"],
            r_in["ts128_q75"],
            h,
            bist30_med=r_in["bist_l128"],
            bist30_ret_1d=r_in["bist30_return_pct"],
            days_since_neg_shock=r_in["feat_days_since_neg_shock"],
            days_since_pos_shock=r_in["feat_days_since_pos_shock"],
            bist30_vs_sma50=r_in["mkt_bist30_vs_sma50"],
            bist30_ret_20d=r_in["mkt_bist30_ret_20d"],
            ts_row=r_in,
        )
        return float(p_out)

    # 1. Inspect ASELS 10D during 2026-09-10 to 2026-10-09 (the exact window from user screenshot!)
    asels_10d = eval90[(eval90["symbol"] == "ASELS") & (eval90["horizon"] == "10d")].copy()
    asels_crisis = asels_10d[asels_10d["trade_date"] >= "2026-09-10"].copy()
    print("=== DEEP DIVE: ASELS 10D Forecast vs Realized During 'Fon Krizi' (2026-09-10 to 2026-10-09) ===")
    print(f"{'Date':<10} | {'Actual 10D':>10} | {'Before (No News)':>16} | {'After (3F+News)':>16} | {'Regime / Trap':<28} | {'MLB 5D%':>8} | {'FK_EWM':>6}")
    print("-" * 112)
    hits_before_w, hits_after_w = 0, 0
    for r in asels_crisis.to_dict("records"):
        p_bef = eval_row(r, "10d", False)
        p_aft = eval_row(r, "10d", True)
        act = float(r["actual"])
        hb = check_horizon_hit(p_bef, act, "10d")
        ha = check_horizon_hit(p_aft, act, "10d")
        hits_before_w += int(hb)
        hits_after_w += int(ha)
        reg_lbl = f"{r['agentic_news_regime']} (Trap={r['is_passive_absorption_trap']})"
        print(
            f"{r['trade_date']:<10} | {act:+9.2f}% | {p_bef:+8.2f}% ({'HIT ' if hb else 'MISS'}) | "
            f"{p_aft:+8.2f}% ({'HIT ' if ha else 'MISS'}) | {reg_lbl:<28} | {r['mlb_5d_share_pct']:+7.2f}% | {r['fon_krizi_ewm5']:6.3f}"
        )
    n_w = len(asels_crisis)
    print(f"--> ASELS 10D Sept-Oct Window Hit Rate: Before = {hits_before_w}/{n_w} ({hits_before_w/max(1,n_w)*100:.1f}%) -> After = {hits_after_w}/{n_w} ({hits_after_w/max(1,n_w)*100:.1f}%)!")

    # 2. Full Zero-Leakage Evaluation across PRE-90 (N=6,792) and EVAL-90 (N=3,240)
    for split_name, sub_df in [
        ("PRE-90 (Strictly Prior Calibration Set, N=6,792)", pre_df),
        ("EVAL-90 (Untouched 90-Session Backtest, N=3,240)", eval90),
    ]:
        print(f"\n=== {split_name} ===")
        hrs_bef, hrs_aft = [], []
        edge_bef_tot, edge_aft_tot = 0, 0
        for h in HORIZONS_ALL:
            sh = sub_df[sub_df["horizon"] == h]
            recs = sh.to_dict("records")
            pb_list = [eval_row(r, h, False) for r in recs]
            pa_list = [eval_row(r, h, True) for r in recs]
            hb_list = [check_horizon_hit(p, r["actual"], h) for p, r in zip(pb_list, recs)]
            ha_list = [check_horizon_hit(p, r["actual"], h) for p, r in zip(pa_list, recs)]
            hr_b = float(np.mean(hb_list) * 100.0)
            hr_a = float(np.mean(ha_list) * 100.0)
            hrs_bef.append(hr_b)
            hrs_aft.append(hr_a)

            eb, ea = 0, 0
            for sym, g in sh.groupby("symbol"):
                grecs = g.to_dict("records")
                g_pb = [eval_row(r, h, False) for r in grecs]
                g_pa = [eval_row(r, h, True) for r in grecs]
                g_hb = float(np.mean([check_horizon_hit(p, r["actual"], h) for p, r in zip(g_pb, grecs)]) * 100.0)
                g_ha = float(np.mean([check_horizon_hit(p, r["actual"], h) for p, r in zip(g_pa, grecs)]) * 100.0)
                up_r, dn_r = constant_call_hit_rates(g["actual"].to_numpy(), HORIZON_DAYS[h])
                mx = max(up_r, dn_r)
                if g_hb > mx:
                    eb += 1
                if g_ha > mx:
                    ea += 1
            edge_bef_tot += eb
            edge_aft_tot += ea
            print(
                f"  {h:<4} | Before: {hr_b:5.2f}% ({eb}/6 edge) | After (3F+News): {hr_a:5.2f}% ({ea}/6 edge) | "
                f"Delta: {hr_a - hr_b:+5.2f} pp"
            )
        print(
            f"  MEAN | Before: {np.mean(hrs_bef):5.2f}% ({edge_bef_tot}/36 edge) | "
            f"After (3F+News): {np.mean(hrs_aft):5.2f}% ({edge_aft_tot}/36 edge) | "
            f"Delta: {np.mean(hrs_aft) - np.mean(hrs_bef):+5.2f} pp"
        )


if __name__ == "__main__":
    main()
