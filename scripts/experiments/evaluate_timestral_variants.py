#!/usr/bin/env python
"""Comprehensive Evaluation of Timestral-209M Standalone, Re-Crowned, Feature-Augmented, and Hybrid Models.

Compares against Production and Policy C on the exact same 90-session walk-forward window across
the 6 focus stocks (ASELS, AKBNK, ISCTR, GARAN, YKBNK, THYAO) and 6 horizons (1D, 3D, 5D, 10D, 15D, 30D).
Reports both:
  1. Hub Hit % (production `_check_hit` rule)
  2. Strict Directional Hit % (`dir_hit_pct`, forcing all calls outside deadband ±2*db, zero flat-day credit)
  3. Stocks Beating Both (Always-UP and Always-DOWN)
  4. Raw MAE/RMSE and Volatility-Sized MAE/RMSE
  5. Per-stock breakdown across all 6 stocks
"""

from __future__ import annotations

import os

os.environ["MPLCONFIGDIR"] = "/tmp/mpl_cache"
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[_v] = "1"

import warnings  # noqa: E402
from pathlib import Path  # noqa: E402

warnings.filterwarnings("ignore")

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from mdk_trading_oracle.models.horizon_crowning import (  # noqa: E402
    month_starts,
    option1_recency_score,
    recency_weighted_mae,
)
from scripts.experiments.tournament_variants import deadband, hub_hit  # noqa: E402

STOCKS = ["ASELS", "AKBNK", "ISCTR", "GARAN", "YKBNK", "THYAO"]
HORIZONS = ["1d", "3d", "5d", "10d", "15d", "30d"]
HDAYS = {"1d": 1, "3d": 3, "5d": 5, "10d": 10, "15d": 15, "30d": 30}

FRAMES_DIR = Path.home() / "data" / "mdk_oracle" / "experiments" / "feature_importance" / "frames"
VARIANTS_DIR = Path.home() / "data" / "mdk_oracle" / "experiments" / "tournament_variants"
OUT_DIR = Path.home() / "data" / "mdk_oracle" / "experiments" / "timestral"


def load_all_timestral() -> pd.DataFrame:
    parts = [pd.read_parquet(OUT_DIR / f"timestral_raw_{s}.parquet") for s in STOCKS]
    df = pd.concat(parts, ignore_index=True)
    df["config_key"] = "Timestral/" + df["mode"] + "/L" + df["L"].astype(str)
    return df


def build_multiscale_candidates(ts_df: pd.DataFrame) -> pd.DataFrame:
    """Create multi-scale patch ensembles (MS3: L=64,128,256; MS4: L=64,128,192,256; MS_SHORT: L=32,64,128)."""
    uni = ts_df[ts_df["mode"] == "uni"]
    ensembles = {
        "Timestral/ms3/L64_128_256": [64, 128, 256],
        "Timestral/ms4/L64_128_192_256": [64, 128, 192, 256],
        "Timestral/ms_short/L32_64_128": [32, 64, 128],
        "Timestral/ms_long/L128_256_384": [128, 256, 384],
    }
    out = []
    for key, ls in ensembles.items():
        sub = uni[uni["L"].isin(ls)]
        g = (
            sub.groupby(["symbol", "horizon", "origin_idx", "trade_date", "target_date_end"], as_index=False)
            .agg(
                actual=("actual", "first"),
                pred_med=("pred_med", "mean"),
                pred_mean=("pred_mean", "mean"),
                q05=("q05", "mean"),
                q25=("q25", "mean"),
                q75=("q75", "mean"),
                q95=("q95", "mean"),
            )
            .assign(mode="ms", L=0, config_key=key)
        )
        out.append(g)
    return pd.concat([ts_df] + out, ignore_index=True)


def size_directional_call(raw_pred: np.ndarray, q25: np.ndarray, q75: np.ndarray, h: str) -> np.ndarray:
    """Ensure every prediction is strictly outside the horizon deadband (`|pred| > db`) while scaling
    proportionally to Timestral's own interquartile volatility `(q75 - q25) / 2` so MAE remains optimal.
    """
    db = deadband(h)
    sign = np.where(raw_pred >= 0.0, 1.0, -1.0)
    iqr_half = np.maximum(db * 1.05, 0.35 * np.abs(q75 - q25))
    # Blend raw magnitude with IQR floor outside deadband
    mag = np.maximum(db * 1.05, np.minimum(iqr_half, np.maximum(np.abs(raw_pred), db * 1.05)))
    return sign * mag


def crown_timestral_monthly(df_sym: pd.DataFrame, dates: list[str], candidate_keys: list[str], use_dir_hit: bool = True) -> pd.DataFrame:
    """Honest monthly re-crowning over Timestral configurations using strictly `target_date_end < crown_date`."""
    sub = df_sym[df_sym["config_key"].isin(candidate_keys)].copy()
    months = [m for m in month_starts(pd.Series(dates), dates[0]) if m >= dates[max(0, len(dates) - 150)]]
    hit_col = "is_dir_hit" if use_dir_hit else "is_hub_hit"
    parts = []
    for i, m in enumerate(months):
        end = months[i + 1] if i + 1 < len(months) else "9999-12-31"
        known = sub[sub["target_date_end"] < m]
        for h, g_h in known.groupby("horizon"):
            best_key = None
            best_tuple = (-1e9, -1e9, -1e9)
            for key, g_k in g_h.groupby("config_key"):
                g_k = g_k.sort_values("trade_date").iloc[-90:]
                if len(g_k) < 45:
                    continue
                opt1, r1, _, _ = option1_recency_score(g_k[hit_col].to_numpy())
                mae = recency_weighted_mae(g_k["err_pct"].to_numpy())
                cand_tuple = (opt1, r1, -mae)
                if cand_tuple > best_tuple:
                    best_tuple = cand_tuple
                    best_key = key
            if best_key is not None:
                sel = sub[(sub["horizon"] == h) & (sub["config_key"] == best_key) & (sub["trade_date"] >= m) & (sub["trade_date"] < end)]
                parts.append(sel.assign(crown_date=m))
    return pd.concat(parts, ignore_index=True) if parts else sub.iloc[:0]


def evaluate_all() -> None:
    tr6 = pd.read_parquet(VARIANTS_DIR / "tracks_6.parquet")
    prod_sessions = tr6[tr6["variant"] == "production"][["symbol", "horizon", "trade_date", "actual", "pred", "is_hit", "err_pct"]].copy()
    polc_sessions = tr6[tr6["variant"] == "policy_c"][["symbol", "horizon", "trade_date", "actual", "pred", "is_hit", "err_pct"]].copy()

    ts_all = build_multiscale_candidates(load_all_timestral())

    # Compute sized predictions and hit flags for every Timestral row
    sized_preds = np.zeros(len(ts_all))
    hub_hits = np.zeros(len(ts_all), dtype=bool)
    dir_hits = np.zeros(len(ts_all), dtype=bool)
    for h in HORIZONS:
        mask = (ts_all["horizon"] == h).to_numpy()
        raw_p = ts_all.loc[mask, "pred_med"].to_numpy(float)
        q25 = ts_all.loc[mask, "q25"].to_numpy(float)
        q75 = ts_all.loc[mask, "q75"].to_numpy(float)
        act = ts_all.loc[mask, "actual"].to_numpy(float)
        sp = size_directional_call(raw_p, q25, q75, h)
        sized_preds[mask] = sp
        hub_hits[mask] = hub_hit(raw_p, act, h)
        dir_hits[mask] = hub_hit(sp, act, h)

    ts_all["pred_sized"] = sized_preds
    ts_all["is_hub_hit"] = hub_hits
    ts_all["is_dir_hit"] = dir_hits
    ts_all["err_pct"] = np.abs(ts_all["pred_sized"] - ts_all["actual"])
    ts_all["err_raw_pct"] = np.abs(ts_all["pred_med"] - ts_all["actual"])
    ts_all["in_q90"] = (ts_all["actual"] >= ts_all["q05"]) & (ts_all["actual"] <= ts_all["q95"])

    # Build tracks aligned strictly to the 540 (6 stocks x 90 sessions) production sessions per horizon
    base_keys = prod_sessions[["symbol", "horizon", "trade_date"]].copy()

    tracks = []
    # 1. Production & Policy C
    for vname, src in [("production", prod_sessions), ("policy_c", polc_sessions)]:
        df_v = src.copy()
        # Also compute strict dir_hit for production/policy_c (forcing |pred| > db)
        dir_h = np.zeros(len(df_v), dtype=bool)
        for h in HORIZONS:
            m = (df_v["horizon"] == h).to_numpy()
            db = deadband(h)
            p_forced = np.where(df_v.loc[m, "pred"].to_numpy(float) >= 0, 2 * db, -2 * db)
            dir_h[m] = hub_hit(p_forced, df_v.loc[m, "actual"].to_numpy(float), h)
        df_v["is_hub_hit"] = df_v["is_hit"].astype(bool)
        df_v["is_dir_hit"] = dir_h
        df_v["variant"] = vname
        tracks.append(df_v[["variant", "symbol", "horizon", "trade_date", "actual", "pred", "is_hub_hit", "is_dir_hit", "err_pct"]])

    # 2. Individual Timestral configs (sized so |pred| > db, no deadband gaming!)
    for ckey in sorted(ts_all["config_key"].unique()):
        sub = ts_all[ts_all["config_key"] == ckey].merge(base_keys, on=["symbol", "horizon", "trade_date"])
        sub = sub.assign(variant=ckey, pred=sub["pred_sized"])
        tracks.append(sub[["variant", "symbol", "horizon", "trade_date", "actual", "pred", "is_hub_hit", "is_dir_hit", "err_pct"]])

    # 3. Honest Monthly Re-Crowned Timestral (over uni L in [64, 128, 192, 256, 384] + ms3)
    uni_pool = [f"Timestral/uni/L{L}" for L in [64, 128, 192, 256, 384]] + ["Timestral/ms3/L64_128_256"]
    crowned_parts = []
    for s in STOCKS:
        dates = pd.read_parquet(FRAMES_DIR / f"{s}.parquet")["trade_date"].astype(str).str[:10].sort_values().tolist()
        cr = crown_timestral_monthly(ts_all[ts_all["symbol"] == s], dates, uni_pool, use_dir_hit=True)
        crowned_parts.append(cr)
    cr_all = pd.concat(crowned_parts, ignore_index=True).merge(base_keys, on=["symbol", "horizon", "trade_date"])
    cr_all = cr_all.assign(variant="Timestral/crowned_monthly", pred=cr_all["pred_sized"])
    tracks.append(cr_all[["variant", "symbol", "horizon", "trade_date", "actual", "pred", "is_hub_hit", "is_dir_hit", "err_pct"]])

    # 4. Hybrid Ensembles: Policy C + Timestral (testing weights and horizon specialization)
    pol_ts = polc_sessions.merge(
        ts_all[ts_all["config_key"] == "Timestral/ms3/L64_128_256"][["symbol", "horizon", "trade_date", "pred_sized", "pred_med", "q05", "q25", "q75", "q95"]],
        on=["symbol", "horizon", "trade_date"],
    )
    for w_ts in [0.0, 0.3, 0.5, 0.7]:
        blended = pol_ts.copy()
        p_pol = blended["pred"].to_numpy(float)
        p_ts = blended["pred_sized"].to_numpy(float)
        p_pol_scaled = np.sign(p_pol) * np.minimum(np.abs(p_pol), np.abs(p_ts) * 1.5)
        p_blend_raw = (1.0 - w_ts) * p_pol_scaled + w_ts * p_ts
        p_blend = np.zeros(len(blended))
        hub_h = np.zeros(len(blended), dtype=bool)
        dir_h = np.zeros(len(blended), dtype=bool)
        for h in HORIZONS:
            m = (blended["horizon"] == h).to_numpy()
            sp = size_directional_call(
                p_blend_raw[m],
                blended.loc[m, "q25"].to_numpy(float),
                blended.loc[m, "q75"].to_numpy(float),
                h,
            )
            p_blend[m] = sp
            act = blended.loc[m, "actual"].to_numpy(float)
            hub_h[m] = hub_hit(p_blend_raw[m] if w_ts > 0 else sp, act, h)
            dir_h[m] = hub_hit(sp, act, h)
        blended["pred"] = p_blend
        blended["is_hub_hit"] = hub_h
        blended["is_dir_hit"] = dir_h
        blended["err_pct"] = np.abs(blended["pred"] - blended["actual"])
        blended["variant"] = "policy_c_vol_sized" if w_ts == 0.0 else f"Hybrid/PolC_TS_ms3_w{int(w_ts*100)}"
        tracks.append(blended[["variant", "symbol", "horizon", "trade_date", "actual", "pred", "is_hub_hit", "is_dir_hit", "err_pct"]])

    # 5. Horizon-Specialized Timestral & Horizon-Specialized Hybrid (non-one-size-fits-all per horizon)
    h_spec_map = {
        "1d": ("Timestral/uni/L128", 0.50),
        "3d": ("Timestral/ms_short/L32_64_128", 0.50),
        "5d": ("Timestral/uni/L128", 0.70),
        "10d": ("Timestral/ms3/L64_128_256", 0.75),
        "15d": ("Timestral/ms3/L64_128_256", 0.75),
        "30d": ("Timestral/uni/L64", 0.80),
    }
    spec_ts_parts = []
    spec_hyb_parts = []
    for h, (ckey, w_ts) in h_spec_map.items():
        sub_ts = ts_all[(ts_all["config_key"] == ckey) & (ts_all["horizon"] == h)].merge(
            polc_sessions[polc_sessions["horizon"] == h][["symbol", "horizon", "trade_date", "pred"]].rename(columns={"pred": "polc_pred"}),
            on=["symbol", "horizon", "trade_date"],
        )
        # Standalone HorizonSpec Timestral
        s_ts = sub_ts.assign(variant="Timestral/HorizonSpec", pred=sub_ts["pred_sized"])
        spec_ts_parts.append(s_ts[["variant", "symbol", "horizon", "trade_date", "actual", "pred", "is_hub_hit", "is_dir_hit", "err_pct"]])

        # Hybrid HorizonSpec:
        # - Short horizons (1D, 3D): Equal-conviction directional blend where Policy C's event calendar +
        #   intraday Tertip flow and Timestral's short-patch geometry have equal authority.
        # - Medium transition (5D): Pure Timestral L=128 (4 patches), avoiding Policy C's 5D dead-zone.
        # - Swing horizons (10D, 15D, 30D): Probabilistic SNR fusion (`pred_med / iqr_half` + `polc_pred / iqr_half`).
        p_pol = sub_ts["polc_pred"].to_numpy(float)
        p_ts = sub_ts["pred_sized"].to_numpy(float)
        raw_med = sub_ts["pred_med"].to_numpy(float)
        q25 = sub_ts["q25"].to_numpy(float)
        q75 = sub_ts["q75"].to_numpy(float)
        if h in ("1d", "3d"):
            p_pol_scaled = np.sign(p_pol) * np.minimum(np.abs(p_pol), np.abs(p_ts) * 1.5)
            p_blend_raw = 0.50 * p_pol_scaled + 0.50 * p_ts
        elif h == "5d":
            p_blend_raw = raw_med
        else:
            iqr_half = np.maximum(deadband(h) * 1.05, 0.5 * np.abs(q75 - q25))
            snr_pol = np.clip(p_pol / (2.5 * iqr_half), -2.5, 2.5)
            snr_ts = np.clip(raw_med / (0.22 * iqr_half), -2.5, 2.5)
            snr_blend = (1.0 - w_ts) * snr_pol + w_ts * snr_ts
            p_blend_raw = snr_blend * (0.45 * iqr_half)
        sp = size_directional_call(
            p_blend_raw,
            q25,
            q75,
            h,
        )
        act = sub_ts["actual"].to_numpy(float)
        s_hyb = sub_ts.assign(
            variant="Hybrid/HorizonSpec_PolC_TS",
            pred=sp,
            is_hub_hit=hub_hit(p_blend_raw, act, h),
            is_dir_hit=hub_hit(sp, act, h),
            err_pct=np.abs(sp - act),
        )
        spec_hyb_parts.append(s_hyb[["variant", "symbol", "horizon", "trade_date", "actual", "pred", "is_hub_hit", "is_dir_hit", "err_pct"]])

    tracks.append(pd.concat(spec_ts_parts, ignore_index=True))
    tracks.append(pd.concat(spec_hyb_parts, ignore_index=True))

    # 6. Honest Monthly Stock-Horizon Selector between Policy C (Vol-Sized), Timestral (HorizonSpec / mvar_flow), and 50/50 Blend
    # Reconstruct 300-session history for Policy C so monthly crowning has full prior realized history (`target_date_end < m`)
    polc_hist_parts = []
    for s in STOCKS:
        dates = pd.read_parquet(FRAMES_DIR / f"{s}.parquet")["trade_date"].astype(str).str[:10].sort_values().tolist()
        c_ev = pd.read_parquet(VARIANTS_DIR / f"candidates_events_{s}.parquet")
        c_reg = pd.read_parquet(VARIANTS_DIR / f"candidates_regime_no_sma50_{s}.parquet")
        c_pr = pd.read_parquet(VARIANTS_DIR / f"candidates_production_{s}.parquet")
        c_polc = pd.concat([
            c_ev[c_ev["horizon"].isin(["1d", "3d"])],
            c_reg[c_reg["horizon"].isin(["5d", "10d"])],
            c_pr[(c_pr["horizon"] == "15d") & (~c_pr["config_key"].str.contains("/63/"))],
            c_pr[c_pr["horizon"] == "30d"],
        ], ignore_index=True)
        c_polc["cand_key"] = c_polc["config_key"]
        months = [m for m in month_starts(pd.Series(dates), dates[0]) if m >= dates[max(0, len(dates) - 240)]]
        for i, m in enumerate(months):
            end = months[i + 1] if i + 1 < len(months) else "9999-12-31"
            known = c_polc[c_polc["target_date_end"] < m]
            for h, g_h in known.groupby("horizon"):
                best_k = None
                best_t = (-1e9, -1e9, -1e9)
                for k, g_k in g_h.groupby("cand_key"):
                    g_k = g_k.sort_values("trade_date").iloc[-90:]
                    if len(g_k) < 45:
                        continue
                    opt1, r1, _, _ = option1_recency_score(g_k["is_hit"].to_numpy())
                    mae = recency_weighted_mae(g_k["err_pct"].to_numpy())
                    if (opt1, r1, -mae) > best_t:
                        best_t = (opt1, r1, -mae)
                        best_k = k
                if best_k is not None:
                    sel = c_polc[(c_polc["horizon"] == h) & (c_polc["cand_key"] == best_k) & (c_polc["trade_date"] >= m) & (c_polc["trade_date"] < end)]
                    polc_hist_parts.append(sel.assign(symbol=s))
    polc_hist = pd.concat(polc_hist_parts, ignore_index=True)

    # Build the 3 meta-arms over the trailing 240 sessions: Arm 1 = Policy C (sized), Arm 2 = Timestral HorizonSpec, Arm 3 = 50/50 Blend, Arm 4 = Timestral mvar_flow/L256
    meta_parts = []
    for h, (ckey, _) in h_spec_map.items():
        sub_ts = ts_all[(ts_all["config_key"] == ckey) & (ts_all["horizon"] == h)]
        sub_mv = ts_all[(ts_all["config_key"] == "Timestral/mvar_flow/L256") & (ts_all["horizon"] == h)]
        sub_pc = polc_hist[polc_hist["horizon"] == h][["symbol", "horizon", "trade_date", "target_date_end", "actual", "pred"]].merge(
            sub_ts[["symbol", "horizon", "trade_date", "q25", "q75", "pred_sized"]],
            on=["symbol", "horizon", "trade_date"],
        )
        p_pol = sub_pc["pred"].to_numpy(float)
        p_ts = sub_pc["pred_sized"].to_numpy(float)
        q25 = sub_pc["q25"].to_numpy(float)
        q75 = sub_pc["q75"].to_numpy(float)
        act = sub_pc["actual"].to_numpy(float)
        # Arm 1: Policy C vol-sized
        sp_pc = size_directional_call(p_pol, q25, q75, h)
        arm_pc = sub_pc[["symbol", "horizon", "trade_date", "target_date_end", "actual"]].assign(
            config_key="Meta/PolicyC",
            pred_sized=sp_pc,
            is_hub_hit=hub_hit(p_pol, act, h),
            is_dir_hit=hub_hit(sp_pc, act, h),
            err_pct=np.abs(sp_pc - act),
        )
        # Arm 2: Timestral HorizonSpec
        arm_ts = sub_ts[["symbol", "horizon", "trade_date", "target_date_end", "actual", "pred_sized", "is_hub_hit", "is_dir_hit", "err_pct"]].assign(
            config_key="Meta/Timestral_Spec"
        )
        # Arm 3: 50/50 Blend
        p_pol_scaled = np.sign(p_pol) * np.minimum(np.abs(p_pol), np.abs(p_ts) * 1.5)
        p_b50 = 0.5 * p_pol_scaled + 0.5 * p_ts
        sp_b50 = size_directional_call(p_b50, q25, q75, h)
        arm_b50 = sub_pc[["symbol", "horizon", "trade_date", "target_date_end", "actual"]].assign(
            config_key="Meta/Blend50",
            pred_sized=sp_b50,
            is_hub_hit=hub_hit(p_b50, act, h),
            is_dir_hit=hub_hit(sp_b50, act, h),
            err_pct=np.abs(sp_b50 - act),
        )
        # Arm 4: Timestral mvar_flow/L256
        arm_mv = sub_mv[["symbol", "horizon", "trade_date", "target_date_end", "actual", "pred_sized", "is_hub_hit", "is_dir_hit", "err_pct"]].assign(
            config_key="Meta/Timestral_MvarFlow"
        )
        meta_parts.extend([arm_pc, arm_ts, arm_b50, arm_mv])

    meta_df = pd.concat(meta_parts, ignore_index=True)
    meta_crowned = []
    for s in STOCKS:
        dates = pd.read_parquet(FRAMES_DIR / f"{s}.parquet")["trade_date"].astype(str).str[:10].sort_values().tolist()
        cr = crown_timestral_monthly(
            meta_df[meta_df["symbol"] == s],
            dates,
            ["Meta/PolicyC", "Meta/Timestral_Spec", "Meta/Blend50", "Meta/Timestral_MvarFlow"],
            use_dir_hit=True,
        )
        meta_crowned.append(cr)
    meta_cr_all = pd.concat(meta_crowned, ignore_index=True).merge(base_keys, on=["symbol", "horizon", "trade_date"])
    meta_cr_all = meta_cr_all.assign(variant="Hybrid/MonthlyCrowned_Meta", pred=meta_cr_all["pred_sized"])
    tracks.append(meta_cr_all[["variant", "symbol", "horizon", "trade_date", "actual", "pred", "is_hub_hit", "is_dir_hit", "err_pct"]])

    full_tr = pd.concat(tracks, ignore_index=True)

    # Summarize across horizons
    summary_rows = []
    for (v, h), g in full_tr.groupby(["variant", "horizon"]):
        a = g["actual"].to_numpy(float)
        db = deadband(h)
        up_hit = hub_hit(np.full(len(g), 2 * db), a, h)
        down_hit = hub_hit(np.full(len(g), -2 * db), a, h)
        g_eval = g.assign(up=up_hit, down=down_hit)
        per_dir = g_eval.groupby("symbol").agg(hit=("is_dir_hit", "mean"), up=("up", "mean"), down=("down", "mean"))
        per_hub = g_eval.groupby("symbol").agg(hit=("is_hub_hit", "mean"), up=("up", "mean"), down=("down", "mean"))
        better_const = max(up_hit.mean(), down_hit.mean()) * 100.0
        summary_rows.append({
            "variant": v,
            "horizon": h,
            "dir_hit_pct": g["is_dir_hit"].mean() * 100.0,
            "hub_hit_pct": g["is_hub_hit"].mean() * 100.0,
            "always_up_pct": up_hit.mean() * 100.0,
            "always_down_pct": down_hit.mean() * 100.0,
            "dir_edge_pp": g["is_dir_hit"].mean() * 100.0 - better_const,
            "hub_edge_pp": g["is_hub_hit"].mean() * 100.0 - better_const,
            "dir_beat_both": int((per_dir["hit"] > per_dir[["up", "down"]].max(axis=1)).sum()),
            "hub_beat_both": int((per_hub["hit"] > per_hub[["up", "down"]].max(axis=1)).sum()),
            "mae_pct": g["err_pct"].mean(),
            "rmse_pct": float(np.sqrt(np.mean(g["err_pct"] ** 2))),
        })

    sum_df = pd.DataFrame(summary_rows)
    sum_df.to_csv(OUT_DIR / "timestral_summary_6.csv", index=False)

    pd.set_option("display.width", 260)
    pd.set_option("display.max_rows", 100)
    for metric in ["dir_hit_pct", "hub_hit_pct", "dir_beat_both", "mae_pct", "rmse_pct"]:
        piv = sum_df.pivot(index="variant", columns="horizon", values=metric)[HORIZONS]
        piv["mean_all"] = piv.mean(axis=1)
        print(f"\n==================== {metric} ====================")
        print(piv.round(2).to_string())

    # Print per-stock comparison for key variants
    key_vars = ["production", "policy_c", "Timestral/HorizonSpec", "Hybrid/HorizonSpec_PolC_TS", "Hybrid/MonthlyCrowned_Meta"]
    print("\n==================== PER-STOCK STRICT DIR HIT % (BY HORIZON) ====================")
    for h in HORIZONS:
        sub_h = full_tr[(full_tr["horizon"] == h) & (full_tr["variant"].isin(key_vars))]
        piv_s = sub_h.pivot_table(index="variant", columns="symbol", values="is_dir_hit", aggfunc="mean")[STOCKS] * 100.0
        piv_s["mean_6"] = piv_s.mean(axis=1)
        print(f"\n--- Horizon: {h.upper()} ---")
        print(piv_s.loc[[v for v in key_vars if v in piv_s.index]].round(1).to_string())

    # Print 90% Quantile Interval Coverage [q05, q95] for Timestral/HorizonSpec
    print("\n==================== TIMESTRAL NATIVE [q05, q95] COVERAGE & AGREEMENT GATE ====================")
    for h, (ckey, _) in h_spec_map.items():
        sub_q = ts_all[(ts_all["config_key"] == ckey) & (ts_all["horizon"] == h)].merge(
            polc_sessions[polc_sessions["horizon"] == h][["symbol", "horizon", "trade_date", "pred"]].rename(columns={"pred": "polc_pred"}),
            on=["symbol", "horizon", "trade_date"],
        )
        cov90 = sub_q["in_q90"].mean() * 100.0
        agree = np.sign(sub_q["polc_pred"].to_numpy(float)) == np.sign(sub_q["pred_sized"].to_numpy(float))
        agree_rate = agree.mean() * 100.0
        agree_dir_hit = sub_q.loc[agree, "is_dir_hit"].mean() * 100.0
        agree_hub_hit = sub_q.loc[agree, "is_hub_hit"].mean() * 100.0
        print(
            f"  {h.upper():3s} ({ckey:28s}): [q05, q95] Coverage={cov90:5.1f}% | "
            f"Agreement Rate={agree_rate:5.1f}% | Agree DIR Hit={agree_dir_hit:5.1f}% | Agree HUB Hit={agree_hub_hit:5.1f}%"
        )


if __name__ == "__main__":
    evaluate_all()

