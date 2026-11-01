#!/usr/bin/env python
"""Dedicated Out-of-the-Box Experiment: Two-Stage Decoupled Direction-First + Amplitude-Second Architecture.

Tests whether decoupling Stage 1 (Pure Directional Sign Specialist) from Stage 2 (Volatility/Excursion Amplitude)
improves strict full-amplitude directional hit rates across 1D, 3D, 5D, 10D, 15D, 30D on the 6 focus stocks.

Ideas Tested (Zero-Leakage Walk-Forward on 540 Sessions/Horizon):
  0. Current Baseline (BIST30 Shock-Gated + Full Amplitude)
  1. Multi-Scale Patch Curvature & Cross-Horizon Trajectory Coherence:
     - Uses `pred_mean` on L=128/MS3 (which beats `pred_med` by +2.8pp on 10D and +1.5pp on 15D)
     - Uses Cross-Horizon Trajectory Slope (`ts_3d - ts_1d`, `ts_10d - ts_3d`, `ts_15d - ts_5d`, `ts_30d - ts_10d`)
       decoded simultaneously inside the same 32-step Timestral output patch!
  2. Sector Peer-Basket Consensus (BANK 4-peer consensus for AKBNK/GARAN/ISCTR/YKBNK; Industrials for ASELS/THYAO):
     - Institutional order flows on BIST move the banking index (`XBANK`) as a tightly coupled block.
     - Averaging the peer group's Policy C + Timestral signals cancels single-stock idiosyncratic noise.
  3. Zero-Leakage Online Expert Hedge / Expanding-Window Specialist Weighting:
     - Tracks each specialist's realized directional accuracy strictly up to `target_date_end < origin_date`
       (stock Timestral, BIST30 Timestral, Peer Consensus, Policy C, Cross-Horizon Slope) and weights them by
       their realized edge on that specific `(symbol, horizon)`.
  4. Walk-Forward Logistic / Margin Direction Classifier (`Stage 1 Direction-Only Specialist`):
     - Trains a regularized L2 Logistic Classifier strictly on prior realized (`target_date_end < origin_date`)
       sessions to predict `sign(y) in {-1, +1}` directly from Timestral multi-scale patch outputs + institutional
       BofA (`MLB`), `BIG5`, `KAMU` order-flow features + BIST30 shock/regime states.
"""

from __future__ import annotations

import os

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[_v] = "1"

import warnings  # noqa: E402
from pathlib import Path  # noqa: E402

warnings.filterwarnings("ignore")

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import torch  # noqa: E402
from timestral import TimestralForecaster  # noqa: E402

from mdk_trading_oracle.models.timestral_forecaster import sanitize_bist_log_prices  # noqa: E402
from scripts.experiments.evaluate_timestral_variants import (  # noqa: E402
    FRAMES_DIR,
    HORIZONS,
    STOCKS,
    VARIANTS_DIR,
    build_multiscale_candidates,
    load_all_timestral,
)

HDAYS = {"1d": 1, "3d": 3, "5d": 5, "10d": 10, "15d": 15, "30d": 30}
BANKS = {"AKBNK", "GARAN", "ISCTR", "YKBNK"}
INDUST = {"ASELS", "THYAO"}


def build_experiment_dataset() -> pd.DataFrame:
    # 1. Compute BIST30 Timestral across L in [32, 64, 128, 256]
    df_ak = pd.read_parquet(FRAMES_DIR / "AKBNK.parquet").sort_values("trade_date").reset_index(drop=True)
    df_ak["trade_date"] = df_ak["trade_date"].astype(str).str[:10]
    bist_ret = df_ak["bist30_return_pct"].fillna(0.0).to_numpy(float)
    bist_px = 100.0 * np.cumprod(1.0 + np.clip(bist_ret / 100.0, -0.0995, 0.0995))
    bist_log_px = sanitize_bist_log_prices(bist_px)

    model_dir = Path.home() / "data" / "mdk_oracle" / "models" / "Timestral-209M"
    forecaster = TimestralForecaster.from_pretrained(str(model_dir), device="cpu")
    origins = list(range(len(df_ak) - 300, len(df_ak)))
    h_steps = {"1d": 0, "3d": 2, "5d": 4, "10d": 9, "15d": 14, "30d": 29}

    bist_preds: dict[int, dict[tuple[str, str], tuple[float, float]]] = {L: {} for L in [32, 64, 128, 256]}
    for L in [32, 64, 128, 256]:
        ctxs = [bist_log_px[idx - L + 1 : idx + 1] - bist_log_px[idx - L + 1] for idx in origins]
        batch = torch.tensor(np.stack(ctxs, axis=0)[:, None, :], dtype=torch.float32)
        with torch.no_grad():
            res = forecaster.forecast(batch, prediction_length=30, clip_nonnegative=False)
        q = res.quantiles.numpy()[:, 0, :, :]
        last_val = np.array([bist_log_px[idx] - bist_log_px[idx - L + 1] for idx in origins])
        for h, s_idx in h_steps.items():
            med = (np.exp((q[:, s_idx, 10] - last_val) / 100.0) - 1.0) * 100.0
            mean_v = (np.exp((q[:, s_idx, :].mean(axis=-1) - last_val) / 100.0) - 1.0) * 100.0
            for i, idx in enumerate(origins):
                tdate = df_ak.loc[idx, "trade_date"]
                bist_preds[L][(tdate, h)] = (float(med[i]), float(mean_v[i]))

    bist_rows = []
    for idx in origins:
        tdate = df_ak.loc[idx, "trade_date"]
        for h in HORIZONS:
            m32, mn32 = bist_preds[32][(tdate, h)]
            m64, mn64 = bist_preds[64][(tdate, h)]
            m128, mn128 = bist_preds[128][(tdate, h)]
            m256, mn256 = bist_preds[256][(tdate, h)]
            bist_rows.append({
                "trade_date": tdate,
                "horizon": h,
                "bist_l64": m64,
                "bist_l128": m128,
                "bist_mean128": mn128,
                "bist_ms_short": (m32 + m64 + m128) / 3.0,
                "bist_ms3": (m64 + m128 + m256) / 3.0,
                "bist_ms3_mean": (mn64 + mn128 + mn256) / 3.0,
            })
    df_bist = pd.DataFrame(bist_rows)

    # 2. Load all Timestral multi-scale candidates per stock
    ts_all = build_multiscale_candidates(load_all_timestral())
    piv_parts = []
    for ckey, col_prefix in [
        ("Timestral/uni/L32", "ts32"),
        ("Timestral/uni/L64", "ts64"),
        ("Timestral/uni/L128", "ts128"),
        ("Timestral/uni/L192", "ts192"),
        ("Timestral/uni/L256", "ts256"),
        ("Timestral/ms_short/L32_64_128", "tsshort"),
        ("Timestral/ms3/L64_128_256", "tsms3"),
        ("Timestral/mvar_flow/L128", "tsmv128"),
    ]:
        sub = ts_all[ts_all["config_key"] == ckey][
            ["symbol", "horizon", "trade_date", "target_date_end", "actual", "pred_med", "pred_mean", "q05", "q25", "q75", "q95"]
        ].copy()
        sub = sub.rename(columns={
            "pred_med": f"{col_prefix}_med",
            "pred_mean": f"{col_prefix}_mean",
            "q05": f"{col_prefix}_q05",
            "q25": f"{col_prefix}_q25",
            "q75": f"{col_prefix}_q75",
            "q95": f"{col_prefix}_q95",
        })
        piv_parts.append(sub)

    base = piv_parts[0]
    for p in piv_parts[1:]:
        drop_c = [c for c in ["target_date_end", "actual"] if c in p.columns]
        base = base.merge(p.drop(columns=drop_c), on=["symbol", "horizon", "trade_date"])

    # 3. Cross-horizon Timestral signals at the same (symbol, trade_date):
    # Inside a single forward pass of patch_size=32, Timestral outputs 1d, 3d, 5d, 10d, 15d, 30d simultaneously!
    xh = base.pivot(index=["symbol", "trade_date"], columns="horizon", values=["ts128_med", "tsms3_med", "ts64_med", "ts128_mean", "tsms3_mean"])
    xh.columns = [f"xh_{c[0]}_{c[1]}" for c in xh.columns]
    xh = xh.reset_index()
    base = base.merge(xh, on=["symbol", "trade_date"])

    # 4. Reconstruct 240-session Policy C point-in-time predictions & candidate ensemble mean
    polc_parts = []
    for s in STOCKS:
        c_ev = pd.read_parquet(VARIANTS_DIR / f"candidates_events_{s}.parquet")
        c_reg = pd.read_parquet(VARIANTS_DIR / f"candidates_regime_no_sma50_{s}.parquet")
        c_pr = pd.read_parquet(VARIANTS_DIR / f"candidates_production_{s}.parquet")
        pool = pd.concat([
            c_ev[c_ev["horizon"].isin(["1d", "3d"])],
            c_reg[c_reg["horizon"].isin(["5d", "10d"])],
            c_pr[(c_pr["horizon"] == "15d") & (~c_pr["config_key"].str.contains("/63/"))],
            c_pr[c_pr["horizon"] == "30d"],
        ], ignore_index=True)
        # Also compute the median & vote across all 75 (or 60) candidate models at each (horizon, trade_date)!
        ens = (
            pool.groupby(["horizon", "trade_date"], as_index=False)
            .agg(
                polc_ens_med=("pred", "median"),
                polc_ens_mean=("pred", "mean"),
                polc_ens_vote=("pred", lambda x: float(np.mean(np.sign(x)))),
            )
            .assign(symbol=s)
        )
        # Also get the 126/252 lookback candidate vote
        long_pool = pool[pool["config_key"].str.contains("/126/|/252/|/378/|/504/")]
        ens_long = (
            long_pool.groupby(["horizon", "trade_date"], as_index=False)
            .agg(
                polc_long_med=("pred", "median"),
                polc_long_vote=("pred", lambda x: float(np.mean(np.sign(x)))),
            )
            .assign(symbol=s)
        )
        polc_parts.append(ens.merge(ens_long, on=["symbol", "horizon", "trade_date"]))
    df_ens = pd.concat(polc_parts, ignore_index=True)

    # Also merge the crowned Policy C prediction on the 90-session evaluation window (and earlier months)
    tr6 = pd.read_parquet(VARIANTS_DIR / "tracks_6.parquet")
    polc_tr = tr6[tr6["variant"] == "policy_c"][["symbol", "horizon", "trade_date", "pred"]].rename(columns={"pred": "polc_crowned"})

    base = base.merge(df_ens, on=["symbol", "horizon", "trade_date"], how="left")
    base = base.merge(polc_tr, on=["symbol", "horizon", "trade_date"], how="left")
    base["polc_crowned"] = base["polc_crowned"].fillna(base["polc_ens_med"])

    # 5. Merge stock & BIST30 institutional/regime features
    f_parts = []
    for s in STOCKS:
        df = pd.read_parquet(FRAMES_DIR / f"{s}.parquet")
        df["symbol"] = s
        df["trade_date"] = df["trade_date"].astype(str).str[:10]
        f_parts.append(df[[
            "symbol", "trade_date", "bist30_return_pct", "feat_bist30_ret_today_pct",
            "feat_days_since_pos_shock", "feat_days_since_neg_shock", "feat_kamu_post_shock_defense",
            "mkt_breadth_above_sma20", "mkt_bist30_ret_5d", "mkt_bist30_ret_10d", "mkt_bist30_ret_20d",
            "mkt_bist30_vs_sma20", "mkt_bist30_vs_sma50", "mkt_bist30_drawdown_60d", "mkt_bist30_vol_20d",
            "feat_total_inst_net_share_today", "feat_ret_today_pct", "feat_close_ret_5d", "feat_close_ret_10d",
            "feat_close_ret_20d", "feat_price_vs_sma20_pct", "feat_price_vs_sma50_pct"
        ]])
    f_all = pd.concat(f_parts, ignore_index=True)
    base = base.merge(f_all, on=["symbol", "trade_date"]).merge(df_bist, on=["trade_date", "horizon"])

    # 6. Compute Sector Peer-Basket Consensus (at the exact same trade_date & horizon)
    base["peer_group"] = np.where(base["symbol"].isin(BANKS), "BANK", "INDUST")
    peer_agg = (
        base.groupby(["peer_group", "horizon", "trade_date"], as_index=False)
        .agg(
            peer_ts128_med=("ts128_med", "mean"),
            peer_tsms3_med=("tsms3_med", "mean"),
            peer_ts64_med=("ts64_med", "mean"),
            peer_polc_vote=("polc_ens_vote", "mean"),
            peer_polc_med=("polc_crowned", "mean"),
        )
    )
    # Also all-6 market basket consensus
    mkt_agg = (
        base.groupby(["horizon", "trade_date"], as_index=False)
        .agg(
            mkt6_ts128_med=("ts128_med", "mean"),
            mkt6_tsms3_med=("tsms3_med", "mean"),
            mkt6_ts64_med=("ts64_med", "mean"),
            mkt6_polc_vote=("polc_ens_vote", "mean"),
            mkt6_polc_med=("polc_crowned", "mean"),
        )
    )
    base = base.merge(peer_agg, on=["peer_group", "horizon", "trade_date"]).merge(mkt_agg, on=["horizon", "trade_date"])
    return base


if __name__ == "__main__":
    df = build_experiment_dataset()
    out_path = Path.home() / "data" / "mdk_oracle" / "experiments" / "timestral" / "direction_experiment_data.parquet"
    df.to_parquet(out_path, index=False)
    print(f"Saved {len(df)} rows to {out_path}")
