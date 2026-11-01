#!/usr/bin/env python
"""Zero-Leakage Multi-Horizon Experiment Arena for Mistral AI's Timestral-209M.

Evaluates `mistralai/Timestral-209M` across the 6 focus stocks (ASELS, AKBNK, ISCTR, GARAN, YKBNK, THYAO)
and all 6 horizons (1D, 3D, 5D, 10D, 15D, 30D) over the trailing 300 origins (90-session graded track
with 210 prior origins for honest monthly re-crowning and tabular feature training).
"""

from __future__ import annotations

import os

os.environ["MPLCONFIGDIR"] = "/tmp/mpl_cache"
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[_v] = "1"

import time  # noqa: E402
import warnings  # noqa: E402
from multiprocessing import Pool  # noqa: E402
from pathlib import Path  # noqa: E402

warnings.filterwarnings("ignore")

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import torch  # noqa: E402

STOCKS = ["ASELS", "AKBNK", "ISCTR", "GARAN", "YKBNK", "THYAO"]
HORIZONS = ["1d", "3d", "5d", "10d", "15d", "30d"]
HDAYS = {"1d": 1, "3d": 3, "5d": 5, "10d": 10, "15d": 15, "30d": 30}
CONTEXT_LENGTHS = [32, 64, 96, 128, 192, 256, 384, 512]

MODEL_DIR = Path.home() / "data" / "mdk_oracle" / "models" / "Timestral-209M"
FRAMES_DIR = Path.home() / "data" / "mdk_oracle" / "experiments" / "feature_importance" / "frames"
VARIANTS_DIR = Path.home() / "data" / "mdk_oracle" / "experiments" / "tournament_variants"
OUT_DIR = Path.home() / "data" / "mdk_oracle" / "experiments" / "timestral"
HISTORY = 300
DISPLAY = 90


def _compute_stock_timestral_forecasts(sym: str) -> str:
    """Compute and cache 300-origin Timestral-209M forecasts for a single stock across all context lengths."""
    torch.set_num_threads(2)
    out_file = OUT_DIR / f"timestral_raw_{sym}.parquet"
    if out_file.exists():
        return str(out_file)

    from timestral import TimestralForecaster

    from mdk_trading_oracle.data.silver.tertip_ml_forecaster import adjusted_price
    from mdk_trading_oracle.data.silver.tertip_multi_horizon_forecaster import compute_forward_terminal_return

    forecaster = TimestralForecaster.from_pretrained(MODEL_DIR, device="cpu")
    df = pd.read_parquet(FRAMES_DIR / f"{sym}.parquet").sort_values("trade_date").reset_index(drop=True)
    px = adjusted_price(df, "close")
    close_raw = df["close_price"].astype(float)
    log_px = (100.0 * np.log(px.to_numpy(float))).astype(np.float32)

    # Also build cumulative institutional flow share for multivariate test
    mlb_sh = (df["mlb_flow"] / df["total_turnover_tl"].replace(0, np.nan) * 100.0).fillna(0.0).cumsum().to_numpy(np.float32)
    inst_sh = df["feat_total_inst_net_share_today"].fillna(0.0).cumsum().to_numpy(np.float32)

    dates = [str(d)[:10] for d in df["trade_date"].tolist()]
    n_total = len(df)
    start_idx = max(max(CONTEXT_LENGTHS), n_total - HISTORY)
    origins = list(range(start_idx, n_total - 1))

    # Precompute ground-truth forward terminal returns for each horizon
    actual_by_h = {}
    for h, n_days in HDAYS.items():
        _, r_term = compute_forward_terminal_return(close_raw, n_days, px)
        actual_by_h[h] = r_term.to_numpy(float)

    rows = []
    # 1. Univariate log_px across all context lengths L
    for L in CONTEXT_LENGTHS:
        ctx = np.stack([log_px[t - L + 1 : t + 1] for t in origins], axis=0)[:, None, :]
        fc = forecaster.forecast(ctx, prediction_length=30, clip_nonnegative=False)
        q = fc.quantiles.numpy()[:, 0, :, :]  # (B, 30, 21)
        last_val = ctx[:, 0, -1]  # (B,)

        for h, n_days in HDAYS.items():
            step_idx = n_days - 1
            q_step = q[:, step_idx, :]  # (B, 21)
            # Convert log-return diff back to exact percentage return
            med_log_diff = q_step[:, 10] - last_val
            mean_log_diff = q_step.mean(axis=-1) - last_val
            q05_log_diff = q_step[:, 1] - last_val
            q25_log_diff = q_step[:, 5] - last_val
            q75_log_diff = q_step[:, 15] - last_val
            q95_log_diff = q_step[:, 19] - last_val

            med_pct = (np.exp(med_log_diff / 100.0) - 1.0) * 100.0
            mean_pct = (np.exp(mean_log_diff / 100.0) - 1.0) * 100.0
            q05_pct = (np.exp(q05_log_diff / 100.0) - 1.0) * 100.0
            q25_pct = (np.exp(q25_log_diff / 100.0) - 1.0) * 100.0
            q75_pct = (np.exp(q75_log_diff / 100.0) - 1.0) * 100.0
            q95_pct = (np.exp(q95_log_diff / 100.0) - 1.0) * 100.0

            act_arr = actual_by_h[h]
            for idx_pos, t in enumerate(origins):
                if t + n_days >= n_total:
                    continue
                act = float(act_arr[t])
                if not np.isfinite(act):
                    continue
                rows.append({
                    "symbol": sym,
                    "horizon": h,
                    "mode": "uni",
                    "L": L,
                    "origin_idx": t,
                    "trade_date": dates[t],
                    "target_date_end": dates[t + n_days],
                    "actual": act,
                    "pred_med": float(med_pct[idx_pos]),
                    "pred_mean": float(mean_pct[idx_pos]),
                    "q05": float(q05_pct[idx_pos]),
                    "q25": float(q25_pct[idx_pos]),
                    "q75": float(q75_pct[idx_pos]),
                    "q95": float(q95_pct[idx_pos]),
                })

    # 2. Multivariate [log_px, cum_mlb_share, cum_inst_share] for L in [64, 128, 256]
    for L in (64, 128, 256):
        mctx = np.stack(
            [
                np.stack([log_px[t - L + 1 : t + 1], mlb_sh[t - L + 1 : t + 1], inst_sh[t - L + 1 : t + 1]], axis=0)
                for t in origins
            ],
            axis=0,
        )
        fc = forecaster.forecast(mctx, prediction_length=30, clip_nonnegative=False)
        q = fc.quantiles.numpy()[:, 0, :, :]
        last_val = mctx[:, 0, -1]
        for h, n_days in HDAYS.items():
            step_idx = n_days - 1
            q_step = q[:, step_idx, :]
            med_pct = (np.exp((q_step[:, 10] - last_val) / 100.0) - 1.0) * 100.0
            mean_pct = (np.exp((q_step.mean(axis=-1) - last_val) / 100.0) - 1.0) * 100.0
            q05_pct = (np.exp((q_step[:, 1] - last_val) / 100.0) - 1.0) * 100.0
            q25_pct = (np.exp((q_step[:, 5] - last_val) / 100.0) - 1.0) * 100.0
            q75_pct = (np.exp((q_step[:, 15] - last_val) / 100.0) - 1.0) * 100.0
            q95_pct = (np.exp((q_step[:, 19] - last_val) / 100.0) - 1.0) * 100.0
            act_arr = actual_by_h[h]
            for idx_pos, t in enumerate(origins):
                if t + n_days >= n_total:
                    continue
                act = float(act_arr[t])
                if not np.isfinite(act):
                    continue
                rows.append({
                    "symbol": sym,
                    "horizon": h,
                    "mode": "mvar_flow",
                    "L": L,
                    "origin_idx": t,
                    "trade_date": dates[t],
                    "target_date_end": dates[t + n_days],
                    "actual": act,
                    "pred_med": float(med_pct[idx_pos]),
                    "pred_mean": float(mean_pct[idx_pos]),
                    "q05": float(q05_pct[idx_pos]),
                    "q25": float(q25_pct[idx_pos]),
                    "q75": float(q75_pct[idx_pos]),
                    "q95": float(q95_pct[idx_pos]),
                })

    pd.DataFrame(rows).to_parquet(out_file)
    return str(out_file)


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    with Pool(6) as pool:
        for i, p in enumerate(pool.imap_unordered(_compute_stock_timestral_forecasts, STOCKS), 1):
            print(f"Completed {i}/6: {p} ({time.time() - t0:.1f}s)", flush=True)


if __name__ == "__main__":
    main()

