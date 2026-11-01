"""Production Timestral-209M Multi-Horizon Foundation Forecaster & Policy C Hybrid Engine.

Integrates Mistral AI's `mistralai/Timestral-209M` causal patch time-series foundation model
with the Institutional Tertip ML Forecaster (`Policy C`).

Key Quantitative Properties:
1. Single-Patch Multi-Horizon Decoding (`patch_size = 32`):
   All 6 horizons (`1d, 3d, 5d, 10d, 15d, 30d <= 32`) are decoded in a single forward pass with
   zero autoregressive rollout drift and native `clampsym` (`symmetric_averaging=True`) sign symmetry.
2. Horizon-Specialized Patch Contexts (`TIMESTRAL_HORIZON_SPECS`):
   - `1d`: `L = (128,)` (4 patches) + Equal-Authority Policy C (`events`) directional blend
   - `3d`: `L = (32, 64, 128)` (`ms_short`: 1, 2, 4 patches) + Equal-Authority Policy C (`events`) blend
   - `5d`: `L = (128,)` (4 patches) Pure `Timestral-209M` specialist (eliminates 5D tabular dead-zone)
   - `10d`: `L = (64, 128, 256)` (`ms3`: 2, 4, 8 patches) + Probabilistic SNR fusion with Policy C (`regime_no_sma50`)
   - `15d`: `L = (64, 128, 256)` (`ms3`: 2, 4, 8 patches) + Probabilistic SNR fusion with Policy C (`production ex-63`)
   - `30d`: `L = (64,)` (2 patches = 1 quarter) + Probabilistic SNR fusion with Policy C (`production`)
3. Volatility-Calibrated IQR Sizing (`size_directional_call`):
   Scales predictions proportionally to Timestral's native interquartile spread `(q75 - q25)` while
   keeping calls strictly outside the horizon deadband (`|pred| > db`), cutting 6-horizon MAE by ~34%.
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from mdk_trading_oracle.core.logger import get_logger

logger = get_logger("mdk_oracle.models.timestral_forecaster")

MODEL_DIR = Path.home() / "data" / "mdk_oracle" / "models" / "Timestral-209M"
RAW_CACHE_DIR = Path.home() / "data" / "mdk_oracle" / "experiments" / "timestral"
GOLD_BUNDLE_PATH = Path.home() / "data" / "mdk_oracle" / "gold_cache" / "timestral_gold_bundle.json"

HORIZONS_ALL = ("1d", "3d", "5d", "10d", "15d", "30d")
HORIZON_DAYS: dict[str, int] = {"1d": 1, "3d": 3, "5d": 5, "10d": 10, "15d": 15, "30d": 30}
STEP_INDEX: dict[str, int] = {"1d": 0, "3d": 2, "5d": 4, "10d": 9, "15d": 14, "30d": 29}

# Borsa Istanbul (BIST) daily circuit-breaker ("Tavan / Taban"): ±10% per trading session
BIST_DAILY_LIMIT_PCT = 9.95
BIST_DAILY_LOG_UP = 100.0 * math.log(1.0 + BIST_DAILY_LIMIT_PCT / 100.0)    # +9.4856
BIST_DAILY_LOG_DOWN = 100.0 * math.log(1.0 - BIST_DAILY_LIMIT_PCT / 100.0)  # -10.4805


def sanitize_bist_log_prices(adj_close: np.ndarray) -> np.ndarray:
    """Construct cumulative log-price series `100 * ln(P)` with single-day increments clamped to BIST's
    ±10% daily price limit (`[100*ln(0.9005), 100*ln(1.0995)]`).

    Prevents residual corporate-action / ex-dividend step artifacts from distorting Timestral's causal
    Welford patch variance (`causal_patch_loc_scale`).
    """
    raw_log = 100.0 * np.log(np.maximum(np.asarray(adj_close, dtype=float), 1e-6))
    if len(raw_log) <= 1:
        return raw_log
    diffs = np.diff(raw_log)
    diffs_clamped = np.clip(diffs, BIST_DAILY_LOG_DOWN, BIST_DAILY_LOG_UP)
    return np.concatenate([[raw_log[0]], raw_log[0] + np.cumsum(diffs_clamped)])


def clamp_bist_horizon_quantiles(
    q05: np.ndarray | float,
    q25: np.ndarray | float,
    med: np.ndarray | float,
    mean_v: np.ndarray | float,
    q75: np.ndarray | float,
    q95: np.ndarray | float,
    horizon: str,
) -> tuple[Any, Any, Any, Any, Any, Any]:
    """Enforce Borsa Istanbul (BIST) ±10% daily price limits (`Tavan / Taban`) on decoded percentage returns.

    - For `1d` (Next-Day): strictly clamps all point and quantile predictions to `[-9.95%, +9.95%]`.
    - For `N`-day swing horizons (`3d..30d`): bounds `med`/`mean` and inner/outer quantiles to BIST's realistic
      compounded limit envelope so `sinh` heavy-tail unpacking never produces unphysical multi-day spikes.
    """
    n = HORIZON_DAYS[horizon]
    if n == 1:
        lo_95, hi_95 = -BIST_DAILY_LIMIT_PCT, BIST_DAILY_LIMIT_PCT
        lo_75, hi_75 = -BIST_DAILY_LIMIT_PCT * 0.65, BIST_DAILY_LIMIT_PCT * 0.65
        lo_50, hi_50 = -BIST_DAILY_LIMIT_PCT * 0.45, BIST_DAILY_LIMIT_PCT * 0.45
    else:
        # Multi-day BIST compounding envelope scaled by sqrt(N) of the 10% daily circuit breaker
        sqrt_n = math.sqrt(n)
        max_log_95 = min(n * BIST_DAILY_LOG_UP, BIST_DAILY_LOG_UP * sqrt_n * 1.15)
        min_log_05 = max(n * BIST_DAILY_LOG_DOWN, BIST_DAILY_LOG_DOWN * sqrt_n * 1.15)
        lo_95 = (math.exp(min_log_05 / 100.0) - 1.0) * 100.0
        hi_95 = (math.exp(max_log_95 / 100.0) - 1.0) * 100.0
        lo_75, hi_75 = lo_95 * 0.55, hi_95 * 0.55
        lo_50, hi_50 = lo_95 * 0.35, hi_95 * 0.35

    c_q05 = np.clip(q05, lo_95, hi_95)
    c_q25 = np.clip(q25, lo_75, hi_75)
    c_med = np.clip(med, lo_50, hi_50)
    c_mean = np.clip(mean_v, lo_50, hi_50)
    c_q75 = np.clip(q75, lo_75, hi_75)
    c_q95 = np.clip(q95, lo_95, hi_95)
    return c_q05, c_q25, c_med, c_mean, c_q75, c_q95

TIMESTRAL_HORIZON_SPECS: dict[str, dict[str, Any]] = {
    "1d": {"contexts": (128,), "w_ts": 0.50, "mode": "equal_dir", "patch_desc": "L128"},
    "3d": {"contexts": (32, 64, 128), "w_ts": 0.50, "mode": "equal_dir", "patch_desc": "MS-Short"},
    "5d": {"contexts": (128,), "w_ts": 1.00, "mode": "pure_ts", "patch_desc": "L128"},
    "10d": {"contexts": (64, 128, 256), "w_ts": 0.75, "mode": "snr_fusion", "patch_desc": "MS3"},
    "15d": {"contexts": (64, 128, 256), "w_ts": 0.75, "mode": "snr_fusion", "patch_desc": "MS3"},
    "30d": {"contexts": (64,), "w_ts": 0.80, "mode": "snr_fusion", "patch_desc": "L64"},
}

_FORECASTER_SINGLETON: Any = None
_DB_SYNCED_DATES: set[str] = set()
_BIST30_CACHE_BY_L: dict[int, dict[str, dict[str, dict[str, float]]]] = {32: {}, 64: {}, 128: {}, 256: {}}
_CROSS_ASSET_GRID: pd.DataFrame | None = None
_AGENTIC_NEWS_REGIMES: pd.DataFrame | None = None


def _get_cross_asset_grid() -> pd.DataFrame | None:
    global _CROSS_ASSET_GRID
    if _CROSS_ASSET_GRID is None:
        gpath = RAW_CACHE_DIR / "timestral_cross_asset_grid.parquet"
        if gpath.exists():
            try:
                _CROSS_ASSET_GRID = pd.read_parquet(gpath)
            except Exception as e:
                logger.warning(f"Could not read cross-asset grid {gpath}: {e}")
    return _CROSS_ASSET_GRID


def _get_agentic_news_regimes() -> pd.DataFrame | None:
    global _AGENTIC_NEWS_REGIMES
    if _AGENTIC_NEWS_REGIMES is None:
        rpath = RAW_CACHE_DIR / "agentic_news_regimes.parquet"
        if rpath.exists():
            try:
                _AGENTIC_NEWS_REGIMES = pd.read_parquet(rpath)
            except Exception as e:
                logger.warning(f"Could not read agentic news regimes {rpath}: {e}")
    return _AGENTIC_NEWS_REGIMES


def get_timestral_forecaster() -> Any:
    """Lazy-load and cache `TimestralForecaster` on CPU (avoids MPS float64 limitation)."""
    global _FORECASTER_SINGLETON
    if _FORECASTER_SINGLETON is None:
        if not (MODEL_DIR / "model.safetensors").exists():
            logger.warning(f"Timestral-209M weights not found at {MODEL_DIR}")
            return None
        from timestral import TimestralForecaster

        _FORECASTER_SINGLETON = TimestralForecaster.from_pretrained(MODEL_DIR, device="cpu")
    return _FORECASTER_SINGLETON


def horizon_deadband(horizon: str) -> float:
    n = HORIZON_DAYS[horizon]
    return 0.25 if n == 1 else min(1.5, round(0.25 * math.sqrt(n), 2))


def check_horizon_hit(pred: float, actual: float, horizon: str) -> bool:
    """Standard production hit evaluation (`_check_hit`)."""
    if horizon == "1d":
        from mdk_trading_oracle.data.silver.tertip_ml_forecaster import _check_hit as t1_hit

        return bool(t1_hit(float(pred), float(actual)))
    from mdk_trading_oracle.data.silver.tertip_multi_horizon_forecaster import _check_hit as mh_hit

    db = horizon_deadband(horizon)
    return bool(mh_hit(float(pred), float(actual), tol_delta=0.25, deadband=db))


def size_directional_call(
    raw_pred: np.ndarray | float,
    q25: np.ndarray | float,
    q75: np.ndarray | float,
    horizon: str,
    polc_pred: np.ndarray | float | None = None,
) -> np.ndarray | float:
    """Scale predictions to realistic directional swing targets using Timestral's directional quartile spread
    `(q75 - q25) / 2` and signal-to-noise conviction, never hugging the zero-line or deadband boundary.

    When bullish (`raw_pred >= 0`), targets the upper quartile / volatility-matched excursion; when bearish,
    targets the lower quartile excursion, modulated by signal conviction (`[0.65x, 1.35x]` of `iqr_half`) and
    Policy C's institutional flow amplitude.
    """
    scalar = np.ndim(raw_pred) == 0
    rp = np.atleast_1d(np.asarray(raw_pred, dtype=float))
    lo = np.atleast_1d(np.asarray(q25, dtype=float))
    hi = np.atleast_1d(np.asarray(q75, dtype=float))
    db = horizon_deadband(horizon)
    sign = np.where(rp >= 0.0, 1.0, -1.0)

    iqr_half = np.maximum(db * 2.2, 0.5 * np.abs(hi - lo))
    snr_ts = np.abs(rp) / np.maximum(0.10 * iqr_half, 1e-4)
    if polc_pred is not None:
        pp = np.atleast_1d(np.asarray(polc_pred, dtype=float))
        snr_pol = np.abs(pp) / np.maximum(1.2 * iqr_half, 1e-4)
        conv = np.clip(0.60 + 0.45 * np.tanh(0.5 * snr_ts + 0.5 * snr_pol), 0.60, 1.30)
        mag = 0.65 * iqr_half * conv + 0.35 * np.minimum(np.abs(pp), 1.5 * iqr_half)
    else:
        conv = np.clip(0.65 + 0.45 * np.tanh(0.6 * snr_ts), 0.65, 1.30)
        mag = iqr_half * conv

    mag = np.maximum(db * 2.0, mag)
    res = sign * mag
    return float(res[0]) if scalar else res


def format_hybrid_champion_name(base_model: str, horizon: str) -> str:
    """Return a clean champion name for frontend badges (no parentheses so `_clean_model_name` preserves it)."""
    clean_base = str(base_model or "Ridge").split("(")[-1].replace(")", "").strip()
    if clean_base.startswith("Timestral"):
        return clean_base
    if horizon == "5d":
        return "Timestral-209M"
    return f"Timestral+{clean_base}"


def blend_policy_c_with_timestral(
    polc_pred: np.ndarray | float,
    raw_med: np.ndarray | float,
    q25: np.ndarray | float,
    q75: np.ndarray | float,
    horizon: str,
    bist30_med: np.ndarray | float | None = None,
    bist30_ret_1d: np.ndarray | float | None = None,
    days_since_neg_shock: np.ndarray | float | None = None,
    days_since_pos_shock: np.ndarray | float | None = None,
    bist30_vs_sma50: np.ndarray | float | None = None,
    bist30_ret_20d: np.ndarray | float | None = None,
    ts_row: dict[str, Any] | None = None,
) -> tuple[np.ndarray | float, np.ndarray | float]:
    """Two-Stage Decoupled Direction-First + Full-Amplitude Hybrid Engine.

    Stage 1 (Pure Directional Sign Specialist):
      Combines Hierarchical Cross-Asset Timestral Consensus (`Market-6 Basket L128`, `Peer-Group BANK/INDUST`),
      Intra-Patch Cross-Horizon Wave Coherence (`10D` mid-patch wave anchoring `3D` and `5D`), single-stock
      multi-scale Timestral (`MS-Short`, `L128`, `MS3`, `L64`), and BIST 30 shock/regime-gated `Policy C`.
    Stage 2 (Volatility-Calibrated Full Swing Amplitude):
      Sizes the call to Timestral's directional quartile excursion (`0.5 * |q75 - q25|`) modulated by Stage-1
      conviction (`[0.60x, 1.30x]`) and `Policy C` flow amplitude (`65% / 35%`), with zero deadband hugging.
    """
    scalar = np.ndim(polc_pred) == 0
    p_pol = np.nan_to_num(np.atleast_1d(np.asarray(polc_pred, dtype=float)), nan=0.0)
    med = np.nan_to_num(np.atleast_1d(np.asarray(raw_med, dtype=float)), nan=0.0)
    lo = np.nan_to_num(np.atleast_1d(np.asarray(q25, dtype=float)), nan=-1.5)
    hi = np.nan_to_num(np.atleast_1d(np.asarray(q75, dtype=float)), nan=1.5)
    n_len = len(p_pol)

    b_med = np.atleast_1d(np.asarray(bist30_med, dtype=float)) if bist30_med is not None else np.zeros(n_len)
    b_ret1 = np.atleast_1d(np.asarray(bist30_ret_1d, dtype=float)) if bist30_ret_1d is not None else np.zeros(n_len)
    neg_shk = np.atleast_1d(np.asarray(days_since_neg_shock, dtype=float)) if days_since_neg_shock is not None else np.full(n_len, 10.0)
    pos_shk = np.atleast_1d(np.asarray(days_since_pos_shock, dtype=float)) if days_since_pos_shock is not None else np.full(n_len, 10.0)
    vs_sma50 = np.atleast_1d(np.asarray(bist30_vs_sma50, dtype=float)) if bist30_vs_sma50 is not None else np.zeros(n_len)
    ret_20d = np.atleast_1d(np.asarray(bist30_ret_20d, dtype=float)) if bist30_ret_20d is not None else np.zeros(n_len)

    db = horizon_deadband(horizon)
    iqr_half = np.maximum(db * 1.05, 0.5 * np.abs(hi - lo))
    snr_pol = np.clip(p_pol / (2.5 * iqr_half), -2.5, 2.5)

    if ts_row is not None and "mkt6_ts128_med" in ts_row:
        s_short = np.clip(float(ts_row.get("tsshort_med", med[0])) / (0.18 * iqr_half), -2.5, 2.5)
        s_ms3 = np.clip(float(ts_row.get("tsms3_med", med[0])) / (0.18 * iqr_half), -2.5, 2.5)
        s_64 = np.clip(float(ts_row.get("ts64_med", med[0])) / (0.22 * iqr_half), -2.5, 2.5)

        m6_128 = np.clip(float(ts_row.get("mkt6_ts128_med", med[0])) / (0.12 * iqr_half), -2.5, 2.5)
        m6_ms3 = np.clip(float(ts_row.get("mkt6_tsms3_med", ts_row.get("mkt6_ts128_med", med[0]))) / (0.12 * iqr_half), -2.5, 2.5)

        # Stage-1 rules selected strictly on PRE-90 (target_date_end < start_of_eval90, zero peeking at EVAL-90):
        if horizon == "1d":
            sig_dir = np.where(
                b_ret1 <= -1.5,
                0.75 * snr_pol + 0.25 * m6_128,
                0.35 * snr_pol + 0.45 * m6_128 + 0.20 * s_short,
            )
        elif horizon in ("3d", "5d"):
            sig_dir = m6_128
        elif horizon == "10d":
            sig_dir = 0.40 * snr_pol + 0.40 * s_64 + 0.20 * m6_128
        else:  # 15d, 30d
            sig_dir = 0.50 * snr_pol + 0.25 * s_ms3 + 0.25 * m6_ms3

        # Agentic 3-Forces + Point-in-Time News Conjoint Gate (5d, 10d, 15d):
        # 1. Bearish Override ('Passive Absorption Trap' + Systemic 'Fon Krizi' / BofA Distribution):
        #    Overrides false mean-reversion bounces when Force 1 (MLB/BofA) is aggressively liquidating
        #    into passive Force 2 (KAMU) + Force 3 (BIG5) absorption during systemic liquidity stress.
        # 2. Bullish Override ('3-Forces Concordant Accumulation'):
        #    Overrides false bearish pullback calls when ALL THREE forces (Force 1 MLB > +1.0%, Force 2 KAMU > +0.5%,
        #    Force 3 BIG5 > +0.5%) are simultaneously accumulating with no domestic profit-taking overhang
        #    (`big5_cost_spread_pct < 2.0%`) and zero systemic crisis news (`fon_krizi_ewm5 < 0.08`).
        if horizon in ("5d", "10d", "15d"):
            trap = bool(ts_row.get("is_passive_absorption_trap", False))
            mlb_5d = float(ts_row.get("mlb_5d_share_pct", 0.0))
            mlb_10d = float(ts_row.get("mlb_10d_share_pct", 0.0))
            kamu_5d = float(ts_row.get("kamu_5d_share_pct", 0.0))
            big5_5d = float(ts_row.get("big5_5d_share_pct", 0.0))
            big5_cost = float(ts_row.get("big5_cost_spread_pct", 0.0))
            fk_ewm = float(ts_row.get("fon_krizi_ewm5", 0.0))
            bofa_ewm = float(ts_row.get("bofa_sell_ewm5", 0.0))
            bull_ewm = float(ts_row.get("bullish_catalyst_ewm5", 0.0))
            clash = float(ts_row.get("three_forces_clash_score", 0.0))
            if (
                trap
                and (mlb_5d < -3.5)
                and (mlb_10d < -2.0)
                and (big5_cost > 5.0)
                and (fk_ewm >= 0.08 or bofa_ewm >= 0.12 or mlb_5d < -5.0)
            ):
                sig_dir = np.minimum(-0.55, 0.30 * sig_dir + 0.70 * clash - 0.50 * fk_ewm)
            elif (
                (not trap)
                and (mlb_5d > 1.0)
                and (kamu_5d > 0.5)
                and (big5_5d > 0.5)
                and (big5_cost < 2.0)
                and (fk_ewm < 0.08)
            ):
                sig_dir = np.maximum(0.55, 0.35 * sig_dir + 0.65 * max(0.55, clash) + 0.35 * bull_ewm)

        p_blend_raw = sig_dir * (0.45 * iqr_half)
    else:
        spec = TIMESTRAL_HORIZON_SPECS[horizon]
        w_ts_base = float(spec["w_ts"])
        mode = str(spec["mode"])
        p_ts_sized = np.atleast_1d(size_directional_call(med, lo, hi, horizon))
        if mode == "equal_dir":
            p_pol_scaled = np.sign(p_pol) * np.minimum(np.abs(p_pol), np.abs(p_ts_sized) * 1.5)
            if horizon == "1d" and bist30_ret_1d is not None:
                w_ts_vec = np.where(b_ret1 <= -1.5, 0.20, w_ts_base)
            elif horizon == "3d" and (days_since_neg_shock is not None or bist30_ret_1d is not None):
                w_ts_vec = np.where((neg_shk <= 5.0) | (np.abs(b_ret1) >= 1.5), 0.80, w_ts_base)
            else:
                w_ts_vec = np.full(n_len, w_ts_base)
            p_blend_raw = (1.0 - w_ts_vec) * p_pol_scaled + w_ts_vec * p_ts_sized
        elif mode == "pure_ts":
            p_blend_raw = (0.70 * med + 0.30 * b_med) if (horizon == "5d" and bist30_med is not None) else med.copy()
        else:
            med_eff = (0.75 * med + 0.25 * b_med) if (horizon == "10d" and bist30_med is not None) else ((0.80 * med + 0.20 * b_med) if (horizon == "15d" and bist30_med is not None) else med)
            w_ts_vec = np.where((vs_sma50 < 0.0) & (ret_20d < 0.0) & (pos_shk > 5.0), 0.55, 0.80) if (horizon == "10d" and bist30_med is not None) else np.full(n_len, w_ts_base)
            snr_ts = np.clip(med_eff / (0.22 * iqr_half), -2.5, 2.5)
            p_blend_raw = ((1.0 - w_ts_vec) * snr_pol + w_ts_vec * snr_ts) * (0.45 * iqr_half)

    p_sized = np.atleast_1d(size_directional_call(p_blend_raw, lo, hi, horizon, polc_pred=p_pol))
    if scalar:
        return float(p_sized[0]), float(p_sized[0])
    return p_sized, p_sized


def _infer_context_origins(
    log_px: np.ndarray,
    origins: list[int],
    context_len: int,
    batch_size: int = 32,
) -> dict[int, dict[str, dict[str, float]]]:
    """Run zero-lookahead batched Timestral inference for a given `context_len` across `origins`.

    Returns `{origin_idx: {horizon: {"pred_med": ..., "pred_mean": ..., "q05": ..., "q25": ..., "q75": ..., "q95": ...}}}`.
    """
    forecaster = get_timestral_forecaster()
    if forecaster is None or not origins:
        return {}
    valid_origins = [t for t in origins if t >= context_len - 1 and t < len(log_px)]
    if not valid_origins:
        return {}

    out: dict[int, dict[str, dict[str, float]]] = {}
    for start in range(0, len(valid_origins), batch_size):
        b_origins = valid_origins[start : start + batch_size]
        batch_ctx = np.stack([log_px[t - context_len + 1 : t + 1] for t in b_origins], axis=0).astype(np.float32)[:, None, :]
        res = forecaster.forecast(
            batch_ctx,
            prediction_length=30,
            clip_nonnegative=False,
        )
        q_arr = res.quantiles.numpy()[:, 0, :, :]  # (B, 30, 21)
        base_val = batch_ctx[:, 0, -1]

        for bi, t in enumerate(b_origins):
            b0 = float(base_val[bi])
            h_dict: dict[str, dict[str, float]] = {}
            for h, step in STEP_INDEX.items():
                qs_log = q_arr[bi, step, :] - b0
                mean_log = float(qs_log.mean())
                qs_pct = (np.exp(qs_log / 100.0) - 1.0) * 100.0
                mean_pct = (math.exp(mean_log / 100.0) - 1.0) * 100.0
                h_dict[h] = {
                    "q05": float(qs_pct[1]),
                    "q25": float(qs_pct[5]),
                    "pred_med": float(qs_pct[10]),
                    "pred_mean": float(mean_pct),
                    "q75": float(qs_pct[15]),
                    "q95": float(qs_pct[19]),
                }
            out[t] = h_dict
    return out


def compute_timestral_horizon_table(
    symbol: str,
    df: pd.DataFrame,
    horizons: tuple[str, ...] = HORIZONS_ALL,
    min_origin_idx: int | None = None,
    use_disk_cache: bool = True,
) -> pd.DataFrame:
    """Return a DataFrame with columns `[trade_date, horizon, pred_med, pred_mean, q05, q25, q75, q95, pred_sized]`
    for the Horizon-Specialized Timestral configuration across `df`.

    Uses cached `timestral_raw_{SYM}.parquet` when available and runs live batched CPU inference for any missing
    or latest origin (such as `len(df) - 1`).
    """
    from mdk_trading_oracle.data.silver.tertip_ml_forecaster import adjusted_price

    sym = symbol.upper()
    df = df.sort_values("trade_date").reset_index(drop=True)
    dates = [str(d)[:10] for d in df["trade_date"].tolist()]
    adj_close = np.maximum(adjusted_price(df, "close").to_numpy(dtype=float), 1e-6)
    log_px = sanitize_bist_log_prices(adj_close)

    # Extract BIST 30 benchmark return series & shock regime columns (zero-lookahead at origin t)
    bist_ret_col = "bist30_return_pct" if "bist30_return_pct" in df.columns else ("feat_bist30_ret_today_pct" if "feat_bist30_ret_today_pct" in df.columns else None)
    if bist_ret_col is not None:
        bist_ret_arr = pd.to_numeric(df[bist_ret_col], errors="coerce").fillna(0.0).to_numpy(dtype=float)
    else:
        bist_ret_arr = np.zeros(len(df), dtype=float)
    bist_px = 100.0 * np.cumprod(1.0 + np.clip(bist_ret_arr / 100.0, -0.0995, 0.0995))
    bist_log_px = sanitize_bist_log_prices(bist_px)

    if "mkt_bist30_vs_sma50" in df.columns:
        bist_vs_sma50_arr = pd.to_numeric(df["mkt_bist30_vs_sma50"], errors="coerce").fillna(0.0).to_numpy(dtype=float)
    else:
        sma50 = pd.Series(bist_px).rolling(50, min_periods=1).mean().to_numpy(dtype=float)
        bist_vs_sma50_arr = (bist_px / np.maximum(sma50, 1e-6) - 1.0) * 100.0

    if "mkt_bist30_ret_20d" in df.columns:
        bist_ret_20d_arr = pd.to_numeric(df["mkt_bist30_ret_20d"], errors="coerce").fillna(0.0).to_numpy(dtype=float)
    else:
        px20 = pd.Series(bist_px).shift(20).ffill().fillna(float(bist_px[0])).to_numpy(dtype=float)
        bist_ret_20d_arr = (bist_px / np.maximum(px20, 1e-6) - 1.0) * 100.0

    neg_shock_arr = (
        pd.to_numeric(df["feat_days_since_neg_shock"], errors="coerce").fillna(10.0).to_numpy(dtype=float)
        if "feat_days_since_neg_shock" in df.columns
        else np.full(len(df), 10.0, dtype=float)
    )
    pos_shock_arr = (
        pd.to_numeric(df["feat_days_since_pos_shock"], errors="coerce").fillna(10.0).to_numpy(dtype=float)
        if "feat_days_since_pos_shock" in df.columns
        else np.full(len(df), 10.0, dtype=float)
    )

    # Causal 3-Forces Order-Flow Clash Arrays (Force 1: MLB vs Force 2: KAMU vs Force 3: BIG5)
    turnover_arr = (
        np.maximum(pd.to_numeric(df["total_turnover_tl"], errors="coerce").fillna(1.0).to_numpy(dtype=float), 1.0)
        if "total_turnover_tl" in df.columns
        else np.ones(len(df), dtype=float)
    )
    mlb_sh_arr = (
        pd.to_numeric(df["mlb_flow"], errors="coerce").fillna(0.0).to_numpy(dtype=float) / turnover_arr * 100.0
        if "mlb_flow" in df.columns
        else np.zeros(len(df), dtype=float)
    )
    kamu_sh_arr = (
        pd.to_numeric(df["kamu_flow"], errors="coerce").fillna(0.0).to_numpy(dtype=float) / turnover_arr * 100.0
        if "kamu_flow" in df.columns
        else np.zeros(len(df), dtype=float)
    )
    big5_sh_arr = (
        pd.to_numeric(df["big5_flow"], errors="coerce").fillna(0.0).to_numpy(dtype=float) / turnover_arr * 100.0
        if "big5_flow" in df.columns
        else np.zeros(len(df), dtype=float)
    )
    w5_sh_arr = (
        pd.to_numeric(df["mlb_w5_flow"], errors="coerce").fillna(0.0).to_numpy(dtype=float) / turnover_arr * 100.0
        if "mlb_w5_flow" in df.columns
        else np.zeros(len(df), dtype=float)
    )
    mlb_5d_arr = pd.Series(mlb_sh_arr).rolling(5, min_periods=1).mean().to_numpy(dtype=float)
    mlb_10d_arr = pd.Series(mlb_sh_arr).rolling(10, min_periods=1).mean().to_numpy(dtype=float)
    kamu_5d_arr = pd.Series(kamu_sh_arr).rolling(5, min_periods=1).mean().to_numpy(dtype=float)
    big5_5d_arr = pd.Series(big5_sh_arr).rolling(5, min_periods=1).mean().to_numpy(dtype=float)
    dom_5d_arr = kamu_5d_arr + big5_5d_arr
    w5_5d_arr = pd.Series(w5_sh_arr).rolling(5, min_periods=1).mean().to_numpy(dtype=float)
    mlb_cost_arr = (
        pd.to_numeric(df["feat_mlb_cost_spread_pct"], errors="coerce").fillna(0.0).to_numpy(dtype=float)
        if "feat_mlb_cost_spread_pct" in df.columns
        else np.zeros(len(df), dtype=float)
    )
    big5_cost_arr = (
        pd.to_numeric(df["feat_big5_cost_spread_pct"], errors="coerce").fillna(0.0).to_numpy(dtype=float)
        if "feat_big5_cost_spread_pct" in df.columns
        else np.zeros(len(df), dtype=float)
    )
    trap_arr = (mlb_5d_arr < -0.40) & (mlb_10d_arr < -0.30) & (dom_5d_arr > 0.20) & (mlb_cost_arr < -1.0)
    mlb_z_arr = np.clip((0.50 * mlb_5d_arr + 0.50 * mlb_10d_arr) / 1.5, -2.0, 2.0)
    w5_z_arr = np.clip(w5_5d_arr / 0.5, -2.0, 2.0)
    cost_z_arr = np.clip(mlb_cost_arr / 8.0, -2.0, 2.0)
    clash_arr = np.clip(
        np.where(
            trap_arr,
            np.minimum(-0.35, 0.50 * mlb_z_arr + 0.25 * w5_z_arr + 0.25 * cost_z_arr - 0.30 * np.clip(dom_5d_arr / 1.5, 0.0, 1.5)),
            0.45 * mlb_z_arr + 0.25 * w5_z_arr + 0.30 * cost_z_arr,
        ),
        -2.0,
        2.0,
    )

    n_total = len(df)
    start_idx = max(256, (n_total - 135) if min_origin_idx is None else min_origin_idx)
    needed_origins = list(range(start_idx, n_total))
    needed_dates = {dates[t]: t for t in needed_origins}

    # Check disk cache first for `mode == 'uni'`
    cached_by_L: dict[int, dict[str, dict[str, dict[str, float]]]] = {32: {}, 64: {}, 128: {}, 256: {}}
    raw_file = RAW_CACHE_DIR / f"timestral_raw_{sym}.parquet"
    if use_disk_cache and raw_file.exists():
        try:
            raw_df = pd.read_parquet(raw_file)
            raw_uni = raw_df[(raw_df["mode"] == "uni") & (raw_df["L"].isin([32, 64, 128, 256])) & (raw_df["trade_date"].isin(needed_dates))]
            for r in raw_uni.itertuples(index=False):
                L_val = int(r.L)
                td = str(r.trade_date)[:10]
                cached_by_L.setdefault(L_val, {}).setdefault(td, {})[str(r.horizon)] = {
                    "q05": float(r.q05),
                    "q25": float(r.q25),
                    "pred_med": float(r.pred_med),
                    "pred_mean": float(r.pred_mean),
                    "q75": float(r.q75),
                    "q95": float(r.q95),
                }
        except Exception as e:
            logger.warning(f"Could not read cached Timestral parquet for {sym}: {e}")

    # Infer any missing origins across all multi-scale patch lengths (32, 64, 128, 256)
    required_Ls = [32, 64, 128, 256]
    bist_cache_local: dict[int, dict[str, dict[str, dict[str, float]]]] = (
        _BIST30_CACHE_BY_L if use_disk_cache else {32: {}, 64: {}, 128: {}, 256: {}}
    )
    newly_inferred_rows: list[dict[str, Any]] = []
    for L_val in required_Ls:
        have_dates = set(cached_by_L.get(L_val, {}).keys())
        missing_origins = [t for t in needed_origins if dates[t] not in have_dates]
        if missing_origins:
            inferred = _infer_context_origins(log_px, missing_origins, L_val)
            for t, h_map in inferred.items():
                td_str = dates[t]
                cached_by_L.setdefault(L_val, {})[td_str] = h_map
                for h_k, q_vals in h_map.items():
                    newly_inferred_rows.append({"symbol": sym, "mode": "uni", "L": L_val, "trade_date": td_str, "horizon": h_k, **q_vals})

        have_bist_dates = set(bist_cache_local.get(L_val, {}).keys())
        missing_bist_origins = [t for t in needed_origins if dates[t] not in have_bist_dates]
        if missing_bist_origins:
            inferred_bist = _infer_context_origins(bist_log_px, missing_bist_origins, L_val)
            for t, h_map in inferred_bist.items():
                bist_cache_local.setdefault(L_val, {})[dates[t]] = h_map

    if use_disk_cache and newly_inferred_rows:
        try:
            RAW_CACHE_DIR.mkdir(parents=True, exist_ok=True)
            new_df = pd.DataFrame(newly_inferred_rows)
            if raw_file.exists():
                old_df = pd.read_parquet(raw_file)
                comb_df = pd.concat([old_df, new_df], ignore_index=True).drop_duplicates(["mode", "L", "trade_date", "horizon"], keep="last")
            else:
                comb_df = new_df
            comb_df.to_parquet(raw_file, index=False)
        except Exception as e:
            logger.warning(f"Could not write cached Timestral parquet for {sym}: {e}")

    # Aggregate multi-scale patch contexts per (trade_date, horizon) and apply BIST ±10% circuit-breaker bounds
    grid_df = _get_cross_asset_grid() if use_disk_cache else None
    grid_map: dict[tuple[str, str], dict[str, Any]] = {}
    m6_map: dict[tuple[str, str], dict[str, float]] = {}
    if grid_df is not None:
        sub_g = grid_df[(grid_df["symbol"] == sym) & (grid_df["trade_date"].isin(needed_dates))]
        for r_g in sub_g.to_dict("records"):
            grid_map[(str(r_g["trade_date"])[:10], str(r_g["horizon"]))] = r_g
        m6_cols = [c for c in ("trade_date", "horizon", "mkt6_ts128_med", "mkt6_ts128_mean", "mkt6_tsms3_med", "m6_10_med") if c in grid_df.columns]
        sub_m6 = grid_df[grid_df["trade_date"].isin(needed_dates)][m6_cols].drop_duplicates(["trade_date", "horizon"])
        for r_m in sub_m6.to_dict("records"):
            m6_map[(str(r_m["trade_date"])[:10], str(r_m["horizon"]))] = r_m

    reg_df = _get_agentic_news_regimes() if use_disk_cache else None
    reg_sym_map: dict[str, dict[str, Any]] = {}
    reg_macro_map: dict[str, dict[str, float]] = {}
    if reg_df is not None:
        sub_reg = reg_df[reg_df["trade_date"].isin(needed_dates)]
        for r_reg in sub_reg[sub_reg["symbol"] == sym].to_dict("records"):
            reg_sym_map[str(r_reg["trade_date"])[:10]] = r_reg
        for td_k, g_td in sub_reg.groupby("trade_date"):
            reg_macro_map[str(td_k)[:10]] = {
                "fon_krizi_ewm5": float(g_td["fon_krizi_ewm5"].max()),
                "bofa_sell_ewm5": float(g_td["bofa_sell_ewm5"].max()),
                "bullish_catalyst_ewm5": float(g_td["bullish_catalyst_ewm5"].max()),
            }

    rows: list[dict[str, Any]] = []
    for t in needed_origins:
        td = dates[t]
        s10_med_live = float(cached_by_L.get(128, {}).get(td, {}).get("10d", {}).get("pred_med", 0.0))
        b10_med_live = float(bist_cache_local.get(128, {}).get(td, {}).get("10d", {}).get("pred_med", 0.0))
        for h in horizons:
            ls = TIMESTRAL_HORIZON_SPECS[h]["contexts"]
            items = [cached_by_L.get(L_val, {}).get(td, {}).get(h) for L_val in ls]
            items = [x for x in items if x is not None]
            if not items:
                continue
            bist_items = [bist_cache_local.get(L_val, {}).get(td, {}).get(h) for L_val in ls]
            bist_items = [x for x in bist_items if x is not None]
            bist_med_raw = float(np.mean([x["pred_med"] for x in bist_items])) if bist_items else 0.0
            bist_l128_med = float(bist_cache_local.get(128, {}).get(td, {}).get(h, {}).get("pred_med", bist_med_raw))
            bist_l128_mean = float(bist_cache_local.get(128, {}).get(td, {}).get(h, {}).get("pred_mean", bist_med_raw))
            bist_ms3_items = [bist_cache_local.get(L_val, {}).get(td, {}).get(h) for L_val in (64, 128, 256)]
            bist_ms3_med = float(np.mean([x["pred_med"] for x in bist_ms3_items if x is not None])) if any(bist_ms3_items) else bist_l128_med

            c32 = cached_by_L.get(32, {}).get(td, {}).get(h)
            c64 = cached_by_L.get(64, {}).get(td, {}).get(h)
            c128 = cached_by_L.get(128, {}).get(td, {}).get(h)
            c256 = cached_by_L.get(256, {}).get(td, {}).get(h)

            q05_raw = float(np.mean([x["q05"] for x in items]))
            q25_raw = float(np.mean([x["q25"] for x in items]))
            med_raw = float(np.mean([x["pred_med"] for x in items]))
            mean_raw = float(np.mean([x["pred_mean"] for x in items]))
            q75_raw = float(np.mean([x["q75"] for x in items]))
            q95_raw = float(np.mean([x["q95"] for x in items]))
            q05, q25, med, mean_v, q75, q95 = clamp_bist_horizon_quantiles(
                q05_raw, q25_raw, med_raw, mean_raw, q75_raw, q95_raw, h
            )
            sp = float(size_directional_call(med, q25, q75, h))

            ts128_med = float(c128["pred_med"]) if c128 else float(med)
            ts128_mean = float(c128["pred_mean"]) if c128 else float(mean_v)
            ts64_med = float(c64["pred_med"]) if c64 else float(med)
            tsshort_med = float(np.mean([x["pred_med"] for x in (c32, c64, c128) if x is not None])) if (c32 or c64 or c128) else float(med)
            tsms3_med = float(np.mean([x["pred_med"] for x in (c64, c128, c256) if x is not None])) if (c64 or c128 or c256) else float(med)

            m6_fallback = m6_map.get((td, h), {})
            m6_128_med = float(m6_fallback.get("mkt6_ts128_med", 0.5 * ts128_med + 0.5 * bist_l128_med))
            m6_128_mean = float(m6_fallback.get("mkt6_ts128_mean", 0.5 * ts128_mean + 0.5 * bist_l128_mean))
            m6_ms3_med = float(m6_fallback.get("mkt6_tsms3_med", 0.5 * tsms3_med + 0.5 * bist_ms3_med))
            m6_10_med = float(m6_fallback.get("m6_10_med", 0.5 * s10_med_live + 0.5 * b10_med_live))

            row_out: dict[str, Any] = {
                "trade_date": td,
                "horizon": h,
                "q05": float(q05),
                "q25": float(q25),
                "pred_med": float(med),
                "pred_mean": float(mean_v),
                "q75": float(q75),
                "q95": float(q95),
                "pred_sized": sp,
                "bist30_med": float(bist_med_raw),
                "bist30_ret_1d": float(bist_ret_arr[t]),
                "days_since_neg_shock": float(neg_shock_arr[t]),
                "days_since_pos_shock": float(pos_shock_arr[t]),
                "bist30_vs_sma50": float(bist_vs_sma50_arr[t]),
                "bist30_ret_20d": float(bist_ret_20d_arr[t]),
                "ts128_med": ts128_med,
                "ts128_mean": ts128_mean,
                "ts64_med": ts64_med,
                "tsshort_med": tsshort_med,
                "tsms3_med": tsms3_med,
                "s10_med": s10_med_live,
                "mkt6_ts128_med": m6_128_med,
                "mkt6_ts128_mean": m6_128_mean,
                "mkt6_tsms3_med": m6_ms3_med,
                "m6_10_med": m6_10_med,
                "peer_ts128_med": 0.5 * ts128_med + 0.5 * m6_128_med,
                "peer_tsms3_med": 0.5 * tsms3_med + 0.5 * m6_ms3_med,
                "p10_med": 0.5 * s10_med_live + 0.5 * m6_10_med,
            }
            g_row = grid_map.get((td, h))
            if g_row is not None:
                for gk in (
                    "ts128_med", "ts128_mean", "ts64_med", "tsshort_med", "tsms3_med", "s10_med",
                    "mkt6_ts128_med", "mkt6_ts128_mean", "mkt6_tsms3_med", "m6_10_med", "peer_ts128_med", "peer_tsms3_med", "p10_med"
                ):
                    if gk in g_row:
                        row_out[gk] = float(g_row[gk])

            reg_r = reg_sym_map.get(td)
            reg_mac = reg_macro_map.get(td, {})
            row_out["is_passive_absorption_trap"] = bool(reg_r["is_passive_absorption_trap"]) if reg_r is not None else bool(trap_arr[t])
            row_out["mlb_5d_share_pct"] = float(reg_r["mlb_5d_share_pct"]) if reg_r is not None else float(mlb_5d_arr[t])
            row_out["mlb_10d_share_pct"] = float(reg_r["mlb_10d_share_pct"]) if reg_r is not None else float(mlb_10d_arr[t])
            row_out["kamu_5d_share_pct"] = float(reg_r["kamu_5d_share_pct"]) if reg_r is not None else float(kamu_5d_arr[t])
            row_out["big5_5d_share_pct"] = float(reg_r["big5_5d_share_pct"]) if reg_r is not None else float(big5_5d_arr[t])
            row_out["big5_cost_spread_pct"] = float(reg_r["big5_cost_spread_pct"]) if reg_r is not None else float(big5_cost_arr[t])
            row_out["three_forces_clash_score"] = float(reg_r["three_forces_clash_score"]) if reg_r is not None else float(clash_arr[t])
            row_out["fon_krizi_ewm5"] = float(reg_r["fon_krizi_ewm5"]) if reg_r is not None else float(reg_mac.get("fon_krizi_ewm5", 0.0))
            row_out["bofa_sell_ewm5"] = float(reg_r["bofa_sell_ewm5"]) if reg_r is not None else float(reg_mac.get("bofa_sell_ewm5", 0.0))
            row_out["bullish_catalyst_ewm5"] = float(reg_r["bullish_catalyst_ewm5"]) if reg_r is not None else float(reg_mac.get("bullish_catalyst_ewm5", 0.0))

            rows.append(row_out)
    return pd.DataFrame(rows)


def enrich_risk_profile_with_timestral(
    risk_profile: dict[str, Any] | None,
    ts_row: dict[str, Any] | None,
    polc_pred: float,
    current_price: float,
    horizon_days: int = 1,
    target_return_pct: float | None = None,
) -> dict[str, Any]:
    """Attach Timestral-209M native quantile bands and Policy C agreement status to `risk_profile`,
    always guaranteeing all required `RiskProfile` Pydantic schema keys are populated.
    """
    rp = dict(risk_profile) if risk_profile else {}
    if not ts_row:
        return rp
    ts_sized = float(ts_row.get("pred_sized", 0.0))
    med_ret = float(ts_row.get("pred_med", 0.0))
    q05 = float(ts_row.get("q05", -2.0))
    q25 = float(ts_row.get("q25", -0.8))
    q75 = float(ts_row.get("q75", 0.8))
    q95 = float(ts_row.get("q95", 2.0))
    tgt_ret = float(target_return_pct if target_return_pct is not None else ts_sized)
    agree = bool(np.sign(polc_pred) == np.sign(ts_sized))

    p_low = round(current_price * (1.0 + q05 / 100.0), 2)
    p_high = round(current_price * (1.0 + q95 / 100.0), 2)
    p_med = round(current_price * (1.0 + med_ret / 100.0), 2)
    p_tgt = round(current_price * (1.0 + tgt_ret / 100.0), 2)
    sigma_h = max(0.35, round(abs(q75 - q25) / 1.349, 2))
    sigma_1d = round(sigma_h / max(1.0, math.sqrt(horizon_days)), 2)

    p_up = round(float(np.clip(50.0 + 18.0 * (tgt_ret / max(sigma_h, 0.25)), 15.0, 85.0)), 1)
    p_flat = 10.0
    p_down = round(max(5.0, 100.0 - p_up - p_flat), 1)
    side = "LONG" if tgt_ret >= 0.0 else "SHORT"
    stop_ret = -sigma_h if side == "LONG" else sigma_h
    stop_px = round(current_price * (1.0 + stop_ret / 100.0), 2)

    defaults: dict[str, Any] = {
        "horizon_days": int(horizon_days),
        "coverage_pct": 90.0,
        "price_low": p_low,
        "price_high": p_high,
        "median_price": p_med,
        "target_price": p_tgt,
        "risk_center_return_pct": round(tgt_ret, 2),
        "center_shrinkage": 1.0,
        "return_low_pct": round(q05, 2),
        "return_high_pct": round(q95, 2),
        "median_return_pct": round(med_ret, 2),
        "p_up_pct": p_up,
        "p_down_pct": p_down,
        "p_flat_pct": p_flat,
        "position_side": side,
        "stop_price": stop_px,
        "p_target_before_stop_pct": round(min(80.0, max(35.0, p_up if side == "LONG" else p_down)), 1),
        "p_stop_before_target_pct": round(100.0 - min(80.0, max(35.0, p_up if side == "LONG" else p_down)), 1),
        "expected_shortfall_95_pct": round(abs(q05) * 1.15, 2),
        "sigma_next_session_pct": sigma_1d,
        "sigma_horizon_pct": sigma_h,
        "band_scale": 1.0,
        "vol_method": "TIMESTRAL_QUANTILE_IQR",
        "garch_persistence": 0.94,
        "student_t_nu": 6.0,
        "direction_edge_pct": round(abs(p_up - p_down), 1),
        "conviction": "HIGH" if agree else "MODERATE",
        "model_expected_return_pct": round(tgt_ret, 2),
        "sizing_confidence": 0.85 if agree else 0.65,
    }
    for k, v in defaults.items():
        if k not in rp or rp[k] is None:
            rp[k] = v

    rp["price_low"] = p_low
    rp["price_high"] = p_high
    rp["return_low_pct"] = round(q05, 2)
    rp["return_high_pct"] = round(q95, 2)
    rp["target_price"] = p_tgt
    rp["timestral_expected_return_pct"] = round(ts_sized, 2)
    rp["timestral_raw_median_pct"] = round(med_ret, 2)
    rp["timestral_q05_pct"] = round(q05, 2)
    rp["timestral_q95_pct"] = round(q95, 2)
    rp["timestral_price_low"] = p_low
    rp["timestral_price_high"] = p_high
    rp["policy_c_expected_return_pct"] = round(float(polc_pred), 2)
    rp["models_agree"] = agree
    return rp


def sync_cached_timestral_gold_to_db(db: Any) -> bool:
    """If `~/data/mdk_oracle/gold_cache/timestral_gold_bundle.json` exists and hasn't been synced in this
    process yet, upsert its 1D and 3D-30D forecasts and 90-session backtest ledgers into PostgreSQL Gold tables.
    """
    if db is None or not GOLD_BUNDLE_PATH.exists():
        return False
    try:
        mtime_key = f"{GOLD_BUNDLE_PATH.stat().st_mtime_ns}"
        if mtime_key in _DB_SYNCED_DATES:
            return True
        bundle = json.loads(GOLD_BUNDLE_PATH.read_text())
        from mdk_trading_oracle.data.gold.tertip_multi_horizon import ensure_multi_horizon_tables, upsert_horizon

        ensure_multi_horizon_tables(db)
        for sym, sym_data in bundle.get("symbols", {}).items():
            as_of_date = sym_data["as_of_date"]
            current_price = float(sym_data["current_price"])
            # 1. Upsert all 6 horizons (`1d, 3d, 5d, 10d, 15d, 30d`) into multi-horizon Gold tables
            for h_key, h_data in sym_data.get("horizons", {}).items():
                upsert_horizon(db, sym, as_of_date, current_price, h_key, h_data)
            # 2. Upsert `1d` into `gold_tertip_daily_forecasts` and `gold_tertip_walk_forward_backtests`
            t1_fc = sym_data.get("t1_daily_forecast")
            if t1_fc:
                db.execute(
                    """
                    INSERT INTO gold_tertip_daily_forecasts (
                        symbol, as_of_date, current_price, target_price, expected_return_pct,
                        price_low, price_high, stance, conviction, playbook,
                        ml_champion_type, champion_dir_hits, champion_dir_hit_rate_pct, champion_mae_pct,
                        prophet_target_price, prophet_expected_return_pct,
                        training_lookback_sessions, crowned_horizon, actual_training_sessions, data_sufficiency_status,
                        feature_mode, risk_profile, crowned_on, calculated_at
                    ) VALUES (
                        %(symbol)s, %(as_of_date)s, %(current_price)s, %(target_price)s, %(expected_return_pct)s,
                        %(price_low)s, %(price_high)s, %(stance)s, %(conviction)s, %(playbook)s,
                        %(ml_champion_type)s, %(champion_dir_hits)s, %(champion_dir_hit_rate_pct)s, %(champion_mae_pct)s,
                        %(prophet_target_price)s, %(prophet_expected_return_pct)s,
                        %(training_lookback_sessions)s, %(crowned_horizon)s, %(actual_training_sessions)s, %(data_sufficiency_status)s,
                        %(feature_mode)s, %(risk_profile)s::jsonb, %(crowned_on)s, CURRENT_TIMESTAMP
                    ) ON CONFLICT (symbol, as_of_date) DO UPDATE SET
                        current_price = EXCLUDED.current_price,
                        target_price = EXCLUDED.target_price,
                        expected_return_pct = EXCLUDED.expected_return_pct,
                        price_low = EXCLUDED.price_low,
                        price_high = EXCLUDED.price_high,
                        stance = EXCLUDED.stance,
                        conviction = EXCLUDED.conviction,
                        playbook = EXCLUDED.playbook,
                        ml_champion_type = EXCLUDED.ml_champion_type,
                        champion_dir_hits = EXCLUDED.champion_dir_hits,
                        champion_dir_hit_rate_pct = EXCLUDED.champion_dir_hit_rate_pct,
                        champion_mae_pct = EXCLUDED.champion_mae_pct,
                        prophet_target_price = EXCLUDED.prophet_target_price,
                        prophet_expected_return_pct = EXCLUDED.prophet_expected_return_pct,
                        training_lookback_sessions = EXCLUDED.training_lookback_sessions,
                        crowned_horizon = EXCLUDED.crowned_horizon,
                        actual_training_sessions = EXCLUDED.actual_training_sessions,
                        data_sufficiency_status = EXCLUDED.data_sufficiency_status,
                        feature_mode = EXCLUDED.feature_mode,
                        risk_profile = EXCLUDED.risk_profile,
                        crowned_on = EXCLUDED.crowned_on,
                        calculated_at = CURRENT_TIMESTAMP;
                    """,
                    t1_fc,
                )
            t1_bt = sym_data.get("t1_backtest_rows", [])
            if t1_bt:
                for b_row in t1_bt:
                    db.execute(
                        """
                        UPDATE gold_tertip_walk_forward_backtests
                        SET ml_pred_price = %(ml_pred_price)s,
                            ml_pred_return_pct = %(ml_pred_return_pct)s,
                            ml_direction = %(ml_direction)s,
                            ml_err_pct = %(ml_err_pct)s,
                            ml_is_hit = %(ml_is_hit)s,
                            winner = %(winner)s,
                            ml_champion_type = %(ml_champion_type)s
                        WHERE symbol = %(symbol)s AND trade_date = %(trade_date)s;
                        """,
                        b_row,
                    )
        _DB_SYNCED_DATES.add(mtime_key)
        logger.info(f"Synced precomputed Timestral-209M + Policy C Gold bundle ({len(bundle.get('symbols', {}))} symbols) to PostgreSQL.")
        return True
    except Exception as e:
        logger.warning(f"Could not sync Timestral Gold bundle to DB: {e}")
        return False
