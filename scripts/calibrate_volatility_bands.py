#!/usr/bin/env python
"""Conformal calibration of GJR-GARCH-t / FHS forecast bands against audited out-of-sample ledgers.

Zero-leakage protocol (per symbol):
  1. GJR-GARCH-t parameters and the standardized residual pool are estimated strictly on adjusted
     returns *before* the first ledger origin date.
  2. Parameters are held fixed and the variance recursion is filtered forward, so the state at each
     origin T only uses information up to T Close.
  3. For each ledger row the forecast distribution is simulated around the persisted prediction and a
     grid of dispersion scales is tested for inclusion of the realized (adjusted) terminal return.
  4. The scale is fitted on the earlier half of origin dates and coverage is reported on the later half
     (honest holdout); the persisted scale is then refitted on all rows.

Usage:
    .venv/bin/python scripts/calibrate_volatility_bands.py            # report + write config
    .venv/bin/python scripts/calibrate_volatility_bands.py --dry-run  # report only
"""

from __future__ import annotations

import argparse
import math
import sys
import zlib
from collections import defaultdict

import numpy as np
import pandas as pd

from mdk_trading_oracle.core.db import PostgresManager
from mdk_trading_oracle.data.silver.tertip_ml_forecaster import DEADBAND_PCT
from mdk_trading_oracle.data.silver.tertip_multi_horizon_forecaster import get_horizon_deadband
from mdk_trading_oracle.models.stochastic_risk import (
    DEFAULT_COVERAGE,
    GARCH_FIT_LOOKBACK,
    QUANTILE_GRID,
    VolatilityState,
    directional_reliability,
    filter_conditional_variance,
    fit_center_shrinkage,
    fit_garch_params,
    fit_scale_for_coverage,
    inclusion_matrix,
    scale_index,
    simulate_log_paths,
    write_band_calibration,
)


def _load_prices(db: PostgresManager, symbol: str) -> pd.DataFrame:
    df = db.execute(
        """
        SELECT trade_date, COALESCE(adj_close_price, close_price) AS adj_px, daily_return_pct
        FROM silver_daily_stock_summary WHERE symbol = %s ORDER BY trade_date ASC;
        """,
        (symbol,),
    ).df()
    df["trade_date"] = df["trade_date"].astype(str).str[:10]
    df["log_ret"] = np.log(df["adj_px"].astype(float)).diff() * 100.0
    df["legacy_sd20_pct"] = df["daily_return_pct"].astype(float).rolling(20, min_periods=10).std() * 100.0
    return df.reset_index(drop=True)


def _load_ledger_rows(db: PostgresManager, symbol: str) -> list[tuple[str, int, float]]:
    """Return (origin_date, n_days, predicted_return_pct) rows from both audited ledgers."""
    rows: list[tuple[str, int, float]] = []
    # T+1 ledger: trade_date is the realized session; origin is the prior session (resolved later).
    for d, pred in db.execute(
        "SELECT trade_date, ml_pred_return_pct FROM gold_tertip_walk_forward_backtests "
        "WHERE symbol = %s AND ml_pred_return_pct IS NOT NULL",
        (symbol,),
    ).fetchall():
        rows.append(("T1:" + str(d)[:10], 1, float(pred)))
    for d, n, pred in db.execute(
        "SELECT trade_date, horizon_days, pred_return_pct FROM gold_tertip_multi_horizon_backtests "
        "WHERE symbol = %s AND NOT COALESCE(is_pending, FALSE) AND pred_return_pct IS NOT NULL",
        (symbol,),
    ).fetchall():
        rows.append((str(d)[:10], int(n), float(pred)))
    return rows


def calibrate(db: PostgresManager, coverage: float, n_paths: int) -> dict:
    symbols = [r[0] for r in db.execute("SELECT DISTINCT symbol FROM gold_tertip_daily_forecasts ORDER BY 1").fetchall()]
    records: dict[int, list[dict]] = defaultdict(list)
    methods: dict[str, int] = defaultdict(int)

    for sym in symbols:
        px = _load_prices(db, sym)
        pos = {d: i for i, d in enumerate(px["trade_date"])}
        raw_rows = _load_ledger_rows(db, sym)
        rows = []
        for origin, n, pred in raw_rows:
            if origin.startswith("T1:"):
                i_real = pos.get(origin[3:])
                if i_real is None or i_real < 1:
                    continue
                i = i_real - 1
            else:
                i = pos.get(origin)
                if i is None:
                    continue
            if i + n < len(px):
                rows.append((i, n, pred))
        if not rows:
            continue

        cutoff = min(i for i, _, _ in rows)
        pre = px["log_ret"].iloc[1 : cutoff + 1].dropna().to_numpy()[-GARCH_FIT_LOOKBACK:]
        if len(pre) < 120:
            continue
        mu = pre.mean()
        params = fit_garch_params(pre)
        methods[params.method] += 1
        s2_pre = filter_conditional_variance(pre - mu, params, float(np.var(pre[:60])))
        z = (pre - mu) / np.sqrt(np.maximum(s2_pre[:-1], 1e-12))
        pool = (z - z.mean()) / z.std()

        # Filter forward with fixed parameters across the full series (point-in-time at every origin)
        all_r = px["log_ret"].fillna(0.0).to_numpy() - mu
        s2_all = filter_conditional_variance(all_r[1:], params, float(np.var(pre[:60])))
        # s2_all[k] = Var(return at position k+1 | info through position k)

        adj = px["adj_px"].astype(float).to_numpy()
        rng = np.random.default_rng(zlib.crc32(sym.encode("utf-8")))
        for i, n, pred in rows:
            realized = (adj[i + n] / adj[i] - 1.0) * 100.0
            state = VolatilityState(params=params, sigma2_next=float(s2_all[i]), std_resid=pool, n_obs=len(pre), as_of_date="")
            paths = simulate_log_paths(state, n, n_paths, rng)[:, -1:]
            deadband = DEADBAND_PCT if n == 1 else get_horizon_deadband(n)

            legacy_sd = float(px["legacy_sd20_pct"].iloc[i]) if pd.notna(px["legacy_sd20_pct"].iloc[i]) else 2.5
            if n > 1:
                legacy_sd = max(1.0, legacy_sd * math.sqrt(n / 5.0))
            target = 1.0 + pred / 100.0
            legacy_in = target * (1 - 1.645 * legacy_sd / 100) <= 1 + realized / 100 <= target * (1 + 1.645 * legacy_sd / 100)

            records[n].append({
                "date": px["trade_date"].iloc[i],
                "pred": pred,
                "realized": realized,
                "deadband": deadband,
                "legacy_in": bool(legacy_in),
                "q": np.quantile(paths[:, 0], QUANTILE_GRID),
            })
        print(f"  {sym:6s} {params.method:12s} persistence={params.persistence:.3f} rows={len(rows)}", file=sys.stderr)

    result = {"coverage": coverage, "fit_methods": dict(methods), "horizons": {}}
    print(
        f"\n{'Horizon':>8} {'Rows':>6} {'Legacy':>7} {'Raw':>7} | {'Kappa':>6} {'Scale':>6} {'Holdout':>8} "
        f"{'P(dir) model/real':>18} | {'Kappa':>6} {'Scale':>6}"
    )
    for n in sorted(records):
        recs = sorted(records[n], key=lambda r: r["date"])
        R = {k: np.array([r[k] for r in recs]) for k in ("pred", "realized", "deadband", "legacy_in", "q")}
        dates = np.array([r["date"] for r in recs])
        split = sorted(set(dates))[len(set(dates)) // 2]
        first = dates < split

        def fit(mask: np.ndarray) -> tuple[float, float]:
            kappa = fit_center_shrinkage(R["pred"][mask], R["realized"][mask])
            incl = inclusion_matrix(R["q"][mask], kappa * R["pred"][mask], R["realized"][mask], coverage)
            return kappa, fit_scale_for_coverage(incl, coverage)

        k_h, c_h = fit(first)
        hold = ~first
        incl_hold = inclusion_matrix(R["q"][hold], k_h * R["pred"][hold], R["realized"][hold], coverage)
        holdout_cov = float(incl_hold[:, scale_index(c_h)].mean() * 100)
        p_model, p_real = directional_reliability(
            R["q"][hold], R["pred"][hold], k_h, c_h, R["realized"][hold], R["deadband"][hold]
        )
        raw = inclusion_matrix(R["q"], R["pred"], R["realized"], coverage)[:, scale_index(1.0)].mean() * 100
        legacy_cov = float(R["legacy_in"].mean() * 100)
        kappa, scale = fit(np.ones(len(recs), dtype=bool))
        print(
            f"{n:>7}d {len(recs):>6} {legacy_cov:>6.1f}% {raw:>6.1f}% | {k_h:>6.3f} {c_h:>6.2f} {holdout_cov:>7.1f}% "
            f"{p_model:>8.1f}% /{p_real:>6.1f}% | {kappa:>6.3f} {scale:>6.2f}"
        )
        result["horizons"][f"{n}d"] = {
            "scale": round(scale, 3),
            "center_shrinkage": round(kappa, 4),
            "n_rows": len(recs),
            "legacy_band_coverage_pct": round(legacy_cov, 1),
            "uncalibrated_coverage_pct": round(float(raw), 1),
            "holdout_center_shrinkage": round(k_h, 4),
            "holdout_scale": round(c_h, 3),
            "holdout_coverage_pct": round(holdout_cov, 1),
            "holdout_model_directional_prob_pct": round(p_model, 1),
            "holdout_realized_directional_freq_pct": round(p_real, 1),
        }
    return result


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--coverage", type=float, default=DEFAULT_COVERAGE)
    ap.add_argument("--n-paths", type=int, default=1500)
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    db = PostgresManager(read_only=True)
    result = calibrate(db, args.coverage, args.n_paths)
    if not args.dry_run:
        path = write_band_calibration(result)
        print(f"\nCalibration written to {path}")


if __name__ == "__main__":
    main()
