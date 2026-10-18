#!/usr/bin/env python
"""Preview: volatility-based sizing + calibrated probabilities on top of the crowned regressors (experiment only).

For each requested stock:
  1. Backtest scorecard (from `direction_size_experiment.py` outputs): current forecasts vs REG_Z sizing at a 100% call
     rate (direction unchanged, size from the GJR-GARCH-t distribution scaled by signal-to-noise confidence).
  2. Live preview for the latest session: current forecast vs new sized forecast, P(up) / P(down) beyond the
     deadband and the calibrated 90% range from the stochastic risk engine.

Usage:
    .venv/bin/python scripts/experiments/sizing_preview.py --symbols ASELS,GARAN,AKBNK,YKBNK,ISCTR,TTKOM
"""

from __future__ import annotations

import argparse
import os
import sys
import warnings
import zlib
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "1")
warnings.filterwarnings("ignore")
sys.path.insert(0, str(Path(__file__).resolve().parent))

import direction_size_experiment as ds  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

H = list(ds.HORIZONS)


def backtest_table(res: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (sym, h), g in res.groupby(["symbol", "horizon"]):
        new = ds.build_new_forecasts(g, "REG_Z", 1.0)
        old_called = g["old_pred"].abs() > g["deadband"]
        o = ds.scorecard(g, "old_pred", old_called)
        n = ds.scorecard(new, "new_pred", new["new_called"])
        rows.append({
            "symbol": sym, "horizon": h,
            "strict_old": o["strict_precision"], "strict_new": n["strict_precision"],
            "lenient_old": o["legacy_lenient_hit"], "lenient_new": n["legacy_lenient_hit"],
            "size_ratio_old": o["size_ratio_on_hits"], "size_ratio_new": n["size_ratio_on_hits"],
            "median_size_old": o["median_abs_pred"], "median_size_new": n["median_abs_pred"],
            "over5_old": o["share_above_5pct"], "over5_new": n["share_above_5pct"],
        })
    return pd.DataFrame(rows)


def live_preview(sym: str, history: pd.DataFrame) -> list[dict]:
    from mdk_trading_oracle.core.db import PostgresManager
    from mdk_trading_oracle.data.silver.tertip_ml_forecaster import DEADBAND_PCT, get_tertip_ml_forecast
    from mdk_trading_oracle.data.silver.tertip_multi_horizon_forecaster import (
        get_horizon_deadband,
        get_multi_horizon_forecaster_payload,
    )
    from mdk_trading_oracle.models import stochastic_risk as sr

    db = PostgresManager(read_only=True)
    t1 = get_tertip_ml_forecast(db, sym, force_refresh=True)
    mh = get_multi_horizon_forecaster_payload(db, sym, force_recompute=True, n_eval_sessions=20)
    as_of = str(t1["as_of_date"])[:10]
    price = float(t1["latest_close_price"])
    state = sr.get_volatility_state(db, sym, as_of)
    current = {"1d": (float(t1["expected_return_pct"]), t1.get("risk_profile"))}
    for h, d in mh.get("horizons", {}).items():
        fc = d["live_forecast"]
        current[h] = (float(fc["expected_return_pct"]), fc.get("risk_profile"))
    out = []
    for h in H:
        if h not in current or state is None:
            continue
        n = ds.HORIZONS[h]
        cur_ret, risk = current[h]
        rng = np.random.default_rng(zlib.crc32(f"{sym}:{as_of}:{h}".encode()))
        term = (np.exp(sr.simulate_log_paths(state, n, 4000, rng)[:, -1] / 100.0) - 1.0) * 100.0
        up = cur_ret > 0
        side = term[term > 0] if up else -term[term < 0]
        med, q90, sigma_h = float(np.median(side)), float(np.quantile(side, 0.9)), float(np.std(term))
        # Confidence: signal-to-noise rank against this stock/horizon's own backtest history (no future data)
        hist = history[(history.symbol == sym) & (history.horizon == h)]
        z_hist = (hist["old_pred"] / hist["sigma_h"].replace(0, np.nan)).abs().dropna()
        z_now = abs(cur_ret) / sigma_h if sigma_h > 0 else 0.0
        conf = float((z_hist < z_now).mean()) if len(z_hist) >= ds.MIN_EDGE_HISTORY else 0.5
        size = min((0.75 + 0.5 * conf) * med, q90)
        if n == 1:
            size = float(ds.t1_compress(np.array([size]))[0])
        new_ret = size if up else -size
        db_h = DEADBAND_PCT if n == 1 else get_horizon_deadband(n)
        out.append({
            "symbol": sym, "as_of": as_of, "price": round(price, 2), "horizon": h,
            "current_%": round(cur_ret, 2), "new_%": round(new_ret, 2),
            "new_target": round(price * (1 + new_ret / 100), 2),
            "P(up)%": risk.get("p_up_pct") if risk else None, "P(down)%": risk.get("p_down_pct") if risk else None,
            "range_90": f"{risk['price_low']:.2f}-{risk['price_high']:.2f}" if risk else None,
            "confidence": round(conf, 2), "deadband_%": db_h,
        })
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--symbols", required=True)
    args = ap.parse_args()
    symbols = [s.strip().upper() for s in args.symbols.split(",") if s.strip()]
    res = pd.concat([pd.read_parquet(ds.OUT_DIR / f"{s}.parquet") for s in symbols], ignore_index=True)

    bt = backtest_table(res)
    live = pd.DataFrame([row for s in symbols for row in live_preview(s, res)])
    out_dir = ds.OUT_DIR / "preview"
    out_dir.mkdir(parents=True, exist_ok=True)
    bt.to_csv(out_dir / "backtest_scorecard.csv", index=False)
    live.to_csv(out_dir / "live_preview.csv", index=False)

    agg = bt.groupby("horizon")[[c for c in bt.columns if c not in ("symbol", "horizon")]].mean().reindex(H)
    print("## Backtest scorecard (mean of selected stocks)\n")
    print(agg.round(2).to_string())
    print("\n## Live preview\n")
    print(live.to_string(index=False))


if __name__ == "__main__":
    main()
