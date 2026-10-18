#!/usr/bin/env python
"""Fit the forecast-sizing reference used by `stochastic_risk.size_forecast`.

Confidence for volatility-based sizing is the percentile of a forecast's signal-to-noise ratio
(|model forecast| / horizon sigma) within the walk-forward distribution of the crowned regressors, per horizon.
Reads the zero-look-ahead outputs of `scripts/experiments/direction_size_experiment.py` and merges 21 quantiles per
horizon into `config/volatility_calibration.yaml` under `sizing`.

Usage:
    .venv/bin/python scripts/calibrate_forecast_sizing.py [--source ~/data/mdk_oracle/experiments/direction_size]
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from mdk_trading_oracle.models.stochastic_risk import write_sizing_reference

DEFAULT_SOURCE = Path.home() / "data" / "mdk_oracle" / "experiments" / "direction_size"
LEVELS = np.linspace(0.0, 1.0, 21)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--source", type=Path, default=DEFAULT_SOURCE)
    args = ap.parse_args()
    files = sorted(f for f in args.source.glob("*.parquet") if f.stem.isupper())  # per-stock outputs only
    if not files:
        raise SystemExit(f"No experiment outputs found in {args.source}")
    d = pd.concat([pd.read_parquet(f, columns=["symbol", "horizon", "n_days", "old_pred", "sigma_h"]) for f in files])
    d = d[(d["sigma_h"] > 0) & (d["old_pred"] != 0)]
    d["snr"] = d["old_pred"].abs() / d["sigma_h"]
    horizons = {}
    for h, g in d.groupby("horizon"):
        q = np.quantile(g["snr"], LEVELS)
        horizons[h] = {"snr_quantiles": [round(float(x), 5) for x in q], "n_rows": int(len(g)), "n_symbols": int(g["symbol"].nunique())}
        print(f"{h:>4}: n={len(g):6d}  median snr={np.median(g['snr']):.3f}  p90={q[18]:.3f}")
    path = write_sizing_reference({"method": "REG_Z_SNR_PERCENTILE", "source": str(args.source), "horizons": horizons})
    print(f"Sizing reference written to {path}")


if __name__ == "__main__":
    main()
