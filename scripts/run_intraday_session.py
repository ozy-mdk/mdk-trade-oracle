#!/usr/bin/env python
"""Intraday Session Forecaster runner.

    # one-off walk-forward history (replaces SIMULATED rows only; ~25 min for 31 stocks on 8 processes)
    .venv/bin/python scripts/run_intraday_session.py --simulate
    .venv/bin/python scripts/run_intraday_session.py --simulate --symbols ASELS,GARAN --start 2022-07-01

    # nightly (also runs as the first step of `run_pipeline.py --target gold`): grade new sessions, publish the next one
    .venv/bin/python scripts/run_intraday_session.py --nightly
    .venv/bin/python scripts/run_intraday_session.py --nightly --restate   # overwrite tonight's published signal (logged)
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from mdk_trading_oracle.core.db import PostgresManager  # noqa: E402
from mdk_trading_oracle.data.gold.intraday_session import (  # noqa: E402
    SIM_START,
    simulate_history,
    update_gold_intraday_session,
)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--simulate", action="store_true")
    ap.add_argument("--nightly", action="store_true")
    ap.add_argument("--restate", action="store_true")
    ap.add_argument("--symbols", default="")
    ap.add_argument("--start", default=SIM_START)
    ap.add_argument("--workers", type=int, default=8)
    a = ap.parse_args()
    syms = [s.strip().upper() for s in a.symbols.split(",") if s.strip()] or None
    db = PostgresManager()
    if a.simulate:
        print(json.dumps(simulate_history(db, syms, start=a.start, workers=a.workers), default=str, indent=1), flush=True)
    if a.nightly:
        print(json.dumps(update_gold_intraday_session(db, syms, restate=a.restate), default=str, indent=1), flush=True)


if __name__ == "__main__":
    main()
