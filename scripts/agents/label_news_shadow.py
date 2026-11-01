#!/usr/bin/env python
"""Label stored headlines with the configured LLM provider in shadow mode. Labels never reach a forecast.

Usage:
    .venv/bin/python scripts/agents/label_news_shadow.py --limit 5
    .venv/bin/python scripts/agents/label_news_shadow.py --live-only
    .venv/bin/python scripts/agents/label_news_shadow.py --provider gemini_cli --limit 5
"""

from __future__ import annotations

import argparse
import sys

from mdk_trading_oracle.agents.news_classifier import run_shadow_labeling
from mdk_trading_oracle.agents.providers import ProviderError
from mdk_trading_oracle.core.db import PostgresManager


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--limit", type=int, default=None, help="Maximum headlines this run; capped by config")
    ap.add_argument("--provider", default=None, help="Override active_provider from config/agents.yaml")
    ap.add_argument("--live-only", action="store_true", help="Skip BACKFILL headlines")
    args = ap.parse_args()
    try:
        print(run_shadow_labeling(PostgresManager(), args.limit, args.provider, args.live_only))
    except ProviderError as e:
        print(f"[ERROR] {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
