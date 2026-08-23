#!/usr/bin/env python3
"""Interactive Data Preparation & Discovery Script for MDK Trading Oracle.

Scans raw trade feeds, displays detailed ranking tables of discovered instruments and brokers,
and synchronizes the YAML catalogs under config/ with full visibility.
"""

import argparse
import sys
from pathlib import Path

# Add project root to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from mdk_trading_oracle.data.discovery import RawDataInspector


def main():
    parser = argparse.ArgumentParser(
        description="Inspect raw BIST data feeds and prepare/sync instrument & broker catalogs."
    )
    parser.add_argument(
        "--glob",
        "-g",
        type=str,
        default=None,
        help="Optional custom glob path to raw CSV files",
    )
    parser.add_argument(
        "--sync",
        "-s",
        action="store_true",
        help="Synchronize discovered instruments & brokers into config/instruments.yaml and config/brokers.yaml",
    )
    args = parser.parse_args()

    inspector = RawDataInspector(raw_glob=args.glob)
    inspector.print_interactive_report()

    if args.sync:
        print("\n🔄 Synchronizing YAML catalogs...")
        res = inspector.sync_to_yaml_catalogs()
        print(f"✅ Successfully wrote {res['instruments_count']} instruments to {res['instruments_file']}")
        print(f"✅ Successfully wrote {res['brokers_count']} brokers to {res['brokers_file']}\n")
    else:
        print(
            "\n💡 [Tip] Run with '--sync' to automatically write these discovered instruments and brokers to config/ YAMLs:"
        )
        print("   python scripts/prepare_data_catalog.py --sync\n")


if __name__ == "__main__":
    main()
