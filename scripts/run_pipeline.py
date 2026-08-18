"""Standalone script to execute the complete MDK Medallion pipeline."""

import sys
from pathlib import Path

# Add src to sys.path
SRC_DIR = Path(__file__).resolve().parents[1] / "src"
sys.path.insert(0, str(SRC_DIR))

from mdk_trading_oracle.core.db import DuckDBManager
from mdk_trading_oracle.ingestion.file_loader import FileIngestor
from mdk_trading_oracle.pipeline.bronze_to_silver import BronzeToSilverPipeline
from mdk_trading_oracle.pipeline.silver_to_gold import SilverToGoldPipeline
from mdk_trading_oracle.oracle.evaluator import OracleEvaluator


def main():
    print("🚀 Running MDK Trading Oracle Medallion Pipeline...")
    db = DuckDBManager()
    db.initialize_schema()

    # 1. Ingest Bronze
    ingestor = FileIngestor(db)
    bronze_res = ingestor.ingest_all_bronze()
    print(f"Bronze: Ingested {len(bronze_res)} files.")

    # 2. Transform Silver
    b2s = BronzeToSilverPipeline(db)
    silver_res = b2s.run()
    print(f"Silver: Generated {silver_res['silver_broker_transactions_count']} transactions.")

    # 3. Transform Gold
    s2g = SilverToGoldPipeline(db)
    gold_res = s2g.run()
    print(f"Gold: Generated {gold_res['gold_bofa_flow_metrics_count']} flow records.")

    # 4. Oracle Decisions
    evaluator = OracleEvaluator(db)
    signals = evaluator.evaluate_latest_signals()
    print(f"Oracle: Generated {len(signals)} actionable signals.")

    print("\n--- Signals Summary ---")
    for s in signals:
        print(f"[{s.signal.value:^11}] {s.symbol:<6} | Net: {s.bofa_net_tl:>12,.0f} TL | Z-Score: {s.bofa_flow_zscore:>+5.2f} | Conf: {s.confidence:.0%}")


if __name__ == "__main__":
    main()
