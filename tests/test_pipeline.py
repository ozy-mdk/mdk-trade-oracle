"""End-to-end integration tests for Medallion transformations and Oracle rules."""

from datetime import datetime, timedelta
import polars as pl
import pytest
from mdk_trading_oracle.core.db import DuckDBManager
from mdk_trading_oracle.core.types import SignalType
from mdk_trading_oracle.oracle.evaluator import OracleEvaluator
from mdk_trading_oracle.oracle.signals import RuleEngine
from mdk_trading_oracle.pipeline.bronze_to_silver import BronzeToSilverPipeline
from mdk_trading_oracle.pipeline.silver_to_gold import SilverToGoldPipeline


@pytest.fixture
def in_memory_db(tmp_path):
    """Fixture providing initialized in-memory DuckDB with synthetic data."""
    db = DuckDBManager(in_memory=True)
    db.initialize_schema()

    # Insert sample Bronze trades directly
    conn = db.get_connection()
    base_time = datetime(2026, 8, 1, 10, 0, 0)

    for i in range(15):
        t_time = base_time + timedelta(days=i)
        # BofA buying AKBNK heavily
        conn.execute("""
            INSERT INTO bronze_raw_trades (
                trade_id, timestamp, symbol, price, volume, buyer_broker_id, seller_broker_id, raw_source
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?);
        """, [
            f"T_{i}_1", t_time, "AKBNK", 50.0 + i * 0.5, 10000.0, "BOFA", "GAR", "mock"
        ])
        # Regular trades
        conn.execute("""
            INSERT INTO bronze_raw_trades (
                trade_id, timestamp, symbol, price, volume, buyer_broker_id, seller_broker_id, raw_source
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?);
        """, [
            f"T_{i}_2", t_time, "AKBNK", 50.0 + i * 0.5, 5000.0, "ISY", "YKR", "mock"
        ])

    return db


def test_pipeline_transformation_flow(in_memory_db, tmp_path):
    """Test full Medallion pipeline execution and signal generation."""
    db = in_memory_db

    # 1. Bronze -> Silver
    b2s = BronzeToSilverPipeline(db)
    # Redirect parquet exports to tmp_path
    b2s.settings.silver_dir = tmp_path
    s_res = b2s.run()
    assert s_res["silver_broker_transactions_count"] > 0
    assert s_res["silver_daily_broker_summary_count"] > 0

    # 2. Silver -> Gold
    s2g = SilverToGoldPipeline(db)
    s2g.settings.gold_dir = tmp_path
    g_res = s2g.run()
    assert g_res["gold_bofa_flow_metrics_count"] > 0

    # Verify Gold data contents
    gold_df = db.query_pl("SELECT * FROM gold_bofa_flow_metrics WHERE symbol = 'AKBNK'")
    assert len(gold_df) == 15
    assert gold_df["bofa_net_tl"].sum() > 0

    # 3. Oracle Evaluation
    evaluator = OracleEvaluator(db)
    signals = evaluator.evaluate_latest_signals()
    assert len(signals) >= 1

    akbnk_signal = next(s for s in signals if s.symbol == "AKBNK")
    assert akbnk_signal.signal in [SignalType.STRONG_BUY, SignalType.BUY]
    assert akbnk_signal.confidence >= 0.60
    assert len(akbnk_signal.reasons) > 0


def test_rule_engine_heuristics():
    """Test individual heuristic branches of the Oracle rule engine."""
    # Strong Buy Case
    buy_row = {
        "symbol": "GARAN",
        "date_val": "2026-08-18",
        "bofa_net_tl": 50_000_000.0,
        "bofa_net_share": 0.25,
        "bofa_volume_share": 0.30,
        "bofa_flow_zscore_20d": 2.2,
        "bofa_flow_acceleration_5d": 15_000_000.0,
        "bofa_net_tl_roll_3d": 40_000_000.0,
    }
    sig_buy = RuleEngine.evaluate_row(buy_row)
    assert sig_buy.signal == SignalType.STRONG_BUY
    assert sig_buy.confidence >= 0.80

    # Strong Sell Case
    sell_row = {
        "symbol": "GARAN",
        "date_val": "2026-08-18",
        "bofa_net_tl": -50_000_000.0,
        "bofa_net_share": -0.25,
        "bofa_volume_share": 0.30,
        "bofa_flow_zscore_20d": -2.2,
        "bofa_flow_acceleration_5d": -15_000_000.0,
        "bofa_net_tl_roll_3d": -40_000_000.0,
    }
    sig_sell = RuleEngine.evaluate_row(sell_row)
    assert sig_sell.signal == SignalType.STRONG_SELL
    assert sig_sell.confidence >= 0.80

    # Neutral Case
    neutral_row = {
        "symbol": "GARAN",
        "date_val": "2026-08-18",
        "bofa_net_tl": 100_000.0,
        "bofa_net_share": 0.01,
        "bofa_volume_share": 0.05,
        "bofa_flow_zscore_20d": 0.1,
        "bofa_flow_acceleration_5d": 0.0,
        "bofa_net_tl_roll_3d": 50_000.0,
    }
    sig_neutral = RuleEngine.evaluate_row(neutral_row)
    assert sig_neutral.signal == SignalType.NEUTRAL
