"""Tests for raw data discovery, entity inspection, and YAML catalog synchronization."""

from pathlib import Path
from tempfile import TemporaryDirectory

import duckdb
import yaml

from mdk_trading_oracle.data.discovery import RawDataInspector


def test_raw_data_inspector_in_memory():
    """Test discovery methods on an in-memory sample dataset."""
    inspector = RawDataInspector()

    # Mock the read connection to return synthetic trade data
    mem_conn = duckdb.connect(":memory:")
    mem_conn.execute("""
        CREATE TABLE bronze_raw_trades (
            trade_id VARCHAR,
            timestamp TIMESTAMP,
            symbol VARCHAR,
            price DOUBLE,
            volume DOUBLE,
            buyer_broker_id VARCHAR,
            seller_broker_id VARCHAR,
            raw_source VARCHAR
        );
        INSERT INTO bronze_raw_trades VALUES
            ('1', '2026-03-10 10:00:00', 'THYAO.E', 300.0, 1000.0, 'MLB', 'ISY', 'test'),
            ('2', '2026-03-10 11:00:00', 'AKBNK.E', 60.0, 2000.0, 'YKR', 'MLB', 'test');
    """)

    inspector._get_read_connection = lambda: mem_conn

    # 1. Test Summary
    summary = inspector.inspect_dataset_summary()
    assert summary["total_trades"] == 2
    assert summary["distinct_symbols"] == 2
    assert summary["distinct_brokers"] == 2

    # 2. Test Instruments Discovery
    instruments = inspector.discover_instruments()
    symbols = [inst["symbol"] for inst in instruments]
    assert "THYAO" in symbols
    assert "AKBNK" in symbols

    # 3. Test Brokers Discovery
    brokers = inspector.discover_brokers()
    codes = [b["code"] for b in brokers]
    assert "MLB" in codes
    assert "ISY" in codes
    assert "YKR" in codes

    # 4. Test YAML Catalog Synchronization
    with TemporaryDirectory() as tmp_dir:
        tmp_inst = Path(tmp_dir) / "instruments.yaml"
        tmp_brk = Path(tmp_dir) / "brokers.yaml"

        sync_res = inspector.sync_to_yaml_catalogs(
            instruments_path=tmp_inst,
            brokers_path=tmp_brk,
        )

        assert sync_res["instruments_count"] == 2
        assert sync_res["brokers_count"] == 3
        assert tmp_inst.exists()
        assert tmp_brk.exists()

        with open(tmp_inst, "r", encoding="utf-8") as f:
            inst_data = yaml.safe_load(f)
            assert len(inst_data["instruments"]) == 2

        with open(tmp_brk, "r", encoding="utf-8") as f:
            brk_data = yaml.safe_load(f)
            assert len(brk_data["brokers"]) == 3
