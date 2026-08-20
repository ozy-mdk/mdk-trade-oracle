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
    assert summary["distinct_brokers"] == 3

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


def test_raw_data_inspector_reads_source_csv_schema(tmp_path):
    """Raw CSV discovery maps source feed columns before Bronze ingestion."""
    csv_file = tmp_path / "trades.csv"
    csv_file.write_text(
        "symbol,signal_time_text,price,quantity,bidask,buyer,seller\n"
        "THYAO,2026-03-02T10:00:00.000+0300,300.0,1000,,MLB,ISY\n"
        "AKBNK,2026-03-03T11:00:00.000+0300,60.0,2000,,YKR,MLB\n",
        encoding="utf-8",
    )
    inspector = RawDataInspector(raw_glob=csv_file.as_posix())

    summary = inspector.inspect_dataset_summary()
    instruments = inspector.discover_instruments()
    brokers = inspector.discover_brokers()

    assert summary["total_trades"] == 2
    assert summary["min_date"] == "2026-03-02"
    assert summary["max_date"] == "2026-03-03"
    assert summary["trading_days"] == 2
    assert summary["distinct_symbols"] == 2
    assert summary["distinct_brokers"] == 3
    assert {item["symbol"] for item in instruments} == {"THYAO", "AKBNK"}
    assert {item["code"] for item in brokers} == {"MLB", "ISY", "YKR"}


def test_broker_discovery_excludes_missing_codes():
    """Null buyer or seller fields must not become catalog broker entries."""
    inspector = RawDataInspector()
    mem_conn = duckdb.connect(":memory:")
    mem_conn.execute("""
        CREATE TABLE bronze_raw_trades (
            timestamp TIMESTAMP,
            symbol VARCHAR,
            price DOUBLE,
            volume BIGINT,
            buyer_broker_id VARCHAR,
            seller_broker_id VARCHAR
        );
        INSERT INTO bronze_raw_trades VALUES
            ('2026-03-02 10:00:00', 'THYAO', 300.0, 1000, NULL, 'MLB'),
            ('2026-03-02 10:01:00', 'THYAO', 301.0, 500, 'IYM', NULL),
            ('2026-03-02 10:02:00', 'THYAO', 302.0, 250, '', 'YKR');
    """)
    inspector._get_read_connection = lambda: mem_conn

    codes = {item["code"] for item in inspector.discover_brokers()}

    assert codes == {"MLB", "IYM", "YKR"}
