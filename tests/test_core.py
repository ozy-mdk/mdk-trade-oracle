"""Unit tests for core configuration, database, and domain types."""

from datetime import datetime

from mdk_trading_oracle.core.config import get_settings
from mdk_trading_oracle.core.db import DuckDBManager
from mdk_trading_oracle.core.types import (
    RawTradeRecord,
)


def test_settings_load():
    """Verify settings initialize and dynamic paths are valid."""
    settings = get_settings()
    assert settings.default_market == "BIST"
    assert settings.primary_institution in ["MLB", "BOFA"]
    assert settings.project_root.exists()
    assert settings.data_dir.exists()
    assert settings.raw_data_dir.exists()
    assert settings.database_dir.exists()
    assert settings.duckdb_path == settings.database_path
    assert len(settings.get_brokers()) > 0
    assert len(settings.get_instruments()) > 0


def test_duckdb_schema_initialization():
    """Test DuckDB in-memory initialization and table creation for Bronze layer."""
    db = DuckDBManager(in_memory=True)
    db.initialize_schema()

    conn = db.get_connection()
    tables = [row[0] for row in conn.execute("SHOW TABLES;").fetchall()]

    expected_tables = [
        "bronze_raw_trades",
        "bronze_brokers",
        "bronze_instruments",
    ]
    for tbl in expected_tables:
        assert tbl in tables


def test_domain_types():
    """Test Pydantic model serialization and validation."""
    trade = RawTradeRecord(
        trade_id="TRD_001",
        timestamp=datetime.now(),
        symbol="AKBNK",
        price=58.50,
        volume=1000.0,
        buyer_broker_id="BOFA",
        seller_broker_id="ISY",
        raw_source="test.csv",
    )
    assert trade.symbol == "AKBNK"
    assert trade.buyer_broker_id == "BOFA"


def test_turkish_time_utilities():
    """Test Turkish Time (Europe/Istanbul / UTC+3) utilities and DuckDB timezone configuration."""
    from mdk_trading_oracle.core.time import now_turkey, now_turkey_naive, today_turkey
    
    now_trt = now_turkey()
    assert now_trt.tzinfo is not None
    assert str(now_trt.tzinfo) == "Europe/Istanbul"
    
    now_naive = now_turkey_naive()
    assert now_naive.tzinfo is None
    
    today = today_turkey()
    assert today == now_trt.date()
    
    # Test DuckDB timezone setting
    db = DuckDBManager(in_memory=True)
    conn = db.get_connection()
    tz_setting = conn.execute("SELECT current_setting('TimeZone');").fetchone()[0]
    assert tz_setting == "Europe/Istanbul"

