"""Unit tests for core configuration, database, and types."""

from datetime import date, datetime
import pytest
from mdk_trading_oracle.core.config import get_settings
from mdk_trading_oracle.core.db import DuckDBManager
from mdk_trading_oracle.core.types import (
    OracleDecisionSignal,
    RawTradeRecord,
    SignalType,
    TradeSide,
)


def test_settings_load():
    """Verify settings initialize and dynamic paths are valid."""
    settings = get_settings()
    assert settings.default_market == "BIST"
    assert settings.primary_institution == "BOFA"
    assert settings.project_root.exists()
    assert len(settings.get_brokers()) > 0
    assert len(settings.get_instruments()) > 0


def test_duckdb_schema_initialization():
    """Test DuckDB in-memory initialization and table creation."""
    db = DuckDBManager(in_memory=True)
    db.initialize_schema()

    conn = db.get_connection()
    tables = [row[0] for row in conn.execute("SHOW TABLES;").fetchall()]

    expected_tables = [
        "bronze_raw_trades",
        "silver_broker_transactions",
        "silver_daily_broker_summary",
        "silver_brokers",
        "silver_instruments",
        "gold_bofa_flow_metrics",
        "oracle_decision_signals",
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

    signal = OracleDecisionSignal(
        signal_id="SIG_001",
        date_val=date(2026, 8, 18),
        symbol="AKBNK",
        signal=SignalType.STRONG_BUY,
        confidence=0.85,
        bofa_net_tl=25_000_000.0,
        bofa_net_share=0.22,
        bofa_flow_zscore=2.1,
        summary="STRONG_BUY signal on AKBNK",
        reasons=["High institutional net inflow", "Z-score > 2.0"],
    )
    assert signal.signal == SignalType.STRONG_BUY
    assert signal.confidence == 0.85
