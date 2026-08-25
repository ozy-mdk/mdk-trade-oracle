"""Unit tests for BIST 30 Index Membership, Historical Changes, and Dynamic Constituent Synchronization."""

import tempfile
from pathlib import Path

import pandas as pd
import pytest

from mdk_trading_oracle.core.db import DuckDBManager
from mdk_trading_oracle.data.bronze.ingestor import BronzeIngestor
from mdk_trading_oracle.data.bronze.schema import initialize_bronze_schema
from mdk_trading_oracle.models.stock_reaction.orchestrator import StockReactionOrchestrator


@pytest.fixture
def temp_db(tmp_path):
    """Create an isolated test DuckDB instance."""
    db_file = tmp_path / "test_bist30.duckdb"
    db = DuckDBManager(db_path=db_file)
    initialize_bronze_schema(db)
    yield db
    db.close()


def test_bronze_bist30_schema_creation(temp_db):
    """Verify Bronze BIST 30 tables are created with proper schemas."""
    conn = temp_db.get_connection()
    tables = [
        r[0]
        for r in conn.execute(
            "SELECT table_name FROM information_schema.tables WHERE table_schema = 'main';"
        ).fetchall()
    ]
    assert "bronze_bist30_membership" in tables
    assert "bronze_bist30_changes" in tables
    assert "bronze_bist30_stock_periods" in tables


def test_bist30_ingestion_and_point_in_time_query(temp_db):
    """Verify BIST 30 Excel ingestion, active flag, and point-in-time constituent resolution."""
    raw_file = (
        Path.home()
        / "data"
        / "mdk_oracle"
        / "00_raw_data"
        / "bist_30_list_with_changes"
        / "BIST30_uyelik_ve_degisim_tarihi.xlsx"
    )

    ingestor = BronzeIngestor(temp_db)

    if raw_file.exists():
        res = ingestor.ingest_bist30_membership(file_path=raw_file, force=True)
        assert res["status"] == "success"
        assert res["membership_rows"] >= 500
        assert res["changes_rows"] >= 15
        assert res["periods_rows"] >= 45
        assert res["active_symbols_count"] == 30

        # 1. Test active constituents
        active_symbols = ingestor.get_bist30_symbols(active_only=True)
        assert len(active_symbols) == 30
        assert "TRALT" in active_symbols
        assert "DSTKF" in active_symbols
        assert "MGROS" in active_symbols
        assert "GUBRF" in active_symbols
        assert "ARCLK" not in active_symbols
        assert "HALKB" not in active_symbols

        # 2. Test historical point-in-time: 2022-02-01
        pit_2022 = ingestor.get_bist30_symbols(as_of_date="2022-02-01")
        assert len(pit_2022) == 30
        assert "ARCLK" in pit_2022
        assert "DSTKF" not in pit_2022

        # 3. Test historical point-in-time: 2024-05-01
        pit_2024 = ingestor.get_bist30_symbols(as_of_date="2024-05-01")
        assert len(pit_2024) == 30
        assert "BRSAN" in pit_2024
        assert "ARCLK" not in pit_2024
    else:
        # Fallback synthetic test
        with tempfile.TemporaryDirectory() as td:
            csv_path = Path(td) / "bist30_dummy.csv"
            df = pd.DataFrame(
                [
                    {"symbol": "AKBNK", "start_date": "2022-01-01", "end_date": None, "status": "Aktif"},
                    {"symbol": "GARAN", "start_date": "2022-01-01", "end_date": None, "status": "Aktif"},
                    {"symbol": "ARCLK", "start_date": "2022-01-01", "end_date": "2024-03-31", "status": "Sona erdi"},
                ]
            )
            df.to_csv(csv_path, index=False)
            res = ingestor.ingest_bist30_membership(file_path=csv_path, force=True)
            assert res["status"] == "success"

            active = ingestor.get_bist30_symbols(active_only=True)
            assert "AKBNK" in active
            assert "GARAN" in active


def test_orchestrator_symbol_resolution_with_db(temp_db):
    """Verify StockReactionOrchestrator resolves symbols dynamically from Bronze BIST 30 membership."""
    raw_file = (
        Path.home()
        / "data"
        / "mdk_oracle"
        / "00_raw_data"
        / "bist_30_list_with_changes"
        / "BIST30_uyelik_ve_degisim_tarihi.xlsx"
    )

    ingestor = BronzeIngestor(temp_db)
    if raw_file.exists():
        ingestor.ingest_bist30_membership(file_path=raw_file, force=True)

    orch = StockReactionOrchestrator(db=temp_db, symbols=None)
    assert len(orch.symbols) == 30
    assert "TRALT" in orch.symbols
    assert "DSTKF" in orch.symbols
    assert "THYAO" in orch.symbols
