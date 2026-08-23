"""Unit and integration tests for Corporate Actions and Share Adjustment Engine (Pay Düzeltme)."""

import datetime as dt
from decimal import Decimal

import pytest

from mdk_trading_oracle.core.db import DuckDBManager
from mdk_trading_oracle.data.bronze import BronzeIngestor, initialize_bronze_schema
from mdk_trading_oracle.data.silver import (
    Action,
    CorporateActionEngine,
    SilverTransformer,
    initialize_silver_schema,
)


def test_single_split_adjustment():
    """Test standard single 2:1 stock split."""
    actions = [
        Action(
            action_date=dt.date(2023, 8, 25),
            symbol="ASELS",
            target_symbol=None,
            multiplier=Decimal("2.0"),
            note="%100 bedelsiz",
        )
    ]
    periods = CorporateActionEngine.build_adjustment_periods(actions)
    assert len(periods) == 2

    # Period 1: Prior to split
    p1 = periods[0]
    assert p1.source_symbol == "ASELS"
    assert p1.effective_from == dt.date(1900, 1, 1)
    assert p1.effective_to == dt.date(2023, 8, 24)
    assert p1.canonical_symbol == "ASELS"
    assert p1.quantity_factor == Decimal("2.0")
    assert not p1.has_unresolved_paid_action

    # Period 2: On and after split
    p2 = periods[1]
    assert p2.source_symbol == "ASELS"
    assert p2.effective_from == dt.date(2023, 8, 25)
    assert p2.effective_to == dt.date(9999, 12, 31)
    assert p2.canonical_symbol == "ASELS"
    assert p2.quantity_factor == Decimal("1.0")
    assert not p2.has_unresolved_paid_action


def test_multi_split_compounding_sasa():
    """Test SASA's 3 historical bonus issues compounding to 36.8x."""
    actions = [
        Action(
            action_date=dt.date(2022, 5, 5),
            symbol="SASA",
            target_symbol=None,
            multiplier=Decimal("2.0"),
            note="%100 kar payından bedelsiz",
        ),
        Action(
            action_date=dt.date(2023, 5, 23),
            symbol="SASA",
            target_symbol=None,
            multiplier=Decimal("2.3"),
            note="%130 bedelsiz",
        ),
        Action(
            action_date=dt.date(2024, 8, 12),
            symbol="SASA",
            target_symbol=None,
            multiplier=Decimal("8.0"),
            note="%700 bedelsiz",
        ),
    ]
    periods = CorporateActionEngine.build_adjustment_periods(actions)
    assert len(periods) == 4

    # Period 1: 1900-01-01 to 2022-05-04 -> factor = 2.0 * 2.3 * 8.0 = 36.8
    assert periods[0].effective_from == dt.date(1900, 1, 1)
    assert periods[0].effective_to == dt.date(2022, 5, 4)
    assert periods[0].quantity_factor == Decimal("36.8")

    # Period 2: 2022-05-05 to 2023-05-22 -> factor = 2.3 * 8.0 = 18.4
    assert periods[1].effective_from == dt.date(2022, 5, 5)
    assert periods[1].effective_to == dt.date(2023, 5, 22)
    assert periods[1].quantity_factor == Decimal("18.4")

    # Period 3: 2023-05-23 to 2024-08-11 -> factor = 8.0
    assert periods[2].effective_from == dt.date(2023, 5, 23)
    assert periods[2].effective_to == dt.date(2024, 8, 11)
    assert periods[2].quantity_factor == Decimal("8.0")

    # Period 4: 2024-08-12 to 9999-12-31 -> factor = 1.0
    assert periods[3].effective_from == dt.date(2024, 8, 12)
    assert periods[3].effective_to == dt.date(9999, 12, 31)
    assert periods[3].quantity_factor == Decimal("1.0")


def test_ticker_rename_chain_kozal_tralt():
    """Test symbol change KOZAL -> TRALT with prior 21x split on KOZAL."""
    actions = [
        Action(
            action_date=dt.date(2023, 2, 17),
            symbol="KOZAL",
            target_symbol=None,
            multiplier=Decimal("21.0"),
            note="%2000 bedelsiz",
        ),
        Action(
            action_date=dt.date(2025, 11, 24),
            symbol="KOZAL",
            target_symbol="TRALT",
            multiplier=Decimal("1.0"),
            note="Borsa İstanbul işlem kodu değişikliği",
        ),
    ]
    periods = CorporateActionEngine.build_adjustment_periods(actions)

    # KOZAL periods
    kozal_periods = [p for p in periods if p.source_symbol == "KOZAL"]
    assert len(kozal_periods) == 3

    assert kozal_periods[0].effective_to == dt.date(2023, 2, 16)
    assert kozal_periods[0].canonical_symbol == "TRALT"
    assert kozal_periods[0].quantity_factor == Decimal("21.0")

    assert kozal_periods[1].effective_from == dt.date(2023, 2, 17)
    assert kozal_periods[1].effective_to == dt.date(2025, 11, 23)
    assert kozal_periods[1].canonical_symbol == "TRALT"
    assert kozal_periods[1].quantity_factor == Decimal("1.0")

    # TRALT target symbol period
    tralt_periods = [p for p in periods if p.source_symbol == "TRALT"]
    assert len(tralt_periods) == 1
    assert tralt_periods[0].canonical_symbol == "TRALT"
    assert tralt_periods[0].quantity_factor == Decimal("1.0")


def test_unresolved_rights_issue_flag():
    """Test that paid action notes flag has_unresolved_paid_action correctly."""
    actions = [
        Action(
            action_date=dt.date(2022, 10, 12),
            symbol="HEKTS",
            target_symbol=None,
            multiplier=Decimal("1.4418604"),
            note="%44.18604 bedelsiz kısmı; ayrıca %150 bedelli uygulanmadı",
        ),
        Action(
            action_date=dt.date(2024, 9, 18),
            symbol="HEKTS",
            target_symbol=None,
            multiplier=Decimal("1.0"),
            note="%233.20158 bedelli; rüçhan kullanım fiyatı 1 TL",
        ),
    ]
    periods = CorporateActionEngine.build_adjustment_periods(actions)
    assert len(periods) == 3

    # Before 2022-10-12: future events contain bedelli -> flagged True
    assert periods[0].has_unresolved_paid_action is True
    # Between 2022-10-12 and 2024-09-17: future event on 2024-09-18 contains bedelli -> True
    assert periods[1].has_unresolved_paid_action is True
    # After 2024-09-18: no future bedelli -> False
    assert periods[2].has_unresolved_paid_action is False


def test_monetary_invariance():
    """Verify that turnover (TL) is invariant to adjustment multipliers."""
    raw_price = Decimal("120.0")
    raw_volume = Decimal("50000")
    factor = Decimal("2.0")

    raw_turnover = raw_price * raw_volume
    adj_price = raw_price / factor
    adj_volume = raw_volume * factor
    adj_turnover = adj_price * adj_volume

    assert raw_turnover == Decimal("6000000.0")
    assert adj_turnover == raw_turnover


def test_duckdb_corporate_actions_pipeline_integration():
    """Integration test with in-memory DuckDB validating schema, ingestion, and Silver periods."""
    db = DuckDBManager(":memory:")
    initialize_bronze_schema(db)
    initialize_silver_schema(db)

    # 1. Ingest Bronze
    ingestor = BronzeIngestor(db)
    ingest_res = ingestor.ingest_corporate_actions()
    assert ingest_res["status"] == "success"
    assert ingest_res["total_rows"] == 23

    # 2. Build Silver periods
    transformer = SilverTransformer(db)
    res_actions = transformer.transform_corporate_action_adjustment_periods()
    assert res_actions["status"] == "success"

    conn = db.get_connection()
    period_count = conn.execute("SELECT COUNT(*) FROM silver_corporate_action_adjustment_periods;").fetchone()[0]
    assert period_count == 42

    # Verify SASA periods in DuckDB
    sasa_p1 = conn.execute("""
        SELECT quantity_factor, canonical_symbol 
        FROM silver_corporate_action_adjustment_periods 
        WHERE source_symbol = 'SASA' AND effective_from = '1900-01-01';
    """).fetchone()
    assert pytest.approx(sasa_p1[0], 0.0001) == 36.8
    assert sasa_p1[1] == "SASA"

    # Verify KOZAL canonical mapping in DuckDB
    kozal_p1 = conn.execute("""
        SELECT quantity_factor, canonical_symbol 
        FROM silver_corporate_action_adjustment_periods 
        WHERE source_symbol = 'KOZAL' AND effective_from = '1900-01-01';
    """).fetchone()
    assert pytest.approx(kozal_p1[0], 0.0001) == 21.0
    assert kozal_p1[1] == "TRALT"
