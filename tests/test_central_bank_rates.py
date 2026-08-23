"""Unit and integration tests for Central Bank (TCMB) interest rates ingestion & market sync."""

import tempfile
from datetime import date, timedelta
from pathlib import Path

import pandas as pd

from mdk_trading_oracle.core.db import DuckDBManager
from mdk_trading_oracle.data.bronze import BronzeIngestor, initialize_bronze_schema
from mdk_trading_oracle.data.silver import SilverTransformer, initialize_silver_schema


def test_central_bank_rates_ingestion_and_upsert():
    """Test reading Excel/CSV rate files, upserting into Bronze, and recalculating rate changes."""
    db = DuckDBManager(in_memory=True)
    conn = db.get_connection()
    initialize_bronze_schema(db)

    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp_path = Path(tmp_dir)

        # 1. Create a mock Excel file with historical rate data
        dates = [
            "2026-01-05",
            "2026-01-06",
            "2026-01-07",
            "2026-01-22",
            "2026-01-23",  # Rate cut from 38% to 37%
            "2026-01-26",
        ]
        rates = [38.0, 38.0, 38.0, 38.0, 37.0, 37.0]
        df_jan = pd.DataFrame({"Tarih": pd.to_datetime(dates), "1 Hafta Repo Faizi (%)": rates})
        jan_file = tmp_path / "tcmb_repo_jan_2026.xlsx"
        df_jan.to_excel(jan_file, index=False)

        ingestor = BronzeIngestor(db)

        # Ingest January file
        res = ingestor.ingest_central_bank_rates(file_path=jan_file, sync_market_dates=False)
        assert res["status"] == "success"
        assert res["rows_ingested"] == 6

        # Verify Bronze table
        rows = conn.execute("""
            SELECT rate_date, interest_rate, rate_change, is_rate_change_day, is_forward_filled, raw_source
            FROM bronze_central_bank_rates
            ORDER BY rate_date ASC;
        """).fetchall()

        assert len(rows) == 6
        assert rows[0][0] == date(2026, 1, 5)
        assert rows[0][1] == 38.0
        assert rows[0][2] == 0.0  # Initial rate change
        assert rows[0][3] is False
        assert rows[0][4] is False  # Not forward-filled

        # Verify rate change on 2026-01-23
        row_cut = [r for r in rows if r[0] == date(2026, 1, 23)][0]
        assert row_cut[1] == 37.0
        assert row_cut[2] == -1.0  # -1% drop
        assert row_cut[3] is True  # is_rate_change_day

        # 2. Test Idempotent Ingestion (Skipping unchanged file when force=False)
        res_repeat = ingestor.ingest_central_bank_rates(file_path=jan_file, force=False, sync_market_dates=False)
        assert res_repeat["rows_ingested"] == 6  # Single file explicitly targeted re-processes cleanly

        # 3. Create a second mock CSV file with February rate data (New monthly arrival)
        feb_dates = [
            "2026-02-02",
            "2026-02-03",
            "2026-02-19",
            "2026-02-20",  # Rate hike from 37% to 39.5%
            "2026-02-23",
        ]
        feb_rates = [37.0, 37.0, 37.0, 39.5, 39.5]
        df_feb = pd.DataFrame({"Date": pd.to_datetime(feb_dates), "Repo Rate": feb_rates})
        feb_file = tmp_path / "tcmb_repo_feb_2026.csv"
        df_feb.to_csv(feb_file, index=False)

        # Ingest February file
        res_feb = ingestor.ingest_central_bank_rates(file_path=feb_file, sync_market_dates=False)
        assert res_feb["status"] == "success"

        # Verify that total rows grew and Jan data was not lost
        total_rows = conn.execute("SELECT COUNT(*) FROM bronze_central_bank_rates;").fetchone()[0]
        assert total_rows == 11  # 6 from Jan + 5 from Feb

        feb_cut = conn.execute("""
            SELECT interest_rate, rate_change, is_rate_change_day 
            FROM bronze_central_bank_rates 
            WHERE rate_date = '2026-02-20';
        """).fetchone()
        assert feb_cut[0] == 39.5
        assert feb_cut[1] == 2.5
        assert feb_cut[2] is True


def test_forward_fill_market_synchronization():
    """Test forward-filling rates to keep in sync with market trading days when no new rate file arrived."""
    db = DuckDBManager(in_memory=True)
    conn = db.get_connection()
    initialize_bronze_schema(db)

    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp_path = Path(tmp_dir)

        # Base rate file ending on 2026-03-10
        base_dates = ["2026-03-06", "2026-03-09", "2026-03-10"]
        base_rates = [45.0, 45.0, 45.0]
        df_base = pd.DataFrame({"Tarih": pd.to_datetime(base_dates), "1 Hafta Repo Faizi (%)": base_rates})
        rate_file = tmp_path / "tcmb_base.xlsx"
        df_base.to_excel(rate_file, index=False)

        ingestor = BronzeIngestor(db)
        ingestor.ingest_central_bank_rates(file_path=rate_file, sync_market_dates=False)

        # Add market trades in bronze_raw_trades extending through 2026-03-15
        conn.execute("""
            INSERT INTO bronze_raw_trades (trade_id, timestamp, symbol, price, volume, buyer_broker_id, seller_broker_id, raw_source)
            VALUES 
                ('t1', '2026-03-09 10:00:00', 'THYAO', 300.0, 100.0, 'MLB', 'ISY', 'test'),
                ('t2', '2026-03-15 15:00:00', 'THYAO', 305.0, 100.0, 'MLB', 'GAR', 'test');
        """)

        # Execute Forward-Fill Synchronization
        sync_res = ingestor.sync_central_bank_rates_to_market()
        assert sync_res["status"] == "success"
        assert sync_res["forward_filled_count"] == 5  # March 11, 12, 13, 14, 15

        # Check forward-filled rows
        ffilled_rows = conn.execute("""
            SELECT rate_date, interest_rate, rate_change, is_forward_filled, raw_source
            FROM bronze_central_bank_rates
            WHERE rate_date > '2026-03-10'
            ORDER BY rate_date ASC;
        """).fetchall()

        assert len(ffilled_rows) == 5
        for r in ffilled_rows:
            assert r[1] == 45.0  # Carried forward previous rate
            assert r[2] == 0.0  # No rate change
            assert r[3] is True  # is_forward_filled = TRUE
            assert r[4] == "forward_fill_market_sync"

        # 4. Test Overwriting when newer official CBRT file arrives for those dates
        newer_dates = ["2026-03-11", "2026-03-12", "2026-03-13", "2026-03-16"]
        # Suppose on March 13 CBRT changed rate to 47.0%
        newer_rates = [45.0, 45.0, 47.0, 47.0]
        df_newer = pd.DataFrame({"Tarih": pd.to_datetime(newer_dates), "1 Hafta Repo Faizi (%)": newer_rates})
        newer_file = tmp_path / "tcmb_newer.xlsx"
        df_newer.to_excel(newer_file, index=False)

        ingestor.ingest_central_bank_rates(file_path=newer_file, sync_market_dates=False)

        # Verify that March 11, 12, 13 are now official (is_forward_filled = FALSE) and March 13 is 47.0%
        m13_row = conn.execute("""
            SELECT interest_rate, rate_change, is_rate_change_day, is_forward_filled 
            FROM bronze_central_bank_rates 
            WHERE rate_date = '2026-03-13';
        """).fetchone()

        assert m13_row[0] == 47.0
        assert m13_row[1] == 2.0
        assert m13_row[2] is True
        assert m13_row[3] is False  # Overwritten by official file


def test_silver_macro_rates_transformation():
    """Test Silver transformation generating daily macro rate features and rolling metrics."""
    db = DuckDBManager(in_memory=True)
    conn = db.get_connection()
    initialize_bronze_schema(db)
    initialize_silver_schema(db)

    # Insert synthetic Bronze rates
    base_date = date(2026, 1, 1)
    records = []
    current_rate = 40.0
    for i in range(45):
        d = base_date + timedelta(days=i)
        if i == 20:
            current_rate = 42.5  # Rate hike on day 20
        records.append(
            (
                d.strftime("%Y-%m-%d"),
                "1_week_repo",
                current_rate,
                2.5 if i == 20 else 0.0,
                True if i == 20 else False,
                False,
                "test_feed",
            )
        )

    conn.executemany(
        """
        INSERT INTO bronze_central_bank_rates (
            rate_date, rate_type, interest_rate, rate_change, is_rate_change_day, is_forward_filled, raw_source
        ) VALUES (?, ?, ?, ?, ?, ?, ?);
    """,
        records,
    )

    transformer = SilverTransformer(db)
    res = transformer.transform_daily_macro_rates()
    assert res["status"] == "success"
    assert res["rows"] == 45

    # Verify columns in silver_daily_macro_rates
    sample = conn.execute("""
        SELECT trade_date, interest_rate, rate_change, is_rate_change_day, days_since_last_rate_change, rolling_30d_rate_mean, is_forward_filled, rate_change_decay_bps
        FROM silver_daily_macro_rates
        ORDER BY trade_date ASC;
    """).fetchall()

    assert len(sample) == 45
    # Day 20 (Rate hike date: +2.5% = +250 bps, day 0)
    day_20 = sample[20]
    assert day_20[1] == 42.5
    assert day_20[2] == 2.5
    assert day_20[3] is True
    assert day_20[4] == 0  # 0 days since rate change on decision day itself
    assert day_20[7] == 250.0  # +250 bps / max(1, 0) = 250.0

    # Day 21 (1 day after rate hike: day 1 denominator is 1)
    day_21 = sample[21]
    assert day_21[4] == 1
    assert day_21[7] == 250.0  # +250 bps / max(1, 1) = 250.0

    # Day 25 (5 days after rate hike: day 5 denominator is 5)
    day_25 = sample[25]
    assert day_25[1] == 42.5
    assert day_25[2] == 0.0
    assert day_25[3] is False
    assert day_25[4] == 5  # 5 days since last rate change
    assert day_25[5] > 40.0  # Rolling 30d mean includes higher rates
    assert day_25[7] == 50.0  # +250 bps / 5 = 50.0
