"""Tests for resumable annual ZIP ingestion into DuckDB Bronze."""

import zipfile

from mdk_trading_oracle.core.db import DuckDBManager
from mdk_trading_oracle.data.bronze import BronzeIngestor, initialize_bronze_schema

HEADER = "symbol,signal_time_text,price,quantity,bidask,buyer,seller\n"


def test_zip_ingestor_preserves_mysql_parity_fields_and_is_resumable(tmp_path):
    archive_dir = tmp_path / "archives"
    archive_dir.mkdir()
    archive_path = archive_dir / "2026.zip"
    with zipfile.ZipFile(archive_path, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        archive.writestr(
            "2026/2026-01-02/THYAO.csv",
            HEADER
            + "THYAO,2026-01-02T09:55:01.123+0300,300.125000,10,,MLB,IYM\n"
            + "THYAO,2026-01-02T09:55:02.000+0300,300.250000,20,A,MLB,YKR\n",
        )
        archive.writestr(
            "2026/2026-01-02/AKBNK.csv",
            HEADER + "AKBNK,2026-01-02T10:00:00.000+0300,60.500000,100,,IYM,MLB",
        )
        archive.writestr("__MACOSX/2026/._THYAO.csv", "ignored")

    db = DuckDBManager(in_memory=True)
    initialize_bronze_schema(db)
    ingestor = BronzeIngestor(db)

    first = ingestor.ingest_bist_zip_archives(
        archive_dir=archive_dir,
        temp_dir=tmp_path / "temp",
        batch_mb=1,
    )
    second = ingestor.ingest_bist_zip_archives(
        archive_dir=archive_dir,
        temp_dir=tmp_path / "temp",
        batch_mb=1,
    )

    assert first["processed_files"] == 2
    assert first["rows_ingested"] == 3
    assert first["total_loaded_files"] == 2
    assert second["processed_files"] == 0
    assert second["rows_ingested"] == 0

    rows = db.get_connection().execute("""
        SELECT timestamp, symbol, price, volume, bidask,
               buyer_broker_id, seller_broker_id, raw_source
        FROM bronze_raw_trades
        ORDER BY timestamp;
    """).fetchall()
    assert rows[0][0].isoformat(sep=" ") == "2026-01-02 09:55:01.123000"
    assert f"{rows[0][2]:.6f}" == "300.125000"
    assert rows[0][3:] == (10, None, "MLB", "IYM", "2026.zip")
    assert rows[2][3:] == (100, None, "IYM", "MLB", "2026.zip")
