"""Tests for Medallion Lakehouse layers (Bronze, Silver, Gold) and MedallionPipeline."""

import tempfile
from pathlib import Path

from mdk_trading_oracle.core.db import DuckDBManager
from mdk_trading_oracle.data.bronze import BronzeIngestor, initialize_bronze_schema
from mdk_trading_oracle.data.gold import GoldFeatureEngineer, initialize_gold_schema
from mdk_trading_oracle.data.pipeline import MedallionPipeline
from mdk_trading_oracle.data.silver import SilverTransformer, initialize_silver_schema


def test_medallion_pipeline_in_memory():
    """Test full Bronze -> Silver -> Gold transformation flow on synthetic tick data across all 6 Silver tables."""
    db = DuckDBManager(in_memory=True)
    conn = db.get_connection()

    # 1. Initialize Bronze, Silver, Gold Schemas
    initialize_bronze_schema(db)
    initialize_silver_schema(db)
    initialize_gold_schema(db)

    # Insert synthetic Bronze trades across various intraday times (Turkish Time TRT)
    conn.execute("""
        INSERT INTO bronze_raw_trades (trade_id, timestamp, symbol, price, volume, buyer_broker_id, seller_broker_id, raw_source)
        VALUES 
            ('t1', '2026-03-16 10:15:00', 'THYAO', 300.0, 1000.0, 'MLB', 'ISY', 'test'),  -- Window 1: Day Start (09:55 - 10:30)
            ('t2', '2026-03-16 11:00:00', 'THYAO', 305.0, 2000.0, 'MLB', 'GAR', 'test'),  -- Window 2: First Reaction (10:30 - 11:30)
            ('t3', '2026-03-16 13:00:00', 'THYAO', 302.0, 500.0, 'YKR', 'MLB', 'test'),   -- Window 3: Midday Follow-up (11:30 - 14:30)
            ('t4', '2026-03-16 15:15:00', 'AKBNK', 60.0, 5000.0, 'GAR', 'AKB', 'test'),   -- Window 4: Afternoon Reaction (14:30 - 16:00)
            ('t5', '2026-03-16 17:30:00', 'AKBNK', 62.0, 3000.0, 'MLB', 'YKR', 'test');   -- Window 5: Closing Session (16:00 - 18:15)
    """)

    # 2. Execute Silver Transformations
    silver = SilverTransformer(db)
    silver_res = silver.run_all()
    assert silver_res["status"] == "success"
    assert "silver_daily_macro_rates" in silver_res

    # --- Verify Table 1: silver_daily_broker_summary ---
    broker_summary = conn.execute("""
        SELECT broker_id, sector, buy_volume, sell_volume, net_volume, net_flow_tl, buy_vwap, broker_symbol_turnover_share
        FROM silver_daily_broker_summary
        WHERE symbol = 'THYAO' AND broker_id = 'MLB';
    """).fetchone()

    assert broker_summary is not None
    assert broker_summary[1] == "Transportation"
    assert broker_summary[2] == 3000.0  # Buy Vol: 1000 + 2000
    assert broker_summary[3] == 500.0  # Sell Vol: 500
    assert broker_summary[4] == 2500.0  # Net Vol: 3000 - 500
    # Buy turnover = 1000*300 + 2000*305 = 910,000. Buy VWAP = 910000 / 3000 = 303.333...
    assert abs(broker_summary[6] - (910000.0 / 3000.0)) < 1e-4

    # --- Verify Table 2: silver_daily_broker_overview ---
    bofa_overview = conn.execute("""
        SELECT broker_id, is_primary_target, total_buy_turnover_tl, total_sell_turnover_tl, net_flow_tl, 
               market_turnover_rank, is_top_5_broker, is_monday
        FROM silver_daily_broker_overview
        WHERE broker_id = 'MLB';
    """).fetchone()

    assert bofa_overview is not None
    assert bofa_overview[1] is True  # is_primary_target
    # BofA bought: 910,000 (THYAO) + 186,000 (AKBNK) = 1,096,000
    assert bofa_overview[2] == (910000.0 + 186000.0)
    # BofA sold: 151,000 (THYAO)
    assert bofa_overview[3] == 151000.0
    assert bofa_overview[4] == (1096000.0 - 151000.0)
    assert bofa_overview[5] == 1  # BofA is #1 rank in turnover
    assert bofa_overview[6] is True  # is_top_5_broker
    assert bofa_overview[7] is True  # 2026-03-16 is a Monday!

    # --- Verify Table 3: silver_daily_stock_summary ---
    stock_summary = conn.execute("""
        SELECT symbol, sector, open_price, high_price, low_price, close_price, total_volume, 
               top_buyer_broker_id, bofa_net_flow_tl, top_5_concentration_ratio
        FROM silver_daily_stock_summary
        WHERE symbol = 'THYAO';
    """).fetchone()

    assert stock_summary is not None
    assert stock_summary[1] == "Transportation"
    assert stock_summary[2] == 300.0  # Open
    assert stock_summary[3] == 305.0  # High
    assert stock_summary[4] == 300.0  # Low
    assert stock_summary[5] == 302.0  # Close
    assert stock_summary[6] == 3500.0  # Total Volume
    assert stock_summary[7] == "MLB"  # Top Buyer Broker
    assert stock_summary[8] == 759000.0  # BofA Net Flow TL
    assert stock_summary[9] == 1.0  # CR5 is 100% since all 3 brokers <= 5

    # --- Verify Table 4: silver_daily_sector_summary ---
    sector_summary = conn.execute("""
        SELECT sector, broker_id, buy_turnover_tl, sell_turnover_tl, net_flow_tl, active_symbols_count
        FROM silver_daily_sector_summary
        WHERE sector = 'Transportation' AND broker_id = 'MLB';
    """).fetchone()

    assert sector_summary is not None
    assert sector_summary[2] == 910000.0
    assert sector_summary[3] == 151000.0
    assert sector_summary[4] == 759000.0
    assert sector_summary[5] == 1

    # --- Verify Table 5: silver_intraday_broker_window_summary ---
    intraday_bofa = conn.execute("""
        SELECT window_name, window_order, buy_volume, sell_volume, net_flow_tl
        FROM silver_intraday_broker_window_summary
        WHERE symbol = 'THYAO' AND broker_id = 'MLB'
        ORDER BY window_order ASC;
    """).fetchall()

    assert len(intraday_bofa) == 3  # Day Start (buy 1000), First Reaction (buy 2000), Midday (sell 500)
    # Window 1: Day Start window (10:15)
    assert intraday_bofa[0][0] == "day_start"
    assert intraday_bofa[0][2] == 1000.0
    # Window 2: First Reaction window (11:00)
    assert intraday_bofa[1][0] == "first_reaction"
    assert intraday_bofa[1][2] == 2000.0
    # Window 3: Midday Follow-up window (13:00)
    assert intraday_bofa[2][0] == "midday_followup"
    assert intraday_bofa[2][3] == 500.0

    # --- Verify Table 6: silver_intraday_sector_window_summary ---
    intraday_sector = conn.execute("""
        SELECT sector, window_name, net_flow_tl
        FROM silver_intraday_sector_window_summary
        WHERE sector = 'Transportation' AND broker_id = 'MLB' AND window_name = 'day_start';
    """).fetchone()

    assert intraday_sector is not None
    assert intraday_sector[2] == 300000.0  # 1000 * 300.0

    # 3. Test Gold Layer compatibility
    gold = GoldFeatureEngineer(db)
    gold_res = gold.run_all()
    assert gold_res["status"] == "success"

    gold_signals = conn.execute("""
        SELECT symbol, bofa_net_flow_tl, bofa_accum_5d_tl
        FROM gold_institutional_daily_signals
        WHERE symbol = 'THYAO';
    """).fetchone()

    assert gold_signals is not None
    assert gold_signals[1] == 759000.0
    assert gold_signals[2] == 759000.0


def test_bronze_incremental_ingestion_and_logging():
    """Test that BronzeIngestor detects new files, avoids duplicate loading, and tracks logs."""
    db = DuckDBManager(in_memory=True)
    conn = db.get_connection()
    initialize_bronze_schema(db)
    ingestor = BronzeIngestor(db)

    with tempfile.TemporaryDirectory() as tmpdir:
        tmp_path = Path(tmpdir)
        day1_dir = tmp_path / "2026/03_march/raw_csv/2026-03-02"
        day1_dir.mkdir(parents=True, exist_ok=True)

        csv1 = day1_dir / "THYAO.csv"
        csv1.write_text(
            "symbol,signal_time_text,price,quantity,buyer,seller\n"
            "THYAO,2026-03-02 08:00:00,300,100,MLB,ISY\n"
            "THYAO,2026-03-02 08:05:00,301,200,MLB,GAR\n"
        )

        # 1. First Ingestion: Ingests 1 file, 2 rows
        res1 = ingestor.ingest_incremental(tmp_path)
        assert res1["status"] == "success"
        assert res1["pending_files"] == 1
        assert conn.execute("SELECT COUNT(*) FROM bronze_raw_trades;").fetchone()[0] == 2
        assert conn.execute("SELECT COUNT(*) FROM bronze_ingestion_log;").fetchone()[0] == 1

        # 2. Second Ingestion (No changes): Skips ingestion, 0 duplicates
        res2 = ingestor.ingest_incremental(tmp_path)
        assert res2["status"] == "up_to_date"
        assert res2["pending_files"] == 0
        assert conn.execute("SELECT COUNT(*) FROM bronze_raw_trades;").fetchone()[0] == 2

        # 3. Third Ingestion (Add a new file): Only ingests the new file
        csv2 = day1_dir / "AKBNK.csv"
        csv2.write_text(
            "symbol,signal_time_text,price,quantity,buyer,seller\nAKBNK,2026-03-02 08:10:00,60,500,GAR,MLB\n"
        )

        res3 = ingestor.ingest_incremental(tmp_path)
        assert res3["status"] == "success"
        assert res3["pending_files"] == 1
        assert conn.execute("SELECT COUNT(*) FROM bronze_raw_trades;").fetchone()[0] == 3
        assert conn.execute("SELECT COUNT(*) FROM bronze_ingestion_log;").fetchone()[0] == 2


def test_bronze_selective_date_partition_update():
    """Test selective atomic deletion and re-ingestion of a single date partition."""
    db = DuckDBManager(in_memory=True)
    conn = db.get_connection()
    initialize_bronze_schema(db)
    ingestor = BronzeIngestor(db)

    with tempfile.TemporaryDirectory() as tmpdir:
        tmp_path = Path(tmpdir)
        day1_dir = tmp_path / "2026/03_march/raw_csv/2026-03-02"
        day2_dir = tmp_path / "2026/03_march/raw_csv/2026-03-03"
        day1_dir.mkdir(parents=True, exist_ok=True)
        day2_dir.mkdir(parents=True, exist_ok=True)

        (day1_dir / "THYAO.csv").write_text(
            "symbol,signal_time_text,price,quantity,buyer,seller\nTHYAO,2026-03-02 08:00:00,300,100,MLB,ISY\n"
        )
        (day2_dir / "THYAO.csv").write_text(
            "symbol,signal_time_text,price,quantity,buyer,seller\nTHYAO,2026-03-03 08:00:00,310,500,MLB,ISY\n"
        )

        # Ingest both days
        ingestor.ingest_incremental(tmp_path)
        assert conn.execute("SELECT COUNT(*) FROM bronze_raw_trades;").fetchone()[0] == 2

        # Update Day 1 data with 3 trades instead of 1
        (day1_dir / "THYAO.csv").write_text(
            "symbol,signal_time_text,price,quantity,buyer,seller\n"
            "THYAO,2026-03-02 08:00:00,300,100,MLB,ISY\n"
            "THYAO,2026-03-02 08:10:00,302,200,MLB,ISY\n"
            "THYAO,2026-03-02 08:20:00,304,300,MLB,ISY\n"
        )

        # Ingest ONLY date 2026-03-02
        date_res = ingestor.ingest_date("2026-03-02", search_dir=tmp_path)
        assert date_res["status"] == "success"
        assert date_res["deleted_previous_trades"] == 1
        assert date_res["new_trades_ingested"] == 3

        # Verify Day 2 remains intact (1 trade), Day 1 now has 3 trades -> Total 4 trades
        assert (
            conn.execute(
                "SELECT COUNT(*) FROM bronze_raw_trades WHERE CAST(timestamp AS DATE) = '2026-03-02';"
            ).fetchone()[0]
            == 3
        )
        assert (
            conn.execute(
                "SELECT COUNT(*) FROM bronze_raw_trades WHERE CAST(timestamp AS DATE) = '2026-03-03';"
            ).fetchone()[0]
            == 1
        )
        assert conn.execute("SELECT COUNT(*) FROM bronze_raw_trades;").fetchone()[0] == 4


def test_pipeline_dag_resolution():
    """Test MedallionPipeline DAG layer resolution."""
    pipeline = MedallionPipeline(DuckDBManager(in_memory=True))

    assert pipeline._resolve_layers("catalog") == ["catalog"]
    assert pipeline._resolve_layers("bronze") == ["bronze"]
    assert pipeline._resolve_layers("silver") == ["bronze", "silver"]
    assert pipeline._resolve_layers("gold") == ["bronze", "silver", "gold"]
    assert pipeline._resolve_layers("all") == ["bronze", "silver", "gold"]
    assert pipeline._resolve_layers("silver", resolve_dependencies=False) == ["silver"]
