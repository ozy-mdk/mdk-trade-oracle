from mdk_trading_oracle.core.db import DuckDBManager
from mdk_trading_oracle.data.bronze import initialize_bronze_schema
from mdk_trading_oracle.data.gold import GoldFeatureEngineer, initialize_gold_schema
from mdk_trading_oracle.data.silver import SilverTransformer, initialize_silver_schema


def test_medallion_pipeline_in_memory():
    """Test full Bronze -> Silver -> Gold transformation flow on synthetic tick data."""
    db = DuckDBManager(in_memory=True)
    conn = db.get_connection()

    # 1. Initialize Bronze
    initialize_bronze_schema(db)
    initialize_silver_schema(db)
    initialize_gold_schema(db)

    # Insert synthetic Bronze trades
    conn.execute("""
        INSERT INTO bronze_raw_trades (trade_id, timestamp, symbol, price, volume, buyer_broker_id, seller_broker_id, raw_source)
        VALUES 
            ('t1', '2026-03-15 10:00:00', 'THYAO', 300.0, 1000.0, 'MLB', 'ISY', 'test'),
            ('t2', '2026-03-15 11:00:00', 'THYAO', 305.0, 2000.0, 'MLB', 'GAR', 'test'),
            ('t3', '2026-03-15 12:00:00', 'THYAO', 302.0, 500.0, 'YKR', 'MLB', 'test'),
            ('t4', '2026-03-15 14:00:00', 'AKBNK', 60.0, 5000.0, 'GAR', 'AKB', 'test');
    """)

    # 2. Test Silver Transformations
    silver = SilverTransformer(db)
    silver_res = silver.run_all()
    assert silver_res["status"] == "success"

    # Verify Silver Daily Broker Summary
    broker_summary = conn.execute("""
        SELECT broker_id, buy_volume, sell_volume, net_volume, net_flow_tl, buy_vwap
        FROM silver_daily_broker_summary
        WHERE symbol = 'THYAO' AND broker_id = 'MLB';
    """).fetchone()

    assert broker_summary is not None
    # MLB bought 1000 + 2000 = 3000, sold 500
    assert broker_summary[1] == 3000.0
    assert broker_summary[2] == 500.0
    assert broker_summary[3] == 2500.0
    # Buy turnover = 1000*300 + 2000*305 = 300,000 + 610,000 = 910,000. Buy VWAP = 910000 / 3000 = 303.333...
    assert abs(broker_summary[5] - (910000.0 / 3000.0)) < 1e-4

    # Verify Silver Market Daily
    market_daily = conn.execute("""
        SELECT symbol, open_price, high_price, low_price, close_price, total_volume, bofa_net_flow_tl
        FROM silver_market_daily
        WHERE symbol = 'THYAO';
    """).fetchone()

    assert market_daily is not None
    assert market_daily[1] == 300.0  # Open
    assert market_daily[2] == 305.0  # High
    assert market_daily[3] == 300.0  # Low
    assert market_daily[4] == 302.0  # Close
    assert market_daily[5] == 3500.0  # Total Volume
    # BofA net flow = 910,000 (buy) - 151,000 (sell: 500*302) = 759,000
    assert market_daily[6] == (910000.0 - 151000.0)

    # 3. Test Gold Feature Engineering
    gold = GoldFeatureEngineer(db)
    gold_res = gold.run_all()
    assert gold_res["status"] == "success"

    # Verify Gold Institutional Daily Signals
    gold_signals = conn.execute("""
        SELECT symbol, bofa_net_flow_tl, bofa_accum_5d_tl
        FROM gold_institutional_daily_signals
        WHERE symbol = 'THYAO';
    """).fetchone()

    assert gold_signals is not None
    assert gold_signals[1] == (910000.0 - 151000.0)
    assert gold_signals[2] == (910000.0 - 151000.0)
