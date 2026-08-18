"""Realistic synthetic data generator for BIST trades with simulated BofA flow."""

import csv
from datetime import datetime, timedelta
import random
import uuid
from pathlib import Path

# Paths
ROOT = Path(__file__).resolve().parents[1]
BRONZE_DIR = ROOT / "data" / "01_bronze"
BRONZE_DIR.mkdir(parents=True, exist_ok=True)

SYMBOLS = ["AKBNK", "GARAN", "ISCTR", "YKBNK", "THYAO", "TUPRS", "EREGL"]
BROKERS = ["BOFA", "QNB", "YKR", "ISY", "GAR", "AKY", "TAC"]

BASE_PRICES = {
    "AKBNK": 58.50,
    "GARAN": 112.00,
    "ISCTR": 13.80,
    "YKBNK": 31.20,
    "THYAO": 310.00,
    "TUPRS": 165.00,
    "EREGL": 51.00,
}


def generate_mock_trades(num_days: int = 30, trades_per_day: int = 400):
    """Generate realistic synthetic trade records for BIST symbols."""
    random.seed(42)
    start_date = datetime.now().date() - timedelta(days=num_days)

    rows = []
    current_prices = {k: v for k, v in BASE_PRICES.items()}

    for day_offset in range(num_days):
        trade_date = start_date + timedelta(days=day_offset)
        # Skip weekends
        if trade_date.weekday() >= 5:
            continue

        # Simulate BofA sentiment for this day (bullish, bearish, or neutral)
        bofa_bias = random.choice(["bullish", "bearish", "neutral", "neutral"])

        for _ in range(trades_per_day):
            symbol = random.choice(SYMBOLS)
            base_p = current_prices[symbol]
            # Drift price slightly
            drift = random.gauss(0, 0.002) * base_p
            price = round(max(1.0, base_p + drift), 2)
            current_prices[symbol] = price

            # Volume in shares
            vol = float(random.randint(50, 5000))

            # Pick buyer and seller brokers
            if bofa_bias == "bullish" and random.random() < 0.45:
                buyer = "BOFA"
                seller = random.choice([b for b in BROKERS if b != "BOFA"])
            elif bofa_bias == "bearish" and random.random() < 0.45:
                seller = "BOFA"
                buyer = random.choice([b for b in BROKERS if b != "BOFA"])
            else:
                buyer = random.choice(BROKERS)
                seller = random.choice([b for b in BROKERS if b != buyer])

            trade_time = datetime.combine(
                trade_date,
                datetime.min.time()
            ) + timedelta(
                hours=random.randint(10, 17),
                minutes=random.randint(0, 59),
                seconds=random.randint(0, 59)
            )

            rows.append({
                "trade_id": f"TRD_{trade_date.strftime('%Y%m%d')}_{uuid.uuid4().hex[:8]}",
                "timestamp": trade_time.strftime("%Y-%m-%d %H:%M:%S"),
                "symbol": symbol,
                "price": price,
                "volume": vol,
                "buyer_broker_id": buyer,
                "seller_broker_id": seller,
            })

    return rows


def main():
    print("Generating realistic synthetic BIST trades with BofA institutional flow...")
    rows = generate_mock_trades(num_days=30, trades_per_day=500)

    # Write CSV (zero-dependency standard library)
    csv_output_path = BRONZE_DIR / "sample_bist_trades.csv"
    with open(csv_output_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["trade_id", "timestamp", "symbol", "price", "volume", "buyer_broker_id", "seller_broker_id"])
        writer.writeheader()
        writer.writerows(rows)

    print(f"✔ Successfully created mock trade CSV dataset: {csv_output_path} ({len(rows):,} trades)")

    # If polars is available, also write parquet
    try:
        import polars as pl
        df = pl.DataFrame(rows)
        parquet_output_path = BRONZE_DIR / "sample_bist_trades.parquet"
        df.write_parquet(parquet_output_path)
        print(f"✔ Also saved Parquet copy: {parquet_output_path}")
    except ImportError:
        pass


if __name__ == "__main__":
    main()
