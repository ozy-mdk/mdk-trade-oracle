# MDK Trading Oracle — Repository Rules & Operating Guidelines

## 🏛 System Architecture & Design Philosophy
This repository implements a high-performance **Databricks-style Medallion Data Lakehouse** (DuckDB + Polars + Python 3.9) for BIST institutional flow tracking (specifically Bank of America / Merrill Lynch `MLB`).

- **Bronze Layer (`bronze_raw_trades`, `bronze_instruments`, `bronze_brokers`)**: Raw tick-by-tick trade captures (microsecond timestamps, buyer/seller broker IDs) and reference entity dimensions.
- **Silver Layer (`silver_daily_broker_summary`, `silver_market_daily`)**: Cleaned, daily aggregated broker turnarounds, VWAP prices, and market OHLCV.
- **Gold Layer (`gold_institutional_daily_signals`)**: Feature-engineered rolling 5-day / 20-day institutional accumulation metrics and BofA flow Z-scores.

---

## 📂 Storage & Directory Layout (Strict Separation)
- **Source Code**: `/Users/ozkanyildirim/.gemini/antigravity-ide/scratch/mdk-trading-oracle/`
- **External Data Lakehouse (NEVER inside Git)**: `/Users/ozkanyildirim/data/mdk_oracle/`
  - Raw Landings: `00_raw_data/2026/03_march/raw_csv/**/*.csv` (945 files, 1.94 GB, 36.8M+ trades)
  - DuckDB File: `database/mdk_oracle.duckdb`

---

## 🐍 Python Environment & Concurrency Rules
- **Virtual Environment**: Always use `.venv` at project root:
  - Binary path: `.venv/bin/python`, `.venv/bin/pytest`, `.venv/bin/ruff`
  - Jupyter Kernel: `Python 3.9 (mdk-trading-oracle)`
- **DuckDB Concurrency & File Locks (CRITICAL)**:
  - DuckDB acquires an exclusive file write lock on `.duckdb` files.
  - **Pipelines & Ingestors**: Use write mode via `DuckDBManager()`.
  - **Notebooks & Exploratory Queries**: MUST use `read_only=True` via `DuckDBManager(read_only=True)` or `duckdb.connect(path, read_only=True)` to prevent blocking pipeline executions.

---

## ⚡ CLI & Pipeline Commands
- **Full Lakehouse Pipeline**:
  ```bash
  .venv/bin/python scripts/run_pipeline.py --target all
  # Or with catalog auto-sync:
  .venv/bin/python scripts/run_pipeline.py --target all --sync-catalog
  ```
- **Inspect & Sync Data Catalogs**:
  ```bash
  .venv/bin/python scripts/prepare_data_catalog.py         # Visual Dry-Run
  .venv/bin/python scripts/prepare_data_catalog.py --sync  # Sync to config/*.yaml
  ```
- **Run Tests & Linting**:
  ```bash
  .venv/bin/pytest
  .venv/bin/ruff check .
  ```

---

## 🎯 Domain Metadata
- **Primary Institution**: `MLB` (Bank of America / Merrill Lynch)
- **Top Domestic Banks**: `IYM` (İş Yatırım), `YKR` (Yapı Kredi), `AKM` (Ak Yatırım), `GRM` (Garanti BBVA), `ZRY` (Ziraat)
- **Equities Universe**: 45 liquid BIST stocks (BIST 30 + liquid BIST 50: `THYAO`, `TUPRS`, `ASELS`, `AKBNK`, `ISCTR`, `KCHOL`, `YKBNK`, etc.)
