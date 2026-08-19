# MDK Trading Oracle — Repository Rules & Operating Guidelines

## 🎯 Core Project Mission
The primary objective of **MDK Trading Oracle** is to analyze and track institutional order flows on Borsa Istanbul (BIST)—specifically **Bank of America (BofA / clearing code `MLB`)**—to detect institutional accumulation/distribution patterns, algorithmic footprints, and volume surges, and translate them into **actionable trading signals and concrete decision items for individual traders**.

---

## 🏛 System Architecture (Medallion Lakehouse)
A high-performance local-first lakehouse powered by **DuckDB + Polars + Python 3.9**:

- **Bronze Layer (`bronze_raw_trades`, `bronze_instruments`, `bronze_brokers`)**:
  - Exact tick-by-tick executed trades (microsecond timestamps, buyer/seller broker clearing IDs).
  - Dimension reference tables for all tracked equities and brokerages.
- **Silver Layer (`silver_daily_broker_summary`, `silver_market_daily`)**:
  - Cleaned, daily aggregated broker turnarounds, buy/sell volume, net flow (TL), and VWAP prices.
  - Daily market OHLCV and BofA market share / net flow metrics.
- **Gold Layer (`gold_institutional_daily_signals`)**:
  - Feature-engineered rolling 5-day / 20-day institutional accumulation metrics and BofA flow Z-scores.
  - Actionable pattern flags for momentum breakouts and institutional reversals.

---

## 📂 Portable Storage & Directory Separation
To allow seamless portability across different team members' local machines:

- **Source Code (Repository Root)**: `./` (where `pyproject.toml`, `src/`, `config/`, `notebooks/`, `scripts/` reside).
- **External Data Lakehouse (Configurable & Portable)**:
  - Default path: `~/data/mdk_oracle/` (or configured via `DATA_DIR` in `.env`).
  - Raw Landings: `~/data/mdk_oracle/00_raw_data/<year>/<month>/raw_csv/**/*.csv`
  - DuckDB Storage: `~/data/mdk_oracle/database/mdk_oracle.duckdb`
  - **Rule**: Never hardcode absolute user-specific home paths (e.g. `/Users/ozkanyildirim/`). Always use `Path.home() / "data" / "mdk_oracle"` or `get_settings().data_dir`.

---

## 📅 Incremental & Multi-Month Data Ingestion
- Current baseline dataset: March 2026 (945 CSV files, 21 trading days, 36.8M+ trades).
- The pipeline supports adding new daily and monthly raw data feeds under `00_raw_data/<year>/<month>/raw_csv/`.
- Use `--sync-catalog` during ingestion to auto-discover any new stock tickers or broker codes.

---

## 🐍 Python Environment & Concurrency Rules
- **Virtual Environment**: Always use `.venv` at project root:
  - Binary path: `.venv/bin/python`, `.venv/bin/pytest`, `.venv/bin/ruff`
  - Jupyter Kernel: `Python 3.9 (mdk-trading-oracle)`
- **DuckDB Concurrency & File Locks (CRITICAL)**:
  - DuckDB uses exclusive file write locks.
  - **Pipelines & Ingestors**: Use write mode via `DuckDBManager()`.
  - **Notebooks & Analytical Queries**: MUST use `read_only=True` (`DuckDBManager(read_only=True)` or `duckdb.connect(path, read_only=True)`) to ensure notebooks never block pipeline executions.

---

## ⚡ Key CLI & Pipeline Commands
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

## 🏦 Primary Market Participants & Universe
- **Primary Institutional Target**: **Bank of America (BofA) [Clearing Code: `MLB`]** — algorithmic execution and high-impact institutional flow.
- **Domestic Major Banks**: `IYM` (İş Yatırım), `YKR` (Yapı Kredi), `AKM` (Ak Yatırım), `GRM` (Garanti BBVA), `ZRY` (Ziraat), `DZY` (Deniz), `VKY` (Vakıf), `HLY` (Halk).
- **Equities Universe**: 45 liquid BIST stocks (BIST 30 + liquid BIST 50).
