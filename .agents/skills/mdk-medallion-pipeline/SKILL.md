---
name: mdk-medallion-pipeline
description: >-
  Orchestrate and execute the Medallion Data Lakehouse pipeline (Bronze, Silver, Gold layers)
  for MDK Trading Oracle. Use when ingesting tick data, computing daily broker turnarounds/VWAP,
  generating institutional rolling flow signals, or troubleshooting DuckDB transformations.
---

# MDK Medallion Lakehouse Pipeline Skill

This skill provides step-by-step procedures for running, modifying, and debugging the **Medallion Lakehouse Pipeline** (`src/mdk_trading_oracle/data/`).

---

## 1. Lakehouse Architecture Overview

- **Bronze Layer (`src/mdk_trading_oracle/data/bronze/`)**:
  - `bronze_raw_trades`: Ingests raw microsecond tick executions (36.8M+ trades for March 2026).
  - `bronze_brokers` & `bronze_instruments`: Synchronizes dimension tables from YAML configs.
- **Silver Layer (`src/mdk_trading_oracle/data/silver/`)**:
  - `silver_daily_broker_summary`: Daily aggregated buy/sell volume, net flow (TL), and VWAP per broker & symbol.
  - `silver_market_daily`: Market OHLCV, total volume/turnover, and BofA net flow per trading day.
- **Gold Layer (`src/mdk_trading_oracle/data/gold/`)**:
  - `gold_institutional_daily_signals`: Computes rolling 5-day / 20-day cumulative BofA flow, volume shares, and 20-day rolling Z-scores.

---

## 2. Common Execution Commands

Run using the project virtual environment (`.venv/bin/python`):

```bash
# 1. Full Lakehouse Pipeline (Bronze -> Silver -> Gold)
.venv/bin/python scripts/run_pipeline.py --target all

# 2. Pipeline with Raw Discovery & Catalog Auto-Sync
.venv/bin/python scripts/run_pipeline.py --target all --sync-catalog

# 3. Target Specific Layer
.venv/bin/python scripts/run_pipeline.py --target silver

# 4. Via Typer CLI
.venv/bin/mdk-oracle pipeline run --target all
```

---

## 3. Concurrency & Lock Conflict Troubleshooting

If you encounter `DuckDB lock conflict: ... is currently locked by another process (PID xxxx)`:
1. **Identify the holding process**: `ps aux | grep <PID>` (usually an active Jupyter Notebook kernel).
2. **Terminate the conflicting process**: `kill <PID>`.
3. **In Jupyter Notebooks**: Always connect with `read_only=True`:
   ```python
   from mdk_trading_oracle.core.db import DuckDBManager
   db = DuckDBManager(read_only=True)
   ```

---

## 4. Verification & Testing

Always verify pipeline changes with automated tests:
```bash
.venv/bin/pytest
.venv/bin/ruff check .
```
