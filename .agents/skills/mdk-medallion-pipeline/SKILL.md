---
name: mdk-medallion-pipeline
description: >-
  Orchestrate and execute the Medallion Data Lakehouse pipeline (Bronze, Silver, Gold layers)
  for MDK Trading Oracle. Use when ingesting tick data, computing daily broker turnarounds/VWAP,
  generating institutional rolling flow signals, executing predictive Gold models, or troubleshooting DuckDB transformations.
---

# MDK Medallion Lakehouse Pipeline Skill

This skill provides step-by-step procedures for running, modifying, and debugging the **Medallion Lakehouse Pipeline** (`src/mdk_trading_oracle/data/`).

---

## 🤝 1. Collaborative Interaction & Pipeline Workflows

When adding new tables, transforming features, or building predictive models:
1. **Plan Together First**: Discuss table schemas, microstructural metrics, and modeling hypotheses.
2. **Review & Confirm**: Align on column names, data leakage guarantees ($T-1$ Close), and verification tests.
3. **Execute ("Let's Go!")**: Implement changes cleanly across Bronze, Silver, and Gold layers with complete test coverage.

---

## 🏛 2. Lakehouse Architecture Overview

- **Bronze Layer (`src/mdk_trading_oracle/data/bronze/`)**:
  - `bronze_raw_trades`: Ingests raw microsecond tick executions across all available monthly/daily partitions under `~/data/mdk_oracle/00_raw_data/`.
  - `bronze_brokers` & `bronze_instruments`: Synchronizes dimension tables from YAML configs.
- **Silver Layer (`src/mdk_trading_oracle/data/silver/`)**:
  - `silver_daily_broker_summary`: Daily aggregated buy/sell volume, net flow (TL), and VWAP per broker & symbol.
  - `silver_daily_broker_overview`: Macro daily broker turnover, volume shares, and net flows.
  - `silver_daily_stock_summary`: Daily symbol OHLCV and market statistics (also syncs `silver_market_daily`).
  - `silver_daily_sector_summary`: Daily aggregated sector breadth and net flow distributions.
  - `silver_intraday_broker_window_summary`: Intraday execution flows across 4 time windows (Opening, Midday, Afternoon, Closing).
  - `silver_intraday_sector_window_summary`: Intraday sector rotation dynamics across 4 time windows.
- **Gold Layer (`src/mdk_trading_oracle/data/gold/`, `src/mdk_trading_oracle/models/`)**:
  - `gold_institutional_daily_signals`: Rolling 5-day / 20-day cumulative BofA flow, volume shares, and 20-day rolling Z-scores.
  - `gold_bofa_day_start_forecasts`: Model 1 Day-Start Forecaster output populated via `DayStartForecaster(model_type="auto")` with champion predictions, 90% credible ranges, and institutional playbooks.

---

## ⚡ 3. Common Execution Commands

Run using the project virtual environment (`.venv/bin/python`):

```bash
# 1. Full Lakehouse Pipeline (Bronze -> Silver -> Gold + Model 1 Forecasts)
.venv/bin/python scripts/run_pipeline.py --target all

# 2. Pipeline with Raw Discovery & Catalog Auto-Sync (for newly added data)
.venv/bin/python scripts/run_pipeline.py --target all --sync-catalog

# 3. Target Specific Layer
.venv/bin/python scripts/run_pipeline.py --target silver
.venv/bin/python scripts/run_pipeline.py --target gold

# 4. Via Typer CLI
.venv/bin/mdk-oracle pipeline run --target all
```

---

## 🔒 4. Concurrency & Lock Conflict Troubleshooting

If you encounter `DuckDB lock conflict: ... is currently locked by another process`:
1. **Identify the holding process**: `ps aux | grep <PID>` (usually an active Jupyter Notebook kernel).
2. **Terminate the conflicting process**: `kill <PID>`.
3. **In Jupyter Notebooks**: Always connect with `read_only=True`:
   ```python
   from mdk_trading_oracle.core.db import DuckDBManager
   db = DuckDBManager(read_only=True)
   ```

---

## 🧪 5. Verification & Testing

Always verify pipeline changes with automated tests:
```bash
.venv/bin/pytest tests/ -v
.venv/bin/ruff check .
```
