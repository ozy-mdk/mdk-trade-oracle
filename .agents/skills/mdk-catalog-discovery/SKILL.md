---
name: mdk-catalog-discovery
description: >-
  Inspect raw CSV trade feeds, auto-discover equities and brokerage entities,
  and synchronize YAML metadata catalogs for MDK Trading Oracle.
  Use when new raw trade dumps (daily or monthly) are added, when checking data completeness, or auditing data coverage.
---

# MDK Catalog Discovery & Data Preparation Skill

This skill guides the discovery of new instruments, broker codes, raw CSV files, and macroeconomic policy rate data across all partitions in `~/data/mdk_oracle/00_raw_data/` (or path configured in `$DATA_DIR`).

---

## 1. Discovery Workflows

### A. Dry-Run Visual Inspection
Scans the raw CSV landing directory (`00_raw_data/<year>/<month>/raw_csv/`) and prints rich ranking tables of discovered equities and brokers without modifying YAML files:
```bash
.venv/bin/python scripts/prepare_data_catalog.py
# Or CLI
.venv/bin/mdk-oracle data inspect
```

### B. Synchronize YAML Metadata Catalogs
Extracts all unique equities and brokers and writes structured definitions to `config/instruments.yaml` and `config/brokers.yaml`:
```bash
.venv/bin/python scripts/prepare_data_catalog.py --sync
# Or CLI
.venv/bin/mdk-oracle data sync-catalog
```

### C. Macroeconomic Central Bank Rates Discovery & Ingestion
Scans `00_raw_data/central_bank_interest_rates/` for updated Central Bank 1-Week Repo rate files (`.xlsx`, `.xls`, `.csv`, `.parquet`), performs idempotent upserts into `bronze_central_bank_rates`, and forward-fills through market dates:
```bash
.venv/bin/mdk-oracle load-rates
```

---

## 2. Interactive Discovery & Audit Notebook

For a visual, step-by-step audit of data inventory, in-scope vs out-of-scope boundaries, Central Bank policy rate decisions (2022–2026), and zero-loss completeness verification:
- Open [`notebooks/00_data_discovery_and_catalog_analysis.ipynb`](file:///Users/ozkanyildirim/.gemini/antigravity-ide/scratch/mdk-trading-oracle/notebooks/00_data_discovery_and_catalog_analysis.ipynb)
- Select Jupyter Kernel: **`Python 3.9 (mdk-trading-oracle)`**

---

## 3. Data Completeness Verification Formula

Verify zero data loss across layers:
$$\text{Expected Daily Rows} = \text{Unique Equities} \times \text{Trading Days}$$
$$\text{Raw Bronze Trades Ingested} \implies \text{Silver Daily Summary Rows} \implies \text{Gold Actionable Signal Rows}$$
$$\text{Bronze Central Bank Rates (2022–2026)} \implies \text{Silver Daily Macro Rates (Aligned \& Continuous)}$$
Ensure that:
- Every raw CSV trade file and Central Bank rate update is accounted for.
- Total market turnover calculated in Bronze matches Silver and Gold sums identically.
- Central Bank policy rates have zero missing dates across all historical BIST trading sessions.
