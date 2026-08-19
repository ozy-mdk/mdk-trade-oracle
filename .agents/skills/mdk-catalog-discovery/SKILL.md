---
name: mdk-catalog-discovery
description: >-
  Inspect raw CSV trade feeds, auto-discover equities and brokerage entities,
  and synchronize YAML metadata catalogs for MDK Trading Oracle.
  Use when new raw trade dumps are added, when checking data completeness, or auditing data coverage.
---

# MDK Catalog Discovery & Data Preparation Skill

This skill guides the discovery of new instruments, broker codes, and raw CSV files in `/Users/ozkanyildirim/data/mdk_oracle/00_raw_data/`.

---

## 1. Discovery Workflows

### A. Dry-Run Visual Inspection
Scans the raw CSV landing directory and prints rich ranking tables of discovered equities and brokers without modifying YAML files:
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

---

## 2. Interactive Discovery & Audit Notebook

For a visual, step-by-step audit of data inventory, in-scope vs out-of-scope boundaries, and 100% zero-loss completeness verification:
- Open [`notebooks/00_data_discovery_and_catalog_analysis.ipynb`](file:///Users/ozkanyildirim/.gemini/antigravity-ide/scratch/mdk-trading-oracle/notebooks/00_data_discovery_and_catalog_analysis.ipynb)
- Select Jupyter Kernel: **`Python 3.9 (mdk-trading-oracle)`**

---

## 3. Data Completeness Formula

Verify zero data loss across layers:
$$\text{Expected Daily Rows} = \text{Unique Equities (45)} \times \text{Trading Days (21)} = 945\text{ rows}$$
$$\text{Bronze Rows} = 36,818,222 \implies \text{Silver Market Rows} = 945 \implies \text{Gold Signal Rows} = 945$$
