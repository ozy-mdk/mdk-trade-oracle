# MDK Trading Oracle — Repository Rules & Operating Guidelines

## 🎯 Core Project Mission
The primary objective of **MDK Trading Oracle** is to analyze and track institutional order flows on Borsa Istanbul (BIST)—specifically **Bank of America (BofA / clearing code `MLB`)**—to detect institutional accumulation/distribution patterns, algorithmic footprints, and volume surges, and translate them into **actionable trading signals and concrete decision items for individual traders**.

---

## 🤝 Collaborative Interaction & Modeling Workflow
Our development philosophy follows a disciplined, collaborative workflow:
1. **Plan Together First**: Before implementing any new model, pipeline expansion, or architectural change:
   - Discuss and formulate the quantitative rationale, microstructure dynamics, feature clusters, and trade playbooks together.
   - Outline the plan, candidate models, and verification strategy clearly.
2. **Align on the Approach**: Solicit feedback, refine assumptions, and resolve any design questions.
3. **Execute ("Let's Go!")**: Once the plan is confirmed, proceed with full-speed execution, including:
   - Zero-leakage feature extraction
   - Walk-forward candidate arena evaluation
   - Production pipeline & interactive Jupyter notebook integration
   - Automated tests (`pytest`), linting (`ruff`), and Git sync (`develop` / `main`).

---

## 🏛 System Architecture (Medallion Lakehouse)
A high-performance local-first lakehouse powered by **DuckDB + Polars + Python 3.9**:

- **Bronze Layer (`bronze_raw_trades`, `bronze_instruments`, `bronze_brokers`)**:
  - Exact tick-by-tick executed trades (microsecond timestamps, buyer/seller broker clearing IDs).
  - Dimension reference tables for all tracked equities and brokerages.
- **Silver Layer (`silver_daily_broker_summary`, `silver_daily_broker_overview`, `silver_daily_stock_summary`, `silver_daily_sector_summary`, `silver_intraday_broker_window_summary`, `silver_intraday_sector_window_summary`)**:
  - Cleaned, daily aggregated broker turnarounds, buy/sell volume, net flow (TL), and VWAP prices.
  - Daily sector breadth and 4-window intraday execution splits (`Window 1` opening 09:55-10:30, `Window 2` midday, `Window 3` afternoon, `Window 4` closing 17:00-18:10).
- **Gold Layer (`gold_institutional_daily_signals`, `gold_bofa_day_start_forecasts`)**:
  - Feature-engineered rolling 5-day / 20-day institutional accumulation metrics and BofA flow Z-scores.
  - Extensible quantitative predictive models (Model 1: `DayStartForecaster`, Model 2: `IntradayExpansionForecaster`, etc.) designed to host **10+ Gold models**.

---

## 🔬 Gold Layer Quantitative Modeling Standards

1. **Extensible Modeling Framework**:
   - All models inherit from `BaseForecaster` and register in `ModelRegistry` (`@ModelRegistry.register("model_name")`).
   - Predictions produce structured `ForecastResult` instances containing continuous flows, direction classifications, conviction probabilities, 90% credible ranges, and institutional execution playbooks.
2. **Zero Data Leakage**:
   - All features must be computed **strictly from $T-1$ Close data** (or prior completed intraday windows). Future session information must never leak into training or feature sets.
3. **The 7 Quantitative Feature Clusters**:
   - **Cluster 1**: Prior Closing Window Momentum (Window 4 net flow, MOC momentum)
   - **Cluster 2**: Multi-Day Inventory & Sector Saturation Z-Scores (5d/20d rolling accumulation)
   - **Cluster 3**: Cost Basis & Unrealized PnL (Spread from 20-day Volume-Weighted Buy Price)
   - **Cluster 4**: Top-5 Domestic Competitor Posture & Flow Delta (`IYM`, `YKR`, `AKM`, `GRM`, `ZRY`)
   - **Cluster 5**: Institutional Hegemony & Market Share Control (BofA volume share)
   - **Cluster 6**: Sector Cross-Sectional Stress & Breadth (Banking vs Transportation vs Industry)
   - **Cluster 7**: Calendar & Temporal Dynamics (`is_monday`, `is_friday`, day of week)
4. **Candidate Model Arena & Baselines**:
   - Every modeling objective must benchmark against rigorous baselines:
     - `Baseline 0`: Naive Window 4 Persistence (carries yesterday's closing flow forward)
     - `Baseline 1`: 5-Day Historical Moving Average
     - `Machine Learning`: LightGBM Regressor (non-linear tree interactions)
     - `Probabilistic Bayesian Ridge`: Analytical Bayesian Ridge Regression
     - `Full Bayesian MCMC / GLM`: PyMC Bayesian Model (informative shrinkage priors)
5. **Expanding-Window Walk-Forward Validation (Zero Lookahead Bias)**:
   - Models must be evaluated chronologically: train on $1 \dots t-1$ to predict $t$, expanding the window day-by-day.
   - Models are scored on **Out-of-Sample Hit Rate (%)**, **90% PICP Credible Coverage (%)**, and **RMSE (TL M)**.
6. **Automated Champion Selection (`model_type="auto"`) & Dual Delivery**:
   - The exact same `DayStartModelArena` runs in both the production pipeline and interactive research notebooks (`03_bofa_day_start_modeling.ipynb`), crowning and tagging the champion model in DuckDB Gold tables.
7. **Actionable Trader Decision Outputs**:
   - Predictions must map to clear trader items:
     - **Direction**: `STRONG_ACCUMULATE`, `ACCUMULATE`, `NEUTRAL`, `DISTRIBUTE`, `STRONG_DISTRIBUTE`
     - **Playbooks**: `SQUEEZE_LONG`, `LIQUIDITY_FADE`, `MOMENTUM_EXPANSION`, `DEFENSE_SUPPORT`, `SECTOR_ROTATION`, `NEUTRAL_WAIT`
     - **Sector Guidance**: `top_predicted_buy_sector`, `top_predicted_sell_sector`.

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
