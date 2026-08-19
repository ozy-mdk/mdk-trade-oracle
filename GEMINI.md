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

## 🔬 Gold Layer Quantitative Modeling Blueprint & Standards

The Gold Layer is designed as an extensible multi-model suite hosting **10+ distinct institutional models** (e.g. Model 1: Day-Start Flow, Model 2: Sector Buy/Sell Ratio & Rotation, Model 3: Intraday Flow Expansion, etc.). 

Every new model adheres to this **Universal Modeling Blueprint**:

1. **Modular Extensible Framework**:
   - Every model inherits from `BaseForecaster`, extracts features via a dedicated `FeatureExtractor`, and registers in `ModelRegistry` (`@ModelRegistry.register("model_name")`).
   - Predictions produce structured `ForecastResult` instances containing continuous predictions, direction classifications, conviction probabilities, 90% credible ranges, and institutional execution playbooks.
2. **Zero Data Leakage**:
   - Predictive features must be computed **strictly from prior completed windows / $T-1$ Close data**. Future session information must never leak into training or feature sets.
3. **Problem-Specific Quantitative Feature Clusters**:
   - Each modeling objective formulates targeted feature clusters (e.g. Model 1 uses the 7 Opening Clusters: Closing Momentum, Multi-Day Inventory Saturation, Cost Basis PnL, Competitor Deltas, Hegemony, Sector Breadth, Calendar Dynamics; Model 2 will formulate Sector Imbalance & Buy/Sell Ratio clusters, etc.).
4. **Candidate Model Arena & Baselines**:
   - Every modeling objective benchmarks candidate paradigms:
     - `Baselines`: Naive Persistence, Historical Moving Averages
     - `Machine Learning`: Non-linear tree ensembles (LightGBM)
     - `Probabilistic Bayesian`: Analytical Bayesian Ridge & Full Bayesian GLM / MCMC (PyMC)
5. **Expanding-Window Walk-Forward Validation (Zero Lookahead Bias)**:
   - Evaluated chronologically: train on $1 \dots t-1$ to predict $t$, expanding the window day-by-day.
   - Scored on **Out-of-Sample Hit Rate (%)**, **90% PICP Credible Coverage (%)**, and **RMSE / MAE**.
6. **Automated Champion Selection (`model_type="auto"`) & Dual Delivery**:
   - A dedicated `ModelArena` runs in both the production pipeline and interactive research notebooks, crowning and tagging the champion model in DuckDB Gold tables.
7. **Actionable Trader Decision Outputs**:
   - Continuous predictions are translated into concrete trader action items:
     - **Directional Conviction Levels**: `STRONG_ACCUMULATE`, `ACCUMULATE`, `NEUTRAL`, `DISTRIBUTE`, `STRONG_DISTRIBUTE`
     - **Institutional Playbooks**: Context-driven trade blueprints (e.g. `SQUEEZE_LONG`, `LIQUIDITY_FADE`, `MOMENTUM_EXPANSION`, `DEFENSE_SUPPORT`, `SECTOR_ROTATION`, `NEUTRAL_WAIT`).
     - **Actionable Guidance**: Top predicted buy/sell sectors or equities.

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
