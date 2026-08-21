# MDK Trading Oracle — Repository Rules & Operating Guidelines

## 1. Core Project Mission
The primary objective of **MDK Trading Oracle** is to analyze and track institutional order flows on Borsa Istanbul (BIST)—specifically **Bank of America (BofA / clearing code `MLB`)**—to detect institutional accumulation/distribution patterns, algorithmic footprints, and volume surges, and translate them into **actionable trading signals and concrete decision items for individual traders**.

---

## 2. Collaborative Interaction & Modeling Workflow
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

## 3. System Architecture (Medallion Lakehouse)
A high-performance local-first lakehouse powered by **DuckDB + Polars + Python 3.9**:

- **Bronze Layer (`bronze_raw_trades`, `bronze_central_bank_rates`, `bronze_instruments`, `bronze_brokers`)**:
  - Exact tick-by-tick executed trades (microsecond timestamps, buyer/seller broker clearing IDs).
  - Official Central Bank (TCMB) 1-Week Repo policy interest rates, rate changes, and decision day flags.
  - Dimension reference tables for all tracked equities and brokerages.
- **Silver Layer (`silver_daily_broker_summary`, `silver_daily_broker_overview`, `silver_daily_stock_summary`, `silver_daily_sector_summary`, `silver_daily_macro_rates`, `silver_intraday_broker_window_summary`, `silver_intraday_sector_window_summary`)**:
  - Cleaned, daily aggregated broker turnarounds, buy/sell volume, net flow (TL), and VWAP prices.
  - Daily macroeconomic interest rates enriched with days elapsed since last MPC rate change and 30-day rolling rate averages.
  - Daily sector breadth and 4-window intraday execution splits (`Window 1` opening 09:55-10:30, `Window 2` midday, `Window 3` afternoon, `Window 4` closing 17:00-18:10).
- **Gold Layer (`gold_institutional_daily_signals`, `gold_bofa_*_forecasts`, `gold_bofa_*_performance`, `gold_bofa_*_backtests`)**:
  - Feature-engineered rolling 5-day / 20-day institutional accumulation metrics and BofA flow Z-scores.
  - Three distinct Gold table categories per predictive model:
    1. **Live Upcoming Forecasts (`gold_bofa_*_forecasts`)**: Strictly holds only active live predictions for upcoming session $T+1$.
    2. **Historical Performance Ledgers (`gold_bofa_*_performance`)**: Permanent audited tracking ledgers recording past forecasts matched against actual realized Window 1 market data from Silver.
    3. **Simulation Backtests (`gold_bofa_*_backtests`)**: Full historical walk-forward simulation ledgers for calibration, residual diagnostics, and tournament benchmarking.
  - Extensible quantitative predictive models designed to host **10+ Gold models**:
    - **Model 1: Macro Day-Start Forecaster (`DayStartForecaster`)**: Forecasts exchange-wide BofA opening net flow ($TL$), directional conviction, 90% credible intervals, and macro execution playbooks.
    - **Model 2: Sector Day-Start Forecaster (`SectorDayStartForecaster`)**: Forecasts BofA's capital allocation and sector rotation across all 26 tracked BIST sectors at the open.

---

## 4. Gold Layer Quantitative Modeling Blueprint & Standards

The Gold Layer is designed as an extensible multi-model suite hosting **10+ distinct institutional models** (e.g. Model 1: Macro Day-Start Flow, Model 2: Sector Day-Start Allocation, Model 3: Intraday Flow Expansion, etc.). 

Every new model adheres to this **Universal Modeling Blueprint**:

1. **Modular Extensible Framework**:
   - Every model inherits from `BaseForecaster`, extracts features via a dedicated `FeatureExtractor`, and registers in `ModelRegistry` (`@ModelRegistry.register("model_name")`).
   - Predictions produce structured `ForecastResult` instances containing continuous predictions, direction classifications, conviction probabilities, 90% credible ranges, and institutional execution playbooks.
2. **Zero Data Leakage & Retrospective Anchoring**:
   - Predictive features must be computed **strictly from prior completed windows / $T-1$ Close data**. Future session information must never leak into training or feature sets.
   - When evaluating any historical date retrospectively, the 12-month training lookback window dynamically and strictly anchors backwards from that exact reference date ($[T - 12\text{ months}, T-1]$).
3. **Rigorous & Clean Target Variables**:
   - Target formulation is strictly mathematically grounded: continuous regression target $y = \text{target\_open\_net\_flow\_tl}$ and derived binary/conviction direction $\text{target\_open\_direction}$.
4. **Problem-Specific Quantitative Feature Clusters**:
   - **Model 1 (7 Macro Clusters)**: Closing Momentum, Multi-Day Inventory Saturation, Cost Basis PnL, Competitor Deltas, Hegemony, Sector Breadth, Calendar Dynamics.
   - **Model 2 (5 Sector Clusters)**: Sector Closing Momentum, Sector Competitor Imbalance, Sector Dominance & Wallet Share, Sector Multi-Day Accumulation, Macro Context & Calendar Dynamics.
5. **Candidate Model Arena & Baselines**:
   - Every modeling objective benchmarks 5 candidate paradigms:
     - `Baselines`: Naive Persistence (prior W4 flow), Historical Moving Averages (5-day rolling mean).
     - `Machine Learning`: Non-linear tree ensembles (LightGBM).
     - `Probabilistic Bayesian`: Analytical Bayesian Ridge & Full Bayesian GLM / MCMC (PyMC).
6. **Three-Table Persistence Architecture ($T+1$ Forecasts vs Performance Ledgers vs Simulation Backtests)**:
   - Every predictive Gold model strictly separates live predictions, audited performance tracking, and simulation ledgers:
     - `forecast_next_day()`: Generates live inference for upcoming trading session $T+1$. Persisted to `gold_bofa_*_forecasts` with `replace_active=True` (strictly holds only $T+1$).
     - `reconcile_and_update_performance_ledger()`: Reconciles past forecasts against realized actual Window 1 market data, logging `actual_open_net_flow_tl`, `error_open_net_flow_tl`, `absolute_error_tl`, `is_direction_hit`, and `is_inside_90_ci` into `gold_bofa_*_performance`.
     - `backtest_all_history()`: Out-of-sample historical walk-forward simulation across past sessions persisted to dedicated `gold_bofa_*_backtests` tables.
7. **Zero-Lookahead Point-In-Time Historical Backfill Engine**:
   - Allows running retrospective point-in-time forecasts for past missed sessions (`--backfill-dates` or `--backfill-missing`).
   - For every target date $T_{target}$, data cutoff is strictly $T_{as\_of} = \max(\text{dates} < T_{target})$, hiding all subsequent data.
   - Configurable discovery lookback window (`default_lookback_months: 2` in `config/default.yaml`, overridable via `--backfill-lookback-months` or `--backfill-lookback-days`).
   - Upserts into `gold_bofa_*_performance` for only the targeted dates while preserving all other historical performance records.
8. **Automated On-The-Fly Champion Selection (`model_type="auto"`) & No Hardcoding**:
   - In both the production pipeline and interactive research notebooks, models are never hardcoded. A dedicated `ModelArena` runs tournaments on the fly across candidate paradigms, crowning and tagging the champion model based on multi-criteria metrics (Out-of-sample Hit Rate %, 90% PICP %, and RMSE).
9. **Interactive Research Notebook Standards & Clean Presentation**:
   - Every modeling notebook (`notebooks/03_*.ipynb`, `notebooks/04_*.ipynb`, etc.) must provide:
     1. **Live Upcoming Session Signal Card ($T+1$)**: Prominent executive card with forecasted net flow ($TL$), 90% credible ranges, directional badges, institutional playbooks, and sector rotation allocations.
     2. **Performance Ledger & Historical Backtest View**: Actual vs. predicted track record with 90% confidence interval ribbons and interactive dropdown session inspectors.
     3. **DuckDB Gold Table Verification**: Direct queries inspecting persisted production records (live forecasts, performance ledgers, and backtest simulations).
   - **Clean & Professional Typography (No Excessive Emojis)**:
     - Keep documentation, markdown cells, headers, section titles, comments, and card templates clean, crisp, and professional.
     - **Do NOT use excessive emojis** in headers, text blocks, or notebooks. Use standard structured markdown headers, clean typography, tables, and minimal functional status badges (`[PASS]`, `[FAIL]`, `HIT`, `MISS`).
10. **Actionable Trader Decision Outputs**:
    - Continuous predictions are translated into concrete trader action items:
      - **Directional Conviction Levels**: `STRONG_ACCUMULATE`, `ACCUMULATE`, `NEUTRAL`, `DISTRIBUTE`, `STRONG_DISTRIBUTE`
      - **Institutional Playbooks**: Context-driven trade blueprints (`SQUEEZE_LONG`, `LIQUIDITY_FADE`, `MOMENTUM_EXPANSION`, `DEFENSE_SUPPORT`, `SECTOR_ROTATION`, `NEUTRAL_WAIT`).
      - **Actionable Guidance**: Top predicted buy/sell sectors or equities.

---

## 5. Portable Storage & Directory Separation
To allow seamless portability across different team members' local machines:

- **Source Code (Repository Root)**: `./` (where `pyproject.toml`, `src/`, `config/`, `notebooks/`, `scripts/` reside).
- **External Data Lakehouse (Configurable & Portable)**:
  - Default path: `~/data/mdk_oracle/` (or configured via `DATA_DIR` in `.env`).
  - Raw Landings:
    - BIST Trades: `~/data/mdk_oracle/00_raw_data/<year>/<month>/raw_csv/**/*.csv`
    - Central Bank Rates: `~/data/mdk_oracle/00_raw_data/central_bank_interest_rates/**/*.*` (`.xlsx`, `.xls`, `.csv`, `.parquet`)
  - DuckDB Storage: `~/data/mdk_oracle/database/mdk_oracle.duckdb`
  - **Rule**: Never hardcode absolute user-specific home paths (e.g. `/Users/ozkanyildirim/`). Always use `Path.home() / "data" / "mdk_oracle"` or `get_settings().data_dir`.

---

## 6. Incremental & Multi-Month Data Ingestion
- Current baseline dataset: March 2026 (945 CSV files, 21 trading days, 36.8M+ trades) + Central Bank 1-week repo rate history (1,157 records from 2022 to 2026).
- The pipeline supports adding new daily and monthly raw data feeds under `00_raw_data/<year>/<month>/raw_csv/` and monthly Central Bank policy updates under `00_raw_data/central_bank_interest_rates/`.
- **Idempotent Upserting**: Central Bank files are upserted (`INSERT OR REPLACE`) to preserve historical series while updating new rates.
- **Continuous Forward-Fill Sync**: When market trading dates advance beyond the latest CBRT file, the pipeline forward-fills the latest known rate (`is_forward_filled = TRUE`) so daily models never have date gaps.
- Use `--sync-catalog` during ingestion to auto-discover any new stock tickers or broker codes.

---

## 7. Python Environment & Concurrency Rules
- **Virtual Environment**: Always use `.venv` at project root:
  - Binary path: `.venv/bin/python`, `.venv/bin/pytest`, `.venv/bin/ruff`
  - Jupyter Kernel: `Python 3.9 (mdk-trading-oracle)`
- **DuckDB Concurrency & File Locks (CRITICAL)**:
  - DuckDB uses exclusive file write locks.
  - **Pipelines & Ingestors**: Use write mode via `DuckDBManager()`.
  - **Notebooks & Analytical Queries**: MUST use `read_only=True` (`DuckDBManager(read_only=True)` or `duckdb.connect(path, read_only=True)`) to ensure notebooks never block pipeline executions.

---

## 8. Key CLI & Pipeline Commands
- **Full Lakehouse Pipeline**:
  ```bash
  .venv/bin/python scripts/run_pipeline.py --target all
  # Or with catalog auto-sync:
  .venv/bin/python scripts/run_pipeline.py --target all --sync-catalog
  ```
- **Central Bank Rates Ingestion & Market Sync**:
  ```bash
  .venv/bin/mdk-oracle load-rates
  ```
- **Daily Gold Layer Execution & Live Inference ($T+1$)**:
  ```bash
  .venv/bin/python scripts/run_pipeline.py --target gold
  ```
- **Historical Point-in-Time Performance Backfilling**:
  ```bash
  # Backfill missing sessions within default 2-month window:
  .venv/bin/python scripts/run_pipeline.py --target gold --backfill-missing

  # Custom lookback window (e.g., 3 months or 45 days):
  .venv/bin/python scripts/run_pipeline.py --target gold --backfill-missing --backfill-lookback-months 3
  .venv/bin/python scripts/run_pipeline.py --target gold --backfill-missing --backfill-lookback-days 45

  # Backfill specific missed dates:
  .venv/bin/python scripts/run_pipeline.py --target gold --backfill-dates 2026-03-10,2026-03-18
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

## 9. Primary Market Participants & Universe
- **Primary Institutional Target**: **Bank of America (BofA) [Clearing Code: `MLB`]** — algorithmic execution and high-impact institutional flow.
- **Domestic Major Banks**: `IYM` (İş Yatırım), `YKR` (Yapı Kredi), `AKM` (Ak Yatırım), `GRM` (Garanti BBVA), `ZRY` (Ziraat), `DZY` (Deniz), `VKY` (Vakıf), `HLY` (Halk).
- **Equities Universe**: 45 liquid BIST stocks (BIST 30 + liquid BIST 50).

