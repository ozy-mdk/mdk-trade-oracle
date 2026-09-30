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
A high-performance lakehouse powered by **PostgreSQL 16 + TimescaleDB + Polars + Python 3.9 on Apple Silicon M5 Mac Pro**:

- **Bronze Layer (`bronze_raw_trades`, `bronze_central_bank_rates`, `bronze_bist_index_benchmarks`, `bronze_corporate_actions`, `bronze_bist30_membership`, `bronze_bist30_changes`, `bronze_bist30_stock_periods`, `bronze_instruments`, `bronze_brokers`)**:
  - Exact tick-by-tick executed trades (microsecond timestamps, buyer/seller broker clearing IDs) stored in a TimescaleDB Hypertable with columnar compression.
  - Official Central Bank (TCMB) 1-Week Repo policy interest rates, rate changes, and decision day flags.
  - Official BIST 30 (`XU030`) benchmark historical OHLCV data.
  - Historical corporate actions (stock splits, rights issues, ticker symbol changes).
  - BIST 30 index membership snapshots, quarterly rebalancing changes, and continuous stock periods.
  - Dimension reference tables for all tracked equities and brokerages.
- **Silver Layer (`silver_candles_1m`, `silver_candles_5m`, `silver_candles_1d`, `silver_corporate_action_adjustment_periods`, `silver_daily_broker_summary`, `silver_daily_broker_overview`, `silver_daily_stock_summary`, `silver_daily_sector_summary`, `silver_daily_macro_rates`, `silver_daily_benchmark_index`, `silver_bofa_historical_flow_thresholds`, `silver_intraday_broker_window_summary`, `silver_intraday_sector_window_summary`)**:
  - Continuous aggregate hypertables (`silver_candles_1m`, `silver_candles_5m`, `silver_candles_1d`) computing sub-millisecond OHLCV, volume, and BofA institutional net turnover for the React candlestick frontend.
  - Cleaned, daily aggregated broker turnarounds, buy/sell volume, net flow (TL), and VWAP prices.
  - Precision corporate action adjustment periods (`silver_corporate_action_adjustment_periods`) providing continuous `quantity_factor` and `canonical_symbol` mappings with zero monetary distortion ($\text{Turnover TL} = \text{Conserved}$).
  - Adjusted prices and returns enriched directly in `silver_daily_stock_summary` (`adj_close_price`, `adj_daily_return_pct`, `adj_market_vwap`, `adj_total_volume`, `adj_bofa_total_vwap`).
  - Daily macroeconomic interest rates enriched with days elapsed since last MPC rate hike/cut, rate change deltas, rate spreads vs 30-day mean, and daily carry costs.
  - Daily BIST 30 benchmark metrics including rolling 5-day / 20-day returns, 20-day historical volatility, trend relative to 20-day SMA, and BIST 30 shock regimes (`is_shock_day`, `shock_type` POSITIVE/NEGATIVE >= 3%, `days_since_last_positive_shock`, `days_since_last_negative_shock`).
  - Empirical flow percentile profiles (`silver_bofa_historical_flow_thresholds` across 27 scopes: 1 Macro ALL + 26 BIST sectors) computing $P_{25}, P_{50}, P_{85}$ for positive buy flows and negative sell flows.
  - Institutional FIFO Tertip Mechanism (`INTRADAY_MATCHED_FIFO_V1` across 7 institutions: `MLB`, `IYM`, `YKR`, `AKM`, `GRM`, `ZRY`, `TRA`):
    - `silver_broker_fifo_daily`: Historical point-in-time time-series logging daily matched flow, intraday PnL, residual flow, carry FIFO realized PnL, open stock inventory, average unit cost, and MTM valuation for every session $T$.
    - `silver_broker_fifo_lot_entries`: Permanent immutable lot creation records (`opened_quantity`, `opened_value_tl`, `opened_unit_cost`).
    - `silver_broker_fifo_lots`: Currently open FIFO lots as of the latest completed session.
    - `silver_broker_fifo_lot_realizations`: Audited closure events logging partial/full lot exits and realized PnL.
    - `silver_broker_fifo_lot_lifecycle`: Open-to-close lifecycle summary view.
  - Daily sector breadth and 5-window intraday execution splits in Turkish Time (TRT): `Window 1` (day_start) opening 09:55-10:30, `Window 2` (first_reaction) 10:30-11:30, `Window 3` (midday_followup) 11:30-14:30, `Window 4` (afternoon_reaction) 14:30-16:00, `Window 5` (closing_session) closing 16:00-18:15.
  - **Turkish Timezone Mandate**: All data, window partitions, log outputs, and database models operate strictly in **Turkish Time (`Europe/Istanbul` / TRT / UTC+3)** with no Central European Time (CET/CEST) or UTC conversions.
- **Gold Layer (`gold_institutional_daily_signals`) & Institutional Predictive Hub**:
  - Feature-engineered rolling 5-day / 20-day institutional accumulation metrics and BofA flow Z-scores persisted to `gold_institutional_daily_signals`.
  - **Institutional Tertip ML Forecaster (`TertipMLForecaster`)**: Live predictive engine powered by 17 lean microstructure features, point-in-time FIFO inventory tracking, intraday matched volume, carry FIFO PnL, carry costs, macro rates, and benchmark index momentum.
  - **Walk-Forward Tournament Arena**: Dynamic candidate model arena benchmarking LightGBM, Bayesian Ridge, and Moving Average baselines on the fly, with champion selection crowned primarily by **Directional Hit Rate %** (with MAE tie-breaker).
  - **Trader Workstation Integration**: Real-time signal cards for upcoming session $T+1$ with forecasted flow, credible ranges, directional badges, institutional playbooks (`SQUEEZE_LONG`, `MOMENTUM_EXPANSION`, `LIQUIDITY_FADE`, `DEFENSE_SUPPORT`), and 30-Day performance track records.

---

## 4. Institutional Machine Learning Forecasting Blueprint & Standards

The predictive architecture is centered around the **Tertip Machine Learning Forecaster**, translating microstructural order flow into high-conviction decision items:

1. **Tertip Microstructure & Inventory Foundation**:
   - Volume alone does not disclose institutional intent. By tracking the **`INTRADAY_MATCHED_FIFO_V1`** mechanism across institutions (`MLB`, `IYM`, `YKR`, `AKM`, `GRM`, `ZRY`, `TRA`), the model distinguishes between intraday scalping/market making and strategic carry inventory accumulation/liquidation.
   - Captures inventory saturation, cost basis spread, and carried unrealized PnL to forecast liquidation pressure, short squeezes, and defense accumulation.
2. **17 Lean Zero-Leakage Features**:
   - All predictive features are computed **strictly from prior completed windows / $T-1$ Close data**. Future session information never leaks into training or feature sets.
   - The training lookback dynamically and strictly anchors backwards 12 months from the evaluation date ($[T - 12\text{ months}, T-1]$).
   - Core lean feature set:
     - *Tertip Inventory Dynamics*: `feat_tertip_inventory_flow_yesterday_tl`, `feat_tertip_unrealized_pnl_yesterday_tl`, `feat_tertip_carry_pnl_yesterday_tl`, `feat_tertip_intraday_pnl_yesterday_tl`.
     - *Macro Rates & Carry Costs*: `feat_macro_repo_rate_pct`, `feat_macro_daily_carry_cost_bps`, `feat_days_since_last_cbrt_decision`.
     - *Benchmark Index Posture*: `feat_bist30_volatility_20d_pct`, `feat_bist30_trend_vs_20d_sma_pct`.
     - *Calendar Dynamics*: `feat_day_of_week`, `feat_is_monday`, `feat_is_friday`.
3. **Walk-Forward Model Tournament & Directional Champion Selection**:
   - Walk-forward candidate arena benchmarks:
     - `Baselines`: Historical Moving Averages (5-day rolling mean).
     - `Machine Learning`: Non-linear gradient boosting (LightGBM).
     - `Probabilistic Bayesian`: Analytical Bayesian Ridge Regression.
   - **Primary Champion Criterion**: Out-of-sample **Directional Hit Rate %** (percentage of sessions where the predicted sign correctly matched the actual realized market flow direction), using Mean Absolute Error (MAE) as the secondary tie-breaker.
4. **Actionable Trader Playbooks & Dynamic Thresholds**:
   - Translates predicted flows into actionable context blueprints:
     - **`SQUEEZE_LONG`**: Strong positive flow expectation with heavy competitor delta — follow aggressive opening accumulation.
     - **`MOMENTUM_EXPANSION`**: Extreme opening accumulation — institutional momentum continuation.
     - **`LIQUIDITY_FADE`**: High negative flow with deep unrealized gains — expect profit-taking and fade intraday dips.
     - **`DEFENSE_SUPPORT`**: Underwater carried inventory with positive flow — institutional defense accumulation.
     - **`NEUTRAL_WAIT`**: Sub-threshold flow — wait for intraday confirmation.
5. **Interactive Research & Serving Standards**:
   - Clean, professional presentation with zero excessive emojis.
   - Frontend Predictive Hub displays 30-Day performance ledger, directional hit tally pills, backtest visualizer, and live $T+1$ signal cards.

---

## 5. Portable Storage & Directory Separation
To allow seamless portability across different team members' local machines:

- **Source Code (Repository Root)**: `./` (where `pyproject.toml`, `src/`, `config/`, `notebooks/`, `scripts/` reside).
- **External Data Lakehouse (Configurable & Portable)**:
  - Default path: `~/data/mdk_oracle/` (or configured via `DATA_DIR` in `.env`).
  - Raw Landings:
    - BIST Trades: `~/data/mdk_oracle/00_raw_data/<year>/<month>/raw_csv/**/*.csv`
    - Central Bank Rates: `~/data/mdk_oracle/00_raw_data/central_bank_interest_rates/**/*.*` (`.xlsx`, `.xls`, `.csv`, `.parquet`)
  - Database: PostgreSQL 16 + TimescaleDB (`PG_HOST`, `PG_PORT`, `PG_DATABASE`, `PG_USER` in `.env`).
  - **Rule**: Never hardcode absolute user-specific home paths (e.g. `/Users/ozkanyildirim/`). Always use `Path.home() / "data" / "mdk_oracle"` or `get_settings().data_dir`.

---

## 6. Full Historical Dataset & Apple Silicon M5 Mac Pro Architecture
- **Full History Scope**: Powered by the **Apple Silicon M5 Mac Pro**, the platform hosts the complete historical dataset spanning **2022 to 2026+ (2.2+ Billion executed trades, 4.5+ years of daily and intraday order flow)**, along with full Central Bank interest rate history (1,180+ policy sessions) and BIST 30 benchmark OHLCV series.
- **Hardware Acceleration**: Takes full advantage of the M5 Mac Pro's high-bandwidth unified memory architecture, multi-core SIMD processing, and TimescaleDB columnar compression policies.
- The pipeline supports continuous ingestion of new daily and monthly raw trade dumps under `00_raw_data/<year>/<month>/raw_csv/` and Central Bank policy updates under `00_raw_data/central_bank_interest_rates/`.
- **Idempotent Upserting**: Central Bank files are upserted (`ON CONFLICT DO UPDATE`) to preserve historical series while updating new rates.
- **Continuous Forward-Fill Sync**: When market trading dates advance beyond the latest CBRT file, the pipeline forward-fills the latest known rate (`is_forward_filled = TRUE`) so daily models never have date gaps.
- Use `--sync-catalog` during ingestion to auto-discover any new stock tickers or broker codes.

---

## 7. Concurrency & Serving Rules
- **Virtual Environment**: Always use `.venv` at project root:
  - Binary path: `.venv/bin/python`, `.venv/bin/pytest`, `.venv/bin/ruff`
  - Jupyter Kernel: `Python 3.9 (mdk-trading-oracle)`
- **Multi-User PostgreSQL MVCC Concurrency**:
  - PostgreSQL eliminates single-writer file lock bottlenecks.
  - High-throughput parallel reads and writes run simultaneously without collisions:
    - Batch pipelines and model trainers write to hypertables using binary `COPY` protocol.
    - React frontend and FastAPI asynchronously serve candlestick charts from continuous aggregate views (`silver_candles_5m`).
    - Multiple Jupyter notebooks query analytical data concurrently without blocking.


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
- **Corporate Actions Ingestion & Share Adjustment Periods**:
  ```bash
  .venv/bin/mdk-oracle load-corporate-actions
  ```
- **BIST 30 Membership Ingestion & Dynamic Constituent Tracking**:
  ```bash
  .venv/bin/mdk-oracle load-bist30
  ```
- **Daily Gold Layer Execution (Institutional Signals)**:
  ```bash
  .venv/bin/python scripts/run_pipeline.py --target gold
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


