---
name: mdk-institutional-flow-analysis
description: >-
  Domain knowledge, signal definitions, quantitative feature engineering, and predictive modeling workflows
  for tracking BIST institutional order flows (specifically Bank of America / BofA `MLB`). Use when building
  predictive models, designing feature clusters, executing walk-forward tournaments, or generating actionable
  trading signals for individual traders.
---

# MDK Institutional Flow Analysis & Signal Modeling Skill

This skill documents domain-specific metrics, institutional broker classifications, quantitative feature engineering, and probabilistic modeling architectures for Borsa Istanbul (BIST) centered on **Bank of America (BofA / `MLB`)** and domestic competitor desks.

---

## 1. Collaborative Interaction & Modeling Workflow

Whenever designing, extending, or refining the predictive modeling architecture:

1. **Plan Together First**:
   - Discuss the market microstructure hypothesis (e.g. "How does overnight carried inventory affect opening liquidity and morning liquidation?").
   - Align on the **Lean Feature Clusters** (zero data leakage from $T-1$ Close).
   - Agree on candidate model types and trader decision outputs (directional hit rate, playbooks, credible ranges).
2. **Review & Confirm ("Let's Go!")**:
   - Present a clear technical implementation plan.
   - Once approved by the user, execute end-to-end with unit tests, pipeline integrations, and interactive frontend verification.

---

## 2. Primary Institutional Target & Competitor Matrix

- **Primary Target**: **`MLB`** (Bank of America / Merrill Lynch Yatırım Bank A.Ş.) — dominant foreign algorithmic flow driver.
- **Top 5 Domestic Competitor Powerhouses**:
  - `IYM` (İş Yatırım) — largest domestic aggregator.
  - `YKR` (Yapı Kredi) — active institutional & prop desk.
  - `AKM` (Ak Yatırım) — high-volume domestic participant.
  - `GRM` (Garanti BBVA) — institutional liquidity provider.
  - `ZRY` (Ziraat) — public/state institution flow.
- **Retail Panic Indicator**:
  - `TRA` (Tera Yatırım) — key proxy for aggressive retail speculation and stop-loss cascading.

---

## 3. Institutional FIFO Tertip Mechanism & Microstructure Alpha (`INTRADAY_MATCHED_FIFO_V1`)

Volume alone is insufficient to identify institutional intentions. The lakehouse implements an exact point-in-time FIFO matching engine:
- **Intraday Match vs. Overnight Carry**: Day $T$ buy and sell executions are matched intraday at respective VWAPs ($\min(\text{buy}, \text{sell})$), generating `intraday_realized_pnl_tl`.
- **Residual Directional Queue**: The remaining daily net volume $(\text{buy} - \text{sell})$ creates new FIFO entries or consumes existing FIFO lots (`silver_broker_fifo_lots`).
- **Point-in-Time Daily History**: `silver_broker_fifo_daily` tracks the position state (`LONG`, `SHORT`, `FLAT`), open stock quantity, FIFO cost basis, and unrealized MTM PnL for every single session $T$.
- **Predictive Edge**: Knowing whether BofA or competitor desks enter session $T+1$ with heavily saturated inventory or deep unrealized gains/losses provides strong predictive signals for morning liquidation pressures, short squeezes, and defense accumulation.

---

## 4. Target Variables & Modeling Rigor

Targets are defined with mathematical rigor around continuous net flow (TL) and derived directional conviction:

| Target Variable | Data Type | Mathematical Formulation | Role & Description |
| :--- | :--- | :--- | :--- |
| `target_net_flow_tl` | Continuous (`float64`) | Net Flow(T, MLB) = Sum over trades(Buy Value - Sell Value) | **Primary Continuous Target**: Total net executed capital in TL by BofA for session T. |
| `target_direction` | Categorical (`str`) | Direction(T) = BUY if Net Flow > 0 else SELL | Derived directional binary outcome (`BUY` vs `SELL`). |

### Probabilistic Derivation:
1. **Primary Regression Target**: The forecaster fits continuous net flow targets, producing point forecast mu_hat and standard error sigma_hat.
2. **Directional Calibration**: Direction and conviction tiers are derived directly from the sign and magnitude of mu_hat.

---

## 5. The 21 Lean Feature Set (Zero Data Leakage)

All features are computed **strictly from prior completed windows / $T-1$ Close data** (18:10 TRT). A strict 12-month trailing training lookback window ($[T - 12\text{ months}, T-1]$) prevents temporal contamination:

1. **Tertip Inventory & Flow Posture**:
   - `feat_tertip_inventory_flow_yesterday_tl`: Net residual carried inventory flow from session $T-1$.
   - `feat_tertip_unrealized_pnl_yesterday_tl`: Carried mark-to-market unrealized PnL ($TL$).
   - `feat_tertip_carry_pnl_yesterday_tl`: Realized carry FIFO PnL from closing multi-day inventory.
   - `feat_tertip_intraday_pnl_yesterday_tl`: Intraday matched scalping PnL.
2. **Aggregate Net Institutional Flow & Execution Breakdown**:
   - `feat_total_inst_net_share_today`: Aggregate institutional net order flow (MLB + BIG5 + KAMU net flow as % of total market turnover). Eliminates false directional drift caused by symmetric two-way scalping on deadlock sessions.
   - Execution breakdown shares (buy, sell, realized PnL) for MLB, BIG5, and KAMU.
3. **Macro Rates, Regime Shocks & Carry Costs**:
   - `feat_macro_repo_rate_pct`: Prevailing Central Bank (TCMB) 1-week repo policy interest rate.
   - `feat_macro_daily_carry_cost_bps`: Daily cost of carry calculated as $\text{Rate} / 365$.
   - `feat_days_since_pos_shock`: Calendar sessions elapsed since the last positive index shock ($\ge +3\%$).
   - `feat_days_since_neg_shock`: Calendar sessions elapsed since the last negative index shock ($\le -3\%$).
4. **Benchmark Index (BIST 30) Momentum & Volatility**:
   - `feat_bist30_volatility_20d_pct`: 20-day annualized historical return volatility of `XU030`.
   - `feat_bist30_trend_vs_20d_sma_pct`: Distance of BIST 30 close price from its 20-day Simple Moving Average.
5. **Calendar Dynamics**:
   - `feat_day_of_week`: Day index (Monday = 0, Friday = 4).
   - `feat_is_monday`: Monday rebalancing indicator.
   - `feat_is_friday`: Friday institutional hedging / risk-off indicator.
6. **Univariate Baseline Momentum Prior**:
   - `feat_prophet_ret_today_pct`: Zero-leakage Prophet 1-step baseline forecast for session $T$ (trained strictly up to session $T-1$). Serves as an overextension/momentum-exhaustion gauge against actual institutional net order flow.

---

## 6. Walk-Forward Candidate Arena & 3-Criteria Tournament Champion Selection

To prevent overfitting and adapt to changing market regimes, model and lookback horizon selection is performed through the walk-forward tournament arena (`test_multi_horizon_180d_arena.py` / `export_tertip_180d_track.py`).

The **tournament run is the single source of truth** for crowning the optimal model and lookback horizon (3M, 6M, 12M) per stock, persisted to `config/tertip_crowned_models.yaml` and PostgreSQL `gold_tertip_daily_forecasts`.

```
                            12-MONTH TRAILING DATASET (e.g. 250 Sessions)
┌────────────────────────────────────────────────────────┬─────────────────────────────┐
│             Rich Historical Training Base              │  Trailing Arena Tournament  │
│                     [Day 1 … 220]                      │        [Day 221 … 250]      │
│                     (220 Days)                         │        (Last 30 Days)       │
└────────────────────────────────────────────────────────┴─────────────────────────────┘
                                                                    │
Step 1:  Train on [Day 1 … 220] ──► Predict Day 221 ────────────────┤ Out-of-Sample Result 1
Step 2:  Train on [Day 1 … 221] ──► Predict Day 222 ────────────────┤ Out-of-Sample Result 2
...                                                                 │
Step 30: Train on [Day 1 … 249] ──► Predict Day 250 ────────────────┘ Out-of-Sample Result 30
```

### Candidate Paradigms Benchmarked:
1. `XGBoost`: Gradient boosting decision trees capturing non-linear feature interactions with low variance. Exceptionally strong on high-beta momentum stocks (`ASELS`, `ASTOR`, `YKBNK`, `TUPRS`).
2. `LightGBM`: Leaf-wise gradient boosting optimized for split speed and asymmetric flows (`GARAN` 80.0%, `TOASO` 80.0%, `SAHOL`).
3. `BayesianRidge`: Analytical conjugate prior regression providing robust posterior distributions on liquid bank/conglomerate giants (`AKBNK`, `BIMAS`, `KCHOL`, `FROTO`).
4. `Huber`: Robust M-estimator resilient against outlier volatility spikes (`PETKM` 76.7%, `PGSUS`, `TCELL`).
5. `Ridge`: L2-regularized linear model with strong generalization and structural stability (`GUBRF`, `ISCTR`, `SASA`).
6. `Prophet`: Zero-leakage univariate prior baseline.

> [!IMPORTANT]
> **Strict Retirement of Deep Neural/Transformer Architectures (`TFT`)**:
> Experimental evaluation confirmed that deep sequence models like Temporal Fusion Transformer (TFT) impose severe runtime latency bottlenecks (>70 seconds per stock vs ~4 ms for tabular models) while suffering from regime drift over BIST order flow. Under proper multi-horizon lookback calibration, **`XGBoost (3m)` decisively beat TFT on `ASELS`** (66.7% directional hits vs 63.3%, 1.94% MAE vs 2.04%, and 78.9% significant move accuracy). Deep neural networks are permanently retired from production crowning in favor of lean, microsecond-latency tabular models.

### Automated BIST 30 Tournament Execution:
To benchmark all 30 BIST 30 stocks across 15 candidate configurations (5 models $\times$ 3 horizons) and crown models via composite loss:
```bash
.venv/bin/python scripts/run_full_bist30_crowned_tournament.py
```
This script acts as the automated tournament referee: it computes out-of-sample directional hits, hit MAE, and miss MAE over the trailing 30 evaluation sessions, selects the champion for each equity, writes the results to `config/tertip_crowned_models.yaml`, and synchronizes `gold_tertip_daily_forecasts`.

### Strategic Tournament Crowning vs. Daily Pipeline Execution:
1. **Periodic Strategic Crowning (`scripts/run_full_bist30_crowned_tournament.py`)**:
   - Run occasionally (e.g. quarterly, or after major macro interest rate or regulatory regime shifts).
   - Establishes the champion model and lookback horizon per stock in `config/tertip_crowned_models.yaml`.
2. **Daily Incremental Pipeline (`scripts/run_pipeline.py --target all`)**:
   - Run daily (or whenever newly arrived raw trades are ingested).
   - Consumes the crowned configuration, updates Bronze and Silver tables, automatically audits past completed days into `gold_tertip_walk_forward_backtests`, generates live $T+1$ opportunity snapshots in `gold_tertip_daily_forecasts`, and invalidates frontend caches for instant (~4 ms) trader workstation serving.

### The 3 Core Tournament Selection Criteria:
1. **Criterion 1: Percentage of Directional Hits (`hit_rate_pct`)**
   - Evaluated against a calibrated $\pm 0.25\%$ (25 bps) market consolidation deadband matching BIST equity tick microstructure.
   - Measures sign accuracy: correct directional calls / total sessions $\times 100.0$.
2. **Criterion 2: Percentage Price Error in Directional Hits (`hit_mae_pct`)**
   - Measures calibration accuracy on winning calls.
   - Predicting $+0.5\%$ when realized is $+1.5\%$ (error $1.0\%$) is rewarded significantly more than predicting $+0.5\%$ when realized is $+10.0\%$ (error $9.5\%$).
3. **Criterion 3: Percentage Price Error in Directional Misses (`miss_mae_pct`)**
   - Measures downside capital protection and severity of bad signals.
   - Predicting $+0.5\%$ when realized is $-0.5\%$ (error $1.0\%$) is far less damaging than predicting $+0.5\%$ when realized is $-5.0\%$ (error $5.5\%$, catastrophic false signal).

### Composite Tournament Loss Function:
```
Tournament Loss = (100.0 - hit_rate_pct) + 1.0 * hit_mae_pct + 2.5 * miss_mae_pct
```

The champion model and lookback horizon for each equity is selected by minimizing this composite loss, simultaneously prioritizing directional correctness, price calibration accuracy, and downside drawdown avoidance.

### Volatility Calibration & Sample Weighting Scheme:
In financial time series, quiet consolidation sessions (|Delta| < 1.0%) outnumber breakout/expansion sessions (|Delta| >= 1.0% and >= 2.0%) by 3:1 to 4:1. Traditional unweighted MSE loss penalizes errors on the abundant quiet days, causing models to consistently underestimate extreme moves.

To solve this distribution inequality without triggering false breakout alarms on quiet days, models support calibrated piecewise sample weighting during training:
- **Quiet Consolidation Sessions** (|Return| < 1.0%): Down-weighted to `0.75x`.
- **Significant Move Sessions** (1.0% <= |Return| < 2.0%): Elevated to `2.50x`.
- **Extreme Move Sessions** (|Return| >= 2.0%): Elevated to `3.00x`.

This expands the model's dynamic range and sensitivity on breakout days while maintaining strict capital preservation and low false-positive rates on consolidation days.

### Multi-Horizon Lookbacks (3M, 6M, 12M):
High-beta, defense, and momentum equities (e.g. ASELS, ASTOR, TRALT) perform drastically better with adaptive 3M (63 sessions) or 6M (126 sessions) lookbacks because distant 12-month regime data dilutes the current high-volatility algorithmic order flow. The tournament arena dynamically crowns the lookback horizon per stock:
- **3M Lookback (63 sessions)**: ~40% of BIST 30 equities (fast-adapting momentum & high-beta).
- **6M Lookback (126 sessions)**: ~33% of BIST 30 equities (cyclical & institutional rotation).
- **12M Lookback (252 sessions)**: ~27% of BIST 30 equities (macro-congruent large caps like ISCTR, TCELL, PETKM).

---

## 7. Multi-Horizon Swing Forecasting Architecture (3D, 5D, 10D, 15D, 30D)

While next-day (T+1) forecasts provide immediate opening tactical cues, swing traders operate over forward holding periods. The platform implements an institutional multi-horizon swing forecasting engine (`tertip_multi_horizon_forecaster.py`):

### A. Direct Horizon Forecasting vs. Recursive Chaining:
In quantitative finance, forecasting multiple horizons can follow two distinct paradigms:
1. **Recursive / Chained Forecasting (REJECTED)**:
   - Predicting 3D, feeding that prediction to estimate 5D, and cascading into 10D.
   - *Fatal Flaw*: Compounding error accumulation. Any estimation error on early horizons is magnified exponentially in subsequent horizons, leading to model hallucination.
2. **Direct Horizon Specialists (ADOPTED)**:
   - Every forward horizon (3D, 5D, 10D, 15D, 30D) is an independent specialist trained directly on actual, ground-truth market features as of session T Close.
   - For horizon N, the target is the terminal return:
     `Forward Terminal Return (%) = (Close_{t+N} - Close_t) / Close_t * 100.0`
     `Forward Target Price = Close_t * (1.0 + Pred_Return / 100.0)`
   - Cross-horizon confluence is evaluated only at the decision layer (post-inference).

### B. Zero-Drift Target Demeaning (Eliminating Secular Trend Contamination):
Equities that experienced multi-quarter secular bull runs (e.g. ASELS surging +530% from 71 TL to 375 TL over 18M) produce historical training samples where forward returns average +3% to +5% per 10 sessions.
- **The "Heatwave / Always-Sunny" Failure Mode**:
  - Unadjusted regression models fit an intercept reflecting past secular momentum:
    `Pred_Return = Historical_Mean (+5.0%) + Beta * Flow`
  - When institutions dump stock at cycle peaks (-2.0% flow impact), the unadjusted model outputs:
    `Pred_Return = +5.0% - 2.0% = +3.0% (BUY)`
  - The model stubbornly shouts BUY into cyclical drawdowns, creating severe false-positive buy traps (ASELS 10D win rate was trapped at 45.0%).
- **Mathematical Correction (Target Demeaning)**:
  - Before model fitting, the mean of historical training targets across the lookback window is subtracted:
    `y_train = y_train_raw - mean(y_train_raw)`
  - Live prediction evaluates deviation from the 0.0% neutral baseline:
    `Pred_Demeaned = 0.0% - 2.0% = -2.0% (SELL)`
- **Quantitative Rationale**: Demeaning strips obsolete secular drift, forcing the model to explain abnormal alpha purely through institutional net flow deltas (`MLB`, `BIG5`, `KAMU`) and momentum divergences.
- **Empirical Impact**:
  - ASELS 10-day hit rate surged from 45.0% to 60.0% (loss reduced from 92.26 to 72.34).
  - ASELS 5-day hit rate leaped to 71.7% (43/60 hits, loss 56.64).
  - AKBNK 10-day win rate was preserved at 81.7% (loss 48.81), and 15D/30D at 83.3%.

### C. Multi-Horizon Flow & Divergence Core Feature Clusters:
1. **Multi-Day Institutional Flow Shares**:
   - `feat_mlb_flow_3d_share`, `feat_mlb_flow_5d_share`, `feat_mlb_flow_10d_share`: BofA executed net flow as % of multi-day turnover.
   - `feat_total_inst_5d_share`: Combined institutional flow share (MLB + BIG5).
2. **Flow Divergence & Rollover Dynamics**:
   - `feat_mlb_flow_divergence_3d_10d`: Difference between fast 3-day and slow 10-day BofA flow share. Detects sudden institutional distribution at cycle peaks before 10-day moving averages turn down.
   - `feat_inst_flow_divergence_5d_10d`: Institutional consensus divergence.
   - `feat_ret_rollover_3d_10d`, `feat_ret_rollover_5d_20d`: Multi-day momentum deceleration.
3. **Decayed Flow & Volatility**:
   - `feat_mlb_ewma_flow_share_5d`: Exponentially-decayed BofA flow (half-life ~5 days).
   - `feat_stock_vol_10d`, `feat_stock_vol_20d`, `atr_14d_pct`: Multi-day realized volatility and ATR envelopes.

### D. The Lean Feature Space Principle (Ablation Findings):
During short-horizon tuning on AKBNK, ASELS, and ISCTR, candidate intraday ratio features were evaluated:
- `feat_mlb_w5_urgency` (Window 5 closing flow share vs day)
- `feat_price_vs_vwap_5d_pct` (Distance from 5-day rolling VWAP)
- `feat_ret_3d_vs_1d_accel` (Short-term acceleration tension)
- `feat_intraday_spread_pct` (Intraday high-low range)
- **Experimental Outcome**: Adding these 4 collinear features diluted regularized coefficients on short lookbacks, causing ASELS 5D to drop from 71.7% to 66.7% and ISCTR 5D to drop from 61.7% to 55.0%.
- **Architectural Decision**: Rejected the collinear additions. The lean 21-25 feature set provides the highest signal-to-noise ratio and superior out-of-sample generalization.

### E. Multi-Horizon Candidate Arena & Crowning (`scripts/run_multi_horizon_tournament.py`):
Evaluates candidate models (`BayesianRidge`, `LightGBM`, `XGBoost`, `Ridge`, `Huber`) across 5 training lookbacks (3M: 63s, 6M: 126s, 12M: 252s, 18M: 378s, 24M: 504s) for each of the 5 forward horizons.
- Champion selection minimizes 3-criteria composite tournament loss over trailing 60 evaluation sessions:
  `Tournament Loss = (100.0 - hit_rate_pct) + 1.0 * hit_mae_pct + 2.5 * miss_mae_pct`
- Crowned models persisted to `config/tertip_multi_horizon_crowned_models.yaml`, `gold_tertip_multi_horizon_forecasts`, and `gold_tertip_multi_horizon_backtests`.
- Fast execution via multiprocessing with single-threaded OpenMP/BLAS controls preventing Apple Silicon thread contention.

### F. Univariate Prophet Baseline Level Prior:
Prophet baseline models always predict actual price levels rather than percentage changes, providing a robust, non-distorted univariate foundation for multi-day horizons.

---

## 8. Actionable Decision Items, Empirical Quantiles & Institutional Trade Playbooks

Continuous flow forecasts are translated into tradeable decision items calibrated from empirical distribution thresholds:

### A. Empirical Percentile Conviction Tiers
- **`STRONG_BUY`**: y_hat >= P_85(historical buy flow)
- **`BUY`**: P_50 <= y_hat < P_85
- **`WEAK_BUY`**: P_25 <= y_hat < P_50
- **`NEUTRAL`**: |y_hat| < P_25
- **`WEAK_SELL`**: P_25 <= |y_hat| < P_50 and y_hat < 0
- **`SELL`**: P_50 <= |y_hat| < P_85 and y_hat < 0
- **`STRONG_SELL`**: |y_hat| >= P_85 and y_hat < 0

### B. Institutional Execution Playbooks
- **`SQUEEZE_LONG`**: Positive flow expectation with heavy competitor delta — follow aggressive opening accumulation.
- **`MOMENTUM_EXPANSION`**: Extreme opening accumulation (>= P_85) — institutional momentum continuation.
- **`BUY ABSORPTION REBOUND`**: Buying flow absorbing heavy retail selling pressure — expect sharp mean reversion bounce.
- **`STRONG SELL PRESSURE`**: Aggressive institutional distribution — reduce long exposure or enter tactical short.
- **`LIQUIDITY_FADE`**: High negative flow expectation with BofA holding > +5% unrealized gains — expect profit-taking and fade intraday dips.
- **`DEFENSE_SUPPORT`**: Underwater carried inventory (< -4% cost basis spread) with positive flow — institutional defense accumulation.
- **`NEUTRAL_WAIT`**: Sub-threshold flow (|y_hat| < P_25) — wait for intraday confirmation.

---

## 9. Trader Workstation & Gold Predictive Hub (`OracleHubDashboard.tsx`)

The quantitative signals are served directly through an interactive **React 18 + TradingView Lightweight Charts** trading workstation (`frontend/`):

1. **Live Upcoming Session Signal Card (T+1)**:
   - Displays upcoming trade date, forecasted net flow (TL), target price, price range [P_low, P_high], directional conviction badge, and institutional execution playbook.
2. **30-Day Performance Track Record**:
   - Header summary pill displays overall directional accuracy across the last 30 sessions (e.g. `20/30 (66.7%)`).
   - Detailed ledger table showing session-by-session predicted flow, actual realized flow, prediction error, and transparent `[HIT]` / `[MISS]` status badges.
3. **Walk-Forward Backtest Visualizer**:
   - Interactive chart plotting realized market flow vs. model predictions across historical sessions.

---

## 10. Predicted Opportunity Actions & Tactical Radar (`OpportunityActionsDashboard.tsx`)

A dedicated workstation tab designed for instant institutional pre-market and intraday opportunity discovery across all 30 BIST 30 constituent equities:

1. **Market Breadth & Bias Ribbon**:
   - Live constituent bias breakdown (e.g. 15 Bullish / 11 Bearish / 4 Consolidation).
   - High-conviction move tally (|Delta| >= 2.0%) and significant move tally (|Delta| >= 1.0%).
   - As-of session close date targeting upcoming T+1 execution.
2. **Dual Spotlight Decks (Highest Upside vs. Highest Downside)**:
   - **Top Bullish Setups**: Ranks top 5 equities by expected positive return, displaying target price, expected upside %, playbook strategy, crowned model/horizon, 30D win rate %, and BofA (MLB) net flow TL.
   - **Top Short / Fade Setups**: Ranks top 5 equities by expected downside, highlighting institutional distribution pressure and profit-taking fade targets.
3. **Full Universe Opportunity Matrix (All 30 Equities Retained & Highlighted)**:
   - All 30 equities are kept in a single unified matrix with tiered conviction badges:
     - **Tier 1: High Conviction** (|Expected Return| >= 2.0%): Standout glowing badges (`★ High Conviction`).
     - **Tier 2: Significant** (1.0% <= |Expected Return| < 2.0%): Actionable directional setups.
     - **Tier 3: Consolidation** (< 1.0%): Range-bound / watch setups.
   - Bidirectional visual return gauge centered at zero (rose for negative, emerald for positive).
   - Quick filter pills: `All Equities`, `Big Moves (|Δ| ≥ 2%)`, `Significant (|Δ| ≥ 1%)`, `Bullish Longs`, `Bearish Shorts`, `BofA Buying`, `BofA Selling`.
   - One-click launch actions to inspect any stock in **Candlestick & Order Flow** or **Gold Predictive Hub**.

---

## 11. Concrete Trader Execution Playbook Workflow

Individual and prop traders utilize the MDK Trading Oracle through a disciplined 3-step execution loop:

```
┌─────────────────────────────────┐
│ 1. Pre-Market Scan              │
│    Predicted Opportunity Actions│ ──► Identify Top 3 Longs & Top 3 Shorts
│    (Radar & Breadth Ribbon)     │     Check Market Breadth & Big Move Tally
└─────────────────────────────────┘
                 │
                 ▼
┌─────────────────────────────────┐
│ 2. Microstructure Deep-Dive     │
│    Gold Predictive Hub          │ ──► Verify Crowned Horizon (3M vs 6M vs 12M)
│    (Pillars, Inventory, 30D Win)│     Inspect FIFO Tertip Cost Basis & Carry PnL
└─────────────────────────────────┘
                 │
                 ▼
┌─────────────────────────────────┐
│ 3. Tactical Intraday Execution  │
│    Candlestick & Order Flow     │ ──► Window 1 (09:55-10:30 TRT): Opening confirmation
│    (Continuous Aggs, VWAP, Spreads)   Trade against Target Price Range [Low, High]
└─────────────────────────────────┘
```
