# Research Handoff — Stochastic Risk Engine, Data Fixes & Hit-Rate Experiments (2026-10-17 → 2026-10-18)

Context for continuing in a new session. Read `AGENTS.md` first; this note records what changed, what was measured,
what was rejected and why, and what is still open. All numbers are walk-forward / zero look-ahead unless stated.

---

## 1. TL;DR

- **Price ranges are now honest.** A GJR-GARCH-t + filtered-historical-simulation engine replaced the old
  `±1.645 × 20d vol` bands. Realized coverage of the "90%" range went from **37–52% (swing) / 78% (T+1)** to **~90%**.
- **Two data bugs fixed:** stock splits leaked into ML targets/features (raw prices), and the FIFO engine never applied
  splits to inventory. Both corrected; FIFO lots now restate on bonus/split events.
- **ZRY removed from BIG5** (it stays in KAMU), silver backfilled.
- **Volatility-based sizing is live:** forecast size now matches realized moves (realized/forecast ≈ 1.0 vs 0.44 before),
  T+1 forecasts >5% dropped from 16.7% to ~0% (none above ~7%). Published forecasts carry the raw model output
  (`model_expected_return_pct`), `sizing_confidence`, calibrated P(up/down) and a `conviction` label (LOW/MODERATE/HIGH).
- **Hit rates could not be improved with the current features.** 10 regressors, 6 classifier variants, confidence
  gating, market-relative targets and re-crowning all land at the same level. The key finding: **out of sample, the
  crowned tournament champions score ≈ random (45.6% vs 44.9% strict)**. Displayed hit rates (~54%+) are mostly
  selection luck from crowning on the same sessions that are displayed.
- **The hub's T+1 track record is inflated by a look-ahead leak** (74% displayed vs 55% clean; strict 70% vs 47%):
  ledger rows are re-created with a model that already trained on the outcome being graded (§4.5). Open — fix first.
- **Next step (new session): fix the leak, make track records honest, then find new information** (features) that
  carries real out-of-sample directional signal.

---

## 2. What was built / changed (code)

| Area | Change | Files |
|---|---|---|
| Stochastic risk engine | GJR-GARCH(1,1)-t on adjusted close-to-close log returns (EWMA fallback), mean-reverting horizon variance, FHS Monte Carlo, Ito-consistent centring, conformal per-horizon `scale` + `center_shrinkage`, trader `risk_profile` (90% band, median, P(up/down/flat), 1σ stop, P(target before stop), ES95) | `src/mdk_trading_oracle/models/stochastic_risk.py`, `scripts/calibrate_volatility_bands.py`, `config/volatility_calibration.yaml` |
| Integration | Bands + `risk_profile` in live T+1 and multi-horizon forecasts, gold tables (`risk_profile JSONB`), API (`RiskProfile` model) | `tertip_ml_forecaster.py`, `tertip_multi_horizon_forecaster.py`, `data/gold/*.py`, `api/app.py` |
| Split leakage in ML | Targets, realized returns, momentum/SMA/ATR/Bollinger and Prophet now use back-adjusted prices (`adjusted_price()`); raw close stays the price level and FIFO comparison basis | `tertip_ml_forecaster.py`, `tertip_multi_horizon_forecaster.py`, `gold/tertip_forecasts.py`, `tests/test_split_adjustment.py` |
| FIFO corporate actions | `apply_corporate_action()`: on the first session ≥ action date, open lots get qty×m, unit cost÷m, TL cost basis conserved; `quantity_factor_applied`, `opened_quantity_adjusted`, `quantity_factor_at_close`, `total_quantity_closed_adjusted` for lifecycle reconciliation | `tertip_engine.py`, `transformations.py`, `schema.py`, `docs/TERTIP_FIFO_MECHANISM.md` |
| ZRY grouping | `BIG_FIVE_BROKERS = (YKR, IYM, AKM, GRM)`; ZRY only in KAMU. `top_5_domestic_net_flow_tl` backfilled in `silver_daily_stock_summary` (verified new = old − ZRY flow) | `tertip_analytics.py`, `transformations.py`, configs, API/frontend labels, skill doc |
| Forecast sizing (production) | `size_forecast()`: direction from the crowned model, size = (0.75–1.25 × confidence) × conditional median move from the GJR-GARCH-t FHS, capped at the conditional 90th pct; T+1 > 5% compressed towards 7%. Confidence = percentile of signal-to-noise (|model| / horizon σ) vs the validated walk-forward reference (`sizing` section of `config/volatility_calibration.yaml`, fitted by `scripts/calibrate_forecast_sizing.py`). Risk distribution stays centred on the calibrated model output; target/stop odds use the published size. `conviction` = probability edge in the forecast direction (<2 pp LOW, <10 MODERATE, else HIGH). Applied to live T+1 (champion, pure ML, pure Prophet), gold T+1 and multi-horizon live forecasts. **Walk-forward backtests / tournaments still use `calibrate_expected_return`** (to be aligned in the ledger rebuild) | `models/stochastic_risk.py`, `tertip_ml_forecaster.py`, `tertip_multi_horizon_forecaster.py`, `gold/tertip_forecasts.py`, `api/app.py` |
| Test isolation | Tests that wrote to **production** (benchmark wipe + fake 2099 rows, BIST30 membership, corporate actions) now use sandbox schemas; `tests/conftest.py` drops the session's sandboxes. Fake rows deleted, 170 stale sandbox schemas dropped | `core/db.py`, `tests/*`, `tests/conftest.py` |

Commits so far on `develop`: `5670c6b`, `86dc3cc` (user), `6e6a30b`, `82e2c11`. **Uncommitted:** ZRY/BIG5 change, test
isolation fix, raw-feed locale fix (`ingestor.py`, `tests/data/test_raw_number_locale.py`),
`scripts/experiments/{direction_size_experiment,hit_rate_experiments,sizing_preview,ledger_leak_check}.py`, this note.
`config/brokers.yaml` / `config/instruments.yaml` diffs are only re-ordering from `--sync-catalog`.

## 3. Data state (as of 2026-10-18)

- Bronze/silver complete through **2026-10-16** (incl. FIFO restatement, ZRY backfill).
- Gold T+1 and multi-horizon: **all 29 stocks as of 2026-10-16**, rebuilt on clean data (locale fix, ZRY/BIG5) with
  volatility-based sizing (gold run of 2026-10-18). The T+1 ledger history still contains leaky rows (§4.5).

---

## 4. Measured results

### 4.1 Price-range coverage (nominal 90%), all BIST 30

| Horizon | Old coverage | New coverage | Old width | New width |
|---|---|---|---|---|
| 1D | 78.1% | 90.4% | 7.6% | 8.4% |
| 3D | 37.5% | 90.1% | 5.9% | 14.7% |
| 5D | 37.0% | 90.2% | 7.5% | 19.0% |
| 10D | 38.8% | 90.7% | 10.4% | 26.4% |
| 15D | 43.8% | 90.3% | 12.7% | 31.6% |
| 30D | 51.8% | 90.3% | 17.3% | 45.3% |

Old multi-horizon bands had a √(n/5) instead of √n bug and used open→close volatility. Under-covered stocks remaining:
TUPRS (78%), ASTOR (80%), SASA/PETKM (85%) → per-stock calibration is a possible refinement.

### 4.2 Forecast sizing (crowned regressor direction unchanged, REG_Z sizing, 29 stocks)

| | 1D | 3D | 5D | 10D | 15D | 30D |
|---|---|---|---|---|---|---|
| Median forecast, current | 3.41% | 4.36% | 5.96% | 8.48% | 9.00% | 9.02% |
| Median forecast, new | 1.56% | 2.70% | 3.53% | 5.08% | 6.33% | 9.07% |
| Realized ÷ forecast on hits, current | 0.44 | 0.65 | 0.66 | 0.71 | 0.87 | 1.08 |
| Realized ÷ forecast on hits, new | 0.99 | 1.03 | 1.09 | 1.02 | 1.07 | 0.97 |

Cause of the old behaviour: `calibrate_expected_return` leaves |raw| ≤ deadband near zero and inflates anything above it
to ≥ conviction × ATR (× √(n/2)), producing a bimodal "near zero or capped" distribution.

### 4.3 Hit-rate experiments (all rejected)

Strict accuracy = called direction AND move beyond the horizon deadband; random baseline ≈ 45%.

| Experiment | Result |
|---|---|
| 5 new regressors (SVR, ARD, ExtraTrees, CatBoost, monotone LGBM) | Same level; adding them to the tournament made champions slightly worse (more selection noise) |
| 3-class calibrated classifiers (LR / LGBM, calibrated, balanced, prior de-biased) | Worse than regressors (classifiers learn the trailing base rate) |
| Confidence gating (call 50–100% of days) | Precision flat across call rates — models do not know when they are right |
| Regressor + classifier agreement | +0.5–1.8 pp at some horizons, drops ~40% of calls; not worth it |
| Market-relative target (stock vs BIST30, β=1, per stock) | Worse; significant at 5D (−4.2 pp) and 10D (−2.6 pp) |
| Re-crowning on corrected data (75-config grid) | Out-of-sample champions −5.7 pp (split stocks) / −14.5 pp (control) vs current crowns |

### 4.4 The selection-bias finding (most important)

15 stocks, most recent ~90 sessions, strict %:

| Method | All horizons |
|---|---|
| Random | 44.9 |
| Crowned **before** the period (true out-of-sample) | 45.6 |
| Average of the 75 configs | 45.9 |
| Fixed Ridge/252/FIFO_CORE, trained per stock (no selection) | 47.8 (30D: 58.4) |
| Current crowns (selected on these sessions) | 54.3 |
| Best of 75 in hindsight | 63.4 |

Example: ASELS 10D — crowned BayesianRidge/378/FIFO_PLUS_YTD scored 73% in its crowning window and 35% in the
following period. GARAN 15D: 74% → 42%. The multi-horizon hub backtest shows the crowning window, so it displays the
"73%", not the "35%".

**Re-run on clean data (2026-10-18, after the locale fix and ZRY/BIG5 change):** all conclusions in 4.2–4.4 hold.
Current crowned regressor strict 47.1–54.5%; REG_Z sizing realized/forecast 0.92–1.02 with no T+1 forecast >5%;
market-relative target still worse; re-crowning: current crowns 59.3% (control, in-sample) vs 44.9% out-of-sample.
Six-stock live preview (ASELS, GARAN, AKBNK, YKBNK, ISCTR, TTKOM):
`~/data/mdk_oracle/experiments/direction_size/preview/live_preview.csv`.

### 4.5 Look-ahead leak in the T+1 "Audited Walk-Forward Reality Track" (OPEN — fix in next session)

**What the trader sees:** the Gold Predictive Hub's "Audited Walk-Forward Reality Track (1D)" (e.g. ASELS, XGBoost 18m)
shows ~70% directional hit rate and "Big Moves 11/13 (85%)". Those rows are not genuine out-of-sample forecasts.

**Where it happens:** `src/mdk_trading_oracle/data/gold/tertip_forecasts.py`, `update_gold_tertip_forecasts()`.
Each evening (close of session T) the job does two things with **one** freshly trained model:

1. *Live forecast for T+1* — trains on rows up to T-1 (`tr_full[f_cols].iloc[:-1]`; every label already realized)
   and predicts from row T's features. **Correct, no leak.**
2. *Ledger row for session T* ("what did we forecast for T, were we right?") — instead of reading the forecast that was
   actually published at the T-1 close, it **re-creates** it: predicts from row T-1's features with the same model.
   But row T-1 is in that model's training set **with its label r(T-1→T), which is exactly the outcome being graded**.

Training row = (features at day d) → (label = return d→d+1). Row T-1's label *is* session T's result.

**Concrete example (ASELS, evening run of 2026-10-16):**

| Last training rows | Label (answer) |
|---|---|
| row 2026-10-13 | return 13→14 Oct = −0.78% |
| row 2026-10-14 | return 14→15 Oct = +0.86% |
| **row 2026-10-15** | **return 15→16 Oct = −3.06% ← the 16 Oct outcome** |

| Ledger "forecast for 16 Oct" made by… | Forecast | Actual | Graded |
|---|---|---|---|
| production (row 15 Oct + its answer in training) | −0.23% | −3.06% | hit |
| what was knowable at the 15 Oct close (row 15 Oct excluded) | +0.14% | −3.06% | miss |

**Measured inflation** (`scripts/experiments/ledger_leak_check.py`, last 30 sessions, all 29 stocks, crowned configs):

| | Lenient hit (hub rule) | Strict on calls | Big-move sign |
|---|---|---|---|
| Leaky (production ledger) | 74.3% | 70.0% | 86.1% |
| Clean (row T-1 excluded) | 55.4% | 47.0% | 50.8% |
| Random baseline | ~50% | ~45% | ~50% |

By model (lenient, clean → leaky): XGBoost 52.2 → 74.1, LightGBM 51.3 → 72.0, Ridge 60.0 → 80.0, Huber 61.1 → 85.6,
BayesianRidge 63.3 → 71.7. ASELS: clean 33% (big moves 5/12) vs leaky 60% (10/12).

**Compounding effect — selection:** the T+1 crowning (`config/tertip_crowned_models.yaml`, calibrated 2026-10-12 on a
60-session window) picks the best of many configs on the same sessions the hub displays (26 of the 30 shown ASELS
sessions are inside the crowning window). See §4.4: crowned-before-the-period champions score ≈ random.

**Proposed fix (to implement and validate in the next session):**
1. **Grade stored forecasts, never re-create them.** The ledger row for session T = the T+1 forecast persisted at the
   T-1 close in `gold_tertip_daily_forecasts` (as_of_date = T-1) vs the realized close of T. Leak-free by construction
   and identical to what traders saw. Persist the exact model/config/version used with each forecast.
2. **Rebuild history** of `gold_tertip_walk_forward_backtests` with a clean walk-forward (model trained only on labels
   realized before the forecast origin) **and** simulated monthly crowning (configs chosen only from data before each
   month). Keep the old table as `_legacy` for comparison.
3. **Hub labelling:** "Live since crowning (date)" track vs "Simulated monthly crowning" backtest; never show the
   crowning window as a track record. Apply the same rule to the multi-horizon hub (180-session crowning ledger).
4. **Guard test:** assert that for every ledger row the training labels end strictly before the forecast origin.
5. Re-check the T+1 tournament code paths (`run_30d_walk_forward_arena` is zero-lookahead; the 180d/export scripts and
   any other "append latest session" code should be audited for the same pattern).

### 4.6 Raw-feed number-locale bug (fixed)

Files for 2026-10-14..16 use English number format ("68.65", "3,340") while the ingester assumed Turkish ("68,25",
"43.817"), so prices were ingested ×100. `_parse_csv` now detects the decimal separator per file
(`tests/data/test_raw_number_locale.py`); the three days were re-ingested and silver rebuilt. Gold rows computed on
2026-10-18 from those days are invalid until the next gold run.

### 4.7 Feature pipeline audit (focus stocks, since 2024)

- **No look-ahead in features.** No feature uses future rows; next-day correlations are small while same-day
  correlations of flow features are large (price impact), as expected.
- **Flows are computed correctly.** MLB net flow, turnover and volume recomputed from `bronze_raw_trades` match silver
  exactly (AKBNK, ASELS 2026-10-16).
- **Broker coverage is fine for liquid names.** MLB and all BIG5/KAMU desks trade the focus stocks on 99.7–100% of
  days. (`silver_broker_fifo_daily` only has rows on days a broker trades; inventory/cost features drop a desk to 0 on
  its non-trading days — matters for illiquid stocks only.)
- **FIXED 2026-10-18 (see below) — FIFO features ignored position side.** `extract_3pillar_time_series` sums `open_stock_quantity` and blends
  `fifo_avg_cost` across LONG and SHORT ledgers. Latest examples: YKBNK BIG5 = YKR long 244M + GRM short 200M counted as
  444M "inventory"; MLB short 176.7M in YKBNK counted as inventory. Share of broker-days SHORT since 2024: VKY 99%,
  TRA 97%, HLY 89%, GRM 44%, MLB 35%, IYM 26%, YKR 25%, AKM 18%, ZRY 10%. Affects cost spreads, tertip shares,
  profitable shares, inventory z-score. Shorts are largely artificial: the ledger starts flat in Jan 2022, so
  persistent net sellers (custodian-like flow) look permanently short. Fix direction: signed quantities, side-aware
  PnL (`unrealized_pnl_tl` is already side-aware) and position size relative to ADV instead of side-blind shares.
- **Fix implemented** (`institutional_positions()` in `tertip_ml_forecaster.py`, test in
  `tests/test_split_adjustment.py`): per-broker FIFO positions are carried forward to every trading day, quantities are
  signed (LONG +, SHORT −) and marked to market side-aware. Feature names kept, new meaning:
  `feat_{p}_cost_spread_pct` = group unrealized PnL / cost basis; `feat_{p}_tertip_share` = net signed position value
  in % of 20d median turnover; `feat_{p}_profitable_tertip_share` = % of gross position held by desks in profit;
  `feat_tertip_inventory_zscore` on MLB's signed position. YKBNK 2026-10-16: MLB −176.7M (short), BIG5 net +319M
  (was 719M side-blind), KAMU +227M. Effect: next-day IC of these features stays ≈ 0 (|IC| ≤ 0.023). Hit rates with
  the *crowned* configs drop (1D −4.1 pp, 10D −7.8 pp) because those configs were selected on the old features;
  with a fixed never-selected config (Ridge/252/FIFO_CORE) old vs new features are identical (47.6% → 47.6% strict,
  random 45.6%, no horizon significant). Conclusion: correct now, but FIFO inventory carries no next-day edge here;
  re-crown (or move to fixed configs) before trusting crowned numbers. Gold of 2026-10-16 was built with the OLD
  features — rerun gold after re-crowning.
- **Degenerate features:** `feat_mlb/big5/kamu_profitable_tertip_share` are 0 / at clip bounds 14–41% of days;
  `feat_kamu_post_shock_defense` is 0 on 68% of days.
- **Minor:** `feat_ret_today/yesterday`, shock-distance features use open→close (no overnight gap); YTD features are
  lagged one extra session; `lookback_days` in `extract_3pillar_time_series` is ignored (full history loaded).
- **Where the signal is:** next-day Spearman IC is tiny overall (median |IC| 0.014 across 68 features) but a small,
  **consistent** MLB-flow effect exists: `feat_mlb_turnover_intensity` 0.059, `feat_bofa_aggression_ratio` 0.058,
  `feat_mlb_ewma_flow_share_5d` 0.058, `feat_mlb_buy_share_today` 0.054 — same sign on 6/6 stocks (per-stock
  significance ≈ 0.076). Same-day IC of the same features is 0.4–0.53: **MLB flow is mostly priced in the same
  session**. Research leads: (a) intraday — early-window MLB/institutional flow vs the rest of the session;
  (b) a lean, fixed linear model on the few MLB-flow features instead of 35–44 features with tree models.
  Full table: `~/data/mdk_oracle/experiments/feature_ic_focus6.csv`.

---

## 5. Limitations & caveats

- Out-of-sample windows are short (~90–180 sessions) and from one market regime; multi-day horizons overlap, so 15D/30D
  rest on few independent periods.
- `center_shrinkage` and band `scale` are pooled per horizon across stocks; regime-dependent (first half κ≈0).
- Paid rights issues (bedelli, multiplier 1.0) and ticker renames (IPEKE→TRENJ, KOZAA→TRMET, KOZAL→TRALT) are not
  merged in FIFO.
- `tertip_analytics.py` next-day return still uses raw closes (descriptive only).
- Crowned configs in `config/*crowned*.yaml` were selected on pre-fix data.
- BIG5 features changed (ZRY removed) after all experiments in §4 were run; only the six-stock preview uses the new grouping.

## 6. Open decisions

0. **Fix the ledger look-ahead leak (§4.5)** — highest priority; all displayed T+1 track records are inflated.
1. **Honest hit-rate display:** show only sessions after the crowning date (live track) and/or a simulated monthly
   re-crowning backtest, instead of the crowning-window backtest.
2. **Replace 75-config crowning with a fixed configuration** (still trained per stock) — validate on all 29 stocks first.
3. ~~Promote sizing + probabilities to production~~ — **done 2026-10-18** (live forecasts only; backtests to follow).
   Open sub-questions: cap on swing horizons (30D can exceed ±10% for volatile names); frontend display of
   `conviction` / P(up/down) (API fields exist, UI not yet updated).
4. Full gold run — done 2026-10-18 on clean data; rerun after the ledger fix.

## 7. Agreed plan for the next phase (user, 2026-10-18)

Focus set: ASELS, GARAN, AKBNK, YKBNK, ISCTR, TTKOM.

1. **Leak fix + comparison on the focus set:** ledger rows grade the forecast stored at the previous close
   (`gold_tertip_daily_forecasts`), history rebuilt with a clean walk-forward; compare leaky vs clean per stock.
2. **Modelling / feature research on the focus set** — current direction skill is ≈ random out of sample.
   Honest protocol for every idea:
   - selection only on data before the scored period; strict direction accuracy vs random, per stock, with bootstrap CIs;
   - a fixed simple per-stock model first (e.g. Ridge/252) so gains come from information, not selection;
   - one feature family at a time, each with an institutional-behaviour hypothesis;
   - keep a family only if it beats random beyond the CI on most focus stocks.

## 8. Next research question (feature ideas)

What information is *not* in today's features but plausibly predicts direction out of sample? Ideas to evaluate with the
honest protocol (crown/select strictly before the scored period):
- Intraday timing of institutional flow (opening W1 vs closing W5 aggression, VWAP stretch) — previously rejected as
  noisy on short lookbacks; re-test with honest evaluation.
- Order-book / trade-size signatures (block prints, algorithmic slicing) from the tick data.
- Cross-stock / sector institutional rotation (still per-stock models, cross-sectional inputs).
- Derivatives (VİOP single-stock futures basis, open interest) if available.
- Event calendar (MPC meetings, index rebalancing, dividend/split dates).

## 9. How to reproduce

```bash
.venv/bin/python scripts/calibrate_volatility_bands.py --dry-run                     # band coverage report
.venv/bin/python scripts/experiments/model_candidate_arena.py --report-only          # model zoo comparison
.venv/bin/python scripts/experiments/direction_size_experiment.py --report-only      # sizing / gating / classifiers
.venv/bin/python scripts/experiments/hit_rate_experiments.py --part report           # re-crowning + relative target
.venv/bin/python scripts/experiments/sizing_preview.py --symbols ASELS,GARAN,AKBNK,YKBNK,ISCTR,TTKOM
.venv/bin/python scripts/experiments/ledger_leak_check.py                           # leaky vs clean T+1 ledger
```

Experiment outputs live under `~/data/mdk_oracle/experiments/` (`model_candidate_arena/`, `direction_size/`,
`hit_rate/`). Tests: `.venv/bin/pytest` is safe again (sandboxed); 90 passed before the
ZRY change.
