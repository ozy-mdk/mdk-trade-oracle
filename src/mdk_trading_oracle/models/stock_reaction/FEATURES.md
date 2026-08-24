# Model 3: BIST30 Stock Intraday Reaction Forecaster — Feature & Input Specification

## 1. Overview & Modeling Scope
The **BIST30 Stock Intraday Reaction Forecaster** predicts execution-aware intraday return percentages for individual BIST30 equities across three critical trading windows:
- **Window 2 (`first_reaction`, 10:30 – 11:30 TRT)**: Morning continuation or reversal following the opening auction.
- **Window 3 (`midday_followup`, 11:30 – 14:30 TRT)**: Midday trend development and institutional position building.
- **Window 5 (`closing_session`, 16:00 – 18:15 TRT)**: End-of-day resolution and closing auction dynamics.

All input features are strictly constructed from data available up to **$T_{\text{feature\_cutoff}} = 10:30\text{ TRT}$ (end of Window 1)** on session $T$, guaranteeing **zero lookahead bias / data leakage**.

---

## 2. Target Variables (Day $T$, Window $w \in \{\text{W2, W3, W5}\}$, Stock $i$)

| Target Variable | Type | Mathematical Formulation | Role & Description |
| :--- | :--- | :--- | :--- |
| `target_w2_return_pct` | `float64` | $y_{i, \text{W2}} = \frac{\text{VWAP}_{i, \text{W2}} - P_{i, \text{W1\_ref}}}{P_{i, \text{W1\_ref}}} \times 100$ | **Continuous Target (W2)**: Execution-aware return % from W1 reference price to W2 VWAP. |
| `target_w3_return_pct` | `float64` | $y_{i, \text{W3}} = \frac{\text{VWAP}_{i, \text{W3}} - P_{i, \text{W1\_ref}}}{P_{i, \text{W1\_ref}}} \times 100$ | **Continuous Target (W3)**: Execution-aware return % from W1 reference price to W3 VWAP. |
| `target_w5_return_pct` | `float64` | $y_{i, \text{W5}} = \frac{\text{VWAP}_{i, \text{W5}} - P_{i, \text{W1\_ref}}}{P_{i, \text{W1\_ref}}} \times 100$ | **Continuous Target (W5)**: Execution-aware return % from W1 reference price to W5 VWAP. |

*Reference Price $P_{i, \text{W1\_ref}}$*: BofA Window 1 Buy VWAP if available, falling back to Window 1 Market VWAP.

---

## 3. The 8 Quantitative Feature Clusters (47 Features)

### Cluster 1: BofA W1 Execution Signal (9 Features)
*Same-day opening auction and initial 35-minute execution footprint (09:55 – 10:30 TRT).*

| Feature Name | Type / Lag | Mathematical / SQL Formulation | Default / Coalesce | Microstructure & Behavioral Rationale |
| :--- | :--- | :--- | :--- | :--- |
| `feat_bofa_w1_buy_vol` | Flow (W1) | $\text{BuyVol}_{\text{MLB}, i, \text{W1}}$ | `0.0` | BofA W1 total bought share volume in stock $i$. |
| `feat_bofa_w1_sell_vol` | Flow (W1) | $\text{SellVol}_{\text{MLB}, i, \text{W1}}$ | `0.0` | BofA W1 total sold share volume in stock $i$. |
| `feat_bofa_w1_buy_tl` | Flow (W1) | $\text{BuyTurnover}_{\text{MLB}, i, \text{W1}}$ | `0.0` | BofA W1 gross buy turnover in TL. |
| `feat_bofa_w1_sell_tl` | Flow (W1) | $\text{SellTurnover}_{\text{MLB}, i, \text{W1}}$ | `0.0` | BofA W1 gross sell turnover in TL. |
| `feat_bofa_w1_net_flow_tl` | Flow (W1) | $\text{BuyTurnover}_{\text{MLB}} - \text{SellTurnover}_{\text{MLB}}$ | `0.0` | Net directional capital commitment by BofA in W1. |
| `feat_bofa_w1_net_vol` | Flow (W1) | $\text{BuyVol}_{\text{MLB}} - \text{SellVol}_{\text{MLB}}$ | `0.0` | Net lot inventory delta accumulated by BofA in W1. |
| `feat_bofa_w1_vol_share` | Ratio (W1) | $\frac{\text{Vol}_{\text{MLB}, \text{W1}}}{\text{Vol}_{\text{Market}, \text{W1}}}$ | `0.0` | BofA's share of total opening volume (pricing power). |
| `feat_bofa_w1_direction_sign` | Flag (W1) | $\text{sign}(\text{Net Flow}_{\text{MLB}, \text{W1}})$ | `0.0` | Directional sign (+1.0 buy, -1.0 sell, 0.0 neutral). |
| `feat_bofa_w1_market_vwap` | Price (W1) | $\frac{\text{Turnover}_{\text{Market}, \text{W1}}}{\text{Vol}_{\text{Market}, \text{W1}}}$ | `0.0` | Market-wide VWAP during W1 opening session. |

---

### Cluster 2: Multi-Broker W1 Alignment & Retail Contra-Signal (9 Features)
*Interaction between BofA, the Top-5 domestic market makers, and retail aggressive aggregator (TRA).*

| Feature Name | Type / Lag | Mathematical / SQL Formulation | Default / Coalesce | Microstructure & Behavioral Rationale |
| :--- | :--- | :--- | :--- | :--- |
| `feat_comp_w1_net_flow_tl` | Flow (W1) | $\sum_{b \in \text{Domestic5, TRA}} \text{Net Flow}_{b, i, \text{W1}}$ | `0.0` | Aggregate competitor net flow in stock $i$ in W1. |
| `feat_iym_w1_net_flow_tl` | Flow (W1) | $\text{Net Flow}_{\text{IYM}, i, \text{W1}}$ | `0.0` | İş Yatırım opening net flow. |
| `feat_ykr_w1_net_flow_tl` | Flow (W1) | $\text{Net Flow}_{\text{YKR}, i, \text{W1}}$ | `0.0` | Yapı Kredi opening net flow. |
| `feat_akm_w1_net_flow_tl` | Flow (W1) | $\text{Net Flow}_{\text{AKM}, i, \text{W1}}$ | `0.0` | Ak Yatırım opening net flow. |
| `feat_grm_w1_net_flow_tl` | Flow (W1) | $\text{Net Flow}_{\text{GRM}, i, \text{W1}}$ | `0.0` | Garanti BBVA opening net flow. |
| `feat_zry_w1_net_flow_tl` | Flow (W1) | $\text{Net Flow}_{\text{ZRY}, i, \text{W1}}$ | `0.0` | Ziraat opening net flow. |
| `feat_tra_w1_net_flow_tl` | Flow (W1) | $\text{Net Flow}_{\text{TRA}, i, \text{W1}}$ | `0.0` | Tera (retail aggressive order flow proxy) net flow. |
| `feat_w1_bofa_comp_alignment` | Flag (W1) | $\text{sign}(\text{Flow}_{\text{MLB}} \times \text{Flow}_{\text{Comp}})$ | `0.0` | +1.0 = aligned flow, -1.0 = institutional battle/divergence. |
| `feat_w1_bofa_tra_contra_signal` | Flag (W1) | $\text{BofA Buy} \land \text{TRA Sell} \to +1.0$ | `0.0` | Retail panic selling into BofA accumulation (bullish squeeze). |

---

### Cluster 3: T-1 Stock Momentum & Technical Posture (6 Features)
*Pre-existing stock trend, mean reversion distance, and historical volatility.*

| Feature Name | Type / Lag | Mathematical / SQL Formulation | Default / Coalesce | Microstructure & Behavioral Rationale |
| :--- | :--- | :--- | :--- | :--- |
| `feat_stock_ret_t1_1d` | Return ($T-1$) | $\text{Return}_{i, T-1}$ | `0.0` | Prior day adjusted close return %. |
| `feat_stock_ret_t1_5d` | Return ($T-1$) | $\sum_{k=1}^5 \text{Return}_{i, T-k}$ | `0.0` | Rolling 5-day cumulative return %. |
| `feat_stock_ret_t1_20d` | Return ($T-1$) | $\frac{P_{T-1}}{P_{T-21}} \times 100 - 100$ | `0.0` | Rolling 20-day cumulative return %. |
| `feat_stock_dist_sma20_t1` | Technical ($T-1$) | $\frac{P_{T-1}}{\text{SMA20}(P)} \times 100 - 100$ | `0.0` | % distance from 20-day simple moving average. |
| `feat_stock_vol_20d_t1` | Volatility ($T-1$) | $\sigma_{20d}(\text{Return}_i)$ | `0.0` | 20-day annualized return standard deviation. |
| `feat_stock_intraday_range_t1` | Technical ($T-1$) | $\frac{P_{\text{Close}} - P_{\text{Open}}}{P_{\text{Open}}} \times 100$ | `0.0` | Prior day intraday candle body %. |

---

### Cluster 4: T-1 Institutional FIFO Tertip & Inventory Posture (7 Features)
*Point-in-time open inventory positions, cost basis spread %, and unrealized PnL.*

| Feature Name | Type / Lag | Mathematical / SQL Formulation | Default / Coalesce | Microstructure & Behavioral Rationale |
| :--- | :--- | :--- | :--- | :--- |
| `feat_bofa_t1_open_qty` | Inventory ($T-1$) | $\text{OpenQty}_{\text{MLB}, i, T-1}$ | `0.0` | BofA open carry lot inventory. |
| `feat_bofa_t1_cost_spread_pct` | Basis ($T-1$) | $\frac{P_{T-1} - \text{AvgCost}_{\text{MLB}}}{\text{AvgCost}_{\text{MLB}}} \times 100$ | `0.0` | BofA unrealized profit/loss margin on inventory. |
| `feat_bofa_t1_unrealized_pnl_tl` | PnL ($T-1$) | $\text{UnrealizedPnL}_{\text{MLB}, i, T-1}$ | `0.0` | Total mark-to-market unrealized PnL in TL. |
| `feat_tra_t1_open_qty` | Inventory ($T-1$) | $\text{OpenQty}_{\text{TRA}, i, T-1}$ | `0.0` | Tera (retail) open carry lot inventory. |
| `feat_tra_t1_cost_spread_pct` | Basis ($T-1$) | $\frac{P_{T-1} - \text{AvgCost}_{\text{TRA}}}{\text{AvgCost}_{\text{TRA}}} \times 100$ | `0.0` | Tera retail margin (underwater retail = panic dump risk). |
| `feat_dom5_t1_open_qty` | Inventory ($T-1$) | $\sum_{b \in \text{Domestic5}} \text{OpenQty}_{b, i, T-1}$ | `0.0` | Domestic bank combined open inventory. |
| `feat_dom5_t1_unrealized_pnl_tl` | PnL ($T-1$) | $\sum_{b \in \text{Domestic5}} \text{PnL}_{b, i, T-1}$ | `0.0` | Domestic bank combined unrealized PnL in TL. |

---

### Cluster 5: T-1 Multi-Day Accumulation & Broker Flow Deltas (5 Features)
*Medium-term accumulation trends and multi-session institutional flow imbalances.*

| Feature Name | Type / Lag | Mathematical / SQL Formulation | Default / Coalesce | Microstructure & Behavioral Rationale |
| :--- | :--- | :--- | :--- | :--- |
| `feat_bofa_accum_5d_t1_tl` | Rolling ($T-1$) | $\sum_{k=1}^5 \text{Net Flow}_{\text{MLB}, i, T-k}$ | `0.0` | 5-day cumulative BofA net flow in TL. |
| `feat_bofa_accum_20d_t1_tl` | Rolling ($T-1$) | $\sum_{k=1}^{20} \text{Net Flow}_{\text{MLB}, i, T-k}$ | `0.0` | 20-day cumulative BofA net flow in TL. |
| `feat_bofa_flow_zscore_t1` | Statistical ($T-1$) | $\frac{\text{Flow}_{T-1} - \mu_{20d}}{\sigma_{20d}}$ | `0.0` | Standardized 20-day flow Z-score. |
| `feat_comp_accum_5d_t1_tl` | Rolling ($T-1$) | $\sum_{k=1}^5 \text{Net Flow}_{\text{Comp}, i, T-k}$ | `0.0` | 5-day cumulative competitor net flow in TL. |
| `feat_bofa_comp_delta_t1_tl` | Delta ($T-1$) | $\text{Flow}_{\text{MLB}, T-1} - \text{Flow}_{\text{Comp}, T-1}$ | `0.0` | Daily flow divergence between BofA and competitors. |

---

### Cluster 6: T-1 Sector Breadth & Peer Relative Spread (3 Features)
*Sector-wide tailwinds and idiosyncratic stock alpha vs industry peers.*

| Feature Name | Type / Lag | Mathematical / SQL Formulation | Default / Coalesce | Microstructure & Behavioral Rationale |
| :--- | :--- | :--- | :--- | :--- |
| `feat_sector_ret_t1` | Return ($T-1$) | $\text{Return}_{\text{Sector}, T-1}$ | `0.0` | Prior day sector return %. |
| `feat_sector_bofa_flow_t1` | Flow ($T-1$) | $\text{Net Flow}_{\text{MLB}, \text{Sector}, T-1}$ | `0.0` | Prior day BofA sector-level net flow. |
| `feat_peer_spread_t1` | Spread ($T-1$) | $\text{Return}_{i, T-1} - \text{Return}_{\text{Sector}, T-1}$ | `0.0` | Excess return over sector peer group (relative momentum). |

---

### Cluster 7: Macro Interest Rates & Carry Dynamics (4 Features)
*Monetary policy regime, carry costs, and repo rate shock dynamics.*

| Feature Name | Type / Lag | Mathematical / SQL Formulation | Default / Coalesce | Microstructure & Behavioral Rationale |
| :--- | :--- | :--- | :--- | :--- |
| `feat_macro_repo_rate_t1` | Rate ($T-1$) | $r_{\text{TCMB}, T-1}$ | `45.0` | TCMB 1-Week Repo policy interest rate %. |
| `feat_macro_rate_delta_t1` | Delta ($T-1$) | $\Delta r_{\text{TCMB}, T-1}$ | `0.0` | Rate change delta at latest MPC decision. |
| `feat_macro_days_since_decision_t1` | Decay ($T-1$) | $T - T_{\text{decision}}$ | `10.0` | Calendar days elapsed since last rate hike/cut. |
| `feat_macro_carry_t1` | Carry ($T-1$) | $\frac{r_{\text{TCMB}}}{360} \times 100$ | `0.125` | Daily funding carry cost percentage. |

---

### Cluster 8: Calendar & Temporal Seasonality (4 Features)
*Calendar effects, weekend positioning, and monthly settlement rhythms.*

| Feature Name | Type / Lag | Mathematical / SQL Formulation | Default / Coalesce | Microstructure & Behavioral Rationale |
| :--- | :--- | :--- | :--- | :--- |
| `feat_day_of_week` | Calendar | $\text{DayOfWeek}(T)$ | `1` | Day of week (1=Sunday/Monday index). |
| `feat_is_monday` | Binary | $\mathbb{I}\{\text{Monday}\}$ | `0` | Monday binary flag (weekend news digestion). |
| `feat_is_friday` | Binary | $\mathbb{I}\{\text{Friday}\}$ | `0` | Friday binary flag (weekend position de-risking). |
| `feat_day_of_month` | Calendar | $\text{DayOfMonth}(T)$ | `15` | Day of month (month-end rebalancing). |

---

## 4. Feature Summary & Verification Matrix

| Cluster Index | Cluster Name | Feature Count | Primary Source Table |
| :--- | :--- | :---: | :--- |
| 1 | BofA W1 Execution Signal | 9 | `silver_intraday_broker_window_summary` |
| 2 | Multi-Broker W1 Alignment | 9 | `silver_intraday_broker_window_summary` |
| 3 | T-1 Stock Momentum | 6 | `silver_daily_stock_summary` |
| 4 | T-1 Institutional FIFO Inventory | 7 | `silver_broker_fifo_daily` |
| 5 | T-1 Multi-Day Accumulation | 5 | `silver_daily_broker_summary` |
| 6 | T-1 Sector Breadth & Spread | 3 | `silver_daily_stock_summary`, `silver_daily_sector_summary` |
| 7 | Macro Interest Rates & Carry | 4 | `silver_daily_macro_rates` |
| 8 | Calendar & Temporal Seasonality | 4 | Inline Temporal Derivation |
| **Total** | **8 Semantic Clusters** | **47 Features** | **DuckDB Silver Lakehouse** |
