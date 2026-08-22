# Model 1: Macro Day-Start Forecaster — Feature & Input Specification

## 1. Overview & Modeling Scope
The **Macro Day-Start Forecaster** predicts Bank of America's (BofA / `MLB`) aggregate exchange-wide opening net flow ($TL$) during **Window 1 (09:55 – 10:30 TRT)** on trading day $T$.

All input features are strictly constructed from completed historical market sessions up to **$T-1$ Close (18:10 TRT)**, guaranteeing **zero lookahead bias / data leakage**.

---

## 2. Target Variables (Day $T$, Window 1)

| Target Variable | Type | Mathematical Formulation | Role & Description |
| :--- | :--- | :--- | :--- |
| `target_open_net_flow_tl` | `float64` | $y_T = \text{Buy Value}_{\text{MLB}, \text{W1}, T} - \text{Sell Value}_{\text{MLB}, \text{W1}, T}$ | **Continuous Target**: Net executed flow in TL across all liquid equities during Window 1. |
| `target_open_direction` | `str` | $\begin{cases} \text{BUY}, & \text{if } y_T > 0 \\ \text{SELL}, & \text{if } y_T \le 0 \end{cases}$ | **Binary Direction**: Derived directional label for classification and hit-rate scoring. |

---

## 3. The 8 Quantitative Feature Clusters (33 Features)

### Cluster 1: Prior Closing Window Momentum (Window 4: 17:00 – 18:10 TRT)
*Unfinished institutional Market-on-Close (MOC) and VWAP benchmark programs carry over into the next morning's opening auction.*

| Feature Name | Type / Lag | Mathematical / SQL Formulation | Default / Coalesce | Microstructure & Behavioral Rationale |
| :--- | :--- | :--- | :--- | :--- |
| `feat_bofa_w4_net_flow_tl` | Flow ($T-1$) | $\text{Buy}_{\text{MLB}, \text{W4}} - \text{Sell}_{\text{MLB}, \text{W4}}$ | `0.0` | Heavy late-day accumulation signals parent order continuation at market open. |
| `feat_bofa_w4_turnover_tl` | Flow ($T-1$) | $\text{Buy}_{\text{MLB}, \text{W4}} + \text{Sell}_{\text{MLB}, \text{W4}}$ | `0.0` | High closing turnover confirms high conviction behind the directional move. |
| `feat_w4_flow_acceleration_ratio` | Ratio ($T-1$) | $\frac{\text{Net Flow}_{\text{MLB}, \text{W4}}}{\|\text{Net Flow}_{\text{MLB}, \text{Day}}\| + \epsilon}$ | `0.0` | Proportion of daily net flow concentrated in final 70 mins ($> 0.5$ signals urgency). |

---

### Cluster 2: Multi-Day Inventory & Sector Saturation
*Algorithmic risk limits enforce multi-day portfolio exposure ceilings. Consecutive accumulation leads to exhaustion or rotation.*

| Feature Name | Type / Lag | Mathematical / SQL Formulation | Default / Coalesce | Microstructure & Behavioral Rationale |
| :--- | :--- | :--- | :--- | :--- |
| `feat_bofa_cum_net_flow_5d_tl` | Rolling ($T-1$) | $\sum_{k=1}^5 \text{Net Flow}_{\text{MLB}, T-k}$ | `0.0` | 5-day rolling net inventory. Extreme saturation triggers mean-reversion. |
| `feat_top5_cum_net_flow_5d_tl` | Rolling ($T-1$) | $\sum_{k=1}^5 \sum_{b \in \text{Top5}} \text{Net Flow}_{b, T-k}$ | `0.0` | 5-day rolling net inventory of top domestic brokerages (`IYM`, `YKR`, `AKM`, `GRM`, `ZRY`). |
| `feat_bofa_flow_zscore_20d` | Z-Score ($T-1$) | $\frac{\text{Flow}_{\text{MLB}, T-1} - \mu_{20d}}{\sigma_{20d} + \epsilon}$ | `0.0` | Statistical significance of yesterday's flow ($|Z| > 2.0$ marks institutional tail events). |
| `feat_bofa_banking_flow_prev_day` | Flow ($T-1$) | $\text{Net Flow}_{\text{MLB}, \text{Banking}, T-1}$ | `0.0` | BofA's previous-day net flow in the high-beta Banking sector. |
| `feat_bofa_transport_flow_prev_day` | Flow ($T-1$) | $\text{Net Flow}_{\text{MLB}, \text{Transportation}, T-1}$ | `0.0` | BofA's previous-day net flow in Transportation (`THYAO`, `PGSUS`). |
| `feat_bofa_holding_flow_prev_day` | Flow ($T-1$) | $\text{Net Flow}_{\text{MLB}, \text{Holding}, T-1}$ | `0.0` | BofA's previous-day net flow in Conglomerate Holdings (`KCHOL`, `SAHOL`). |
| `feat_bofa_energy_flow_prev_day` | Flow ($T-1$) | $\text{Net Flow}_{\text{MLB}, \text{Energy}, T-1}$ | `0.0` | BofA's previous-day net flow in Energy & Refining (`TUPRS`, `PETKM`). |
| `feat_bofa_defense_flow_prev_day` | Flow ($T-1$) | $\text{Net Flow}_{\text{MLB}, \text{Defense}, T-1}$ | `0.0` | BofA's previous-day net flow in Defense & Tech (`ASELS`). |

---

### Cluster 3: Institutional Cost Basis & Unrealized PnL
*The spread between current market price and BofA's 20-day buy VWAP governs profit-taking vs defense support behaviors.*

| Feature Name | Type / Lag | Mathematical / SQL Formulation | Default / Coalesce | Microstructure & Behavioral Rationale |
| :--- | :--- | :--- | :--- | :--- |
| `feat_bofa_cost_basis_spread_20d_pct` | Spread ($T-1$) | $\frac{\bar{P}_{\text{Close}, T-1} - \text{VWAP}_{\text{Buy}, 20d}}{\text{VWAP}_{\text{Buy}, 20d}} \times 100$ | `0.0` | $> +5\%$ unrealized gain prompts liquidity fades; $< -4\%$ prompts defense buying. |
| `feat_prev_day_close_vs_vwap_spread_pct` | Spread ($T-1$) | $\frac{\bar{P}_{\text{Close}, T-1} - \text{VWAP}_{\text{Market}, T-1}}{\text{VWAP}_{\text{Market}, T-1}} \times 100$ | `0.0` | Yesterday's closing price premium/discount relative to intraday market VWAP. |

---

### Cluster 4: Top-5 Domestic Competitor Closing Posture & Imbalance Delta
*Order flow is adversarial. Divergence between BofA and domestic market makers exposes liquidity squeezes.*

| Feature Name | Type / Lag | Mathematical / SQL Formulation | Default / Coalesce | Microstructure & Behavioral Rationale |
| :--- | :--- | :--- | :--- | :--- |
| `feat_top5_domestic_w4_net_flow_tl` | Flow ($T-1$) | $\sum_{b \in \text{Top5}} \text{Net Flow}_{b, \text{W4}, T-1}$ | `0.0` | Window 4 closing net flow of top domestic brokerages. |
| `feat_top5_domestic_prev_day_net_flow_tl` | Flow ($T-1$) | $\sum_{b \in \text{Top5}} \text{Net Flow}_{b, \text{Full Day}, T-1}$ | `0.0` | Full-day net flow of top domestic brokerages. |
| `feat_bofa_vs_top5_w4_flow_delta_tl` | Delta ($T-1$) | $\text{Flow}_{\text{MLB}, \text{W4}} - \text{Flow}_{\text{Top5}, \text{W4}}$ | `0.0` | Closing flow divergence ($> +30\text{M TL}$ signals foreign flow overpowering domestic resistance). |
| `feat_bofa_vs_top5_total_flow_delta_tl` | Delta ($T-1$) | $\text{Flow}_{\text{MLB}, \text{Day}} - \text{Flow}_{\text{Top5}, \text{Day}}$ | `0.0` | Full-day flow divergence between BofA and domestic competitors. |
| `feat_top5_banking_flow_prev_day` | Flow ($T-1$) | $\sum_{b \in \text{Top5}} \text{Net Flow}_{b, \text{Banking}, T-1}$ | `0.0` | Domestic institutions' posture in Banking on $T-1$. |

---

### Cluster 5: Institutional Hegemony & Market Concentration
*Quantifies institutional dominance over exchange volume. High concentration amplifies signal fidelity and reduces noise.*

| Feature Name | Type / Lag | Mathematical / SQL Formulation | Default / Coalesce | Microstructure & Behavioral Rationale |
| :--- | :--- | :--- | :--- | :--- |
| `feat_bofa_prev_day_net_flow_tl` | Flow ($T-1$) | $\text{Buy}_{\text{MLB}, T-1} - \text{Sell}_{\text{MLB}, T-1}$ | `0.0` | Total full-day net flow of BofA on $T-1$. |
| `feat_bofa_prev_day_turnover_tl` | Flow ($T-1$) | $\text{Buy}_{\text{MLB}, T-1} + \text{Sell}_{\text{MLB}, T-1}$ | `0.0` | Total full-day gross turnover generated by BofA on $T-1$. |
| `feat_bofa_prev_day_market_share` | Ratio ($T-1$) | $\frac{\text{Turnover}_{\text{MLB}, T-1}}{\text{Turnover}_{\text{Market}, T-1}}$ | `0.0` | BofA's volume market share of total exchange liquidity. |
| `feat_bofa_prev_day_turnover_rank` | Rank ($T-1$) | $\text{DenseRank}(\text{Turnover}_{T-1})$ | `10` | BofA liquidity rank across all active brokerages (1 = most active). |
| `feat_institutional_hegemony_share` | Ratio ($T-1$) | $\frac{\text{Turnover}_{\text{MLB}} + \text{Turnover}_{\text{Top5}}}{\text{Turnover}_{\text{Market}}}$ | `0.0` | Combined market share of BofA + Top 5 domestic institutions. |
| `feat_avg_cr5_concentration` | Ratio ($T-1$) | $\frac{1}{N}\sum_{s=1}^N \text{CR5}_s$ | `0.0` | Cross-sectional 5-broker volume concentration across all liquid stocks. |

---

### Cluster 6: Sector Cross-Sectional Stress & Volatility Breadth
*Market-wide dispersion and volatility dictate whether algorithms operate in risk-on expansion or risk-off deleveraging.*

| Feature Name | Type / Lag | Mathematical / SQL Formulation | Default / Coalesce | Microstructure & Behavioral Rationale |
| :--- | :--- | :--- | :--- | :--- |
| `feat_market_avg_return_pct` | Return ($T-1$) | $\frac{1}{N}\sum_{s=1}^N \frac{P_{s, \text{Close}} - P_{s, \text{Prev}}}{P_{s, \text{Prev}}} \times 100$ | `0.0` | Average daily return across all liquid tracked equities. |
| `feat_market_avg_range_pct` | Volatility ($T-1$) | $\frac{1}{N}\sum_{s=1}^N \frac{P_{s, \text{High}} - P_{s, \text{Low}}}{P_{s, \text{Low}}} \times 100$ | `0.0` | Average high-low price range across equities (intraday dispersion proxy). |

---

### Cluster 7: Calendar Dynamics & Seasonality
*Institutional mandates follow recurring calendar schedules (Monday new-week asset allocations vs Friday delta hedging).*

| Feature Name | Type / Lag | Mathematical / SQL Formulation | Default / Coalesce | Microstructure & Behavioral Rationale |
| :--- | :--- | :--- | :--- | :--- |
| `day_of_week` | Integer ($T$) | $\text{DayOfWeek}(\text{trade\_date})$ | `1..5` | Day of week integer ($1 = \text{Monday}, \dots, 5 = \text{Friday}$). |
| `is_monday` | Binary ($T$) | $\text{day\_of\_week} = 1$ | `FALSE` | Monday session indicator (high initial capital re-allocation). |
| `is_friday` | Binary ($T$) | $\text{day\_of\_week} = 5$ | `FALSE` | Friday session indicator (weekend exposure reduction / delta hedging). |

---

### Cluster 8: Macro Interest Rate Dynamics & Shock Impulse
*Monetary policy benchmarks equity cost of capital and margin financing costs. Rate decisions trigger immediate decay-weighted shocks.*

| Feature Name | Type / Lag | Mathematical / SQL Formulation | Default / Coalesce | Microstructure & Behavioral Rationale |
| :--- | :--- | :--- | :--- | :--- |
| `feat_macro_interest_rate` | Level ($T-1$) | $\text{Rate}_{T-1}$ (TCMB 1-Week Repo %) | `45.0` | Baseline financing benchmark and equity hurdle rate. |
| `feat_macro_rate_shock_decay` | Impulse ($T-1$) | $\frac{\Delta\text{Rate}_{\text{bps}, T-1}}{\max(1, \text{days\_since\_change}_{T-1})}$ | `0.0` | Decay-weighted rate shock ($100\%$ on $T=0,1$, decaying at $1/d$ on $T \ge 2$). |
| `feat_macro_rate_spread_vs_30d_mean` | Delta ($T-1$) | $(\text{Rate}_{T-1} - \overline{\text{Rate}}_{30d, T-1}) \times 100$ | `0.0` | Policy stance vs 30-day moving average (monetary tightening/easing). |
| `feat_macro_daily_carry_cost_bps` | Level ($T-1$) | $\frac{\text{Rate}_{T-1}}{365} \times 100\text{ (bps)}$ | `125.0` | Overnight carry and financing cost of holding long equity exposure. |

---

### Cluster 9: Benchmark Index (BIST 30) Momentum & Volatility
*Official BIST 30 benchmark dynamics capture market-wide beta, index arbitrage pressure, and broad momentum follow-through.*

| Feature Name | Type / Lag | Mathematical / SQL Formulation | Default / Coalesce | Microstructure & Behavioral Rationale |
| :--- | :--- | :--- | :--- | :--- |
| `feat_bist30_prev_day_return_pct` | Return ($T-1$) | $\frac{P_{\text{Close}, T-1} - P_{\text{Close}, T-2}}{P_{\text{Close}, T-2}} \times 100$ | `0.0` | Close-to-close return of the BIST 30 index. Strong closing momentum triggers morning basket orders. |
| `feat_bist30_prev_day_intraday_return_pct` | Return ($T-1$) | $\frac{P_{\text{Close}, T-1} - P_{\text{Open}, T-1}}{P_{\text{Open}, T-1}} \times 100$ | `0.0` | Intraday trend from Open to Close on $T-1$, measuring day-session conviction. |
| `feat_bist30_prev_day_range_pct` | Volatility ($T-1$) | $\frac{P_{\text{High}, T-1} - P_{\text{Low}, T-1}}{P_{\text{Low}, T-1}} \times 100$ | `0.0` | Intraday trading range of BIST 30 index measuring market volatility and price expansion. |
| `feat_bist30_cum_return_5d` | Momentum ($T-1$) | $\frac{P_{\text{Close}, T-1} - P_{\text{Close}, T-6}}{P_{\text{Close}, T-6}} \times 100$ | `0.0` | 5-day rolling cumulative return of the benchmark index. |
| `feat_bist30_trend_vs_20d_sma` | Trend ($T-1$) | $\frac{P_{\text{Close}, T-1}}{\text{SMA}_{20}(P_{\text{Close}, T-1})} - 1.0$ | `0.0` | Benchmark price relative to its 20-day Simple Moving Average. |
| `feat_bist30_volatility_20d` | Volatility ($T-1$) | $\sigma_{20d}(\text{daily\_returns})$ | `0.0` | 20-day rolling annualized/daily volatility of BIST 30 returns (risk regime classifier). |

