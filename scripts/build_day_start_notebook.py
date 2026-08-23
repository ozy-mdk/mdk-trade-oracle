"""Builds the 03_bofa_day_start_modeling.ipynb notebook."""

import nbformat as nbf

nb = nbf.v4.new_notebook()

# Metadata
nb.metadata = {
    "kernelspec": {"display_name": "Python 3.9 (mdk-trading-oracle)", "language": "python", "name": "python3"},
    "language_info": {"name": "python", "version": "3.9.5"},
}

cells = [
    nbf.v4.new_markdown_cell(r"""# MDK Trading Oracle — Gold Layer Model 1: Day-Start Institutional Forecaster
### *"How Will Bank of America (BofA / `MLB`) Start the Day?"*

---

## 1. Model Objective & Quantitative Mission

On **Borsa Istanbul (BIST)**, the first 30 minutes of continuous trading (**Window 1 `day_start`**: 09:55 – 10:30 TRT / 07:55 – 08:30 UTC) dictate the opening trend and liquidity posture of the entire market. 

Foreign institutional algorithms—chiefly **Bank of America (Clearing Code: `MLB`)**—account for 15–25%+ of total market liquidity and execute massive programmatic flows.

### Primary Modeling Goal
Predict BofA's **directional conviction**, **net flow magnitude (TL)**, and **execution playbook** for the upcoming morning opening auction (**Window 1**) using **strictly $T-1$ Close data (zero lookahead leakage)** before the market opens.

---

## 2. Target Variables & Multi-Target Architecture

The model trains against actual executed trade metrics extracted from DuckDB Silver table `silver_intraday_broker_window_summary` for **Window 1 (`day_start`)**:

| Target Variable | Data Type | Mathematical Formulation | Role & Description |
| :--- | :--- | :--- | :--- |
| `target_open_net_flow_tl` | Continuous (`float64`) | $$\text{Net Flow}_{T, \text{W1}} = \sum_{i \in \text{Trades}_{T, \text{W1}, \text{MLB}}} (\text{Buy Value}_i - \text{Sell Value}_i)$$ | **Primary Training Target ($y$)**: Total net executed capital in Turkish Lira (TL) by BofA across the market in Window 1. |
| `target_open_direction` | Categorical (`str`) | $$\text{Direction}_T = \begin{cases} \text{BUY}, & \text{if } \text{Net Flow}_{T, \text{W1}} > 0 \\ \text{SELL}, & \text{if } \text{Net Flow}_{T, \text{W1}} \le 0 \end{cases}$$ | **Derived Target**: Directional binary outcome (`BUY` vs `SELL`) computed directly from posterior probability. |

*(Note: Sector-specific opening net flows and allocations are forecasted in **Model 2: Sector Day-Start Forecaster**).*

---

### Business Overview: Why Unified Modeling Matters for the Trading Desk

In live market operations, individual traders face an overwhelming amount of disconnected data. Standard machine learning systems often train separate, independent models for each variable—e.g. one model predicting volume, another predicting Buy/Sell direction, and a third predicting net flow. 

This creates two critical flaws on the trading desk:
1. **Contradictory Signals**: An isolated classifier might say "BUY" while a separate volume model predicts negative net flow.
2. **Missing Risk Bands**: A simple Buy/Sell arrow does not tell the trader how confident the algorithm is or what the potential downside risk is.

**Our Approach**:
We train a **Single Primary Probabilistic Model** on continuous net flow ($\text{TL}$) and mathematically derive directional conviction, confidence percentages, and trade playbooks from the exact same posterior distribution. 

* **No Contradictions**: Flow magnitude, sign, confidence, and execution playbooks are always 100% harmonized.
* **Smart Position Sizing**: The trader sees the exact expected capital flow (e.g. $+42.5\text{M TL}$), the probability score (e.g. $84.2\%$ conviction), and the $90\%$ credible range (e.g. $[+15\text{M}, +70\text{M TL}]$) to size risk before the opening bell.

---

### Technical Architecture: How Multi-Target Forecasting is Managed

```
                               ┌────────────────────────────────────────────────────────┐
                               │  Primary Model Target: target_open_net_flow_tl (y)     │
                               │  (Trained via Bayesian Ridge / PyMC GLM / LightGBM)   │
                               └───────────────────────────┬────────────────────────────┘
                                                           │
                                   Outputs: Mean (μ̂) + Posterior Uncertainty (σ̂)
                                                           │
        ┌──────────────────────────────────────────────────┼──────────────────────────────────────────────────┐
        ▼                                                  ▼                                                  ▼
┌───────────────────────────────┐          ┌───────────────────────────────┐          ┌───────────────────────────────┐
│ 1. Continuous Net Flow        │          │ 2. Direction & Confidence     │          │ 3. Institutional Playbook     │
│ Point Forecast: μ̂             │          │ Direction = BUY if μ̂ > 0      │          │ Derived from:                 │
│ 90% Range: [μ̂ ± 1.645σ̂]      │          │ P(BUY) = 1 - Φ(0; μ̂, σ̂)       │          │ • μ̂ (Predicted Flow)          │
│                               │          │ Confidence = max(P, 1-P)      │          │ • Competitor W4 Delta         │
│                               │          │ Class: STRONG_ACCUMULATE, etc.│          │ • 20d Cost Basis Spread       │
└───────────────────────────────┘          └───────────────────────────────┘          └───────────────────────────────┘
```

1. **Direct Regression Training**:
   The model is trained strictly on $y = \text{target\_open\_net\_flow\_tl}$. The fitted model outputs both the point forecast $\hat{\mu}$ (mean flow in TL) and the posterior standard deviation $\hat{\sigma}$.
2. **Exact 90% Credible Range**:
   $$\text{Credible Range}_{90\%} = \big[\hat{\mu} - 1.645\hat{\sigma},\; \hat{\mu} + 1.645\hat{\sigma}\big]$$
3. **Probabilistic Direction & Conviction Formulation**:
   Directional probability is calculated using the Gaussian Cumulative Distribution Function ($\Phi$):
   $$P(\text{BUY}) = P(\text{Flow} > 0) = 1 - \Phi\left(\frac{0 - \hat{\mu}}{\hat{\sigma}}\right)$$
   $$\text{Direction Confidence} = \max\Big(P(\text{BUY}), 1 - P(\text{BUY})\Big)$$
4. **Directional Conviction Levels**:
   * $\hat{\mu} > +50\text{M TL} \text{ and Confidence} \ge 70\% \implies$ **`STRONG_ACCUMULATE`**
   * $\hat{\mu} > +10\text{M TL} \implies$ **`ACCUMULATE`**
   * $-10\text{M} \le \hat{\mu} \le +10\text{M TL} \implies$ **`NEUTRAL`**
   * $\hat{\mu} < -10\text{M TL} \implies$ **`DISTRIBUTE`**
   * $\hat{\mu} < -50\text{M TL} \text{ and Confidence} \ge 70\% \implies$ **`STRONG_DISTRIBUTE`**
5. **Ground-Truth Benchmark Targets**:
   `target_open_turnover_tl` and `target_open_market_share` are extracted in the Silver tables as ground-truth metrics for walk-forward performance auditing and position sizing baseline calculations.

---

## 3. The 7 Quantitative Feature Clusters (Input Features & Formulas)

All 26 predictive features are constructed **strictly from historical data up to $T-1$ Close** (or completed prior sessions) with **zero data leakage**.

```
┌──────────────────────────────────────────────────────────────────────────────────────────────────┐
│                      T-1 Trading Session (Close at 18:10 TRT)                                    │
│  [W4 Momentum] + [Multi-Day Inventory] + [Cost Basis PnL] + [Competitor Delta] + [Calendar]     │
└───────────────────────────────────────────────┬──────────────────────────────────────────────────┘
                                                │ (Zero Data Leakage / Lagged by 1 Day)
                                                ▼
┌──────────────────────────────────────────────────────────────────────────────────────────────────┐
│                   Day T Opening Session Window 1 (09:55 - 10:30 TRT)                             │
│       Forecast: Predicted Net Flow (TL), 90% Credible Range, Conviction, Playbook               │
└──────────────────────────────────────────────────────────────────────────────────────────────────┘
```

### Cluster 1: Prior Closing Window Momentum (Window 4: 17:00 – 18:10 TRT)
* **Rationale**: Institutional Market-on-Close (MOC) and VWAP benchmark programs unfinished by 18:10 TRT routinely carry over into the next morning's opening auction.
* **Key Features**:
  1. `feat_bofa_w4_net_flow_tl`: Net flow (TL) executed by BofA during Window 4 on $T-1$.
     $$\text{W4 Net Flow}_{T-1} = \text{Buy Value}_{\text{MLB}, \text{W4}, T-1} - \text{Sell Value}_{\text{MLB}, \text{W4}, T-1}$$
     *Expectation*: Positive W4 flow indicates strong opening continuation buying; heavy late selling carries into morning weakness.
  2. `feat_bofa_w4_turnover_tl`: Total turnover (TL) of BofA in Window 4 on $T-1$.
     *Expectation*: Higher volume confirms conviction behind the closing move.
  3. `feat_w4_flow_acceleration_ratio`: Proportion of full-day flow concentrated in the final 70 minutes.
     $$\text{Flow Acceleration Ratio}_{T-1} = \frac{\text{Net Flow}_{\text{MLB}, \text{W4}, T-1}}{|\text{Net Flow}_{\text{MLB}, \text{Full Day}, T-1}| + \epsilon}$$
     *Expectation*: Values $> 0.5$ indicate aggressive end-of-day parent order urgency.

### Cluster 2: Multi-Day Inventory & Sector Saturation
* **Rationale**: Algorithms operate under strict portfolio risk ceilings. After 3–5 consecutive days of net accumulation, exposure limits trigger rebalancing or mean-reversion.
* **Key Features**:
  4. `feat_bofa_cum_net_flow_5d_tl`: 5-day rolling cumulative net flow of BofA (TL).
     $$\text{Cum Net Flow}_{5d, T-1} = \sum_{k=1}^{5} \text{Net Flow}_{\text{MLB}, T-k}$$
  5. `feat_top5_cum_net_flow_5d_tl`: 5-day rolling cumulative net flow of Top 5 domestic powerhouses (`IYM`, `YKR`, `AKM`, `GRM`, `ZRY`).
     $$\text{Top5 Cum Net Flow}_{5d, T-1} = \sum_{k=1}^{5} \sum_{b \in \text{Top5}} \text{Net Flow}_{b, T-k}$$
  6. `feat_bofa_flow_zscore_20d`: Statistical Z-score of BofA's $T-1$ net flow relative to its 20-day historical distribution.
     $$Z_{20d, T-1} = \frac{\text{Net Flow}_{\text{MLB}, T-1} - \mu_{20d}(\text{Net Flow}_{\text{MLB}})}{\sigma_{20d}(\text{Net Flow}_{\text{MLB}}) + \epsilon}$$
     *Expectation*: $|Z| > 2.0$ represents statistical tail events (extreme institutional pressure).
  7. `feat_bofa_banking_flow_prev_day`, `feat_bofa_transport_flow_prev_day`, `feat_bofa_holding_flow_prev_day`, `feat_bofa_energy_flow_prev_day`, `feat_bofa_defense_flow_prev_day`: BofA net flow by specific sector on $T-1$.
     *Expectation*: Detects sector rotation (e.g. rotating out of Banking into Transportation).

### Cluster 3: Institutional Cost Basis & Unrealized PnL
* **Rationale**: The distance between current price and BofA's 20-day Volume-Weighted Buy Price (VWAP) dictates institutional behavior. Large profits invite profit-taking; underwater positions trigger defense.
* **Key Features**:
  8. `feat_bofa_cost_basis_spread_20d_pct`: Percentage spread between yesterday's market close and BofA's 20-day rolling Buy VWAP.
     $$\text{Cost Basis Spread}_{20d, T-1} = \frac{\bar{P}_{\text{Close}, T-1} - \text{VWAP}_{\text{Buy}, 20d, T-1}}{\text{VWAP}_{\text{Buy}, 20d, T-1}}$$
     where $\text{VWAP}_{\text{Buy}, 20d} = \frac{\sum_{k=1}^{20} \text{Buy Turnover}_{T-k}}{\sum_{k=1}^{20} \text{Buy Volume}_{T-k}}$.
     *Expectation*:
     * Spread $> +5\%$ (High Profit): Algorithmic profit-taking / liquidity fade.
     * Spread $< -4\%$ (Underwater Inventory): Institutional defense support buying.
  9. `feat_prev_day_close_vs_vwap_spread_pct`: Spread between $T-1$ Close and $T-1$ Market VWAP.
     $$\text{Close vs VWAP Spread}_{T-1} = \frac{\bar{P}_{\text{Close}, T-1} - \text{VWAP}_{\text{Market}, T-1}}{\text{VWAP}_{\text{Market}, T-1}}$$

### Cluster 4: Top-5 Domestic Competitor Closing Posture & Imbalance Delta
* **Rationale**: Order flow is adversarial. BofA trades directly against domestic liquidity providers (`IYM`, `YKR`, `AKM`, `GRM`, `ZRY`).
* **Key Features**:
  10. `feat_top5_domestic_w4_net_flow_tl`: Net flow of Top-5 domestic brokers in Window 4 on $T-1$.
  11. `feat_top5_domestic_prev_day_net_flow_tl`: Total daily net flow of Top-5 domestic brokers on $T-1$.
  12. `feat_bofa_vs_top5_w4_flow_delta_tl`: Window 4 flow divergence between BofA and domestic powerhouses.
      $$\Delta \text{W4 Flow}_{T-1} = \text{Net Flow}_{\text{MLB}, \text{W4}, T-1} - \text{Net Flow}_{\text{Top5}, \text{W4}, T-1}$$
      *Expectation*: A large delta ($> +30\text{M TL}$) signals an institutional squeeze where foreign flow overwhelms domestic resistance.
  13. `feat_bofa_vs_top5_total_flow_delta_tl`: Full-day flow divergence between BofA and Top-5 brokers on $T-1$.
  14. `feat_top5_banking_flow_prev_day`: Domestic institutions' flow in Banking on $T-1$.

### Cluster 5: Institutional Hegemony & Market Concentration
* **Rationale**: Quantifies institutional dominance over the market. High institutional market share minimizes noise and amplifies signal reliability.
* **Key Features**:
  15. `feat_bofa_prev_day_net_flow_tl`: Full-day net flow of BofA on $T-1$.
  16. `feat_bofa_prev_day_turnover_tl`: Total turnover generated by BofA on $T-1$.
  17. `feat_bofa_prev_day_market_share`: BofA's percentage of total exchange turnover on $T-1$.
      $$\text{BofA Market Share}_{T-1} = \frac{\text{Turnover}_{\text{MLB}, T-1}}{\text{Turnover}_{\text{Market}, T-1}}$$
  18. `feat_bofa_prev_day_turnover_rank`: BofA's liquidity ranking (1 = most active broker).
  19. `feat_institutional_hegemony_share`: Combined market turnover share of BofA + Top 5 domestic institutions.
      $$\text{Hegemony Share}_{T-1} = \frac{\text{Turnover}_{\text{MLB}, T-1} + \text{Turnover}_{\text{Top5}, T-1}}{\text{Turnover}_{\text{Market}, T-1}}$$
  20. `feat_avg_cr5_concentration`: Average 5-broker volume concentration ratio across all liquid equities.

### Cluster 6: Sector Cross-Sectional Stress & Volatility Breadth
* **Rationale**: Market-wide dispersion and volatility dictate whether algorithms operate in risk-on expansion or risk-off deleveraging modes.
* **Key Features**:
  21. `feat_market_avg_return_pct`: Average daily return across all tracked liquid equities on $T-1$.
  22. `feat_market_avg_range_pct`: Average daily high-low price range across tracked stocks (volatility proxy).
      $$\overline{\text{Range}}_{T-1} = \frac{1}{N}\sum_{s=1}^{N} \frac{P_{s, \text{High}, T-1} - P_{s, \text{Low}, T-1}}{P_{s, \text{Low}, T-1}}$$

### Cluster 7: Calendar & Temporal Seasonality
* **Rationale**: Institutional mandates follow recurring calendar patterns (Monday morning new portfolio allocations vs Friday closing hedges).
* **Key Features**:
  23. `day_of_week`: Integer representing day of week ($1 = \text{Monday}, \dots, 5 = \text{Friday}$).
  24. `is_monday`: Boolean flag for Monday morning sessions (high re-allocation volume).
  25. `is_friday`: Boolean flag for Friday sessions (weekend delta hedging).

---

## 4. Candidate Model Suite & Probabilistic Architecture

We benchmark 5 candidate paradigms to find the optimal trade-off between predictive accuracy and uncertainty quantification:

```
                  ┌──────────────────────────────────────────────┐
                  │          DayStartModelArena Tournament       │
                  └──────────────────────┬───────────────────────┘
                                         │
     ┌───────────────────┬───────────────┴───────────────┬───────────────────┐
     ▼                   ▼                               ▼                   ▼
┌──────────────┐  ┌──────────────┐               ┌──────────────┐     ┌──────────────┐
│  Baseline 0  │  │  Baseline 1  │               │   LightGBM   │     │   Bayesian   │
│   Naive W4   │  │ 5-Day Roll.  │               │  Non-Linear  │     │ Ridge / PyMC │
│  Persistence │  │     Mean     │               │   Ensemble   │     │ GLM Posterior│
└──────────────┘  └──────────────┘               └──────────────┘     └──────────────┘
```

1. **`DayStartNaivePersistenceModel` (Baseline 0)**: Carries yesterday's closing Window 4 flow forward: $\hat{y}_t = y_{t-1, \text{W4}}$.
2. **`DayStartRollingMeanModel` (Baseline 1)**: Historical 5-day moving average: $\hat{y}_t = \frac{1}{5}\sum_{k=1}^5 y_{t-k}$.
3. **`DayStartLightGBMModel`**: Gradient boosting regressor capturing non-linear interactions between competitor deltas and cost basis spreads with L2 shrinkage.
4. **`DayStartBayesianModel`**: Bayesian Ridge Regression with analytical conjugate priors, producing full posterior Gaussian distributions:
   $$\hat{y}_t \sim \mathcal{N}(\mu_{\text{post}}, \sigma_{\text{post}}^2) \implies 90\% \text{ CI} = [\mu - 1.645\sigma, \mu + 1.645\sigma]$$
5. **`DayStartPyMCModel`**: Full Bayesian GLM with informative regularizing priors $\beta \sim \mathcal{N}(0, 0.5)$ and $\sigma \sim \text{HalfNormal}(1.0)$ via MAP estimation and NUTS MCMC sampling.

---

## 5. Actionable Decision Items for Individual Traders

Continuous predictions are translated into discrete, tradeable decision playbooks:

### A. Directional Conviction Levels
* **`STRONG_ACCUMULATE`**: Predicted net flow $> +50\text{M TL}$ with conviction $\ge 70\%$.
* **`ACCUMULATE`**: Predicted net flow $> +10\text{M TL}$.
* **`NEUTRAL`**: Predicted net flow between $-10\text{M}$ and $+10\text{M TL}$.
* **`DISTRIBUTE`**: Predicted net flow $< -10\text{M TL}$.
* **`STRONG_DISTRIBUTE`**: Predicted net flow $< -50\text{M TL}$ with conviction $\ge 70\%$.

### B. Institutional Execution Playbooks
* **`SQUEEZE_LONG`**: High positive flow expectation with massive competitor flow delta ($> +30\text{M TL}$) — trade long momentum.
* **`MOMENTUM_EXPANSION`**: Large opening accumulation ($> +40\text{M TL}$) — follow early breakout.
* **`DEFENSE_SUPPORT`**: Underwater inventory ($< -4\%$ cost basis spread) with positive flow — institutional defense buying.
* **`LIQUIDITY_FADE`**: High negative flow expectation with BofA holding $> +5\%$ unrealized gains — expect profit-taking / fade intraday dips.
* **`SECTOR_ROTATION`**: Monday morning capital shifts between Banking and Transportation.
* **`NEUTRAL_WAIT`**: Ambiguous flow — wait for Window 2 intraday confirmation.
"""),
    nbf.v4.new_code_cell("""import duckdb
import polars as pl
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import ipywidgets as widgets
from IPython.display import display, HTML

from mdk_trading_oracle.core.db import DuckDBManager
from mdk_trading_oracle.core.config import get_settings
from mdk_trading_oracle.models.day_start import (
    DayStartFeatureExtractor,
    DayStartForecaster,
    DayStartModelArena,
    DayStartNaivePersistenceModel,
    DayStartRollingMeanModel,
    DayStartBayesianModel,
    DayStartPyMCModel,
    DayStartLightGBMModel,
)

settings = get_settings()
print(f"[OK] DuckDB Database: {settings.duckdb_path}")
print(f"[OK] Data Directory: {settings.data_dir}")
"""),
    nbf.v4.new_markdown_cell("""## 2. Feature Extraction: Assembling the 7 Feature Clusters

We extract features computed strictly at $T-1$ Close from our 6 Silver fact tables with **zero data leakage**.
"""),
    nbf.v4.new_code_cell("""db = DuckDBManager(read_only=True)
extractor = DayStartFeatureExtractor(db, target_broker_id="MLB")
df_pl = extractor.extract_features()
df = df_pl.to_pandas()

print(f"[OK] Extracted {len(df)} historical trading sessions with {len(df.columns)} features.")
display(df.head(5)[["trade_date", "day_of_week", "is_monday", "feat_bofa_w4_net_flow_tl", 
                   "feat_bofa_vs_top5_w4_flow_delta_tl", "feat_bofa_cost_basis_spread_20d_pct", 
                   "target_open_net_flow_tl", "target_open_direction"]])
"""),
    nbf.v4.new_markdown_cell("""## 3. Feature Importance & Correlation Analysis

How do yesterday's closing signals, competitor imbalances, and cost basis spreads correlate with today's opening net flow?
"""),
    nbf.v4.new_code_cell("""# Calculate correlations with the target opening net flow
feat_cols = [c for c in df.columns if c.startswith("feat_") or c in ["is_monday", "is_friday"]]
corrs = df[feat_cols + ["target_open_net_flow_tl"]].corr()["target_open_net_flow_tl"].drop("target_open_net_flow_tl").sort_values()

fig_corr = px.bar(
    x=corrs.values,
    y=corrs.index,
    orientation="h",
    title="Feature Correlations with Day-Start Opening Net Flow (Window 1)",
    labels={"x": "Pearson Correlation", "y": "Feature Name"},
    color=corrs.values,
    color_continuous_scale="RdBu_r",
    height=600
)
fig_corr.update_layout(template="plotly_dark", showlegend=False)
fig_corr.show()
"""),
    nbf.v4.new_markdown_cell(r"""## 4. Multi-Model Arena & Auto-Champion Tournament (Walk-Forward Validation)

We run an expanding-window **Walk-Forward Validation Tournament** across all 5 candidate models.
Models are trained strictly on past trading sessions ($1 \dots t-1$) to forecast session $t$, guaranteeing **zero lookahead bias**.
"""),
    nbf.v4.new_code_cell("""X = df.drop(columns=["target_open_net_flow_tl", "target_open_direction"], errors="ignore")
y = df["target_open_net_flow_tl"]

# Run Automated Walk-Forward Tournament across all 5 candidates
arena = DayStartModelArena()
scoreboard_df, champion_model = arena.run_tournament(X, y, min_train_samples=5, eval_window_days=20)

champion_name = scoreboard_df.iloc[0]["Model"]
champ_hit_rate = scoreboard_df.iloc[0]["hit_rate_pct"]
champ_picp = scoreboard_df.iloc[0]["picp_90_pct"]
champ_rmse = scoreboard_df.iloc[0]["rmse_million_tl"]

display(HTML(f\"\"\"
<div style="background: linear-gradient(135deg, #1b4332 0%, #081c15 100%); padding: 18px 24px; border-radius: 12px; border-left: 6px solid #52b788; margin-bottom: 20px; color: #fff; box-shadow: 0 4px 15px rgba(0,0,0,0.3);">
    <h3 style="margin: 0; color: #52b788;">Champion Crowned by Auto-Arena: {champion_name}</h3>
    <p style="margin: 6px 0 0 0; font-size: 14px; opacity: 0.95;">
        <b>Out-of-Sample Hit Rate:</b> <span style="color: #74c69d; font-weight: bold;">{champ_hit_rate:.1f}%</span> &nbsp;|&nbsp; 
        <b>90% Credible Interval Coverage (PICP):</b> <span style="color: #74c69d; font-weight: bold;">{champ_picp:.1f}%</span> &nbsp;|&nbsp; 
        <b>RMSE:</b> {champ_rmse:.2f}M TL
    </p>
</div>
\"\"\"))

display(HTML("<h3>Out-of-Sample Walk-Forward Scoreboard</h3>"))
display(scoreboard_df.style.highlight_max(subset=["hit_rate_pct", "picp_90_pct"], color="#1b4332")
                           .highlight_min(subset=["mae_million_tl", "rmse_million_tl"], color="#1b4332"))
"""),
    nbf.v4.new_markdown_cell("""## 5. Live Next-Day Forecast (Actionable Trading Signal for Tomorrow)

Using the dynamically crowned champion model from the Tournament, we extract features strictly from the latest market close and generate the **live forecast for the upcoming morning opening auction**.
"""),
    nbf.v4.new_code_cell("""# Initialize Forecaster with the dynamically crowned champion
forecaster = DayStartForecaster(db, model_type=champion_model.model_name)
live_forecast = forecaster.forecast_next_day()

# Display Live Signal Card for Traders
next_date = live_forecast.forecast_date
pred_flow = live_forecast.predicted_net_flow_tl / 1e6
lower_90 = live_forecast.predicted_flow_lower_90 / 1e6
upper_90 = live_forecast.predicted_flow_upper_90 / 1e6
direction = live_forecast.predicted_direction
confidence = live_forecast.direction_confidence * 100
playbook = live_forecast.predicted_playbook
buy_sec = live_forecast.top_predicted_buy_sector
sell_sec = live_forecast.top_predicted_sell_sector

dir_color = "#06d6a0" if "ACCUMULATE" in direction else ("#ef476f" if "DISTRIBUTE" in direction else "#ffd166")

display(HTML(f\"\"\"
<div style="background: linear-gradient(135deg, #0d1b2a 0%, #1b263b 100%); padding: 24px; border-radius: 14px; border: 2px solid {dir_color}; box-shadow: 0 8px 25px rgba(0,0,0,0.4); margin-bottom: 25px; color: #fff;">
    <div style="display: flex; justify-content: space-between; align-items: center; border-bottom: 1px solid rgba(255,255,255,0.15); padding-bottom: 12px; margin-bottom: 16px;">
        <h2 style="margin: 0; color: #e0e1dd; font-size: 20px;">Live Forecast for Upcoming Session: <span style="color: #00b4d8;">{next_date}</span></h2>
        <span style="background: {dir_color}; color: #000; font-weight: bold; padding: 6px 14px; border-radius: 20px; font-size: 14px;">{direction} ({confidence:.1f}% Conviction)</span>
    </div>
    <div style="display: grid; grid-template-columns: repeat(3, 1fr); gap: 16px; font-size: 15px;">
        <div style="background: rgba(255,255,255,0.05); padding: 14px; border-radius: 8px;">
            <div style="color: #778da9; font-size: 13px; text-transform: uppercase;">Expected Opening Net Flow</div>
            <div style="font-size: 22px; font-weight: bold; color: {dir_color}; margin-top: 4px;">{pred_flow:+,.2f} M TL</div>
            <div style="color: #778da9; font-size: 12px; margin-top: 2px;">90% CI: [{lower_90:+,.1f}M, {upper_90:+,.1f}M]</div>
        </div>
        <div style="background: rgba(255,255,255,0.05); padding: 14px; border-radius: 8px;">
            <div style="color: #778da9; font-size: 13px; text-transform: uppercase;">Institutional Playbook</div>
            <div style="font-size: 20px; font-weight: bold; color: #00b4d8; margin-top: 4px;">{playbook}</div>
            <div style="color: #778da9; font-size: 12px; margin-top: 2px;">Champion Model: {champion_name}</div>
        </div>
        <div style="background: rgba(255,255,255,0.05); padding: 14px; border-radius: 8px;">
            <div style="color: #778da9; font-size: 13px; text-transform: uppercase;">Top Sector Rotation Focus</div>
            <div style="font-size: 16px; margin-top: 4px;"><b>Buy:</b> <span style="color: #52b788;">{buy_sec}</span></div>
            <div style="font-size: 16px; margin-top: 2px;"><b>Sell:</b> <span style="color: #e63946;">{sell_sec}</span></div>
        </div>
    </div>
</div>
\"\"\"))
"""),
    nbf.v4.new_markdown_cell("""## 6. Historical Backtest Track Record: Predicted vs Actual Opening Net Flow & 90% Confidence Ribbon

Visualizing historical out-of-sample and backtested performance of the crowned Champion model against actual opening net flows.
"""),
    nbf.v4.new_code_cell("""# Generate historical backtest track record using crowned champion
backtest_forecasts = forecaster.backtest_all_history()

predictions = []
lowers = []
uppers = []
playbooks = []
directions = []
confidences = []

for res in backtest_forecasts:
    predictions.append(res.predicted_net_flow_tl / 1e6)
    lowers.append(res.predicted_flow_lower_90 / 1e6)
    uppers.append(res.predicted_flow_upper_90 / 1e6)
    playbooks.append(res.predicted_playbook)
    directions.append(res.predicted_direction)
    confidences.append(res.direction_confidence)

chart_df = df.copy()
chart_df["trade_date"] = chart_df["trade_date"].astype(str).str.slice(0, 10)
chart_df["pred_flow_m"] = predictions
chart_df["lower_90_m"] = lowers
chart_df["upper_90_m"] = uppers
chart_df["actual_flow_m"] = df["target_open_net_flow_tl"] / 1e6
chart_df["playbook"] = playbooks
chart_df["pred_direction"] = directions
chart_df["confidence"] = confidences

fig = go.Figure()

# 90% Confidence Interval Shaded Band
fig.add_trace(go.Scatter(
    x=chart_df["trade_date"].tolist() + chart_df["trade_date"].tolist()[::-1],
    y=chart_df["upper_90_m"].tolist() + chart_df["lower_90_m"].tolist()[::-1],
    fill="toself",
    fillcolor="rgba(0, 180, 216, 0.15)",
    line=dict(color="rgba(255,255,255,0)"),
    name="90% Credible Interval",
    hoverinfo="skip"
))

# Predicted Flow
fig.add_trace(go.Scatter(
    x=chart_df["trade_date"],
    y=chart_df["pred_flow_m"],
    mode="lines+markers",
    name="Predicted Opening Net Flow (TL M)",
    line=dict(color="#00b4d8", width=3),
    marker=dict(size=8, symbol="diamond")
))

# Actual Flow
fig.add_trace(go.Scatter(
    x=chart_df["trade_date"],
    y=chart_df["actual_flow_m"],
    mode="lines+markers",
    name="Actual Window 1 Net Flow (TL M)",
    line=dict(color="#ffb703", width=2, dash="dash"),
    marker=dict(size=7, symbol="circle")
))

fig.update_layout(
    title="Bank of America Day-Start Backtest: Predicted vs Actual Opening Net Flow (Million TL)",
    xaxis_title="Trading Date",
    yaxis_title="Net Flow (Million TL)",
    template="plotly_dark",
    hovermode="x unified",
    height=550
)
fig.show()
"""),
    nbf.v4.new_markdown_cell("""## 7. Interactive Historical Session Inspector & Playbook Breakdown

Select any historical date to inspect the model's opening conviction, competitor closing posture, and sector allocation forecast.
"""),
    nbf.v4.new_code_cell("""date_options = chart_df["trade_date"].tolist()

date_dropdown = widgets.Dropdown(
    options=date_options,
    value=date_options[-1] if date_options else None,
    description="Date:",
    style={"description_width": "initial"}
)

output = widgets.Output()

def update_session(change):
    with output:
        output.clear_output()
        sel_date = change["new"]
        if not sel_date:
            return
        matches = chart_df[chart_df["trade_date"] == str(sel_date)]
        if len(matches) == 0:
            display(HTML(f"<p style='color: orange;'>No data for {sel_date}</p>"))
            return
        row = matches.iloc[0]
        
        dir_color = "#06d6a0" if "ACCUMULATE" in row["pred_direction"] else ("#ef476f" if "DISTRIBUTE" in row["pred_direction"] else "#ffd166")
        
        html_card = f\"\"\"
        <div style="background: #1e1e2e; padding: 20px; border-radius: 12px; border-left: 6px solid {dir_color}; margin-bottom: 15px; color: #fff;">
            <h3 style="margin-top: 0;">Trading Session: {sel_date} ({'Monday (Rebalancing)' if row['is_monday'] else 'Regular Session'})</h3>
            <table style="width: 100%; border-collapse: collapse; font-size: 15px;">
                <tr>
                    <td><b>Predicted Direction:</b> <span style="color: {dir_color}; font-weight: bold;">{row['pred_direction']}</span></td>
                    <td><b>Conviction / Confidence:</b> {row['confidence'] * 100:.1f}%</td>
                </tr>
                <tr>
                    <td><b>Forecasted Net Flow:</b> {row['pred_flow_m']:+,.2f} M TL</td>
                    <td><b>90% Credible Range:</b> [{row['lower_90_m']:+,.2f} M, {row['upper_90_m']:+,.2f} M]</td>
                </tr>
                <tr>
                    <td><b>Institutional Playbook:</b> <span style="color: #00b4d8; font-weight: bold;">{row['playbook']}</span></td>
                    <td><b>Actual Window 1 Flow:</b> {row['actual_flow_m']:+,.2f} M TL</td>
                </tr>
                <tr>
                    <td><b>Yesterday W4 Net Flow:</b> {row['feat_bofa_w4_net_flow_tl'] / 1e6:+,.2f} M TL</td>
                    <td><b>Top-5 Competitor Closing Delta:</b> {row['feat_bofa_vs_top5_w4_flow_delta_tl'] / 1e6:+,.2f} M TL</td>
                </tr>
            </table>
        </div>
        \"\"\"
        display(HTML(html_card))

date_dropdown.observe(update_session, names="value")
display(date_dropdown, output)
if date_dropdown.value:
    update_session({"new": date_dropdown.value})
"""),
    nbf.v4.new_markdown_cell("""## 8. Gold Tables & Production Performance Ledgers in DuckDB

Verifying the persisted production tables in DuckDB:
1. `gold_bofa_day_start_forecasts`: Pure upcoming live forecasts ($T+1$) strictly for tomorrow's market open.
2. `gold_bofa_day_start_performance`: Permanent historical performance tracking ledger recording prior forecasts matched against realized actual Window 1 market data.
3. `gold_bofa_day_start_backtests`: Dedicated historical walk-forward backtest simulation ledger.
"""),
    nbf.v4.new_code_cell("""conn = db.get_connection()

print("1. Live Active Upcoming Forecast (gold_bofa_day_start_forecasts) - Strictly T+1:")
gold_forecasts_df = conn.execute(\"\"\"
    SELECT 
        forecast_date,
        day_of_week,
        is_monday,
        predicted_open_net_flow_tl / 1e6 AS pred_net_flow_m_tl,
        predicted_direction,
        direction_confidence,
        predicted_playbook,
        top_predicted_buy_sector,
        top_predicted_sell_sector,
        model_name
    FROM gold_bofa_day_start_forecasts
    ORDER BY forecast_date DESC;
\"\"\").df()
display(gold_forecasts_df)

print("2. Historical Performance Tracking Ledger (gold_bofa_day_start_performance) - Latest 5 Sessions:")
gold_perf_df = conn.execute(\"\"\"
    SELECT 
        trade_date,
        predicted_open_net_flow_tl / 1e6 AS pred_m_tl,
        actual_open_net_flow_tl / 1e6 AS actual_m_tl,
        error_open_net_flow_tl / 1e6 AS error_m_tl,
        absolute_error_tl / 1e6 AS abs_error_m_tl,
        predicted_direction,
        actual_direction,
        is_direction_hit,
        is_inside_90_ci,
        predicted_playbook
    FROM gold_bofa_day_start_performance
    ORDER BY trade_date DESC
    LIMIT 5;
\"\"\").df()
display(gold_perf_df)
"""),
    nbf.v4.new_markdown_cell("""### Historical Backtest Performance Dashboard (`gold_bofa_day_start_backtests`)

Calculating executive summary KPIs directly from the DuckDB backtest ledger:
"""),
    nbf.v4.new_code_cell("""# Summary KPIs from DuckDB
backtest_kpis = conn.execute(\"\"\"
    SELECT 
        COUNT(*) AS total_sessions,
        ROUND(AVG(CASE WHEN is_direction_hit THEN 1.0 ELSE 0.0 END) * 100, 1) AS hit_rate_pct,
        ROUND(AVG(CASE WHEN is_inside_90_ci THEN 1.0 ELSE 0.0 END) * 100, 1) AS picp_90_pct,
        ROUND(AVG(ABS(error_open_net_flow_tl)) / 1e6, 2) AS mae_m_tl,
        ROUND(SQRT(AVG(POWER(error_open_net_flow_tl, 2))) / 1e6, 2) AS rmse_m_tl
    FROM gold_bofa_day_start_backtests;
\"\"\").df().iloc[0]


kpi_html = f\"\"\"
<div style="display: flex; gap: 15px; margin-bottom: 20px;">
    <div style="flex: 1; background: #1e1e2e; padding: 15px; border-radius: 10px; border-left: 5px solid #00b4d8; color: #fff;">
        <div style="font-size: 12px; color: #888; text-transform: uppercase;">Total Evaluated Sessions</div>
        <div style="font-size: 24px; font-weight: bold; margin-top: 5px;">{int(backtest_kpis['total_sessions'])}</div>
    </div>
    <div style="flex: 1; background: #1e1e2e; padding: 15px; border-radius: 10px; border-left: 5px solid #06d6a0; color: #fff;">
        <div style="font-size: 12px; color: #888; text-transform: uppercase;">Out-of-Sample Hit Rate</div>
        <div style="font-size: 24px; font-weight: bold; margin-top: 5px; color: #06d6a0;">{backtest_kpis['hit_rate_pct']:.1f}%</div>
    </div>
    <div style="flex: 1; background: #1e1e2e; padding: 15px; border-radius: 10px; border-left: 5px solid #ffd166; color: #fff;">
        <div style="font-size: 12px; color: #888; text-transform: uppercase;">90% Credible Coverage (PICP)</div>
        <div style="font-size: 24px; font-weight: bold; margin-top: 5px; color: #ffd166;">{backtest_kpis['picp_90_pct']:.1f}%</div>
    </div>
    <div style="flex: 1; background: #1e1e2e; padding: 15px; border-radius: 10px; border-left: 5px solid #ef476f; color: #fff;">
        <div style="font-size: 12px; color: #888; text-transform: uppercase;">Mean Absolute Error (MAE)</div>
        <div style="font-size: 24px; font-weight: bold; margin-top: 5px;">{backtest_kpis['mae_m_tl']:,.2f} M TL</div>
    </div>
</div>
\"\"\"
display(HTML(kpi_html))
"""),
    nbf.v4.new_code_cell("""print("2. Full Day-Start Backtest Ledger (gold_bofa_day_start_backtests):")
gold_backtests_full_df = conn.execute(\"\"\"
    SELECT 
        trade_date,
        day_of_week,
        is_monday,
        predicted_open_net_flow_tl / 1e6 AS pred_net_flow_m,
        actual_open_net_flow_tl / 1e6 AS act_net_flow_m,
        error_open_net_flow_tl / 1e6 AS error_m,
        predicted_direction,
        actual_direction,
        is_direction_hit,
        is_inside_90_ci,
        direction_confidence,
        predicted_playbook,
        model_name
    FROM gold_bofa_day_start_backtests
    ORDER BY trade_date DESC;
\"\"\").df()

# Display formatted full backtest table
display(gold_backtests_full_df)
"""),
    nbf.v4.new_code_cell("""print("3. Backtest Performance by Session Type (Day of Week):")
dow_perf_df = conn.execute(\"\"\"
    SELECT 
        day_of_week,
        COUNT(*) AS sessions,
        ROUND(AVG(CASE WHEN is_direction_hit THEN 1.0 ELSE 0.0 END) * 100, 1) AS hit_rate_pct,
        ROUND(AVG(CASE WHEN is_inside_90_ci THEN 1.0 ELSE 0.0 END) * 100, 1) AS picp_90_pct,
        ROUND(AVG(ABS(error_open_net_flow_tl)) / 1e6, 2) AS mae_m_tl,
        ROUND(AVG(actual_open_net_flow_tl) / 1e6, 2) AS avg_actual_flow_m
    FROM gold_bofa_day_start_backtests
    GROUP BY day_of_week
    ORDER BY sessions DESC;
\"\"\").df()
display(dow_perf_df)
"""),
]

nb.cells.extend(cells)

with open("notebooks/03_bofa_day_start_modeling.ipynb", "w") as f:
    nbf.write(nb, f)

print("[OK] Successfully generated notebooks/03_bofa_day_start_modeling.ipynb!")
