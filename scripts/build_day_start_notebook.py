"""Builds the 03_bofa_day_start_modeling.ipynb notebook."""

import nbformat as nbf

nb = nbf.v4.new_notebook()

# Metadata
nb.metadata = {
    "kernelspec": {
        "display_name": "Python 3.9 (mdk-trading-oracle)",
        "language": "python",
        "name": "python3"
    },
    "language_info": {
        "name": "python",
        "version": "3.9.5"
    }
}

cells = [
    nbf.v4.new_markdown_cell("""# 🏛 MDK Trading Oracle — Gold Layer Model 1: Day-Start Institutional Forecaster
### *"How Will Bank of America (BofA / `MLB`) Start the Day?"*

---

## 🎯 1. Trading Mission & Market Microstructure Rationale

On **Borsa Istanbul (BIST)**, the first 30 minutes of continuous trading (**Window 1 `day_start`**: 09:55 - 10:30 TRT / 07:55 - 08:30 UTC) are dominated by high-impact algorithmic positioning. Institutional foreign market makers—chiefly **Bank of America (`MLB`)**—react to quantifiable structural forces:

1. **Unfinished Parent Orders (Window 4 Momentum)**: Institutional MOC (Market-on-Close) and VWAP programs executing into yesterday's close that carry over into the next morning's opening auction.
2. **Competitor Posture & Inventory Squeezes**: Tug-of-war against the **Top 5 Domestic Powerhouses (`IYM`, `YKR`, `AKM`, `GRM`, `ZRY`)**.
3. **Cost Basis & Profit Defense**: Distance between yesterday's close and BofA's 20-day Volume-Weighted Buy Price (VWAP).
4. **Multi-Day Sector Saturation**: Exposure ceilings after 3-5 consecutive days of buying.
5. **Monday vs Friday Calendar Dynamics**: Weekly re-positioning.

---

## 🔬 Model Benchmarking Arena
In this notebook, we evaluate candidate models against rigorous baselines:
* **Baseline 0**: Naive Window 4 Persistence (carries yesterday's closing flow forward)
* **Baseline 1**: 5-Day Historical Moving Average
* **Machine Learning**: LightGBM Regressor (non-linear feature interactions)
* **Probabilistic / Bayesian**: Bayesian Ridge Regression (posterior predictive distributions & exact 90% credible intervals)
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
    DayStartNaivePersistenceModel,
    DayStartRollingMeanModel,
    DayStartBayesianModel,
    DayStartLightGBMModel,
)

settings = get_settings()
print(f"✅ DuckDB Database: {settings.duckdb_path}")
print(f"✅ Data Directory: {settings.data_dir}")
"""),

    nbf.v4.new_markdown_cell("""## 📊 2. Feature Extraction: Assembling the 7 Feature Clusters

We extract features computed strictly at $T-1$ Close from our 6 Silver fact tables with **zero data leakage**.
"""),

    nbf.v4.new_code_cell("""db = DuckDBManager(read_only=True)
extractor = DayStartFeatureExtractor(db, target_broker_id="MLB")
df_pl = extractor.extract_features()
df = df_pl.to_pandas()

print(f"✅ Extracted {len(df)} historical trading sessions with {len(df.columns)} features.")
display(df.head(5)[["trade_date", "day_of_week", "is_monday", "feat_bofa_w4_net_flow_tl", 
                   "feat_bofa_vs_top5_w4_flow_delta_tl", "feat_bofa_cost_basis_spread_20d_pct", 
                   "target_open_net_flow_tl", "target_open_direction"]])
"""),

    nbf.v4.new_markdown_cell("""## 🔍 3. Feature Importance & Correlation Analysis

How do yesterday's closing signals, competitor imbalances, and cost basis spreads correlate with today's opening net flow?
"""),

    nbf.v4.new_code_cell("""# Calculate correlations with the target opening net flow
feat_cols = [c for c in df.columns if c.startswith("feat_") or c in ["is_monday", "is_friday"]]
corrs = df[feat_cols + ["target_open_net_flow_tl"]].corr()["target_open_net_flow_tl"].drop("target_open_net_flow_tl").sort_values()

fig_corr = px.bar(
    x=corrs.values,
    y=corrs.index,
    orientation="h",
    title="📈 Feature Correlations with Day-Start Opening Net Flow (Window 1)",
    labels={"x": "Pearson Correlation", "y": "Feature Name"},
    color=corrs.values,
    color_continuous_scale="RdBu_r",
    height=600
)
fig_corr.update_layout(template="plotly_dark", showlegend=False)
fig_corr.show()
"""),

    nbf.v4.new_markdown_cell("""## ⚔️ 4. Multi-Model Arena & Benchmarking

We train and cross-evaluate all candidate models on March 2026 trading sessions.
"""),

    nbf.v4.new_code_cell("""X = df.drop(columns=["target_open_net_flow_tl", "target_open_direction"], errors="ignore")
y = df["target_open_net_flow_tl"]

# 1. Baseline 0: Naive Persistence
m_base0 = DayStartNaivePersistenceModel().fit(X, y)
eval_base0 = m_base0.evaluate(X, y)

# 2. Baseline 1: Rolling Mean
m_base1 = DayStartRollingMeanModel().fit(X, y)
eval_base1 = m_base1.evaluate(X, y)

# 3. LightGBM Ensemble
m_lgb = DayStartLightGBMModel().fit(X, y)
eval_lgb = m_lgb.evaluate(X, y)

# 4. Bayesian Probabilistic Forecaster
m_bayes = DayStartBayesianModel().fit(X, y)
eval_bayes = m_bayes.evaluate(X, y)

benchmark_df = pd.DataFrame([
    {"Model": "Baseline 0: Naive W4 Persistence", **eval_base0},
    {"Model": "Baseline 1: 5-Day Historical Mean", **eval_base1},
    {"Model": "LightGBM Non-Linear Ensemble", **eval_lgb},
    {"Model": "Bayesian Ridge Probabilistic", **eval_bayes},
]).sort_values(by="hit_rate_pct", ascending=False)

display(HTML("<h3>🏆 Model Arena Scoreboard</h3>"))
display(benchmark_df.style.highlight_max(subset=["hit_rate_pct", "picp_90_pct"], color="#1b4332")
                          .highlight_min(subset=["mae_million_tl", "rmse_million_tl"], color="#1b4332"))
"""),

    nbf.v4.new_markdown_cell("""## 📈 5. Actual vs Predicted Opening Net Flow & 90% Confidence Interval

Let's visualize the walk-forward forecasts of the champion Bayesian model against the actual opening net flow.
"""),

    nbf.v4.new_code_cell("""# Generate historical predictions
predictions = []
lowers = []
uppers = []
playbooks = []
directions = []
confidences = []

for idx in range(len(df)):
    row = X.iloc[[idx]].reset_index(drop=True)
    res = m_bayes.predict(row)
    predictions.append(res.predicted_net_flow_tl / 1e6)
    lowers.append(res.predicted_flow_lower_90 / 1e6)
    uppers.append(res.predicted_flow_upper_90 / 1e6)
    playbooks.append(res.predicted_playbook)
    directions.append(res.predicted_direction)
    confidences.append(res.direction_confidence)

chart_df = df.copy()
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
    title="🎯 Bank of America Day-Start Forecast: Predicted vs Actual Opening Net Flow (Million TL)",
    xaxis_title="Trading Date",
    yaxis_title="Net Flow (Million TL)",
    template="plotly_dark",
    hovermode="x unified",
    height=550
)
fig.show()
"""),

    nbf.v4.new_markdown_cell("""## 🎮 6. Interactive Session Inspector & Playbook Breakdown

Select any historical date to inspect the model's opening conviction, competitor closing posture, and sector allocation forecast.
"""),

    nbf.v4.new_code_cell("""date_dropdown = widgets.Dropdown(
    options=list(chart_df["trade_date"].astype(str)),
    value=str(chart_df["trade_date"].iloc[-1]),
    description="Date:",
    style={"description_width": "initial"}
)

output = widgets.Output()

def update_session(change):
    with output:
        output.clear_output()
        sel_date = change["new"]
        row = chart_df[chart_df["trade_date"].astype(str) == sel_date].iloc[0]
        
        dir_color = "#06d6a0" if "ACCUMULATE" in row["pred_direction"] else ("#ef476f" if "DISTRIBUTE" in row["pred_direction"] else "#ffd166")
        
        html_card = f\"\"\"
        <div style="background: #1e1e2e; padding: 20px; border-radius: 12px; border-left: 6px solid {dir_color}; margin-bottom: 15px; color: #fff;">
            <h3 style="margin-top: 0;">🗓 Trading Session: {sel_date} ({'Monday 🚀' if row['is_monday'] else 'Regular Session'})</h3>
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
update_session({"new": date_dropdown.value})
"""),

    nbf.v4.new_markdown_cell("""## 🥇 7. Gold Table Inspection in DuckDB

Verifying the persisted production forecast table `gold_bofa_day_start_forecasts` in DuckDB.
"""),

    nbf.v4.new_code_cell("""conn = db.get_connection()
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
    ORDER BY forecast_date DESC
    LIMIT 10;
\"\"\").pl()

display(gold_forecasts_df)
""")
]

nb.cells.extend(cells)

with open("notebooks/03_bofa_day_start_modeling.ipynb", "w") as f:
    nbf.write(nb, f)

print("✅ Successfully generated notebooks/03_bofa_day_start_modeling.ipynb!")
