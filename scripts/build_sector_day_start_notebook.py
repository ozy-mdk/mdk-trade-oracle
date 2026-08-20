"""Builds the 04_bofa_sector_day_start_modeling.ipynb notebook."""

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
    nbf.v4.new_markdown_cell(r"""# 🏛 MDK Trading Oracle — Gold Layer Model 2: Sector Day-Start Allocation Forecaster
### *"Where Will Bank of America (BofA / `MLB`) Allocate Capital Across Sectors at the Open?"*

---

## 🎯 1. Model Objective & Microstructure Rationale

While **Model 1** forecasts BofA's total macro opening flow across the exchange, **Model 2** resolves the **cross-sectional sector allocation and rotation problem**:

* **Why Sector-Level Modeling?**: Institutional algorithms rarely buy or sell the entire market uniformly. On any given morning, BofA may aggressively **accumulate Banking (`AKBNK`, `GARAN`, `ISCTR`)** while simultaneously **distributing Transportation (`THYAO`, `PGSUS`)** or defending **Holdings (`KCHOL`, `SAHOL`)**.
* **Zero Lookahead Leakage**: All predictive input features are computed **strictly from historical data up to $T-1$ Close (18:10 TRT)**.

---

## 🎯 2. Sector Target Variables

For each tracked sector $s \in \{\text{Banking}, \text{Transportation}, \text{Defense \& Tech}, \text{Energy \& Refining}, \text{Holding}, \dots\}$:

| Target Variable | Data Type | Mathematical Formulation | Role & Description |
| :--- | :--- | :--- | :--- |
| `target_sector_open_net_flow_tl` | Continuous (`float64`) | $$\text{Net Flow}_{s, T, \text{W1}} = \sum_{i \in \text{Trades}_{T, \text{W1}, \text{MLB}, \text{Sector } s}} (\text{Buy Value}_i - \text{Sell Value}_i)$$ | **Primary Sector Target ($y_s$)**: Net executed capital in TL by BofA in sector $s$ during Window 1. |
| `target_sector_open_direction` | Categorical (`str`) | $$\text{Direction}_{s, T} = \begin{cases} \text{BUY}, & \text{if } \text{Net Flow}_{s, T, \text{W1}} > 0 \\ \text{SELL}, & \text{if } \text{Net Flow}_{s, T, \text{W1}} \le 0 \end{cases}$$ | **Derived Direction**: Sector opening direction (`BUY` vs `SELL`). |

---

## 🧠 3. The 5 Sector Quantitative Feature Clusters

All features are constructed strictly from $T-1$ Close data:

1. **Sector Closing Momentum (W4)**: BofA's closing window net flow and turnover in sector $s$ (`feat_sector_bofa_w4_net_flow_tl`).
2. **Sector Competitor Imbalance**: Domestic Top 5 flow in sector $s$ and BofA vs Top-5 delta in that sector (`feat_sector_bofa_vs_top5_w4_delta_tl`).
3. **Sector Market Dominance & Wallet Share**: BofA's turnover share within sector $s$ (`feat_sector_bofa_market_share`) and share of BofA's total daily wallet (`feat_sector_bofa_share_of_wallet`).
4. **Sector Multi-Day Inventory Saturation**: 5-day cumulative sector flow and 20-day sector flow Z-score (`feat_sector_bofa_flow_zscore_20d`).
5. **Macro Context & Calendar Dynamics**: Total BofA macro net flow on $T-1$, and calendar flags (`is_monday`, `is_friday`).
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
from mdk_trading_oracle.models.sector_day_start import (
    SectorDayStartFeatureExtractor,
    SectorDayStartForecaster,
    SectorDayStartModelArena,
)

settings = get_settings()
print(f"✅ DuckDB Database: {settings.duckdb_path}")
print(f"✅ Data Directory: {settings.data_dir}")
"""),

    nbf.v4.new_markdown_cell("""## 📊 2. Sector Feature Extraction across BIST Industries

We extract sector-level feature matrices for all tracked liquid sectors at $T-1$ Close with **zero data leakage**.
"""),

    nbf.v4.new_code_cell("""db = DuckDBManager(read_only=True)
extractor = SectorDayStartFeatureExtractor(db, target_broker_id="MLB")
tracked_sectors = extractor.get_tracked_sectors(min_session_count=10)

print(f"✅ Tracked Liquid Sectors ({len(tracked_sectors)}): {', '.join(tracked_sectors[:8])}...")

df_pl = extractor.extract_features()
df = df_pl.to_pandas()

print(f"✅ Total Sector-Session Observations: {len(df)} across {df['sector'].nunique()} sectors.")
display(df.head(6)[["trade_date", "sector", "feat_sector_bofa_w4_net_flow_tl", 
                   "feat_sector_bofa_vs_top5_w4_delta_tl", "feat_sector_bofa_cum_net_flow_5d_tl", 
                   "target_sector_open_net_flow_tl", "target_sector_open_direction"]])
"""),

    nbf.v4.new_markdown_cell("""## ⚔️ 3. Sector Model Arena: Walk-Forward Tournament

Running expanding-window walk-forward validation across candidate models (Naive Persistence, Rolling Mean, LightGBM, Bayesian Ridge, PyMC GLM).
"""),

    nbf.v4.new_code_cell("""# Benchmark candidates across the banking sector as a representative high-liquidity benchmark
df_banking = df[df["sector"] == "Banking"].copy()
X_bank = df_banking.drop(columns=["target_sector_open_net_flow_tl", "target_sector_open_direction"], errors="ignore")
y_bank = df_banking["target_sector_open_net_flow_tl"]

arena = SectorDayStartModelArena()
scoreboard_df, champion_model = arena.run_tournament(X_bank, y_bank, min_train_samples=5)

champion_name = scoreboard_df.iloc[0]["Model"]
champ_hit_rate = scoreboard_df.iloc[0]["hit_rate_pct"]
champ_picp = scoreboard_df.iloc[0]["picp_90_pct"]
champ_rmse = scoreboard_df.iloc[0]["rmse_million_tl"]

display(HTML(f\"\"\"
<div style="background: linear-gradient(135deg, #1b4332 0%, #081c15 100%); padding: 18px 24px; border-radius: 12px; border-left: 6px solid #52b788; margin-bottom: 20px; color: #fff; box-shadow: 0 4px 15px rgba(0,0,0,0.3);">
    <h3 style="margin: 0; color: #52b788;">🏆 Sector Champion Crowned: {champion_name}</h3>
    <p style="margin: 6px 0 0 0; font-size: 14px; opacity: 0.95;">
        <b>Out-of-Sample Hit Rate:</b> <span style="color: #74c69d; font-weight: bold;">{champ_hit_rate:.1f}%</span> &nbsp;|&nbsp; 
        <b>90% Credible Interval Coverage (PICP):</b> <span style="color: #74c69d; font-weight: bold;">{champ_picp:.1f}%</span> &nbsp;|&nbsp; 
        <b>RMSE:</b> {champ_rmse:.2f}M TL
    </p>
</div>
\"\"\"))

display(scoreboard_df.style.highlight_max(subset=["hit_rate_pct", "picp_90_pct"], color="#1b4332")
                           .highlight_min(subset=["mae_million_tl", "rmse_million_tl"], color="#1b4332"))
"""),

    nbf.v4.new_markdown_cell("""## 🎮 4. Interactive Sector Forecast Explorer & Heat Matrix

Select any sector from the dropdown to inspect BofA's predicted vs actual opening net flow and 90% credible ranges.
"""),

    nbf.v4.new_code_cell("""# Run production Sector Forecaster across all sectors
forecaster = SectorDayStartForecaster(db, model_type="auto")
all_sector_forecasts = forecaster.train_and_forecast_all(sectors=tracked_sectors[:8])

records = []
for f in all_sector_forecasts:
    records.append({
        "trade_date": str(f.forecast_date)[:10],
        "sector": f.top_predicted_buy_sector,
        "pred_flow_m": f.predicted_net_flow_tl / 1e6,
        "lower_90_m": f.predicted_flow_lower_90 / 1e6,
        "upper_90_m": f.predicted_flow_upper_90 / 1e6,
        "pred_direction": f.predicted_direction,
        "confidence": f.direction_confidence,
        "playbook": f.predicted_playbook,
    })

chart_df = pd.DataFrame(records)
# Merge actuals
actuals_df = df[["trade_date", "sector", "target_sector_open_net_flow_tl"]].copy()
actuals_df["trade_date"] = actuals_df["trade_date"].astype(str).str.slice(0, 10)
actuals_df["actual_flow_m"] = actuals_df["target_sector_open_net_flow_tl"] / 1e6
chart_df = chart_df.merge(actuals_df, on=["trade_date", "sector"], how="left")

sector_dropdown = widgets.Dropdown(
    options=sorted(chart_df["sector"].unique().tolist()),
    value="Banking" if "Banking" in chart_df["sector"].unique() else chart_df["sector"].iloc[0],
    description="Sector:",
    style={"description_width": "initial"}
)

plot_output = widgets.Output()

def update_sector_plot(change):
    with plot_output:
        plot_output.clear_output()
        sel_sector = change["new"]
        sub = chart_df[chart_df["sector"] == sel_sector].sort_values(by="trade_date")
        if len(sub) == 0:
            return
            
        fig = go.Figure()
        
        # 90% Confidence Interval Band
        fig.add_trace(go.Scatter(
            x=sub["trade_date"].tolist() + sub["trade_date"].tolist()[::-1],
            y=sub["upper_90_m"].tolist() + sub["lower_90_m"].tolist()[::-1],
            fill="toself",
            fillcolor="rgba(0, 180, 216, 0.15)",
            line=dict(color="rgba(255,255,255,0)"),
            name="90% Credible Interval",
            hoverinfo="skip"
        ))
        
        # Predicted
        fig.add_trace(go.Scatter(
            x=sub["trade_date"],
            y=sub["pred_flow_m"],
            mode="lines+markers",
            name="Predicted Net Flow (TL M)",
            line=dict(color="#00b4d8", width=3),
            marker=dict(size=8, symbol="diamond")
        ))
        
        # Actual
        fig.add_trace(go.Scatter(
            x=sub["trade_date"],
            y=sub["actual_flow_m"],
            mode="lines+markers",
            name="Actual Window 1 Net Flow (TL M)",
            line=dict(color="#ffb703", width=2, dash="dash"),
            marker=dict(size=7, symbol="circle")
        ))
        
        fig.update_layout(
            title=f"🎯 BofA Opening Flow in Sector: {sel_sector} (Predicted vs Actual)",
            xaxis_title="Trade Date",
            yaxis_title="Net Flow (Million TL)",
            template="plotly_dark",
            hovermode="x unified",
            height=500
        )
        fig.show()

sector_dropdown.observe(update_sector_plot, names="value")
display(sector_dropdown, plot_output)
if sector_dropdown.value:
    update_sector_plot({"new": sector_dropdown.value})
"""),

    nbf.v4.new_markdown_cell("""## 🔄 5. Cross-Sectional Sector Rotation & Capital Allocation

Visualizing BofA's latest predicted capital allocation across all sectors simultaneously (detecting long/short sector pair trades).
"""),

    nbf.v4.new_code_cell("""latest_date = chart_df["trade_date"].max()
latest_df = chart_df[chart_df["trade_date"] == latest_date].sort_values(by="pred_flow_m", ascending=False)

fig_rot = px.bar(
    latest_df,
    x="pred_flow_m",
    y="sector",
    orientation="h",
    title=f"🔄 BofA Predicted Opening Sector Allocation for Session: {latest_date}",
    labels={"pred_flow_m": "Forecasted Opening Net Flow (TL M)", "sector": "Industry Sector"},
    color="pred_flow_m",
    color_continuous_scale="RdBu",
    height=450
)
fig_rot.update_layout(template="plotly_dark", showlegend=False)
fig_rot.show()
"""),

    nbf.v4.new_markdown_cell("""## 🥇 6. Gold Sector Table Inspection in DuckDB

Verifying the persisted production forecast table `gold_bofa_sector_day_start_forecasts` in DuckDB.
"""),

    nbf.v4.new_code_cell("""conn = db.get_connection()
gold_sector_df = conn.execute(\"\"\"
    SELECT 
        forecast_date,
        sector,
        day_of_week,
        predicted_open_net_flow_tl / 1e6 AS pred_flow_m_tl,
        predicted_direction,
        direction_confidence,
        predicted_playbook,
        model_name
    FROM gold_bofa_sector_day_start_forecasts
    ORDER BY forecast_date DESC, pred_flow_m_tl DESC
    LIMIT 12;
\"\"\").pl()

display(gold_sector_df)
""")
]

nb.cells.extend(cells)

with open("notebooks/04_bofa_sector_day_start_modeling.ipynb", "w") as f:
    nbf.write(nb, f)

print("✅ Successfully generated notebooks/04_bofa_sector_day_start_modeling.ipynb!")
