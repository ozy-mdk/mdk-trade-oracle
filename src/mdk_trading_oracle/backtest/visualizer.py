"""Interactive Plotly visualization suite for model backtest and performance diagnostics."""

from typing import List, Optional

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from mdk_trading_oracle.backtest.types import BacktestSummary, SliceMetrics, TargetUnit


class BacktestVisualizer:
    """Generates production-grade, dark-themed interactive visualizations for model performance."""

    THEME = {
        "paper_bgcolor": "#0f172a",
        "plot_bgcolor": "#1e293b",
        "font_family": "Inter, -apple-system, BlinkMacSystemFont, sans-serif",
        "font_color": "#e2e8f0",
        "grid_color": "#334155",
        "zero_line_color": "#64748b",
        "green": "#10b981",
        "red": "#ef4444",
        "blue": "#3b82f6",
        "amber": "#f59e0b",
        "purple": "#8b5cf6",
        "cyan": "#06b6d4",
    }

    @classmethod
    def _format_value(cls, val: float, unit: TargetUnit) -> str:
        """Format numerical value with proper unit scaling."""
        if unit == TargetUnit.TL:
            return f"{val / 1e6:+.2f}M TL"
        return f"{val:+.2f}%"

    @classmethod
    def _format_abs_value(cls, val: float, unit: TargetUnit) -> str:
        """Format non-signed numerical value with proper unit scaling."""
        if unit == TargetUnit.TL:
            return f"{val / 1e6:,.2f}M TL"
        return f"{val:.2f}%"

    @classmethod
    def plot_track_record(
        cls,
        df: pd.DataFrame,
        summary: Optional[BacktestSummary] = None,
        title: Optional[str] = None,
        date_col: str = "trade_date",
        actual_col: str = "actual_open_net_flow_tl",
        predicted_col: str = "predicted_open_net_flow_tl",
        lower_90_col: str = "predicted_open_flow_lower_90",
        upper_90_col: str = "predicted_open_flow_upper_90",
        is_hit_col: str = "is_direction_hit",
        unit: TargetUnit = TargetUnit.TL,
    ) -> go.Figure:
        """Plot time-series track record: Actuals vs Predictions with 90% credible ribbons and hit/miss markers."""
        data = df.copy().dropna(subset=[actual_col, predicted_col])
        if date_col in data.columns:
            data[date_col] = pd.to_datetime(data[date_col])
            data = data.sort_values(by=date_col).reset_index(drop=True)

        scale = 1e6 if unit == TargetUnit.TL else 1.0
        y_label = "Net Flow (Million TL)" if unit == TargetUnit.TL else "Return (%)"

        x_vals = data[date_col] if date_col in data.columns else list(range(len(data)))
        actual_scaled = data[actual_col] / scale
        pred_scaled = data[predicted_col] / scale

        fig = go.Figure()

        # 90% Confidence Ribbon (if available)
        has_ci = lower_90_col in data.columns and upper_90_col in data.columns
        if has_ci:
            upper_scaled = data[upper_90_col] / scale
            lower_scaled = data[lower_90_col] / scale

            # Upper bound line (invisible)
            fig.add_trace(
                go.Scatter(
                    x=x_vals,
                    y=upper_scaled,
                    mode="lines",
                    line=dict(width=0),
                    hoverinfo="skip",
                    showlegend=False,
                )
            )

            # Lower bound line filled up to upper bound
            fig.add_trace(
                go.Scatter(
                    x=x_vals,
                    y=lower_scaled,
                    mode="lines",
                    line=dict(width=0),
                    fill="tonexty",
                    fillcolor="rgba(59, 130, 246, 0.15)",  # Translucent blue
                    name="90% Credible Range",
                    hoverinfo="skip",
                )
            )

        # Realized Actual Path
        fig.add_trace(
            go.Scatter(
                x=x_vals,
                y=actual_scaled,
                mode="lines+markers",
                name="Realized Actual",
                line=dict(color="#94a3b8", width=1.8),
                marker=dict(size=5, color="#cbd5e1"),
                hovertemplate="<b>Actual</b>: %{y:+.2f}<extra></extra>",
            )
        )

        # Predicted Forecast Path
        fig.add_trace(
            go.Scatter(
                x=x_vals,
                y=pred_scaled,
                mode="lines+markers",
                name="Predicted Forecast",
                line=dict(color=cls.THEME["blue"], width=2.5, dash="dot"),
                marker=dict(size=6, color=cls.THEME["blue"]),
                hovertemplate="<b>Predicted</b>: %{y:+.2f}<extra></extra>",
            )
        )

        # Hit / Miss Evaluation Markers
        if is_hit_col in data.columns:
            hits_df = data[data[is_hit_col] == True]  # noqa: E712
            miss_df = data[data[is_hit_col] == False]  # noqa: E712

            if not hits_df.empty:
                fig.add_trace(
                    go.Scatter(
                        x=hits_df[date_col] if date_col in hits_df.columns else hits_df.index,
                        y=hits_df[actual_col] / scale,
                        mode="markers",
                        name="Direction HIT",
                        marker=dict(size=10, symbol="circle", color=cls.THEME["green"], line=dict(width=1.5, color="#ffffff")),
                        hovertemplate="<b>HIT</b><br>Actual: %{y:+.2f}<extra></extra>",
                    )
                )

            if not miss_df.empty:
                fig.add_trace(
                    go.Scatter(
                        x=miss_df[date_col] if date_col in miss_df.columns else miss_df.index,
                        y=miss_df[actual_col] / scale,
                        mode="markers",
                        name="Direction MISS",
                        marker=dict(size=10, symbol="x", color=cls.THEME["red"], line=dict(width=2)),
                        hovertemplate="<b>MISS</b><br>Actual: %{y:+.2f}<extra></extra>",
                    )
                )

        # Zero reference line
        fig.add_hline(y=0, line_width=1, line_dash="dash", line_color=cls.THEME["zero_line_color"])

        plot_title = title or (
            f"<b>{summary.model_name if summary else 'Model'} — Backtest Track Record</b>"
            f"<br><span style='font-size: 12px; color: #94a3b8;'>"
            f"Hit Rate: {summary.directional.hit_rate_pct:.1f}% | PICP 90%: {summary.probabilistic.picp_90_pct:.1f}% | MAE: {cls._format_abs_value(summary.regression.mae, unit)}"
            f"</span>"
            if summary
            else "<b>Model Backtest Track Record: Predicted vs Realized</b>"
        )

        fig.update_layout(
            title={"text": plot_title, "x": 0.03, "font": {"size": 16, "family": cls.THEME["font_family"]}},
            template="plotly_dark",
            paper_bgcolor=cls.THEME["paper_bgcolor"],
            plot_bgcolor=cls.THEME["plot_bgcolor"],
            font=dict(color=cls.THEME["font_color"], family=cls.THEME["font_family"]),
            xaxis=dict(gridcolor=cls.THEME["grid_color"], title="Session Date"),
            yaxis=dict(gridcolor=cls.THEME["grid_color"], title=y_label, zerolinecolor=cls.THEME["zero_line_color"]),
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1.0),
            margin=dict(t=80, b=50, l=60, r=40),
            height=480,
        )

        return fig

    @classmethod
    def plot_parity_and_residuals(
        cls,
        df: pd.DataFrame,
        summary: Optional[BacktestSummary] = None,
        title: Optional[str] = None,
        actual_col: str = "actual_open_net_flow_tl",
        predicted_col: str = "predicted_open_net_flow_tl",
        is_hit_col: str = "is_direction_hit",
        unit: TargetUnit = TargetUnit.TL,
    ) -> go.Figure:
        """Generate 2-panel parity scatter (Predicted vs Actual with 45-deg line) and error distribution."""
        data = df.copy().dropna(subset=[actual_col, predicted_col])
        scale = 1e6 if unit == TargetUnit.TL else 1.0
        unit_str = "M TL" if unit == TargetUnit.TL else "%"

        y_true = data[actual_col] / scale
        y_pred = data[predicted_col] / scale
        residuals = y_pred - y_true

        fig = make_subplots(
            rows=1,
            cols=2,
            subplot_titles=(
                "<b>Parity Scatter (Actual vs Predicted)</b>",
                "<b>Residual Distribution (Error = Pred - Actual)</b>",
            ),
            horizontal_spacing=0.12,
        )

        # Panel 1: Parity Scatter
        colors = [
            cls.THEME["green"] if h else cls.THEME["red"]
            for h in data.get(is_hit_col, [True] * len(data))
        ]

        fig.add_trace(
            go.Scatter(
                x=y_true,
                y=y_pred,
                mode="markers",
                name="Forecast Points",
                marker=dict(size=8, color=colors, line=dict(width=1, color="#ffffff")),
                hovertemplate=f"Actual: %{{x:+.2f}} {unit_str}<br>Predicted: %{{y:+.2f}} {unit_str}<extra></extra>",
            ),
            row=1,
            col=1,
        )

        # 45-degree reference line
        min_val = min(float(y_true.min()), float(y_pred.min()))
        max_val = max(float(y_true.max()), float(y_pred.max()))
        fig.add_trace(
            go.Scatter(
                x=[min_val, max_val],
                y=[min_val, max_val],
                mode="lines",
                name="Ideal Fit (y=x)",
                line=dict(color="#64748b", dash="dash", width=1.5),
                hoverinfo="skip",
            ),
            row=1,
            col=1,
        )

        # Panel 2: Residual Histogram
        fig.add_trace(
            go.Histogram(
                x=residuals,
                nbinsx=25,
                name="Residuals",
                marker=dict(color=cls.THEME["blue"], line=dict(color="#0f172a", width=1)),
                hovertemplate=f"Residual: %{{x:+.2f}} {unit_str}<br>Count: %{{y}}<extra></extra>",
            ),
            row=1,
            col=2,
        )

        # Mean bias vertical line in panel 2
        mean_bias = float(residuals.mean())
        fig.add_vline(
            x=mean_bias,
            line_width=2,
            line_dash="dot",
            line_color=cls.THEME["amber"],
            row=1,
            col=2,
        )

        fig.update_layout(
            title={
                "text": title or "<b>Model Residual Diagnostics & Parity Analysis</b>",
                "x": 0.03,
                "font": {"size": 16, "family": cls.THEME["font_family"]},
            },
            template="plotly_dark",
            paper_bgcolor=cls.THEME["paper_bgcolor"],
            plot_bgcolor=cls.THEME["plot_bgcolor"],
            font=dict(color=cls.THEME["font_color"], family=cls.THEME["font_family"]),
            showlegend=False,
            margin=dict(t=70, b=50, l=50, r=40),
            height=420,
        )

        fig.update_xaxes(title_text=f"Realized Actual ({unit_str})", gridcolor=cls.THEME["grid_color"], row=1, col=1)
        fig.update_yaxes(title_text=f"Predicted Output ({unit_str})", gridcolor=cls.THEME["grid_color"], row=1, col=1)
        fig.update_xaxes(title_text=f"Prediction Error ({unit_str})", gridcolor=cls.THEME["grid_color"], row=1, col=2)
        fig.update_yaxes(title_text="Frequency (Count)", gridcolor=cls.THEME["grid_color"], row=1, col=2)

        return fig

    @classmethod
    def plot_cumulative_performance(
        cls,
        df: pd.DataFrame,
        summary: Optional[BacktestSummary] = None,
        title: Optional[str] = None,
        date_col: str = "trade_date",
        actual_col: str = "actual_open_net_flow_tl",
        predicted_col: str = "predicted_open_net_flow_tl",
        is_hit_col: str = "is_direction_hit",
        unit: TargetUnit = TargetUnit.TL,
    ) -> go.Figure:
        """Plot cumulative directional capture and rolling hit-rate drift over time."""
        data = df.copy().dropna(subset=[actual_col, predicted_col])
        if date_col in data.columns:
            data[date_col] = pd.to_datetime(data[date_col])
            data = data.sort_values(by=date_col).reset_index(drop=True)

        scale = 1e6 if unit == TargetUnit.TL else 1.0
        unit_str = "Million TL" if unit == TargetUnit.TL else "%"

        y_true = data[actual_col].to_numpy()
        y_pred = data[predicted_col].to_numpy()

        directional_capture = (np.sign(y_pred) * y_true) / scale
        cum_capture = np.cumsum(directional_capture)

        hits = data[is_hit_col].to_numpy(dtype=bool) if is_hit_col in data.columns else (y_true * y_pred > 0)
        cum_hit_rate = (np.cumsum(hits) / (np.arange(len(hits)) + 1)) * 100.0

        x_vals = data[date_col] if date_col in data.columns else list(range(len(data)))

        fig = make_subplots(
            rows=2,
            cols=1,
            shared_xaxes=True,
            vertical_spacing=0.10,
            subplot_titles=(
                f"<b>Cumulative Directional Strategy Capture ({unit_str})</b>",
                "<b>Expanding Cumulative Hit Rate (%)</b>",
            ),
        )

        # Panel 1: Cumulative PnL/Capture
        fig.add_trace(
            go.Scatter(
                x=x_vals,
                y=cum_capture,
                mode="lines",
                name="Cumulative Capture",
                line=dict(color=cls.THEME["cyan"], width=2.5),
                fill="tozeroy",
                fillcolor="rgba(6, 182, 212, 0.12)",
                hovertemplate=f"Cumulative Capture: %{{y:+.2f}} {unit_str}<extra></extra>",
            ),
            row=1,
            col=1,
        )

        # Panel 2: Cumulative Hit Rate
        fig.add_trace(
            go.Scatter(
                x=x_vals,
                y=cum_hit_rate,
                mode="lines+markers",
                name="Cumulative Hit Rate %",
                line=dict(color=cls.THEME["green"], width=2),
                marker=dict(size=4),
                hovertemplate="Hit Rate: %{y:.1f}%<extra></extra>",
            ),
            row=2,
            col=1,
        )

        # 50% baseline reference in panel 2
        fig.add_hline(y=50.0, line_width=1.5, line_dash="dash", line_color="#ef4444", row=2, col=1)

        fig.update_layout(
            title={
                "text": title or "<b>Cumulative Performance & Temporal Stability Drift</b>",
                "x": 0.03,
                "font": {"size": 16, "family": cls.THEME["font_family"]},
            },
            template="plotly_dark",
            paper_bgcolor=cls.THEME["paper_bgcolor"],
            plot_bgcolor=cls.THEME["plot_bgcolor"],
            font=dict(color=cls.THEME["font_color"], family=cls.THEME["font_family"]),
            showlegend=False,
            margin=dict(t=70, b=50, l=60, r=40),
            height=500,
        )

        fig.update_yaxes(title_text=f"Total ({unit_str})", gridcolor=cls.THEME["grid_color"], row=1, col=1)
        fig.update_yaxes(title_text="Hit Rate (%)", gridcolor=cls.THEME["grid_color"], row=2, col=1)
        fig.update_xaxes(title_text="Session Date", gridcolor=cls.THEME["grid_color"], row=2, col=1)

        return fig

    @classmethod
    def plot_conviction_and_calibration(
        cls,
        summary: BacktestSummary,
        title: Optional[str] = None,
    ) -> go.Figure:
        """Plot conviction tier win rates and 90% credible interval calibration coverage."""
        fig = make_subplots(
            rows=1,
            cols=2,
            subplot_titles=(
                "<b>Hit Rate by Conviction Tier</b>",
                "<b>90% Credible Interval Coverage (PICP 90%)</b>",
            ),
            horizontal_spacing=0.15,
        )

        # Panel 1: Conviction tiers
        rates = summary.directional.conviction_hit_rates
        counts = summary.directional.conviction_counts

        if rates:
            tiers = list(rates.keys())
            hit_rates = [rates[t] for t in tiers]
            sample_counts = [counts.get(t, 0) for t in tiers]

            colors = [
                cls.THEME["green"] if hr >= 60.0 else (cls.THEME["amber"] if hr >= 50.0 else cls.THEME["red"])
                for hr in hit_rates
            ]

            fig.add_trace(
                go.Bar(
                    x=tiers,
                    y=hit_rates,
                    name="Conviction Hit Rate",
                    marker=dict(color=colors, line=dict(color="#0f172a", width=1.5)),
                    text=[f"{hr:.1f}%<br>(n={sc})" for hr, sc in zip(hit_rates, sample_counts)],
                    textposition="outside",
                    hovertemplate="Tier: %{x}<br>Hit Rate: %{y:.1f}%<extra></extra>",
                ),
                row=1,
                col=1,
            )

            # 50% baseline line
            fig.add_hline(y=50.0, line_width=1.5, line_dash="dash", line_color="#94a3b8", row=1, col=1)

        # Panel 2: PICP 90% Calibration
        picp = summary.probabilistic.picp_90_pct
        picp_color = cls.THEME["green"] if abs(picp - 90.0) <= 8.0 else cls.THEME["amber"]

        fig.add_trace(
            go.Bar(
                x=["Nominal Target (90%)", f"Realized ({picp:.1f}%)"],
                y=[90.0, picp],
                name="Calibration",
                marker=dict(color=["#3b82f6", picp_color], line=dict(color="#0f172a", width=1.5)),
                text=["90.0% Nominal", f"{picp:.1f}% Actual"],
                textposition="inside",
                textfont=dict(size=14, color="#ffffff"),
                hovertemplate="%{x}: %{y:.1f}%<extra></extra>",
            ),
            row=1,
            col=2,
        )

        fig.update_layout(
            title={
                "text": title or f"<b>{summary.model_name} — Conviction & Probabilistic Calibration</b>",
                "x": 0.03,
                "font": {"size": 16, "family": cls.THEME["font_family"]},
            },
            template="plotly_dark",
            paper_bgcolor=cls.THEME["paper_bgcolor"],
            plot_bgcolor=cls.THEME["plot_bgcolor"],
            font=dict(color=cls.THEME["font_color"], family=cls.THEME["font_family"]),
            showlegend=False,
            margin=dict(t=70, b=50, l=50, r=40),
            height=420,
        )

        fig.update_yaxes(title_text="Hit Rate (%)", range=[0, 115], gridcolor=cls.THEME["grid_color"], row=1, col=1)
        fig.update_yaxes(title_text="Coverage (%)", range=[0, 110], gridcolor=cls.THEME["grid_color"], row=1, col=2)

        return fig

    @classmethod
    def plot_slice_leaderboard(
        cls,
        slice_metrics: List[SliceMetrics],
        title: Optional[str] = None,
        max_display: int = 26,
        unit: TargetUnit = TargetUnit.TL,
    ) -> go.Figure:
        """Plot ranked horizontal leaderboard across discrete slices (e.g. sectors or symbols)."""
        if not slice_metrics:
            return go.Figure()

        displayed = sorted(slice_metrics, key=lambda s: s.hit_rate_pct, reverse=False)[-max_display:]

        keys = [s.slice_key for s in displayed]
        hit_rates = [s.hit_rate_pct for s in displayed]
        sample_counts = [s.sample_count for s in displayed]
        maes = [s.mae for s in displayed]

        colors = [
            cls.THEME["green"] if hr >= 60.0 else (cls.THEME["amber"] if hr >= 50.0 else cls.THEME["red"])
            for hr in hit_rates
        ]

        scale = 1e6 if unit == TargetUnit.TL else 1.0
        unit_str = "M TL" if unit == TargetUnit.TL else "%"

        fig = go.Figure(
            go.Bar(
                x=hit_rates,
                y=keys,
                orientation="h",
                marker=dict(color=colors, line=dict(color="#0f172a", width=1)),
                text=[f"{hr:.1f}% (n={sc})" for hr, sc in zip(hit_rates, sample_counts)],
                textposition="outside",
                customdata=np.column_stack([sample_counts, [m / scale for m in maes]]),
                hovertemplate=(
                    "<b>%{y}</b><br>"
                    "Hit Rate: %{x:.1f}%<br>"
                    "Samples: %{customdata[0]}<br>"
                    f"MAE: %{{customdata[1]:.2f}} {unit_str}<extra></extra>"
                ),
            )
        )

        # 50% baseline
        fig.add_vline(x=50.0, line_width=1.5, line_dash="dash", line_color="#94a3b8")

        fig.update_layout(
            title={
                "text": title or "<b>Cross-Entity Backtest Leaderboard (Ranked by Out-of-Sample Hit Rate)</b>",
                "x": 0.03,
                "font": {"size": 16, "family": cls.THEME["font_family"]},
            },
            template="plotly_dark",
            paper_bgcolor=cls.THEME["paper_bgcolor"],
            plot_bgcolor=cls.THEME["plot_bgcolor"],
            font=dict(color=cls.THEME["font_color"], family=cls.THEME["font_family"]),
            xaxis=dict(title="Out-of-Sample Directional Hit Rate (%)", range=[0, 115], gridcolor=cls.THEME["grid_color"]),
            yaxis=dict(gridcolor=cls.THEME["grid_color"]),
            margin=dict(t=70, b=50, l=100, r=60),
            height=max(400, len(displayed) * 22),
        )

        return fig

    @classmethod
    def plot_stock_window_matrix(
        cls,
        df: pd.DataFrame,
        title: Optional[str] = None,
        symbol_col: str = "symbol",
        window_col: str = "window_name",
        is_hit_col: str = "is_direction_hit",
    ) -> go.Figure:
        """Plot Model 3 Stock x Window directional hit rate heatmap matrix."""
        if df.empty or symbol_col not in df.columns or window_col not in df.columns:
            return go.Figure()

        # Compute hit rate per (symbol, window)
        grouped = df.groupby([symbol_col, window_col])[is_hit_col].agg(["count", "sum"]).reset_index()
        grouped["hit_rate_pct"] = (grouped["sum"] / grouped["count"]) * 100.0

        # Map window names to clean display labels
        window_display_names = {
            "first_reaction": "W2: First Reaction (10:30-11:30)",
            "midday_followup": "W3: Midday (11:30-14:30)",
            "closing_session": "W5: Closing (16:00-18:15)",
            "w2": "W2: First Reaction (10:30-11:30)",
            "w3": "W3: Midday (11:30-14:30)",
            "w5": "W5: Closing (16:00-18:15)",
            "W2": "W2: First Reaction (10:30-11:30)",
            "W3": "W3: Midday (11:30-14:30)",
            "W5": "W5: Closing (16:00-18:15)",
        }
        grouped["_win_disp"] = grouped[window_col].map(lambda x: window_display_names.get(x, str(x)))

        pivot_df = grouped.pivot(index=symbol_col, columns="_win_disp", values="hit_rate_pct")

        # Chronological column ordering
        desired_order = [
            "W2: First Reaction (10:30-11:30)",
            "W3: Midday (11:30-14:30)",
            "W5: Closing (16:00-18:15)",
        ]
        col_order = [c for c in desired_order if c in pivot_df.columns] + [
            c for c in pivot_df.columns if c not in desired_order
        ]
        if col_order:
            pivot_df = pivot_df[col_order]

        symbols = pivot_df.index.tolist()
        windows = pivot_df.columns.tolist()
        z_vals = pivot_df.values

        fig = go.Figure(
            go.Heatmap(
                z=z_vals,
                x=windows,
                y=symbols,
                colorscale=[
                    [0.0, "#ef4444"],   # Red for low hit rate
                    [0.5, "#334155"],   # Neutral slate for 50%
                    [1.0, "#10b981"],   # Green for high hit rate
                ],
                zmin=30.0,
                zmax=80.0,
                colorbar=dict(title="Hit Rate %"),
                text=np.round(z_vals, 1),
                texttemplate="%{text}%",
                hoverongaps=False,
                hovertemplate="<b>%{y}</b> | %{x}<br>Hit Rate: %{z:.1f}%<extra></extra>",
            )
        )

        fig.update_layout(
            title={
                "text": title or "<b>Model 3: Stock Intraday Reaction Performance Matrix (Symbol x Window)</b>",
                "x": 0.03,
                "font": {"size": 16, "family": cls.THEME["font_family"]},
            },
            template="plotly_dark",
            paper_bgcolor=cls.THEME["paper_bgcolor"],
            plot_bgcolor=cls.THEME["plot_bgcolor"],
            font=dict(color=cls.THEME["font_color"], family=cls.THEME["font_family"]),
            xaxis=dict(title="Intraday Reaction Window", gridcolor=cls.THEME["grid_color"]),
            yaxis=dict(title="BIST 30 Symbol", gridcolor=cls.THEME["grid_color"]),
            margin=dict(t=70, b=50, l=80, r=40),
            height=max(450, len(symbols) * 18),
        )

        return fig

    @classmethod
    def format_executive_scorecard_html(cls, summary: BacktestSummary) -> str:
        """Format an executive HTML card summarizing key backtest KPIs with clean status badges."""
        unit = summary.target_unit
        hit_rate = summary.directional.hit_rate_pct
        picp = summary.probabilistic.picp_90_pct
        mae_str = cls._format_abs_value(summary.regression.mae, unit)
        rmse_str = cls._format_abs_value(summary.regression.rmse, unit)

        hit_color = "#10b981" if hit_rate >= 55.0 else ("#f59e0b" if hit_rate >= 50.0 else "#ef4444")
        picp_color = "#10b981" if abs(picp - 90.0) <= 8.0 else "#f59e0b"

        card_html = f"""
        <div style="background-color: #0f172a; border: 1px solid #334155; border-radius: 10px; padding: 20px; font-family: Inter, sans-serif; color: #e2e8f0; margin-bottom: 20px;">
            <div style="display: flex; justify-content: space-between; align-items: center; border-bottom: 1px solid #1e293b; padding-bottom: 12px; margin-bottom: 16px;">
                <div>
                    <span style="font-size: 18px; font-weight: 700; color: #ffffff;">{summary.model_name}</span>
                    <span style="font-size: 13px; color: #94a3b8; margin-left: 10px;">Target: {summary.target_name}</span>
                </div>
                <div>
                    <span style="background-color: #1e293b; color: #93c5fd; padding: 4px 10px; border-radius: 6px; font-size: 12px; font-weight: 600;">{summary.total_samples:,} Sessions</span>
                </div>
            </div>

            <div style="display: grid; grid-template-columns: repeat(auto-fit, minmax(140px, 1fr)); gap: 14px;">
                <div style="background-color: #1e293b; padding: 12px; border-radius: 8px;">
                    <div style="font-size: 11px; color: #94a3b8; text-transform: uppercase; font-weight: 600;">Direction Hit Rate</div>
                    <div style="font-size: 22px; font-weight: 700; color: {hit_color}; margin-top: 4px;">{hit_rate:.1f}%</div>
                    <div style="font-size: 11px; color: #64748b; margin-top: 2px;">Baseline: 50.0%</div>
                </div>

                <div style="background-color: #1e293b; padding: 12px; border-radius: 8px;">
                    <div style="font-size: 11px; color: #94a3b8; text-transform: uppercase; font-weight: 600;">90% CI Coverage</div>
                    <div style="font-size: 22px; font-weight: 700; color: {picp_color}; margin-top: 4px;">{picp:.1f}%</div>
                    <div style="font-size: 11px; color: #64748b; margin-top: 2px;">Target: 90.0% PICP</div>
                </div>

                <div style="background-color: #1e293b; padding: 12px; border-radius: 8px;">
                    <div style="font-size: 11px; color: #94a3b8; text-transform: uppercase; font-weight: 600;">Mean Abs Error (MAE)</div>
                    <div style="font-size: 22px; font-weight: 700; color: #e2e8f0; margin-top: 4px;">{mae_str}</div>
                    <div style="font-size: 11px; color: #64748b; margin-top: 2px;">RMSE: {rmse_str}</div>
                </div>

                <div style="background-color: #1e293b; padding: 12px; border-radius: 8px;">
                    <div style="font-size: 11px; color: #94a3b8; text-transform: uppercase; font-weight: 600;">Profit Factor Proxy</div>
                    <div style="font-size: 22px; font-weight: 700; color: #38bdf8; margin-top: 4px;">{summary.trading.profit_factor:.2f}x</div>
                    <div style="font-size: 11px; color: #64748b; margin-top: 2px;">Win/Loss: {summary.trading.win_loss_ratio:.2f}</div>
                </div>

                <div style="background-color: #1e293b; padding: 12px; border-radius: 8px;">
                    <div style="font-size: 11px; color: #94a3b8; text-transform: uppercase; font-weight: 600;">Correlation (r)</div>
                    <div style="font-size: 22px; font-weight: 700; color: #a855f7; margin-top: 4px;">{summary.regression.pearson_r:+.2f}</div>
                    <div style="font-size: 11px; color: #64748b; margin-top: 2px;">R²: {summary.regression.r2:.2f}</div>
                </div>
            </div>
        </div>
        """
        return card_html
