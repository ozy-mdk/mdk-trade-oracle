"""Visualizer: Interactive Plotly waterfall charts, cluster donut charts, and markdown cards for model explainability."""

from typing import Optional

import plotly.graph_objects as go

from mdk_trading_oracle.explainability.types import GlobalExplanation, LocalExplanation


def plot_waterfall(
    local_exp: LocalExplanation,
    max_display: int = 8,
    title: Optional[str] = None,
) -> go.Figure:
    """Generate an interactive Plotly waterfall chart decomposing a live T+1 forecast.

    Args:
        local_exp: LocalExplanation instance from ModelExplainer.
        max_display: Maximum number of individual features to display before grouping into 'Other Features'.
        title: Optional custom plot title.

    Returns:
        Plotly go.Figure waterfall chart.
    """
    unit = local_exp.unit
    is_tl = unit == "TL"

    # Sort attributions by absolute impact
    sorted_attrs = sorted(local_exp.feature_attributions, key=lambda a: a.attribution_abs, reverse=True)

    displayed = sorted_attrs[:max_display]
    remaining = sorted_attrs[max_display:]

    x_labels = ["Baseline E[y]"]
    y_values = [local_exp.base_value]
    measures = ["absolute"]
    text_labels = [
        f"{local_exp.base_value / 1e6:+.1f}M TL" if is_tl else f"{local_exp.base_value:+.2f}%"
    ]
    hover_texts = [f"Baseline Expected Value: {text_labels[0]}"]

    for a in displayed:
        x_labels.append(a.feature_name.replace("feat_", ""))
        y_values.append(a.attribution)
        measures.append("relative")
        val_str = f"{a.attribution / 1e6:+.1f}M TL" if is_tl else f"{a.attribution:+.2f}%"
        text_labels.append(val_str)
        hover_texts.append(
            f"Feature: {a.feature_name}<br>"
            f"Cluster: {a.cluster_name}<br>"
            f"Feature Value: {a.feature_value:,.2f}<br>"
            f"Attribution: {val_str}"
        )

    # Group remaining features
    if remaining:
        other_sum = sum(a.attribution for a in remaining)
        x_labels.append("Other Features")
        y_values.append(other_sum)
        measures.append("relative")
        val_str = f"{other_sum / 1e6:+.1f}M TL" if is_tl else f"{other_sum:+.2f}%"
        text_labels.append(val_str)
        hover_texts.append(f"Sum of {len(remaining)} other features: {val_str}")

    # Final forecast bar
    x_labels.append("Final Forecast")
    y_values.append(local_exp.predicted_value)
    measures.append("total")
    final_str = f"{local_exp.predicted_value / 1e6:+.1f}M TL" if is_tl else f"{local_exp.predicted_value:+.2f}%"
    text_labels.append(final_str)
    hover_texts.append(f"Predicted Output: {final_str}")

    fig = go.Figure(
        go.Waterfall(
            name="SHAP Attribution",
            orientation="v",
            measure=measures,
            x=x_labels,
            textposition="outside",
            text=text_labels,
            y=y_values,
            hovertext=hover_texts,
            hoverinfo="text",
            connector={"line": {"color": "#64748b", "width": 1.5}},
            decreasing={"marker": {"color": "#ef4444"}},  # Red for negative headwinds
            increasing={"marker": {"color": "#10b981"}},  # Green for positive catalysts
            totals={"marker": {"color": "#3b82f6"}},      # Blue for total / baseline
        )
    )

    plot_title = (
        title
        or f"{local_exp.model_name} — Microstructure Driver Waterfall ({local_exp.target_broker_or_symbol})"
    )

    fig.update_layout(
        title={
            "text": f"<b>{plot_title}</b>",
            "x": 0.05,
            "font": {"size": 16, "family": "Inter, sans-serif"},
        },
        showlegend=False,
        template="plotly_dark",
        paper_bgcolor="#0f172a",
        plot_bgcolor="#1e293b",
        font={"color": "#e2e8f0", "family": "Inter, sans-serif"},
        margin=dict(t=60, b=80, l=60, r=40),
        xaxis={"tickangle": -35, "gridcolor": "#334155"},
        yaxis={
            "title": f"Flow Impact ({unit})" if not is_tl else "Flow Impact (TL)",
            "gridcolor": "#334155",
            "zerolinecolor": "#64748b",
        },
        height=480,
    )

    return fig


def plot_cluster_donut(
    global_exp: GlobalExplanation,
    title: Optional[str] = None,
) -> go.Figure:
    """Generate an interactive Plotly donut chart showing % alpha share by semantic microstructure cluster.

    Args:
        global_exp: GlobalExplanation instance from ModelExplainer.
        title: Optional custom plot title.

    Returns:
        Plotly go.Figure donut chart.
    """
    df = global_exp.cluster_importance_df
    if df.empty:
        return go.Figure()

    labels = df["cluster_name"].tolist()
    values = df["relative_importance_pct"].tolist()

    colors = [
        "#3b82f6",  # Blue
        "#10b981",  # Emerald
        "#f59e0b",  # Amber
        "#8b5cf6",  # Purple
        "#06b6d4",  # Cyan
        "#ec4899",  # Pink
        "#64748b",  # Slate
        "#f97316",  # Orange
        "#14b8a6",  # Teal
        "#6366f1",  # Indigo
    ]

    fig = go.Figure(
        data=[
            go.Pie(
                labels=labels,
                values=values,
                hole=0.55,
                marker=dict(colors=colors[: len(labels)], line=dict(color="#0f172a", width=2)),
                textinfo="label+percent",
                hoverinfo="label+percent+value",
                textfont=dict(size=12, family="Inter, sans-serif"),
            )
        ]
    )

    plot_title = title or f"{global_exp.model_name} — Microstructure Cluster Alpha Share"

    fig.update_layout(
        title={
            "text": f"<b>{plot_title}</b>",
            "x": 0.05,
            "font": {"size": 16, "family": "Inter, sans-serif"},
        },
        template="plotly_dark",
        paper_bgcolor="#0f172a",
        plot_bgcolor="#1e293b",
        font={"color": "#e2e8f0", "family": "Inter, sans-serif"},
        legend=dict(orientation="h", yanchor="bottom", y=-0.2, xanchor="center", x=0.5),
        margin=dict(t=60, b=60, l=40, r=40),
        height=450,
    )

    return fig


def format_markdown_card(local_exp: LocalExplanation) -> str:
    """Format a clean, markdown executive summary table of catalysts and headwinds."""
    is_tl = local_exp.unit == "TL"

    def fmt_val(v: float) -> str:
        if is_tl:
            return f"{v / 1e6:+.2f}M TL"
        return f"{v:+.2f}%"

    pred_str = fmt_val(local_exp.predicted_value)
    base_str = fmt_val(local_exp.base_value)

    lines = [
        f"### Forecast Attribution Breakdown: {local_exp.target_broker_or_symbol} ({pred_str})",
        f"- **Expected Market Baseline (E[y])**: `{base_str}`",
        f"- **Net Forecast**: `{pred_str}`",
        "",
        "#### Top Catalysts (Bullish Drivers)",
        "| Feature | Semantic Cluster | Value | Attribution |",
        "| :--- | :--- | :---: | :---: |",
    ]

    if local_exp.top_positive_drivers:
        for d in local_exp.top_positive_drivers:
            lines.append(
                f"| `{d.feature_name}` | {d.cluster_name} | {d.feature_value:,.2f} | **{fmt_val(d.attribution)}** |"
            )
    else:
        lines.append("| *None* | - | - | - |")

    lines.extend(
        [
            "",
            "#### Top Headwinds (Bearish Drag)",
            "| Feature | Semantic Cluster | Value | Attribution |",
            "| :--- | :--- | :---: | :---: |",
        ]
    )

    if local_exp.top_negative_drivers:
        for d in local_exp.top_negative_drivers:
            lines.append(
                f"| `{d.feature_name}` | {d.cluster_name} | {d.feature_value:,.2f} | **{fmt_val(d.attribution)}** |"
            )
    else:
        lines.append("| *None* | - | - | - |")

    lines.extend(
        [
            "",
            "#### Microstructure Cluster Rollup",
            "| Cluster | Net Contribution | Share (%) | Top Driver |",
            "| :--- | :---: | :---: | :--- |",
        ]
    )

    for c in local_exp.cluster_attributions:
        lines.append(
            f"| **{c.cluster_name}** | {fmt_val(c.total_attribution)} | {c.percentage_share:.1f}% | `{c.top_feature}` |"
        )

    return "\n".join(lines)
