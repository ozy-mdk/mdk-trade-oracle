"""Executive report generator and cross-model performance benchmarking."""

from typing import List

import pandas as pd

from mdk_trading_oracle.backtest.types import BacktestSummary, TargetUnit


class BacktestReportGenerator:
    """Generates markdown executive reports and comparative tables across predictive models."""

    @classmethod
    def compare_models(cls, summaries: List[BacktestSummary]) -> pd.DataFrame:
        """Construct a side-by-side comparative DataFrame benchmark across multiple models.

        Args:
            summaries: List of BacktestSummary objects to benchmark.

        Returns:
            pd.DataFrame ranking models on directional accuracy, calibration, and regression error.
        """
        rows = []
        for s in summaries:
            scale_str = "M TL" if s.target_unit == TargetUnit.TL else "%"
            mae_disp = s.regression.mae / 1e6 if s.target_unit == TargetUnit.TL else s.regression.mae
            rmse_disp = s.regression.rmse / 1e6 if s.target_unit == TargetUnit.TL else s.regression.rmse

            rows.append(
                {
                    "Model": s.model_name,
                    "Target Scope": s.target_name,
                    "Total Sessions": s.total_samples,
                    "Hit Rate (%)": round(s.directional.hit_rate_pct, 1),
                    "PICP 90% (%)": round(s.probabilistic.picp_90_pct, 1),
                    "MAE": f"{mae_disp:.2f} {scale_str}",
                    "RMSE": f"{rmse_disp:.2f} {scale_str}",
                    "Pearson r": f"{s.regression.pearson_r:+.2f}",
                    "R²": f"{s.regression.r2:.2f}",
                    "Profit Factor": f"{s.trading.profit_factor:.2f}x",
                }
            )

        df_comp = pd.DataFrame(rows)
        return df_comp

    @classmethod
    def generate_markdown_summary(cls, summary: BacktestSummary) -> str:
        """Format an executive markdown report block for a single model backtest."""
        unit = summary.target_unit
        is_tl = unit == TargetUnit.TL
        scale = 1e6 if is_tl else 1.0
        unit_str = "M TL" if is_tl else "%"

        status_hit = "[PASS]" if summary.directional.hit_rate_pct >= 55.0 else "[WARN]"
        status_calib = "[CALIBRATED]" if abs(summary.probabilistic.coverage_bias_pct) <= 8.0 else "[DRIFT]"

        lines = [
            f"## Backtest Performance Summary: {summary.model_name}",
            f"- **Target Definition**: `{summary.target_name}`",
            f"- **Sample Universe**: `{summary.total_samples:,}` sessions ({summary.start_date} to {summary.end_date})",
            "",
            "### Primary Quantitative Scorecard",
            "| Evaluation Lens | Metric | Realized Value | Benchmark / Target | Status |",
            "| :--- | :--- | :---: | :---: | :---: |",
            f"| **Directional Accuracy** | Sign Hit Rate % | **{summary.directional.hit_rate_pct:.1f}%** | > 50.0% (Coin Toss) | {status_hit} |",
            f"| **Probabilistic Calibration** | 90% CI Coverage (PICP) | **{summary.probabilistic.picp_90_pct:.1f}%** | 90.0% Nominal | {status_calib} |",
            f"| **Regression Precision** | Mean Absolute Error (MAE) | **{summary.regression.mae / scale:,.2f} {unit_str}** | Minimized | [OK] |",
            f"| **Regression Dispersion** | Root Mean Square Error (RMSE) | **{summary.regression.rmse / scale:,.2f} {unit_str}** | Minimized | [OK] |",
            f"| **Linear Concordance** | Pearson Correlation (r) | **{summary.regression.pearson_r:+.2f}** | > 0.00 | [OK] |",
            f"| **Trading Utility Proxy** | Profit Factor | **{summary.trading.profit_factor:.2f}x** | > 1.00x | [OK] |",
            f"| **Trading Utility Proxy** | Win / Loss Ratio | **{summary.trading.win_loss_ratio:.2f}** | > 1.00 | [OK] |",
            "",
        ]

        if summary.directional.conviction_hit_rates:
            lines.extend(
                [
                    "### Conviction Tier Breakdown",
                    "| Conviction Tier | Sample Count | Realized Hit Rate % |",
                    "| :--- | :---: | :---: |",
                ]
            )
            for tier, hr in summary.directional.conviction_hit_rates.items():
                cnt = summary.directional.conviction_counts.get(tier, 0)
                lines.append(f"| `{tier}` | {cnt} | **{hr:.1f}%** |")
            lines.append("")

        if summary.slices:
            for slice_name, slice_list in summary.slices.items():
                lines.extend(
                    [
                        f"### Sub-segment Breakdown: `{slice_name}`",
                        "| Slice Key | Samples | Hit Rate % | MAE | PICP 90% | Directional Capture |",
                        "| :--- | :---: | :---: | :---: | :---: | :---: |",
                    ]
                )
                for s in slice_list[:10]:
                    lines.append(
                        f"| **{s.slice_key}** | {s.sample_count} | {s.hit_rate_pct:.1f}% | "
                        f"{s.mae / scale:,.2f} {unit_str} | {s.picp_90_pct:.1f}% | "
                        f"{s.directional_capture / scale:+,.2f} {unit_str} |"
                    )
                if len(slice_list) > 10:
                    lines.append(f"| *... and {len(slice_list) - 10} more slices* | - | - | - | - | - |")
                lines.append("")

        return "\n".join(lines)
