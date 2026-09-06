"""Pure statistical and quantitative calculation engine for backtest performance."""

from typing import Dict, List, Optional, Union

import numpy as np
import pandas as pd

from mdk_trading_oracle.backtest.types import (
    BacktestSummary,
    DirectionalMetrics,
    ProbabilisticMetrics,
    RegressionMetrics,
    SliceMetrics,
    TargetUnit,
    TradingUtilityMetrics,
)
from mdk_trading_oracle.core.logger import get_logger
from mdk_trading_oracle.core.time import now_turkey_iso

logger = get_logger("mdk_oracle.backtest.metrics")


class BacktestMetricsCalculator:
    """Universal calculator for quantitative backtest performance metrics across any target definition."""

    @classmethod
    def calculate_summary(
        cls,
        df: pd.DataFrame,
        model_name: str,
        target_name: str = "open_net_flow_tl",
        actual_col: str = "actual_open_net_flow_tl",
        predicted_col: str = "predicted_open_net_flow_tl",
        lower_90_col: str = "predicted_open_flow_lower_90",
        upper_90_col: str = "predicted_open_flow_upper_90",
        direction_col: str = "predicted_direction",
        actual_direction_col: Optional[str] = "actual_direction",
        is_hit_col: Optional[str] = "is_direction_hit",
        is_inside_ci_col: Optional[str] = "is_inside_90_ci",
        date_col: str = "trade_date",
        target_unit: Union[TargetUnit, str] = TargetUnit.TL,
        slice_columns: Optional[List[str]] = None,
    ) -> BacktestSummary:
        """Compute full backtest performance summary from evaluated records.

        Args:
            df: DataFrame containing backtest/performance rows.
            model_name: Name of evaluated model.
            target_name: Semantic name of target variable.
            actual_col: Column name containing realized ground truth.
            predicted_col: Column name containing point prediction.
            lower_90_col: Column name for lower 90% credible bound.
            upper_90_col: Column name for upper 90% credible bound.
            direction_col: Column name for predicted directional conviction.
            actual_direction_col: Optional column for realized direction.
            is_hit_col: Optional column indicating precomputed direction hit.
            is_inside_ci_col: Optional column indicating precomputed CI containment.
            date_col: Date column for temporal ordering.
            target_unit: TargetUnit enum (TL or %).
            slice_columns: List of columns to aggregate slices across (e.g. ['sector', 'window_name']).

        Returns:
            BacktestSummary containing all computed metric categories and slice breakdowns.
        """
        if df.empty:
            raise ValueError(f"Cannot calculate backtest metrics on empty DataFrame for model '{model_name}'.")

        # Create sorted copy by date if present
        data = df.copy()
        if date_col in data.columns:
            data[date_col] = pd.to_datetime(data[date_col])
            data = data.sort_values(by=date_col).reset_index(drop=True)

        unit_enum = TargetUnit(target_unit) if isinstance(target_unit, str) else target_unit

        # Filter out rows where actual or predicted are NaN
        data = data.dropna(subset=[actual_col, predicted_col])
        if data.empty:
            raise ValueError(f"No valid non-null rows remaining in backtest dataset for model '{model_name}'.")

        y_true = data[actual_col].to_numpy(dtype=float)
        y_pred = data[predicted_col].to_numpy(dtype=float)

        lower_90 = data[lower_90_col].to_numpy(dtype=float) if lower_90_col in data.columns else None
        upper_90 = data[upper_90_col].to_numpy(dtype=float) if upper_90_col in data.columns else None

        # Regression Metrics
        reg_metrics = cls.calculate_regression(y_true, y_pred)

        # Directional Metrics
        dir_metrics = cls.calculate_directional(
            data=data,
            y_true=y_true,
            y_pred=y_pred,
            direction_col=direction_col,
            is_hit_col=is_hit_col,
        )

        # Probabilistic Metrics
        prob_metrics = cls.calculate_probabilistic(
            data=data,
            y_true=y_true,
            lower_90=lower_90,
            upper_90=upper_90,
            is_inside_ci_col=is_inside_ci_col,
        )

        # Simulated Trading Utility Metrics
        trading_metrics = cls.calculate_trading_utility(y_true, y_pred)

        # Slice Breakdowns
        slice_dict: Dict[str, List[SliceMetrics]] = {}
        if slice_columns:
            for sc in slice_columns:
                if sc in data.columns:
                    slice_dict[sc] = cls.calculate_slice_metrics(
                        data=data,
                        slice_col=sc,
                        actual_col=actual_col,
                        predicted_col=predicted_col,
                        lower_90_col=lower_90_col if lower_90 is not None else None,
                        upper_90_col=upper_90_col if upper_90 is not None else None,
                        is_hit_col=is_hit_col,
                        is_inside_ci_col=is_inside_ci_col,
                    )

        start_date = data[date_col].min().date() if date_col in data.columns else None
        end_date = data[date_col].max().date() if date_col in data.columns else None

        return BacktestSummary(
            model_name=model_name,
            target_name=target_name,
            target_unit=unit_enum,
            total_samples=len(data),
            start_date=start_date,
            end_date=end_date,
            regression=reg_metrics,
            directional=dir_metrics,
            probabilistic=prob_metrics,
            trading=trading_metrics,
            slices=slice_dict,
            calculated_at=now_turkey_iso(),
        )

    @classmethod
    def calculate_regression(cls, y_true: np.ndarray, y_pred: np.ndarray) -> RegressionMetrics:
        """Compute continuous regression performance metrics."""
        errors = y_pred - y_true
        abs_errors = np.abs(errors)

        mae = float(np.mean(abs_errors))
        rmse = float(np.sqrt(np.mean(errors**2)))
        mean_bias = float(np.mean(errors))
        max_error = float(np.max(abs_errors))

        # Normalized MAE: standardized by std(y_true) or 1.0 if zero std
        std_y = float(np.std(y_true))
        normalized_mae = mae / std_y if std_y > 1e-9 else 0.0

        # Pearson correlation
        if len(y_true) > 1 and std_y > 1e-9 and np.std(y_pred) > 1e-9:
            corr_mat = np.corrcoef(y_true, y_pred)
            pearson_r = float(corr_mat[0, 1]) if not np.isnan(corr_mat[0, 1]) else 0.0
        else:
            pearson_r = 0.0

        # Coefficient of determination R^2
        ss_res = float(np.sum(errors**2))
        ss_tot = float(np.sum((y_true - np.mean(y_true)) ** 2))
        r2 = float(1.0 - (ss_res / ss_tot)) if ss_tot > 1e-9 else 0.0

        return RegressionMetrics(
            mae=mae,
            rmse=rmse,
            pearson_r=pearson_r,
            r2=r2,
            mean_bias=mean_bias,
            normalized_mae=normalized_mae,
            max_error=max_error,
        )

    @classmethod
    def calculate_directional(
        cls,
        data: pd.DataFrame,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        direction_col: str = "predicted_direction",
        is_hit_col: Optional[str] = None,
    ) -> DirectionalMetrics:
        """Compute directional hit rates, conviction breakdowns, and monotonicity."""
        total = len(y_true)
        if is_hit_col and is_hit_col in data.columns and not data[is_hit_col].isnull().all():
            hits_arr = data[is_hit_col].to_numpy(dtype=bool)
        else:
            # Sign agreement: (y_true * y_pred > 0) or both zero
            hits_arr = (y_true * y_pred > 0) | ((y_true == 0) & (y_pred == 0))

        hits = int(np.sum(hits_arr))
        misses = total - hits
        hit_rate_pct = float(hits / total * 100.0) if total > 0 else 0.0

        conviction_hit_rates: Dict[str, float] = {}
        conviction_counts: Dict[str, int] = {}

        if direction_col in data.columns:
            data_with_hits = data.copy()
            data_with_hits["_hit_eval"] = hits_arr
            grouped = data_with_hits.groupby(direction_col)
            for tier, group in grouped:
                c_total = len(group)
                c_hits = int(group["_hit_eval"].sum())
                tier_str = str(tier)
                conviction_counts[tier_str] = c_total
                conviction_hit_rates[tier_str] = float(c_hits / c_total * 100.0) if c_total > 0 else 0.0

        # Compute Monotonicity Score (Rank correlation between conviction tier and win rate)
        tier_order = {
            "STRONG_BUY": 3,
            "BUY": 2,
            "WEAK_BUY": 1,
            "NEUTRAL": 0,
            "WEAK_SELL": 1,
            "SELL": 2,
            "STRONG_SELL": 3,
            "RALLY": 2,
            "STRONG_RALLY": 3,
            "DECLINE": 2,
            "STRONG_DECLINE": 3,
        }

        monotonicity = cls._compute_monotonicity(conviction_hit_rates, tier_order)

        return DirectionalMetrics(
            hit_rate_pct=hit_rate_pct,
            total_predictions=total,
            hits=hits,
            misses=misses,
            conviction_hit_rates=conviction_hit_rates,
            conviction_counts=conviction_counts,
            monotonicity_score=monotonicity,
        )

    @classmethod
    def calculate_probabilistic(
        cls,
        data: pd.DataFrame,
        y_true: np.ndarray,
        lower_90: Optional[np.ndarray],
        upper_90: Optional[np.ndarray],
        is_inside_ci_col: Optional[str] = None,
    ) -> ProbabilisticMetrics:
        """Compute probabilistic calibration, 90% interval coverage (PICP), and interval width (MPIW)."""
        total = len(y_true)
        if is_inside_ci_col and is_inside_ci_col in data.columns and not data[is_inside_ci_col].isnull().all():
            inside_ci = data[is_inside_ci_col].to_numpy(dtype=bool)
            picp_90_pct = float(np.sum(inside_ci) / total * 100.0) if total > 0 else 0.0
        elif lower_90 is not None and upper_90 is not None:
            inside_ci = (y_true >= lower_90) & (y_true <= upper_90)
            picp_90_pct = float(np.sum(inside_ci) / total * 100.0) if total > 0 else 0.0
        else:
            picp_90_pct = 0.0

        coverage_bias = picp_90_pct - 90.0

        if lower_90 is not None and upper_90 is not None:
            widths = upper_90 - lower_90
            mpiw = float(np.mean(widths))

            # Winkler Score for 90% interval (alpha = 0.10)
            alpha = 0.10
            under = np.maximum(0, lower_90 - y_true)
            over = np.maximum(0, y_true - upper_90)
            winkler = widths + (2.0 / alpha) * under + (2.0 / alpha) * over
            interval_score = float(np.mean(winkler))
        else:
            mpiw = 0.0
            interval_score = 0.0

        return ProbabilisticMetrics(
            picp_90_pct=picp_90_pct,
            mpiw=mpiw,
            coverage_bias_pct=coverage_bias,
            interval_score=interval_score,
        )

    @classmethod
    def calculate_trading_utility(cls, y_true: np.ndarray, y_pred: np.ndarray) -> TradingUtilityMetrics:
        """Compute directional trading utility and capture proxies."""
        # Simulated trade direction: sign of prediction (+1 or -1)
        # Directional capture = sign(y_pred) * y_true
        pred_sign = np.sign(y_pred)
        # Where prediction is 0, capture is 0
        directional_pnl = pred_sign * y_true

        total_capture = float(np.sum(directional_pnl))
        cum_pnl = np.cumsum(directional_pnl).tolist()

        wins = directional_pnl[directional_pnl > 0]
        losses = directional_pnl[directional_pnl < 0]

        mean_win = float(np.mean(wins)) if len(wins) > 0 else 0.0
        mean_loss = float(np.abs(np.mean(losses))) if len(losses) > 0 else 0.0
        win_loss_ratio = mean_win / mean_loss if mean_loss > 1e-9 else (1.0 if mean_win > 0 else 0.0)

        sum_wins = float(np.sum(wins)) if len(wins) > 0 else 0.0
        sum_losses = float(np.sum(np.abs(losses))) if len(losses) > 0 else 0.0
        profit_factor = sum_wins / sum_losses if sum_losses > 1e-9 else (99.0 if sum_wins > 0 else 0.0)

        return TradingUtilityMetrics(
            directional_capture_total=total_capture,
            win_loss_ratio=win_loss_ratio,
            profit_factor=profit_factor,
            cumulative_directional_pnl=cum_pnl,
        )

    @classmethod
    def calculate_slice_metrics(
        cls,
        data: pd.DataFrame,
        slice_col: str,
        actual_col: str,
        predicted_col: str,
        lower_90_col: Optional[str] = None,
        upper_90_col: Optional[str] = None,
        is_hit_col: Optional[str] = None,
        is_inside_ci_col: Optional[str] = None,
    ) -> List[SliceMetrics]:
        """Aggregate performance across discrete categories (e.g. sectors, symbols, windows)."""
        slices: List[SliceMetrics] = []
        for key, group in data.groupby(slice_col):
            if group.empty:
                continue
            y_t = group[actual_col].to_numpy(dtype=float)
            y_p = group[predicted_col].to_numpy(dtype=float)
            count = len(group)

            err = y_p - y_t
            mae = float(np.mean(np.abs(err)))
            rmse = float(np.sqrt(np.mean(err**2)))

            # Hit Rate
            if is_hit_col and is_hit_col in group.columns and not group[is_hit_col].isnull().all():
                hits = float(np.sum(group[is_hit_col].to_numpy(dtype=bool)))
            else:
                hits = float(np.sum((y_t * y_p > 0) | ((y_t == 0) & (y_p == 0))))
            hit_rate_pct = hits / count * 100.0 if count > 0 else 0.0

            # PICP 90
            if is_inside_ci_col and is_inside_ci_col in group.columns and not group[is_inside_ci_col].isnull().all():
                inside = float(np.sum(group[is_inside_ci_col].to_numpy(dtype=bool)))
                picp = inside / count * 100.0 if count > 0 else 0.0
            elif lower_90_col and upper_90_col and lower_90_col in group.columns and upper_90_col in group.columns:
                low = group[lower_90_col].to_numpy(dtype=float)
                high = group[upper_90_col].to_numpy(dtype=float)
                inside = float(np.sum((y_t >= low) & (y_t <= high)))
                picp = inside / count * 100.0 if count > 0 else 0.0
            else:
                picp = 0.0

            # Directional Capture
            capture = float(np.sum(np.sign(y_p) * y_t))

            slices.append(
                SliceMetrics(
                    slice_type=slice_col,
                    slice_key=str(key),
                    sample_count=count,
                    hit_rate_pct=hit_rate_pct,
                    mae=mae,
                    rmse=rmse,
                    picp_90_pct=picp,
                    directional_capture=capture,
                )
            )

        # Sort by sample count and hit rate
        slices.sort(key=lambda s: (s.hit_rate_pct, s.sample_count), reverse=True)
        return slices

    @staticmethod
    def _compute_monotonicity(conviction_rates: Dict[str, float], tier_order: Dict[str, int]) -> float:
        """Compute rank monotonicity score between conviction tiers and win rates."""
        tier_items = [(tier_order[k], v) for k, v in conviction_rates.items() if k in tier_order]
        if len(tier_items) < 2:
            return 0.0

        levels = [item[0] for item in tier_items]
        rates = [item[1] for item in tier_items]

        if len(set(levels)) <= 1 or len(set(rates)) <= 1:
            return 0.0

        corr = np.corrcoef(levels, rates)[0, 1]
        return float(corr) if not np.isnan(corr) else 0.0
