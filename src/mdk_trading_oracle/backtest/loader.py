"""DuckDB reader and standardized loader for backtest and performance ledgers across all models."""

from datetime import date
from typing import List, Optional, Union

import pandas as pd

from mdk_trading_oracle.backtest.metrics import BacktestMetricsCalculator
from mdk_trading_oracle.backtest.types import BacktestSummary, TargetUnit
from mdk_trading_oracle.core.db import DuckDBManager
from mdk_trading_oracle.core.logger import get_logger

logger = get_logger("mdk_oracle.backtest.loader")


class BacktestLoader:
    """Standardized loader extracting backtest and performance datasets directly from DuckDB."""

    def __init__(self, db: Optional[DuckDBManager] = None):
        """Initialize loader with a read-only DuckDB connection."""
        self.db = db or DuckDBManager(read_only=True)

    def load_day_start(
        self,
        use_performance_ledger: bool = False,
        start_date: Optional[Union[date, str]] = None,
        end_date: Optional[Union[date, str]] = None,
    ) -> pd.DataFrame:
        """Load Model 1: Macro Day-Start opening flow backtest/performance dataset.

        Args:
            use_performance_ledger: If True, reads from `gold_bofa_day_start_performance`,
                                   otherwise reads from `gold_bofa_day_start_backtests`.
            start_date: Optional start date filter.
            end_date: Optional end date filter.

        Returns:
            pd.DataFrame containing historical predictions, actuals, and evaluation tags.
        """
        table_name = (
            "gold_bofa_day_start_performance" if use_performance_ledger else "gold_bofa_day_start_backtests"
        )
        query = f"SELECT * FROM {table_name}"
        conditions = []
        if start_date:
            conditions.append(f"trade_date >= '{start_date}'")
        if end_date:
            conditions.append(f"trade_date <= '{end_date}'")

        if conditions:
            query += " WHERE " + " AND ".join(conditions)

        query += " ORDER BY trade_date ASC"

        conn = self.db.get_connection()
        df = conn.execute(query).df()
        logger.info(f"Loaded {len(df)} records from `{table_name}`.")
        return df

    def load_sector_day_start(
        self,
        sectors: Optional[List[str]] = None,
        use_performance_ledger: bool = False,
        start_date: Optional[Union[date, str]] = None,
        end_date: Optional[Union[date, str]] = None,
    ) -> pd.DataFrame:
        """Load Model 2: Sector Day-Start opening flow backtest/performance dataset.

        Args:
            sectors: Optional list of sectors to filter by (e.g. ['XBANK', 'XUSIN']).
            use_performance_ledger: If True, reads from `gold_bofa_sector_day_start_performance`.
            start_date: Optional start date filter.
            end_date: Optional end date filter.

        Returns:
            pd.DataFrame containing sector predictions, actuals, and evaluation tags.
        """
        table_name = (
            "gold_bofa_sector_day_start_performance"
            if use_performance_ledger
            else "gold_bofa_sector_day_start_backtests"
        )
        query = f"SELECT * FROM {table_name}"
        conditions = []
        if sectors:
            sector_list_sql = ", ".join([f"'{s}'" for s in sectors])
            conditions.append(f"sector IN ({sector_list_sql})")
        if start_date:
            conditions.append(f"trade_date >= '{start_date}'")
        if end_date:
            conditions.append(f"trade_date <= '{end_date}'")

        if conditions:
            query += " WHERE " + " AND ".join(conditions)

        query += " ORDER BY trade_date ASC, sector ASC"

        conn = self.db.get_connection()
        df = conn.execute(query).df()
        logger.info(f"Loaded {len(df)} records from `{table_name}`.")
        return df

    def load_stock_reaction(
        self,
        windows: Optional[List[str]] = None,
        symbols: Optional[List[str]] = None,
        use_performance_ledger: bool = False,
        start_date: Optional[Union[date, str]] = None,
        end_date: Optional[Union[date, str]] = None,
    ) -> pd.DataFrame:
        """Load Model 3: Stock Intraday Reaction percentage return backtest/performance dataset.

        Args:
            windows: Reaction windows to load (defaults to ['w2', 'w3', 'w5']).
            symbols: Optional list of equity symbols to filter by.
            use_performance_ledger: If True, reads from `gold_bofa_stock_reaction_<w>_performance`.
            start_date: Optional start date filter.
            end_date: Optional end date filter.

        Returns:
            pd.DataFrame combining reaction windows for individual stocks.
        """
        target_windows = [w.lower() for w in (windows or ["w2", "w3", "w5"])]
        suffix = "performance" if use_performance_ledger else "backtests"

        conn = self.db.get_connection()
        frames = []
        for w in target_windows:
            table_name = f"gold_bofa_stock_reaction_{w}_{suffix}"
            try:
                query = f"SELECT * FROM {table_name}"
                conditions = []
                if symbols:
                    sym_list_sql = ", ".join([f"'{s}'" for s in symbols])
                    conditions.append(f"symbol IN ({sym_list_sql})")
                if start_date:
                    conditions.append(f"trade_date >= '{start_date}'")
                if end_date:
                    conditions.append(f"trade_date <= '{end_date}'")

                if conditions:
                    query += " WHERE " + " AND ".join(conditions)

                query += " ORDER BY trade_date ASC, symbol ASC"
                w_df = conn.execute(query).df()
                frames.append(w_df)
            except Exception as e:
                logger.warning(f"Could not load table `{table_name}`: {e}")

        if not frames:
            return pd.DataFrame()

        df = pd.concat(frames, ignore_index=True)
        logger.info(f"Loaded {len(df)} stock reaction records across windows: {target_windows}.")
        return df

    def summarize_day_start(
        self,
        use_performance_ledger: bool = False,
        start_date: Optional[Union[date, str]] = None,
        end_date: Optional[Union[date, str]] = None,
    ) -> BacktestSummary:
        """Load Model 1 data and compute full statistical backtest summary."""
        df = self.load_day_start(
            use_performance_ledger=use_performance_ledger,
            start_date=start_date,
            end_date=end_date,
        )
        return BacktestMetricsCalculator.calculate_summary(
            df=df,
            model_name="Model 1: Macro Day-Start Forecaster",
            target_name="Window 1 Opening Net Flow (TL)",
            actual_col="actual_open_net_flow_tl",
            predicted_col="predicted_open_net_flow_tl",
            lower_90_col="predicted_open_flow_lower_90",
            upper_90_col="predicted_open_flow_upper_90",
            direction_col="predicted_direction",
            is_hit_col="is_direction_hit",
            is_inside_ci_col="is_inside_90_ci",
            target_unit=TargetUnit.TL,
            slice_columns=["day_of_week", "is_monday", "predicted_playbook"],
        )

    def summarize_sector_day_start(
        self,
        sectors: Optional[List[str]] = None,
        use_performance_ledger: bool = False,
        start_date: Optional[Union[date, str]] = None,
        end_date: Optional[Union[date, str]] = None,
    ) -> BacktestSummary:
        """Load Model 2 data and compute full cross-sector backtest summary."""
        df = self.load_sector_day_start(
            sectors=sectors,
            use_performance_ledger=use_performance_ledger,
            start_date=start_date,
            end_date=end_date,
        )
        return BacktestMetricsCalculator.calculate_summary(
            df=df,
            model_name="Model 2: Sector Day-Start Forecaster",
            target_name="Sector Opening Net Flow (TL)",
            actual_col="actual_open_net_flow_tl",
            predicted_col="predicted_open_net_flow_tl",
            lower_90_col="predicted_open_flow_lower_90",
            upper_90_col="predicted_open_flow_upper_90",
            direction_col="predicted_direction",
            is_hit_col="is_direction_hit",
            is_inside_ci_col="is_inside_90_ci",
            target_unit=TargetUnit.TL,
            slice_columns=["sector", "predicted_playbook", "day_of_week"],
        )

    def summarize_stock_reaction(
        self,
        windows: Optional[List[str]] = None,
        symbols: Optional[List[str]] = None,
        use_performance_ledger: bool = False,
        start_date: Optional[Union[date, str]] = None,
        end_date: Optional[Union[date, str]] = None,
    ) -> BacktestSummary:
        """Load Model 3 data and compute full stock return reaction backtest summary."""
        df = self.load_stock_reaction(
            windows=windows,
            symbols=symbols,
            use_performance_ledger=use_performance_ledger,
            start_date=start_date,
            end_date=end_date,
        )
        return BacktestMetricsCalculator.calculate_summary(
            df=df,
            model_name="Model 3: Stock Intraday Reaction Forecaster",
            target_name="Intraday Stock Reaction Return (%)",
            actual_col="actual_return_pct",
            predicted_col="predicted_return_pct",
            lower_90_col="predicted_return_lower_90",
            upper_90_col="predicted_return_upper_90",
            direction_col="predicted_direction",
            is_hit_col="is_direction_hit",
            is_inside_ci_col="is_inside_90_ci",
            target_unit=TargetUnit.PERCENTAGE,
            slice_columns=["window_name", "symbol", "predicted_playbook"],
        )
