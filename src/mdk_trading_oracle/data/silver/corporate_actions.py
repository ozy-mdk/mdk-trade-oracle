"""Corporate Actions & Share Adjustment Period Engine (Pay Düzeltme).

Computes continuous, non-overlapping historical adjustment periods with cumulative
quantity factors (splits/bonus shares), canonical symbol graph resolution,
and unresolved rights issue flags.
"""

from __future__ import annotations

import datetime as dt
from dataclasses import dataclass
from decimal import Decimal
from typing import Any, Dict, List, Optional, Set, Tuple

import polars as pl

from mdk_trading_oracle.core.db import DuckDBManager
from mdk_trading_oracle.core.logger import get_logger
from mdk_trading_oracle.data.silver.schema import initialize_silver_schema

logger = get_logger("mdk_oracle.data.silver.corporate_actions")

MIN_DATE = dt.date(1900, 1, 1)
MAX_DATE = dt.date(9999, 12, 31)
ONE = Decimal("1")


@dataclass(frozen=True)
class Action:
    """Discrete corporate action event."""

    action_date: dt.date
    symbol: str
    target_symbol: Optional[str]
    multiplier: Decimal
    note: str


@dataclass(frozen=True)
class Period:
    """Continuous point-in-time share adjustment period."""

    source_symbol: str
    effective_from: dt.date
    effective_to: dt.date
    canonical_symbol: str
    quantity_factor: Decimal
    has_unresolved_paid_action: bool


class CorporateActionEngine:
    """Engine that resolves chained corporate actions into continuous point-in-time adjustment periods."""

    def __init__(self, db: Optional[DuckDBManager] = None):
        self.db = db or DuckDBManager()

    @staticmethod
    def resolve_adjustment_chain(
        source_symbol: str,
        by_symbol: Dict[str, List[Action]],
    ) -> Tuple[str, List[Action]]:
        """Traverse ticker rename chains and collect all subsequent corporate action events."""
        current = source_symbol
        floor_date = MIN_DATE
        events: List[Action] = []
        visited: Set[str] = set()

        while True:
            if current in visited:
                raise ValueError(f"Ticker symbol rename cycle detected for: {source_symbol}")
            visited.add(current)

            current_events = sorted(
                (row for row in by_symbol.get(current, []) if row.action_date >= floor_date),
                key=lambda row: (row.action_date, row.symbol),
            )
            events.extend(current_events)

            renames = [row for row in current_events if row.target_symbol]
            if not renames:
                return current, sorted(events, key=lambda row: (row.action_date, row.symbol))

            if len(renames) != 1:
                raise ValueError(f"Multiple symbol renames found for symbol on overlapping dates: {current}")

            rename = renames[0]
            current = rename.target_symbol
            floor_date = rename.action_date

    @classmethod
    def build_adjustment_periods(cls, actions: List[Action]) -> List[Period]:
        """Build continuous disjoint adjustment periods from discrete corporate actions.

        Args:
            actions: List of validated corporate action events.

        Returns:
            List[Period]: Continuous periods from MIN_DATE (1900-01-01) to MAX_DATE (9999-12-31).
        """
        by_symbol: Dict[str, List[Action]] = {}
        symbols: Set[str] = set()

        for row in actions:
            by_symbol.setdefault(row.symbol, []).append(row)
            symbols.add(row.symbol)
            if row.target_symbol:
                symbols.add(row.target_symbol)

        result: List[Period] = []
        for source_symbol in sorted(symbols):
            canonical_symbol, events = cls.resolve_adjustment_chain(source_symbol, by_symbol)
            breakpoints = sorted({row.action_date for row in events})
            starts = [MIN_DATE, *breakpoints]

            for index, effective_from in enumerate(starts):
                effective_to = breakpoints[index] - dt.timedelta(days=1) if index < len(breakpoints) else MAX_DATE
                future = [row for row in events if row.action_date > effective_from]
                factor = ONE
                for row in future:
                    factor *= row.multiplier

                result.append(
                    Period(
                        source_symbol=source_symbol,
                        effective_from=effective_from,
                        effective_to=effective_to,
                        canonical_symbol=canonical_symbol,
                        quantity_factor=factor,
                        has_unresolved_paid_action=any("bedelli" in row.note.casefold() for row in future),
                    )
                )

        return result

    def execute_and_persist(self) -> Dict[str, Any]:
        """Read `bronze_corporate_actions`, compute adjustment periods, and persist to Silver."""
        initialize_silver_schema(self.db)
        conn = self.db.get_connection()

        # Check if bronze table exists and has rows
        has_bronze = conn.execute("""
            SELECT COUNT(*) FROM information_schema.tables WHERE table_name = 'bronze_corporate_actions';
        """).fetchone()[0]

        if has_bronze == 0 or conn.execute("SELECT COUNT(*) FROM bronze_corporate_actions;").fetchone()[0] == 0:
            logger.info("No corporate actions found in Bronze. Attempting auto-ingestion...")
            from mdk_trading_oracle.data.bronze.ingestor import BronzeIngestor

            ingestor = BronzeIngestor(self.db)
            ingestor.ingest_corporate_actions()

        # Fetch actions from Bronze
        raw_rows = conn.execute("""
            SELECT action_date, symbol, target_symbol, quantity_multiplier, note
            FROM bronze_corporate_actions
            ORDER BY action_date ASC, symbol ASC;
        """).fetchall()

        if not raw_rows:
            logger.warning("No corporate actions available to process.")
            return {"adjustment_periods_count": 0, "status": "empty"}

        actions = [
            Action(
                action_date=r[0],
                symbol=r[1].strip().upper(),
                target_symbol=r[2].strip().upper() if r[2] else None,
                multiplier=Decimal(str(r[3])),
                note=r[4] or "",
            )
            for r in raw_rows
        ]

        periods = self.build_adjustment_periods(actions)

        # Convert to Polars DataFrame for DuckDB registration
        period_rows = [
            {
                "source_symbol": p.source_symbol,
                "effective_from": p.effective_from,
                "effective_to": p.effective_to,
                "canonical_symbol": p.canonical_symbol,
                "quantity_factor": float(p.quantity_factor),
                "has_unresolved_paid_action": p.has_unresolved_paid_action,
            }
            for p in periods
        ]

        df_periods = pl.DataFrame(period_rows)
        conn.register("df_periods_temp", df_periods)
        conn.execute("""
            CREATE OR REPLACE TABLE silver_corporate_action_adjustment_periods AS
            SELECT 
                source_symbol,
                effective_from,
                effective_to,
                canonical_symbol,
                quantity_factor,
                has_unresolved_paid_action,
                CURRENT_TIMESTAMP AS calculated_at
            FROM df_periods_temp
            ORDER BY source_symbol ASC, effective_from ASC;
        """)
        conn.unregister("df_periods_temp")

        logger.info(
            f"Successfully built `silver_corporate_action_adjustment_periods`: "
            f"{len(periods)} periods across {len({p.source_symbol for p in periods})} symbols."
        )

        return {
            "adjustment_periods_count": len(periods),
            "source_symbols_count": len({p.source_symbol for p in periods}),
            "status": "success",
        }
