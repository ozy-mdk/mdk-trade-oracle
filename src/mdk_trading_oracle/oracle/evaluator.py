"""Oracle evaluation and decision generator."""

import json
from typing import Dict, List, Optional
import polars as pl
from mdk_trading_oracle.core.db import DuckDBManager
from mdk_trading_oracle.core.logger import get_logger
from mdk_trading_oracle.core.types import OracleDecisionSignal
from mdk_trading_oracle.oracle.signals import RuleEngine

logger = get_logger("mdk_oracle.evaluator")


class OracleEvaluator:
    """Evaluates Gold metrics to produce actionable trading decision signals."""

    def __init__(self, db_manager: DuckDBManager):
        self.db = db_manager

    def evaluate_latest_signals(self) -> List[OracleDecisionSignal]:
        """Evaluate signals for all symbols on their latest available trading date."""
        conn = self.db.get_connection()
        logger.info("Evaluating Oracle signals from Gold layer...")

        query = """
            WITH latest_records AS (
                SELECT *,
                       ROW_NUMBER() OVER (PARTITION BY symbol ORDER BY date_val DESC) as rn
                FROM gold_bofa_flow_metrics
            )
            SELECT *
            FROM latest_records
            WHERE rn = 1
            ORDER BY bofa_net_tl DESC;
        """
        df = self.db.query_pl(query)
        if df.is_empty():
            logger.warning("No records found in Gold layer to evaluate.")
            return []

        signals: List[OracleDecisionSignal] = []
        for row in df.iter_rows(named=True):
            sig = RuleEngine.evaluate_row(row)
            signals.append(sig)

            # Persist into DuckDB table
            conn.execute("""
                INSERT OR REPLACE INTO oracle_decision_signals (
                    signal_id, date_val, symbol, signal, confidence,
                    bofa_net_tl, bofa_net_share, bofa_flow_zscore, summary, reasons
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?);
            """, [
                sig.signal_id,
                sig.date_val,
                sig.symbol,
                sig.signal.value,
                sig.confidence,
                sig.bofa_net_tl,
                sig.bofa_net_share,
                sig.bofa_flow_zscore,
                sig.summary,
                json.dumps(sig.reasons),
            ])

        logger.info(f"Oracle generated {len(signals)} signals.")
        return signals

    def get_symbol_signal(self, symbol: str) -> Optional[OracleDecisionSignal]:
        """Evaluate or fetch the latest signal for a specific symbol."""
        query = """
            SELECT *
            FROM gold_bofa_flow_metrics
            WHERE symbol = ?
            ORDER BY date_val DESC
            LIMIT 1;
        """
        df = self.db.query_pl(query, [symbol])
        if df.is_empty():
            return None
        row = df.iter_rows(named=True).__next__()
        return RuleEngine.evaluate_row(row)
