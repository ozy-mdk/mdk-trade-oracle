"""Signal rules and heuristics for institutional decision support."""

import json
import uuid
from datetime import date
from typing import Any, Dict, List
from mdk_trading_oracle.core.types import OracleDecisionSignal, SignalType


class RuleEngine:
    """Evaluates rule-based signals based on BofA flow heuristics."""

    @staticmethod
    def evaluate_row(row: Dict[str, Any]) -> OracleDecisionSignal:
        """Evaluate flow metrics dictionary for a single day/symbol."""
        symbol = row.get("symbol", "UNKNOWN")
        date_val = row.get("date_val")
        if isinstance(date_val, str):
            date_val = date.fromisoformat(date_val)

        net_tl = float(row.get("bofa_net_tl", 0.0))
        net_share = float(row.get("bofa_net_share", 0.0))
        vol_share = float(row.get("bofa_volume_share", 0.0))
        zscore = float(row.get("bofa_flow_zscore_20d", 0.0))
        accel = float(row.get("bofa_flow_acceleration_5d", 0.0))
        roll_3d = float(row.get("bofa_net_tl_roll_3d", 0.0))

        reasons: List[str] = []
        signal = SignalType.NEUTRAL
        confidence = 0.50

        # Heuristic 1: Aggressive Institutional Accumulation
        if net_share > 0.15 and zscore > 1.2:
            signal = SignalType.STRONG_BUY
            confidence = min(0.95, 0.70 + (zscore * 0.10) + (net_share * 0.5))
            reasons.append(f"Heavy BofA net buying ({net_tl:,.0f} TL), representing {net_share:.1%} of market volume.")
            reasons.append(f"Order flow Z-score is strong positive at +{zscore:.2f} standard deviations.")
            if accel > 0:
                reasons.append(f"Positive 5-day flow acceleration (+{accel:,.0f} TL).")

        # Heuristic 2: Moderate Accumulation
        elif net_share > 0.05 and net_tl > 0:
            signal = SignalType.BUY
            confidence = min(0.80, 0.60 + (net_share * 0.4))
            reasons.append(f"Net positive BofA inflow of {net_tl:,.0f} TL ({net_share:.1%} net share).")
            if roll_3d > 0:
                reasons.append(f"Sustained 3-day average net buy flow of {roll_3d:,.0f} TL.")

        # Heuristic 3: Aggressive Institutional Distribution / Dumping
        elif net_share < -0.15 and zscore < -1.2:
            signal = SignalType.STRONG_SELL
            confidence = min(0.95, 0.70 + (abs(zscore) * 0.10) + (abs(net_share) * 0.5))
            reasons.append(f"Heavy BofA net selling ({net_tl:,.0f} TL), representing {abs(net_share):.1%} net market dump.")
            reasons.append(f"Order flow Z-score is deeply negative at {zscore:.2f} standard deviations.")
            if accel < 0:
                reasons.append(f"Negative 5-day flow deceleration ({accel:,.0f} TL).")

        # Heuristic 4: Moderate Distribution
        elif net_share < -0.05 and net_tl < 0:
            signal = SignalType.SELL
            confidence = min(0.80, 0.60 + (abs(net_share) * 0.4))
            reasons.append(f"Net negative BofA outflow of {net_tl:,.0f} TL ({abs(net_share):.1%} net sell share).")
            if roll_3d < 0:
                reasons.append(f"Sustained 3-day average net sell flow of {roll_3d:,.0f} TL.")

        else:
            signal = SignalType.NEUTRAL
            confidence = 0.50
            reasons.append("BofA activity is within normal market equilibrium parameters.")
            reasons.append(f"Volume share at {vol_share:.1%}, with neutral net share of {net_share:.1%}.")

        summary = f"{signal.value} signal on {symbol} (Confidence: {confidence:.0%})"

        return OracleDecisionSignal(
            signal_id=str(uuid.uuid4()),
            date_val=date_val,
            symbol=symbol,
            signal=signal,
            confidence=round(confidence, 3),
            bofa_net_tl=net_tl,
            bofa_net_share=round(net_share, 4),
            bofa_flow_zscore=round(zscore, 2),
            summary=summary,
            reasons=reasons,
        )
