"""Model 3: BIST30 Stock Intraday Reaction Forecaster.

Forecasts intraday stock returns for BIST30 equities across three reaction windows:
  - W2 (first_reaction):   10:30-11:30 TRT  — momentum continuation / reversal after open
  - W3 (midday_followup):  11:30-14:30 TRT  — mid-session trend
  - W5 (closing_session):  16:00-18:15 TRT  — end-of-day resolution

Input signals: W1 (day_start, 09:55-10:30 TRT) BofA + multi-broker execution patterns.
Target: stock adjusted return % from W1 reference price to window-end VWAP.
"""

from mdk_trading_oracle.models.stock_reaction.forecaster import StockReactionForecaster
from mdk_trading_oracle.models.stock_reaction.orchestrator import StockReactionOrchestrator

__all__ = ["StockReactionForecaster", "StockReactionOrchestrator"]
