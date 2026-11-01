"""Three-Forces conditioning rules for swing forecasts (5d, 10d, 15d).

Force 1 is MLB, Force 2 is KAMU, Force 3 is BIG5. Rationale, test results and the choice of default mode are recorded
in docs/AGENTIC_DECISIONS.md.
"""

from __future__ import annotations

from functools import lru_cache
from typing import Any

import numpy as np

from mdk_trading_oracle.core.config import get_settings

GATED_HORIZONS = ("5d", "10d", "15d")
GATE_MODES = ("off", "flow_only", "flow_news")


@lru_cache(maxsize=1)
def get_gate_mode() -> str:
    mode = str(get_settings().load_yaml("agents.yaml").get("gate", {}).get("mode", "flow_only"))
    if mode not in GATE_MODES:
        raise ValueError(f"gate.mode must be one of {GATE_MODES}, got {mode!r}")
    return mode


def gate_direction(row: dict[str, Any], mode: str) -> str | None:
    """Return BEARISH, BULLISH or None for one stock-session under `mode`."""
    if mode == "off":
        return None
    use_news = mode == "flow_news"
    news_available = use_news and bool(row.get("news_available", True))
    fk_ewm = float(row.get("fon_krizi_ewm5", 0.0)) if news_available else 0.0
    bofa_ewm = float(row.get("bofa_sell_ewm5", 0.0)) if news_available else 0.0

    trap = bool(row.get("is_passive_absorption_trap", False))
    mlb_5d = float(row.get("mlb_5d_share_pct", 0.0))
    mlb_10d = float(row.get("mlb_10d_share_pct", 0.0))
    kamu_5d = float(row.get("kamu_5d_share_pct", 0.0))
    big5_5d = float(row.get("big5_5d_share_pct", 0.0))
    big5_cost = float(row.get("big5_cost_spread_pct", 0.0))

    if (
        trap
        and mlb_5d < -3.5
        and mlb_10d < -2.0
        and big5_cost > 5.0
        and (fk_ewm >= 0.08 or bofa_ewm >= 0.12 or mlb_5d < -5.0)
    ):
        return "BEARISH"
    # Missing news must not read as "no crisis" when the rule is meant to check for one.
    if use_news and not news_available:
        return None
    if (not trap) and mlb_5d > 1.0 and kamu_5d > 0.5 and big5_5d > 0.5 and big5_cost < 2.0 and fk_ewm < 0.08:
        return "BULLISH"
    return None


def apply_three_forces_gate(sig_dir: Any, row: dict[str, Any], horizon: str, mode: str | None = None) -> Any:
    """Override the Stage-1 directional signal when the gate fires; otherwise return it unchanged."""
    if horizon not in GATED_HORIZONS:
        return sig_dir
    mode = get_gate_mode() if mode is None else mode
    direction = gate_direction(row, mode)
    if direction is None:
        return sig_dir
    news_available = mode == "flow_news" and bool(row.get("news_available", True))
    fk_ewm = float(row.get("fon_krizi_ewm5", 0.0)) if news_available else 0.0
    bull_ewm = float(row.get("bullish_catalyst_ewm5", 0.0)) if news_available else 0.0
    clash = float(row.get("three_forces_clash_score", 0.0))
    if direction == "BEARISH":
        return np.minimum(-0.55, 0.30 * sig_dir + 0.70 * clash - 0.50 * fk_ewm)
    return np.maximum(0.55, 0.35 * sig_dir + 0.65 * max(0.55, clash) + 0.35 * bull_ewm)
