#!/usr/bin/env python
"""Compare LLM shadow labels with the keyword scorer, and report how far the live-collection requirement has come."""

from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np

from mdk_trading_oracle.agents.providers import load_agents_config
from mdk_trading_oracle.core.db import PostgresManager

_spec = importlib.util.spec_from_file_location("scorer", Path(__file__).with_name("score_agentic_news_regimes.py"))
scorer = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(scorer)

PAIRS = {
    "fon_krizi": "fon_krizi_news_intensity",
    "bofa_sell": "bofa_sell_news_intensity",
    "kamu_floor": "kamu_floor_news_intensity",
    "bullish_catalyst": "bullish_catalyst_intensity",
    "geo_escalation": "geo_escalation_intensity",
    "geo_peace": "geo_peace_intensity",
}
ACTIVE = 0.5


def main() -> None:
    db = PostgresManager()
    df = db.query_df(
        """
        SELECT l.*, h.title, h.category, h.collection_mode
        FROM silver_news_llm_labels l JOIN bronze_news_headlines h USING (headline_id)
        """
    )
    live = db.query_df(
        "SELECT count(*) AS n, count(DISTINCT published_at_trt::date) AS days FROM bronze_news_headlines WHERE collection_mode = 'LIVE'"
    ).iloc[0]
    required = load_agents_config()["news"]["min_live_sessions_for_review"]
    print(f"LIVE headlines: {live['n']} across {live['days']} days; forecast review requires {required} live sessions")
    if df.empty:
        print("No labels yet.")
        return
    print(f"Labels: {len(df)}; model(s): {sorted(df['model'].unique())}; prompt(s): {sorted(df['prompt_version'].unique())}")
    scores = [scorer.score_single_headline(t, c) for t, c in zip(df["title"], df["category"])]
    print(f"{'dimension':<18}{'agree':>8}{'llm>=0.5':>10}{'kw>=0.5':>9}{'mean |diff|':>13}")
    for llm_col, kw_col in PAIRS.items():
        llm = df[llm_col].to_numpy(float)
        kw = np.array([s[kw_col] for s in scores], dtype=float)
        agree = np.mean((llm >= ACTIVE) == (kw >= ACTIVE)) * 100.0
        print(f"{llm_col:<18}{agree:7.1f}%{int((llm >= ACTIVE).sum()):>10}{int((kw >= ACTIVE).sum()):>9}{np.abs(llm - kw).mean():>13.3f}")


if __name__ == "__main__":
    main()
