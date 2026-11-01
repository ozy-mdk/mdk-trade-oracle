"""Shadow-mode LLM headline classifier. Labels are stored for comparison and never reach a forecast.

Only the headline text is sent: no dates, prices or outcomes.
"""

from __future__ import annotations

import json
import time
import uuid
from typing import Any

import pandas as pd
from pydantic import BaseModel, Field, ValidationError

from mdk_trading_oracle.agents.providers import (
    CliProvider,
    ProviderError,
    get_provider,
    load_agents_config,
    read_prompt,
)
from mdk_trading_oracle.core.logger import get_logger
from mdk_trading_oracle.data.bronze.news import ensure_news_tables, now_trt

logger = get_logger(__name__)

TASK = "news_classifier"
INTENSITY_FIELDS = ("fon_krizi", "bofa_sell", "kamu_floor", "bullish_catalyst", "geo_escalation", "geo_peace")


class HeadlineLabel(BaseModel):
    id: int
    fon_krizi: float = Field(ge=0.0, le=1.0)
    bofa_sell: float = Field(ge=0.0, le=1.0)
    kamu_floor: float = Field(ge=0.0, le=1.0)
    bullish_catalyst: float = Field(ge=0.0, le=1.0)
    geo_escalation: float = Field(ge=0.0, le=1.0)
    geo_peace: float = Field(ge=0.0, le=1.0)
    is_market_wide: bool
    stocks: list[str]


LABEL_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "items": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "id": {"type": "integer"},
                    **{f: {"type": "number", "minimum": 0, "maximum": 1} for f in INTENSITY_FIELDS},
                    "is_market_wide": {"type": "boolean"},
                    "stocks": {"type": "array", "items": {"type": "string"}},
                },
                "required": ["id", *INTENSITY_FIELDS, "is_market_wide", "stocks"],
                "additionalProperties": False,
            },
        }
    },
    "required": ["items"],
    "additionalProperties": False,
}


def build_user_prompt(titles: list[str]) -> str:
    lines = [f"{i}: {' '.join(t.split())}" for i, t in enumerate(titles)]
    return "Headlines:\n" + "\n".join(lines)


def validate_labels(data: dict[str, Any], expected: int) -> list[HeadlineLabel]:
    """Reject a whole batch unless it contains exactly one valid result for every headline."""
    try:
        labels = [HeadlineLabel(**item) for item in data.get("items", [])]
    except (TypeError, ValidationError) as e:
        raise ProviderError(f"Labels failed validation: {e}") from e
    if sorted(x.id for x in labels) != list(range(expected)):
        raise ProviderError(f"Expected one label per id 0..{expected - 1}, got {sorted(x.id for x in labels)}")
    return labels


def _log_run(db: Any, provider: str, model: str, version: str, n: int, usage: dict[str, Any], duration: float, status: str, error: str | None, started: Any) -> None:
    db.execute(
        """
        INSERT INTO agent_llm_runs (run_id, task, provider, model, prompt_version, n_items, input_tokens, output_tokens,
            cache_read_tokens, cache_write_tokens, notional_cost_usd, duration_seconds, status, error, started_at_trt)
        VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
        """,
        [
            uuid.uuid4().hex, TASK, provider, model, version, n, usage.get("input_tokens"), usage.get("output_tokens"),
            usage.get("cache_read_tokens"), usage.get("cache_write_tokens"), usage.get("notional_cost_usd"), duration,
            status, error, started,
        ],
    )


def select_unlabeled(db: Any, prompt_version: str, provider: str, model: str, limit: int, live_only: bool = False) -> pd.DataFrame:
    live_clause = "AND h.collection_mode = 'LIVE'" if live_only else ""
    return db.query_df(
        f"""
        SELECT h.headline_id, h.title FROM bronze_news_headlines h
        WHERE NOT EXISTS (
            SELECT 1 FROM silver_news_llm_labels l
            WHERE l.headline_id = h.headline_id AND l.prompt_version = %s AND l.provider = %s AND l.model = %s
        ) {live_clause}
        ORDER BY h.published_at_trt DESC, h.headline_id
        LIMIT %s
        """,
        [prompt_version, provider, model, limit],
    )


def label_batch(db: Any, batch: pd.DataFrame, provider: CliProvider, model: str, task: dict[str, Any]) -> int:
    version = task["prompt_version"]
    titles = batch["title"].tolist()
    schema_json = json.dumps(LABEL_SCHEMA, separators=(",", ":"))
    system_prompt = read_prompt(task["prompt_file"]).replace("{schema_json}", schema_json)
    started, t0 = now_trt(), time.monotonic()
    try:
        result = provider.complete_json(system_prompt, build_user_prompt(titles), LABEL_SCHEMA, model, int(task["timeout_seconds"]))
        labels = validate_labels(result.data, len(titles))
    except ProviderError as e:
        _log_run(db, provider.name, model, version, len(titles), {}, time.monotonic() - t0, "ERROR", str(e)[:1000], started)
        raise
    ids = batch["headline_id"].tolist()
    labeled_at = now_trt()
    for label in labels:
        db.execute(
            """
            INSERT INTO silver_news_llm_labels (headline_id, prompt_version, provider, model, fon_krizi, bofa_sell,
                kamu_floor, bullish_catalyst, geo_escalation, geo_peace, is_market_wide, stocks, labeled_at_trt)
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
            ON CONFLICT DO NOTHING
            """,
            [
                ids[label.id], version, provider.name, model, label.fon_krizi, label.bofa_sell, label.kamu_floor,
                label.bullish_catalyst, label.geo_escalation, label.geo_peace, label.is_market_wide,
                ",".join(s.strip().upper() for s in label.stocks), labeled_at,
            ],
        )
    _log_run(db, provider.name, model, version, len(titles), result.usage, result.duration_seconds, "OK", None, started)
    return len(labels)


def run_shadow_labeling(db: Any, limit: int | None = None, provider_name: str | None = None, live_only: bool = False) -> dict[str, Any]:
    """Label unlabeled headlines up to the configured daily cap. Stops at the first provider failure."""
    ensure_news_tables(db)
    cfg = load_agents_config()
    task = cfg["tasks"][TASK]
    provider = get_provider(provider_name)
    model = provider.model_for_tier(task["model_tier"])
    cap = int(task["max_headlines_per_run"])
    pending = select_unlabeled(db, task["prompt_version"], provider.name, model, min(limit or cap, cap), live_only)
    size = int(task["batch_size"])
    labeled = 0
    for start in range(0, len(pending), size):
        labeled += label_batch(db, pending.iloc[start:start + size], provider, model, task)
        logger.info(f"Labeled {labeled}/{len(pending)} headlines with {provider.name}/{model}")
    return {"provider": provider.name, "model": model, "prompt_version": task["prompt_version"], "labeled": labeled}
