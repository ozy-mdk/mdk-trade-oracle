"""Append-only store for point-in-time news headlines, LLM shadow labels and agent run logs.

A headline's publication time and precision are frozen when first seen, so a later harvest can never move it to a
different session. Later harvests update only `last_seen_at_trt`. Decisions: docs/AGENTIC_DECISIONS.md.
"""

from __future__ import annotations

import hashlib
from datetime import datetime, time
from typing import Any, Iterable, Sequence
from zoneinfo import ZoneInfo

import pandas as pd

TRT = ZoneInfo("Europe/Istanbul")
SESSION_CLOSE_TIME = time(18, 10)
LIVE_COLLECTION_WINDOW_HOURS = 72
PRECISION_EXACT = "EXACT_SECOND_VERIFIED"
PRECISION_DATE_ONLY = "DATE_ONLY_SHIFTED_T+1"

NEWS_DDL = """
CREATE TABLE IF NOT EXISTS bronze_news_headlines (
    headline_id VARCHAR(32) PRIMARY KEY,
    scope VARCHAR(32) NOT NULL,
    category VARCHAR(32) NOT NULL,
    title TEXT NOT NULL,
    link TEXT,
    published_at_trt TIMESTAMP NOT NULL,
    timestamp_precision VARCHAR(32) NOT NULL,
    first_seen_at_trt TIMESTAMP NOT NULL,
    last_seen_at_trt TIMESTAMP NOT NULL,
    collection_mode VARCHAR(16) NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_bronze_news_published ON bronze_news_headlines (published_at_trt);

CREATE TABLE IF NOT EXISTS silver_news_regimes_daily (
    symbol VARCHAR(16) NOT NULL,
    trade_date DATE NOT NULL,
    agentic_news_regime VARCHAR(48),
    is_passive_absorption_trap BOOLEAN,
    three_forces_clash_score DOUBLE PRECISION,
    mlb_5d_share_pct DOUBLE PRECISION,
    mlb_10d_share_pct DOUBLE PRECISION,
    kamu_5d_share_pct DOUBLE PRECISION,
    big5_5d_share_pct DOUBLE PRECISION,
    mlb_cost_spread_pct DOUBLE PRECISION,
    big5_cost_spread_pct DOUBLE PRECISION,
    fon_krizi_ewm5 DOUBLE PRECISION,
    bofa_sell_ewm5 DOUBLE PRECISION,
    kamu_floor_ewm5 DOUBLE PRECISION,
    bullish_catalyst_ewm5 DOUBLE PRECISION,
    news_article_count INTEGER,
    PRIMARY KEY (symbol, trade_date)
);

CREATE TABLE IF NOT EXISTS silver_news_llm_labels (
    headline_id VARCHAR(32) NOT NULL,
    prompt_version VARCHAR(48) NOT NULL,
    provider VARCHAR(32) NOT NULL,
    model VARCHAR(64) NOT NULL,
    fon_krizi DOUBLE PRECISION NOT NULL,
    bofa_sell DOUBLE PRECISION NOT NULL,
    kamu_floor DOUBLE PRECISION NOT NULL,
    bullish_catalyst DOUBLE PRECISION NOT NULL,
    geo_escalation DOUBLE PRECISION NOT NULL,
    geo_peace DOUBLE PRECISION NOT NULL,
    is_market_wide BOOLEAN NOT NULL,
    stocks TEXT NOT NULL,
    labeled_at_trt TIMESTAMP NOT NULL,
    PRIMARY KEY (headline_id, prompt_version, provider, model)
);

CREATE TABLE IF NOT EXISTS agent_llm_runs (
    run_id VARCHAR(32) PRIMARY KEY,
    task VARCHAR(48) NOT NULL,
    provider VARCHAR(32) NOT NULL,
    model VARCHAR(64) NOT NULL,
    prompt_version VARCHAR(48) NOT NULL,
    n_items INTEGER NOT NULL,
    input_tokens BIGINT,
    output_tokens BIGINT,
    cache_read_tokens BIGINT,
    cache_write_tokens BIGINT,
    notional_cost_usd DOUBLE PRECISION,
    duration_seconds DOUBLE PRECISION,
    status VARCHAR(16) NOT NULL,
    error TEXT,
    started_at_trt TIMESTAMP NOT NULL
);

CREATE TABLE IF NOT EXISTS gold_agentic_gate_audit (
    symbol VARCHAR(16) NOT NULL,
    horizon VARCHAR(8) NOT NULL,
    trade_date DATE NOT NULL,
    actual_return_pct DOUBLE PRECISION NOT NULL,
    pred_off_pct DOUBLE PRECISION NOT NULL,
    pred_flow_pct DOUBLE PRECISION NOT NULL,
    pred_news_pct DOUBLE PRECISION NOT NULL,
    hit_off BOOLEAN NOT NULL,
    hit_flow BOOLEAN NOT NULL,
    hit_news BOOLEAN NOT NULL,
    PRIMARY KEY (symbol, horizon, trade_date)
);
"""


def normalize_tr(text: str) -> str:
    """Fold Turkish text to lowercase with unified `i` (`İ/I/ı -> i`) and normalized apostrophes."""
    if not text:
        return ""
    s = text.replace("İ", "i").replace("I", "i").replace("ı", "i").lower()
    s = s.replace("İ".lower(), "i").replace("ı", "i")
    for ch in ("’", "‘", "´", "`", "''", '"'):
        s = s.replace(ch, "'")
    return s


STOCK_ALIASES: dict[str, tuple[str, ...]] = {
    "AEFES": ("aefes", "anadolu efes"),
    "AKBNK": ("akbnk", "akbank"),
    "ASELS": ("asels", "aselsan"),
    "ASTOR": ("astor", "astor enerji"),
    "BIMAS": ("bimas", "bim birlesik", "bim magaza"),
    "EKGYO": ("ekgyo", "emlak konut"),
    "ENKAI": ("enkai", "enka insaat", "enka inşaat"),
    "EREGL": ("eregl", "erdemir", "eregli demir", "ereğli demir"),
    "FROTO": ("froto", "ford otosan", "ford otomotiv"),
    "GARAN": ("garan", "garanti bbva", "garanti bankasi"),
    "GUBRF": ("gubrf", "gubre fabrikalari", "gübre fabrikalari"),
    "ISCTR": ("isctr", "is bankasi", "iş bankasi"),
    "KCHOL": ("kchol", "koc holding", "koç holding"),
    "KRDMD": ("krdmd", "kardemir"),
    "MGROS": ("mgros", "migros"),
    "PETKM": ("petkm", "petkim"),
    "PGSUS": ("pgsus", "pegasus"),
    "SAHOL": ("sahol", "sabanci holding", "sabancı holding"),
    "SASA": ("sasa", "sasa polyester"),
    "SISE": ("sise", "sisecam", "şişecam"),
    "TAVHL": ("tavhl", "tav havalimanlari"),
    "TCELL": ("tcell", "turkcell"),
    "THYAO": ("thyao", "türk hava yollari", "turk hava yollari"),
    "TOASO": ("toaso", "tofas", "tofaş"),
    "TRALT": ("tralt", "altin sertifikasi", "altın sertifikası"),
    "TTKOM": ("ttkom", "türk telekom", "turk telekom"),
    "TUPRS": ("tuprs", "tupras", "tüpraş"),
    "VAKBN": ("vakbn", "vakifbank", "vakıfbank"),
    "YKBNK": ("ykbnk", "yapi kredi", "yapı kredi"),
}


def detect_explicit_title_stocks(title: str) -> set[str]:
    """Identify which stocks are explicitly named in `title`."""
    t = normalize_tr(title)
    matched: set[str] = set()
    for sym, aliases in STOCK_ALIASES.items():
        if any(alias in t for alias in aliases):
            matched.add(sym)
    return matched


def ensure_news_tables(db: Any) -> None:
    db.execute(NEWS_DDL)


def now_trt() -> datetime:
    return datetime.now(TRT).replace(tzinfo=None, microsecond=0)


def headline_id(scope: str, title: str) -> str:
    key = f"{scope}|{' '.join(title.split()).casefold()}"
    return hashlib.sha256(key.encode("utf-8")).hexdigest()[:32]


def is_placeholder_timestamp(pub_dt_trt: datetime, pub_raw: str = "") -> bool:
    """Google News replaces the time of articles older than about 3 days with 07:00:00 GMT (10:00:00 TRT)."""
    return ("07:00:00" in pub_raw) or (pub_dt_trt.hour == 10 and pub_dt_trt.minute == 0 and pub_dt_trt.second == 0)


def map_pubdate_to_trading_session(pub_dt_trt: datetime, trading_dates: Sequence[str], date_only: bool) -> str | None:
    """First session whose 18:10 TRT close is at or after publication; None if before the calendar or still pending.

    A date-only article may have appeared after that day's close, so it always maps to a later session.
    """
    pub_date = pub_dt_trt.strftime("%Y-%m-%d")
    if not trading_dates or pub_date < trading_dates[0]:
        return None
    after_close = date_only or pub_dt_trt.time() > SESSION_CLOSE_TIME
    for d in trading_dates:
        if d > pub_date or (d == pub_date and not after_close):
            return d
    return None


def collection_mode(published_at: datetime, first_seen_at: datetime, backfill: bool) -> str:
    """LIVE only when the harvest saw the article soon after publication, which rules out retrieval hindsight."""
    if backfill or (first_seen_at - published_at).total_seconds() > LIVE_COLLECTION_WINDOW_HOURS * 3600:
        return "BACKFILL"
    return "LIVE"


def upsert_headlines(db: Any, rows: Iterable[dict[str, Any]], seen_at: datetime, backfill: bool = False) -> int:
    """Insert unseen headlines and return how many were new. An existing headline keeps its frozen fields."""
    new_count = 0
    for r in rows:
        published = pd.Timestamp(r["published_at_trt"]).to_pydatetime()
        cur = db.execute(
            """
            INSERT INTO bronze_news_headlines (headline_id, scope, category, title, link, published_at_trt,
                timestamp_precision, first_seen_at_trt, last_seen_at_trt, collection_mode)
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
            ON CONFLICT (headline_id) DO UPDATE SET last_seen_at_trt = EXCLUDED.last_seen_at_trt
            RETURNING (xmax = 0)
            """,
            [
                headline_id(r["scope"], r["title"]), r["scope"], r["category"], r["title"], r.get("link"), published,
                r["timestamp_precision"], seen_at, seen_at, collection_mode(published, seen_at, backfill),
            ],
        )
        row = cur.fetchone()
        new_count += int(bool(row and row[0]))
    return new_count


def load_trading_dates(db: Any) -> list[str]:
    df = db.query_df("SELECT DISTINCT trade_date::text AS d FROM silver_daily_stock_summary ORDER BY 1")
    return df["d"].tolist()


def load_headlines_with_sessions(db: Any, trading_dates: Sequence[str]) -> pd.DataFrame:
    """Headlines mapped to their causal session. Those waiting for the next session stay out until it exists."""
    df = db.query_df(
        """
        SELECT headline_id, scope, category, title, link, published_at_trt, timestamp_precision,
               first_seen_at_trt, collection_mode
        FROM bronze_news_headlines ORDER BY published_at_trt, headline_id
        """
    )
    if df.empty:
        return df.assign(effective_trade_date=pd.Series(dtype=str))
    df["effective_trade_date"] = [
        map_pubdate_to_trading_session(pd.Timestamp(p).to_pydatetime(), trading_dates, tp == PRECISION_DATE_ONLY)
        for p, tp in zip(df["published_at_trt"], df["timestamp_precision"])
    ]
    df = df[df["effective_trade_date"].notna()].reset_index(drop=True)
    df["published_at_trt"] = pd.to_datetime(df["published_at_trt"]).dt.strftime("%Y-%m-%d %H:%M:%S")
    return df
