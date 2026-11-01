#!/usr/bin/env python
"""Deterministic Point-in-Time BIST News & 3-Forces Microstructure Harvester.

Part of the MDK Trading Oracle Agentic Intelligence System ($0 external API cost).
1. Queries Google News TR RSS across macro/crisis topics ('Fon Krizi', SPK liquidation/investigations,
   Takasbank margin calls, TCMB/CDS shocks) and stock-specific catalysts (ASELS, AKBNK, GARAN, ISCTR, YKBNK, THYAO).
2. Appends unseen headlines to `bronze_news_headlines`. Publication time and precision are frozen at first sight;
   sessions are mapped when scoring (`data/bronze/news.py`), so a later harvest cannot rewrite history.
3. Extracts the daily 3-Forces Microstructure Clash Vector (`MLB` vs `KAMU` vs `BIG5`) from Silver/Gold frames.
"""

from __future__ import annotations

import argparse
import email.utils
import sys
import urllib.parse
import urllib.request
import xml.etree.ElementTree as ET
from datetime import timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from mdk_trading_oracle.core.db import PostgresManager
from mdk_trading_oracle.data.bronze.news import (
    PRECISION_DATE_ONLY,
    PRECISION_EXACT,
    TRT,
    ensure_news_tables,
    is_placeholder_timestamp,
    now_trt,
    upsert_headlines,
)

OUT_DIR = Path.home() / "data" / "mdk_oracle" / "experiments" / "timestral"
FRAMES_DIR = Path.home() / "data" / "mdk_oracle" / "experiments" / "feature_importance" / "frames"
BIST30_STOCKS = [
    "AEFES", "AKBNK", "ASELS", "ASTOR", "BIMAS", "EKGYO", "ENKAI", "EREGL",
    "FROTO", "GARAN", "GUBRF", "ISCTR", "KCHOL", "KRDMD", "MGROS", "PETKM",
    "PGSUS", "SAHOL", "SASA", "SISE", "TAVHL", "TCELL", "THYAO", "TOASO",
    "TRALT", "TTKOM", "TUPRS", "VAKBN", "YKBNK",
]

NEWS_QUERIES: list[tuple[str, str, str]] = [
    # (scope, category, query)
    ("MACRO", "FON_KRIZI", "Borsa İstanbul fon krizi OR SPK fon tasfiye OR manipülasyon soruşturma"),
    ("MACRO", "MARGIN_CALL", "Borsa İstanbul teminat tamamlama çağrısı OR kredili işlem satış OR ViOP teminat"),
    ("MACRO", "BOFA_FLOW", "Borsa İstanbul BofA satış OR Bank of America BIST satış OR yabancı çıkışı"),
    ("MACRO", "KAMU_DEFENSE", "Borsa İstanbul Varlık Fonu alım OR Ziraat Yatırım BIST destek OR BES fonları"),
    ("MACRO", "TCMB_MACRO", "Borsa İstanbul Merkez Bankası faiz OR enflasyon muhasebesi bilanço baskısı"),
    ("MACRO", "GEO_HORMUZ_ENERGY", "Hürmüz Boğazı OR İran İsrail Orta Doğu gerilim Borsa İstanbul OR petrol TÜPRAŞ savunma ASELSAN"),
    ("MACRO", "GEO_UKRAINE_BLACKSEA", "Rusya Ukrayna savaş Karadeniz Borsa İstanbul OR barış müzakere ENKAI THYAO havacılık"),
    ("ASELS", "STOCK_NEWS", "ASELSAN hisse düşüş OR BofA Aselsan satış OR Aselsan KAP sözleşme"),
    ("THYAO", "STOCK_NEWS", "THYAO hisse OR Türk Hava Yolları Borsa İstanbul"),
    ("AKBNK", "STOCK_NEWS", "AKBNK hisse OR Borsa bankacılık endeksi XBANK"),
    ("GARAN", "STOCK_NEWS", "GARAN hisse OR Garanti BBVA Borsa İstanbul"),
    ("ISCTR", "STOCK_NEWS", "ISCTR hisse OR İş Bankası Borsa İstanbul"),
    ("YKBNK", "STOCK_NEWS", "YKBNK hisse OR Yapı Kredi Borsa İstanbul"),
    ("BIST30_HOLDINGS", "STOCK_NEWS", "KCHOL OR SAHOL OR SISE OR ENKAI hisse Borsa İstanbul BofA"),
    ("BIST30_INDUST", "STOCK_NEWS", "TUPRS OR EREGL OR KRDMD OR PETKM OR SASA OR GUBRF OR ASTOR hisse Borsa İstanbul"),
    ("BIST30_AUTO_TRANS", "STOCK_NEWS", "FROTO OR TOASO OR PGSUS OR TAVHL hisse Borsa İstanbul"),
    ("BIST30_CONS_TELE", "STOCK_NEWS", "BIMAS OR MGROS OR AEFES OR TCELL OR TTKOM OR EKGYO OR VAKBN hisse Borsa İstanbul"),
]


def harvest_google_news_rss() -> tuple[pd.DataFrame, int]:
    """Fetch public Google News TR RSS items. Returns the rows and how many queries failed."""
    rows: list[dict[str, Any]] = []
    failures = 0

    for scope, category, query in NEWS_QUERIES:
        url = f"https://news.google.com/rss/search?q={urllib.parse.quote(query)}&hl=tr&gl=TR&ceid=TR:tr"
        req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
        try:
            xml_bytes = urllib.request.urlopen(req, timeout=12).read()
            root = ET.fromstring(xml_bytes)
        except Exception as e:
            print(f"[WARN] Failed RSS query ({scope}/{category}): {e}")
            failures += 1
            continue

        for item in root.findall(".//item"):
            title = (item.findtext("title") or "").strip()
            pub_raw = (item.findtext("pubDate") or "").strip()
            link = (item.findtext("link") or "").strip()
            if not title or not pub_raw:
                continue
            try:
                dt = email.utils.parsedate_to_datetime(pub_raw)
                if dt.tzinfo is None:
                    dt = dt.replace(tzinfo=timezone.utc)
                dt_trt = dt.astimezone(TRT)
            except Exception:
                continue
            rows.append({
                "scope": scope,
                "category": category,
                "published_at_trt": dt_trt.strftime("%Y-%m-%d %H:%M:%S"),
                "timestamp_precision": PRECISION_DATE_ONLY if is_placeholder_timestamp(dt_trt, pub_raw) else PRECISION_EXACT,
                "title": title,
                "link": link,
            })
    return pd.DataFrame(rows), failures


def seed_from_parquet(db: Any) -> int:
    """One-time import of the pre-database parquet. Every row is BACKFILL: it was retrieved long after publication."""
    legacy = OUT_DIR / "harvested_bist_news.parquet"
    if not legacy.exists():
        print(f"No legacy file at {legacy}")
        return 0
    df = pd.read_parquet(legacy)
    return upsert_headlines(db, df.to_dict("records"), now_trt(), backfill=True)


def extract_3forces_clash_features(db: Any) -> pd.DataFrame:
    """Compute strictly point-in-time 3-Forces (`MLB` vs `KAMU` vs `BIG5`) clash & distress features
    across all active BIST 30 stocks (`BIST30_STOCKS`), using cached parquet frames when available and
    querying PostgreSQL `extract_3pillar_time_series` for all other BIST 30 constituents.
    """
    from mdk_trading_oracle.data.silver.tertip_ml_forecaster import extract_3pillar_time_series

    latest = str(db.query_df("SELECT max(trade_date)::text AS d FROM silver_daily_stock_summary")["d"].iloc[0])
    parts: list[pd.DataFrame] = []
    for sym in BIST30_STOCKS:
        p_path = FRAMES_DIR / f"{sym}.parquet"
        df = pd.read_parquet(p_path).sort_values("trade_date").reset_index(drop=True) if p_path.exists() else pd.DataFrame()
        # A cached frame behind the database would freeze the flow features, so it is only used when current.
        if df.empty or str(df["trade_date"].iloc[-1])[:10] != latest:
            df = extract_3pillar_time_series(db, sym, lookback_days=1250).sort_values("trade_date").reset_index(drop=True)
        if df.empty or len(df) < 60:
            continue
        dates = [str(d)[:10] for d in df["trade_date"].tolist()]
        turnover = np.maximum(df["total_turnover_tl"].to_numpy(float), 1.0)

        mlb_sh = df["mlb_flow"].to_numpy(float) / turnover * 100.0
        kamu_sh = df["kamu_flow"].to_numpy(float) / turnover * 100.0
        big5_sh = df["big5_flow"].to_numpy(float) / turnover * 100.0
        mlb_w5_sh = df["mlb_w5_flow"].to_numpy(float) / turnover * 100.0

        mlb_5d = pd.Series(mlb_sh).rolling(5, min_periods=1).mean().to_numpy(float)
        mlb_10d = pd.Series(mlb_sh).rolling(10, min_periods=1).mean().to_numpy(float)
        kamu_5d = pd.Series(kamu_sh).rolling(5, min_periods=1).mean().to_numpy(float)
        big5_5d = pd.Series(big5_sh).rolling(5, min_periods=1).mean().to_numpy(float)
        w5_5d = pd.Series(mlb_w5_sh).rolling(5, min_periods=1).mean().to_numpy(float)

        mlb_cost = pd.to_numeric(df["feat_mlb_cost_spread_pct"], errors="coerce").fillna(0.0).to_numpy(float)
        big5_cost = pd.to_numeric(df["feat_big5_cost_spread_pct"], errors="coerce").fillna(0.0).to_numpy(float)
        kamu_cost = pd.to_numeric(df["feat_kamu_cost_spread_pct"], errors="coerce").fillna(0.0).to_numpy(float)
        ret_5d = pd.Series(df["close_price"].to_numpy(float)).pct_change(5).fillna(0.0).to_numpy(float) * 100.0
        ret_10d = pd.Series(df["close_price"].to_numpy(float)).pct_change(10).fillna(0.0).to_numpy(float) * 100.0

        # Passive absorption trap: MLB selling over 5D/10D (< -0.5% of turnover) while KAMU+BIG5 absorb (> +0.3%)
        # AND MLB FIFO inventory is underwater (mlb_cost < -1.0%) while BIG5 still sits on prior gains (big5_cost > 5%)
        dom_5d = kamu_5d + big5_5d
        is_absorption_trap = (
            (mlb_5d < -0.40)
            & (mlb_10d < -0.30)
            & (dom_5d > 0.20)
            & (mlb_cost < -1.0)
        )

        # Continuous 3-Forces directional pressure (strictly causal at session T close):
        # Driven by MLB 5D/10D flow + W5 closing sweep + MLB cost spread vs BIG5 profit-taking overhang
        mlb_z = np.clip((0.50 * mlb_5d + 0.50 * mlb_10d) / 1.5, -2.0, 2.0)
        w5_z = np.clip(w5_5d / 0.5, -2.0, 2.0)
        cost_z = np.clip(mlb_cost / 8.0, -2.0, 2.0)

        # When absorption trap is active, domestic buying is a contrarian bearish overhang rather than bullish support
        clash_score = np.where(
            is_absorption_trap,
            np.minimum(-0.35, 0.50 * mlb_z + 0.25 * w5_z + 0.25 * cost_z - 0.30 * np.clip(dom_5d / 1.5, 0.0, 1.5)),
            0.45 * mlb_z + 0.25 * w5_z + 0.30 * cost_z,
        )

        parts.append(pd.DataFrame({
            "symbol": sym,
            "trade_date": dates,
            "close_price": df["close_price"].to_numpy(float),
            "mlb_5d_share_pct": np.round(mlb_5d, 3),
            "mlb_10d_share_pct": np.round(mlb_10d, 3),
            "kamu_5d_share_pct": np.round(kamu_5d, 3),
            "big5_5d_share_pct": np.round(big5_5d, 3),
            "mlb_w5_5d_share_pct": np.round(w5_5d, 3),
            "mlb_cost_spread_pct": np.round(mlb_cost, 2),
            "big5_cost_spread_pct": np.round(big5_cost, 2),
            "kamu_cost_spread_pct": np.round(kamu_cost, 2),
            "ret_5d_pct": np.round(ret_5d, 2),
            "ret_10d_pct": np.round(ret_10d, 2),
            "is_passive_absorption_trap": is_absorption_trap,
            "three_forces_clash_score": np.round(np.clip(clash_score, -2.0, 2.0), 4),
        }))

    out = pd.concat(parts, ignore_index=True)
    # Also compute Market-6 cross-sectional 3-Forces distress (how many of the 6 focus stocks are in MLB distribution / absorption trap)
    mkt_trap_breadth = out.groupby("trade_date")["is_passive_absorption_trap"].transform("mean")
    mkt_clash_mean = out.groupby("trade_date")["three_forces_clash_score"].transform("mean")
    out["mkt6_absorption_trap_breadth"] = np.round(mkt_trap_breadth, 3)
    out["mkt6_three_forces_clash_mean"] = np.round(mkt_clash_mean, 4)
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--seed-from-parquet", action="store_true", help="Import the legacy parquet as BACKFILL first")
    ap.add_argument("--skip-clash", action="store_true", help="Only harvest headlines")
    args = ap.parse_args()

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    db = PostgresManager()
    ensure_news_tables(db)
    if args.seed_from_parquet:
        print(f"Seeded {seed_from_parquet(db)} legacy headlines as BACKFILL", flush=True)

    df_news, failures = harvest_google_news_rss()
    new_count = upsert_headlines(db, df_news.to_dict("records"), now_trt()) if not df_news.empty else 0
    total = db.query_df("SELECT count(*) AS n FROM bronze_news_headlines")["n"].iloc[0]
    print(f"Fetched {len(df_news)} items, {new_count} new, {total} stored, {failures}/{len(NEWS_QUERIES)} queries failed", flush=True)
    if failures == len(NEWS_QUERIES):
        print("[ERROR] Every RSS query failed", file=sys.stderr)
        sys.exit(1)

    if not args.skip_clash:
        df_3f = extract_3forces_clash_features(db)
        clash_path = OUT_DIR / "three_forces_daily_clash.parquet"
        df_3f.to_parquet(clash_path, index=False)
        print(f"Saved {len(df_3f)} stock-session rows to {clash_path}", flush=True)


if __name__ == "__main__":
    main()
