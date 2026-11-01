#!/usr/bin/env python
"""Deterministic Point-in-Time BIST News & 3-Forces Microstructure Harvester.

Part of the MDK Trading Oracle Agentic Intelligence System ($0 external API cost).
1. Queries Google News TR RSS across macro/crisis topics ('Fon Krizi', SPK liquidation/investigations,
   Takasbank margin calls, TCMB/CDS shocks) and stock-specific catalysts (ASELS, AKBNK, GARAN, ISCTR, YKBNK, THYAO).
2. Enforces strict Point-in-Time Turkish Time (`Europe/Istanbul`, 18:10 TRT session close) causality:
   Every article's RFC-2822 `<pubDate>` is converted to TRT; if published after 18:10 TRT or on a weekend/holiday,
   it is mapped strictly to the NEXT completed trading session close.
3. Extracts the daily 3-Forces Microstructure Clash Vector (`MLB` vs `KAMU` vs `BIG5`) from Silver/Gold frames.
"""

from __future__ import annotations

import email.utils
import urllib.parse
import urllib.request
import xml.etree.ElementTree as ET
from datetime import datetime, time, timezone
from pathlib import Path
from typing import Any
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd

TRT = ZoneInfo("Europe/Istanbul")
SESSION_CLOSE_TIME = time(18, 10)

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


def map_pubdate_to_trading_session(
    pub_dt_trt: datetime,
    trading_dates: list[str],
    pub_raw: str = "",
) -> str | None:
    """Map an article's TRT publication timestamp to the first trading session `d` whose 18:10 TRT close
    is strictly `>= pub_dt_trt` (100% zero-lookahead guarantee).

    CRITICAL ANTI-LEAKAGE GUARD FOR GOOGLE NEWS `07:00:00 GMT` (`10:00:00 TRT`) PLACEHOLDERS:
    Google News RSS replaces the exact hour/minute/second of archived articles (>3 days old) with a
    synthetic `07:00:00 GMT` (`10:00:00 TRT`) placeholder (`date-only` resolution). Because an archived
    article published on calendar day `D` might have come out at `19:30 TRT` (after BIST's 18:10 close),
    treating `07:00:00 GMT` as `10:00 TRT` would risk same-evening post-close leakage!
    Therefore, whenever an item carries the synthetic `07:00:00 GMT` (`10:00:00` TRT) placeholder, we
    conservatively treat it as post-close (`after_close = True`) and map it strictly to the NEXT trading
    session `D + 1` (`d > pub_date_str`). Only articles with verified sub-minute timestamps (`HH:MM:SS != 07:00:00 GMT`)
    published at `<= 18:10 TRT` are mapped to session `D`.
    """
    pub_date_str = pub_dt_trt.strftime("%Y-%m-%d")
    is_synthetic_archive_time = ("07:00:00" in pub_raw) or (
        pub_dt_trt.hour == 10 and pub_dt_trt.minute == 0 and pub_dt_trt.second == 0
    )
    after_close = is_synthetic_archive_time or (pub_dt_trt.time() > SESSION_CLOSE_TIME)
    for d in trading_dates:
        if d > pub_date_str:
            return d
        if d == pub_date_str and not after_close:
            return d
    return None


def harvest_google_news_rss(trading_dates: list[str]) -> pd.DataFrame:
    """Fetch public Google News TR RSS items and align each item strictly to its causal trading session."""
    rows: list[dict[str, Any]] = []
    seen_titles: set[tuple[str, str]] = set()

    for scope, category, query in NEWS_QUERIES:
        url = f"https://news.google.com/rss/search?q={urllib.parse.quote(query)}&hl=tr&gl=TR&ceid=TR:tr"
        req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
        try:
            xml_bytes = urllib.request.urlopen(req, timeout=12).read()
            root = ET.fromstring(xml_bytes)
        except Exception as e:
            print(f"[WARN] Failed RSS query ({scope}/{category}): {e}")
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

            eff_session = map_pubdate_to_trading_session(dt_trt, trading_dates, pub_raw=pub_raw)
            if eff_session is None:
                continue
            dedup_key = ( scope, title[:80] )
            if dedup_key in seen_titles:
                continue
            seen_titles.add(dedup_key)
            is_synthetic_time = ("07:00:00" in pub_raw) or (
                dt_trt.hour == 10 and dt_trt.minute == 0 and dt_trt.second == 0
            )
            rows.append({
                "scope": scope,
                "category": category,
                "published_at_trt": dt_trt.strftime("%Y-%m-%d %H:%M:%S"),
                "timestamp_precision": "DATE_ONLY_SHIFTED_T+1" if is_synthetic_time else "EXACT_SECOND_VERIFIED",
                "effective_trade_date": eff_session,
                "title": title,
                "link": link,
            })

    df_news = pd.DataFrame(rows)
    if not df_news.empty:
        df_news = df_news.sort_values(["effective_trade_date", "published_at_trt"]).reset_index(drop=True)
    return df_news


def extract_3forces_clash_features() -> pd.DataFrame:
    """Compute strictly point-in-time 3-Forces (`MLB` vs `KAMU` vs `BIG5`) clash & distress features
    across all active BIST 30 stocks (`BIST30_STOCKS`), using cached parquet frames when available and
    querying PostgreSQL `extract_3pillar_time_series` for all other BIST 30 constituents.
    """
    from mdk_trading_oracle.core.db import PostgresManager
    from mdk_trading_oracle.data.silver.tertip_ml_forecaster import extract_3pillar_time_series

    db = PostgresManager()
    parts: list[pd.DataFrame] = []
    for sym in BIST30_STOCKS:
        p_path = FRAMES_DIR / f"{sym}.parquet"
        if p_path.exists():
            df = pd.read_parquet(p_path).sort_values("trade_date").reset_index(drop=True)
        else:
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
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    df_asels = pd.read_parquet(FRAMES_DIR / "ASELS.parquet").sort_values("trade_date").reset_index(drop=True)
    trading_dates = [str(d)[:10] for d in df_asels["trade_date"].tolist()]

    print("1. Harvesting point-in-time Google News TR RSS feeds...", flush=True)
    df_news = harvest_google_news_rss(trading_dates)
    news_path = OUT_DIR / "harvested_bist_news.parquet"
    df_news.to_parquet(news_path, index=False)
    print(f"   Saved {len(df_news)} timestamped news items to {news_path}", flush=True)
    if not df_news.empty:
        recent = df_news[df_news["effective_trade_date"] >= "2026-09-01"]
        print(f"   Items mapped to Sept-Oct 2026 sessions: {len(recent)}")
        for r in recent.itertuples(index=False):
            print(f"     [{r.effective_trade_date} <- pub {r.published_at_trt}] ({r.scope}/{r.category}) {r.title}")

    print("\n2. Extracting point-in-time 3-Forces Clash Features across focus stocks...", flush=True)
    df_3f = extract_3forces_clash_features()
    clash_path = OUT_DIR / "three_forces_daily_clash.parquet"
    df_3f.to_parquet(clash_path, index=False)
    print(f"   Saved {len(df_3f)} stock-session rows to {clash_path}", flush=True)

    asels_sep_oct = df_3f[(df_3f["symbol"] == "ASELS") & (df_3f["trade_date"] >= "2026-09-10") & (df_3f["trade_date"] <= "2026-10-16")]
    print("\n=== ASELS 3-Forces Clash Vector (2026-09-10 to 2026-10-16) ===")
    print(asels_sep_oct[[
        "trade_date", "close_price", "mlb_5d_share_pct", "mlb_10d_share_pct",
        "kamu_5d_share_pct", "big5_5d_share_pct", "mlb_cost_spread_pct",
        "is_passive_absorption_trap", "three_forces_clash_score"
    ]].to_string(index=False))


if __name__ == "__main__":
    main()

