#!/usr/bin/env python
"""Point-in-Time BIST News Regime Scorer & 3-Forces Microstructure Fusion Engine.

Part of the MDK Trading Oracle Agentic Intelligence System.
Enforces strict zero-leakage causality:
1. Every headline in `bronze_news_headlines` is verified against `published_at_trt <= effective_trade_date 18:10 TRT`.
2. Headlines are scored at their `effective_trade_date` into four `[0.0, 1.0]` intensity dimensions:
   - `fon_krizi_news_intensity`: SPK fund liquidations ('fon krizi', 'tasfiye'), margin calls ('teminat tamamlama'),
     Takasbank defaults ('temerrüt'), manipulation arrests ('gözaltı', 'soruşturma', 'manipülasyon'),
     brokerage bond redemption stress ('itfa baskısı'), forced margin selling ('kredili işlem satış').
   - `bofa_sell_news_intensity`: BofA ('MLB') institutional selling ('BofA satış', 'net satış', 'yabancı çıkışı',
     'sert düşüş', 'taban').
   - `kamu_floor_news_intensity`: State/pension floor support ('Ziraat Yatırım', 'Varlık Fonu', 'destek alımı')
     during market selloffs.
   - `bullish_catalyst_intensity`: Major sovereign/export contracts and institutional accumulation
     ('ihracat sözleşmesi', 'KAP', 'hedef fiyat', 'Al tavsiyesi', 'dev alım', 'rekor kar').
3. Computes strictly causal 5-session (`ewm(halflife=5, adjust=False)`) and 10-session (`ewm(halflife=10, adjust=False)`)
   exponentially decaying news memory per `symbol` and `trade_date`.
4. Merges the causal news memory with `three_forces_daily_clash.parquet` and persists:
   - `silver_news_regimes_daily` (served by the API)
   - `~/data/mdk_oracle/experiments/timestral/agentic_news_regimes.parquet` (read by the forecaster)
   - `~/data/mdk_oracle/experiments/timestral/agentic_news_regimes_summary.json`
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from mdk_trading_oracle.core.db import PostgresManager
from mdk_trading_oracle.data.bronze.news import (
    detect_explicit_title_stocks,
    ensure_news_tables,
    load_headlines_with_sessions,
    load_trading_dates,
    normalize_tr,
)

OUT_DIR = Path.home() / "data" / "mdk_oracle" / "experiments" / "timestral"
FOCUS_STOCKS = [
    "AEFES", "AKBNK", "ASELS", "ASTOR", "BIMAS", "EKGYO", "ENKAI", "EREGL",
    "FROTO", "GARAN", "GUBRF", "ISCTR", "KCHOL", "KRDMD", "MGROS", "PETKM",
    "PGSUS", "SAHOL", "SASA", "SISE", "TAVHL", "TCELL", "THYAO", "TOASO",
    "TRALT", "TTKOM", "TUPRS", "VAKBN", "YKBNK",
]


def _kw_list(*items: str) -> tuple[str, ...]:
    return tuple(normalize_tr(x) for x in items)


FON_STRONG_KW = _kw_list(
    "fon krizi",
    "fon kriz",
    "fon vurgun",
    "fon skandal",
    "fon soygun",
    "fon soruşturma",
    "fon mağdur",
    "tasfiye",
    "teminat tamamlama",
    "teminat çağrısı",
    "temerrüt",
    "gözaltı",
    "manipülasyon",
    "itfa baskısı",
    "kredili işlem satış",
    "mal varlığı donduruldu",
    "mal varlığına el konuldu",
    "işlem yasağı",
    "yakalama kararı",
)

FON_MOD_KW = _kw_list(
    "soruşturma",
    "suç duyurusu",
    "açığa satış hareketliliği",
    "açığa satış listesi",
    "fon dolandırıcı",
    "likidite kriz",
    "operasyon",
)

BOFA_BUY_OVERRIDE_KW = _kw_list(
    "bofa alıma geçti",
    "bofa borsada alıma döndü",
)

BOFA_STRONG_KW = _kw_list(
    "bofa satış",
    "bofa'dan hisse satışları",
    "bofa'dan hisselerde net satışlar",
    "bofa'dan borsa istanbul'da büyük satışlar",
    "bofa'dan borsa istanbul'da satışlar",
    "bofa baskısı",
    "bofa çıkıştı",
    "net satış",
    "yabancı çıkışı",
    "sert düşüş",
    "sert düştü",
    "taban oldu",
    "devre kesti",
    "devre kesici",
    "dev satış",
)

BOFA_MOD_KW = _kw_list(
    "neden düştü",
    "değer kaybetti",
    "düşüş yüzde",
    "aldığı ve sattığı hisseler",
    "alıp sattığı hisseler",
)

KAMU_STRONG_KW = _kw_list(
    "ziraat yatırım",
    "varlık fonu",
    "destek alımı",
    "borsa destek",
    "bes fonları",
)

KAMU_MOD_KW = _kw_list(
    "ziraat bankası",
    "ziraat'tan",
    "kamu",
)

BULL_STRONG_KW = _kw_list(
    "ihracat sözleşmesi",
    "ihracat sözleşmeleri",
    "yeni sözleşme",
    "dev anlaşma",
    "savunma ortaklığı",
    "hedef fiyat",
    "al tavsiyesi",
    "'al' tavsiyesi",
    "'al' önerisi",
    "dev alım",
    "rekor alım",
    "rekor kar",
    "beklentileri aştı",
)

BULL_MOD_KW = _kw_list(
    "al sinyali",
    "'al' veriyor",
    "alıma geçti",
    "alıma döndü",
    "yüksek alım",
    "model portföy",
    "en çok önerilen",
    "prim potansiyeli",
    "temettü",
    "güne yükselişle başladı",
    "güçlü kapanış",
)

GEO_ESCALATION_KW = _kw_list(
    "hürmüz",
    "hurmuz",
    "iran israil",
    "orta doğu gerilim",
    "orta dogu gerilim",
    "savaş",
    "savas",
    "füze",
    "fuze",
    "hava sahası",
    "petrol fiyatları fırladı",
    "brent petrol",
    "jeopolitik risk",
    "karadeniz gerilim",
)

GEO_PEACE_KW = _kw_list(
    "barış müzakere",
    "baris muzakere",
    "ateşkes",
    "ateskes",
    "ukrayna yeniden inşa",
)

# Sectoral transmission of Geopolitical / Regional Crises on BIST 30:
# - Defense / Gold / Refining (ASELS, TRALT, TUPRS): Geopolitical escalation (Hormuz/Middle East) acts as a safe-haven or margin hedge (+0.45 bullish, 0.25 risk).
# - Aviation & Tourism (THYAO, PGSUS, TAVHL): Airspace closures + Brent oil spike act as a severe bearish shock (+0.90 crisis/risk).
# - Black Sea / Russia-Ukraine Reconstruction (ENKAI, SISE, AEFES): Escalation is bearish (+0.75 risk); peace/ceasefire is a major bullish catalyst (+0.90 bull).
GEO_DEFENSE_ENERGY_STOCKS = {"ASELS", "TRALT", "TUPRS"}
GEO_AVIATION_STOCKS = {"THYAO", "PGSUS", "TAVHL"}
GEO_BLACKSEA_RECON_STOCKS = {"ENKAI", "SISE", "AEFES"}


KAP_WORD = re.compile(r"\bkap\b")


def score_single_headline(title: str, category: str) -> dict[str, float]:
    """Score a single Turkish financial headline into the four `[0.0, 1.0]` regime intensities."""
    t = normalize_tr(title)
    cat = (category or "").upper()

    # 1. Fon Krizi / Liquidation / Margin Call / Default / Manipulation Investigation
    fon_hits_strong = sum(1 for k in FON_STRONG_KW if k in t)
    fon_hits_mod = sum(1 for k in FON_MOD_KW if k in t)
    fon_score = 0.0
    if fon_hits_strong >= 2:
        fon_score = 1.0
    elif fon_hits_strong == 1:
        fon_score = 0.85 if fon_hits_mod == 0 else 1.0
    elif fon_hits_mod >= 1:
        fon_score = 0.65
    if cat in ("FON_KRIZI", "MARGIN_CALL") and fon_score > 0.0:
        fon_score = min(1.0, fon_score + 0.15)
    elif cat in ("FON_KRIZI", "MARGIN_CALL") and ("fon" in t or "spk" in t or "borsa" in t):
        fon_score = max(fon_score, 0.70)

    # 2. BofA Distribution / Foreign Outflow / Sharp Selloff
    bofa_buy_override = any(k in t for k in BOFA_BUY_OVERRIDE_KW)
    bofa_score = 0.0
    if not bofa_buy_override:
        b_strong = sum(1 for k in BOFA_STRONG_KW if k in t)
        b_mod = sum(1 for k in BOFA_MOD_KW if k in t)
        if b_strong >= 2:
            bofa_score = 1.0
        elif b_strong == 1:
            bofa_score = 0.85 if b_mod == 0 else 1.0
        elif b_mod >= 1:
            bofa_score = 0.55
        if cat == "BOFA_FLOW" and bofa_score > 0.0:
            bofa_score = min(1.0, bofa_score + 0.15)

    # 3. Kamu Floor Defense (Ziraat Yatırım / Varlık Fonu / BES support buying)
    k_strong = sum(1 for k in KAMU_STRONG_KW if k in t)
    k_mod = sum(1 for k in KAMU_MOD_KW if k in t)
    kamu_score = 0.0
    if k_strong >= 1:
        kamu_score = 0.90 if ("alim" in t or "destek" in t or "milyar" in t) else 0.75
    elif k_mod >= 1 and cat == "KAMU_DEFENSE":
        kamu_score = 0.65

    # 4. Bullish Sovereign / KAP / Institutional Accumulation Catalyst
    # KAP needs a whole-word match: as a substring it fires on "kapsam", "kapanış" and "kapandı".
    bu_strong = sum(1 for k in BULL_STRONG_KW if k in t) + int(bool(KAP_WORD.search(t)))
    bu_mod = sum(1 for k in BULL_MOD_KW if k in t)
    bull_score = 0.0
    if bu_strong >= 2:
        bull_score = 1.0
    elif bu_strong == 1:
        bull_score = 0.85 if bu_mod == 0 else 1.0
    elif bu_mod >= 1:
        bull_score = 0.65

    # 5. Regional / Geopolitical Crisis (Hormuz Strait, Iran-Israel, Russia-Ukraine War / Peace)
    geo_esc = sum(1 for k in GEO_ESCALATION_KW if k in t)
    geo_peace = sum(1 for k in GEO_PEACE_KW if k in t)
    geo_esc_score = 0.85 if geo_esc >= 1 and geo_peace == 0 else 0.0
    geo_peace_score = 0.85 if geo_peace >= 1 else 0.0

    # Suppress false bullish triggers on severe crash/default/arrest headlines unless explicit contract/target
    if fon_score >= 0.85 or (bofa_score >= 0.85 and not bofa_buy_override):
        if "ihracat sözleşme" not in t and "hedef fiyat" not in t and "dev alim" not in t:
            bull_score = 0.0

    return {
        "fon_krizi_news_intensity": round(float(np.clip(fon_score, 0.0, 1.0)), 4),
        "bofa_sell_news_intensity": round(float(np.clip(bofa_score, 0.0, 1.0)), 4),
        "kamu_floor_news_intensity": round(float(np.clip(kamu_score, 0.0, 1.0)), 4),
        "bullish_catalyst_intensity": round(float(np.clip(bull_score, 0.0, 1.0)), 4),
        "geo_escalation_intensity": round(float(np.clip(geo_esc_score, 0.0, 1.0)), 4),
        "geo_peace_intensity": round(float(np.clip(geo_peace_score, 0.0, 1.0)), 4),
    }


def verify_zero_leakage(df_news: pd.DataFrame) -> None:
    """Strictly verify that every article's `published_at_trt` is `<= effective_trade_date 18:10:00`."""
    for row in df_news.itertuples(index=False):
        pub_str = str(row.published_at_trt)
        eff_date = str(row.effective_trade_date)[:10]
        cutoff_str = f"{eff_date} 18:10:00"
        if pub_str > cutoff_str:
            raise ValueError(
                f"Zero-leakage violation detected: published_at_trt={pub_str} > cutoff={cutoff_str} "
                f"for title={row.title!r}"
            )


def classify_daily_regime(row: pd.Series) -> str:
    """Classify each (symbol, trade_date) row into structured BIST microstructure regimes."""
    fon_mem = max(float(row["fon_krizi_news_intensity"]), float(row["fon_krizi_ewm5"]))
    bofa_mem = max(float(row["bofa_sell_news_intensity"]), float(row["bofa_sell_ewm5"]))
    kamu_mem = max(float(row["kamu_floor_news_intensity"]), float(row["kamu_floor_ewm5"]))
    bull_mem = max(float(row["bullish_catalyst_intensity"]), float(row["bullish_catalyst_ewm5"]))

    is_trap = bool(row["is_passive_absorption_trap"])
    clash = float(row["three_forces_clash_score"])
    kamu_flow = float(row["kamu_5d_share_pct"])
    mlb_flow = float(row["mlb_5d_share_pct"])

    # Priority 1: Systemic Fon Krizi Liquidation (forced fund unwinds / margin calls / passive absorption trap)
    if fon_mem >= 0.35 and (is_trap or clash <= -0.35 or float(row["fon_krizi_news_intensity"]) >= 0.70):
        return "FON_KRIZI_LIQUIDATION"

    # Priority 2: Kamu Floor Defense during selloff (Ziraat / Varlık Fonu absorbing selloff)
    if (kamu_mem >= 0.25 or (kamu_flow >= 2.0 and is_trap)) and mlb_flow < -1.0:
        return "KAMU_FLOOR_DEFENSE"

    # Priority 3: Explicit BofA Distribution
    if bofa_mem >= 0.30 or (is_trap and mlb_flow < -1.5):
        return "BOFA_DISTRIBUTION"

    # Priority 4: Bullish Sovereign / KAP Catalyst (only when not overwhelmed by active Fon Krizi liquidation)
    if bull_mem >= 0.35 and clash > -1.0:
        return "BULLISH_SOVEREIGN_OR_KAP_CATALYST"

    if fon_mem >= 0.35:
        return "FON_KRIZI_LIQUIDATION"

    return "NEUTRAL_FLOW"


def build_agentic_news_regimes(db: Any) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Score headlines, compute strictly causal EWM(5) and EWM(10) news memory, and merge with 3-Forces clash."""
    clash_path = OUT_DIR / "three_forces_daily_clash.parquet"

    df_news = load_headlines_with_sessions(db, load_trading_dates(db))
    df_clash = pd.read_parquet(clash_path)

    # 1. Verify 100% Point-in-Time Zero-Leakage rule
    verify_zero_leakage(df_news)

    # 2. Score each headline at its effective_trade_date
    scored_rows: list[dict[str, Any]] = []
    for row in df_news.itertuples(index=False):
        scores = score_single_headline(str(row.title), str(row.category))
        explicit_stocks = detect_explicit_title_stocks(str(row.title))
        scored_rows.append({
            "scope": row.scope,
            "category": row.category,
            "published_at_trt": row.published_at_trt,
            "effective_trade_date": str(row.effective_trade_date)[:10],
            "title": row.title,
            "explicit_stocks": sorted(explicit_stocks),
            **scores,
        })
    df_scored_news = pd.DataFrame(scored_rows)

    # 3. Map each headline strictly at `effective_trade_date` to `(symbol, trade_date)`
    #    - Direct stock mention in title gets full weight (1.0).
    #    - If no stock is explicitly named in the title (e.g., Borsa Istanbul market-wide news harvested under
    #      MACRO or general BIST RSS query), treat as systemic market-wide news across all stocks.
    daily_records: list[dict[str, Any]] = []
    for r in df_scored_news.itertuples(index=False):
        eff_date = r.effective_trade_date
        explicit_set = set(r.explicit_stocks)
        is_market_wide = (r.scope == "MACRO") or (len(explicit_set) == 0)

        for sym in FOCUS_STOCKS:
            if sym in explicit_set or (r.scope == sym and len(explicit_set) == 0):
                w_fon, w_bofa, w_kamu, w_bull = 1.0, 1.0, 1.0, 1.0
            elif is_market_wide:
                w_fon = 1.0
                w_kamu = 0.95
                w_bofa = 0.85 if not explicit_set else 0.50
                w_bull = 0.50 if not explicit_set else 0.0
            else:
                # Headline explicitly names other stock(s) and not `sym`
                w_fon = 0.45 if r.fon_krizi_news_intensity >= 0.80 else 0.0
                w_bofa, w_kamu, w_bull = 0.0, 0.0, 0.0

            fon_val = r.fon_krizi_news_intensity * w_fon
            bofa_val = r.bofa_sell_news_intensity * w_bofa
            kamu_val = r.kamu_floor_news_intensity * w_kamu
            bull_val = r.bullish_catalyst_intensity * w_bull

            # Sector-Conditional Geopolitical Routing (Hormuz / Middle East / Ukraine-Russia):
            geo_esc_v = float(getattr(r, "geo_escalation_intensity", 0.0))
            geo_peace_v = float(getattr(r, "geo_peace_intensity", 0.0))
            if geo_esc_v > 0.0:
                if sym in GEO_DEFENSE_ENERGY_STOCKS:
                    bull_val = max(bull_val, 0.55 * geo_esc_v)
                elif sym in GEO_AVIATION_STOCKS:
                    fon_val = max(fon_val, 0.85 * geo_esc_v)
                    bofa_val = max(bofa_val, 0.65 * geo_esc_v)
                else:
                    fon_val = max(fon_val, 0.50 * geo_esc_v)
            if geo_peace_v > 0.0 and sym in (GEO_BLACKSEA_RECON_STOCKS | GEO_AVIATION_STOCKS):
                bull_val = max(bull_val, 0.85 * geo_peace_v)

            if fon_val > 0 or bofa_val > 0 or kamu_val > 0 or bull_val > 0:
                daily_records.append({
                    "symbol": sym,
                    "trade_date": eff_date,
                    "fon_krizi_raw": fon_val,
                    "bofa_sell_raw": bofa_val,
                    "kamu_floor_raw": kamu_val,
                    "bullish_catalyst_raw": bull_val,
                    "article_count": 1,
                })

    df_daily_raw = pd.DataFrame(daily_records)
    if not df_daily_raw.empty:
        def saturating_agg(s: pd.Series) -> float:
            arr = s.to_numpy(float)
            if len(arr) == 0 or np.max(arr) <= 0.0:
                return 0.0
            combined = 1.0 - float(np.prod(1.0 - np.clip(arr * 0.85, 0.0, 0.98)))
            return round(float(np.clip(max(float(np.max(arr)), combined), 0.0, 1.0)), 4)

        df_daily_agg = (
            df_daily_raw.groupby(["symbol", "trade_date"], as_index=False)
            .agg(
                fon_krizi_news_intensity=("fon_krizi_raw", saturating_agg),
                bofa_sell_news_intensity=("bofa_sell_raw", saturating_agg),
                kamu_floor_news_intensity=("kamu_floor_raw", saturating_agg),
                bullish_catalyst_intensity=("bullish_catalyst_raw", saturating_agg),
                news_article_count=("article_count", "sum"),
            )
        )
    else:
        df_daily_agg = pd.DataFrame(
            columns=[
                "symbol",
                "trade_date",
                "fon_krizi_news_intensity",
                "bofa_sell_news_intensity",
                "kamu_floor_news_intensity",
                "bullish_catalyst_intensity",
                "news_article_count",
            ]
        )

    # 4. Merge onto the full `(symbol, trade_date)` grid from `three_forces_daily_clash.parquet`
    df_merged = df_clash.sort_values(["symbol", "trade_date"]).reset_index(drop=True).copy()
    df_merged["trade_date"] = df_merged["trade_date"].astype(str).str.slice(0, 10)
    df_merged = df_merged.merge(df_daily_agg, on=["symbol", "trade_date"], how="left")

    intensity_cols = [
        "fon_krizi_news_intensity",
        "bofa_sell_news_intensity",
        "kamu_floor_news_intensity",
        "bullish_catalyst_intensity",
    ]
    for col in intensity_cols:
        df_merged[col] = df_merged[col].fillna(0.0).astype(float)
    df_merged["news_article_count"] = df_merged["news_article_count"].fillna(0).astype(int)

    # 5. Strictly causal 5-session and 10-session EWM news memory per symbol (ordered by trade_date)
    #    Uses ONLY articles with effective_trade_date <= trade_date (100% zero lookahead).
    ewm_prefixes = {
        "fon_krizi_news_intensity": "fon_krizi",
        "bofa_sell_news_intensity": "bofa_sell",
        "kamu_floor_news_intensity": "kamu_floor",
        "bullish_catalyst_intensity": "bullish_catalyst",
    }
    for src_col, prefix in ewm_prefixes.items():
        df_merged[f"{prefix}_ewm5"] = (
            df_merged.groupby("symbol", group_keys=False)[src_col]
            .transform(lambda s: s.ewm(halflife=5, adjust=False).mean().round(4))
        )
        df_merged[f"{prefix}_ewm10"] = (
            df_merged.groupby("symbol", group_keys=False)[src_col]
            .transform(lambda s: s.ewm(halflife=10, adjust=False).mean().round(4))
        )

    # 6. Compute composite agentic news overhang & synergy with 3-Forces clash
    trap_flag = df_merged["is_passive_absorption_trap"].to_numpy(bool)
    fon_m5 = df_merged["fon_krizi_ewm5"].to_numpy(float)
    bofa_m5 = df_merged["bofa_sell_ewm5"].to_numpy(float)
    kamu_m5 = df_merged["kamu_floor_ewm5"].to_numpy(float)
    bull_m5 = df_merged["bullish_catalyst_ewm5"].to_numpy(float)
    clash_sc = df_merged["three_forces_clash_score"].to_numpy(float)

    kamu_sign = np.where(trap_flag | (clash_sc < -0.35), -0.50, 0.25)
    net_news = np.clip(
        0.75 * bull_m5 - 0.85 * fon_m5 - 0.75 * bofa_m5 + kamu_sign * kamu_m5,
        -1.0,
        1.0,
    )
    df_merged["agentic_net_news_sentiment"] = np.round(net_news, 4)

    regime_composite = np.clip(
        0.65 * clash_sc + 0.85 * net_news - np.where(trap_flag & (fon_m5 > 0.15), 0.35, 0.0),
        -2.0,
        2.0,
    )
    df_merged["agentic_3forces_news_composite"] = np.round(regime_composite, 4)

    df_merged["agentic_news_regime"] = df_merged.apply(classify_daily_regime, axis=1)

    return df_merged, df_scored_news


def build_summary_payload(df_regimes: pd.DataFrame, df_scored_news: pd.DataFrame) -> dict[str, Any]:
    """Generate a structured human-readable JSON summary of the Sept-Oct 2026 Fon Krizi & ASELS 3-Forces battle."""
    sep_oct_news = df_scored_news[
        (df_scored_news["effective_trade_date"] >= "2026-09-14")
        & (df_scored_news["effective_trade_date"] <= "2026-10-23")
    ].copy()

    asels_window = df_regimes[
        (df_regimes["symbol"] == "ASELS")
        & (df_regimes["trade_date"] >= "2026-09-14")
        & (df_regimes["trade_date"] <= "2026-10-23")
    ].sort_values("trade_date")

    regime_counts_all = df_regimes["agentic_news_regime"].value_counts().to_dict()
    regime_counts_sep_oct = (
        df_regimes[
            (df_regimes["trade_date"] >= "2026-09-14")
            & (df_regimes["trade_date"] <= "2026-10-23")
        ]["agentic_news_regime"]
        .value_counts()
        .to_dict()
    )
    asels_regime_counts = asels_window["agentic_news_regime"].value_counts().to_dict()

    asels_timeline: list[dict[str, Any]] = []
    for r in asels_window.itertuples(index=False):
        asels_timeline.append({
            "trade_date": r.trade_date,
            "close_price": float(r.close_price),
            "mlb_5d_share_pct": float(r.mlb_5d_share_pct),
            "mlb_10d_share_pct": float(r.mlb_10d_share_pct),
            "kamu_5d_share_pct": float(r.kamu_5d_share_pct),
            "big5_5d_share_pct": float(r.big5_5d_share_pct),
            "mlb_cost_spread_pct": float(r.mlb_cost_spread_pct),
            "big5_cost_spread_pct": float(r.big5_cost_spread_pct),
            "is_passive_absorption_trap": bool(r.is_passive_absorption_trap),
            "three_forces_clash_score": float(r.three_forces_clash_score),
            "fon_krizi_news_intensity": float(r.fon_krizi_news_intensity),
            "bofa_sell_news_intensity": float(r.bofa_sell_news_intensity),
            "kamu_floor_news_intensity": float(r.kamu_floor_news_intensity),
            "bullish_catalyst_intensity": float(r.bullish_catalyst_intensity),
            "fon_krizi_ewm5": float(r.fon_krizi_ewm5),
            "bofa_sell_ewm5": float(r.bofa_sell_ewm5),
            "kamu_floor_ewm5": float(r.kamu_floor_ewm5),
            "bullish_catalyst_ewm5": float(r.bullish_catalyst_ewm5),
            "agentic_net_news_sentiment": float(r.agentic_net_news_sentiment),
            "agentic_3forces_news_composite": float(r.agentic_3forces_news_composite),
            "agentic_news_regime": str(r.agentic_news_regime),
        })

    top_crisis_headlines = (
        sep_oct_news.sort_values(
            ["fon_krizi_news_intensity", "bofa_sell_news_intensity", "effective_trade_date"],
            ascending=[False, False, True],
        )
        .head(25)[
            [
                "effective_trade_date",
                "published_at_trt",
                "scope",
                "category",
                "fon_krizi_news_intensity",
                "bofa_sell_news_intensity",
                "kamu_floor_news_intensity",
                "bullish_catalyst_intensity",
                "title",
            ]
        ]
        .to_dict(orient="records")
    )

    start_price = float(asels_window.iloc[0]["close_price"]) if not asels_window.empty else 0.0
    end_price = float(asels_window.iloc[-1]["close_price"]) if not asels_window.empty else 0.0
    min_price = float(asels_window["close_price"].min()) if not asels_window.empty else 0.0
    max_price = float(asels_window["close_price"].max()) if not asels_window.empty else 0.0

    return {
        "metadata": {
            "total_news_headlines_scored": int(len(df_scored_news)),
            "sept_oct_2026_headlines_count": int(len(sep_oct_news)),
            "total_stock_session_rows": int(len(df_regimes)),
            "focus_stocks": FOCUS_STOCKS,
            "zero_leakage_verified": True,
            "session_cutoff_trt": "18:10:00 Europe/Istanbul",
            "ewm_halflives_sessions": [5, 10],
        },
        "regime_distribution_all_history": regime_counts_all,
        "regime_distribution_2026_09_14_to_2026_10_23": regime_counts_sep_oct,
        "asels_battle_summary_2026_09_14_to_2026_10_23": {
            "window_sessions": int(len(asels_window)),
            "price_start_tl": start_price,
            "price_end_tl": end_price,
            "price_high_tl": max_price,
            "price_low_tl": min_price,
            "cumulative_return_pct": round((end_price / max(start_price, 1e-9) - 1.0) * 100.0, 2),
            "max_drawdown_from_window_high_pct": round((min_price / max(max_price, 1e-9) - 1.0) * 100.0, 2),
            "passive_absorption_trap_sessions": int(asels_window["is_passive_absorption_trap"].sum()),
            "passive_absorption_trap_share_pct": round(
                float(asels_window["is_passive_absorption_trap"].mean()) * 100.0, 1
            ),
            "min_mlb_5d_share_pct": float(asels_window["mlb_5d_share_pct"].min()),
            "min_mlb_10d_share_pct": float(asels_window["mlb_10d_share_pct"].min()),
            "max_kamu_5d_share_pct": float(asels_window["kamu_5d_share_pct"].max()),
            "max_big5_5d_share_pct": float(asels_window["big5_5d_share_pct"].max()),
            "min_mlb_cost_spread_pct": float(asels_window["mlb_cost_spread_pct"].min()),
            "peak_fon_krizi_ewm5": float(asels_window["fon_krizi_ewm5"].max()),
            "peak_bofa_sell_ewm5": float(asels_window["bofa_sell_ewm5"].max()),
            "peak_kamu_floor_ewm5": float(asels_window["kamu_floor_ewm5"].max()),
            "regime_breakdown": asels_regime_counts,
        },
        "top_sept_oct_2026_crisis_headlines": top_crisis_headlines,
        "asels_daily_ledger_2026_09_14_to_2026_10_23": asels_timeline,
    }


REGIME_TABLE_COLS = [
    "symbol", "trade_date", "agentic_news_regime", "is_passive_absorption_trap", "three_forces_clash_score",
    "mlb_5d_share_pct", "mlb_10d_share_pct", "kamu_5d_share_pct", "big5_5d_share_pct", "mlb_cost_spread_pct",
    "big5_cost_spread_pct", "fon_krizi_ewm5", "bofa_sell_ewm5", "kamu_floor_ewm5", "bullish_catalyst_ewm5",
    "news_article_count",
]


def write_regime_table(db: Any, df_regimes: pd.DataFrame) -> None:
    conn = db.get_connection()
    rows = df_regimes[REGIME_TABLE_COLS].astype(object).values.tolist()
    with conn.cursor() as cur:
        cur.execute("TRUNCATE silver_news_regimes_daily")
        placeholders = ", ".join(["%s"] * len(REGIME_TABLE_COLS))
        cur.executemany(
            f"INSERT INTO silver_news_regimes_daily ({', '.join(REGIME_TABLE_COLS)}) VALUES ({placeholders})", rows
        )


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    db = PostgresManager()
    ensure_news_tables(db)
    df_regimes, df_scored_news = build_agentic_news_regimes(db)
    write_regime_table(db, df_regimes)

    out_parquet = OUT_DIR / "agentic_news_regimes.parquet"
    out_json = OUT_DIR / "agentic_news_regimes_summary.json"

    df_regimes.to_parquet(out_parquet, index=False)
    summary = build_summary_payload(df_regimes, df_scored_news)
    out_json.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"Saved {len(df_regimes)} rows ({len(df_regimes.columns)} cols) -> {out_parquet}")
    print(f"Saved summary JSON -> {out_json}")


if __name__ == "__main__":
    main()
