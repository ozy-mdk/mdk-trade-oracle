---
name: mdk-3forces-news-agent
description: Zero-cost Agentic Market Intelligence & 3-Forces Microstructure Context Engine for Borsa Istanbul (BIST). Harvests point-in-time Turkish financial news (Google News TR RSS, KAP disclosures, SPK bulletins, Fon Krizi & ViOP margin call alerts) with strict 18:10 TRT session-close timestamp guards, combines them with the 3-Forces Clash Vector (1: MLB/BofA, 2: KAMU/Ziraat/Varlık Fonu, 3: BIG5/Domestic Funds), and conditions Timestral-209M + Policy C forecasts with zero data leakage.
---

# MDK 3-Forces & Point-in-Time News Intelligence Skill

## 1. Purpose & Domain Rationale
Borsa Istanbul (BIST) price action is governed by the interaction of **Three Institutional Forces**:
1. **Force 1 — `MLB` (Bank of America / Foreign Algorithmic Flow)**: High-velocity price-setting flow, intraday sweeps, and multi-day accumulation/distribution campaigns.
2. **Force 2 — `KAMU` (`ZRY` Ziraat Yatırım, `VKY` Vakıf, `HLY` Halk / Varlık Fonu / BES)**: State-aligned liquidity provider that acts in two distinct modes:
   - **Active Sovereign Catalyst Mode**:Genuine directional accumulation alongside `MLB` (bullish).
   - **Passive Floor-Defense Mode**: Emergency bid absorption during a systemic crisis (e.g., **Fon Krizi**, SPK fund liquidations, or ViOP margin-call cascades) while `MLB` aggressively unloads inventory.
3. **Force 3 — `BIG5` & Domestic Funds (`IYM`, `YKR`, `AKM`, `GRM`, plus local mutual/hedge funds)**: Domestic institutional inventory that carries deep unrealized gains during bull runs (`feat_big5_cost_spread_pct > +15%`) and suffers forced redemption/collateral selling (`Teminat Tamamlama`) during liquidity crises.

### Why Pure Price + Aggregate Flow Models Get Trapped Without This Skill
- In standard tabular models, `feat_total_inst_net_share_today = (mlb_flow + big5_flow + kamu_flow) / turnover`.
- When **`MLB` sells `-1,500M TL`** on `ASELS` while **`KAMU + BIG5` passively absorb `+1,800M TL`** on limit bids during a **Fon Krizi**, the sum looks *positive* (`+300M TL`) and `Timestral-209M` sees an oversold price dip—causing both models to falsely predict a bullish bounce (`+5% to +6%`) right into a multi-week liquidation overhang!
- This skill resolves that ambiguity by combining **Point-in-Time Web/RSS News Intelligence** with the **Endogenous 3-Forces Clash Vector**.

---

## 2. Zero-Cost Agentic Architecture ($0 External API Spend)

This skill uses a **Neuro-Symbolic / Tool-Grounded Multi-Agent Architecture** that runs entirely on your existing **Gemini subscription inside Antigravity** + free public Turkish feeds + local CPU/M5 Mac Pro:

1. **Tool 1 — Deterministic Point-in-Time News Harvester (`scripts/agents/harvest_3forces_news.py`)**:
   - Queries free **Google News TR RSS** (`hl=tr&gl=TR&ceid=TR:tr`) across Macro/Crisis (`FON_KRIZI`, `MARGIN_CALL`, `BOFA_FLOW`, `KAMU_DEFENSE`, `TCMB_MACRO`) and BIST 30 stock catalysts.
   - **Strict 18:10 TRT Session-Close Guard (`map_pubdate_to_trading_session`)**:
     Every `<pubDate>` (`RFC 2822`) is parsed into `Europe/Istanbul` (`TRT`). Any article published after `18:10 TRT` on day $T$ or on a weekend is strictly mapped to the **next** trading session $T+1$, guaranteeing **100% zero lookahead in historical backtests**.
2. **Tool 2 — Endogenous 3-Forces Clash Extractor (`extract_3forces_clash_features`)**:
   - Detects the **Passive Absorption Trap (`is_passive_absorption_trap`)** at session $T$ Close:
     - `mlb_5d_share_pct < -0.40%` and `mlb_10d_share_pct < -0.30%` (persistent `MLB` multi-day distribution)
     - `kamu_5d_share_pct + big5_5d_share_pct > +0.20%` (passive domestic bid absorption)
     - `mlb_cost_spread_pct < -1.0%` (`MLB` FIFO inventory underwater, driving stop/liquidation pressure)
3. **Subagent — `three-forces-news-analyst` (`scripts/agents/score_agentic_news_regimes.py`)**:
   - Scores point-in-time headlines into causal exponentially decaying regime intensities (`fon_krizi_news_ewm`, `bofa_sell_news_ewm`, `kamu_floor_news_ewm`, `bullish_catalyst_ewm`) and merges them with the 3-Forces Clash Vector into `~/data/mdk_oracle/experiments/timestral/agentic_news_regimes.parquet`.
4. **Quantitative Conditioning (`scripts/experiments/experiment_agentic_3forces_news.py` & `models/timestral_forecaster.py`)**:
   - Conditions Stage-1 directional conviction on the **3-Forces Clash + Crisis Regime**, selected strictly on `PRE-90` (`target_date_end < start_of_eval90`, zero peeking at `EVAL-90`).

---

## 3. Standard Execution Commands

```bash
# 1. Harvest point-in-time Google News TR RSS + extract 3-Forces Clash Vector
.venv/bin/python scripts/agents/harvest_3forces_news.py

# 2. Score point-in-time news regimes & merge with 3-Forces Clash table
.venv/bin/python scripts/agents/score_agentic_news_regimes.py

# 3. Run zero-leakage PRE-90 vs EVAL-90 walk-forward verification experiment
PYTHONPATH=. .venv/bin/python scripts/experiments/experiment_agentic_3forces_news.py
```

