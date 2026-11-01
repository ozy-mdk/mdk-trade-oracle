# Agentic Layer: Decision Log and Handoff

Read this first if you are picking up this work, with any tool or agent. Add a new entry in the same change as the code it describes. Do not rewrite old entries; supersede them with a new one.

## Current state (2026-11-01)

| Part | Status | Where |
|---|---|---|
| Headline store | Append-only; runs when the daily job is run | `bronze_news_headlines`, `data/bronze/news.py` |
| Keyword scoring and regimes | Research only | `scripts/agents/score_agentic_news_regimes.py`, `silver_news_regimes_daily` |
| LLM labels | Shadow mode: stored, never used by a forecast | `agents/news_classifier.py`, `silver_news_llm_labels`, `agent_llm_runs` |
| Forecast gate | `flow_only` by default, for the 6 focus stocks | `models/three_forces_gate.py`, `config/agents.yaml` |
| API and UI | Live | `GET /api/v1/tertip/news-regime`, `frontend/src/components/NewsRegimePanel.tsx` |
| Daily schedule | **Not installed.** Run `scripts/agents/install_daily_schedule.sh` to enable it | Daily job section |
| Gold rebuild | **Not run.** Served forecasts still come from the bundle built before these changes | D-012 |

## Switching provider

Everything provider-specific is in `config/agents.yaml`. Prompts are in `config/agents/prompts/` and contain no provider assumptions.

1. Install and log in to the other CLI.
2. In `config/agents.yaml`, check that provider's `args`, output paths and model names against its real output. The `gemini_cli` entry is an untested template.
3. Run `.venv/bin/python scripts/agents/label_news_shadow.py --provider gemini_cli --limit 5`.
4. Compare with `.venv/bin/python scripts/agents/compare_news_labels.py`. Labels are keyed by provider and model, so nothing is overwritten.
5. Set `active_provider` and change that provider's `status` to `tested`.

A provider must accept the prompt on stdin and print JSON. `structured_output_path` is optional: without it, JSON is parsed from `text_output_path`, and `news_classifier.validate_labels` rejects any batch that does not contain exactly one valid result per headline.

## Daily job

`scripts/agents/run_daily_news.sh` runs harvest, scoring and live-only labeling. `scripts/agents/install_daily_schedule.sh` installs it as a macOS launchd job at 20:00 local time; `--remove` uninstalls it. Until it is installed, each missed day can lose headlines from the point-in-time record. The installer itself has not been run yet. Logs go to `<DATA_DIR>/logs/agentic_news_daily.log`.

Newer headlines stay stored but count toward no session until `silver_daily_stock_summary` contains that session, so market data must keep loading.

## Decisions

### D-001: There was no LLM in the original "agentic" system
**Finding.** Scoring was keyword rules in Python. The `three-forces-news-analyst` spec lived in a Gemini session folder outside the repo and no code called it. **Decision.** Say so plainly, and keep agent specs in the repo (`.agents/agents/`).

### D-002: The original EVAL-90 result is not out-of-sample
**Finding.** The rules and keywords were written while studying the ASELS crisis inside that window. The keyword lists contained literal fragments of headlines from it. **Decision.** Removed those fragments. Relabeled the window "last 90 sessions" with a warning in the script and UI. The earlier "ASELS 10D 50% to 83.3%" claim must not be quoted as a track record.

### D-003: The gate defaults to `flow_only`
**Evidence** (`scripts/experiments/experiment_agentic_3forces_news.py`, 6 focus stocks, hit rates):

| Horizon | PRE-90 off / flow / flow+news | Last 90 off / flow / flow+news |
|---|---|---|
| 5D | 52.50 / 52.67 / 52.67 | 61.11 / 61.67 / 62.22 |
| 10D | 54.21 / 54.82 / 54.74 | 64.44 / 65.00 / 65.37 |
| 15D | 60.65 / 61.20 / 61.20 | 71.48 / 71.85 / 71.67 |

**Decision.** Keyword news adds nothing measurable beyond flow where news can be tested. Only about 2% of stock-sessions before the last 90 sessions have any news. `flow_news` remains available in config.

### D-004: Do not extend the flow rules beyond the 6 focus stocks
**Evidence** (`scripts/experiments/experiment_three_forces_rule_audit.py`, 29 stocks, 2022-2026). A rule makes one constant call, so it is compared with the same call on all sessions of the same stocks (lift). Rows at different horizons reuse the same firing events.

| Split | Rule | Firings per horizon | Lift, 5D / 10D / 15D (pp) | Independent-call lift (pp) |
|---|---|---|---|---|
| Other 23 stocks, all dates | Bearish | 196 | +2.0 / -3.6 / -1.2 | +0.2 / -10.5 / -2.3 |
| Other 23 stocks, all dates | Bullish | 271-279 | +0.4 / -2.4 / +0.5 | +2.8 / +1.9 / +5.3 |
| Focus stocks before 2026 | Bearish | 20 | +21.4 / +18.7 / +26.2 | +14.7 / +33.7 / +27.9 |
| Focus stocks before 2026 | Bullish | 17 | -3.4 / +6.6 / +33.8 | +5.2 / +3.5 / +31.9 |

On the other 23 stocks, mean returns after a bearish firing were above those stocks' normal returns, and after a bullish firing were below them. **Decision.** The rules do not generalize. The focus-stock lift rests on 12-13 independent calls and those stocks were chosen before the rules were written, so it is at best a weak signal. `scripts/generate_timestral_gold_bundle.py` covers only those 6 stocks; the rules must not spread further without new evidence.

### D-005: Headlines are append-only with frozen timing
**Why.** The harvest used to overwrite its file with whatever Google returned that day. Google also turns exact timestamps into date-only placeholders after about 3 days, so an article's session could change between runs. **Decision.** `bronze_news_headlines` freezes publication time and precision at first sight. Later harvests update only `last_seen_at_trt`.

### D-006: Sessions are mapped when reading, not when storing
**Why.** Mapping at harvest needs a calendar that may not yet contain the next session, which silently dropped new articles. Articles older than the calendar also collapsed onto its first date. **Decision.** Keep the 18:10 TRT cutoff and T+1 shift for date-only items, and map in `load_headlines_with_sessions`. Articles before the calendar are excluded.

### D-007: `LIVE` and `BACKFILL` are separate
**Why.** Past headlines were retrieved with search queries written after the events, and Google's results favour recent items. **Decision.** A headline is `LIVE` only if first seen within 72 hours of publication. All headlines imported from the old parquet are `BACKFILL`. Any evaluation of news value must use `LIVE` only.

### D-008: LLM access runs through the CLI under the user's subscription
**Why.** A small daily job for personal use needs no API account. Docs checked 2026-11-01: subscription login works with the unmodified Claude Code binary, while third-party products may not route their users through plan credentials. The docs do not explicitly mention cron. **Decision.** Use `claude -p` through `config/agents.yaml`, and refuse to run if `ANTHROPIC_API_KEY` or `ANTHROPIC_AUTH_TOKEN` is set, because either would bill an API account. If LLM output ever serves other people, move to the API by adding an adapter.

**Watch.** The docs say `--bare` will become the default for `-p`, and bare mode ignores subscription login. The job would then fail with an auth error, which `agent_llm_runs` records.

**Measured.** One batch of 5 headlines on Haiku took about 6 seconds, 1,041 output tokens and 2,590 cache-write tokens. The `notional_cost_usd` column is the CLI's list-price estimate, not a charge.

### D-009: LLM labels stay in shadow mode until they earn a place
**Why.** A model may already know how past events ended, so backfilled labels can leak outcomes. **Decision.** Send headline text only, with no dates or prices. Review only after `news.min_live_sessions_for_review` (60) live sessions. To promote, labels must beat both keyword scoring and `flow_only` on `LIVE` data, and beat always-UP and always-DOWN on the same sessions.

### D-010: Daily stock narratives are deferred
**Why.** Explaining forecasts in prose would make an unproven signal look more credible. **Decision.** Build this only after D-009 has a result.

### D-011: Missing news must not read as calm
**Why.** A missing regime row defaulted news to 0, which satisfied the "no crisis" check in the bullish rule. **Decision.** The forecaster sets `news_available`, and in `flow_news` mode the bullish rule stays closed without news. The API and panel show a stale warning beyond `news.stale_after_sessions`.

### D-012: Production forecasts were left as they are
**Why.** The user asked not to spend effort on production because nobody is using it yet. **Decision.** Gold was not rebuilt or synced. The `flow_only` default and the rescored news take effect at the next run of `scripts/generate_timestral_gold_bundle.py`.

### D-013: Gate results are stored for display, not computed on request
**Decision.** `experiment_agentic_3forces_news.py` writes `gold_agentic_gate_audit`. The API computes always-UP and always-DOWN on the same sessions. Only the 6 focus stocks have this track, and the panel says so for others.

## Open items

- A Gemini CLI is not installed, so `gemini_cli` is unverified.
- `normalize_tr` does not fold all Turkish letters; aliases list both spellings where it matters.
- The same article found by two queries counts twice in the daily intensity.
- The `BIST30_STOCKS` list has 29 names.
