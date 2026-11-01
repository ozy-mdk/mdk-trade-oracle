---
name: mdk-3forces-news-agent
description: Work on the BIST Three-Forces flow rules, point-in-time news store, keyword regimes and shadow-mode LLM headline labels. Use when running or changing the news pipeline, the forecast gate, LLM providers, or the news panel.
---

# MDK Three-Forces and News

Read `docs/AGENTIC_DECISIONS.md` before changing anything here, and add an entry in the same change as your code. `docs/AGENTIC_3FORCES_NEWS_ARCHITECTURE.md` explains how it works.

## Rules

- Never rewrite stored headline timing. `bronze_news_headlines` is append-only.
- Never quote results from the last 90 sessions as out-of-sample for the gate.
- Evaluate news value on `LIVE` headlines only, and against always-UP and always-DOWN on the same sessions.
- LLM labels must not reach a forecast until `docs/AGENTIC_DECISIONS.md` D-009 is met.
- Do not extend the flow rules beyond the 6 focus stocks without new evidence (D-004).
- Keep provider details in `config/agents.yaml` and prompts in `config/agents/prompts/`.
- Keep keyword lists general. Do not add fragments of specific headlines.

## Commands

```bash
PYTHONPATH=. .venv/bin/python scripts/agents/harvest_3forces_news.py
PYTHONPATH=. .venv/bin/python scripts/agents/score_agentic_news_regimes.py
.venv/bin/python scripts/agents/label_news_shadow.py --limit 5
.venv/bin/python scripts/agents/compare_news_labels.py
PYTHONPATH=. .venv/bin/python scripts/experiments/experiment_three_forces_rule_audit.py
.venv/bin/pytest tests/test_agentic_news.py
```
