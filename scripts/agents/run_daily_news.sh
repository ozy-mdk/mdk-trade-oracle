#!/usr/bin/env bash
# Daily agentic news job: harvest, score, then label new live headlines in shadow mode.
# A labeling failure is reported in the exit code, but harvested data is already stored by then.
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$PROJECT_ROOT"
export PYTHONPATH="$PROJECT_ROOT/src:$PROJECT_ROOT"
export PATH="$HOME/.local/bin:/opt/homebrew/bin:/usr/local/bin:$PATH"

echo "=== $(TZ=Europe/Istanbul date '+%Y-%m-%d %H:%M:%S') TRT ==="
.venv/bin/python -W ignore scripts/agents/harvest_3forces_news.py
.venv/bin/python -W ignore scripts/agents/score_agentic_news_regimes.py
.venv/bin/python -W ignore scripts/agents/label_news_shadow.py --live-only
