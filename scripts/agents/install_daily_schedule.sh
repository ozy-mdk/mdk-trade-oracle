#!/usr/bin/env bash
# Install or remove the daily news job as a macOS launchd agent (20:00 local time, after the BIST close).
#   scripts/agents/install_daily_schedule.sh
#   scripts/agents/install_daily_schedule.sh --remove
set -euo pipefail

LABEL="com.mdk-oracle.agentic-news-daily"
PLIST="$HOME/Library/LaunchAgents/$LABEL.plist"
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"

launchctl bootout "gui/$(id -u)/$LABEL" 2>/dev/null || true
if [ "${1:-}" = "--remove" ]; then
    rm -f "$PLIST"
    echo "Removed $LABEL"
    exit 0
fi

DATA_DIR="$(cd "$PROJECT_ROOT" && .venv/bin/python -W ignore -c 'from mdk_trading_oracle.core.config import get_settings; print(get_settings().data_dir)' 2>/dev/null || true)"
DATA_DIR="${DATA_DIR:-$HOME/data/mdk_oracle}"
LOG_DIR="$DATA_DIR/logs"
mkdir -p "$LOG_DIR" "$(dirname "$PLIST")"

cat > "$PLIST" <<EOF
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
  <key>Label</key><string>$LABEL</string>
  <key>ProgramArguments</key>
  <array>
    <string>/bin/bash</string>
    <string>$PROJECT_ROOT/scripts/agents/run_daily_news.sh</string>
  </array>
  <key>StartCalendarInterval</key>
  <dict><key>Hour</key><integer>20</integer><key>Minute</key><integer>0</integer></dict>
  <key>StandardOutPath</key><string>$LOG_DIR/agentic_news_daily.log</string>
  <key>StandardErrorPath</key><string>$LOG_DIR/agentic_news_daily.log</string>
</dict>
</plist>
EOF

launchctl bootstrap "gui/$(id -u)" "$PLIST"
echo "Installed $LABEL. Log: $LOG_DIR/agentic_news_daily.log"
