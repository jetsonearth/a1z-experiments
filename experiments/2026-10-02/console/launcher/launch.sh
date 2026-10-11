#!/bin/bash
# Open the local dashboard immediately; Jetson reconnects in the background.
export PATH=/opt/homebrew/bin:/usr/bin:/bin:/usr/sbin:/sbin
PORT=8780
URL="http://localhost:$PORT/#/run"
LOG="$HOME/Library/Logs/a1z-console-launcher.log"
SCRIPT="$HOME/a1z-experiments/experiments/2026-10-02/console/launcher/local_dashboard.py"
mkdir -p "$HOME/Library/Logs"
exec >>"$LOG" 2>&1
if ! curl -fsS -m 1 "http://127.0.0.1:$PORT/_local/health" | grep -q 'a1z-local-dashboard'; then
  nohup python3 "$SCRIPT" </dev/null >>"$LOG" 2>&1 &
  for _ in {1..30}; do
    curl -fsS -m 1 "http://127.0.0.1:$PORT/_local/health" | grep -q 'a1z-local-dashboard' && break
    sleep 0.1
  done
fi
open "$URL"
