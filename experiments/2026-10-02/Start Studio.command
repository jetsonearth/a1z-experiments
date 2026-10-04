#!/bin/bash
cd "$(dirname "$0")" || exit 1
export A1Z_STUDIO_HARDWARE=1
export A1Z_CAMERA=1
# Reopening the launcher must never replace an existing capture process.
if python3 - <<'CHECK'
import json, os, urllib.request
try:
    port=int(os.environ.get('A1Z_STUDIO_PORT','8769'))
    s=json.load(urllib.request.urlopen(f'http://127.0.0.1:{port}/api/status', timeout=1))
    raise SystemExit(0 if 'hardware_enabled' in s and 'session_episodes' in s else 1)
except (OSError, ValueError):
    raise SystemExit(1)
CHECK
then
    echo 'A1Z 控制台已运行；请使用现有页面。'
    exit 0
fi
exec python3 studio/server.py
