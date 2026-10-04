#!/bin/bash
cd "$(dirname "$0")" || exit 1
export A1Z_MODE=task
exec python3 mac_console.py
