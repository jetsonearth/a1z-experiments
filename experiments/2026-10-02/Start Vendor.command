#!/bin/bash
cd "$(dirname "$0")" || exit 1
export A1Z_MODE=vendor
exec python3 mac_console.py
