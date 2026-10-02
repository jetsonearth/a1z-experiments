#!/bin/bash
cd "$(dirname "$0")" || exit 1
export A1Z_STUDIO_HARDWARE=1
export A1Z_CAMERA=1
exec python3 studio/server.py
