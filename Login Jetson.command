#!/bin/bash
cd "$(dirname "$0")/experiments/2026-10-02" || exit 1
exec python3 login_jetson.py
