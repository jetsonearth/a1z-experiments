#!/bin/bash
# Opens the A1Z Console in the browser: starts the server on the Jetson if needed and tunnels it to this Mac
# (the server listens on the Jetson's 127.0.0.1 only).
PORT=8780
ssh jetson-direct "curl -s -o /dev/null http://127.0.0.1:$PORT/api/usb || (cd ~/a1z-experiments/2026-10-02 && setsid nohup ~/a1z-infer/run.sh console/server.py > /tmp/a1z_console.log 2>&1 < /dev/null &)" || { echo "连不上 Jetson（ssh jetson-direct）"; read -p "按回车关闭"; exit 1; }
lsof -iTCP:$PORT -sTCP:LISTEN >/dev/null 2>&1 || ssh -f -N -o ExitOnForwardFailure=yes -L $PORT:127.0.0.1:$PORT jetson-direct
sleep 2
open "http://localhost:$PORT/#/run"
