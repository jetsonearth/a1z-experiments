#!/bin/bash
# A1Z Console launcher (runs inside "A1Z Console.app"): starts the console server on the Jetson if needed,
# opens the SSH tunnel (the server listens on the Jetson's 127.0.0.1 only) and opens the browser. No Terminal window.
PORT=8780;URL="http://localhost:$PORT/#/run";LOG="$HOME/Library/Logs/a1z-console-launcher.log"
export PATH=/usr/bin:/bin:/usr/sbin:/sbin:/opt/homebrew/bin
exec >>"$LOG" 2>&1;echo "--- $(date)"
alert(){ osascript -e "display alert \"A1Z Console\" message \"$1\" as critical" >/dev/null; }
# Already tunnelled and answering: just open it.
if curl -s -m 2 -o /dev/null "http://localhost:$PORT/api/usb";then open "$URL";exit 0;fi
ssh -o BatchMode=yes -o ConnectTimeout=6 jetson-direct "curl -s -m 2 -o /dev/null http://127.0.0.1:$PORT/api/usb || (cd ~/a1z-experiments/2026-10-02 && setsid nohup ~/a1z-infer/run.sh console/server.py > /tmp/a1z_console.log 2>&1 < /dev/null &)" \
  || { alert "连不上 Jetson。请确认 Jetson 已开机、网线已连上，然后再点一次。";exit 1; }
# A stale tunnel process can hold the port without a live connection: replace it.
pkill -f "ssh -f -N .*-L $PORT:127.0.0.1:$PORT jetson-direct" 2>/dev/null
ssh -f -N -o ExitOnForwardFailure=yes -o ServerAliveInterval=15 -o ServerAliveCountMax=3 -L $PORT:127.0.0.1:$PORT jetson-direct \
  || { alert "SSH 隧道没建起来（本机 $PORT 端口可能被占用）。日志：$LOG";exit 1; }
for _ in $(seq 1 30);do curl -s -m 2 -o /dev/null "http://localhost:$PORT/api/usb" && { open "$URL";exit 0; };sleep 0.5;done
alert "控制台服务器没有响应。Jetson 上的日志：/tmp/a1z_console.log";exit 1
