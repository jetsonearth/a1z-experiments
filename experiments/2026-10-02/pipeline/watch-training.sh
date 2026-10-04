#!/usr/bin/env bash
# Live view of the cloud ACT training log. Ctrl-C only stops watching; training keeps running.
INSTANCE=${1:-54016679}
VAST=~/a1z-experiments/.venv-vast/bin/vastai
while true; do
  read -r IP PORT STATUS < <($VAST show instance "$INSTANCE" --raw | python3 -c "
import sys,json;d=json.load(sys.stdin);p=(d.get('ports') or {}).get('22/tcp') or [{}]
print(d.get('public_ipaddr') or '-',p[0].get('HostPort') or '-',d.get('actual_status'))")
  if [ "$PORT" != "-" ] && ssh -q -o ConnectTimeout=5 -o StrictHostKeyChecking=accept-new -p "$PORT" root@"$IP" test -f /workspace/train.log; then
    echo "每 100 步一行：step 步数 · epch 已过几遍数据 · loss 误差（越低越好）"
    exec ssh -o StrictHostKeyChecking=accept-new -p "$PORT" root@"$IP" "tail -c 200000 -f /workspace/train.log | stdbuf -oL tr '\\r' '\\n' | grep --line-buffered -oE 'step:.*loss:[0-9.]+|Training: +[0-9]+%.*step/s' | awk '/step:/{print; fflush(); next} NR%300==0{print; fflush()}'"
  fi
  echo "$(date +%H:%M:%S) 机器状态：$STATUS，训练还没开始，10 秒后重试…"
  sleep 10
done
