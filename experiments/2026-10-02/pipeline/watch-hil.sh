#!/usr/bin/env bash
# Progress of the HG-DAgger runs on the rented GPU, refreshed every 30 s. Ctrl-C only stops watching.
INSTANCE=${1:-54147414}
VAST=~/a1z-experiments/.venv-vast/bin/vastai
while true; do
  read -r IP PORT STATUS < <($VAST show instance "$INSTANCE" --raw | python3 -c "
import sys,json;d=json.load(sys.stdin);p=(d.get('ports') or {}).get('22/tcp') or [{}]
print(d.get('public_ipaddr') or '-',p[0].get('HostPort') or '-',d.get('actual_status'))")
  clear; echo "$(date +%H:%M:%S)  机器：$STATUS   （loss 越低越好；训完机器会自动停机）"
  if [ "$STATUS" = running ]; then
    ssh -q -o ConnectTimeout=8 -p "$PORT" root@"$IP" '
      for r in "ft:A 微调（从 060000 接着训）" "scratch:B 从头训"; do f=${r%%:*}; echo; echo "${r#*:}"
        tr "\r" "\n" < /workspace/train-$f.log | grep -E "[0-9]+/[0-9]+ \[" | tail -1 | sed -E "s/.*\| *([0-9]+\/[0-9]+) \[([^<]*)<([^,]*),.*/  进度 \1 步   已用 \2   还剩 \3/"
        tr "\r" "\n" < /workspace/train-$f.log | grep -oE "loss:[0-9.]+" | tail -1 | sed "s/^/  /"
        ls /workspace/runs/hil-e1-$f/checkpoints 2>/dev/null | grep -v last | tr "\n" " " | sed "s/^/  已存 checkpoint：/"; echo
      done'
  fi
  sleep 30
done
