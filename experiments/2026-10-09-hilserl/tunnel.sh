#!/bin/bash
# Run on the Mac and leave open during training: the Jetson's 127.0.0.1:50051 reaches the learner on the Vast machine.
#   Jetson actor -> (ssh -R over the cable) -> Mac 127.0.0.1:50051 -> (ssh -L) -> Vast 127.0.0.1:50051 learner
#   tunnel.sh root@HOST PORT          (the Vast instance's SSH address and port; JETSON=... overrides jetson-direct)
set -u
[ $# -eq 2 ] || { sed -n 2,4p "$0"; exit 1; }
P=${LEARNER_PORT:-50051}
OPTS=(-N -o ExitOnForwardFailure=yes -o ServerAliveInterval=15 -o ServerAliveCountMax=3)
ssh "${OPTS[@]}" -L "$P:127.0.0.1:$P" -p "$2" "$1" & A=$!
ssh "${OPTS[@]}" -R "$P:127.0.0.1:$P" "${JETSON:-jetson-direct}" & B=$!
trap 'kill $A $B 2>/dev/null' EXIT INT TERM
sleep 3
echo "隧道已开：Jetson 127.0.0.1:$P → Mac → Vast 127.0.0.1:$P。训练期间别关这个窗口，Ctrl+C 关闭。"
while kill -0 $A 2>/dev/null && kill -0 $B 2>/dev/null; do sleep 2; done
echo "⚠ 有一条隧道断了（Vast 或 Jetson 那一侧）。actor 会报连接出错并停下；重开本脚本后用 --resume 续训。"
