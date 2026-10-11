#!/bin/bash
# HIL-SERL actor on the Jetson, in tmux (survives SSH drops; rerun to reattach).
#   hilserl.sh <snap|check|demo|train|eval> [args]      e.g. hilserl.sh check --crop 200 120 240 240
# train needs the learner running and tunnel.sh open on the Mac.
ARGS=$(printf ' %q' "$@")
EXP='cd $HOME/a1z-experiments/2026-10-09-hilserl && $HOME/a1z-infer/run.sh actor.py'
if ! tmux has-session -t hil 2>/dev/null; then
  tmux new -d -s hil -n actor /bin/bash --noprofile --norc -c "$EXP$ARGS; echo; read -p '按回车关闭'; tmux kill-session -t hil"
fi
exec tmux attach -t hil:actor
