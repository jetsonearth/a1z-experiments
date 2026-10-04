#!/bin/bash
# Shadow mode inside tmux: survives an SSH drop; rerun this script to reattach.
CKPT=${1:-/home/dimos/a1z-infer/checkpoints/060000/pretrained_model}
exec tmux new -A -s shadow /bin/bash --noprofile --norc -c "cd \$HOME/a1z-experiments/2026-10-02 && \$HOME/a1z-infer/run.sh shadow.py --checkpoint $CKPT; echo; read -p \"按回车关闭\""
