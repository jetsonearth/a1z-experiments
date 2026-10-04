#!/bin/bash
# ACT live evaluation in tmux (survives SSH drops; rerun to reattach).
#   eval.sh               policy server + client (+ scripted spiral search after the policy stalls)
#   eval.sh --no-search   policy only
#   eval.sh --no-policy   drag-only check: robot process + client, no model
#   eval.sh --calib       port height calibration (drag mode, keys 1/2/3)
CKPT=${CKPT:-/home/dimos/a1z-infer/checkpoints/060000/pretrained_model}
EXP='cd $HOME/a1z-experiments/2026-10-02 && $HOME/a1z-infer/run.sh'
if ! tmux has-session -t eval 2>/dev/null; then
  if [ "$1" = "--calib" ]; then
    tmux new -d -s eval -n client /bin/bash --noprofile --norc -c "$EXP calib.py; echo; read -p '按回车关闭'; tmux kill-session -t eval"
  elif [ "$1" = "--no-policy" ]; then
    tmux new -d -s eval -n client /bin/bash --noprofile --norc -c "$EXP a1z_client.py $*; echo; read -p '按回车关闭'; tmux kill-session -t eval"
  else
    tmux new -d -s eval -n server /bin/bash --noprofile --norc -c "$EXP -m lerobot.async_inference.policy_server --host=127.0.0.1 --port=8080 --fps=30 --inference_latency=0.033 --obs_queue_timeout=2 2>&1 | tee /tmp/policy_server.log"
    sleep 6
    tmux new-window -t eval -n client /bin/bash --noprofile --norc -c "$EXP a1z_client.py --checkpoint $CKPT $*; echo; read -p '按回车关闭'; tmux kill-session -t eval"
  fi
fi
exec tmux attach -t eval:client
