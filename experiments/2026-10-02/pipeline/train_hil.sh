#!/usr/bin/env bash
# HG-DAgger "train pi_N(i+1) on D" step, on the rented GPU (same LeRobot 0.4.4 lerobot-train as act-v2).
#   train_hil.sh <aggregate dataset dir> <output dir> <init checkpoint pretrained_model dir>
# MODE=finetune (default): start from the previous novice's weights; the normalizer stats are replaced by the
#   aggregate's stats (lerobot-train does this whenever --policy.path is given without --resume).
# MODE=scratch: same recipe as act-v2 (fresh ACT, ImageNet backbone) on the aggregate; the paper's literal reading.
# Everything else (batch 32, lr 1e-5, chunk 100, no augmentation) matches act-v2.
set -euo pipefail
DATA=${1:?aggregate dataset dir}; OUT=${2:?new output dir}; INIT=${3:-}
MODE=${MODE:-finetune}; STEPS=${STEPS:-$([ "$MODE" = scratch ] && echo 60000 || echo 20000)}
COMMON=(--dataset.repo_id=local/a1z_usb_aggregate --dataset.root="$DATA" --policy.device=cuda --policy.push_to_hub=false
        --output_dir="$OUT" --job_name="$(basename "$OUT")" --batch_size=32 --steps="$STEPS" --save_freq=10000
        --log_freq=100 --num_workers=16 --wandb.enable=false --eval_freq=0)
if [ "$MODE" = scratch ]; then
  exec lerobot-train --policy.type=act "${COMMON[@]}"
else
  [ -n "$INIT" ] || { echo "finetune needs the previous novice's pretrained_model dir"; exit 2; }
  exec lerobot-train --policy.path="$INIT" "${COMMON[@]}"
fi
