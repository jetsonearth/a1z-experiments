#!/bin/bash
# Copy the HIL-SERL code from this Mac to the Jetson or the GPU machine. Run on the Mac.
#   sync.sh jetson                  -> ~/a1z-experiments/2026-10-09-hilserl on jetson-direct (JETSON=... to override)
#   sync.sh vast root@HOST PORT     -> ~/a1z-hilserl/{2026-10-02,2026-10-09-hilserl} on the Vast instance
# The Jetson's ~/a1z-experiments/2026-10-02 is never overwritten: the reused files are only compared (checksums) and
# any difference is listed, because the Jetson copy may carry fixes made there.
set -euo pipefail
HERE=$(cd "$(dirname "$0")" && pwd); PREV=$(cd "$HERE/../2026-10-02" && pwd)
CODE=(common.py env.py actor.py learner.py bc.py classifier.py side_camera.py side_preview.py usb_exact.py quick_calib.py test_hilserl.py hilserl.sh README.md setup-log.md)
REUSED=(rl_run.py rl_env.py rl_agent.py resnet10_hilserl.py resnet10_hilserl.safetensors a1z_robot.py robot_proc.py wrist_camera.py teach.py insertion.py usb_watch.py test_rl_env.py)
cd "$HERE"
case "${1:-}" in
  jetson)
    H=${JETSON:-jetson-direct}
    ssh "$H" mkdir -p a1z-experiments/2026-10-09-hilserl
    rsync -av "${CODE[@]}" "$H:a1z-experiments/2026-10-09-hilserl/"
    echo "== 复用文件：Jetson 上的版本和这台 Mac 比（只列差异，不复制）"
    (cd "$PREV" && rsync -nci "${REUSED[@]}" "$H:a1z-experiments/2026-10-02/") || true
    echo "（上面没有列出文件 = 两边一致）" ;;
  vast)
    [ $# -eq 3 ] || { echo "用法：sync.sh vast root@HOST PORT"; exit 1; }
    E="ssh -p $3"
    $E "$2" mkdir -p a1z-hilserl/2026-10-02 a1z-hilserl/2026-10-09-hilserl
    (cd "$PREV" && rsync -av -e "$E" rl_agent.py resnet10_hilserl.py resnet10_hilserl.safetensors "$2:a1z-hilserl/2026-10-02/")
    rsync -av -e "$E" common.py learner.py bc.py vast_setup.sh "$2:a1z-hilserl/2026-10-09-hilserl/" ;;
  *) sed -n 2,6p "$0"; exit 1 ;;
esac
