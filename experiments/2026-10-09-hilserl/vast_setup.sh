#!/bin/bash
# On the GPU machine, once (after sync.sh vast): a venv with LeRobot 0.4.4, the same version as the Jetson, and a
# grpcio new enough for LeRobot's generated gRPC code (>= 1.73.1). Then start the learner in tmux:
#   bash ~/a1z-hilserl/2026-10-09-hilserl/vast_setup.sh
#   cd ~/a1z-hilserl/2026-10-09-hilserl && tmux new -s learner '../.venv/bin/python learner.py --out runs/$(date +%Y%m%d-%H%M%S); read'
set -euo pipefail
cd ~/a1z-hilserl
PY=$(command -v python3.12 || command -v python3.11 || command -v python3.10 || command -v python3)
"$PY" -c 'import sys; assert sys.version_info>=(3,10), "LeRobot 0.4.4 needs Python >= 3.10"'
[ -d .venv ] || "$PY" -m venv .venv
.venv/bin/pip install -q -U pip
.venv/bin/pip install -q "lerobot==0.4.4" "grpcio>=1.73.1" safetensors
.venv/bin/python - <<'EOF'
import torch, grpc, lerobot
print('torch', torch.__version__, 'cuda', torch.cuda.is_available(), torch.cuda.get_device_name(0) if torch.cuda.is_available() else '-')
print('grpc', grpc.__version__)
from lerobot.transport import services_pb2_grpc  # fails here if grpcio is too old
import sys; sys.path.insert(0, '2026-10-09-hilserl')
import common; p = common.make_policy('cuda' if torch.cuda.is_available() else 'cpu'); print('policy ok', sum(x.numel() for x in p.parameters()), 'params')
EOF
