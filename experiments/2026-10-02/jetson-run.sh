#!/bin/bash
# Inference venv launcher: Jetson torch needs pip-provided cuDSS / cuSPARSELt.
SP=$HOME/a1z-infer/.venv/lib/python3.10/site-packages/nvidia
export LD_LIBRARY_PATH=/usr/local/cuda/lib64:$SP/cu12/lib:$SP/cusparselt/lib:${LD_LIBRARY_PATH}
exec $HOME/a1z-infer/.venv/bin/python "$@"
