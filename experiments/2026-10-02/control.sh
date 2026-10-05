#!/bin/bash
# Control-layer test, no model: control.sh [stiffness, default 1.0]
exec "$HOME/a1z-infer/eval.sh" --control --stiffness "${1:-1.0}"
