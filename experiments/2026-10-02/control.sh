#!/bin/bash
# Control-layer test, no model: control.sh [stiffness, default 1.0] [integral gain 1/s, default 0 = off]
exec "$HOME/a1z-infer/eval.sh" --control --stiffness "${1:-1.0}" --ki "${2:-0}"
