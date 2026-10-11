#!/bin/bash
# Control-layer test, no model: control.sh [stiffness] [integral gain 1/s]
# Defaults: dimos kp/kd, integral 2/s capped at 1 deg (2026-10-09). A stiffness argument selects the legacy
# vendor-kp x stiffness gains; pass 0 as the second argument to turn the integral off.
exec "$HOME/a1z-infer/eval.sh" --control ${1:+--stiffness "$1"} ${2:+--ki "$2"}
