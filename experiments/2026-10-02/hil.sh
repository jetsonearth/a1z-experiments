#!/bin/bash
# HG-DAgger collection shortcut: hil.sh [epoch] [stiffness]
# Defaults: epoch 1, default gains (dimos kp/kd + integral, 2026-10-09). A stiffness argument selects the legacy
# vendor-kp x stiffness gains (the HIL rounds of 2026-10-04/05 used 1.5).
exec "$HOME/a1z-infer/eval.sh" --hil --epoch "${1:-1}" ${2:+--stiffness "$2"}
