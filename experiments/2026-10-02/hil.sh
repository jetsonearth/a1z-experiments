#!/bin/bash
# HG-DAgger collection shortcut: hil.sh [epoch] [stiffness]   (defaults: epoch 1, stiffness 1.5)
exec "$HOME/a1z-infer/eval.sh" --hil --epoch "${1:-1}" --stiffness "${2:-1.5}"
