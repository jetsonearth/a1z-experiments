#!/usr/bin/env python3
"""Offline check of robot_proc.grip_ramp: hold->full rises over GRIP_RAMP_S at the 100 Hz supervisor rate, monotonically,
without overshoot; the first step after engage is small (the 2026-10-05 hum came from a single hold->full step)."""
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parent))
from robot_proc import grip_ramp,GRIP_RAMP_S


def main():
    peak=11.0;hold=1.5/peak;full=0.364;dt=0.01;cur=hold;seq=[]
    while cur!=full and len(seq)<1000:cur=grip_ramp(cur,full,full-hold,dt);seq.append(cur)
    t=len(seq)*dt
    assert abs(t-GRIP_RAMP_S)<=2*dt,t                      # reaches full in ~GRIP_RAMP_S
    assert all(b>=a for a,b in zip(seq,seq[1:])) and max(seq)==full  # monotone, no overshoot
    assert (seq[0]-hold)*peak<0.1,(seq[0]-hold)*peak       # first step < 0.1 N m (was ~2 N m)
    assert grip_ramp(full,full,full-hold,dt)==full          # at goal: unchanged
    assert grip_ramp(full,hold,full-hold,1.0)==hold         # long dt: lands exactly on goal
    assert grip_ramp(hold,hold,0.0,dt)==hold                # no span (hold>=full): unchanged
    print(f'grip ramp OK: {hold*peak:.2f} -> {full*peak:.2f} N m in {t:.2f} s, first step {(seq[0]-hold)*peak:.3f} N m')

if __name__=='__main__':main()
