#!/usr/bin/env python3
"""Offline check of robot_proc.grip_ramp: hold->full rises over GRIP_RAMP_S at the 100 Hz supervisor rate, monotonically,
without overshoot; the first step after engage is small (the 2026-10-05 hum came from a single hold->full step)."""
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parent))
from robot_proc import grip_force_levels,grip_ramp,GRIP_RAMP_S


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
    # live force change (2026-10-09): 4 -> 6 N m ramps up from the current level; hold stays capped at full; SDK clip at peak
    f6,h6=grip_force_levels(6.0,1.5,peak);assert abs(f6*peak-6.0)<1e-9 and abs(h6*peak-1.5)<1e-9
    cur=4.0/peak;seq2=[]
    while cur!=f6 and len(seq2)<1000:cur=grip_ramp(cur,f6,f6-h6,dt);seq2.append(cur)
    assert all(b>a for a,b in zip(seq2,seq2[1:])) and seq2[-1]==f6 and len(seq2)*dt<=GRIP_RAMP_S+2*dt
    f1,h1=grip_force_levels(1.0,1.5,peak);assert h1==f1                  # hold never above full
    assert grip_force_levels(20.0,None,peak)==(1.0,None)                  # clipped at the motor peak, single level
    assert grip_ramp(f6,4.0/peak,f6-h6,dt)<f6                             # lowering also ramps (no step to a lower limit)
    print(f'grip ramp OK: {hold*peak:.2f} -> {full*peak:.2f} N m in {t:.2f} s, first step {(seq[0]-hold)*peak:.3f} N m')

if __name__=='__main__':main()
