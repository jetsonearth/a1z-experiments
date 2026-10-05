#!/usr/bin/env python3
"""Offline check of robot_proc.integral_step on a one-joint stick-slip model (no hardware): with the vendor PD alone the
joint stalls ~Fs/kp short of its command; the bounded integral removes most of that error and never exceeds its clamp."""
import sys
from pathlib import Path
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parent))
from robot_proc import integral_step


def simulate(ki,kp=30.0,fs=0.6,fc=0.45,target=np.radians(2.0),secs=4.0,dt=0.01,max_deg=3.0):
    """Quasi-static joint: sticks while |torque| <= fs (N m); when it breaks away it slides to where torque = fc."""
    q=np.zeros(1);off=np.zeros(1);offs=[]
    for _ in range(int(secs/dt)):
        cmd=np.array([target])
        if ki>0:off=integral_step(off,cmd,q,ki,dt,np.radians(max_deg))
        tau=kp*(cmd+off-q)
        if abs(tau[0])>fs:q=cmd+off-np.sign(tau)*fc/kp
        offs.append(off[0])
    return float(np.degrees(target-q[0])),float(np.degrees(np.abs(offs).max()))


def main():
    pd,_=simulate(0.0);pi,peak=simulate(2.0)
    assert 0.8<pd<1.2,pd                 # vendor PD alone: ~fc/kp = 0.86 deg short (measured on the arm: 1-2 deg)
    assert abs(pi)<0.3,pi                # with the integral: error mostly removed
    assert peak<=3.0+1e-9,peak           # clamp holds
    blocked,peak2=simulate(2.0,fs=1e9)   # joint blocked (contact): offset saturates at the clamp, no runaway
    assert abs(peak2-3.0)<1e-6,peak2
    print(f'integral OK: PD-only error {pd:.2f} deg -> with integral {pi:.2f} deg; blocked joint offset capped at {peak2:.1f} deg')

if __name__=='__main__':main()
