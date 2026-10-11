#!/usr/bin/env python3
"""Offline check of robot_proc.engaged_gains: stiffness scaling by default, explicit per-joint kp/kd override, SDK range."""
import sys
from pathlib import Path
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parent))
from robot_proc import engaged_gains,DEFAULT_KP,DEFAULT_KD,gains_label

KP=[30,30,30,20,5,5];KD=[1,1,1,0.5,0.5,0.5]

def main():
    kp,kd=engaged_gains(KP,KD)                                   # default since 2026-10-09: dimos gains
    assert np.allclose(kp,[80,80,80,50,20,20]) and np.allclose(kd,[3,3,3,0.7,0.4,0.4]) and kp.tolist()==DEFAULT_KP and kd.tolist()==DEFAULT_KD
    kp,kd=engaged_gains(KP,KD,1.5)                               # legacy stiffness: vendor kp x 1.5, vendor kd
    assert np.allclose(kp,[45,45,45,30,7.5,7.5]) and np.allclose(kd,KD)
    kp,kd=engaged_gains(KP,KD,kd=[2]*6);assert np.allclose(kp,DEFAULT_KP) and np.allclose(kd,[2]*6)
    assert 'dimos' in gains_label() and '150%' in gains_label(1.5) and '80' in gains_label(None,[80]*6,None)
    kp,kd=engaged_gains(KP,KD,1.5,kp=[80,80,80,50,20,20],kd=[3,3,3,0.7,0.4,0.4])
    assert np.allclose(kp,[80,80,80,50,20,20]) and np.allclose(kd,[3,3,3,0.7,0.4,0.4])  # stiffness ignored with explicit kp
    kp,kd=engaged_gains(KP,KD,2.0,kd=[2,2,2,1,1,1])
    assert np.allclose(kp,[60,60,60,40,10,10]) and np.allclose(kd,[2,2,2,1,1,1])
    for bad in (dict(kp=[201]*6),dict(kd=[5.1]*6),dict(kp=[1]*5),dict(kd=[-1]*6)):
        try:engaged_gains(KP,KD,1.0,**bad);raise AssertionError(bad)
        except ValueError:pass
    print('gains OK')

if __name__=='__main__':main()
