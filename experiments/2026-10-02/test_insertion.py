#!/usr/bin/env python3
"""Offline check of the scripted search/push stage against a toy contact model (runs where pinocchio is installed)."""
import math
import numpy as np
from insertion import Inserter,Kin,SearchConfig

def simulate(hole_offset_mm,seconds=30.0,dt=1/30):
    kin=Kin();q0=np.radians([7,134,-129,80,2,6])
    T0=kin.fk(q0);surface=T0[2,3];hole=T0[:3,3]+np.array([hole_offset_mm[0],hole_offset_mm[1],0])/1000
    cmd=kin.step_towards(q0,T0[:3,3]-[0,0,0.012],T0[:3,:3])  # policy presses ~12 mm below the surface
    ins=Inserter(kin,SearchConfig());q=q0.copy();t=0.0
    while t<seconds:
        stage,target=ins.update(t,q,cmd)
        if target is not None:cmd=target
        Tc=kin.fk(cmd);p=Tc[:3,3].copy()
        in_mouth=np.hypot(*(p[:2]-hole[:2]))<0.0005
        floor=surface-(0.012 if in_mouth else 0.0)
        p[2]=max(p[2],floor)
        q=kin.step_towards(q,p,Tc[:3,:3],iters=2)
        if stage in ('search_failed','push_failed') or (in_mouth and kin.fk(q)[2,3]<surface-0.010):break
        t+=dt
    return ins.stage,[e['stage'] for e in ins.events],round(t,1)

if __name__=='__main__':
    for off in [(0,0),(2,-1),(-3,2.5),(4.5,0),(9,0)]:
        print('hole offset',off,'mm ->',simulate(off))
