#!/usr/bin/env python3
"""Offline check of robot_proc.smooth_step on a one-joint stick-slip model under the vendor MIT law
tau = kp*(cmd-q) + kd*(v_cmd-dq) (no hardware). Legacy = velocity-clipped target with v_cmd=0, as run on 2026-10-05,
where J2 lagged up to 5.6 deg and then broke free at ~2x the command rate. The model is a toy: it checks the direction
and rough size of the effect, not the arm's real numbers."""
import sys
from pathlib import Path
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parent))
from robot_proc import smooth_step


def simulate(smooth,jump_deg=8.0,kp=45.0,kd=1.0,inertia=0.05,fs=2.0,fc=1.4,vmax=0.5,secs=3.0):
    """Returns (peak |dq| rad/s, overshoot deg past the target, max cmd-q lag deg, settle error deg)."""
    target=np.radians(jump_deg);q=0.0;dq=0.0;cmd=np.zeros(1);v=np.zeros(1);ctl_dt=0.01;sub=10;h=ctl_dt/sub
    peak=0.0;over=0.0;lag=0.0;t=0.0
    for _ in range(int(secs/ctl_dt)):
        if smooth:
            vm=vmax*min(1.0,0.2+0.8*t/1.0)
            cmd,v=smooth_step(cmd,v,np.array([target]),vm,3.0,ctl_dt);vff=float(v[0])
        else:cmd=cmd+np.clip(target-cmd,-vmax*ctl_dt,vmax*ctl_dt);vff=0.0
        for _ in range(sub):
            tau=kp*(cmd[0]-q)+kd*(vff-dq)
            if abs(dq)<1e-4 and abs(tau)<=fs:dq=0.0;continue      # stuck
            f=fc*np.sign(dq if abs(dq)>=1e-4 else tau)
            dq+=(tau-f)/inertia*h;q+=dq*h
        t+=ctl_dt;peak=max(peak,abs(dq));over=max(over,np.degrees(q-target));lag=max(lag,np.degrees(abs(cmd[0]-q)))
    return peak,over,lag,np.degrees(abs(target-q))


def main():
    for jump in (3.0,8.0):
        lp,lo,ll,le=simulate(False,jump);sp,so,sl,se=simulate(True,jump)
        print(f'{jump:.0f} deg target jump | legacy: peak {lp:.2f} rad/s, overshoot {lo:+.2f} deg, lag {ll:.1f} deg | '
              f'smooth: peak {sp:.2f} rad/s, overshoot {so:+.2f} deg, lag {sl:.1f} deg')
        assert sp<=lp+1e-6 and sl<=ll+1e-6,(lp,sp,ll,sl)
        if jump>=8:assert sp<0.8*lp,(lp,sp)   # the burst after a large jump is what --smooth is for
        assert sp<=0.5*1.2,sp            # stays near the commanded speed limit
        assert so<=max(lo,0.0)+1e-9      # no more overshoot than legacy
    # smooth_step itself: never passes a fixed target, reaches it, respects vmax and amax
    cmd=np.zeros(2);v=np.zeros(2);tg=np.array([0.3,-0.1]);vs=[]
    for _ in range(400):
        pv=v.copy();cmd,v=smooth_step(cmd,v,tg,0.5,3.0,0.01);vs.append(v.copy())
        arrived=np.isclose(cmd,tg)  # landing on the target zeroes v in one step (residual speed < sqrt(2*amax*step))
        assert np.all(np.abs(v)<=0.5+1e-9) and np.all((np.abs(v-pv)<=3.0*0.01+1e-9)|arrived)
        assert np.all(np.sign(tg)*(tg-cmd)>=-1e-12)
    assert np.allclose(cmd,tg),cmd
    print('smooth_step OK')

if __name__=='__main__':main()
