#!/usr/bin/env python3
"""Scripted last-centimetre insertion that takes over from the learned policy.

ACT brings the plug to within a few millimetres of the port, then rests on the hub surface: it learned
positions, not the push force (kinesthetic demos never recorded the hand's force). This stage is not
learned, only tuned:
  stall  : plug height stops dropping for stall_s while the commanded pose is pressed below it
  search : keep orientation, press press_mm below the stall point, sweep an outward spiral in the
           horizontal plane (pitch_mm per turn, up to radius_mm) - classic peg-in-hole spiral search
  push   : once the plug drops drop_mm below the stall height (it entered the port mouth), stop moving
           sideways and press push_mm down until USB enumerates or push_s elapses
Targets are joint positions sent through the same robot-process safety layer as policy actions
(rate limit, target-gap disengage, heartbeat), so this stage cannot command anything the policy could not.
"""
from collections import deque
from dataclasses import dataclass
import math
import sys

import numpy as np

SDK='/home/dimos/GALAXEA-A1Z'
URDF=SDK+'/a1z/robot_models/a1z/A1Z_G1Z.urdf'


class Kin:
    """Forward kinematics of the gripper finger frame and a damped least-squares pose step (pinocchio)."""
    def __init__(self,urdf=URDF):
        if SDK not in sys.path:sys.path.insert(0,SDK)
        from a1z.robots.kinematics import Kinematics
        import pinocchio as pin
        self.k=Kinematics(urdf);self.pin=pin

    def fk(self,q):
        return self.k.fk(np.asarray(q[:6],dtype=float))

    def step_towards(self,q,p_target,R_target,iters=3,damping=1e-3):
        """Joint vector near q whose finger frame reaches (p_target [m], R_target)."""
        q=np.asarray(q[:6],dtype=float).copy()
        for _ in range(iters):
            T=self.fk(q);err=np.r_[p_target-T[:3,3],self.pin.log3(R_target@T[:3,:3].T)]
            J=np.zeros((6,6));h=1e-5
            for i in range(6):
                dq=q.copy();dq[i]+=h;Ti=self.fk(dq)
                J[:3,i]=(Ti[:3,3]-T[:3,3])/h;J[3:,i]=self.pin.log3(Ti[:3,:3]@T[:3,:3].T)/h
            q=q+J.T@np.linalg.solve(J@J.T+damping*np.eye(6),err)
        return q


@dataclass
class SearchConfig:
    stall_s:float=1.0      # plug height flat for this long ...
    stall_band_mm:float=1.5
    pressed_mm:float=5.0   # ... while the command is at least this far below the plug
    min_policy_s:float=2.0
    press_mm:float=6.0     # how far below the stall point the search target sits (sets the press force)
    pitch_mm:float=1.0     # spiral growth per turn
    radius_mm:float=5.0
    turn_s:float=2.0       # seconds per spiral turn
    search_s:float=14.0
    drop_mm:float=2.0      # plug fell this far below the stall height -> it is in the port mouth
    push_mm:float=15.0
    push_s:float=5.0


class Inserter:
    """Per-trial state machine. Feed it measured/commanded joints each tick; it says what to send."""
    def __init__(self,kin,cfg=SearchConfig()):
        self.kin=kin;self.cfg=cfg;self.stage='policy';self.hist=deque();self.events=[]
        self.anchor=None;self.R0=None;self.t_stage=None;self.drop_xy=None;self.q_cmd=None

    def _enter(self,stage,t,**info):
        self.stage=stage;self.t_stage=t;self.events.append({'t':round(t,3),'stage':stage,**info})

    def update(self,t,q,q_cmd):
        """t: seconds since trial start. Returns (stage, joint_target or None). None = let the policy drive."""
        c=self.cfg;T=self.kin.fk(q);z=T[2,3]*1000
        if self.stage=='policy':
            zc=self.kin.fk(q_cmd)[2,3]*1000
            self.hist.append((t,z,zc))
            while self.hist and t-self.hist[0][0]>c.stall_s:self.hist.popleft()
            span=t-self.hist[0][0] if self.hist else 0
            zs=[h[1] for h in self.hist]
            if (t>=c.min_policy_s and span>=0.9*c.stall_s and max(zs)-min(zs)<c.stall_band_mm
                    and z-zc>=c.pressed_mm):
                self.anchor=T[:3,3].copy();self.R0=T[:3,:3].copy();self.q_cmd=np.asarray(q_cmd[:6],dtype=float).copy()
                self._enter('search',t,stall_z_mm=round(z,2),pressed_mm=round(z-zc,2))
            else:
                return 'policy',None
        dz_drop=self.anchor[2]*1000-z
        if self.stage=='search':
            if dz_drop>=c.drop_mm:
                self.drop_xy=T[:2,3].copy();self._enter('push',t,drop_mm=round(dz_drop,2))
            elif t-self.t_stage>c.search_s:
                self._enter('search_failed',t);return 'search_failed',None
            else:
                s=t-self.t_stage;theta=2*math.pi*s/c.turn_s;r=min(c.radius_mm,c.pitch_mm*theta/(2*math.pi))/1000
                p=self.anchor+np.array([r*math.cos(theta),r*math.sin(theta),-c.press_mm/1000])
                self.q_cmd=self.kin.step_towards(self.q_cmd,p,self.R0);return 'search',self.q_cmd
        if self.stage=='push':
            if t-self.t_stage>c.push_s:
                self._enter('push_failed',t);return 'push_failed',None
            p=np.array([self.drop_xy[0],self.drop_xy[1],self.anchor[2]-c.push_mm/1000])
            self.q_cmd=self.kin.step_towards(self.q_cmd,p,self.R0);return 'push',self.q_cmd
        return self.stage,None
