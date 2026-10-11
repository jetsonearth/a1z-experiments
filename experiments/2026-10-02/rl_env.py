#!/usr/bin/env python3
"""Real-robot RL environment for the last centimetre of the USB insertion (HIL-SERL style, see RL-RECOVERY.md).

Each episode starts with the plug resting next to the port (or hovering a few mm above it), placed there by a scripted
reset relative to the calibrated port. The policy then moves the finger frame by small Cartesian steps (10 Hz, xyz,
orientation fixed to the calibrated inserted orientation) until the plug drops into the port mouth.

Success is geometric, from calib.py's heights under the current grasp: the finger frame is at or below the mouth
height and within success_xy_mm of the inserted position. Pressing on the plastic cannot bring the plug that low, only
entering the port can. The port reference is privileged (reward and reset only); the policy never observes it.

Safety bounds (the RL exploration needs them; each is listed in RL-RECOVERY.md and was approved for this experiment):
xy box around the episode start, z between the inserted height and the hover height, commanded z at most press_mm below
the measured z (bounds the pressing force), commanded xy at most lead_mm ahead of the measured xy. Every joint target
still goes through robot_proc.py's own rate limit, target-gap check and heartbeat.
"""
from dataclasses import dataclass
import json
import math
import time

import numpy as np


@dataclass
class EnvCfg:
    hz:float=10.0
    ticks_per_step:int=3          # 30 Hz client loop, one RL decision every 3 ticks
    step_mm:float=1.5             # max move per RL step and axis
    box_mm:float=10.0             # xy box around the episode start
    press_mm:float=5.0            # commanded z at most this far below the measured z
    lead_mm:float=4.0             # commanded xy at most this far from the measured xy (RL steps)
    top_mm:float=12.0             # travel height above the surface
    floor_mm:float=3.0            # lowest command below the mouth height (training never pushes to full depth)
    slip_tol_mm:float=1.5         # contact height vs calibrated surface: larger = the plug moved in the fingers
    success_xy_mm:float=3.0
    success_dz_mm:float=0.5
    success_hold:int=2
    max_steps:int=80              # 8 s
    r_min:float=2.5
    r_max:float=7.0
    side_frac:float=0.5           # fraction of starts offset along side_axis (beside the port)
    side_axis:str='y'             # base-frame axis along the port's long side
    side_noise_mm:float=1.0
    hover_frac:float=0.25
    hover_dz:tuple=(2.0,6.0)
    crop:tuple|None=None          # (x0,y0,w,h) in the 640x480 wrist image; None = whole image
    img:int=128
    travel_mm_s:float=25.0
    descend_mm_s:float=5.0
    lift_mm_s:float=10.0


class Abort(Exception):
    """Operator abort (x) or quit during a scripted motion or an RL step."""


class Disengaged(Exception):
    """robot_proc left ENGAGED by itself (safety stop)."""


class Port:
    """Port reference from a calib.py file, all in the finger frame under the current grasp: the mouth samples ("plug
    just inside, not pushed") give the port xy, the reference orientation and the success height; the surface samples
    (plug resting next to the port) give the contact height. Inserted samples are not used: pushing to full depth
    slides the plug in the fingers (2026-10-05 21:29, taped + zip-tied: the finger frame moved 8.8 mm sideways from
    the mouth), so they record where the gripper went, not where the port is."""
    def __init__(self,path,kin):
        d=json.load(open(path))
        # a sample whose 1 s average moved more than 1 mm was taken while the arm was still moving: not a reference
        self.dropped=[(i,x['label'],x.get('std_mm')) for i,x in enumerate(d['samples']) if max(x.get('std_mm',[0]))>1.0]
        s=[x for x in d['samples'] if max(x.get('std_mm',[0]))<=1.0]
        mouth=[x for x in s if x['label']=='mouth'];surf=[x for x in s if x['label']=='surface']
        if len(mouth)<1 or not surf:raise ValueError(f'{path}: need mouth and surface samples')
        self.path=str(path)
        pm=np.array([x['p_mm'] for x in mouth],dtype=float)
        self.p_mouth=pm.mean(0);self.q_ref=np.array(mouth[-1]['q'],dtype=float)
        self.R_ref=kin.fk(self.q_ref)[:3,:3].copy()
        self.z_mouth=float(self.p_mouth[2]);self.p_port=self.p_mouth   # xy = port centre
        self.z_surface=float(np.mean([x['p_mm'][2] for x in surf]))
        self.surface_spread=float(np.ptp([x['p_mm'][2] for x in surf]))
        self.mouth_spread=float(np.max(np.linalg.norm(pm-self.p_mouth,axis=1))) if len(pm)>1 else 0.0
        problems=[]
        if self.z_surface-self.z_mouth<1.5:problems.append(f'surface only {self.z_surface-self.z_mouth:.2f} mm above the mouth')
        if self.mouth_spread>1.5:problems.append(f'mouth samples {self.mouth_spread:.1f} mm apart')
        if self.surface_spread>1.0:problems.append(f'surface samples differ by {self.surface_spread:.1f} mm in height')
        if problems:raise ValueError(f'{path}: '+'; '.join(problems)+' - recalibrate')

    def describe(self):
        return {'file':self.path,'p_mouth_mm':self.p_mouth.round(2).tolist(),'z_mouth_mm':round(self.z_mouth,2),'z_surface_mm':round(self.z_surface,2),
                'surface_minus_mouth_mm':round(self.z_surface-self.z_mouth,2),'mouth_spread_mm':round(self.mouth_spread,2),'surface_spread_mm':round(self.surface_spread,2),
                'dropped_unsteady_samples':self.dropped}


class Clock:
    """30 Hz client tick: heartbeat, robot events, operator hook (may raise Abort), pacing."""
    def __init__(self,robot,hz=30.0,hook=None):
        self.robot=robot;self.dt=1.0/hz;self.hook=hook;self.last=time.monotonic();self.events=[];self.max_gap=0.0;self.slow=0;self.slow_at=[]

    def tick(self,need_engaged=True):
        self.robot.heartbeat()
        for e in self.robot.poll_events():
            self.events.append(e)
            if e[0]=='fatal':raise RuntimeError('robot process: '+str(e[1:]))
        if self.hook:self.hook()
        if need_engaged and not self.robot.engaged:raise Disengaged(self.robot.disengage_reason or 'robot left engaged mode')
        now=time.monotonic()
        if now-self.last<1.0:   # >1 s = waiting for the operator between motions, not a stalled tick
            self.max_gap=max(self.max_gap,now-self.last);self.slow+=int(now-self.last>0.1)
            if now-self.last>0.1:self.slow_at.append((round(self.last,3),round(now-self.last,3)))
        time.sleep(max(0.0,self.dt-(now-self.last)));self.last=time.monotonic()


def _rot_interp(pin,R0,R1,frac):
    return R0@pin.exp3(frac*pin.log3(R0.T@R1))


class InsertEnv:
    def __init__(self,robot,kin,port,usb,clock,cfg=EnvCfg()):
        self.robot=robot;self.kin=kin;self.port=port;self.usb=usb;self.clock=clock;self.cfg=cfg
        self.q_cmd=None;self.p_cmd=None;self.R_cmd=None;self.p_start=None;self.contact_dz=None;self.prev_a=np.zeros(3);self.steps=0;self.hold=0
        self.trace=[]

    # ---------- kinematics helpers (mm) ----------
    def p_meas(self):
        q,_=self.robot.measured();T=self.kin.fk(q)
        return T[:3,3]*1000.0,T[:3,:3]

    def in_port(self,p=None):
        p=self.p_meas()[0] if p is None else p
        c=self.cfg
        return bool(p[2]<=self.port.z_mouth+c.success_dz_mm and np.hypot(*(p[:2]-self.port.p_port[:2]))<=c.success_xy_mm)

    def _sync_cmd(self):
        q,_=self.robot.measured();self.q_cmd=np.asarray(q,dtype=float).copy()
        T=self.kin.fk(self.q_cmd);self.p_cmd=T[:3,3]*1000.0;self.R_cmd=T[:3,:3].copy()

    def _send(self,p,R):
        pm,_=self.p_meas()
        p=np.array(p,dtype=float)
        if pm[2]<self.port.z_surface+3.0:   # bounded press near the hub; in free air the command may lead further
            p[2]=max(p[2],pm[2]-self.cfg.press_mm)   # (a 2 mm lead left the arm stuck 12-34 mm above the hub, 2026-10-05)
        self.p_cmd=p;self.R_cmd=R
        self.q_cmd=self.kin.step_towards(self.q_cmd,p/1000.0,R)
        self.robot.send_target(self.q_cmd,'rl')

    def engage(self):
        if not self.robot.engaged:
            if not self.robot.engage():raise Disengaged('engage refused')
        self._sync_cmd()

    def goto(self,p_goal,speed,R_goal=None,stop=None,settle_ticks=6,timeout=20.0):
        """Straight-line Cartesian move of the command at speed mm/s (orientation slerped to R_goal); stop() ends it early."""
        p0=self.p_cmd.copy();R0=self.R_cmd.copy();R1=R0 if R_goal is None else R_goal
        p_goal=np.asarray(p_goal,dtype=float);dist=float(np.linalg.norm(p_goal-p0))
        ang=float(np.linalg.norm(self.kin.pin.log3(R0.T@R1)))
        n=max(1,int(math.ceil(max(dist/(speed*self.clock.dt),ang/(0.3*self.clock.dt)))))  # rotation at most 0.3 rad/s
        t0=time.monotonic()
        for i in range(1,n+1):
            f=i/n;self._send(p0+(p_goal-p0)*f,_rot_interp(self.kin.pin,R0,R1,f));self.clock.tick()
            if stop and stop():return True
            if time.monotonic()-t0>timeout:return False
        for _ in range(settle_ticks):
            self._send(self.p_cmd,self.R_cmd);self.clock.tick()
            if stop and stop():return True
        return False

    # ---------- episode ----------
    def sample_start(self,rng):
        c=self.cfg;r=rng.uniform(c.r_min,c.r_max)
        if rng.uniform()<c.side_frac:
            sign=1.0 if rng.uniform()<0.5 else -1.0;ax=0 if c.side_axis=='x' else 1
            off=np.zeros(2);off[ax]=sign*r;off[1-ax]=rng.uniform(-c.side_noise_mm,c.side_noise_mm)
        else:
            th=rng.uniform(0,2*math.pi);off=r*np.array([math.cos(th),math.sin(th)])
        hover=bool(rng.uniform()<c.hover_frac);dz=float(rng.uniform(*c.hover_dz)) if hover else 0.0
        return {'offset_mm':off.round(3).tolist(),'hover':hover,'hover_dz_mm':round(dz,2)}

    def lift(self,speed=None):
        top=self.port.z_surface+self.cfg.top_mm
        pm,_=self.p_meas()
        self.goto([self.p_cmd[0],self.p_cmd[1],max(top,pm[2])],speed or self.cfg.lift_mm_s,R_goal=self.port.R_ref)

    def reset(self,start):
        """Scripted reset to start (from sample_start). Returns the first observation, or None if the plug fell into the port."""
        c=self.cfg;self.engage();self.lift()
        top=self.port.z_surface+c.top_mm;xy=self.port.p_port[:2]+np.array(start['offset_mm'])
        self.goto([xy[0],xy[1],top],c.travel_mm_s,R_goal=self.port.R_ref)
        if start['hover']:
            self.goto([xy[0],xy[1],self.port.z_surface+start['hover_dz_mm']],c.descend_mm_s)
        else:
            hist=[]
            def blocked():   # for 1 s the command stayed >= 1.5 mm below the plug AND the plug moved < 0.3 mm: it rests
                # on something. Lag alone is not contact: in free air the arm lagged ~3 mm and stuck for seconds (2026-10-05).
                pm,_=self.p_meas();hist.append((self.p_cmd[2]<pm[2]-1.5 and pm[2]<self.port.z_surface+3.0,pm[2]))
                w=hist[-30:]   # only within 3 mm of the calibrated surface can the plug be resting on the hub
                return len(w)==30 and all(x[0] for x in w) and max(x[1] for x in w)-min(x[1] for x in w)<0.3
            self.goto([xy[0],xy[1],self.port.z_surface-1.5],c.descend_mm_s,stop=blocked,settle_ticks=0)
            for _ in range(9):self._send(self.p_cmd,self.R_cmd);self.clock.tick()   # 0.3 s settle, light press
        if self.in_port():return None
        # Grasp check: resting on the surface, the finger frame should sit at the calibrated surface height. If the
        # plug slid in the fingers, this height moves by the slip (the policy and the success test would be off).
        self.contact_dz=None if start['hover'] else round(float(self.p_meas()[0][2]-self.port.z_surface),2)
        self.p_start=self.p_meas()[0].copy();self.prev_a=np.zeros(3);self.steps=0;self.hold=0;self.trace=[]
        return self.obs()

    def begin_episode(self):
        """Start an RL episode from wherever the plug is now (hand-off from another policy, rl_run.py chain)."""
        self._sync_cmd();self.p_start=self.p_meas()[0].copy();self.prev_a=np.zeros(3);self.steps=0;self.hold=0
        self.trace=[];self.contact_dz=None
        return self.obs()

    def obs(self):
        import cv2
        with self.robot._frame_lock:img=self.robot._frame;age=time.monotonic()-self.robot._frame_t
        if img is None or age>0.25:raise RuntimeError(f'wrist camera stale ({age:.2f} s)')
        if self.cfg.crop:x,y,w,h=self.cfg.crop;img=img[y:y+h,x:x+w]
        img=cv2.resize(img,(self.cfg.img,self.cfg.img),interpolation=cv2.INTER_AREA)
        pm,_=self.p_meas()
        low=np.r_[(pm-self.p_start)/10.0,(self.p_cmd-pm)/5.0,self.prev_a].astype(np.float32)
        return {'image':np.ascontiguousarray(img,dtype=np.uint8),'state':low}

    def _finish_step(self,a,info):
        c=self.cfg;self.prev_a=np.asarray(a,dtype=float);self.steps+=1
        pm,_=self.p_meas();ok=self.in_port(pm);self.hold=self.hold+1 if ok else 0
        term=self.hold>=c.success_hold;trunc=(not term) and self.steps>=c.max_steps
        info.update({'p_meas':pm.round(2).tolist(),'p_cmd':self.p_cmd.round(2).tolist(),'in_port':ok})
        self.trace.append(info)
        return self.obs(),float(term),term,trunc,info

    def step(self,a):
        c=self.cfg;a=np.clip(np.asarray(a,dtype=float),-1,1);pm,_=self.p_meas()
        p=self.p_cmd+a*c.step_mm;clamped=[]
        lo=self.p_start[:2]-c.box_mm;hi=self.p_start[:2]+c.box_mm
        if (p[:2]<lo).any() or (p[:2]>hi).any():clamped.append('box')
        p[:2]=np.clip(p[:2],lo,hi)
        zlo=self.port.z_mouth-c.floor_mm;zhi=self.port.z_surface+c.top_mm
        if p[2]<zlo or p[2]>zhi:clamped.append('z')
        p[2]=min(max(p[2],zlo),zhi)
        if p[2]<pm[2]-c.press_mm:clamped.append('press')
        lead=p[:2]-pm[:2];n=float(np.linalg.norm(lead))
        if n>c.lead_mm:p[:2]=pm[:2]+lead*c.lead_mm/n;clamped.append('lead')
        for _ in range(c.ticks_per_step):self._send(p,self.port.R_ref);self.clock.tick()
        return self._finish_step(a,{'a':a.round(3).tolist(),'clamped':clamped,'human':False})

    def human_step(self):
        """One 10 Hz step while the operator drags the floating arm. The label is the measured move, clipped like a policy action."""
        c=self.cfg;p0,_=self.p_meas()
        for _ in range(c.ticks_per_step):self.clock.tick(need_engaged=False)
        p1,_=self.p_meas();raw=(p1-p0)/c.step_mm;a=np.clip(raw,-1,1)
        self._sync_cmd()
        obs,r,term,trunc,info=self._finish_step(a,{'a':a.round(3).tolist(),'clamped':['label_clip'] if (np.abs(raw)>1).any() else [],'human':True})
        trunc=(not term) and self.steps>=2*c.max_steps   # operator time is not cut at the policy limit
        return obs,r,term,trunc,info,a

    def handback(self):
        self.engage()

    def verify_insert(self,timeout=3.0):
        """After the mouth: press on to the inserted height and wait for USB. Returns (usb_connected, z_mm)."""
        t0=time.monotonic();xy=self.p_meas()[0][:2]
        self.goto([xy[0],xy[1],self.port.z_mouth-8.0],4.0,R_goal=self.port.R_ref,stop=lambda:self.usb.connected,settle_ticks=0,timeout=timeout)
        while not self.usb.connected and time.monotonic()-t0<timeout:self._send(self.p_cmd,self.R_cmd);self.clock.tick()
        return bool(self.usb.connected),round(float(self.p_meas()[0][2]),2)


# ---------- scripted baselines, acting through the same step() and bounds ----------
class HoldPolicy:
    name='hold'
    def reset(self,env):pass
    def __call__(self,env,obs):return np.array([0.0,0.0,-1.0])


class SpiralPolicy:
    """Archimedean spiral around the episode start while pressing (insertion.py's search, expressed as env actions)."""
    name='spiral'
    def __init__(self,pitch_mm=1.0,radius_mm=8.0,speed_mm_s=12.0):
        self.pitch=pitch_mm;self.radius=radius_mm;self.speed=speed_mm_s
    def reset(self,env):self.theta=0.0
    def __call__(self,env,obs):
        r=max(0.3,self.pitch*self.theta/(2*math.pi))
        self.theta+=self.speed/env.cfg.hz/r                       # constant path speed
        r=min(self.radius,self.pitch*self.theta/(2*math.pi))
        tgt=env.p_start[:2]+r*np.array([math.cos(self.theta),math.sin(self.theta)])
        pm,_=env.p_meas()
        a=np.r_[(tgt-env.p_cmd[:2])/env.cfg.step_mm,(pm[2]-env.cfg.press_mm-env.p_cmd[2])/env.cfg.step_mm]
        return np.clip(a,-1,1)
