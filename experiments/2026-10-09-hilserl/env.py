#!/usr/bin/env python3
"""Full USB insertion env for HIL-SERL on the A1Z, built on ../2026-10-02/rl_env.InsertEnv.

What changes against last week's last-centimetre env (RL-RECOVERY.md):
- Start: in free air above the port, xy uniform in a disc of start_r_mm around the calibrated port, hover_min..max mm
  above the hub surface (HIL-SERL randomises the start pose in a box around the target; RAM insertion used +-4 cm).
- Reward (cfg.reward): 'classifier' = the paper's binary image classifier on the side camera (classifier.py), the
  default; 'usb' = the device enumerates (kernel signal, lags physical insertion by a few hundred ms). Whatever the
  reward, every step logs the USB signal, the classifier probability and the geometric in_port test, so the
  electrical signal referees the classifier and the true success rate is counted independently.
- Observations: wrist image, side image (fixed D435/D455, cropped to 128x128), 9-d state.
- Bounds: the xy box is around the port (start disc + margin), not around the episode start, and commands may go
  floor_mm below the mouth so the plug can be pushed home.

Unchanged and reused: Cartesian xyz steps at 10 Hz with fixed orientation, the press bound (command at most press_mm
below the measured height near the hub: this is HIL-SERL's reference-error clip), the lateral lead bound, kinesthetic
takeover labels, the 30 Hz heartbeat clock, and every robot_proc safety check underneath.
"""
from dataclasses import dataclass
import math
import time

import numpy as np

import common  # noqa: F401  (puts ../2026-10-02 on sys.path)
from rl_env import Disengaged,EnvCfg,InsertEnv


class UsbStillConnected(RuntimeError):
    """After lifting, the device still enumerates: the plug did not come out of the port (or slipped in the fingers)."""


@dataclass
class FullCfg(EnvCfg):
    start_r_mm:float=10.0         # xy start offset radius around the port
    hover_min_mm:float=4.0        # start height above the hub surface
    hover_max_mm:float=12.0
    top_mm:float=20.0             # travel height above the surface (also the highest command)
    box_margin_mm:float=5.0       # xy box = port +- (start_r_mm + box_margin_mm)
    floor_mm:float=9.0            # lowest command below the mouth height (the plug must reach full depth)
    max_steps:int=100             # 10 s at 10 Hz
    reward:str='clf_usb'          # 'clf_usb' (classifier triggers, USB confirms) | 'classifier' (paper) | 'usb' | 'geometric' (tests)
    clf_hold:int=2                # consecutive classifier positives that trigger success (paper: 1)
    usb_confirm_s:float=1.5       # clf_usb: after the classifier fires, keep pressing up to this long for the USB
    handback_grace_steps:int=30   # after a takeover is handed back the policy always gets this many steps to finish
    crop_side:tuple|None=None     # (x0,y0,w,h) in the 640x480 side image; None = whole image
    lift_mm_s:float=6.0           # slow pull-out after a full insertion
    freeze_zone_mm:float=3.0      # freeze robot_proc's integral while the plug is within this height of the hub surface
    clear_mm:float=10.0           # pull-out: straight up at the measured xy and orientation until this far above the mouth
    press_n:float=8.0             # near the hub, downward push as a pure vertical force (N); 0 = old position press
    press_slew_n:float=1.0        # force change per 30 Hz tick (ramps onto the hub instead of hitting it)
    press_max_off_deg:float=12.0  # cap on the per-joint target offset that carries the force
    press_kxy:float=1.0           # while pressing: lateral spring toward the commanded xy, N/mm ...
    press_fxy_max:float=4.0       # ... capped at this force (the port chamfer, not the arm, centres the plug)
    press_krot:float=2.0          # while pressing: weak orientation spring toward the commanded orientation, N m/rad


class FullInsertEnv(InsertEnv):
    def __init__(self,robot,kin,port,usb,clock,cfg=FullCfg(),side=None,clf=None):
        super().__init__(robot,kin,port,usb,clock,cfg)
        self.side=side;self.clf=clf;self.clf_count=0;self.reset_clf_p=None;self.frozen=False;self.press_f=0.0;self.min_end_step=0
        if cfg.reward in ('classifier','clf_usb') and clf is None:raise ValueError(f"reward '{cfg.reward}' needs a trained classifier (--clf)")

    def sample_start(self,rng):
        c=self.cfg;r=c.start_r_mm*math.sqrt(rng.uniform());th=rng.uniform(0,2*math.pi)
        off=r*np.array([math.cos(th),math.sin(th)])
        return {'offset_mm':off.round(3).tolist(),'hover':True,'hover_dz_mm':round(float(rng.uniform(c.hover_min_mm,c.hover_max_mm)),2)}

    def _send(self,p,R):
        """Near the hub (within freeze_zone_mm of the surface):
        - Integral anti-windup: the joints are stuck in contact, and robot_proc's per-joint integral would otherwise grow
          to its cap and push the plug sideways. Freeze it there (offset kept), resume above the zone.
        - Pressing (target below the plug) is a pure vertical force instead of a position error. Joint PD maps a Cartesian
          position error to a coupled wrench: in the 2026-10-10 check (press 10 mm) the tip got 10-14 N down plus 8-16 N
          sideways and 3-4.5 N m about y, and the plug jammed 2-3 mm deep in 9 of 10 insertions. Here xy and orientation
          stay position-controlled (IK target at the measured height), and the push is added as joint target offsets
          tau/kp with tau = J^T [0,0,-F,0,0,0], i.e. what a Cartesian impedance controller would apply (HIL-SERL's Franka).
          F ramps to press_n over the first press_mm of commanded depth, slew-limited per tick.
          Second check the same day (press_n 5): still 8-15 N sideways and 2-4 N m. The commanded xy was ~1.5 mm off where
          the port held the plug, and the arm is stiff sideways (~11 N/mm in x), so position control fought the port wall.
          So while pressing all six axes are force-controlled about the measured pose: a capped lateral spring toward the
          commanded xy (enough to slide on the hub, small enough for the chamfer to win), a weak orientation spring, and
          the vertical push; no stored position error remains (remote-centre-compliance behaviour, done in software)."""
        c=self.cfg;pm,_=self.p_meas();near=bool(pm[2]<self.port.z_surface+c.freeze_zone_mm)
        if near!=self.frozen and hasattr(self.robot,'freeze_integral'):self.robot.freeze_integral(near)
        self.frozen=near
        p=np.array(p,dtype=float)
        if not (near and c.press_n>0 and p[2]<pm[2]):
            self.press_f=0.0;return super()._send(p,R)
        p[2]=max(p[2],pm[2]-c.press_mm)
        want=c.press_n*min(1.0,(pm[2]-p[2])/max(c.press_mm,1e-6))
        f=self.press_f+float(np.clip(want-self.press_f,-c.press_slew_n,c.press_slew_n));self.press_f=f
        self.p_cmd=p;self.R_cmd=R
        q,_=self.robot.measured();q=np.asarray(q,dtype=float);Rm=self.kin.fk(q)[:3,:3]
        fxy=np.clip(c.press_kxy*(p[:2]-pm[:2]),-c.press_fxy_max,c.press_fxy_max)
        w=np.r_[fxy,-f,c.press_krot*self.kin.pin.log3(R@Rm.T)]
        self.q_cmd=q.copy()   # IK seed when leaving the press: the measured pose, no stored error
        off=np.clip(self._jac(q).T@w/self._joint_kp(),*np.radians([-c.press_max_off_deg,c.press_max_off_deg]))
        self.robot.send_target(q+off,'press',press_n=round(f,2),press_fxy=np.round(fxy,2).tolist())

    def _joint_kp(self):
        kp=(getattr(self.robot,'ready_info',None) or {}).get('kp')
        if kp is None:
            import robot_proc;kp=robot_proc.DEFAULT_KP
        return np.asarray(kp,dtype=float)

    def _jac(self,q,h=1e-5):
        """6x6 finger-frame Jacobian (m and world-frame rad per joint rad), the convention of Kin.step_towards."""
        T=self.kin.fk(q);J=np.zeros((6,6))
        for i in range(6):
            dq=np.array(q,dtype=float);dq[i]+=h;Ti=self.kin.fk(dq)
            J[:3,i]=(Ti[:3,3]-T[:3,3])/h;J[3:,i]=self.kin.pin.log3(Ti[:3,:3]@T[:3,:3].T)/h
        return J

    def engage(self):
        super().engage();self.frozen=False   # robot_proc clears the freeze on every engage

    def lift(self,speed=None):
        """Pull-out that keeps the plug's axis (2026-10-10). The inherited lift starts from the last command, which near the
        hub is up to press_mm below the plug and may lead it sideways, and slerps the orientation to the calibrated R_ref
        while rising; with the plug still in the port both pull it sideways (bent connector after a firm grasp, 2026-10-09).
        Here: re-engage (robot_proc: command := measured joints, integral offset cleared), rise straight up keeping the
        measured xy and orientation until clear_mm above the mouth, and only then turn to R_ref on the way to travel height."""
        c=self.cfg;pm,_=self.p_meas()
        if pm[2]<self.port.z_mouth+c.clear_mm:
            if not self.robot.engage():raise Disengaged('engage refused')
            self.frozen=False   # robot_proc cleared the freeze; _send freezes again while the plug is still near the hub
            self._sync_cmd();p0=self.p_cmd.copy()
            self.goto([p0[0],p0[1],self.port.z_mouth+c.clear_mm],speed or c.lift_mm_s,R_goal=self.R_cmd)
        pm,_=self.p_meas()
        self.goto([self.p_cmd[0],self.p_cmd[1],max(self.port.z_surface+c.top_mm,pm[2])],speed or c.lift_mm_s,R_goal=self.port.R_ref)

    def handback(self):
        super().handback();self.min_end_step=self.steps+self.cfg.handback_grace_steps

    def _crop(self,img,crop):
        import cv2
        if crop:x,y,w,h=crop;img=img[y:y+h,x:x+w]
        return np.ascontiguousarray(cv2.resize(img,(self.cfg.img,self.cfg.img),interpolation=cv2.INTER_AREA),dtype=np.uint8)

    def images(self):
        """Cropped 128x128 wrist and side frames (side None without a side camera)."""
        import time
        with self.robot._frame_lock:img=self.robot._frame;age=time.monotonic()-self.robot._frame_t
        if img is None or age>0.25:raise RuntimeError(f'wrist camera stale ({age:.2f} s)')
        side=None if self.side is None else self._crop(self.side.latest(),self.cfg.crop_side)
        return self._crop(img,self.cfg.crop),side

    def obs(self):
        wrist,side=self.images();pm,_=self.p_meas()
        low=np.r_[(pm-self.p_start)/10.0,(self.p_cmd-pm)/5.0,self.prev_a].astype(np.float32)
        o={'image':wrist,'state':low}
        if side is not None:o['side']=side
        return o

    def reset(self,start):
        obs=super().reset(start)
        if self.usb.connected:raise UsbStillConnected('USB 仍然连着：插头没拔出来，或者在手指里滑了')
        self.clf_count=0;self.min_end_step=0;self.reset_clf_p=None if (self.clf is None or obs is None) else round(self.clf.prob(obs),3)
        return obs

    def _finish_step(self,a,info):
        c=self.cfg;self.prev_a=np.asarray(a,dtype=float);self.steps+=1
        pm,_=self.p_meas();usb=bool(self.usb.connected);ok=self.in_port(pm)
        obs=self.obs();p=None if self.clf is None else round(self.clf.prob(obs),4)
        if c.reward in ('classifier','clf_usb'):
            self.clf_count=self.clf_count+1 if p>=self.clf.threshold else 0;fired=self.clf_count>=c.clf_hold
            if c.reward=='classifier' or not fired:term=fired
            else:
                # 2026-10-10 run: the classifier fired with the plug 1/2-2/3 in (the last 1-2 mm are invisible from our
                # cameras) and the policy learned to stop there. Now a classifier positive only triggers a check: keep
                # pressing until the USB enumerates (~0.7 s after full insertion); if it does not, the episode goes on.
                t=time.monotonic();usb=self.wait_usb(c.usb_confirm_s);term=usb
                info.update({'clf_fire':True,'usb_confirm':usb,'confirm_s':round(time.monotonic()-t,2)})
                if not usb:self.clf_count=0
                pm,_=self.p_meas();ok=self.in_port(pm);obs=self.obs()
        elif c.reward=='usb':term=usb
        else:self.hold=self.hold+1 if ok else 0;term=self.hold>=c.success_hold
        # operator steps count toward max_steps, so a takeover handed back at 8-9 s used to end the episode on the policy's
        # first step (2026-10-11); the policy now always gets handback_grace_steps to finish from the corrected state
        trunc=(not term) and self.steps>=max(c.max_steps,self.min_end_step)
        info.update({'p_meas':pm.round(2).tolist(),'p_cmd':self.p_cmd.round(2).tolist(),'in_port':ok,'usb':usb,'clf_p':p,'press_n':round(self.press_f,2)})
        self.trace.append(info)
        return obs,float(term),term,trunc,info

    def wait_usb(self,timeout):
        """Hold the current command (pressing, when engaged) until the USB enumerates or timeout. Returns connected."""
        t=time.monotonic()
        while not self.usb.connected and time.monotonic()-t<timeout:
            if self.robot.engaged:self._send(self.p_cmd,self.R_cmd)
            self.clock.tick(need_engaged=False)
        return bool(self.usb.connected)

    def step(self,a):
        c=self.cfg;a=np.clip(np.asarray(a,dtype=float),-1,1);pm,_=self.p_meas()
        p=self.p_cmd+a*c.step_mm;clamped=[]
        half=c.start_r_mm+c.box_margin_mm;lo=self.port.p_port[:2]-half;hi=self.port.p_port[:2]+half
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

    def touch(self,side_axis='y',offset_mm=5.0):
        """Grasp check for hover starts: rest the plug on the hub beside the port and compare with the calibrated surface
        height (a slipped plug changes it). Returns contact_dz_mm, or None if the plug went into the port."""
        off=np.zeros(2);off[0 if side_axis=='x' else 1]=offset_mm
        obs=InsertEnv.reset(self,{'offset_mm':off.tolist(),'hover':False,'hover_dz_mm':0.0})
        dz=self.contact_dz;self.lift()
        return None if obs is None else dz

    def insert_from_above(self,timeout=8.0):
        """Scripted test of the force budget: from above the calibrated port, descend straight down (press bound active)
        until USB enumerates. Uses the privileged port position; never part of an RL episode."""
        import time
        c=self.cfg;self.engage();self.lift()
        top=self.port.z_surface+c.hover_min_mm
        self.goto([self.port.p_port[0],self.port.p_port[1],top],c.travel_mm_s,R_goal=self.port.R_ref)
        t0=time.monotonic()
        self.goto([self.port.p_port[0],self.port.p_port[1],self.port.z_mouth-c.floor_mm],c.descend_mm_s,R_goal=self.port.R_ref,
                  stop=lambda:self.usb.connected,settle_ticks=0,timeout=timeout)
        while not self.usb.connected and time.monotonic()-t0<timeout:self._send(self.p_cmd,self.R_cmd);self.clock.tick()
        ok=bool(self.usb.connected);dt=round(time.monotonic()-t0,2);z=round(float(self.p_meas()[0][2]),2);f=round(self.press_f,2)
        self.lift()
        return {'usb':ok,'seconds':dt,'z_end_mm':z,'z_below_mouth_mm':round(self.port.z_mouth-z,2),'press_n':f,'still_connected_after_lift':bool(self.usb.connected)}


__all__=['FullCfg','FullInsertEnv','UsbStillConnected','Disengaged']
