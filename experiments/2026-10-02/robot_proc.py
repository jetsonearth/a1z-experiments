#!/usr/bin/env python3
"""A1Z robot process: the vendor SDK alone in its own interpreter, as during recording.

Camera, gRPC and policy work live in the client process, so they can never take this process's GIL
or slow the SDK's 250 Hz control loop. This process never writes to the terminal (stdout/stderr go to
a file), so a stalled terminal cannot block the control thread either.

Modes: FLOAT (vendor gravity assist: kp 0, gravity comp 1.0, as in recording) and ENGAGED (policy targets,
rate-limited here at 100 Hz, through the SDK's own command_joint_state with all vendor checks).
Flight recorder: 50 Hz, every mode, to flight.jsonl.

Shared memory:
  link  (client -> robot): [heartbeat_t, target_t, j1..j6]
  state (robot -> client): [t, mode, sdk_hz, gripper, q(6), dq(6), eff(6), cmd(6), gripper_temp_C, max_joint_temp_C]
"""
import json
import logging
import os
from pathlib import Path
import queue
import sys
import threading
import time

ROOT=Path(__file__).resolve().parent
FLOAT,ENGAGED=0.0,1.0
STATE_N=4+6*4+2
LINK_N=8


class _Capture(logging.Handler):
    """Keeps SDK log records (rate, warnings) without printing anything."""
    def __init__(self):super().__init__();self.hz=float('nan');self.records=queue.SimpleQueue()
    def emit(self,record):
        msg=record.getMessage()
        if 'Control loop frequency' in msg:
            try:self.hz=float(msg.split(':')[1].split('Hz')[0])
            except Exception:pass
        if record.levelno>=logging.WARNING:self.records.put((record.levelname,msg[:200]))


def smooth_step(cmd,v,target,vmax,amax,dt):
    """Replan a bounded-acceleration point-to-point trajectory from (cmd, v).

    Integrate through acceleration/cruise/braking phase boundaries, including
    boundaries inside this tick. A moving target inside the stopping distance
    requires braking past it and returning; never teleport position or zero a
    nonzero velocity. This limits acceleration, not jerk or measured velocity.
    No robot/task-specific tuning beyond the existing vmax/amax is introduced.
    """
    import math
    import numpy as np
    cmd=np.asarray(cmd,dtype=float);v=np.asarray(v,dtype=float)
    target=np.asarray(target,dtype=float)
    if cmd.shape!=v.shape or target.shape!=cmd.shape:
        raise ValueError('position, velocity and target shapes must match')
    if not (np.isfinite(cmd).all() and np.isfinite(v).all() and np.isfinite(target).all()
            and math.isfinite(vmax) and math.isfinite(amax) and math.isfinite(dt)
            and vmax>0 and amax>0 and dt>=0):
        raise ValueError('finite states, positive limits and nonnegative dt required')
    qout=cmd.copy();vout=v.copy()
    for idx in np.ndindex(cmd.shape):
        q=float(cmd[idx]);u=float(v[idx]);goal=float(target[idx]);remaining=dt
        # A reduced velocity limit cannot instantaneously erase existing speed.
        if abs(u)>vmax:
            acc=-math.copysign(amax,u)
            duration=(abs(u)-vmax)/amax
            h=min(remaining,duration)
            q+=u*h+0.5*acc*h*h;u+=acc*h;remaining-=h
            if remaining<=0:
                qout[idx]=q;vout[idx]=u;continue
        error=goal-q
        stop_error=error-u*abs(u)/(2*amax)
        direction=1.0 if stop_error>=0 else -1.0
        distance=direction*error;speed=direction*u
        peak=min(vmax,math.sqrt(max(0.0,amax*distance+0.5*speed*speed)))
        accel_time=max(0.0,(peak-speed)/amax)
        cruise_distance=max(0.0,distance-(2*peak*peak-speed*speed)/(2*amax))
        cruise_time=cruise_distance/peak if peak>0 else 0.0
        phases=((direction*amax,accel_time),(0.0,cruise_time),
                (-direction*amax,peak/amax))
        finished=True
        for acc,duration in phases:
            h=min(remaining,duration)
            q+=u*h+0.5*acc*h*h;u+=acc*h;remaining-=h
            if h<duration:
                finished=False;break
        if finished:
            # All braking phases have elapsed; only eliminate roundoff here.
            q=goal;u=0.0
        qout[idx]=q;vout[idx]=u
    return qout,vout


GRIP_RAMP_S=0.5


def grip_ramp(cur,goal,span,dt,ramp_s=GRIP_RAMP_S):
    """Move the gripper current limit toward goal at span/ramp_s per second (span = full - hold); never overshoots."""
    step=abs(span)*dt/ramp_s if ramp_s>0 else abs(goal-cur)
    return goal if abs(goal-cur)<=step else cur+(step if goal>cur else -step)


def integral_step(offset,cmd,q,dq,ki,dt,max_rad,still=0.05,gate_rad=0.09,leak_s=0.5):
    """Outer integral loop on joint position (off when ki=0), aimed at static friction only. The vendor loop is PD +
    gravity comp; static friction leaves a steady error of ~friction/kp (1-2 deg at default kp, measured 2026-10-05).
    A joint integrates only while it is stuck (|dq| < still rad/s) with a moderate error (< gate_rad); while it moves
    the offset leaks to zero (time constant leak_s), so it cannot wind up during fast motion and cause overshoot
    (seen 2026-10-05 with a plain integral under the policy: +-3.8 deg swings on J2). Clamped to +-max_rad, so the
    extra torque is bounded by kp*max_rad (in contact this is the pressing force). Reset on every engage/float."""
    import numpy as np
    err=cmd-q;stuck=(np.abs(dq)<still)&(np.abs(err)<gate_rad)
    out=np.where(stuck,offset+ki*err*dt,offset*max(0.0,1-dt/leak_s))
    return np.clip(out,-max_rad,max_rad)


def grip_force_levels(full_nm,hold_nm,peak_nm):
    """(i_full, i_hold) as fractions of the motor peak for a gripper force of full_nm, hold capped at full (None = one level).
    The SDK clips the current limit to 1.0 (= peak_nm)."""
    i_full=min(1.0,max(0.0,float(full_nm)/peak_nm))
    return i_full,(None if hold_nm is None else min(i_full,float(hold_nm)/peak_nm))


# Default gains while engaged since 2026-10-09 (dimos' A1Z gains). Same demo replayed in free air: vendor gains ~9.8 mm,
# these 2.8 mm, these + integral (ki 2/s, cap 1 deg) 0.3-0.5 mm. Static friction (~0.2-0.6 N m per joint) leaves a
# dead band of friction/kp; kd is raised with kp so the stiffer arm does not ring.
DEFAULT_KP=[80.0,80.0,80.0,50.0,20.0,20.0]
DEFAULT_KD=[3.0,3.0,3.0,0.7,0.4,0.4]


def gains_label(stiffness=None,kp=None,kd=None):
    """Short operator-facing description of the engaged gains (startup banners)."""
    if kp is None and kd is None:return f'刚度 {stiffness:.0%} 厂家 kp，厂家 kd（旧配置）' if stiffness is not None else '增益 dimos 默认（kp 80/80/80/50/20/20，kd 3/3/3/0.7/0.4/0.4）'
    return f"增益 kp {kp if kp is not None else ('厂家×'+format(stiffness,'g') if stiffness is not None else 'dimos 默认')}，kd {kd if kd is not None else ('厂家' if stiffness is not None else 'dimos 默认')}"


def engaged_gains(default_kp,default_kd,stiffness=None,kp=None,kd=None):
    """Gains while engaged. Explicit per-joint kp / kd win; otherwise stiffness (legacy) = stiffness x vendor kp with
    vendor kd, as in all runs before 2026-10-09; otherwise DEFAULT_KP / DEFAULT_KD. The SDK rejects kp > 200 or kd > 5."""
    import numpy as np
    legacy=stiffness is not None
    kp_on=np.asarray(kp if kp is not None else (np.asarray(default_kp,dtype=float)*stiffness if legacy else DEFAULT_KP),dtype=float)
    kd_on=np.asarray(kd if kd is not None else (default_kd if legacy else DEFAULT_KD),dtype=float)
    if kp_on.shape!=(6,) or kd_on.shape!=(6,) or kp_on.min()<0 or kp_on.max()>200 or kd_on.min()<0 or kd_on.max()>5:
        raise ValueError(f'gains out of SDK range: kp {kp_on.tolist()} kd {kd_on.tolist()}')
    return kp_on,kd_on


def run(link,state,cmd_q,evt_q,out_dir,stiffness,max_speed,max_gap,ki=0.0,ki_max_deg=3.0,grip_hold_nm=None,smooth=None,kp=None,kd=None):
    out_dir=Path(out_dir)
    log_file=open(out_dir/'robot_proc.log','a',buffering=1)
    os.dup2(log_file.fileno(),1);os.dup2(log_file.fileno(),2)  # never touch the terminal
    import numpy as np
    cap=_Capture();sdk_log=logging.getLogger('a1z');sdk_log.setLevel(logging.INFO);sdk_log.addHandler(cap);sdk_log.propagate=False
    sys.path.insert(0,str(ROOT))
    from teach import build_robot,preflight
    arm=bus=None;flight=open(out_dir/'flight.jsonl','a',buffering=1<<16)
    def event(*e):evt_q.put(e)
    try:
        arm,bus=build_robot(with_gripper=True)
        for attempt in range(3):  # preflight only reads state (motors stay disabled); drivers may still be booting
            try:preflight(arm,bus);break
            except RuntimeError as e:
                if attempt==2:raise
                print('preflight retry:',repr(e),flush=True);time.sleep(1.0)
        # Operator request (2026-10-04): never open the gripper at startup, so a held plug is not dropped.
        # The vendor start() runs gripper.home() = drive open. The SDK documents that homing is not needed for
        # position accuracy (calibrated flash zero), so home() becomes a no-op and the gripper starts in the
        # SDK's own zero-torque free-drive mode, which streams feedback without moving. Once its position is
        # known: closed (norm < 0.5) -> hold closed at the configured torque; open -> hold where it is.
        # Unknown position -> stays zero-torque until the operator presses Space.
        arm.gripper.home=lambda timeout=1.5:True
        arm.set_gripper_free_drive(True)
        arm.start()  # vendor gravity assist
        grip_state='unknown';deadline=time.monotonic()+1.0
        while arm.gripper._motor.last_feedback is None and time.monotonic()<deadline:time.sleep(0.01)
        if arm.gripper._motor.last_feedback is not None:
            norm=arm.gripper.get_feedback_norm();grip_state='closed' if norm<0.5 else 'open'
            arm.set_gripper_free_drive(False)  # first, so free-drive cannot overwrite the target set below
            with arm.gripper._lock:arm.gripper._cmd_norm=0.0 if norm<0.5 else float(norm)
        info=arm.get_robot_info();kp_on,kd_on=engaged_gains(info['default_kp'],info['default_kd'],stiffness,kp,kd)
        kp_off=np.zeros(6);kd_off=np.asarray(info['default_kd'])*0.5  # the SDK's own float gains
        limits=[tuple(l) for l in arm._joint_limits] if arm._joint_limits is not None else None
        mode=FLOAT;cmd=None;last=time.monotonic();last_rec=0;last_flush=time.monotonic();last_limit_warn=0
        i_off=np.zeros(6);i_max=np.radians(ki_max_deg)
        # Integral freeze (2026-10-09, HIL-SERL): in contact the joints are 'stuck' too, so integral_step kept integrating up to
        # the cap and, being per joint, pushed the plug sideways 6-14 mm across the hub (check runs 14:43-14:51). The client
        # freezes the offset near the hub: it keeps the free-air friction compensation, stops growing. Cleared on engage/float.
        ki_frozen=False
        # smooth = {'accel':rad/s^2,'soft_start_s':s,'vel_ff':bool} or None (legacy: velocity-clipped target, vel=0).
        # 2026-10-05 (session 172839): the clipped target jumps 0 -> 0.5 rad/s instantly; J2 lagged up to 5.6 deg behind it
        # (friction + kd*dq with vel=0), then broke free at up to 1.08 rad/s, 2x the command rate, and overshot. Accel
        # limiting, a slow first second and velocity feed-forward (kd acts on cmd velocity - dq, as the SDK's MIT law allows)
        # each reduce that stored lag.
        sm=smooth or {};v_cmd=np.zeros(6);t_engage=0.0
        # Two-level grip (2026-10-05: the G1Z coil reached 94 C holding the plug at the full limit for ~5 min). The SDK reads
        # gripper._i_des (current limit as a fraction of peak) on every step, so outside policy control the hold current
        # drops to grip_hold_nm; heat goes with current squared. Full force only while ENGAGED. None = vendor behaviour.
        from a1z.robots.gripper import MOTOR_PEAK_TORQUE_NM
        i_full=float(arm.gripper._i_des);i_hold=None if grip_hold_nm is None else min(i_full,float(grip_hold_nm)/MOTOR_PEAK_TORQUE_NM)
        # Raise hold->full as a ramp, not a step: on 2026-10-05 (session 170121) a step at engage made the gripper's
        # force-position loop oscillate (torque 1.7-3.7 Nm, opening 0.013-0.020, audible hum) in 6 of 20 trials and
        # twisted the plug in the fingers. Releasing (full->hold) stays a step.
        grip_goal=[i_hold if i_hold is not None else i_full]
        def grip_level(engaged):
            if i_hold is None:return
            grip_goal[0]=i_full if engaged else i_hold
            if not engaged:arm.gripper._i_des=i_hold
        grip_level(False)
        def measured():
            st=arm.get_joint_state()
            g=arm.gripper.get_feedback_norm() if arm.gripper._motor.last_feedback is not None else float('nan')
            return st,float(g)
        def thermal(st):
            """Motor temperatures (arm joints + gripper) and the gripper's measured torque, to locate audible buzzing / heating."""
            out={}
            try:out['temp_mos']=np.round(st['temp_mos'],1).tolist();out['temp_rotor']=np.round(st['temp_rotor'],1).tolist()
            except Exception:pass
            fb=arm.gripper._motor.last_feedback
            if fb is not None:
                for k_in,k_out in (('torque','g_tau'),('temperature_mos','g_temp_mos'),('temperature_rotor','g_temp_rotor'),('temp_mos','g_temp_mos'),('temp_rotor','g_temp_rotor')):
                    v=getattr(fb,k_in,None)
                    if v is not None:out[k_out]=round(float(v),3)
            return out
        def go_float(reason):
            nonlocal mode,i_off,ki_frozen
            i_off=np.zeros(6);ki_frozen=False
            st,_=measured()
            arm.command_joint_state({'pos':st['pos'],'vel':np.zeros(6),'kp':kp_off,'kd':kd_off})
            if mode==ENGAGED:event('disengaged',reason)
            mode=FLOAT;grip_level(False)
        event('ready',{'stiffness':stiffness,'kp':kp_on.tolist(),'kd':kd_on.tolist(),'limits':limits,'gripper':grip_state,'ki':ki,'ki_max_deg':ki_max_deg,
                       'grip_full_nm':round(i_full*MOTOR_PEAK_TORQUE_NM,2),'grip_hold_nm':grip_hold_nm,'smooth':smooth})
        while True:
            tick=time.monotonic()
            if not arm.is_running:event('fatal','厂家控制器已停止');break
            st,g=measured();q=np.asarray(st['pos'],dtype=float)
            while True:
                try:c=cmd_q.get_nowait()
                except queue.Empty:break
                if c[0]=='engage':
                    if not np.isfinite(q).all():event('refused','关节读数无效');continue
                    cmd=q.copy();i_off=np.zeros(6);ki_frozen=False;v_cmd=np.zeros(6);t_engage=time.monotonic();grip_level(True);arm.command_joint_state({'pos':cmd,'vel':np.zeros(6),'kp':kp_on,'kd':kd_on})
                    mode=ENGAGED;link[1]=0.0;event('engaged')
                elif c[0]=='float':go_float(c[1])
                elif c[0]=='gripper':arm.set_gripper_free_drive(False);arm.command_gripper(0.0 if c[1] else 1.0)
                elif c[0]=='grip_force':  # live change of the gripper force (2026-10-09), ramped like engage; never opens the gripper
                    # Optional 3rd field (2026-10-10): also set the floating-mode hold. Without it the hold stays at the startup
                    # grip_hold_nm, so in drag mode (calib, demos, takeovers) raising the force had no effect.
                    if len(c)>2 and c[2] is not None:grip_hold_nm=float(c[2])
                    i_full,i_hold=grip_force_levels(c[1],grip_hold_nm,MOTOR_PEAK_TORQUE_NM)
                    grip_goal[0]=i_full if (mode==ENGAGED or i_hold is None) else i_hold
                    event('grip_force',{'full_nm':round(i_full*MOTOR_PEAK_TORQUE_NM,2),'hold_nm':None if i_hold is None else round(i_hold*MOTOR_PEAK_TORQUE_NM,2)})
                elif c[0]=='integral_freeze':ki_frozen=bool(c[1])   # hold i_off as it is (True) or resume integrating (False)
                elif c[0]=='quit':raise SystemExit
            hb_age=tick-link[0] if link[0]>0 else 0.0
            if arm.gripper._i_des!=grip_goal[0]:
                span=i_full-i_hold if i_hold is not None and i_full>i_hold else i_full
                arm.gripper._i_des=grip_ramp(arm.gripper._i_des,grip_goal[0],span,min(tick-last,0.05),GRIP_RAMP_S)
            if mode==ENGAGED:
                tgt_age=time.monotonic()-link[1] if link[1]>0 else None
                if hb_age>0.5:go_float('client heartbeat lost')
                elif tgt_age is not None and tgt_age<0.5:
                    target=np.array(link[2:8],dtype=float)
                    if not np.isfinite(target).all():go_float('policy target not finite')
                    elif max_gap is not None and np.abs(target-q).max()>max_gap:
                        go_float('policy target too far from measured pose')
                    else:
                        dt=min(tick-last,0.05)
                        vmax=max_speed
                        if sm.get('soft_start_s',0)>0:vmax=max_speed*min(1.0,0.2+0.8*(tick-t_engage)/sm['soft_start_s'])
                        if sm.get('accel',0)>0:cmd,v_cmd=smooth_step(cmd,v_cmd,target,vmax,sm['accel'],dt)
                        else:cmd=cmd+np.clip(target-cmd,-vmax*dt,vmax*dt)
                        if ki>0 and not ki_frozen:i_off=integral_step(i_off,cmd,q,np.asarray(st['vel'],dtype=float),ki,dt,i_max)
                        vel=v_cmd if sm.get('vel_ff') else np.zeros(6)
                        arm.command_joint_state({'pos':cmd+i_off,'vel':vel,'kp':kp_on,'kd':kd_on})
            elif hb_age>5.0:event('fatal','客户端失联超过 5 秒');break
            if limits and tick-last_limit_warn>1.0:
                over=[(i+1,float(np.degrees(v-(lo if v<lo else hi)))) for i,(v,(lo,hi)) in enumerate(zip(q,limits)) if v<lo-0.05 or v>hi+0.05]
                if over:last_limit_warn=tick;event('limit',over)
            hz=cap.hz
            vals=[tick,mode,hz,g,*q,*st['vel'],*st['eff'],*(cmd if cmd is not None else q)]
            for i,v in enumerate(vals):state[i]=float(v)
            if tick-last_rec>=0.02:
                last_rec=tick
                with arm._command_lock:kp_now=arm._command.kp.tolist();cpos=arm._command.pos.tolist()
                # grip = measured opening (NaN without feedback), grip_cmd = commanded opening: the same two values the
                # demonstrations stored as gripper_position / gripper_target, so intervention data exports identically.
                th=thermal(st)
                gt=[th[k] for k in ('g_temp_mos','g_temp_rotor') if k in th];jt=th.get('temp_rotor') or th.get('temp_mos')
                state[STATE_N-2]=float(max(gt)) if gt else float('nan');state[STATE_N-1]=float(max(jt)) if jt else float('nan')
                flight.write(json.dumps({'t':round(tick,4),'mode':int(mode),'hz':hz,'q':q.round(5).tolist(),'dq':np.round(st['vel'],4).tolist(),
                    'eff':np.round(st['eff'],3).tolist(),'cmd_pos':np.round(cpos,5).tolist(),'kp':kp_now,'grip':g,
                    'grip_cmd':float(arm.gripper.get_pos()),'g_i':round(float(arm.gripper._i_des)*MOTOR_PEAK_TORQUE_NM,3),'hb_age':round(hb_age,3),'v_cmd':np.round(v_cmd,4).tolist(),'i_off':np.round(i_off,5).tolist(),'ki_frozen':ki_frozen,**th})+'\n')
            while not cap.records.empty():
                lvl,msg=cap.records.get()
                flight.write(json.dumps({'t':round(tick,4),'sdk':lvl,'msg':msg})+'\n')
            if tick-last_flush>1.0:flight.flush();last_flush=tick
            last=tick
            time.sleep(max(0.0,0.01-(time.monotonic()-tick)))  # 100 Hz supervisor; the SDK loop runs at its own 250 Hz
    except SystemExit:pass
    except Exception as e:event('fatal',repr(e))
    finally:
        if arm is not None:arm.stop()
        if bus is not None:bus.shutdown()
        flight.flush();flight.close()
        event('stopped')
