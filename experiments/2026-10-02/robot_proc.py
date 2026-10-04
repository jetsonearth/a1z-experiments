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
  state (robot -> client): [t, mode, sdk_hz, gripper, q(6), dq(6), eff(6), cmd(6)]
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
STATE_N=4+6*4
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


def run(link,state,cmd_q,evt_q,out_dir,stiffness,max_speed,max_gap):
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
        info=arm.get_robot_info();kp_on=np.asarray(info['default_kp'])*stiffness;kd_on=np.asarray(info['default_kd'])
        kp_off=np.zeros(6);kd_off=np.asarray(info['default_kd'])*0.5  # the SDK's own float gains
        limits=[tuple(l) for l in arm._joint_limits] if arm._joint_limits is not None else None
        mode=FLOAT;cmd=None;last=time.monotonic();last_rec=0;last_flush=time.monotonic();last_limit_warn=0
        def measured():
            st=arm.get_joint_state()
            g=arm.gripper.get_feedback_norm() if arm.gripper._motor.last_feedback is not None else float('nan')
            return st,float(g)
        def go_float(reason):
            nonlocal mode
            st,_=measured()
            arm.command_joint_state({'pos':st['pos'],'vel':np.zeros(6),'kp':kp_off,'kd':kd_off})
            if mode==ENGAGED:event('disengaged',reason)
            mode=FLOAT
        event('ready',{'stiffness':stiffness,'kp':kp_on.tolist(),'limits':limits,'gripper':grip_state})
        while True:
            tick=time.monotonic()
            if not arm.is_running:event('fatal','厂家控制器已停止');break
            st,g=measured();q=np.asarray(st['pos'],dtype=float)
            while True:
                try:c=cmd_q.get_nowait()
                except queue.Empty:break
                if c[0]=='engage':
                    if not np.isfinite(q).all():event('refused','关节读数无效');continue
                    cmd=q.copy();arm.command_joint_state({'pos':cmd,'vel':np.zeros(6),'kp':kp_on,'kd':kd_on})
                    mode=ENGAGED;link[1]=0.0;event('engaged')
                elif c[0]=='float':go_float(c[1])
                elif c[0]=='gripper':arm.set_gripper_free_drive(False);arm.command_gripper(0.0 if c[1] else 1.0)
                elif c[0]=='quit':raise SystemExit
            hb_age=tick-link[0] if link[0]>0 else 0.0
            if mode==ENGAGED:
                tgt_age=time.monotonic()-link[1] if link[1]>0 else None
                if hb_age>0.5:go_float('client heartbeat lost')
                elif tgt_age is not None and tgt_age<0.5:
                    target=np.array(link[2:8],dtype=float)
                    if not np.isfinite(target).all() or np.abs(target-q).max()>max_gap:
                        go_float('policy target too far from measured pose')
                    else:
                        dt=min(tick-last,0.05)
                        cmd=cmd+np.clip(target-cmd,-max_speed*dt,max_speed*dt)
                        arm.command_joint_state({'pos':cmd,'vel':np.zeros(6),'kp':kp_on,'kd':kd_on})
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
                flight.write(json.dumps({'t':round(tick,4),'mode':int(mode),'hz':hz,'q':q.round(5).tolist(),'dq':np.round(st['vel'],4).tolist(),
                    'eff':np.round(st['eff'],3).tolist(),'cmd_pos':np.round(cpos,5).tolist(),'kp':kp_now,'grip':g,'hb_age':round(hb_age,3)})+'\n')
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
