#!/usr/bin/env python3
"""LeRobot Robot adapter for the A1Z arm + G1Z gripper + D405 wrist camera.

The arm runs in its own process (robot_proc.py) so the vendor SDK keeps its 250 Hz loop to itself;
this class lives in the client process with the camera and LeRobot's gRPC client, and talks to the
arm through shared memory. Safety logic (rate limit, target-gap check, heartbeat) runs in the robot
process. The gripper is operator-owned: the policy's gripper output is logged, never applied.
"""
from dataclasses import dataclass
import json
from pathlib import Path
import multiprocessing as mp
import queue
import threading
import time

import numpy as np
from lerobot.robots.config import RobotConfig
from lerobot.robots.robot import Robot

import robot_proc

JOINTS=[f'j{i}' for i in range(1,7)]
LIVE_JPEG='/dev/shm/a1z_live.jpg'  # read by console/server.py


@RobotConfig.register_subclass('a1z')
@dataclass
class A1ZConfig(RobotConfig):
    stiffness:float|None=None    # legacy: kp = stiffness x vendor kp, vendor kd; None = robot_proc.DEFAULT_KP/KD (dimos)
    max_joint_speed:float=4.0    # rad/s target rate = the SDK's own commanded-velocity saturation (our 0.5 cap removed 2026-10-09)
    max_target_gap:float|None=None  # rad; None = no target-gap stop (our 20 deg stop removed 2026-10-09, user: too conservative)
    integral_gain:float=2.0      # 1/s, outer integral on joint position (0 = off); default on since 2026-10-09
    integral_max_deg:float=1.0   # clamp of the integral offset per joint (1 deg with the stiffer default gains)
    smooth:dict|None=None        # {'accel','soft_start_s','vel_ff'}; None = legacy velocity-clipped target
    grip_hold_nm:float|None=1.5  # gripper force outside policy control (None = configured full force always)
    kp:list|None=None            # per-joint kp while engaged; overrides stiffness x vendor kp (e.g. dimos 80,80,80,50,20,20)
    kd:list|None=None            # per-joint kd while engaged; None = vendor default kd
    out_dir:str='.'
    width:int=640
    height:int=480
    fps:int=30


class A1Z(Robot):
    config_class=A1ZConfig
    name='a1z'

    def __init__(self,config:A1ZConfig):
        super().__init__(config)
        self.config=config
        self._proc=None;self._pipe=None
        self._frame=None;self._frame_t=0.0;self._frame_lock=threading.Lock();self._cam_stop=threading.Event()
        self._video=None;self._video_lock=threading.Lock();self._video_path=None;self._video_frames=[]
        self.disengage_reason=None;self.log=[];self.events=[];self._live_t=0.0

    @property
    def observation_features(self):
        return {**{j:float for j in JOINTS},'gripper':float,'wrist':(self.config.height,self.config.width,3)}

    @property
    def action_features(self):
        return {**{j:float for j in JOINTS},'gripper':float}

    @property
    def is_connected(self):
        return self._proc is not None and self._proc.is_alive()

    @property
    def is_calibrated(self):return True
    def calibrate(self):pass
    def configure(self):pass

    # ---------- camera (client process) ----------
    def _camera_loop(self):
        while not self._cam_stop.is_set():
            try:frames=self._pipe.wait_for_frames(1000)
            except RuntimeError:continue
            color=frames.get_color_frame()
            if not color:continue
            host=time.monotonic()  # host receipt time, the same clock and point as the demo camera recorder
            image=np.asanyarray(color.get_data()).copy()
            with self._frame_lock:self._frame=image;self._frame_t=host
            if host-self._live_t>=0.2:self._live_t=host;self._write_live(image)
            with self._video_lock:
                if self._video is not None:
                    self._video.write(image.tobytes());self._video_frames.append({'host_monotonic_s':host,'rgb_frame_number':color.get_frame_number()})

    def _write_live(self,image):
        """Latest wrist frame for the console's live view (~5 Hz JPEG in /dev/shm; never blocks the camera loop on errors)."""
        try:
            import cv2,os
            ok,buf=cv2.imencode('.jpg',image[:,:,::-1],[cv2.IMWRITE_JPEG_QUALITY,70])
            if ok:
                tmp=LIVE_JPEG+'.tmp';open(tmp,'wb').write(buf.tobytes());os.replace(tmp,LIVE_JPEG)
        except Exception:pass

    def start_video(self,path):
        """Per-trial wrist video with the demo recorder's encoder (H.264 ultrafast/zerolatency, crf 23) and a
        per-frame timestamp file, so trial frames can be aligned and exported exactly like demonstrations."""
        from camera_session import StreamingVideo
        with self._video_lock:
            self._video=StreamingVideo(path,self.config.width,self.config.height,self.config.fps)
            self._video_path=Path(path);self._video_frames=[]

    def stop_video(self):
        with self._video_lock:
            if self._video is None:return
            video,path,frames=self._video,self._video_path,self._video_frames;self._video=None
        error=None
        try:video.close()
        except Exception as e:error=repr(e)
        path.with_suffix('.timestamps.json').write_text(json.dumps({'video':path.name,'encoder':'libx264 ultrafast zerolatency crf23 (camera_session.StreamingVideo)',
                                                                  'timing_boundary':'host receipt, not calibrated exposure','encoder_error':error,'frames':frames}))

    def frame_age(self):
        with self._frame_lock:return time.monotonic()-self._frame_t if self._frame is not None else float('inf')

    # ---------- arm (robot process) ----------
    def connect(self,calibrate:bool=True):
        if self._proc is not None:return
        import pyrealsense2 as rs
        from wrist_camera import enable_wrist
        self._pipe=rs.pipeline();cfg=rs.config();self.wrist_serial=enable_wrist(rs,cfg)
        cfg.enable_stream(rs.stream.color,self.config.width,self.config.height,rs.format.rgb8,self.config.fps)  # same as recording
        # Depth must be on too, exactly as in recording: with color alone the D405 color image is ~40% darker
        # under the same light (measured 76 vs 124), which put every inference image off the training distribution.
        cfg.enable_stream(rs.stream.depth,self.config.width,self.config.height,rs.format.z16,self.config.fps)
        self._pipe.start(cfg);self._pipe.wait_for_frames(3000)
        threading.Thread(target=self._camera_loop,daemon=True).start()
        ctx=mp.get_context('spawn')
        self.link=ctx.Array('d',robot_proc.LINK_N,lock=False);self.state=ctx.Array('d',robot_proc.STATE_N,lock=False)
        for i in range(robot_proc.STATE_N):self.state[i]=float('nan')
        self.cmd_q=ctx.Queue();self.evt_q=ctx.Queue()
        self._proc=ctx.Process(target=robot_proc.run,args=(self.link,self.state,self.cmd_q,self.evt_q,self.config.out_dir,
                               self.config.stiffness,self.config.max_joint_speed,self.config.max_target_gap,
                               self.config.integral_gain,self.config.integral_max_deg,self.config.grip_hold_nm,self.config.smooth,self.config.kp,self.config.kd),daemon=False)
        self._proc.start()
        deadline=time.monotonic()+40
        while time.monotonic()<deadline:
            self.heartbeat()
            for e in self.poll_events():
                if e[0]=='ready':self.ready_info=e[1];return
                if e[0]=='fatal':raise RuntimeError('机械臂进程启动失败：'+str(e[1:]))
            time.sleep(0.05)
        raise RuntimeError('机械臂进程启动超时')

    def heartbeat(self):
        self.link[0]=time.monotonic()

    def poll_events(self):
        out=[]
        while True:
            try:e=self.evt_q.get_nowait()
            except queue.Empty:break
            if e[0]=='disengaged':self.disengage_reason=e[1]
            out.append(e);self.events.append((time.time(),)+tuple(e))
        return out

    @property
    def engaged(self):return self.state[1]==robot_proc.ENGAGED

    @property
    def sdk_hz(self):return self.state[2]

    def measured(self):
        s=self.state
        return np.array(s[4:10],dtype=float),float(s[3])

    def commanded(self):
        return np.array(self.state[22:28],dtype=float)

    def temps(self):
        """(gripper motor temperature, hottest arm joint) in deg C from the robot process; NaN until the first reading."""
        return float(self.state[robot_proc.STATE_N-2]),float(self.state[robot_proc.STATE_N-1])

    def get_observation(self):
        q,g=self.measured()
        with self._frame_lock:image=self._frame;age=time.monotonic()-self._frame_t
        if image is None or age>0.25:raise RuntimeError(f'Wrist camera stale ({age:.2f}s)')
        if not np.isfinite(q).all() or not np.isfinite(g):raise RuntimeError('Joint/gripper feedback not valid yet')
        return {**{j:float(v) for j,v in zip(JOINTS,q)},'gripper':g,'wrist':image}

    def engage(self,timeout=1.0):
        self.disengage_reason=None;self.cmd_q.put(('engage',))
        deadline=time.monotonic()+timeout
        while time.monotonic()<deadline:
            self.heartbeat()
            if self.engaged:return True
            time.sleep(0.01)
        return False

    def disengage(self,reason='operator'):
        self.cmd_q.put(('float',reason))
        deadline=time.monotonic()+1.0
        while self.engaged and time.monotonic()<deadline:self.heartbeat();time.sleep(0.01)
        if self.disengage_reason is None:self.disengage_reason=reason

    def send_action(self,action):
        target=np.array([action[j] for j in JOINTS],dtype=np.float64)
        self.send_target(target,'policy',policy_gripper=float(action.get('gripper',np.nan)))
        return action

    def send_target(self,target,stage,**extra):
        """Joint target from the policy or the scripted search; the robot process applies all safety checks."""
        if self.engaged:
            q,_=self.measured()
            self.link[2:8]=np.asarray(target,dtype=float).tolist();self.link[1]=time.monotonic()
            self.log.append({'t':time.monotonic(),'stage':stage,'q':q.tolist(),'policy':np.asarray(target,dtype=float).tolist(),
                             'cmd':self.commanded().tolist(),**extra})

    def freeze_integral(self,frozen:bool):
        """Hold the integral offset at its current value (True) or resume integrating (False); see robot_proc."""
        self.cmd_q.put(('integral_freeze',bool(frozen)))

    def command_gripper(self,closed:bool):
        self.cmd_q.put(('gripper',bool(closed)))

    def set_grip_force(self,nm:float,hold_nm:float|None=None):
        """Change the gripper force (N m at the motor) live; robot_proc ramps to it and answers with a grip_force event.
        hold_nm also changes the force while floating (drag mode); None keeps the hold set at startup."""
        self.cmd_q.put(('grip_force',float(nm),None if hold_nm is None else float(hold_nm)))

    def disconnect(self):
        self._cam_stop.set();self.stop_video()
        if self._proc is not None:
            self.cmd_q.put(('quit',))
            self._proc.join(timeout=8)
            if self._proc.is_alive():self._proc.terminate()
            self._proc=None
        if self._pipe is not None:
            try:self._pipe.stop()
            except Exception:pass
