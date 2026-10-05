#!/usr/bin/env python3
"""Offline test of a1z_client --hil key flow with a fake robot, USB watcher and LeRobot client (no hardware):
g starts, Space takes over (arm floats, policy paused), Space hands back (queue cleared, re-engaged),
Space takes over again, USB connects -> trial ends; segments and labels are recorded; d discards."""
import builtins
import json
from pathlib import Path
from queue import Queue
import sys
import tempfile
import threading
import time
import types

import numpy as np

ROOT=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT))


class FakeRobot:
    def __init__(self,cfg):self.config=cfg;self.log=[];self.events=[];self.disengage_reason=None;self.engaged=False;self.ready_info={'gripper':'closed'};self.calls=[]
    def connect(self):pass
    def heartbeat(self):pass
    def poll_events(self):return []
    def frame_age(self):return 0.0
    @property
    def is_connected(self):return True
    sdk_hz=250.0
    def measured(self):return np.zeros(6),0.0
    def commanded(self):return np.zeros(6)
    def engage(self):self.engaged=True;self.disengage_reason=None;self.calls.append('engage');return True
    def disengage(self,reason='operator'):
        if self.engaged:self.disengage_reason=reason
        self.engaged=False;self.calls.append('float:'+reason)
    def start_video(self,p):self.calls.append('video')
    def stop_video(self):self.calls.append('stop_video')
    def command_gripper(self,c):self.calls.append(f'grip:{c}')
    def disconnect(self):pass


class FakeUsb:
    connected=False
    def start(self):return self
    def stop(self):pass
    def describe(self):return {}


class FakeClient:
    instance=None
    def __init__(self,cfg):
        FakeClient.instance=self;self.action_chunk_size=30
        self.action_queue=Queue();self.action_queue_lock=threading.Lock();self.latest_action=5;self.latest_action_lock=threading.Lock()
        self.must_go=threading.Event();self.logger=types.SimpleNamespace(setLevel=lambda *_:None);self.start_barrier=types.SimpleNamespace(wait=lambda:None)
        self.steps=0
    def start(self):return True
    def receive_actions(self):pass
    def actions_available(self):return True
    def control_loop_action(self):self.steps+=1
    def _ready_to_send_observation(self):return False
    def stop(self):pass


def main():
    import a1z_client,a1z_robot,usb_watch
    fake_robot={}
    a1z_robot.A1Z=lambda cfg:fake_robot.setdefault('r',FakeRobot(cfg))
    usb=FakeUsb();usb_watch.UsbWatcher=lambda:usb
    rc=types.ModuleType('lerobot.async_inference.robot_client');client={}
    rc.RobotClient=FakeClient
    sys.modules['lerobot.async_inference.robot_client']=rc
    import lerobot.async_inference
    lerobot.async_inference.robot_client=rc
    builtins.input=lambda *_:'READY'
    script=[(0.3,'g'),(0.6,' '),(0.9,' '),(1.2,' '),(1.5,'USB'),(2.8,'d'),(3.0,'q'),(3.1,'q')]
    def fake_keys(events,stop):
        t0=time.monotonic()
        for at,k in script:
            while time.monotonic()-t0<at:time.sleep(0.01)
            if k=='USB':usb.connected=True
            else:events.put(k)
    a1z_client.keys=fake_keys
    with tempfile.TemporaryDirectory() as tmp:
        a1z_client.ROOT=Path(tmp)
        sys.argv=['a1z_client.py','--checkpoint','ck','--hil','--epoch','1','--after-insert','0.5']
        a1z_client.main()
        sess=next((Path(tmp)/'hil-runs').iterdir());t=json.loads((sess/'trial-01.json').read_text())
        ctl=[s['controller'] for s in t['segments']]
        assert ctl==['policy','human','policy','human'],ctl
        assert all(s['t1']>s['t0'] for s in t['segments'])
        assert t['interventions']==2 and t['success'] and not t['autonomous_success'],t
        assert t['end_reason']=='usb_inserted',t['end_reason']
        assert t['hil_keep'] is False  # d pressed after the trial
        r=fake_robot['r'];c=FakeClient.instance
        assert r.calls[:4]==['video','engage','float:takeover','engage'],r.calls
        assert c.latest_action==2005  # reset at start and at the one handback
        assert not any(x.startswith('grip') for x in r.calls)  # Space never toggled the gripper during the trial
        print('hil client OK',ctl,r.calls)

if __name__=='__main__':main()
