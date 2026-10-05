#!/usr/bin/env python3
"""Offline test of control_test.py with a fake robot, USB watcher and kinematics (no hardware):
t needs USB; t stores the pose; g moves back to it and ends on USB; r records a drag that stops itself after USB;
p moves to the recorded start, replays, and ends on USB. The fake arm tracks targets with a first-order lag."""
import builtins
import json
from pathlib import Path
import sys
import tempfile
import time
import types

import numpy as np

ROOT=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT))
TAUGHT=np.array([0.1,0.2,0.3,0.0,0.1,0.0]);START=TAUGHT+np.array([0.15,-0.1,0.05,0,0,0])


class FakeRobot:
    def __init__(self,cfg):
        self.config=cfg;self.log=[];self.events=[];self.disengage_reason=None;self.engaged=False;self.ready_info={'gripper':'closed'}
        self.q=START.copy();self.target=None;self.calls=[]
    def connect(self):pass
    def heartbeat(self):pass
    def poll_events(self):return []
    @property
    def is_connected(self):return True
    def measured(self):
        if self.engaged and self.target is not None:self.q=self.q+0.3*(self.target-self.q)
        return self.q.copy(),0.0
    def commanded(self):return self.q.copy()
    def send_target(self,target,stage,**extra):
        self.target=np.asarray(target,float);self.log.append({'t':time.monotonic(),'stage':stage,'q':self.q.tolist(),'policy':self.target.tolist(),'cmd':self.target.tolist()})
    def engage(self):self.engaged=True;self.target=None;self.calls.append('engage');return True
    def disengage(self,reason='operator'):
        if self.engaged:self.disengage_reason=reason
        self.engaged=False;self.calls.append('float:'+reason)
    def start_video(self,p):self.calls.append('video')
    def stop_video(self):pass
    def command_gripper(self,c):self.calls.append(f'grip:{c}')
    def disconnect(self):pass


class FakeUsb:
    """Hand-set while floating; while the arm is engaged it connects when the plug reaches the taught pose."""
    def __init__(self):self.manual=False;self.robot=None
    @property
    def connected(self):
        r=self.robot
        if r is not None and r.engaged:return bool(np.abs(r.q-TAUGHT).max()<2e-3)
        return self.manual
    def start(self):return self
    def stop(self):pass
    def describe(self):return {}


def main():
    import control_test,a1z_robot,usb_watch
    ins=types.ModuleType('insertion')
    class Kin:
        def fk(self,q):T=np.eye(4);T[:3,3]=np.asarray(q[:3])/10;return T  # metres
    ins.Kin=Kin;sys.modules['insertion']=ins
    usb=FakeUsb();robots={}
    def make(cfg):
        r=robots.setdefault('r',FakeRobot(cfg));usb.robot=r;return r
    a1z_robot.A1Z=make;usb_watch.UsbWatcher=lambda:usb
    builtins.input=lambda *_:'READY'
    def drag(to,secs):
        r=robots['r'];q0=r.q.copy();t0=time.monotonic()
        while (w:=min(1,(time.monotonic()-t0)/secs))<1:r.q=q0+(to-q0)*w;time.sleep(0.01)
        r.q=to.copy()
    script=[(0.3,'t'),                                   # refused: USB not connected
            (0.5,lambda:drag(TAUGHT,0.3)),(0.9,'usb1'),(1.1,'t'),(2.9,'t'),
            (1.3,'usb0'),(1.4,lambda:drag(TAUGHT+[0,0,0.05,0,0,0],0.2)),(1.8,'g'),   # A1 from ~5 mm above
            (5.5,'1'),                                   # relabel run 1 by hand
            (5.8,lambda:drag(START,0.2)),(6.2,'r'),(6.3,lambda:drag(TAUGHT,1.0)),(7.4,'usb1'),
            (8.3,'usb0'),(8.4,lambda:drag(START+0.02,0.2)),(8.8,'p'),
            (16.0,'q'),(16.1,'q')]
    def fake_keys(events,stop):
        t0=time.monotonic()
        for at,k in script:
            while time.monotonic()-t0<at:time.sleep(0.01)
            if callable(k):k()
            elif k=='usb1':usb.manual=True
            elif k=='usb0':usb.manual=False
            else:events.put(k)
    control_test.keys=fake_keys
    with tempfile.TemporaryDirectory() as tmp:
        control_test.ROOT=Path(tmp);control_test.RUNS=Path(tmp)/'control-runs'
        sys.argv=['control_test.py','--approach-rate','0.5','--hold','3','--replay-speed','1','--after-insert','0.3']
        control_test.main()
        sess=next(p for p in control_test.RUNS.iterdir() if p.is_dir())
        taught=json.loads((control_test.RUNS/'taught.json').read_text())
        assert np.allclose(taught['q'],TAUGHT),taught
        r1=json.loads((sess/'run-01.json').read_text());r2=json.loads((sess/'run-02.json').read_text())
        assert r1['kind']=='A1 示教点' and r1['success'] and r1['end_reason']=='usb_inserted',(r1['kind'],r1['end_reason'])
        assert r1['operator_label']=='missed' and not r1['operator_label_auto'],r1['operator_label']
        assert abs(r1['start_offset_mm'][2]-5.0)<0.5 and max(abs(x) for x in r1['end_dxyz_mm'])<0.5,(r1['start_offset_mm'],r1['end_dxyz_mm'])
        demo=json.loads(next(sess.glob('demo-*.json')).read_text())
        assert demo['usb_at'] is not None and demo['samples'][-1]['t']>=demo['usb_at']+0.45,demo['usb_at']
        assert np.allclose(demo['samples'][0]['q'],START,atol=1e-6)
        assert r2['kind']=='A2 回放' and r2['success'] and r2['end_reason']=='usb_inserted',(r2['kind'],r2['end_reason'])
        assert r2['operator_label']=='inserted' and r2['operator_label_auto']
        print('control test OK',robots['r'].calls,r1['end_dxyz_mm'],r2['duration_s'])

if __name__=='__main__':main()
