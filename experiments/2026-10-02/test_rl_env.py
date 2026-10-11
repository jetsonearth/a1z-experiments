#!/usr/bin/env python3
"""Offline tests for rl_env.py with a fake arm: the plug follows the command with a rate limit and cannot go below the
hub surface unless it is over the port. No hardware, no pinocchio, no torch."""
import json
import tempfile
import threading
import time
import unittest

import numpy as np

from rl_env import Clock,EnvCfg,InsertEnv,Port,SpiralPolicy

P_INS=np.array([467.99,94.74,166.58]);Z_MOUTH=171.34;Z_SURF=175.7;TOL=1.2


class FakePin:
    @staticmethod
    def log3(R):return np.zeros(3)
    @staticmethod
    def exp3(v):return np.eye(3)


class FakeKin:
    """Joint vector q[:3] is the finger position in metres."""
    pin=FakePin()
    def fk(self,q):
        T=np.eye(4);T[:3,3]=np.asarray(q[:3],dtype=float);return T
    def step_towards(self,q,p,R,iters=3,damping=1e-3):
        out=np.zeros(6);out[:3]=p;return out


class FakeRobot:
    def __init__(self,p):
        self.p=np.array(p,dtype=float);self.target=self.p.copy();self.engaged=False;self.disengage_reason=None
        self._frame_lock=threading.Lock();self._frame=np.zeros((480,640,3),np.uint8);self.sent=[]
    @property
    def _frame_t(self):return time.monotonic()
    def heartbeat(self):
        inside=self.p[2]<Z_SURF-0.3
        if self.engaged:
            d=np.clip(self.target-self.p,-1.0,1.0);self.p=self.p+d
        if inside:   # in the port the walls hold the plug laterally
            v=self.p[:2]-P_INS[:2];n=np.hypot(*v)
            if n>TOL:self.p[:2]=P_INS[:2]+v*TOL/n
        over=np.hypot(*(self.p[:2]-P_INS[:2]))<=TOL
        self.p[2]=max(self.p[2],P_INS[2] if over else Z_SURF)
    def poll_events(self):return []
    def measured(self):
        q=np.zeros(6);q[:3]=self.p/1000.0;return q,0.0
    def engage(self):self.engaged=True;self.target=self.p.copy();return True
    def disengage(self,reason='x'):self.engaged=False;self.disengage_reason=reason
    def send_target(self,q,stage,**k):self.target=np.asarray(q[:3])*1000.0;self.sent.append(self.target.copy())


class FakeUsb:
    connected=False
    def describe(self):return {'fake':True}


def calib_file(**over):
    s=[{'label':'surface','p_mm':[466.9,104.7,Z_SURF],'q':[0]*6},{'label':'mouth','p_mm':[P_INS[0],P_INS[1],Z_MOUTH],'q':[0]*6},
       {'label':'inserted','p_mm':P_INS.tolist(),'q':(P_INS/1000).tolist()+[0,0,0],'usb_connected':True}]
    for x in s:
        if x['label'] in over:x['p_mm']=over[x['label']]
    f=tempfile.NamedTemporaryFile('w',suffix='.json',delete=False);json.dump({'samples':s},f);f.close();return f.name


def make(start_p=(470,100,195),**cfg):
    kin=FakeKin();robot=FakeRobot(start_p);port=Port(calib_file(),kin)
    env=InsertEnv(robot,kin,port,FakeUsb(),Clock(robot,hz=1000),EnvCfg(**cfg))
    return env,robot


class TestEnv(unittest.TestCase):
    def test_port_validation(self):
        with self.assertRaises(ValueError):Port(calib_file(mouth=[467.5,94.7,175.0]),FakeKin())

    def test_port_ignores_slipped_insertion(self):
        port=Port(calib_file(inserted=[473.0,86.0,161.0]),FakeKin())   # the gripper slid on the plug when pushed home
        self.assertAlmostEqual(port.p_port[0],P_INS[0]);self.assertAlmostEqual(port.z_mouth,Z_MOUTH)

    def test_contact_height_check(self):
        env,robot=make()
        env.reset({'offset_mm':[0,5.0],'hover':False,'hover_dz_mm':0})
        self.assertLessEqual(abs(env.contact_dz),0.05)
        env.reset({'offset_mm':[0,-4.0],'hover':True,'hover_dz_mm':3.0})
        self.assertIsNone(env.contact_dz)

    def test_reset_contact_and_hover(self):
        env,robot=make()
        obs=env.reset({'offset_mm':[0,5.0],'hover':False,'hover_dz_mm':0})
        self.assertIsNotNone(obs);self.assertAlmostEqual(robot.p[2],Z_SURF,delta=0.01);self.assertLess(env.p_cmd[2],robot.p[2])
        self.assertEqual(obs['image'].shape,(128,128,3));self.assertEqual(obs['state'].shape,(9,))
        self.assertFalse(env.in_port())
        obs=env.reset({'offset_mm':[-4.0,0],'hover':True,'hover_dz_mm':4.0})
        self.assertAlmostEqual(robot.p[2],Z_SURF+4.0,delta=0.2)

    def test_success_by_moving_to_port(self):
        env,robot=make()
        env.reset({'offset_mm':[0,5.0],'hover':False,'hover_dz_mm':0})
        for i in range(80):
            pm,_=env.p_meas();a=np.r_[(P_INS[:2]-env.p_cmd[:2])/1.5,-1.0]
            obs,r,term,trunc,info=env.step(a)
            if term:break
        self.assertTrue(term);self.assertEqual(r,1.0);self.assertLessEqual(robot.p[2],Z_MOUTH+0.5)

    def test_pressing_beside_port_is_not_success(self):
        env,robot=make()
        env.reset({'offset_mm':[0,-5.0],'hover':False,'hover_dz_mm':0})
        for i in range(20):
            obs,r,term,trunc,info=env.step([0,0,-1])
            self.assertFalse(term);self.assertGreaterEqual(info['p_cmd'][2],info['p_meas'][2]-5.0-1e-6)
        self.assertIn('press',info['clamped'])

    def test_box_lead_and_truncation(self):
        env,robot=make(max_steps=30)
        env.reset({'offset_mm':[5.0,0],'hover':True,'hover_dz_mm':5.0})
        for i in range(30):
            obs,r,term,trunc,info=env.step([1,1,0])
            self.assertLessEqual(abs(info['p_cmd'][0]-env.p_start[0]),10.0+1e-6)
            self.assertLessEqual(np.hypot(*(np.array(info['p_cmd'][:2])-np.array(info['p_meas'][:2]))),4.0+1e-6)
        self.assertTrue(trunc);self.assertFalse(term)

    def test_human_step_label(self):
        env,robot=make()
        env.reset({'offset_mm':[0,5.0],'hover':True,'hover_dz_mm':3.0})
        robot.disengage('takeover')
        orig=robot.heartbeat
        def drag():
            robot.p[1]-=1.0;orig()
        robot.heartbeat=drag
        obs,r,term,trunc,info,a=env.human_step()
        self.assertTrue(info['human']);self.assertAlmostEqual(a[1],-1.0);self.assertIn('label_clip',info['clamped'])
        robot.heartbeat=orig;env.handback();self.assertTrue(robot.engaged)

    def test_spiral_finds_small_offset(self):
        env,robot=make(max_steps=200)
        obs=env.reset({'offset_mm':[1.5,0],'hover':False,'hover_dz_mm':0})
        self.assertIsNotNone(obs)
        pol=SpiralPolicy();pol.reset(env)
        for i in range(200):
            obs,r,term,trunc,info=env.step(pol(env,obs) if i else pol(env,None))
            if term or trunc:break
        self.assertTrue(term)


if __name__=='__main__':unittest.main()
