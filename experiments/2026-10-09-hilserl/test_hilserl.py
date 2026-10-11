#!/usr/bin/env python3
"""Offline tests: policy weight round trip, transition format, buffer routing, the full-insertion env on the fake arm,
the reward classifier on synthetic side frames, and an end-to-end run (learner.py + actor.py --fake over real gRPC on
localhost, classifier as the reward, USB as the referee). No hardware.

  python -m unittest test_hilserl            # all (the end-to-end test takes ~1-2 min on a laptop CPU)
"""
import json
import os
from pathlib import Path
import socket
import subprocess
import sys
import tempfile
import unittest

import numpy as np

import common
from common import IMG,IS_DEMO,IS_INTERVENTION,STATE,VIEW_KEYS,ImageBuffer
from env import FullCfg,FullInsertEnv,UsbStillConnected
from side_camera import FakeSideCamera
import test_rl_env as T
from rl_env import Clock,Port

ROOT=Path(__file__).resolve().parent


class FakeUsbLinked:
    """Enumerates while the fake plug sits at full depth in the port (FakeRobot lets it drop there only over the port)."""
    def __init__(self,robot):self.robot=robot
    @property
    def connected(self):
        p=self.robot.p;return bool(np.hypot(*(p[:2]-T.P_INS[:2]))<=T.TOL and p[2]<=T.P_INS[2]+0.3)
    def describe(self):return {'fake':'linked to the fake plug depth'}


def make_env(clf=None,**cfg):
    cfg.setdefault('reward','classifier' if clf else 'usb')
    kin=T.FakeKin();robot=T.FakeRobot((470,100,195));port=Port(T.calib_file(),kin);usb=FakeUsbLinked(robot)
    robot.ready_info={'kp':[1e4]*6}   # fake joints are metres: 1 N of press moves the fake target 0.1 mm
    env=FullInsertEnv(robot,kin,port,usb,Clock(robot,hz=1000),FullCfg(**cfg),side=FakeSideCamera(usb),clf=clf);return env,robot


def drive_in(env,robot,steps=100):
    """Scripted: over the port, then down. Returns the last step's (r,term,info)."""
    for i in range(steps):
        a=np.clip(np.r_[(T.P_INS[:2]-env.p_cmd[:2])/1.5,-1.0 if np.hypot(*(robot.p[:2]-T.P_INS[:2]))<0.5 else 0.0],-1,1)
        obs,r,term,trunc,info=env.step(a)
        if term:break
    return r,term,info


def synthetic_clf_session(d,n=300):
    """Classifier training data in the classify-mode format: side frames from the fake camera (bright patch = inserted)."""
    rng=np.random.default_rng(0);cam=FakeSideCamera(type('U',(),{'connected':False})())
    import cv2
    y=(rng.uniform(size=n)<0.25)
    side=[];wrist=[]
    for k in y:
        cam.usb.connected=bool(k);img=cam.latest().astype(np.int16)+rng.integers(-20,20,(480,640,3))
        side.append(cv2.resize(np.clip(img,0,255).astype(np.uint8),(128,128),interpolation=cv2.INTER_AREA));wrist.append(np.zeros((128,128,3),np.uint8))
    np.savez_compressed(Path(d)/'clf-001.npz',frames=np.stack(wrist),frames_side=np.stack(side),labels=y,usb=y,t=np.arange(n)*0.1)


def fake_episode(T_=6,human=None,success=True):
    rng=np.random.default_rng(0)
    return {'frames':rng.integers(0,255,(T_+1,128,128,3),dtype=np.uint8),'frames_side':rng.integers(0,255,(T_+1,128,128,3),dtype=np.uint8),'states':rng.normal(size=(T_+1,9)).astype(np.float32),
            'actions':rng.uniform(-1,1,(T_,3)).astype(np.float32),'rewards':np.r_[np.zeros(T_-1),float(success)].astype(np.float32),
            'terminals':np.r_[np.zeros(T_-1),float(success)].astype(np.float32),
            'human':np.zeros(T_,bool) if human is None else np.asarray(human,bool)}


class TestPolicyAndData(unittest.TestCase):
    def test_actor_weights_round_trip(self):
        import torch
        from actor import RemoteAgent
        a=RemoteAgent('cpu');b=RemoteAgent('cpu')
        with torch.no_grad():
            for p in a.policy.actor.parameters():
                if p.requires_grad:p.add_(0.01*torch.randn_like(p))
        params=common.actor_params(a.policy)
        self.assertFalse(any(k.startswith(common.FROZEN_PREFIX) for k in params))
        params['info.updates']=torch.tensor(7);b.load(params)
        g=np.random.default_rng(1);ob={'image':g.integers(0,255,(128,128,3),dtype=np.uint8),'side':g.integers(0,255,(128,128,3),dtype=np.uint8),'state':np.zeros(9,np.float32)}
        np.testing.assert_allclose(a.act(ob,deterministic=True),b.act(ob,deterministic=True),atol=1e-6)
        self.assertEqual(b.info['updates'],7.0)

    def test_transitions_and_flags(self):
        ep=fake_episode(human=[0,1,1,0,0,0])
        tr=common.episode_transitions(ep)
        self.assertEqual(len(tr),6);self.assertEqual(tr[0]['state'][IMG].dtype.__str__(),'torch.uint8')
        self.assertEqual(tuple(tr[0]['state'][VIEW_KEYS['side']].shape),(1,3,128,128))
        self.assertEqual(tuple(tr[0]['state'][IMG].shape),(1,3,128,128))
        self.assertEqual([float(t['complementary_info'][IS_INTERVENTION][0]) for t in tr],[0,1,1,0,0,0])
        self.assertTrue(tr[-1]['done']);self.assertFalse(tr[-1]['truncated'])
        tr=common.episode_transitions(fake_episode(success=False))
        self.assertFalse(tr[-1]['done']);self.assertTrue(tr[-1]['truncated'])
        self.assertTrue(all(float(t['complementary_info'][IS_DEMO][0])==1 for t in common.episode_transitions(ep,demo=True)))

    def test_buffer_sample(self):
        buf=ImageBuffer(10)
        for t in common.episode_transitions(fake_episode(T_=12)):buf.add(t)
        self.assertEqual(len(buf),10)
        b=buf.sample(4,'cpu')
        self.assertEqual(tuple(b['state'][IMG].shape),(4,3,128,128));self.assertLessEqual(float(b['state'][IMG].max()),1.0)
        self.assertEqual(tuple(b['next_state'][VIEW_KEYS['side']].shape),(4,3,128,128))
        self.assertEqual(tuple(b['next_state'][STATE].shape),(4,9))

    def test_learner_routing(self):
        import learner
        args=type('A',(),{'out':tempfile.mkdtemp(),'device':'cpu','batch':8,'utd':1,'lr':3e-4,'online_cap':100,'offline_cap':100,'resume':None,'views':'wrist,side'})()
        L=learner.Learner(args)
        for t in common.episode_transitions(fake_episode(),demo=True):L.route(t)
        self.assertEqual((len(L.online),len(L.offline)),(0,6))
        for t in common.episode_transitions(fake_episode(human=[0,1,1,0,0,0])):L.route(t)
        self.assertEqual((len(L.online),len(L.offline)),(6,8));self.assertEqual(L.counts,{'policy':4,'intervention':2,'demo':6})
        info=L.agent.update(L.online,L.offline)
        self.assertIn('loss_critic',info)


class TestFullEnv(unittest.TestCase):
    def test_start_distribution(self):
        env,_=make_env(start_r_mm=10.0,hover_min_mm=4.0,hover_max_mm=12.0)
        rng=np.random.default_rng(0);s=[env.sample_start(rng) for _ in range(300)]
        r=np.array([np.hypot(*x['offset_mm']) for x in s]);dz=np.array([x['hover_dz_mm'] for x in s])
        self.assertLessEqual(r.max(),10.0);self.assertGreater(r.max(),8.0);self.assertTrue(all(x['hover'] for x in s))
        self.assertTrue(((dz>=4.0)&(dz<=12.0)).all())

    def test_scripted_insertion_succeeds_by_usb(self):
        env,robot=make_env()
        obs=env.reset({'offset_mm':[6.0,-5.0],'hover':True,'hover_dz_mm':8.0})
        self.assertEqual(obs['state'].shape,(9,));self.assertEqual(obs['side'].shape,(128,128,3));self.assertFalse(env.usb.connected)
        r,term,info=drive_in(env,robot)
        self.assertTrue(term);self.assertEqual(r,1.0);self.assertTrue(info['usb'])
        self.assertLess(robot.p[2],T.Z_MOUTH)   # full depth, not just the mouth

    def test_pressing_beside_port_never_succeeds(self):
        env,robot=make_env(max_steps=40)
        env.reset({'offset_mm':[0.0,5.0],'hover':True,'hover_dz_mm':4.0})
        for i in range(40):
            obs,r,term,trunc,info=env.step([0,0,-1])
            self.assertFalse(term);self.assertGreaterEqual(info['p_cmd'][2],info['p_meas'][2]-5.0-1e-6)
        self.assertTrue(trunc);self.assertIn('press',info['clamped'])

    def test_box_is_around_the_port(self):
        env,robot=make_env(start_r_mm=10.0,box_margin_mm=5.0,max_steps=60)
        env.reset({'offset_mm':[9.0,0.0],'hover':True,'hover_dz_mm':8.0})
        for i in range(60):obs,r,term,trunc,info=env.step([1,0,0])
        self.assertLessEqual(info['p_cmd'][0]-env.port.p_port[0],15.0+1e-6);self.assertIn('box',info['clamped'])

    def test_reset_refuses_when_usb_still_connected(self):
        env,robot=make_env()
        env.usb=type('U',(),{'connected':True,'describe':lambda s:{}})()
        with self.assertRaises(UsbStillConnected):env.reset({'offset_mm':[3.0,0.0],'hover':True,'hover_dz_mm':6.0})

    def test_scripted_insert_from_above(self):
        env,robot=make_env()
        r=env.insert_from_above(timeout=5.0)
        self.assertTrue(r['usb']);self.assertFalse(r['still_connected_after_lift']);self.assertGreater(r['z_below_mouth_mm'],0)

    def test_integral_frozen_near_hub_only(self):
        env,robot=make_env();calls=[]
        robot.freeze_integral=lambda f:calls.append(f)
        env.reset({'offset_mm':[0.0,0.0],'hover':True,'hover_dz_mm':8.0})
        self.assertNotIn(True,calls)                      # free air: integrating
        drive_in(env,robot)
        self.assertIn(True,calls)                         # frozen on the way into the port
        env.lift();self.assertEqual(calls[-1],False)       # resumed above the zone

    def test_lift_is_vertical_until_clear(self):
        env,robot=make_env()
        env.reset({'offset_mm':[0.0,0.0],'hover':True,'hover_dz_mm':8.0});drive_in(env,robot)
        pm,_=env.p_meas();R0=env.R_cmd.copy()
        env.p_cmd=env.p_cmd+np.array([3.0,-2.0,-4.0])      # a stale command: sideways lead and press below the plug
        sent=[];orig=env._send
        def rec(p,R):sent.append((np.array(p,dtype=float).copy(),np.array(R).copy()));orig(p,R)
        env._send=rec;env.lift()
        clear=env.port.z_mouth+env.cfg.clear_mm
        below=[(p,R) for p,R in sent if p[2]<clear-1e-6]
        self.assertTrue(below)
        for p,R in below:
            self.assertLess(np.hypot(*(p[:2]-pm[:2])),0.05)   # measured xy, not the stale command's
            self.assertTrue(np.allclose(R,R0,atol=1e-6))       # orientation held while the plug can still be in the port
        self.assertGreaterEqual(sent[-1][0][2],env.port.z_surface+env.cfg.top_mm-1e-6)

    def test_press_is_a_vertical_force(self):
        env,robot=make_env(press_n=10.0,press_mm=5.0)
        env.reset({'offset_mm':[0.0,5.0],'hover':True,'hover_dz_mm':2.0})   # beside the port: the hub surface blocks it
        sent=[];orig=robot.send_target
        def rec(q,stage,**k):sent.append((stage,np.asarray(q,dtype=float).copy(),k));orig(q,stage,**k)
        robot.send_target=rec
        pm,_=env.p_meas()
        for _ in range(15):env._send(np.r_[pm[:2],pm[2]-8.0],env.port.R_ref);env.clock.tick()
        press=[(q,k) for st,q,k in sent if st=='press']
        self.assertTrue(press)
        fs=[k['press_n'] for q,k in press]
        self.assertTrue(all(b-a<=1.0+1e-9 for a,b in zip(fs,fs[1:])))   # slew-limited ramp
        self.assertAlmostEqual(fs[-1],10.0)
        q,k=press[-1];pm2,_=env.p_meas()
        self.assertTrue(np.allclose(q[:2]*1000,pm[:2],atol=0.05))       # no sideways push
        self.assertAlmostEqual((pm2[2]-q[2]*1000),10.0/1e4*1000,delta=0.05)   # depth offset = F/kp (fake J = I)

    def test_press_lateral_spring_is_capped(self):
        env,robot=make_env(press_n=10.0,press_kxy=1.0,press_fxy_max=4.0)
        env.reset({'offset_mm':[0.0,5.0],'hover':True,'hover_dz_mm':2.0})
        sent=[];orig=robot.send_target
        def rec(q,stage,**k):sent.append((stage,k));orig(q,stage,**k)
        robot.send_target=rec
        for _ in range(5):
            pm,_=env.p_meas();env._send(np.r_[pm[0]+10.0,pm[1]-2.0,pm[2]-8.0],env.port.R_ref);env.clock.tick()
        k=[k for st,k in sent if st=='press'][-1]
        self.assertAlmostEqual(k['press_fxy'][0],4.0);self.assertAlmostEqual(k['press_fxy'][1],-2.0,delta=0.2)

    def test_scripted_insert_reaches_full_depth_with_force(self):
        env,robot=make_env(press_n=10.0)
        r=env.insert_from_above(timeout=5.0)
        self.assertTrue(r['usb']);self.assertGreater(r['press_n'],0)

    def test_clf_usb_reward_needs_the_usb(self):
        class AlwaysYes:
            threshold=0.5;views=('side',)
            def prob(self,obs):return 0.99
        env,robot=make_env(clf=AlwaysYes(),reward='clf_usb',usb_confirm_s=0.05)
        env.reset({'offset_mm':[0.0,0.0],'hover':True,'hover_dz_mm':6.0})
        _,r,term,trunc,info=env.step([0.0,0.0,0.0]);self.assertFalse(term)          # hold 2: first positive only counts
        _,r,term,trunc,info=env.step([0.0,0.0,0.0])
        self.assertTrue(info['clf_fire']);self.assertFalse(info['usb_confirm']);self.assertFalse(term);self.assertEqual(r,0.0)
        r,term,info=drive_in(env,robot)
        self.assertTrue(term);self.assertEqual(r,1.0);self.assertTrue(info['usb'])

    def test_late_handback_leaves_the_policy_time(self):
        env,robot=make_env(max_steps=10,handback_grace_steps=5)
        env.reset({'offset_mm':[0.0,5.0],'hover':True,'hover_dz_mm':8.0})
        for _ in range(9):env.step([0.0,0.0,0.0])
        env.steps=12;env.handback()                      # operator held it past max_steps, then handed back
        _,_,term,trunc,_=env.step([0.0,0.0,0.0]);self.assertFalse(trunc)
        for _ in range(3):_,_,term,trunc,_=env.step([0.0,0.0,0.0])
        self.assertFalse(trunc);_,_,term,trunc,_=env.step([0.0,0.0,0.0]);self.assertTrue(trunc)

    def test_touch_reports_contact_height(self):
        env,robot=make_env()
        self.assertAlmostEqual(env.touch('y',5.0),0.0,delta=0.1)


class TestClassifier(unittest.TestCase):
    def test_train_and_reward(self):
        import classifier
        d=Path(tempfile.mkdtemp());synthetic_clf_session(d)
        clf,m=classifier.train([d],str(d/'clf.pt'),views=('side',),steps=150,batch=32,device='cpu',log=lambda *x:None)
        self.assertGreaterEqual(m['accuracy'],0.95,m);self.assertIn('vs_usb',m)
        clf=classifier.RewardClassifier.load(d/'clf.pt')
        env,robot=make_env(clf=clf)
        env.reset({'offset_mm':[4.0,3.0],'hover':True,'hover_dz_mm':6.0})
        self.assertLess(env.reset_clf_p,0.5)
        r,term,info=drive_in(env,robot)
        self.assertTrue(term);self.assertTrue(info['usb']);self.assertGreaterEqual(info['clf_p'],0.5)


class TestExactUsb(unittest.TestCase):
    def test_only_the_exact_port_counts(self):
        import time
        from usb_exact import ExactUsb
        root=Path(tempfile.mkdtemp())
        def plug(p):(root/p).mkdir(parents=True,exist_ok=True);(root/p/'idVendor').write_text('2717')
        plug('1-2.4.2')   # the inner hub itself
        u=ExactUsb('1-2.4.2.1',period_s=0.01,root=str(root)).start()
        self.assertFalse(u.connected)
        plug('1-2.4.2.3');time.sleep(0.05);self.assertFalse(u.connected)   # neighbouring port of the same hub
        plug('1-2.4.2.1');time.sleep(0.05);self.assertTrue(u.connected)
        import shutil;shutil.rmtree(root/'1-2.4.2.1');time.sleep(0.05);self.assertFalse(u.connected)
        u.stop();self.assertEqual([e['connected'] for e in u.events],[True,False])


def free_port():
    s=socket.socket();s.bind(('127.0.0.1',0));p=s.getsockname()[1];s.close();return p


@unittest.skipIf(__import__('platform').machine()=='aarch64' and not os.environ.get('HILSERL_E2E'),
                 'end-to-end runs a CPU learner: not on the Jetson, where it would compete with the 33 ms control tick (HILSERL_E2E=1 forces it)')
class TestEndToEnd(unittest.TestCase):
    """learner.py and actor.py --fake as separate processes over gRPC: demos are sent, the learner trains, weights come
    back between episodes, interventions are routed, stats reach the learner log."""
    def test_train_round_trip(self):
        py=sys.executable;port=free_port();out=Path(tempfile.mkdtemp())
        # one successful scripted "demo" episode in the actor's on-disk format
        demo=out/'demo';demo.mkdir()
        env,robot=make_env();obs=env.reset({'offset_mm':[4.0,0.0],'hover':True,'hover_dz_mm':6.0})
        frames=[obs['image']];states=[obs['state']];acts=[];rews=[];terms=[]
        for i in range(60):
            a=np.clip(np.r_[(T.P_INS[:2]-env.p_cmd[:2])/1.5,-1.0 if np.hypot(*(robot.p[:2]-T.P_INS[:2]))<0.5 else 0.0],-1,1)
            obs,r,term,trunc,info=env.step(a);acts.append(a);rews.append(r);terms.append(float(term));frames.append(obs['image']);states.append(obs['state'])
            if term:break
        self.assertTrue(terms[-1])
        import classifier
        clfdir=out/'clf';clfdir.mkdir();synthetic_clf_session(clfdir)
        classifier.train([clfdir],str(out/'clf.pt'),views=('side',),steps=150,batch=32,device='cpu',log=lambda *x:None)
        env2,robot2=make_env();obs=env2.reset({'offset_mm':[4.0,0.0],'hover':True,'hover_dz_mm':6.0});side=[obs['side']]
        frames=[obs['image']];states=[obs['state']];acts=[];rews=[];terms=[]
        for i in range(60):
            a=np.clip(np.r_[(T.P_INS[:2]-env2.p_cmd[:2])/1.5,-1.0 if np.hypot(*(robot2.p[:2]-T.P_INS[:2]))<0.5 else 0.0],-1,1)
            obs,r,term,trunc,info=env2.step(a);acts.append(a);rews.append(r);terms.append(float(term));frames.append(obs['image']);side.append(obs['side']);states.append(obs['state'])
            if term:break
        self.assertTrue(terms[-1])
        np.savez_compressed(demo/'ep-0001.npz',frames_side=np.stack(side),frames=np.stack(frames),states=np.stack(states).astype(np.float32),actions=np.stack(acts).astype(np.float32),
                            rewards=np.array(rews,np.float32),terminals=np.array(terms,np.float32),human=np.ones(len(acts),bool))
        lr=subprocess.Popen([py,str(ROOT/'learner.py'),'--out',str(out/'learner'),'--port',str(port),'--device','cpu','--batch','16',
                             '--min-online','20','--warmup','20','--push-s','1','--exit-after-s','150'],stdout=subprocess.PIPE,stderr=subprocess.STDOUT,text=True,cwd=ROOT)
        try:
            line=''
            while 'LEARNER_READY' not in line:
                line=lr.stdout.readline()
                if not line:self.fail('learner exited before READY')
            act=subprocess.run([py,str(ROOT/'actor.py'),'train','--fake','--learner-port',str(port),'--episodes','4','--max-steps','25',
                                '--touch-every','2','--demos',str(demo),'--clf',str(out/'clf.pt')],input='READY\n',capture_output=True,text=True,timeout=300,cwd=ROOT)
            self.assertEqual(act.returncode,0,act.stdout[-3000:]+act.stderr[-3000:])
            self.assertIn('已发送 示范 1 条',act.stdout)
            eps=[l for l in act.stdout.splitlines() if l.startswith('EP ')]
            self.assertIn('分类器',act.stdout)
            self.assertEqual(len(eps),4,act.stdout[-2000:])
            sess=[l.split()[-1] for l in act.stdout.splitlines() if l.startswith('SESSION_SAVED')][0]
            self.assertTrue((Path(sess)/'actor-last.pt').exists())
        finally:
            lr.terminate()   # learner saves on SIGTERM; a CPU update in flight can delay that, so kill after a grace period
            try:rest=lr.communicate(timeout=30)[0]
            except subprocess.TimeoutExpired:lr.kill();rest=lr.communicate()[0]
        logs=[json.loads(l) for l in (out/'learner'/'learner.jsonl').read_text().splitlines()]
        self.assertGreaterEqual(len(logs),4);self.assertEqual(logs[-1]['demo'],len(acts))
        self.assertGreater(logs[-1]['policy'],0)
        self.assertGreater(max(l['updates'] for l in logs),0,rest[-2000:])
        self.assertTrue((out/'learner'/'ckpt-last.pt').exists())
        # the learner checkpoint loads into the eval path
        ev=subprocess.run([py,str(ROOT/'actor.py'),'eval','--fake','--n','2','--max-steps','15','--touch-every','0','--ckpt',str(out/'learner'/'ckpt-last.pt'),'--clf',str(out/'clf.pt')],
                          input='READY\n',capture_output=True,text=True,timeout=300,cwd=ROOT)
        self.assertEqual(ev.returncode,0,ev.stdout[-2000:]+ev.stderr[-2000:]);self.assertIn('USB 判定插入成功',ev.stdout)


if __name__=='__main__':unittest.main()
