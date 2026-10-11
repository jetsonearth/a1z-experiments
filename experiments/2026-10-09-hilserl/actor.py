#!/usr/bin/env python3
"""HIL-SERL actor on the A1Z (Jetson): full USB insertion from a randomised start above the port. See README.md.

The learner runs on a GPU machine (learner.py) and is reached over gRPC, normally through SSH tunnels via the Mac
(tunnel.sh), so the actor talks to 127.0.0.1:50051. Operator flow, safety, thermal pause, parking and the
kinesthetic takeover come from ../2026-10-02/rl_run.Runner; this file adds the fixed side camera, the reward
classifier, and the remote learner.

Modes (hilserl.sh <mode> ...):
  snap      wrist + side frames for choosing --crop / --crop-side and checking occlusion (motors stay in float)
  check     tracking test, scripted insertion from above (does the force budget push the plug home?), grasp touch on
            both sides, zero-action resets. Must pass before demo/train.
  classify  reward-classifier data: the arm floats, you move the plug around and into the port; once it is fully
            inserted and charging, press s once: the next --label-s seconds (2) are labelled success, then it reverts
            by itself (paper: click when the task is completed; ~200 positive, ~1000 negative frames, ~5 min).
            --label-toggle restores the old on/off key.
            USB is recorded alongside as the referee, never as a label. Then: classifier.py train runs/<session>
  demo      scripted reset to a random start, then the arm floats and you drag the plug in
  train     online HIL-SERL with the remote learner: Space = take over / hand back
  eval      fixed starts (seed 20261009), deterministic policy, no takeover; USB counts the true successes

Keys: g = start / continue   Space = take over / hand back   s = success label (classify)   x = abort episode
      p = pause after this episode   q q = quit (arm on its support first)
"""
import argparse
import json
import sys
import threading
import time
from pathlib import Path
from queue import Empty,Queue

import numpy as np

import common
from common import STATE,VIEW_KEYS
import rl_run
from rl_run import SMOOTH,Abort,Clock,Disengaged,Keys,Port
from env import FullCfg,FullInsertEnv,UsbStillConnected

ROOT=Path(__file__).resolve().parent
EVAL_SEED=20261009


def newest_calibration():
    """Newest calibration by the time in its name (file mtimes are unreliable: the Jetson clock resets after a power
    cut); a quick_calib.py result replaces the key-3-only file it was derived from."""
    fs=[f for f in (common.PREV/'calibration').glob('port-*.json')]
    quick={f.name.replace('-quick.json','.json') for f in fs if f.name.endswith('-quick.json')}
    fs=[f for f in fs if f.name not in quick]
    return max(fs,key=lambda f:f.name.replace('-quick.json','.json'))
OBS_KEY={'wrist':'image','side':'side'}   # env observation key per view


# ---------- link to the learner (LeRobot transport; lerobot.rl.actor is not imported: it pulls in SO-100/gamepad code) ----------
class Link:
    def __init__(self,host,port):
        import grpc
        from lerobot.transport import services_pb2_grpc
        from lerobot.transport.utils import grpc_channel_options
        # the Jetson login shell exports http(s)_proxy; gRPC would send the learner tunnel through it (UNAVAILABLE)
        self.channel=grpc.insecure_channel(f'{host}:{port}',list(grpc_channel_options())+[('grpc.enable_http_proxy',0)])
        self.stub=services_pb2_grpc.LearnerServiceStub(self.channel)
        self.stop=threading.Event();self.params=Queue();self.transitions=Queue();self.interactions=Queue();self.errors=[]
        self.addr=f'{host}:{port}'

    def connect(self,attempts=30):
        import grpc
        from lerobot.transport import services_pb2
        for i in range(attempts):
            try:self.stub.Ready(services_pb2.Empty());break
            except grpc.RpcError as e:
                if i==0:print(f'等 learner（{self.addr}）…  隧道开了吗？ {e.code()}',flush=True)
                time.sleep(2)
        else:raise RuntimeError(f'连不上 learner {self.addr}')
        for fn in (self._recv_params,self._send_transitions,self._send_interactions):threading.Thread(target=fn,daemon=True).start()

    def _guard(self,fn,name):
        import grpc
        try:fn()
        except grpc.RpcError as e:
            if not self.stop.is_set():self.errors.append(f'{name}: {e.code()} {e.details()}')

    def _recv_params(self):
        from lerobot.transport import services_pb2
        from lerobot.transport.utils import receive_bytes_in_chunks
        self._guard(lambda:receive_bytes_in_chunks(self.stub.StreamParameters(services_pb2.Empty()),self.params,self.stop,'[ACTOR] params'),'params')

    def _stream(self,q,msg_cls):
        from lerobot.transport.utils import send_bytes_in_chunks
        while not self.stop.is_set():
            try:m=q.get(timeout=0.5)
            except Empty:continue
            yield from send_bytes_in_chunks(m,msg_cls)

    def _send_transitions(self):
        from lerobot.transport import services_pb2
        self._guard(lambda:self.stub.SendTransitions(self._stream(self.transitions,services_pb2.Transition)),'transitions')

    def _send_interactions(self):
        from lerobot.transport import services_pb2
        self._guard(lambda:self.stub.SendInteractions(self._stream(self.interactions,services_pb2.InteractionMessage)),'interactions')

    def push_episode(self,ep,views,demo=False):
        self.transitions.put(common.encode_episode(ep,views,demo=demo))   # learner.drain decodes and splits into transitions

    def push_stats(self,d):
        from lerobot.transport.utils import python_object_to_bytes
        self.interactions.put(python_object_to_bytes(d))

    def latest_params(self):
        from lerobot.transport.utils import bytes_to_state_dict
        item=None
        while True:
            try:item=self.params.get_nowait()
            except Empty:break
        return None if item is None else bytes_to_state_dict(item)

    def close(self):
        self.stop.set()
        try:self.channel.close()
        except Exception:pass


class RemoteAgent:
    """Acting copy of the policy. Weights change only between episodes, as in LeRobot's actor."""
    def __init__(self,device,views=common.DEFAULT_VIEWS):
        import torch
        self.torch=torch;self.device=device;self.views=tuple(views);self.policy=common.make_policy(device,self.views);self.policy.eval()
        self.version=0;self.info={};self.params=None

    def act(self,obs,deterministic=False):
        torch=self.torch
        with torch.no_grad():
            b={VIEW_KEYS[v]:torch.from_numpy(obs[OBS_KEY[v]]).to(self.device).permute(2,0,1).unsqueeze(0).float().div_(255.0) for v in self.views}
            b[STATE]=torch.from_numpy(obs['state']).to(self.device).unsqueeze(0)
            with torch.autocast('cuda',dtype=torch.float16,enabled=self.device.startswith('cuda')):
                f=self.policy.actor.encoder.get_cached_image_features(b)
            a,_,mean=self.policy.actor(b,{k:v.float() for k,v in f.items()})
            out=torch.tanh(mean) if deterministic else a
        return out[0].float().cpu().numpy()

    def load(self,params):
        sd={k:v.to(self.device) for k,v in params.items()}
        self.info={k[5:]:float(v) for k,v in params.items() if k.startswith('info.')}
        common.load_actor_params(self.policy,sd);self.params=params;self.version+=1

    def load_file(self,path):
        d=self.torch.load(path,map_location=self.device)
        if 'policy' in d:   # learner checkpoint (whole SAC policy)
            self.policy.load_state_dict(d['policy']);self.info={'updates':d.get('updates')};self.version+=1
        else:self.load(d)

    def save(self,path):
        if self.params is not None:self.torch.save(self.params,path)

    def warmup(self):
        z=np.zeros((common.IMG_SIZE,common.IMG_SIZE,3),np.uint8)
        ob={'image':z,'side':z,'state':np.zeros(common.STATE_DIM,np.float32)}
        for _ in range(3):self.act(ob);self.act(ob,deterministic=True)


class HilRunner(rl_run.Runner):
    def __init__(self,a):
        self.a=a;self.keys=None;self.toggle=False;self.pause=False;self.quit_at=0.0;self.quit=False;self.label_toggle=False
        self.session=ROOT/'runs'/(time.strftime('%Y%m%d-%H%M%S')+'-'+a.mode);self.session.mkdir(parents=True)
        self.log=open(self.session/'episodes.jsonl','a',buffering=1)
        self.n=0;self.results=[];self.link=None;self.side=None;self.views=common.parse_views(a.views)

    def hook(self):
        for k in self.keys.drain():
            if k==' ':self.toggle=True
            elif k.lower()=='s':self.label_toggle=True
            elif k.lower()=='x':raise Abort('operator x')
            elif k.lower()=='p':self.pause=True;print('\n⏸ 这一轮结束后暂停',flush=True)
            elif k.lower()=='q':
                if time.monotonic()-self.quit_at<2:self.quit=True;raise Abort('quit')
                self.quit_at=time.monotonic();print('\n再按一次 q 退出（先确认手臂在支撑上）',flush=True)

    def setup(self):
        a=self.a
        if a.fake:
            import test_rl_env as T
            import test_hilserl as H
            from side_camera import FakeSideCamera
            self.kin=T.FakeKin();calib=Path(T.calib_file())
        else:
            from a1z_robot import A1Z,A1ZConfig
            from insertion import Kin
            from side_camera import SideCamera
            from usb_watch import UsbWatcher
            self.kin=Kin();calib=Path(a.calib) if a.calib else newest_calibration()
        self.port=Port(calib,self.kin) if a.mode not in ('snap','classify') else None
        if self.port:print('标定：',json.dumps(self.port.describe(),ensure_ascii=False),flush=True)
        clf=None
        if a.clf:
            from classifier import RewardClassifier
            clf=RewardClassifier.load(a.clf,'cpu' if a.fake else 'cuda')
            if a.clf_threshold is not None:clf.threshold=a.clf_threshold
            print(f"分类器 {a.clf}：视角 {clf.views}  阈值 {clf.threshold}  验证集 {clf.meta.get('accuracy')}  误报率 {clf.meta.get('false_positive_rate')}",flush=True)
        if input(f'【HIL-SERL {a.mode}】现场就绪（插头已夹好、hub 固定、急停在手边、USB 未连接），输入 READY：').strip()!='READY':sys.exit(0)
        if a.fake:
            self.robot=T.FakeRobot((470,100,195));self.usb=H.FakeUsbLinked(self.robot);self.side=FakeSideCamera(self.usb)
            self.robot.temps=lambda:(50.0,40.0);self.robot.disconnect=lambda:None;self.robot.ready_info={'fake':True}
        else:
            if a.usb_path:
                from usb_exact import ExactUsb
                self.usb=ExactUsb(a.usb_path).start()
            else:self.usb=UsbWatcher().start()
            if self.usb.connected and a.mode not in ('snap','classify'):sys.exit('USB 目标设备现在就是连接状态：先把插头拔出来再启动')
            import subprocess,time as _t
            if subprocess.run(['tmux','kill-session','-t','sidecam'],capture_output=True).returncode==0:_t.sleep(1.5)  # console preview held the camera
            self.side=SideCamera(model=a.side_model).start()
            if a.grip_full:import os;os.environ['A1Z_GRIPPER_NM']=str(a.grip_full)
            self.robot=A1Z(A1ZConfig(stiffness=a.stiffness,out_dir=str(self.session),smooth=SMOOTH,grip_hold_nm=a.grip_hold,
                                     integral_gain=a.ki,integral_max_deg=a.ki_max,kp=a.kp,kd=a.kd))
            self.robot.connect()
        self.keys=Keys();self.clock=Clock(self.robot,hz=1000.0 if a.fake else 30.0,hook=self.hook)
        reward=a.reward if a.mode in ('train','eval','demo') else 'usb'
        if reward in ('classifier','clf_usb') and clf is None and a.mode in ('train','eval','demo'):sys.exit('--reward classifier 需要 --clf classifier.pt（先 classify 采数、classifier.py train）')
        cfg=FullCfg(side_axis=a.side_axis,crop=tuple(a.crop) if a.crop else None,crop_side=tuple(a.crop_side) if a.crop_side else None,
                    max_steps=a.max_steps,press_mm=a.press_mm,press_n=a.press_n,ticks_per_step=max(1,round(a.step_s*30)),step_mm=a.step_mm,hz=1.0/a.step_s,
                    start_r_mm=a.start_r,hover_min_mm=a.hover_min,hover_max_mm=a.hover_max,top_mm=a.top_mm,floor_mm=a.floor_mm,reward=reward,clf_hold=a.clf_hold,usb_confirm_s=a.usb_confirm,handback_grace_steps=a.handback_grace,
                    travel_mm_s=a.travel_speed,descend_mm_s=a.descend_speed,lift_mm_s=a.lift_speed)
        self.env=FullInsertEnv(self.robot,self.kin,self.port,self.usb,self.clock,cfg,side=self.side,clf=clf) if self.port else None
        self.cfg=cfg
        meta={'mode':a.mode,'args':vars(a),'port':self.port.describe() if self.port else None,'env_cfg':cfg.__dict__,'views':self.views,
              'usb':self.usb.describe(),'side_camera':self.side.describe(),'wrist_serial':getattr(self.robot,'wrist_serial',None),
              'classifier':{'path':a.clf,'views':clf.views,'threshold':clf.threshold,'meta':clf.meta} if clf else None,
              'robot_ready':getattr(self.robot,'ready_info',None),'policy':common.POLICY_KW,'t':time.time()}
        (self.session/'session.json').write_text(json.dumps(meta,indent=1,default=str,ensure_ascii=False))
        print(f'会话目录 {self.session}',flush=True)

    def guarded(self,fn,*args,**kw):
        try:return super().guarded(fn,*args,**kw)
        except UsbStillConnected as e:
            self.robot.disengage('usb still connected')
            print(f'\n⚠ {e}。手臂浮动：把插头拔出来、检查夹持（滑了就重新标定），再按 g。',flush=True)
            self.log.write(json.dumps({'n':self.n+1,'aborted':'usb_still_connected','t':time.time()})+'\n')
            return 'quit' if not self.wait_go('按 g 继续，q q 退出') else None

    def touch_gate(self):
        """Hover starts never rest on the hub, so check the grasp height by touching beside the port."""
        print('🔍 夹持检查（脚本动作，不是回合，不用接管）：插头贴一下口旁边的表面，量高度看有没有在指尖里滑',flush=True)
        dz=self.guarded(self.env.touch,self.a.side_axis)
        if dz=='quit':return False
        if dz is None:return True
        print(f'夹持检查：贴表面高度差 {dz:+.2f} mm',flush=True)
        return self.slip_gate({'contact_dz_mm':dz})

    def disk_gate(self,min_gb=1.5):
        """2026-10-11 15:42: the Jetson disk filled up (0 bytes free) during HG-DAgger round 2; saving the episode failed, the
        run crashed and the arm fell. Before every episode: below min_gb free, lift, float, and wait for space."""
        import shutil
        free=lambda:shutil.disk_usage(str(self.session)).free/1e9
        if self.a.fake or free()>=min_gb:return True
        print(f'\n💾 Jetson 硬盘只剩 {free():.2f} GB（< {min_gb} GB），再写回合数据会崩溃。抬起后浮动：把手臂扶回支撑上，清理空间后按 g',flush=True)
        try:self.env.lift()
        except (Abort,Disengaged):pass
        self.robot.disengage('disk full pause')
        while free()<min_gb:
            self.robot.heartbeat();time.sleep(0.5);print(f'\r剩余 {free():.2f} GB',end='',flush=True)
        print()
        return self.wait_go('空间已够。按 g 继续（会重新接合并复位）')

    def thermal_gate(self):
        """Gripper limit from rl_run, plus the arm joints (2026-10-11: J2 reached 82.5-85 C at the end of a 100-episode eval;
        the vendor e-stop is at 90 C rotor). Above --joint-pause the arm floats; put it on its support to cool."""
        if not self.disk_gate():return False
        if not super().thermal_gate():return False
        _,j=self.temps()
        if j is None or j<self.a.joint_pause:return True
        print(f'\n🌡 关节最高 {j}°C ≥ {self.a.joint_pause}°C（厂家 90°C 急停）：抬起后浮动。把手臂扶回支撑上降温，降到 {self.a.joint_resume}°C 以下再按 g',flush=True)
        try:self.env.lift()
        except (Abort,Disengaged):pass
        self.robot.disengage('joint thermal pause')
        while True:
            self.robot.heartbeat();time.sleep(0.5);_,j=self.temps()
            print(f'\r关节最高 {j}°C',end='',flush=True)
            if j is not None and j<self.a.joint_resume:break
        print()
        return self.wait_go('已降温。按 g 继续（会重新接合并复位）')

    # ---------- one episode (rl_run.Runner.episode plus the side view, USB and classifier per step) ----------
    def episode(self,start,actor=None,human_start=False,verify=False,deterministic=False,tag='',allow_toggle=True):
        env=self.env;self.toggle=False
        agent=getattr(self,'agent',None);reset_agent=getattr(agent,'reset',lambda:None)
        obs=env.reset(start);reset_agent()
        if obs is None:
            print('起点直接掉进了口，换一个起点',flush=True);env.lift();return 'fell'
        frames=[obs['image']];side=[obs.get('side')];states=[obs['state']];acts=[];rews=[];terms=[];human=[];infos=[]
        human_mode=False;interventions=0;t0=time.monotonic();t_usb=t_clf=None
        vid=None
        if self.a.video and not self.a.fake:   # full-resolution wrist + side clips of the episode (reports), from the reset on
            vid=self.session/f'ep-{self.n+1:04d}'
            if hasattr(self.robot,'start_video'):self.robot.start_video(str(vid)+'-wrist.mp4')
            if hasattr(self.side,'start_record'):self.side.start_record(str(vid)+'-side.mp4')
        if human_start:self.robot.disengage('takeover');human_mode=True;print('✋ 拖着插头插进去（判定成功会自动结束）',flush=True)
        while True:
            if self.toggle and not allow_toggle:self.toggle=False
            if self.toggle:
                self.toggle=False
                if human_mode:env.handback();reset_agent();human_mode=False;print('▶ 交还策略',flush=True)
                else:self.robot.disengage('takeover');human_mode=True;interventions+=1;print('✋ 你接管了，空格交还',flush=True)
            if human_mode:obs,r,term,trunc,info,a=env.human_step()
            else:
                a=actor(env,obs) if actor is not None else self.agent.act(obs,deterministic=deterministic)
                obs,r,term,trunc,info=env.step(a)
            if info['usb'] and t_usb is None:t_usb=round(time.monotonic()-t0,2)
            if info['clf_p'] is not None and info['clf_p']>=env.clf.threshold and t_clf is None:t_clf=round(time.monotonic()-t0,2)
            acts.append(np.asarray(a,np.float32));rews.append(r);terms.append(float(term));human.append(bool(info['human']));infos.append(info)
            frames.append(obs['image']);side.append(obs.get('side'));states.append(obs['state'])
            if term or trunc:break
        dur=round(time.monotonic()-t0,2);success=bool(terms[-1])
        if human_mode:
            if success:print('✓ 判定成功，松手，1 秒后接合',flush=True)
            t=time.monotonic()
            while time.monotonic()-t<1.0:self.clock.tick(need_engaged=False)
            env.engage()
        # referee: the USB enumerates ~0.7 s after full insertion; a success ended by the classifier alone would be
        # counted as a USB failure if read at once (2026-10-10). Wait up to usb_confirm_s before pulling out.
        usb_at_end=env.wait_usb(env.cfg.usb_confirm_s) if success else bool(self.usb.connected)
        env.lift()
        if vid is not None:   # stop after the pull-out so the clip shows the whole insertion and release
            if hasattr(self.robot,'stop_video'):self.robot.stop_video()
            if hasattr(self.side,'stop_record'):self.side.stop_record()
        ep={'frames':np.stack(frames),'states':np.stack(states).astype(np.float32),'actions':np.stack(acts),
            'rewards':np.array(rews,np.float32),'terminals':np.array(terms,np.float32),'human':np.array(human),
            'usb':np.array([i['usb'] for i in infos]),'clf_p':np.array([np.nan if i['clf_p'] is None else i['clf_p'] for i in infos],np.float32)}
        if side[0] is not None:ep['frames_side']=np.stack(side)
        self.n+=1
        meta={'n':self.n,'tag':tag,'start':start,'success':success,'reward':env.cfg.reward,'usb_success':bool(usb_at_end or any(i['usb'] for i in infos)),
              'clf_success':t_clf is not None,'t_usb_s':t_usb,'t_clf_s':t_clf,'reset_clf_p':env.reset_clf_p,
              'steps':len(acts),'human_steps':int(np.sum(human)),'interventions':interventions,'duration_s':dur,
              'clamped':{k:int(sum(k in i['clamped'] for i in infos)) for k in ('box','z','press','lead','label_clip')},
              'p_start':env.p_start.round(2).tolist(),'p_end':infos[-1]['p_meas'],'temps':self.temps(),'t':time.time()}
        if meta['success']!=meta['usb_success']:
            print(f"⚠ reward 判定 {'成功' if meta['success'] else '失败'}，USB {'连上了' if meta['usb_success'] else '没连上'}",flush=True)
        self.hold_while(lambda:np.savez_compressed(self.session/f'ep-{self.n:04d}.npz',**ep))
        self.log.write(json.dumps(meta)+'\n')
        return ep,meta

    # ---------- modes ----------
    def _save_frame(self,name,img,crop):
        import cv2
        cv2.imwrite(str(self.session/f'{name}.jpg'),img[:,:,::-1])
        if crop:
            x,y,w,h=crop;prev=img.copy();cv2.rectangle(prev,(x,y),(x+w,y+h),(255,0,0),2)
            cv2.imwrite(str(self.session/f'{name}-crop.jpg'),prev[:,:,::-1])
            cv2.imwrite(str(self.session/f'{name}-crop-128.jpg'),cv2.resize(img[y:y+h,x:x+w],(128,128),interpolation=cv2.INTER_AREA)[:,:,::-1])

    def snap(self):
        """Wrist + side frames with the plug held where episodes happen (above the port, at the mouth, fully in)."""
        k=0
        while self.wait_go('夹着插头，拖到 插口上方约 10 mm / 刚进口 / 插到底 等位置，扶稳后按 g 拍两路；q 连按两次结束'):
            time.sleep(0.3);k+=1
            with self.robot._frame_lock:wrist=self.robot._frame.copy()
            self._save_frame(f'snap-{k}-wrist',wrist,self.a.crop);self._save_frame(f'snap-{k}-side',self.side.latest(),self.a.crop_side)
            q,_=self.robot.measured();p=self.kin.fk(q)[:3,3]*1000
            print(f'已保存 snap-{k}-wrist.jpg / snap-{k}-side.jpg   手指 {p.round(1).tolist()} mm   USB {self.usb.connected}',flush=True)
            if self.a.fake:break

    def classify(self):
        """Reward-classifier data at 10 Hz while the arm floats. s toggles the "fully inserted" label (the operator's
        judgement, as in the paper); USB is stored alongside and never used as the label."""
        a=self.a;dt=0.1;label=False;label_until=0.0;buf={'frames':[],'frames_side':[],'labels':[],'usb':[],'t':[]};part=0;last_print=0.0
        def flush():
            nonlocal part
            if not buf['labels']:return
            part+=1;arr={k:np.array(v) for k,v in buf.items()}
            self.hold_while(lambda:np.savez_compressed(self.session/f'clf-{part:03d}.npz',**arr))
            for v in buf.values():v.clear()
        how=('插到底、充上电时按 s（标为成功），离开前再按 s' if a.label_toggle else f'插到底、充上电后按一次 s：接下来 {a.label_s:.0f} 秒标为成功（别动），之后自动恢复')
        if not self.wait_go(f'分类器采数：手臂浮动，你拿着插头在口附近移动、插进去、拔出来。{how}。'
                            f'多录一些"压在口旁边""插一半""插进隔壁的口"。最长 {a.seconds:.0f} 秒，q 连按两次结束。先点一下终端窗口，再按 g 开始'):return
        self.robot.disengage('classify');t0=time.monotonic();n_pos=n_neg=0
        try:
            while time.monotonic()-t0<a.seconds:
                t=time.monotonic()
                while time.monotonic()-t<dt:self.clock.tick(need_engaged=False)
                if self.label_toggle:
                    self.label_toggle=False
                    if a.label_toggle:label=not label;print(f"\n标签 → {'✅ 成功' if label else '⬜ 非成功'}",flush=True)
                    else:label_until=time.monotonic()+a.label_s;print(f"\n✅ 接下来 {a.label_s:.0f} 秒标为成功，别动"+('' if self.usb.connected else '（⚠ USB 现在没接通）'),flush=True)
                if not a.label_toggle:
                    was=label;label=time.monotonic()<label_until
                    if was and not label:print('\n⬜ 成功标记结束，可以拔了',flush=True)
                wrist,side=self.env_images()
                buf['frames'].append(wrist);buf['frames_side'].append(side);buf['labels'].append(label);buf['usb'].append(bool(self.usb.connected))
                buf['t'].append(time.monotonic()-t0);n_pos+=int(label);n_neg+=int(not label)
                if len(buf['labels'])>=600:flush()
                if time.monotonic()-last_print>1.0:
                    last_print=time.monotonic()
                    print(f"\r{time.monotonic()-t0:5.0f}s  标签 {'✅' if label else '⬜'}  USB {'🔌' if self.usb.connected else '  '}  成功 {n_pos}  非成功 {n_neg}   ",end='',flush=True)
        except Abort:pass
        finally:
            flush();print(f'\n已保存 {part} 个文件：成功 {n_pos} 帧，非成功 {n_neg} 帧（论文约 200 / 1000）。下一步：classifier.py train {self.session} --out classifier.pt',flush=True)

    def env_images(self):
        if self.env is not None:return self.env.images()
        crop=FullInsertEnv._crop
        with self.robot._frame_lock:img=self.robot._frame
        return crop(self,img,self.a.crop),crop(self,self.side.latest(),self.a.crop_side)

    def check(self):
        env=self.env;c=env.cfg;report={'track':[],'insert':[],'touch':[],'resets':[]}
        if not self.wait_go('检查：跟踪测试 → 从插口正上方脚本插入（看下压力够不够插到底）→ 两侧贴表面 → 零动作复位。按 g 开始'):return
        def track():
            env.engage();env.lift()
            env.goto([self.port.p_port[0],self.port.p_port[1],self.port.z_surface+c.top_mm],c.travel_mm_s,R_goal=self.port.R_ref)
            out=[]
            for ax in range(3):
                for sgn in (1,-1):
                    p=env.p_cmd.copy();p[ax]+=sgn*c.step_mm
                    for _ in range(30):env._send(p,self.port.R_ref);self.clock.tick()
                    pm,_=env.p_meas();out.append({'axis':'xyz'[ax],'dir':sgn,'err_mm':round(float(np.linalg.norm(p-pm)),2)})
            return out
        r=self.guarded(track)
        if r in (None,'quit'):return
        report['track']=r;errs=[x['err_mm'] for x in r]
        print(f'跟踪：每步 {c.step_mm} mm、停 1 秒后误差 中位 {np.median(errs):.2f} mm，最大 {max(errs):.2f} mm',flush=True)
        for i in range(self.a.n_insert):
            r=self.guarded(env.insert_from_above)
            if r in (None,'quit'):return
            report['insert'].append(r)
            print(f"插入 {i+1}/{self.a.n_insert}：USB {'✓' if r['usb'] else '✗'}  {r['seconds']} s  比口部低 {r['z_below_mouth_mm']} mm  下压 {r.get('press_n')} N"
                  +('  ⚠ 抬起后仍连着' if r['still_connected_after_lift'] else ''),flush=True)
            if r['still_connected_after_lift']:
                self.robot.disengage('usb still connected')
                if not self.wait_go('插头没拔出来：手动拔出、检查夹持后按 g'):return
        for sign in (1,-1):
            r=self.guarded(env.touch,c.side_axis,5.0*sign)
            if r=='quit':return
            report['touch'].append({'side':sign,'contact_dz_mm':r});print(f'贴表面 侧 {sign:+d}：{r} mm',flush=True)
        rng=np.random.default_rng(1)
        for i in range(4):
            start=env.sample_start(rng)
            def zero():
                env.reset(start);pm0=env.p_meas()[0];usb=0
                for _ in range(5):_,_,term,_,info=env.step([0,0,0]);usb+=int(info['usb'])
                env.lift();return {'start':start,'p_start':pm0.round(2).tolist(),'usb_steps':usb}
            r=self.guarded(zero)
            if r in (None,'quit'):return
            report['resets'].append(r);print(f'复位 {i+1}/4：{start}  误报 USB 步数 {r["usb_steps"]}',flush=True)
        report['tick_max_gap_ms']=round(self.clock.max_gap*1000,1)
        ok=(len(errs)==6 and float(np.median(errs))<=0.7 and max(errs)<=1.5
            and len(report['insert'])==self.a.n_insert and all(x['usb'] and not x['still_connected_after_lift'] for x in report['insert'])
            and all(t['contact_dz_mm'] is not None and abs(t['contact_dz_mm'])<=c.slip_tol_mm for t in report['touch'])
            and all(x['usb_steps']==0 for x in report['resets']))
        report['pass']=ok;(self.session/'check.json').write_text(json.dumps(report,indent=1))
        print(('✅ 检查通过' if ok else '❌ 检查未通过，看 check.json')+f'   最大节拍间隔 {report["tick_max_gap_ms"]} ms',flush=True)

    def _episodes(self,dirs,need_success=False,relabel_usb=False):
        """relabel_usb (resent training episodes, 2026-10-10): the reward becomes the USB referee's verdict. The
        classifier was fooled by plugs jammed part-way into the port (14 of 60 episodes, USB never on); keeping those
        rewards would teach the restarted critic the exploit again. Success = the episode's usb_success (USB on during
        the episode or right after the classifier ended it); otherwise no reward and the episode counts as truncated."""
        for d in dirs or []:
            meta={}
            jl=Path(d)/'episodes.jsonl'
            for line in (jl.read_text().splitlines() if jl.exists() else []):
                m=json.loads(line)
                if 'success' in m:meta[m['n']]=m
            for f in sorted(Path(d).glob('ep-*.npz')):
                ep=common.load_episode(f)
                if relabel_usb:
                    m=meta.get(int(f.stem.split('-')[1]))
                    if m is None:continue
                    T=len(ep['actions']);ep['rewards']=np.zeros(T,np.float32);ep['terminals']=np.zeros(T,np.float32)
                    if m['usb_success']:ep['rewards'][-1]=1.0;ep['terminals'][-1]=1.0
                if need_success and not (len(ep['terminals']) and ep['terminals'][-1]):continue
                if any(common.FRAME_KEYS[v] not in ep for v in self.views):print(f'跳过 {f}：缺少视角 {self.views}',flush=True);continue
                yield f,ep

    def train(self):
        a=self.a
        self.agent=RemoteAgent('cpu' if a.fake else 'cuda',self.views);self.agent.warmup()
        self.link=Link(a.learner_host,a.learner_port);self.link.connect()
        cnt={'d':0,'r':0}
        def resend():   # JPEG-encoding hundreds of episodes takes tens of seconds: keep the heartbeat alive meanwhile
            for f,ep in self._episodes(a.demos,need_success=True):self.link.push_episode(ep,self.views,demo=True);cnt['d']+=1
            for f,ep in self._episodes(a.resume,relabel_usb=a.relabel_usb):self.link.push_episode(ep,self.views);cnt['r']+=1
        print('编码并发送示范和续训数据…',flush=True);self.hold_while(resend);nd,nr=cnt['d'],cnt['r']
        print(f'已发送 示范 {nd} 条（只计成功的）、续训 {nr} 条',flush=True)
        if nd==0:print('⚠ 没有示范：论文每个任务 20 条，没有示范时 RL 从零开始基本学不会',flush=True)
        t0=time.monotonic()
        while time.monotonic()-t0<30:
            self.robot.heartbeat();p=self.link.latest_params()
            if p is not None:self.agent.load(p);break
            time.sleep(0.1)
        else:raise RuntimeError('30 秒内没有收到 learner 的权重（两边 --views 要一致）')
        import gc;gc.collect();gc.freeze()
        rng=np.random.default_rng(int(time.time()));recent=[];usb_recent=[]
        if not self.wait_go('在线训练：策略自己找口插入；明显走错或卡住再按空格接管，纠正一下就交还。按 g 开始'):return
        try:
            while True:
                if not self.thermal_gate():return
                if a.touch_every and self.n and self.n%a.touch_every==0 and not self.touch_gate():return
                r=self.guarded(self.episode,self.env.sample_start(rng),tag='train')
                if r=='quit':return
                if r is None or r=='fell':continue
                ep,meta=r
                self.hold_while(lambda:self.link.push_episode(ep,self.views))
                self.link.push_stats({'episode':meta['n'],'success':meta['success'],'usb_success':meta['usb_success'],'clf_success':meta['clf_success'],
                                      'steps':meta['steps'],'human_steps':meta['human_steps'],'interventions':meta['interventions'],
                                      'intervention_rate':meta['human_steps']/max(1,meta['steps']),'duration_s':meta['duration_s'],
                                      't_usb_s':meta['t_usb_s'],'t_clf_s':meta['t_clf_s'],'policy_version':self.agent.version})
                p=self.link.latest_params()
                if p is not None:self.hold_while(lambda:self.agent.load(p))
                if meta['interventions']==0:recent=(recent+[meta['success']])[-20:];usb_recent=(usb_recent+[meta['usb_success']])[-20:]
                rate=f"{sum(recent)}/{len(recent)}（USB {sum(usb_recent)}）" if recent else '-'
                info=self.agent.info
                print(f"EP {meta['n']}: {'✓' if meta['success'] else '✗'} USB{'✓' if meta['usb_success'] else '✗'} {meta['steps']}步 {meta['duration_s']}s "
                      f"接管{meta['interventions']}次  无接管近20轮 {rate}  learner 更新 {int(info.get('updates',0))} α {info.get('temperature','-')}  "
                      f"夹爪 {meta['temps'][0]}°C",flush=True)
                if self.link.errors:print('⚠ 与 learner 的连接出错：',self.link.errors[-3:],flush=True);return
                if a.episodes and meta['n']>=a.episodes:return
                if meta['n']%a.save_every==0:
                    self.hold_while(lambda:self.agent.save(self.session/'actor-last.pt'))
                    if meta['n']%(4*a.save_every)==0:self.hold_while(lambda:self.agent.save(self.session/f"actor-ep{meta['n']:04d}.pt"))
                if self.pause and not self.wait_go('已暂停，按 g 继续（learner 照常在跑）'):return
        finally:
            try:self.agent.save(self.session/'actor-last.pt')
            except Exception:pass
            self.link.close()
            (self.session/'timing.json').write_text(json.dumps({'tick_gaps':self.clock.slow_at}))
            print(f'已保存 {self.session}/actor-last.pt',flush=True)

    def eval(self):
        a=self.a;self.agent=self.load_agent(a.ckpt)
        import gc;gc.collect();gc.freeze()
        rng=np.random.default_rng(EVAL_SEED);starts=[self.env.sample_start(rng) for _ in range(a.n)];res=[]
        if not self.wait_go(f'评测：{a.n} 个固定起点，确定性策略，不接管。按 g 开始'):return
        for i,s in enumerate(starts):
            if a.touch_every and i and i%a.touch_every==0 and not self.touch_gate():break
            while True:
                r=self.guarded(self.episode,s,deterministic=True,tag=f'eval-{i:03d}',allow_toggle=False)
                if r=='quit':return self._eval_summary(res)
                if r is not None:break
            if r=='fell':continue
            ep,meta=r;res.append(meta)
            print(f"[{i+1}/{a.n}] reward {'✓' if meta['success'] else '✗'} USB {'✓' if meta['usb_success'] else '✗'} {meta['duration_s']}s  {meta['steps']} 步",flush=True)
            if not self.thermal_gate():break
            if self.pause:   # p was ignored in eval until 2026-10-11 (ACT run)
                self.pause=False
                if not self.wait_go(f'已暂停（{len(res)}/{a.n}），按 g 继续，q q 结束并汇总'):break
        self._eval_summary(res)

    def load_agent(self,path):
        import bc
        dev='cpu' if self.a.fake else 'cuda'
        if bc.is_act_ckpt(path):ag=bc.ActAgent(path,dev);print(f'ACT 策略 {path}  视角 {ag.views}  {ag.info}',flush=True)
        else:ag=RemoteAgent(dev,self.views);ag.load_file(path)
        ag.warmup();return ag

    def dagger(self):
        """HG-DAgger data collection (Kelly et al. 2019): the ACT novice drives, the operator takes over (space) when it goes
        wrong, corrects, and hands back or finishes. Only operator steps become labels (bc.py). No learner, no classifier:
        success is the USB referee."""
        a=self.a;self.agent=self.load_agent(a.ckpt)
        import gc;gc.collect();gc.freeze()
        rng=np.random.default_rng(int(time.time()));tot={'n':0,'ok':0,'int_eps':0,'human_steps':0,'steps':0}
        if not self.wait_go('HG-DAgger：策略自己插；走错或卡住按空格接管，纠正后空格交还（或直接插到底）。按 g 开始'):return
        while True:
            if not self.thermal_gate():return
            if a.touch_every and self.n and self.n%a.touch_every==0 and not self.touch_gate():return
            r=self.guarded(self.episode,self.env.sample_start(rng),tag='dagger')
            if r=='quit':return
            if r is None or r=='fell':continue
            ep,meta=r;tot['n']+=1;tot['ok']+=meta['usb_success'];tot['int_eps']+=meta['interventions']>0
            tot['human_steps']+=meta['human_steps'];tot['steps']+=meta['steps']
            print(f"DAgger {meta['n']}: USB{'✓' if meta['usb_success'] else '✗'} {meta['steps']}步 {meta['duration_s']}s 接管{meta['interventions']}次（{meta['human_steps']}步）  "
                  f"累计 成功 {tot['ok']}/{tot['n']}  有接管的回合 {tot['int_eps']}  人工步 {tot['human_steps']}  夹爪 {meta['temps'][0]}°C",flush=True)
            if a.episodes and meta['n']>=a.episodes:return
            if self.pause and not self.wait_go('已暂停，按 g 继续'):return

    def _eval_summary(self,res):
        """True success = USB (the referee); the reward signal's own count is reported next to it."""
        k=sum(m['usb_success'] for m in res);n=len(res);t=[m['t_usb_s'] for m in res if m['usb_success'] and m['t_usb_s'] is not None]
        streak=best=0
        for m in res:streak=streak+1 if m['usb_success'] else 0;best=max(best,streak)
        fp=sum(m['success'] and not m['usb_success'] for m in res);fn=sum((not m['success']) and m['usb_success'] for m in res)
        out={'usb_success':k,'n':n,'wilson95':common.wilson(k,n),'longest_streak':best,'cycle_s_median':float(np.median(t)) if t else None,
             'cycle_s_p90':float(np.percentile(t,90)) if t else None,'reward_success':sum(m['success'] for m in res),
             'reward_false_positive':fp,'reward_false_negative':fn,'ckpt':self.a.ckpt}
        (self.session/'eval.json').write_text(json.dumps(out,indent=1))
        print(f"USB 判定插入成功 {k}/{n}  95%CI {out['wilson95']}  最长连续成功 {best}  成功节拍 中位 {out['cycle_s_median']} s   "
              f"reward 误报 {fp} 漏报 {fn}",flush=True)


def main():
    p=argparse.ArgumentParser()
    p.add_argument('mode',choices=['snap','check','classify','demo','train','eval','dagger'])
    p.add_argument('--calib',help='calib.py output (default: newest ../2026-10-02/calibration/port-*.json)')
    p.add_argument('--crop',type=int,nargs=4,metavar=('X','Y','W','H'),help='wrist image crop before resizing to 128x128 (default: cameras.json)')
    p.add_argument('--crop-side',type=int,nargs=4,metavar=('X','Y','W','H'),help='side image crop before resizing to 128x128 (default: cameras.json, saved from the console Cameras page)')
    p.add_argument('--side-model',default=None,help='side camera model substring (D435, D455); default: the RealSense that is not the D405')
    p.add_argument('--views',default='wrist,side',help='policy image views; the learner must use the same')
    p.add_argument('--reward',choices=['clf_usb','classifier','usb'],default='clf_usb',help='clf_usb: classifier triggers, USB confirms (default since 2026-10-11); classifier: paper; USB is logged either way')
    p.add_argument('--usb-confirm',type=float,default=1.5,help='seconds to wait for the USB after a classifier positive (clf_usb reward and the referee)')
    p.add_argument('--clf',help='classifier.pt from classifier.py train')
    p.add_argument('--clf-threshold',type=float,default=None,help='override the saved threshold (0.5)')
    p.add_argument('--clf-hold',type=int,default=2,help='consecutive positives that end an episode (paper: 1)')
    p.add_argument('--seconds',type=float,default=600.0,help='classify: maximum recording time')
    p.add_argument('--label-s',type=float,default=2.0,help='classify: one s press labels this many seconds as success')
    p.add_argument('--label-toggle',action='store_true',help='classify: s toggles the label on/off instead (old behaviour)')
    p.add_argument('--usb-path',default='1-2.4.2.1',help='USB referee: only a device at exactly this sysfs path counts (target port, verified 2026-10-09); '
                   "'' = usb_watch.UsbWatcher with usb_target.txt (any device under the hub)")
    p.add_argument('--side-axis',choices=['x','y'],default='y')
    p.add_argument('--learner-host',default='127.0.0.1');p.add_argument('--learner-port',type=int,default=50051)
    p.add_argument('--stiffness',type=float,default=None,help='legacy: kp = stiffness x vendor kp with vendor kd; default (AGENTS.md, 2026-10-09): dimos gains '
                   'kp 80/80/80/50/20/20, kd 3/3/3/0.7/0.4/0.4 (robot_proc.DEFAULT_KP/KD)')
    p.add_argument('--ki',type=float,default=2.0,help="robot_proc's bounded integral loop, 1/s (0 = off). On by decision 2026-10-09 (AGENTS.md): same-demo "
                   "replay error 2.8 mm without vs 0.3-0.5 mm with the integral, and 2.8 mm exceeds a 1.5 mm RL step")
    p.add_argument('--ki-max',type=float,default=1.0,help='integral offset clamp per joint, deg: bounds the extra force while pressing (1.0 = tested with the dimos gains)')
    p.add_argument('--kp',type=float,nargs=6,default=None,help='per-joint kp while engaged (default: robot_proc.DEFAULT_KP 80/80/80/50/20/20; SDK max 200)')
    p.add_argument('--kd',type=float,nargs=6,default=None,help='per-joint kd while engaged (default: robot_proc.DEFAULT_KD 3/3/3/0.7/0.4/0.4; SDK max 5)')
    p.add_argument('--grip-full',type=float,default=None);p.add_argument('--grip-hold',type=float,default=1.5)
    p.add_argument('--temp-pause',type=float,default=75.0);p.add_argument('--temp-resume',type=float,default=60.0)
    p.add_argument('--joint-pause',type=float,default=78.0,help='pause when the hottest arm joint reaches this, C (vendor e-stop 90 C rotor)')
    p.add_argument('--joint-resume',type=float,default=65.0)
    p.add_argument('--start-r',type=float,default=10.0,help='start xy within this radius of the port, mm (paper RAM insertion: +-40)')
    p.add_argument('--hover-min',type=float,default=20.0);p.add_argument('--hover-max',type=float,default=25.0)   # paper: fixed height, +-1 cm xy
    p.add_argument('--top-mm',type=float,default=35.0,help='highest command and travel height above the hub surface, mm (keep above --hover-max)')
    p.add_argument('--floor-mm',type=float,default=15.0,help='lowest command below the calibrated mouth, mm')
    p.add_argument('--press-mm',type=float,default=5.0,help='near the hub: commanded depth below the plug at which the push reaches --press-n')
    p.add_argument('--press-n',type=float,default=11.0,help='near the hub: downward push as a pure vertical force, N (0 = old position press; raise step by step)')
    p.add_argument('--max-steps',type=int,default=100)
    p.add_argument('--travel-speed',type=float,default=40.0,help='scripted moves in free air, mm/s (was 25)')
    p.add_argument('--descend-speed',type=float,default=10.0,help='scripted descent to the start height / onto the hub, mm/s (was 5)')
    p.add_argument('--lift-speed',type=float,default=10.0,help='scripted lift, incl. pulling the plug out after a success, mm/s (was 6; faster risks slip in the fingers)')
    p.add_argument('--step-s',type=float,default=0.1);p.add_argument('--step-mm',type=float,default=1.5)
    p.add_argument('--n',type=int,default=20,help='demo: successful demos to collect; eval: fixed starts')
    p.add_argument('--n-insert',type=int,default=3,help='check: scripted insertions from above')
    p.add_argument('--demos',nargs='*',help='demo session dirs (runs/...-demo): successful episodes seed the offline buffer')
    p.add_argument('--resume',nargs='*',help='train session dirs to resend (learner restarted)')
    p.add_argument('--relabel-usb',action='store_true',help='with --resume: rewards of the resent episodes from the USB referee, not the classifier')
    p.add_argument('--ckpt',help='eval: actor-*.pt from a train session, the learner ckpt-last.pt, or a bc.py act-*.pt; dagger: an act-*.pt')
    p.add_argument('--save-every',type=int,default=10)
    p.add_argument('--touch-every',type=int,default=10,help='grasp height check beside the port every N episodes (0 = never)')
    p.add_argument('--episodes',type=int,default=0)
    p.add_argument('--handback-grace',type=int,default=30,help='steps the policy always gets after a takeover is handed back')
    p.add_argument('--video',action='store_true',help='save full-resolution wrist and side mp4 per episode (ep-NNNN-wrist.mp4 / -side.mp4)')
    p.add_argument('--fake',action='store_true',help='offline dry run: fake arm, USB and side camera')
    a=p.parse_args()
    if a.mode in ('eval','dagger') and not a.ckpt:p.error(f'{a.mode} needs --ckpt')
    if a.hover_max>a.top_mm-5:p.error(f'--hover-max {a.hover_max} must stay at least 5 mm below --top-mm {a.top_mm}')
    if not a.fake:
        for k,v in common.saved_crops().items():
            if getattr(a,k) is None:setattr(a,k,list(v));print(f"{'--crop' if k=='crop' else '--crop-side'} {' '.join(map(str,v))}（来自 cameras.json）",flush=True)
    if (not a.fake and a.mode in ('classify','demo','train','eval','dagger') and 'wrist' in common.parse_views(a.views) and not a.crop):
        p.error('no wrist crop: set it on the console Cameras page (Wrist) or pass --crop. Without it the whole 640x480 frame '
                'is squashed to 128x128 and the port is a few pixels (2026-10-10/11 runs could not find the port).')
    import fcntl
    lock=(common.PREV/'controller.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    r=HilRunner(a)
    try:
        r.setup()
        getattr(r,a.mode)()
        if a.fake:print('MAX_TICK_GAP_MS',round(r.clock.max_gap*1000,1),flush=True)
    except KeyboardInterrupt:pass
    finally:
        try:r.park()
        except KeyboardInterrupt:pass
        if r.keys:r.keys.stop.set()
        try:
            if getattr(r,'robot',None) is not None:r.robot.disengage('shutdown');r.robot.disconnect()
        except Exception:pass
        if getattr(r,'usb',None) is not None and hasattr(r.usb,'stop'):r.usb.stop()
        if r.side is not None:r.side.stop()
        if r.link:r.link.close()
        print(f'SESSION_SAVED {r.session}',flush=True)


if __name__=='__main__':main()
