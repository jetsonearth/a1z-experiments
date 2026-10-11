#!/usr/bin/env python3
"""HIL-SERL run on the A1Z: plug next to the port -> into the mouth. See RL-RECOVERY.md.

Modes (eval.sh --rl <mode> ...):
  snap    save one wrist frame (and the crop preview) to choose --crop; motors stay in float
  check   calibration margin test + 6 zero-action resets; must pass before demo/train
  demo    scripted reset, then you drag the plug into the port (arm floats); 15-20 of these seed the offline buffer
  train   online SAC/RLPD with interventions: Space = take over / hand back
  eval    fixed starts (seed 20261005), --policy rl,spiral,hold alternating per start, deterministic, USB verify

Keys: g = start / continue   Space = take over / hand back (demo, train)   x = abort episode (arm floats)
      p = pause after this episode   q q = quit (arm on its support first)
"""
import argparse
import json
import math
import sys
import threading
import time
from pathlib import Path
from queue import Empty,Queue

import numpy as np

ROOT=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT))
from rl_env import Abort,Clock,Disengaged,EnvCfg,HoldPolicy,InsertEnv,Port,SpiralPolicy   # noqa: E402

SMOOTH={'accel':3.0,'soft_start_s':1.0,'vel_ff':True}   # a1z_client.py default since 2026-10-05 17:50
EVAL_SEED=20261005


def wilson(k,n,z=1.96):
    if n==0:return (float('nan'),float('nan'))
    p=k/n;d=1+z*z/n;c=(p+z*z/(2*n))/d;h=z*math.sqrt(p*(1-p)/n+z*z/(4*n*n))/d
    return (round(c-h,3),round(c+h,3))


class Keys:
    def __init__(self):
        self.q=Queue();self.stop=threading.Event()
        threading.Thread(target=self._run,daemon=True).start()
    def _run(self):
        import os,termios,tty
        fd=sys.stdin.fileno();saved=termios.tcgetattr(fd) if sys.stdin.isatty() else None
        if saved:tty.setcbreak(fd)
        try:
            while not self.stop.is_set():
                k=os.read(fd,1)
                if not k:break
                self.q.put(k.decode(errors='ignore'))
        finally:
            if saved:termios.tcsetattr(fd,termios.TCSADRAIN,saved)
    def drain(self):
        out=[]
        while True:
            try:out.append(self.q.get_nowait())
            except Empty:return out


class Runner:
    def __init__(self,a):
        self.a=a;self.keys=None;self.toggle=False;self.pause=False;self.quit_at=0.0;self.quit=False
        self.session=ROOT/'rl-runs'/(time.strftime('%Y%m%d-%H%M%S')+'-'+a.mode);self.session.mkdir(parents=True)
        self.log=open(self.session/'episodes.jsonl','a',buffering=1)
        self.n=0;self.results=[]

    # ---------- operator ----------
    def hook(self):
        for k in self.keys.drain():
            if k==' ':self.toggle=True
            elif k.lower()=='x':raise Abort('operator x')
            elif k.lower()=='p':self.pause=True;print('\n⏸ 这一轮结束后暂停',flush=True)
            elif k.lower()=='q':
                if time.monotonic()-self.quit_at<2:self.quit=True;raise Abort('quit')
                self.quit_at=time.monotonic();print('\n再按一次 q 退出（先确认手臂在支撑上）',flush=True)

    def wait_go(self,msg):
        print(msg,flush=True)
        if self.a.fake:return True   # dry run: input() already consumed the piped stdin
        while True:
            self.robot.heartbeat()
            for e in self.robot.poll_events():
                if e[0]=='fatal':raise RuntimeError('robot process: '+str(e[1:]))
            for k in self.keys.drain():
                if k.lower()=='g':self.pause=False;self.toggle=False;return True
                if k.lower()=='q':
                    if time.monotonic()-self.quit_at<2:return False
                    self.quit_at=time.monotonic();print('再按一次 q 退出',flush=True)
            time.sleep(0.03)

    def temps(self):
        g,j=self.robot.temps()
        return (None if not np.isfinite(g) else round(g,1)),(None if not np.isfinite(j) else round(j,1))

    def thermal_gate(self):
        g,_=self.temps()
        if g is None or g<self.a.temp_pause:return True
        print(f'\n🌡 夹爪 {g}°C ≥ {self.a.temp_pause}°C：抬起后浮动降温，降到 {self.a.temp_resume}°C 以下再按 g',flush=True)
        try:self.env.lift()
        except (Abort,Disengaged):pass
        self.robot.disengage('thermal pause')
        while True:
            self.robot.heartbeat();time.sleep(0.5);g,_=self.temps()
            print(f'\r夹爪 {g}°C',end='',flush=True)
            if g is not None and g<self.a.temp_resume:break
        print()
        return self.wait_go('已降温。按 g 继续（会重新接合并复位）')

    def park(self):
        """Never cut power with the arm in the air: A1Z has no brakes and robot_proc's stop disables the motors (on
        2026-10-05 21:48 the check mode returned by itself and the arm fell). Float (vendor gravity assist), then wait
        for the operator to put the arm on its support and press q q."""
        robot=getattr(self,'robot',None)
        if robot is None or self.a.fake or not robot.is_connected:return
        try:robot.disengage('end of run')
        except Exception:pass
        print('\n🛑 结束前：手臂已切到浮动（重力辅助）。把手臂放回支撑上，再连按两次 q 断电退出。',flush=True)
        if self.keys is None:input('手臂放回支撑后按回车：');return
        self.keys.drain();quit_at=0.0
        while robot.is_connected:
            robot.heartbeat()
            if any(e[0]=='fatal' for e in robot.poll_events()):return
            for k in self.keys.drain():
                if k.lower()=='q':
                    if time.monotonic()-quit_at<2:return
                    quit_at=time.monotonic();print('再按一次 q 断电退出（手臂会失去支撑力）',flush=True)
            time.sleep(0.03)

    # ---------- setup ----------
    def setup(self):
        if not self.a.fake:
            from a1z_robot import A1Z,A1ZConfig
            from insertion import Kin
            from usb_watch import UsbWatcher
        a=self.a
        if a.fake:   # offline dry run: fake arm from test_rl_env (no hardware is touched)
            import test_rl_env as T
            self.kin=T.FakeKin();calib=Path(T.calib_file())
        else:
            self.kin=Kin();calib=Path(a.calib) if a.calib else sorted((ROOT/'calibration').glob('port-*.json'))[-1]
        self.port=Port(calib,self.kin) if a.mode!='snap' else None
        if self.port:print('标定：',json.dumps(self.port.describe(),ensure_ascii=False),flush=True)
        if input(f'【RL {a.mode}】现场就绪（插头已夹好、hub 固定、急停在手边），输入 READY：').strip()!='READY':sys.exit(0)
        if a.fake:
            self.usb=T.FakeUsb();self.robot=T.FakeRobot((470,100,195))
            self.robot.temps=lambda:(50.0,40.0);self.robot.disconnect=lambda:None;self.robot.ready_info={'fake':True}
        else:
            self.usb=UsbWatcher().start()
            if a.grip_full:import os;os.environ['A1Z_GRIPPER_NM']=str(a.grip_full)   # read by teach.gripper_max_torque in robot_proc
            self.robot=A1Z(A1ZConfig(stiffness=a.stiffness,out_dir=str(self.session),smooth=SMOOTH,grip_hold_nm=a.grip_hold,
                                     integral_gain=a.ki,integral_max_deg=a.ki_max))
            self.robot.connect()
        self.keys=Keys()
        self.clock=Clock(self.robot,hook=self.hook)
        cfg=EnvCfg(side_axis=a.side_axis,crop=tuple(a.crop) if a.crop else None,max_steps=a.max_steps,press_mm=a.press_mm,success_dz_mm=a.success_dz,
                   ticks_per_step=max(1,round(a.step_s*30)),step_mm=a.step_mm,hz=1.0/a.step_s)
        self.env=InsertEnv(self.robot,self.kin,self.port,self.usb,self.clock,cfg) if self.port else None
        meta={'mode':a.mode,'args':vars(a),'port':self.port.describe() if self.port else None,'env_cfg':cfg.__dict__,
              'usb':self.usb.describe(),'robot_ready':getattr(self.robot,'ready_info',None),'t':time.time()}
        (self.session/'session.json').write_text(json.dumps(meta,indent=1,default=str,ensure_ascii=False))
        print(f'会话目录 {self.session}',flush=True)

    def hold_while(self,fn):
        """Run fn in a worker thread while this thread keeps the 30 Hz heartbeat (and, if engaged, re-sends the current
        target, i.e. holds position). Post-processing between episodes took up to 0.6 s in the dry run, longer than
        robot_proc's 0.5 s heartbeat timeout."""
        out={}
        def run():
            try:out['r']=fn()
            except BaseException as e:out['e']=e
        t=threading.Thread(target=run,daemon=True);t.start()
        while t.is_alive():
            if self.robot.engaged and self.env is not None and self.env.p_cmd is not None:
                self.env._send(self.env.p_cmd,self.env.R_cmd);self.clock.tick()
            else:self.clock.tick(need_engaged=False)
        t.join()
        if 'e' in out:raise out['e']
        return out.get('r')

    # ---------- one episode ----------
    def episode(self,start,actor=None,human_start=False,verify=False,deterministic=False,tag='',allow_toggle=True):
        """Returns (episode, meta), or 'fell' if the start fell into the port. actor(env,obs) -> action; None = agent."""
        env=self.env;self.toggle=False
        obs=env.reset(start)
        if obs is None:
            print('起点直接掉进了口，换一个起点',flush=True);env.lift();return 'fell'
        frames=[obs['image']];states=[obs['state']];acts=[];rews=[];terms=[];human=[];infos=[]
        human_mode=False;interventions=0;t0=time.monotonic();ticks0=len(self.clock.events)
        if human_start:self.robot.disengage('takeover');human_mode=True;print('✋ 拖着插头滑进口里（成功会自动结束）',flush=True)
        if actor is not None and hasattr(actor,'reset'):actor.reset(env)
        while True:
            if self.toggle and not allow_toggle:self.toggle=False
            if self.toggle:
                self.toggle=False
                if human_mode:env.handback();human_mode=False;print('▶ 交还策略',flush=True)
                else:self.robot.disengage('takeover');human_mode=True;interventions+=1;print('✋ 你接管了，空格交还',flush=True)
            if human_mode:obs,r,term,trunc,info,a=env.human_step()
            else:
                a=actor(env,obs) if actor is not None else self.agent.act(obs,deterministic=deterministic)
                obs,r,term,trunc,info=env.step(a)
            acts.append(np.asarray(a,np.float32));rews.append(r);terms.append(float(term));human.append(bool(info['human']));infos.append(info)
            frames.append(obs['image']);states.append(obs['state'])
            if term or trunc:break
        dur=round(time.monotonic()-t0,2);success=bool(terms[-1])
        if human_mode:
            if success:print('✓ 进口了，松手，1 秒后接合',flush=True)
            t=time.monotonic()
            while time.monotonic()-t<1.0:self.clock.tick(need_engaged=False)
            env.engage()
        usb=z_end=None
        if success and verify:usb,z_end=env.verify_insert()
        env.lift()
        ep={'frames':np.stack(frames),'states':np.stack(states).astype(np.float32),'actions':np.stack(acts),
            'rewards':np.array(rews,np.float32),'terminals':np.array(terms,np.float32),'human':np.array(human)}
        self.n+=1
        meta={'n':self.n,'tag':tag,'start':start,'success':success,'steps':len(acts),'human_steps':int(np.sum(human)),
              'interventions':interventions,'duration_s':dur,'usb_verified':usb,'z_end_mm':z_end,
              'clamped':{k:int(sum(k in i['clamped'] for i in infos)) for k in ('box','z','press','lead','label_clip')},
              'p_start':env.p_start.round(2).tolist(),'p_end':infos[-1]['p_meas'],'contact_dz_mm':env.contact_dz,'temps':self.temps(),'t':time.time()}
        self.hold_while(lambda:np.savez_compressed(self.session/f'ep-{self.n:04d}.npz',**ep))
        self.log.write(json.dumps(meta)+'\n')
        return ep,meta

    def slip_gate(self,meta):
        """Pause when the plug rests at a different height than calibrated: it has probably moved in the fingers."""
        dz=meta.get('contact_dz_mm')
        if dz is None or abs(dz)<=self.env.cfg.slip_tol_mm:return True
        print(f'\n⚠ 这一轮压到表面时手指高度比标定差 {dz:+.2f} mm（容差 {self.env.cfg.slip_tol_mm} mm）：插头可能在手指里滑了。'
              '检查夹持；滑了就退出、重新标定（只做 1 和 2），再用 --resume 续训。',flush=True)
        return self.wait_go('确认插头没动、只是误报，就按 g 继续；否则 q q 退出')

    def guarded(self,fn,*args,**kw):
        """Run an episode; abort and safety stops float the arm and wait for g. Returns the result or None."""
        try:return fn(*args,**kw)
        except Abort as e:
            self.robot.disengage('operator abort')
            if self.quit:return 'quit'
            print(f'\n已中止（{e}），手臂浮动。',flush=True)
        except Disengaged as e:
            print(f'\n⚠ 安全停止：{e}。手臂浮动。',flush=True)
        self.log.write(json.dumps({'n':self.n+1,'aborted':True,'t':time.time()})+'\n')
        return 'quit' if not self.wait_go('按 g 继续（会重新接合并复位），q q 退出') else None

    # ---------- modes ----------
    def snap(self):
        """Wrist frames for choosing --crop / --side-axis: drag the plug where an episode would start, press g; repeat; q q ends."""
        import cv2
        k=0
        while self.wait_go('把插头拖到插口旁边、搭在表面上（像训练起点那样），扶稳后按 g 拍一张；q 连按两次结束'):
            time.sleep(0.3)
            with self.robot._frame_lock:img=self.robot._frame.copy()
            k+=1;name=f'snap-{k}'
            cv2.imwrite(str(self.session/f'{name}.jpg'),img[:,:,::-1])
            if self.a.crop:
                x,y,w,h=self.a.crop;prev=img.copy();cv2.rectangle(prev,(x,y),(x+w,y+h),(255,0,0),2)
                cv2.imwrite(str(self.session/f'{name}-crop.jpg'),prev[:,:,::-1])
                cv2.imwrite(str(self.session/f'{name}-crop-128.jpg'),cv2.resize(img[y:y+h,x:x+w],(128,128),interpolation=cv2.INTER_AREA)[:,:,::-1])
            q,_=self.robot.measured();p=self.kin.fk(q)[:3,3]*1000
            print(f'已保存 {self.session}/{name}.jpg   手指位置 {p.round(1).tolist()} mm',flush=True)
            if self.a.fake:break

    def check(self):
        env=self.env;c=env.cfg;rng=np.random.default_rng(1);report={'track':[],'press':[],'zero':[]}
        if not self.wait_go('检查：先在插口上方做毫米级跟踪测试，再两侧各压一次，再做 6 次零动作复位。按 g 开始'):return
        def track():
            env.engage();env.lift()
            top=self.port.z_surface+c.top_mm+5.0
            env.goto([self.port.p_port[0],self.port.p_port[1],top],c.travel_mm_s,R_goal=self.port.R_ref)
            out=[]
            for ax in range(3):
                for sgn in (1,-1):
                    p=env.p_cmd.copy();p[ax]+=sgn*c.step_mm
                    for _ in range(30):env._send(p,self.port.R_ref);self.clock.tick()   # hold 1 s
                    pm,_=env.p_meas();out.append({'axis':'xyz'[ax],'dir':sgn,'err_mm':round(float(np.linalg.norm(p-pm)),2)})
            return out
        r=self.guarded(track)
        if r in (None,'quit'):return
        report['track']=r;errs=[x['err_mm'] for x in r]
        print(f'毫米级跟踪：每步 {c.step_mm} mm、停 1 秒后的误差 中位 {np.median(errs):.2f} mm，最大 {max(errs):.2f} mm  {[x["err_mm"] for x in r]}',flush=True)
        for sign in (1,-1):
            off=np.zeros(2);off[0 if c.side_axis=='x' else 1]=sign*5.0
            def press():
                obs=env.reset({'offset_mm':off.tolist(),'hover':False,'hover_dz_mm':0.0})
                print(f'侧 {sign:+d}：轻压接触时比标定表面 {env.contact_dz:+.2f} mm',flush=True)
                zs=[];leads=[]
                for _ in range(15):
                    _,_,_,_,info=env.step([0,0,-1]);zs.append(info['p_meas'][2]);leads.append(info['p_cmd'][2]-info['p_meas'][2])
                env.lift();return obs is not None,min(zs),min(leads)
            r=self.guarded(press)
            if r in (None,'quit'):return
            ok,zmin,lead=r;margin=round(zmin-(self.port.z_mouth+c.success_dz_mm),2)
            report['press'].append({'side':sign,'contact_dz_mm':env.contact_dz,'z_min_mm':round(zmin,2),'margin_mm':margin,'cmd_minus_meas_mm':round(lead,2)})
            print(f'侧 {sign:+d}：压到最低 z={zmin:.2f}，距成功线 {margin} mm',flush=True)
        for i in range(6):
            start=env.sample_start(rng)
            def zero():
                obs=env.reset(start)
                if obs is None:env.lift()
                if obs is None:return 'fell'
                fp=0
                for _ in range(10):
                    _,_,term,_,info=env.step([0,0,0]);fp+=int(info['in_port'])
                env.lift();return fp
            r=self.guarded(zero)
            if r in (None,'quit'):return
            report['zero'].append({'start':start,'false_in_port_steps':r if r!='fell' else None,'fell_in':r=='fell','temps':self.temps()})
            print(f'复位 {i+1}/6：{start}  误判步数 {r}',flush=True)
        report['tick_max_gap_ms']=round(self.clock.max_gap*1000,1)
        errs=[x['err_mm'] for x in report['track']]
        ok=(len(errs)==6 and float(np.median(errs))<=0.7 and max(errs)<=1.5
            and all(p['margin_mm']>=1.0 and p['contact_dz_mm'] is not None and abs(p['contact_dz_mm'])<=c.slip_tol_mm for p in report['press'])
            and all(z['false_in_port_steps'] in (0,None) for z in report['zero']))
        report['pass']=ok;(self.session/'check.json').write_text(json.dumps(report,indent=1))
        print(('✅ 检查通过' if ok else '❌ 检查未通过，看 check.json')+f'   最大节拍间隔 {report["tick_max_gap_ms"]} ms',flush=True)

    def demo(self):
        rng=np.random.default_rng(int(time.time()));k=0
        if not self.wait_go(f'示范：每轮复位后手臂会浮动，你把插头滑进口里。目标 {self.a.n} 条。按 g 开始'):return
        while k<self.a.n:
            if not self.thermal_gate():return
            r=self.guarded(self.episode,self.env.sample_start(rng),human_start=True,tag='demo',allow_toggle=False)
            if r=='quit':return
            if r is None or r=='fell':continue
            ep,meta=r;k+=int(meta['success'])
            if not self.slip_gate(meta):return
            print(f"示范 {meta['n']}: {'✓' if meta['success'] else '✗'}  {meta['steps']} 步  标签截断 {meta['clamped']['label_clip']}   成功示范 {k}/{self.a.n}",flush=True)
            if self.pause and not self.wait_go('已暂停，按 g 继续'):return

    def _load_dirs(self,dirs,online,offline,demo):
        n=0
        for d in dirs or []:
            bad=set()
            jl=Path(d)/'episodes.jsonl'
            for line in (jl.read_text().splitlines() if jl.exists() else []):
                m=json.loads(line)
                if m.get('contact_dz_mm') is not None and abs(m['contact_dz_mm'])>3.0:bad.add(m['n'])   # reset left it in the air
            if bad:print(f'{d}: 跳过起点不在表面的 {sorted(bad)}',flush=True)
            for f in sorted(Path(d).glob('ep-*.npz')):
                if int(f.stem.split('-')[1]) in bad:continue
                z=np.load(f);ep={k:z[k] for k in z.files};ep['feats']=self.agent.encode(ep['frames'])
                if demo:offline.add_episode(ep)
                else:
                    online.add_episode(ep)
                    if ep['human'].any():offline.add_episode(ep,mask=ep['human'])
                n+=1
        return n

    def train(self):
        from rl_agent import Agent,Buffer,LearnerThread
        a=self.a;self.agent=Agent(device='cuda',utd=a.utd,encoder=a.encoder)
        online=Buffer(a.online_cap);offline=Buffer(a.offline_cap)
        nd=self._load_dirs(a.demos,online,offline,demo=True);nr=self._load_dirs(a.resume,online,offline,demo=False)
        if a.ckpt:self.agent.load(a.ckpt);print(f'已加载 {a.ckpt}',flush=True)
        print(f'示范 {nd} 条 → offline {offline.size}；续训 {nr} 条 → online {online.size}',flush=True)
        self.agent.warmup()   # CUDA kernels and first-call costs (0.5 s act, 1.6 s update) before the arm is engaged
        import gc;gc.collect();gc.freeze()   # full collections over the torch/lerobot object graph stalled all threads (dry run: 343 ms)
        learner=LearnerThread(self.agent,online,offline,min_online=a.min_online,max_ratio=a.max_ratio);learner.start()
        rng=np.random.default_rng(int(time.time()));recent=[]
        if not self.wait_go('在线训练：策略自己找口；明显走错或卡住超过 2 秒再按空格接管，带到口边就交还。按 g 开始'):return
        try:
            while True:
                if not self.thermal_gate():return
                r=self.guarded(self.episode,self.env.sample_start(rng),verify=(a.verify_every>0 and self.n%a.verify_every==a.verify_every-1),tag='train')
                if r=='quit':return
                if r is None or r=='fell':continue
                ep,meta=r
                def post():
                    ep['feats']=self.agent.encode(ep['frames']);online.add_episode(ep)
                    if ep['human'].any():offline.add_episode(ep,mask=ep['human'])
                self.hold_while(post)
                if meta['interventions']==0:recent=(recent+[meta['success']])[-20:]
                rate=f"{sum(recent)}/{len(recent)}" if recent else '-'
                info=self.agent.last_info
                print(f"EP {meta['n']}: {'✓' if meta['success'] else '✗'} {meta['steps']}步 接管{meta['interventions']}次  无接管近20轮 {rate}  "
                      f"更新 {info.get('updates',0)}（{learner.dt_ms} ms） Lc {info.get('loss_critic')} α {info.get('temperature')}  "
                      f"online {online.size} offline {offline.size}  夹爪 {meta['temps'][0]}°C"+(f"  USB {meta['usb_verified']}" if meta['usb_verified'] is not None else ''),flush=True)
                if learner.error:print('learner 出错：',learner.error,flush=True);return
                if not self.slip_gate(meta):return
                if a.episodes and meta['n']>=a.episodes:return
                if meta['n']%a.save_every==0:   # torch.save of ~67 MB took ~0.35 s: keep the heartbeat meanwhile
                    self.hold_while(lambda:self.agent.save(self.session/'ckpt-last.pt',{'episodes':meta['n']}))
                    if meta['n']%(4*a.save_every)==0:self.hold_while(lambda:self.agent.save(self.session/f"ckpt-ep{meta['n']:04d}.pt",{'episodes':meta['n']}))
                if self.pause:
                    learner.paused.set()
                    if not self.wait_go('已暂停，按 g 继续'):return
                    learner.paused.clear()
        finally:
            learner.stop.set();learner.join(timeout=10)
            try:self.hold_while(lambda:self.agent.save(self.session/'ckpt-last.pt',{'episodes':self.n}))
            except Exception:self.agent.save(self.session/'ckpt-last.pt',{'episodes':self.n})
            (self.session/'timing.json').write_text(json.dumps({'tick_gaps':self.clock.slow_at,'learner':learner.trace[-3000:]}))
            print(f'已保存 {self.session}/ckpt-last.pt',flush=True)

    def eval(self):
        a=self.a;policies=a.policy.split(',');actors={}
        if 'rl' in policies:
            from rl_agent import Agent
            self.agent=Agent(device='cuda',encoder=a.encoder);self.agent.load(a.ckpt,optim=False);self.agent.warmup();actors['rl']=None
            import gc;gc.collect();gc.freeze()
        if 'spiral' in policies:actors['spiral']=SpiralPolicy()
        if 'hold' in policies:actors['hold']=HoldPolicy()
        rng=np.random.default_rng(EVAL_SEED);starts=[self.env.sample_start(rng) for _ in range(a.n)]
        stats={p:[] for p in policies}
        if not self.wait_go(f'评测：{a.n} 个固定起点 × {policies}，交替进行。按 g 开始'):return
        for i,s in enumerate(starts):
            for p in policies:
                while True:
                    r=self.guarded(self.episode,s,actor=actors[p],verify=a.verify,deterministic=True,tag=f'eval-{p}-{i:02d}',allow_toggle=False)
                    if r=='quit':self._eval_summary(stats);return
                    if r is not None:break
                if r=='fell':print(f'[{i+1}/{a.n}] {p}: 起点直接进口，跳过',flush=True);continue
                ep,meta=r;stats[p].append(meta)
                if not self.slip_gate(meta):self._eval_summary(stats);return
                print(f"[{i+1}/{a.n}] {p}: {'✓' if meta['success'] else '✗'} {meta['duration_s']}s  USB {meta['usb_verified']}",flush=True)
                if not self.thermal_gate():self._eval_summary(stats);return
        self._eval_summary(stats)

    def chain(self):
        """Same-conditions comparison ACT / HG-DAgger / HG-DAgger -> RL from stored start poses (chain.py)."""
        import chain as C
        from rl_agent import Agent
        a=self.a;groups=a.groups.split(',')
        if 'dagger+rl' in groups:
            self.agent=Agent(device='cuda',encoder=a.encoder);self.agent.load(a.ckpt,optim=False);self.agent.warmup()
        else:self.agent=None
        acts={}
        if 'act' in groups:acts['act']=C.ActRunner(self.robot,a.act_ckpt)
        if any(g.startswith('dagger') for g in groups):acts['dagger']=C.ActRunner(self.robot,a.dagger_ckpt)
        starts=json.load(open(ROOT/'chain_starts.json'))['starts'][:a.n]
        cfg={'act_s':a.act_s,'rl_s':a.rl_s,'min_act_s':3.0,'rl_steps':a.max_steps}
        (self.session/'chain_config.json').write_text(json.dumps({'groups':groups,'cfg':cfg,'act_ckpt':a.act_ckpt,'dagger_ckpt':a.dagger_ckpt,
                                                                  'rl_ckpt':a.ckpt,'starts':starts},indent=1))
        stats={g:[] for g in groups}
        if not self.wait_go(f'对比评测：{len(starts)} 个起始姿态 × {groups}，每个起点轮流跑。按 g 开始（手臂会先移到起始姿态）'):return
        raw=self.robot.get_observation()
        for r in acts.values():r.warmup(raw)
        import gc;gc.collect();gc.freeze()
        log=open(self.session/'chain.jsonl','a',buffering=1)
        for i,s in enumerate(starts):
            order=groups[i%len(groups):]+groups[:i%len(groups)]   # rotate the order so no group always goes first
            for g in order:
                while True:
                    r=self.guarded(C.run_trial,self,self.env,g,s['q'],acts,self.agent,cfg)
                    if r=='quit':self._chain_summary(stats);return
                    if r is not None:break
                r.update(start=i,t=time.time());log.write(json.dumps(r)+'\n');stats[g].append(r)
                ph={'act':'策略自己进口','rl':'RL 接手后进口',None:''}[r['phase']]
                print(f"[{i+1}/{len(starts)}] {g}: {'✓ '+ph if r['success'] else '✗'}  {r['duration_s']}s"
                      +(f"  交接 {r['handoff_s']}s" if r.get('handoff_s') else '')+f"  结束时离口 {r['end_xy_from_port_mm']} mm、比表面 {r['end_dz_surface_mm']:+.1f} mm",flush=True)
                if not self.thermal_gate():self._chain_summary(stats);return
        self._chain_summary(stats)

    def _chain_summary(self,stats):
        out={}
        for g,rs in stats.items():
            k=sum(r['success'] for r in rs);n=len(rs);t=[r['t_success_s'] for r in rs if r['success']]
            out[g]={'success':k,'n':n,'wilson95':wilson(k,n),'by_phase':{p:sum(1 for r in rs if r['phase']==p) for p in ('act','rl')},
                    'median_s':float(np.median(t)) if t else None}
            print(f"{g}: 进口 {k}/{n}  95%CI {out[g]['wilson95']}  其中 RL 接手后进口 {out[g]['by_phase']['rl']}  成功中位用时 {out[g]['median_s']}",flush=True)
        (self.session/'chain_summary.json').write_text(json.dumps(out,indent=1))

    def _eval_summary(self,stats):
        out={}
        for p,ms in stats.items():
            k=sum(m['success'] for m in ms);n=len(ms);usb=sum(bool(m['usb_verified']) for m in ms)
            t=[m['duration_s'] for m in ms if m['success']]
            out[p]={'success':k,'n':n,'wilson95':wilson(k,n),'usb':usb,'median_s':float(np.median(t)) if t else None}
            print(f"{p}: 进口 {k}/{n}  95%CI {out[p]['wilson95']}  USB {usb}/{n}  成功中位用时 {out[p]['median_s']}",flush=True)
        (self.session/'eval.json').write_text(json.dumps(out,indent=1))


def main():
    p=argparse.ArgumentParser()
    p.add_argument('mode',choices=['snap','check','demo','train','eval','chain'])
    p.add_argument('--calib',help='calib.py output (default: newest calibration/port-*.json)')
    p.add_argument('--crop',type=int,nargs=4,metavar=('X','Y','W','H'),help='wrist image crop before resizing to 128x128')
    p.add_argument('--side-axis',choices=['x','y'],default='y')
    p.add_argument('--stiffness',type=float,default=None,help='legacy: kp = stiffness x vendor kp with vendor kd (all runs before 2026-10-09); default: dimos gains 80/80/80/50/20/20, kd 3/3/3/0.7/0.4/0.4')
    p.add_argument('--ki',type=float,default=2.0,help="robot_proc's bounded integral loop, 1/s (0 = vendor PD only). The 21:47 check without it: "
                   "the finger lagged the command by ~3 mm and stuck there (static friction ~1 deg), so 1.5 mm RL steps were not executed")
    p.add_argument('--ki-max',type=float,default=1.0,help='integral offset clamp per joint, deg (bounds the extra force in contact)')
    p.add_argument('--grip-full',type=float,default=None,help='gripper N m while engaged for this run (default: gripper_max_torque.txt, 4 N m); '
                   'the coil went 60->76 C in six episodes at 4 N m')
    p.add_argument('--grip-hold',type=float,default=1.5,help='gripper N m while floating (full force only while engaged); 1.5 as in the 2026-10-05 runs')
    p.add_argument('--temp-pause',type=float,default=75.0);p.add_argument('--temp-resume',type=float,default=60.0)
    p.add_argument('--max-steps',type=int,default=80)
    p.add_argument('--step-s',type=float,default=0.1,help='seconds per RL step: the arm settles this long before the next observation. '
                   'The integral loop needs ~0.5-1 s to remove the friction error (2026-10-05 check: 2.6 mm median error after 1.5 mm steps)')
    p.add_argument('--step-mm',type=float,default=1.5,help='max move per RL step and axis')
    p.add_argument('--success-dz',type=float,default=0.5,help='success when the finger frame is at most this far above the calibrated mouth height (mm); negative = must go deeper')
    p.add_argument('--press-mm',type=float,default=5.0,help='how far the command may run below the plug: sets the pressing force (lower = less slip in the fingers)')
    p.add_argument('--n',type=int,default=20,help='demo: successful demos to collect; eval: number of fixed starts')
    p.add_argument('--demos',nargs='*',help='demo session dirs (rl-runs/...-demo) for the offline buffer')
    p.add_argument('--resume',nargs='*',help='train session dirs whose episodes refill the buffers')
    p.add_argument('--ckpt',help='checkpoint to load (train: continue; eval: required for rl)')
    p.add_argument('--policy',default='rl,spiral')
    p.add_argument('--groups',default='act,dagger,dagger+rl',help='chain: groups to compare, run in rotating order at every start pose')
    p.add_argument('--act-ckpt',default='/home/dimos/a1z-infer/checkpoints/060000/pretrained_model')
    p.add_argument('--dagger-ckpt',default='/home/dimos/a1z-infer/checkpoints/hil-e2-ft/pretrained_model')
    p.add_argument('--act-s',type=float,default=20.0,help='chain: imitation-policy time; dagger+rl hands off earlier when the plug stalls near the hub')
    p.add_argument('--rl-s',type=float,default=12.0,help='chain: extra time after act-s; all groups get act-s + rl-s in total')
    p.add_argument('--encoder',choices=['resnet10','resnet18'],default='resnet10',help="frozen image backbone: HIL-SERL's ResNet-10 (default) or torchvision ResNet-18")
    p.add_argument('--utd',type=int,default=2);p.add_argument('--max-ratio',type=float,default=4.0);p.add_argument('--min-online',type=int,default=100)
    p.add_argument('--online-cap',type=int,default=20000);p.add_argument('--offline-cap',type=int,default=5000)
    p.add_argument('--verify-every',type=int,default=0,help='train: every Nth episode press on to full insertion + USB (0 = never: pulling a fully inserted plug back out may slip it in the fingers)')
    p.add_argument('--verify',action='store_true',help='eval: press on to full insertion + USB after each success (same slip risk)')
    p.add_argument('--save-every',type=int,default=10)
    p.add_argument('--episodes',type=int,default=0,help='train: stop after this many episodes (0 = until q q)')
    p.add_argument('--fake',action='store_true',help='offline dry run with the fake arm from test_rl_env.py')
    a=p.parse_args()
    if a.mode=='eval' and 'rl' in a.policy.split(',') and not a.ckpt:p.error('eval with rl needs --ckpt')
    if a.mode=='chain' and 'dagger+rl' in a.groups.split(',') and not a.ckpt:p.error('chain with dagger+rl needs --ckpt')
    import fcntl
    lock=(ROOT/'controller.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    r=Runner(a)
    try:
        r.setup()
        getattr(r,a.mode)()
        if a.fake:print('MAX_TICK_GAP_MS',round(r.clock.max_gap*1000,1),'gaps>100ms',r.clock.slow,flush=True)
    except KeyboardInterrupt:pass
    finally:
        try:r.park()
        except KeyboardInterrupt:pass
        if r.keys:r.keys.stop.set()
        try:
            if getattr(r,'robot',None) is not None:r.robot.disengage('shutdown');r.robot.disconnect()
        except Exception:pass
        if getattr(r,'usb',None) is not None and hasattr(r.usb,'stop'):r.usb.stop()
        print(f'SESSION_SAVED {r.session}',flush=True)


if __name__=='__main__':main()
