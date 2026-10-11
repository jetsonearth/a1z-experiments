#!/usr/bin/env python3
"""Control-layer test, no model: given the right answer, can the arm put the plug in?

A1 teach-and-return: insert the plug by hand (USB enumerates), press t to store that joint pose. Pull the plug
  out and hold the arm about 2 cm above the port, press g: the arm engages, moves to --pre-mm straight above the
  stored pose, holds --settle s and reports the horizontal error there (free-air accuracy, no contact), then
  descends vertically (IK every 0.5 mm) to exactly the stored joint angles and holds. Tests free-air accuracy
  and the final press separately.
A2 record-and-replay: press r, drag one full insertion yourself; recording stops 0.5 s after USB enumerates
  (or press r again). Pull the plug out, bring the arm near the start, press p: the arm moves to the recorded
  start and replays the motion with its timing. Tests whether a position-only demonstration (no hand force)
  reproduces the insertion, which is what ACT learns from.

Same control path as policy inference: robot_proc.py (vendor gravity comp 1.0, stiffness x vendor kp, 0.5 rad/s
target-rate cap, 20 deg target-gap stop, heartbeat). The gripper is operator-owned as in eval.
A run ends on USB enumeration (+1 s), --hold seconds after the end of the motion, or any key.
After each run: 1 = not in the port mouth, 2 = in the mouth, no charging, 3 = the phone charges (auto on USB enumeration,
which needs the data pins, i.e. slightly deeper than charging).
Files: control-runs/<session>/run-NN.json + .mp4, flight.jsonl; the taught pose persists in control-runs/taught.json.
"""
import argparse
import json
import os
from pathlib import Path
from queue import Queue
import sys
import threading
import time

ROOT=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT))
RUNS=ROOT/'control-runs'
GRIP_MAX_NM=11.0  # G1Z motor peak; the SDK clips the gripper current limit there


def keys(events,stop):
    import termios,tty
    fd=sys.stdin.fileno();saved=termios.tcgetattr(fd) if sys.stdin.isatty() else None
    if saved:tty.setcbreak(fd)
    try:
        while not stop.is_set():
            k=os.read(fd,1)
            if not k:break
            events.put(k.decode(errors='ignore'))
    finally:
        if saved:termios.tcsetattr(fd,termios.TCSADRAIN,saved)


def load_demo(arg):
    """--demo: a saved A2 recording, or 'latest' (newest control-runs/*/demo-*.json by name = session time + clock time)."""
    if not arg:return None
    f=max(RUNS.glob('*/demo-*.json'),key=lambda x:(x.parent.name,x.name),default=None) if arg=='latest' else Path(arg)
    if f is None or not f.exists():raise SystemExit(f'找不到示范文件：{arg}')
    d=json.loads(f.read_text())
    if len(d.get('samples',[]))<2:raise SystemExit(f'示范文件没有轨迹：{f}')
    d['file']=f'{f.parent.name}/{f.name}';return d


def smooth_plan(q0,q1,rate):
    """Cosine-eased joint interpolation q0 -> q1; duration from the largest joint move at `rate` rad/s (>= 1.5 s)."""
    import numpy as np
    q0=np.asarray(q0,float);q1=np.asarray(q1,float);T=max(1.5,float(np.abs(q1-q0).max())/rate)
    ts=np.linspace(0,T,int(T*30)+1)
    return [(float(t),(q0+(q1-q0)*(1-np.cos(np.pi*t/T))/2).tolist()) for t in ts]


def approach_plan(kin,q_now,q_taught,pre_mm,rate,settle_s,descend_mm_s):
    """A1: joint move to pre_mm straight above the taught finger pose (same orientation), hold settle_s,
    then a straight vertical descent (IK every 0.5 mm) ending exactly on the taught joint angles."""
    import numpy as np
    T=kin.fk(q_taught);p0=T[:3,3];R=T[:3,:3]
    def ik(seed,dz_mm):
        q=kin.step_towards(seed,p0+np.array([0,0,dz_mm/1000]),R,iters=8)
        err=np.linalg.norm(kin.fk(q)[:3,3]-(p0+np.array([0,0,dz_mm/1000])))*1000
        if err>0.2:raise ValueError(f'IK did not converge ({err:.2f} mm)')
        return q
    q_pre=ik(np.asarray(q_taught,float),pre_mm)
    plan=smooth_plan(q_now,q_pre,rate);t1=plan[-1][0]+settle_s;plan.append((t1,q_pre.tolist()))
    steps=max(2,int(round(pre_mm/0.5)));seed=q_pre;dt=pre_mm/descend_mm_s/steps
    for i in range(1,steps+1):
        dz=pre_mm*(1-i/steps);seed=np.asarray(q_taught,float) if i==steps else ik(seed,dz)
        plan.append((t1+i*dt,seed.tolist()))
    return plan,t1


def target_at(plan,t):
    """Plan target at time t (linear between samples; holds the last one)."""
    import bisect
    import numpy as np
    ts=[p[0] for p in plan];i=bisect.bisect_right(ts,t)
    if i>=len(plan):return np.asarray(plan[-1][1])
    if i==0:return np.asarray(plan[0][1])
    (ta,qa),(tb,qb)=plan[i-1],plan[i];w=(t-ta)/(tb-ta)
    return np.asarray(qa)+w*(np.asarray(qb)-np.asarray(qa))


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--stiffness',type=float,default=None,help='legacy: kp = stiffness x vendor kp with vendor kd (all runs before 2026-10-09); default: dimos gains 80/80/80/50/20/20, kd 3/3/3/0.7/0.4/0.4');p.add_argument('--kp',type=float,nargs=6,default=None,help='per-joint kp while engaged (overrides --stiffness; default: dimos 80 80 80 50 20 20)')
    p.add_argument('--kd',type=float,nargs=6,default=None,help='per-joint kd while engaged (default: dimos 3 3 3 0.7 0.4 0.4)')
    p.add_argument('--max-speed',type=float,default=4.0,help="rad/s target rate; 4 = the SDK's commanded-velocity saturation")
    p.add_argument('--approach-rate',type=float,default=0.1,help='rad/s of the largest joint while moving to the taught pose / demo start')
    p.add_argument('--replay-speed',type=float,default=0.5,help='A2 playback speed (1 = as demonstrated)')
    p.add_argument('--hold',type=float,default=6.0,help='seconds to hold the final target before giving up')
    p.add_argument('--max-start-gap',type=float,default=None,help='rad: optionally refuse to start farther than this from the taught pose / demo start (default: no check)')
    p.add_argument('--after-insert',type=float,default=1.0)
    p.add_argument('--pre-mm',type=float,default=15.0,help='A1: hover this far straight above the taught pose before descending')
    p.add_argument('--settle',type=float,default=2.0,help='A1: seconds to hold the hover pose before measuring and descending')
    p.add_argument('--descend-speed',type=float,default=5.0,help='A1: vertical descent speed, mm/s')
    p.add_argument('--ki',type=float,default=2.0,help='integral gain 1/s on joint position (0 = off; default 2 since 2026-10-09)')
    p.add_argument('--grip-hold',type=float,default=1.5,help='gripper force (N m) outside policy control; full force only while the arm is engaged (default 1.5; full force only while engaged)')
    p.add_argument('--ki-max',type=float,default=1.0,help='integral offset clamp per joint, deg')
    p.add_argument('--demo',default=None,help="A2: load a saved recording (path, or 'latest' = newest control-runs/*/demo-*.json) so the same demo is replayed across gain settings")
    a=p.parse_args()
    if not 0<a.replay_speed<=1:p.error('--replay-speed must be in (0,1]')
    import fcntl
    import numpy as np
    from a1z_robot import A1Z,A1ZConfig
    from robot_proc import gains_label
    from usb_watch import UsbWatcher
    from insertion import Kin
    kin=Kin();fk=lambda q:kin.fk(q)[:3,3]*1000
    lock=(ROOT/'controller.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    session=RUNS/time.strftime('%Y%m%d-%H%M%S');session.mkdir(parents=True)
    usb=UsbWatcher().start()
    if usb.connected:print('USB 现在是插着的：开始前请先拔出（成功信号以开始时的状态为基线）',flush=True);return
    taught=json.loads((RUNS/'taught.json').read_text()) if (RUNS/'taught.json').exists() else None
    mode='【控制层测试：没有模型。t=记下插好的姿态  g=走回去(A1)  r=录一条示范  p=回放(A2)】'
    print('='*60+'\n'+mode+'\n'+gains_label(a.stiffness,a.kp,a.kd)+'，'+(f'积分补偿 开（{a.ki}/s，上限 {a.ki_max}°）' if a.ki>0 else '积分补偿 关')+'\n'+'='*60,flush=True)
    if taught:print(f"已有示教点（{taught['saved']}）：{np.round(taught['p_mm'],1).tolist()} mm。夹法变了就重新按 t。",flush=True)
    demo=load_demo(a.demo)
    if demo:print(f"已载入示范 {demo['file']}：{demo['samples'][-1]['t']:.1f}s，{len(demo['samples'])} 帧（{'USB 已确认' if demo['usb_at'] is not None else '未经 USB 确认'}）。按 p 回放",flush=True)
    if input(f'{mode} 现场就绪，输入 READY：').strip()!='READY':return
    robot=A1Z(A1ZConfig(stiffness=a.stiffness,kp=a.kp,kd=a.kd,max_joint_speed=a.max_speed,out_dir=str(session),integral_gain=a.ki,integral_max_deg=a.ki_max,grip_hold_nm=a.grip_hold))
    robot.connect()
    events=Queue();stop=threading.Event();results=[];labels={}
    state='float';run=None;pending=None;rec=None
    def finish(reason):
        nonlocal state,run,pending
        robot.disengage(reason);state='float';robot.stop_video()
        r=run;r['end_reason']=robot.disengage_reason if reason=='robot_disengaged' else reason
        r['duration_s']=round(time.monotonic()-r['t0'],2);r['success']=r['inserted_at'] is not None
        r['log']=list(robot.log);qs=np.array([e['q'] for e in r['log']]) if r['log'] else None
        if qs is not None and len(qs):
            ref=np.array(r['ref_p_mm']);P=np.array([fk(q) for q in qs[::3]]+[fk(qs[-1])])
            end=P[-1];r['end_dxyz_mm']=(end-ref).round(2).tolist();r['lowest_dz_mm']=round(float(P[:,2].min()-ref[2]),2)
            r['end_joint_err_deg']=np.degrees(qs[-1]-np.array(r['ref_q'])).round(3).tolist()
            r['end_cmd_minus_q_deg']=np.degrees(np.array(r['log'][-1]['cmd'])-qs[-1]).round(3).tolist()
        (session/f"run-{r['n']:02d}.json").write_text(json.dumps(r))
        results.append(r['success']);pending=r['n']
        d=r.get('end_dxyz_mm')
        print(f"\nRUN {r['n']} ({r['kind']}): {'USB 接通' if r['success'] else 'USB 未接通'} · {r['end_reason']} · {r['duration_s']}s   累计 {sum(results)}/{len(results)}",flush=True)
        if d:print(f"  结束时离目标 x/y/z {d[0]:+.1f} / {d[1]:+.1f} / {d[2]:+.1f} mm   最低时高度差 {r['lowest_dz_mm']:+.1f} mm   "
                   f"指令-实际 最大 {max(abs(x) for x in r['end_cmd_minus_q_deg']):.2f}°",flush=True)
        if r['success']:set_label(r['n'],3,auto=True)
        print('标注：1=没进口   2=进口了没充电   3=充上电了'+('（已自动标 3，可改）' if r['success'] else ''),flush=True)
        print('回到重力辅助。拔出插头后：g=再走一次示教点   p=再回放   t=重新示教   r=重录示范',flush=True)
        run=None
    def set_label(n,label,auto=False):
        f=session/f'run-{n:02d}.json';r=json.loads(f.read_text())
        r['operator_label']={1:'missed',2:'in_mouth',3:'charging'}[label];r['operator_label_auto']=auto;f.write_text(json.dumps(r));labels[n]=label
        print(f"第 {n} 次标为「{ {1:'没进口',2:'进口没充电',3:'充上电'}[label] }」   充上电 {sum(v==3 for v in labels.values())}/{len(labels)}",flush=True)
    def start(kind,plan,ref_q,extra):
        nonlocal state,run
        if usb.connected:print('USB 还插着：先拔出',flush=True);return
        robot.log=[];n=len(results)+1;robot.start_video(session/f'run-{n:02d}.mp4')
        if not robot.engage():robot.stop_video();print('机械臂未进入控制模式，未开始',flush=True);return
        now=time.monotonic();state='run'
        run={'n':n,'kind':kind,'t0':now,'start_wall':time.time(),'inserted_at':None,'grip_nm':grip_nm,'stiffness':a.stiffness,'kp':robot.ready_info.get('kp',a.kp),'kd':robot.ready_info.get('kd',a.kd),'max_speed':a.max_speed,'ki':a.ki,'ki_max_deg':a.ki_max,
             'plan_s':plan[-1][0],'hold_s':a.hold,'ref_q':list(ref_q),'ref_p_mm':fk(ref_q).tolist(),'usb_signal':usb.describe(),
             'clock':'time.monotonic(), shared with flight.jsonl',**extra}
        run['plan']=plan
        print(f'▶ {kind} 开始：{plan[-1][0]:.1f}s 动作 + 最多保持 {a.hold:.0f}s（任意键停止）',flush=True)
    try:
        threading.Thread(target=keys,args=(events,stop),daemon=True).start()
        gs=robot.ready_info.get('gripper');closed=gs=='closed';grip_nm=float(robot.ready_info.get('grip_full_nm') or 4.0);quit_at=0;t_at=0;stopped_at=0;last_print=0;dt=1/30
        print({'closed':'夹爪保持合着','open':'夹爪开着'}.get(gs,'夹爪位置未读到：放松不出力，按空格夹紧'),flush=True)
        print('就绪。空格=开/合夹爪   t=示教(插好时)   g=走回示教点   r=录示范   p=回放   w=显示当前位置   [ ]=夹力 −/+0.5 N·m   连按两次 q=退出',flush=True)
        while True:
            tick=time.monotonic();robot.heartbeat()
            for e in robot.poll_events():
                if e[0]=='limit':print('⚠ 关节超出限位：'+', '.join(f'J{j} 超出 {d:+.0f}°' for j,d in e[1]),flush=True)
                elif e[0]=='fatal':raise RuntimeError('机械臂进程停止：'+str(e[1]))
                elif e[0]=='disengaged' and state=='run':print('安全停止：'+str(e[1]),flush=True)
                elif e[0]=='grip_force':print(f"夹力已设为 {e[1]['full_nm']} N·m（控制中）"+(f"，浮动时 {e[1]['hold_nm']} N·m" if e[1]['hold_nm'] is not None else ''),flush=True)
            q,_=robot.measured()
            if rec is not None:
                rec['samples'].append({'t':round(tick-rec['t0'],4),'q':q.tolist()})
                if usb.connected and rec['usb_at'] is None:rec['usb_at']=round(tick-rec['t0'],3);print(f"USB 已插入 @ {rec['usb_at']}s，0.5s 后自动停止录制",flush=True)
                if rec['usb_at'] is not None and tick-rec['t0']>=rec['usb_at']+0.5:
                    demo=rec;rec=None;f=session/f"demo-{time.strftime('%H%M%S')}.json";f.write_text(json.dumps(demo));demo['file']=f'{session.name}/{f.name}'
                    print(f"录好了：{demo['samples'][-1]['t']:.1f}s，{len(demo['samples'])} 帧 → {f.name}。拔出插头，把手臂带回起点附近，按 p 回放",flush=True)
            while not events.empty():
                k=events.get()
                if k in '[]':  # live gripper force, 0.5 N m steps, also during a run (does not stop it)
                    grip_nm=round(min(GRIP_MAX_NM,max(0.5,grip_nm+(0.5 if k==']' else -0.5))),2);robot.set_grip_force(grip_nm);continue
                if state=='run':finish('operator_stop');stopped_at=tick;continue
                if k in '123' and pending is not None:set_label(pending,int(k));continue
                if k==' ':
                    if tick-stopped_at<1.0:continue
                    closed=not closed;robot.command_gripper(closed);print('夹爪','合' if closed else '开',flush=True)
                elif k.lower()=='t':
                    if not usb.connected and tick-t_at>2:
                        t_at=tick;print('USB 没识别到。确定插到底了就 2 秒内再按一次 t 强制记下（会注明未经 USB 确认）',flush=True);continue
                    t_at=0
                    taught={'q':q.tolist(),'p_mm':fk(q).tolist(),'saved':time.strftime('%Y-%m-%d %H:%M:%S'),'session':session.name,'usb_verified':bool(usb.connected)}
                    (RUNS/'taught.json').write_text(json.dumps(taught));(session/'taught.json').write_text(json.dumps(taught))
                    print(f"示教点已记下：{np.round(taught['p_mm'],1).tolist()} mm。拔出插头，把手臂拖到口上方约 2 cm，按 g",flush=True)
                elif k.lower()=='g':
                    if taught is None:print('还没有示教点：先插好按 t',flush=True);continue
                    if not closed:print('夹爪是开的：先按空格夹紧',flush=True);continue
                    gap=float(np.abs(np.array(taught['q'])-q).max())
                    if a.max_start_gap is not None and gap>a.max_start_gap:print(f'离示教点太远（最大关节差 {np.degrees(gap):.0f}°）：先拖到口上方 2 cm 左右',flush=True);continue
                    above=fk(q)-np.array(taught['p_mm'])
                    try:plan,check=approach_plan(kin,q,taught['q'],a.pre_mm,a.approach_rate,a.settle,a.descend_speed)
                    except ValueError as e:print('算不出口上方的姿态：',e,flush=True);continue
                    start('A1 示教点',plan,taught['q'],{'start_offset_mm':above.round(2).tolist(),'taught':taught,'check_at':check,
                          'pre_mm':a.pre_mm,'settle_s':a.settle,'descend_mm_s':a.descend_speed})
                elif k.lower()=='r':
                    if rec is None:
                        if usb.connected:print('USB 还插着：先拔出再录',flush=True);continue
                        rec={'t0':tick,'usb_at':None,'samples':[],'start_wall':time.time()};print('● 开始录制：拖着完整插一次，USB 识别后自动停止；没识别就插到底后再按 r 停止保存',flush=True)
                    elif rec['samples'] and rec['samples'][-1]['t']>=1.0:
                        demo=rec;rec=None;f=session/f"demo-{time.strftime('%H%M%S')}.json";f.write_text(json.dumps(demo));demo['file']=f'{session.name}/{f.name}'
                        print(f"录制停止并保存（{'USB 已确认' if demo['usb_at'] is not None else '未经 USB 确认'}）：{demo['samples'][-1]['t']:.1f}s → {f.name}。拔出插头，拖回起点附近，按 p 回放",flush=True)
                    else:rec=None;print('录制太短（<1 s），未保存',flush=True)
                elif k.lower()=='p':
                    if demo is None:print('还没有录好的示范：按 r 录一条',flush=True);continue
                    if not closed:print('夹爪是开的：先按空格夹紧',flush=True);continue
                    s=demo['samples'];q_start=np.array(s[0]['q']);gap=float(np.abs(q_start-q).max())
                    if a.max_start_gap is not None and gap>a.max_start_gap:print(f'离示范起点太远（最大关节差 {np.degrees(gap):.0f}°）：先拖回起点附近',flush=True);continue
                    move=smooth_plan(q,q_start,a.approach_rate);t_off=move[-1][0]+0.5
                    plan=move+[(t_off+x['t']/a.replay_speed,x['q']) for x in s]
                    start('A2 回放',plan,s[-1]['q'],{'replay_speed':a.replay_speed,'demo_usb_at':demo['usb_at'],'demo_samples':len(s),'demo_file':demo.get('file')})
                elif k.lower()=='w':
                    print(f'当前夹爪尖位置 x/y/z = {np.round(fk(q),1).tolist()} mm（基座坐标）',flush=True)
                elif k.lower()=='q':
                    if tick-quit_at<2:raise KeyboardInterrupt
                    quit_at=tick;print('再按一次 q 退出（先确认手臂在支撑上）',flush=True)
            if state=='run':
                t=tick-run['t0']
                if not robot.engaged:finish('robot_disengaged')
                else:
                    robot.send_target(target_at(run['plan'],t),run['kind'])
                    if run.get('check_at') is not None and 'hover_dxyz_mm' not in run and t>=run['check_at']:
                        d=fk(q)-np.array(run['ref_p_mm']);run['hover_dxyz_mm']=d.round(2).tolist()
                        print(f"  【空中对准，口上方 {run['pre_mm']:.0f} mm 停稳后】x {d[0]:+.1f}  y {d[1]:+.1f} mm（水平共 {np.hypot(d[0],d[1]):.1f} mm），高度 {d[2]:+.1f} mm（目标 +{run['pre_mm']:.0f}）。开始竖直下降",flush=True)
                    if usb.connected and run['inserted_at'] is None:run['inserted_at']=round(t,2);print(f'USB 已插入 @ {run["inserted_at"]}s',flush=True)
                    if run['inserted_at'] is not None and t-run['inserted_at']>=a.after_insert:finish('usb_inserted')
                    elif t>run['plan_s']+a.hold:finish('hold_timeout')
                    elif tick-last_print>=0.5:
                        last_print=tick;d=fk(q)-np.array(run['ref_p_mm'])
                        print(f"  t={t:4.1f}s  离目标 x/y/z {d[0]:+5.1f}/{d[1]:+5.1f}/{d[2]:+5.1f} mm",flush=True)
            if not robot.is_connected:raise RuntimeError('机械臂进程已退出')
            time.sleep(max(0,dt-(time.monotonic()-tick)))
    except KeyboardInterrupt:pass
    finally:
        stop.set()
        if run is not None:
            try:finish('shutdown')
            except Exception:pass
        usb.stop();robot.disconnect()
        (session/'events.json').write_text(json.dumps(robot.events,default=str))
        lock.close()
        print(f'SESSION_SAVED {session}  USB 接通 {sum(results)}/{len(results)}',flush=True)

if __name__=='__main__':main()
