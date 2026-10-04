#!/usr/bin/env python3
"""Operator client for ACT on the A1Z via LeRobot async inference.

Processes: policy server (own process) | this client: camera, gRPC, keys, video | robot_proc: vendor SDK alone.

Keys (no Enter):
  floating:  Space = open/close gripper   g = start a trial   q q = quit (arm on its support first)
  running:   ANY key = stop -> back to floating gravity assist
A trial also ends by itself: USB enumerated + --after-insert seconds (success), --timeout, or a robot-process
safety stop. Each trial saves trial-NN.json + trial-NN.mp4; the robot process logs flight.jsonl (50 Hz, all modes).
--no-policy: drag-only session (no server, no 'g'), for checking the floating mode and the SDK loop rate.

--hil: HG-DAgger data collection (Kelly et al. 2019, Algorithm 1). The novice (policy) drives; the operator is
the gating function and holds uninterrupted control between takeover and handback:
  policy driving:  Space = take over (arm floats, you drive it)   any other key = end the trial
  you driving:     Space = hand back to the policy                 x = end the trial
Only operator-driven stretches become training labels (export_hil.py); policy stretches are kept, flagged, for
evaluation and later RL. After each HIL trial: d = discard its intervention data (default keep).
"""
import argparse
import json
import logging
import os
from pathlib import Path
from queue import Queue
import sys
import threading
import time

ROOT=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT))


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


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--checkpoint');p.add_argument('--server',default='127.0.0.1:8080');p.add_argument('--no-policy',action='store_true')
    p.add_argument('--stiffness',type=float,default=1.0);p.add_argument('--max-speed',type=float,default=0.5)
    p.add_argument('--timeout',type=float,default=35);p.add_argument('--after-insert',type=float,default=1.0)
    p.add_argument('--actions-per-chunk',type=int,default=30);p.add_argument('--threshold',type=float,default=0.7)
    p.add_argument('--search',action='store_true',help='after the policy stalls on the hub, run the scripted spiral search / push (off by default)')
    p.add_argument('--hil',action='store_true',help='HG-DAgger collection: Space takes over / hands back during a trial')
    p.add_argument('--epoch',type=int,help='HG-DAgger epoch i whose novice policy is being rolled out (recorded in each trial)')
    a=p.parse_args()
    if not a.no_policy and not a.checkpoint:p.error('--checkpoint required unless --no-policy')
    if a.hil and (a.no_policy or a.search):p.error('--hil needs the policy and no scripted search')
    if a.hil and a.epoch is None:p.error('--hil needs --epoch (1 for the first round after behavioral cloning)')
    import fcntl
    import numpy as np
    from a1z_robot import A1Z,A1ZConfig
    from usb_watch import UsbWatcher
    from insertion import Inserter,Kin
    kin=Kin() if a.search and not a.no_policy else None
    lock=(ROOT/'controller.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    session=ROOT/('hil-runs' if a.hil else 'eval-runs')/time.strftime('%Y%m%d-%H%M%S');session.mkdir(parents=True)
    usb=UsbWatcher().start()
    robot=A1Z(A1ZConfig(stiffness=a.stiffness,max_joint_speed=a.max_speed,out_dir=str(session)))
    if a.no_policy:print('拖动测试：只有重力辅助（和录制一样），不加载模型。',flush=True)
    elif a.hil:print(f'HG-DAgger 第 {a.epoch} 轮采集：模型先开，觉得要出问题就按空格接管，你拖着做完或救回来后再按空格交还。刚度 {a.stiffness:.0%}，超时只算模型开的时间 {a.timeout:.0f}s。',flush=True)
    else:print(f'ACT 推理：刚度 {a.stiffness:.0%} 厂家默认，目标速度上限 {a.max_speed} rad/s，超时 {a.timeout:.0f}s。',flush=True)
    print('启动后是厂家重力辅助（和录制一样）。夹爪不会张开：合着就夹紧，开着就不动，读不到位置就放松。机械臂在独立进程里运行。',flush=True)
    mode='【HIL 采集模式：空格=接管/交还，接管数据会录下来】' if a.hil else ('【拖动测试】' if a.no_policy else '【评估模式：不录接管数据，按键=停止】')
    print('='*60+'\n'+mode+'\n'+'='*60,flush=True)
    if input(f'{mode} 现场就绪，输入 READY：').strip()!='READY':return
    robot.connect()
    client=None
    if not a.no_policy:
        import lerobot.async_inference.robot_client as rc
        from lerobot.async_inference.configs import RobotClientConfig
        rc.make_robot_from_config=lambda cfg:robot  # hand our connected robot to LeRobot's client
        cfg=RobotClientConfig(policy_type='act',pretrained_name_or_path=a.checkpoint,robot=robot.config,
                              actions_per_chunk=a.actions_per_chunk,server_address=a.server,policy_device='cuda',
                              chunk_size_threshold=a.threshold,fps=30,aggregate_fn_name='weighted_average')
        client=rc.RobotClient(cfg);client.logger.setLevel(logging.WARNING)
    events=Queue();stop=threading.Event();results=[]
    state='float';trial=None;ins=None;labels={};pending=None
    def segment(controller,now):
        """Close the open control segment and open the next one (controller None = trial over)."""
        segs=trial['segments']
        if segs and segs[-1]['t1'] is None:segs[-1]['t1']=now
        if controller:segs.append({'controller':controller,'t0':now,'t1':None})
    def reset_policy_stream():
        """Fresh start for the novice: drop queued and in-flight chunks predicted from older observations."""
        with client.action_queue_lock:client.action_queue=Queue()
        with client.latest_action_lock:client.latest_action+=1000
        client.must_go.set()
    def policy_time(now):
        """Seconds the policy has been in control this trial (the timeout does not count operator time)."""
        return sum((x['t1'] or now)-x['t0'] for x in trial['segments'] if x['controller']=='policy')
    def finish(reason):
        nonlocal state,trial,ins,pending
        was=state;robot.disengage(reason);segment(None,time.monotonic());state='float';robot.stop_video()
        t=trial;t['end_reason']=reason if was=='human' else (robot.disengage_reason or reason)  # 'takeover' is not an end
        t['duration_s']=round(time.monotonic()-t['t0'],2)
        t['success']=t['inserted_at'] is not None;t['log']=list(robot.log);t['stages']=ins.events if ins else []
        human=[x for x in t['segments'] if x['controller']=='human']
        t['interventions']=len(human);t['human_s']=round(sum(x['t1']-x['t0'] for x in human),2)
        t['autonomous_success']=t['success'] and not human
        if a.hil:t['hil_keep']=True
        (session/f"trial-{t['n']:02d}.json").write_text(json.dumps(t))
        results.append(t['autonomous_success'] if a.hil else t['success'])
        pending=t['n']
        if a.hil:
            print(f"\nTRIAL {t['n']}: {'USB 接通' if t['success'] else 'USB 未接通'} · 接管 {t['interventions']} 次 共 {t['human_s']}s · {t['end_reason']}   "
                  f"模型独立完成 {sum(results)}/{len(results)}",flush=True)
            print('这一轮的接管数据默认保留；纠正得不干净就按 d 丢弃。',flush=True)
        else:
            print(f"\nTRIAL {t['n']}: {'成功' if t['success'] else '失败'} · {t['end_reason']} · {t['duration_s']}s   累计 {sum(results)}/{len(results)}",flush=True)
            if t['success']:set_label(t['n'],3,auto=True)
            print('标注这一轮：1=没找到口   2=找到口（进口了，没插到底）   3=插到底'+('（USB 已接通，已自动标 3，可改）' if t['success'] else ''),flush=True)
        print('已回到重力辅助：拔出插头、带回起点，空格=开/合夹爪（调整插头），夹好后按 g 开始下一轮。',flush=True)
        trial=None;ins=None
    def discard(n):
        f=session/f'trial-{n:02d}.json';t=json.loads(f.read_text());t['hil_keep']=False;f.write_text(json.dumps(t))
        print(f'第 {n} 轮的接管数据已标为丢弃（文件仍保留，导出时跳过）',flush=True)
    def set_label(n,label,auto=False):
        f=session/f'trial-{n:02d}.json';t=json.loads(f.read_text())
        t['operator_label']={1:'missed',2:'found_port',3:'inserted'}[label];t['operator_label_auto']=auto
        f.write_text(json.dumps(t));labels[n]=label
        found=sum(1 for v in labels.values() if v>=2);ins_n=sum(1 for v in labels.values() if v==3)
        print(f"第 {n} 轮标为「{ {1:'没找到口',2:'找到口',3:'插到底'}[label] }」   累计（已标 {len(labels)} 轮）：找到口 {found}/{len(labels)}   插到底 {ins_n}/{len(labels)}",flush=True)
    try:
        if client is not None:
            if not client.start():raise RuntimeError('连不上 policy server（tmux 里 server 窗口是否在运行？）')
            threading.Thread(target=client.receive_actions,daemon=True).start()
            client.start_barrier.wait()  # we are the control-loop party
        threading.Thread(target=keys,args=(events,stop),daemon=True).start()
        print('就绪。空格=开/合夹爪   '+('' if a.no_policy else 'g=开始一轮   运行中按任意键=停止   ')+'连按两次 q=退出',flush=True)
        gs=robot.ready_info.get('gripper');closed=gs=='closed';quit_at=0;stopped_at=0;last_print=0;last_hz=0;dt=1/30
        print({'closed':'夹爪启动时保持原状：合着，已夹紧','open':'夹爪启动时保持原状：开着'}.get(gs,'夹爪位置未读到：保持放松不出力，按空格夹紧'),flush=True)
        while True:
            tick=time.monotonic();robot.heartbeat()
            for e in robot.poll_events():
                if e[0]=='limit':print('⚠ 关节超出限位：'+', '.join(f'J{j} 超出 {d:+.0f}°' for j,d in e[1]),flush=True)
                elif e[0]=='fatal':raise RuntimeError('机械臂进程停止：'+str(e[1]))
                elif e[0]=='disengaged' and state=='run':print('安全停止：'+str(e[1]),flush=True)
            while not events.empty():
                k=events.get()
                if state=='run':
                    if a.hil and k==' ':  # takeover: the operator becomes the gate and holds control until handback
                        robot.disengage('takeover');segment('human',time.monotonic());state='human';stopped_at=tick
                        print('✋ 你接管了（重力辅助）。拖完或救回来后：空格=交还模型   x=结束这一轮',flush=True)
                    else:finish('operator_stop');stopped_at=tick
                    continue
                if state=='human':
                    if k==' ':
                        reset_policy_stream()
                        if robot.engage():segment('policy',time.monotonic());state='run';print('▶ 交还模型（空格=再次接管）',flush=True)
                        else:print('机械臂未进入控制模式，仍由你控制',flush=True)
                    elif k.lower()=='x':finish('operator_end');stopped_at=tick
                    else:print('你在控制中：空格=交还模型   x=结束这一轮',flush=True)
                    continue
                if a.hil and k.lower()=='d' and pending is not None:
                    discard(pending);continue
                if k in '123' and pending is not None and client is not None and not a.hil:
                    set_label(pending,int(k));continue
                if k==' ':
                    if tick-stopped_at<1.0:continue  # a stop key press must not also open the gripper
                    closed=not closed;robot.command_gripper(closed);print('夹爪','合' if closed else '开',flush=True)
                elif k.lower()=='g' and client is not None:
                    if usb.connected:print('USB 还插着：先拔出再开始',flush=True);continue
                    if not closed:print('夹爪是开的：先按空格夹紧插头',flush=True);continue
                    if robot.frame_age()>0.25:print('相机画面过期，稍等',flush=True);continue
                    reset_policy_stream();robot.log=[]  # drop any late chunk from the last trial
                    n=len(results)+1;robot.start_video(session/f'trial-{n:02d}.mp4')
                    if not robot.engage():robot.stop_video();print('机械臂未进入控制模式，未开始',flush=True);continue
                    state='run';ins=Inserter(kin) if kin else None;now=time.monotonic()
                    trial={'n':n,'t0':now,'start_wall':time.time(),'inserted_at':None,'checkpoint':a.checkpoint,'hil':a.hil,'epoch':a.epoch,
                           'segments':[{'controller':'policy','t0':now,'t1':None}],'clock':'time.monotonic(), shared with flight.jsonl and trial-NN.timestamps.json',
                           'stiffness':a.stiffness,'max_speed':a.max_speed,'actions_per_chunk':a.actions_per_chunk,
                           'threshold':a.threshold,'usb_signal':usb.describe()}
                    print('▶ 模型开始'+('（空格=接管，其他键=结束）' if a.hil else '（任意键停止）'),flush=True)
                elif k.lower()=='q':
                    if tick-quit_at<2:raise KeyboardInterrupt
                    quit_at=tick;print('再按一次 q 退出（先确认手臂在支撑上）',flush=True)
            if state=='human':
                if usb.connected and trial['inserted_at'] is None:
                    trial['inserted_at']=round(tick-trial['t0'],2);print(f'USB 已插入 @ {trial["inserted_at"]}s',flush=True)
                if trial['inserted_at'] is not None and tick-trial['t0']-trial['inserted_at']>=a.after_insert:finish('usb_inserted')
            elif state=='run':
                if not robot.engaged:finish(robot.disengage_reason or 'robot_disengaged')
                else:
                    stage,target=('policy',None)
                    if ins:
                        q,_=robot.measured();prev=ins.stage
                        stage,target=ins.update(tick-trial['t0'],q,robot.commanded())
                        if ins.stage!=prev:print('阶段 →',{'search':'螺旋搜索（轻压 + 小螺旋）','push':'插头进口，往下推','search_failed':'搜索超时','push_failed':'下推超时'}.get(ins.stage,ins.stage),flush=True)
                    if stage=='policy':
                        if client.actions_available():client.control_loop_action()
                        if client._ready_to_send_observation():client.control_loop_observation(task='')
                    elif target is not None:robot.send_target(target,stage)
                    elif stage in ('search_failed','push_failed'):finish(stage);continue
                    if usb.connected and trial['inserted_at'] is None:
                        trial['inserted_at']=round(tick-trial['t0'],2);print(f'USB 已插入 @ {trial["inserted_at"]}s',flush=True)
                    if trial['inserted_at'] is not None and tick-trial['t0']-trial['inserted_at']>=a.after_insert:finish('usb_inserted')
                    elif policy_time(tick)>a.timeout:finish('timeout')
                    elif tick-last_print>=0.5:
                        last_print=tick;q,_=robot.measured()
                        print('RUN',json.dumps({'t':round(tick-trial['t0'],1),'queue':client.action_queue.qsize(),
                              'track_err_deg':round(float(np.degrees(np.abs(robot.commanded()-q).max())),1),'sdk_hz':round(robot.sdk_hz),'stage':ins.stage if ins else 'policy'}),flush=True)
            elif tick-last_hz>=5:
                last_hz=tick;print(f'[厂家控制循环 {robot.sdk_hz:.0f} Hz]',flush=True)
            if not robot.is_connected:raise RuntimeError('机械臂进程已退出')
            time.sleep(max(0,dt-(time.monotonic()-tick)))
    except KeyboardInterrupt:pass
    finally:
        stop.set()
        if trial is not None:
            try:finish('shutdown')
            except Exception:pass
        usb.stop()
        if client is not None:client.stop()
        else:robot.disconnect()
        (session/'events.json').write_text(json.dumps(robot.events,default=str))
        lock.close()
        print(f'SESSION_SAVED {session}  成功 {sum(results)}/{len(results)}',flush=True)

if __name__=='__main__':main()
