#!/usr/bin/env python3
"""Supervised vendor-native control. No experiment motion caps or automatic home.
SDK defaults provide PD, gravity, joint limits and minimum-jerk move_joints.
Normal playback ends holding. Disable is an explicit, separate operator action.
"""
import argparse
import fcntl
import json
from pathlib import Path
import threading
import time


def trajectory(path):
    import math
    x=json.loads(Path(path).read_text())
    if 'frames' in x:
        rows=x['frames']
    else:
        if x['metadata'].get('status') != 'motion_complete':
            raise ValueError('Source episode is not complete')
        rows=[(r['t_monotonic_s'],r['q']) for r in x['samples']]
    if len(rows)<2: raise ValueError('Empty trajectory')
    t0=rows[0][0]
    out=[]
    for t,q in rows:
        if len(q)!=6 or not all(math.isfinite(v) for v in [t,*q]):
            raise ValueError('Malformed trajectory')
        if out and t-t0<=out[-1][0]: raise ValueError('Nonmonotonic trajectory')
        out.append((t-t0,q))
    return out


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--execute', action='store_true')
    p.add_argument('--episode',type=Path)
    a=p.parse_args()
    track=trajectory(a.episode) if a.episode else None
    if not a.execute:
        print(json.dumps({'will_connect_hardware':False,'controller':'SDK defaults',
            'frames':len(track) if track else 0,'commands':['hold','float','mark','reset','play','quit']}))
        return
    import numpy as np
    from teach import build_robot, preflight, verify_revision, persist_episode
    root=Path(__file__).resolve().parent
    lock=(root/'controller.lock').open('a')
    fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    rev=verify_revision()
    print('厂家默认位置保持：不会自动归零。请确认底座固定、臂体有支撑、人员离开运动路径。')
    if input('输入 READY 启动当前位置保持：').strip()!='READY': return
    robot=bus=None
    rows=[]; done=threading.Event(); worker=None
    meta={'sdk_revision':rev,'controller':'vendor_native','gravity_comp_factor':1.0,'events':[]}
    try:
        robot,bus=build_robot()
        preflight(robot,bus)
        robot.zero_gravity_mode=False
        robot.start()  # SDK initializes its target from fresh measured position.
        meta.update(kp=robot._default_kp.tolist(),kd=robot._default_kd.tolist(),
                    velocity_limits=robot._vel_limit.tolist(),torque_limits=robot._torque_clip.tolist())
        def sample():
            while not done.wait(.02):
                st=robot.get_joint_state()
                rows.append({'t_monotonic_s':time.monotonic(),'q':st['pos'].tolist(),
                             'dq':st['vel'].tolist(),'running':robot.is_running})
        worker=threading.Thread(target=sample,daemon=True);worker.start()
        marked=None
        print('保持中。hold=当前位置保持；float=厂家重力辅助拖动；mark=记复位点；reset=厂家平滑移动；play=回放；quit=支撑后失能。')
        while robot.is_running:
            try:
                command=input('A1Z> ').strip().lower()
                meta['events'].append({'time':time.monotonic(),'command':command})
                if command=='float': robot.estop()
                elif command=='hold':
                    robot.estop();robot.release()
                elif command=='mark':
                    marked=robot.get_joint_state()['pos'].copy();print('复位点已记录',marked.tolist())
                elif command=='reset':
                    if marked is None: print('先 mark 记录复位点');continue
                    print('目标',marked.tolist())
                    if input('确认路径清空，输入 GO：').strip()!='GO':continue
                    robot.estop();robot.release();robot.move_joints(marked)
                    print('厂家移动函数已返回；继续保持。请现场判断是否到位。')
                elif command=='play':
                    if track is None: print('未选择 episode');continue
                    print('将用厂家 move_joints 接近起点，然后回放，末尾保持。')
                    if input('确认路径清空，输入 GO：').strip()!='GO':continue
                    robot.estop();robot.release()
                    robot.move_joints(np.array(track[0][1]))
                    robot.play_trajectory([(t,np.array(q)) for t,q in track])
                    print('厂家回放函数已返回；继续保持。')
                elif command=='quit':
                    if input('臂体已放稳支撑，输入 DISABLE 失能退出：').strip()=='DISABLE':break
                elif command:print('未知命令')
            except KeyboardInterrupt:
                robot.estop()
                print('\n已调用厂家 estop：暂停指令，保持重力辅助；用 hold 恢复或 quit 退出。')
            except (ValueError,RuntimeError) as e:
                meta['events'].append({'error':repr(e)})
                print('命令未完成：',e)
                # Bad input is not a request to disable a healthy controller.
                if robot.is_running:
                    robot.estop();print('已切换厂家重力辅助模式。')
        if not robot.is_running:meta['controller_fault']=True
    finally:
        if robot is not None: robot.stop()
        # Continue observing briefly if feedback remains available; do not claim
        # this captures motion if the SDK no longer receives motor feedback.
        done.set()
        if worker:worker.join(timeout=1)
        if bus is not None:bus.shutdown()
        out=root/'replay-results';out.mkdir(exist_ok=True)
        path=out/f'{time.monotonic_ns()}-vendor-session.json'
        persist_episode(path,meta,rows)
        print('SESSION_RESULT',path)


if __name__=='__main__':main()
