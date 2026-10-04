#!/usr/bin/env python3
"""Port height calibration in floating mode (vendor gravity assist, no policy, nothing commanded).

Hold the plug at a position and press its key; 1 s of finger-frame positions is averaged and saved.
  1 = plug resting on the hub surface next to the port
  2 = plug just inside the port mouth, not pushed down
  3 = plug fully inserted (USB connected)
Space = gripper open/close, q q = quit (arm on its support first). Press a key several times at
different spots to average. Output: calibration/port-<time>.json, used by eval_metrics.py.
"""
import json
import os
from pathlib import Path
from queue import Queue
import sys
import threading
import time

ROOT=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT))
LABELS={'1':'surface','2':'mouth','3':'inserted'}


def main():
    import fcntl
    import numpy as np
    from a1z_robot import A1Z,A1ZConfig
    from a1z_client import keys
    from insertion import Kin
    from usb_watch import UsbWatcher
    lock=(ROOT/'controller.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    out=ROOT/'calibration';out.mkdir(exist_ok=True)
    path=out/time.strftime('port-%Y%m%d-%H%M%S.json')
    kin=Kin();usb=UsbWatcher().start()
    robot=A1Z(A1ZConfig(out_dir=str(out)))
    print('接口高度校准：只有重力辅助（和录制一样），不加载模型，不主动驱动手臂。',flush=True)
    if input('现场就绪，输入 READY：').strip()!='READY':return
    robot.connect()
    closed=robot.ready_info.get('gripper')=='closed'
    print('1=插头搭在口旁边的表面上   2=刚进口没推下去   3=完全插到底   空格=夹爪   连按两次 q=退出',flush=True)
    events=Queue();stop=threading.Event();threading.Thread(target=keys,args=(events,stop),daemon=True).start()
    samples=[];quit_at=0
    try:
        while True:
            robot.heartbeat();robot.poll_events()
            while not events.empty():
                k=events.get()
                if k==' ':closed=not closed;robot.command_gripper(closed);print('夹爪','合' if closed else '开',flush=True)
                elif k in LABELS:
                    ps=[]
                    t_end=time.monotonic()+1.0
                    while time.monotonic()<t_end:
                        robot.heartbeat();q,_=robot.measured();ps.append(kin.fk(q)[:3,3]*1000);time.sleep(1/30)
                    ps=np.array(ps);q,_=robot.measured()
                    samples.append({'label':LABELS[k],'p_mm':ps.mean(0).round(2).tolist(),'std_mm':ps.std(0).round(2).tolist(),
                                    'q':q.tolist(),'usb_connected':usb.connected,'t':time.time()})
                    path.write_text(json.dumps({'frame':'gripper_finger_rIght_link in base frame, mm','samples':samples},indent=1))
                    print(f"记录 {LABELS[k]}: z={ps[:,2].mean():.1f} mm (晃动 {ps[:,2].std():.2f}) USB={'接通' if usb.connected else '未接通'}",flush=True)
                    by={l:np.array([s['p_mm'] for s in samples if s['label']==l]) for l in LABELS.values()}
                    if len(by['inserted']) and (len(by['surface']) or len(by['mouth'])):
                        z0=by['inserted'][:,2].mean()
                        print('  相对插到底：'+'  '.join(f"{l} {by[l][:,2].mean()-z0:+.1f} mm" for l in ('surface','mouth') if len(by[l])),flush=True)
                elif k.lower()=='q':
                    if time.monotonic()-quit_at<2:raise KeyboardInterrupt
                    quit_at=time.monotonic();print('再按一次 q 退出（先确认手臂在支撑上）',flush=True)
            if not robot.is_connected:raise RuntimeError('机械臂进程已退出')
            time.sleep(1/30)
    except KeyboardInterrupt:pass
    finally:
        stop.set();usb.stop();robot.disconnect();lock.close()
        print('CALIBRATION_SAVED',path if samples else '(no samples)',flush=True)

if __name__=='__main__':main()
