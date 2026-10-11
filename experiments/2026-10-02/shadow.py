#!/usr/bin/env python3
"""Shadow-mode ACT check: vendor gravity assist (same as recording), live wrist camera, live policy.

The policy NEVER commands the arm. The operator moves the arm by hand while each frame's predicted
next joint target is compared with the measured pose. Only OPEN/CLOSE gripper commands are sent,
exactly as during kinesthetic recording.

Keys: Space toggles the gripper; q twice ends the session.
"""
import argparse
import json
from pathlib import Path
import sys
import threading
import time

ROOT=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT));sys.path.insert(0,str(ROOT/'pipeline'))


def main():
    p=argparse.ArgumentParser();p.add_argument('--checkpoint',type=Path,required=True)
    p.add_argument('--device',default='cuda');p.add_argument('--print-every',type=float,default=0.5)
    a=p.parse_args()
    import fcntl
    import numpy as np
    import torch
    import pyrealsense2 as rs
    from policy_runtime import load_checkpoint
    from teach import build_robot,preflight,verify_revision
    model,pre,post=load_checkpoint(a.checkpoint,a.device)
    if model.config.output_features['action'].shape[0]!=7:raise ValueError('Expected 6 joints + gripper')
    def chunk(image,q):
        obs={'observation.images.wrist':torch.from_numpy(image).permute(2,0,1).float()/255.,
             'observation.state':torch.tensor(q,dtype=torch.float32)}
        # fp16 on Jetson: ~26 ms vs ~31 ms, outputs within 0.002 rad of fp32.
        with torch.inference_mode(),torch.autocast('cuda',dtype=torch.float16,enabled=a.device=='cuda'):value=post(model.predict_action_chunk(pre(obs)))
        return value[0].float().cpu().numpy()
    warm=np.zeros((480,640,3),np.uint8)
    for _ in range(3):chunk(warm,np.zeros(7))
    t0=time.monotonic()
    for _ in range(10):chunk(warm,np.zeros(7))
    print('SHADOW_LATENCY_MS',round((time.monotonic()-t0)*100,1),flush=True)
    lock=(ROOT/'controller.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    out=ROOT/'shadow-runs'/time.strftime('%Y%m%d-%H%M%S');out.mkdir(parents=True)
    meta={'checkpoint':str(a.checkpoint),'sdk_revision':verify_revision(),'mode':'shadow; policy never commands arm',
          'gripper_max_torque_nm':(ROOT/'gripper_max_torque.txt').read_text().strip() if (ROOT/'gripper_max_torque.txt').exists() else None}
    rows=[];robot=bus=pipe=None;done=threading.Event()
    try:
        from wrist_camera import enable_wrist
        pipe=rs.pipeline();cfg=rs.config();enable_wrist(rs,cfg)
        cfg.enable_stream(rs.stream.color,640,480,rs.format.rgb8,30)  # same stream as recording
        # Depth must be on too, exactly as in recording: with color alone the D405 color image is ~40% darker
        # under the same light (measured 76 vs 124), which put every inference image off the training distribution.
        cfg.enable_stream(rs.stream.depth,640,480,rs.format.z16,30)
        pipe.start(cfg);pipe.wait_for_frames(3000)
        print('影子模式：厂家重力辅助，与录制相同；模型只预测、不控制手臂。启动时夹爪会张开。',flush=True)
        if input('现场就绪，输入 READY：').strip()!='READY':return
        robot,bus=build_robot(with_gripper=True);preflight(robot,bus);robot.start()
        print('SHADOW_READY  空格：开/合夹爪。结束：先把手臂放回支撑，再连按两次 q（失能后手臂不再受力）。',flush=True)
        grip={'open':robot.gripper.get_pos()>=0.5}
        def commands():
            import termios,tty
            fd=sys.stdin.fileno();saved=termios.tcgetattr(fd) if sys.stdin.isatty() else None
            if saved:tty.setcbreak(fd)  # single keys, no Enter; restored on exit
            try:
                quit_at=0
                while True:
                    key=__import__('os').read(fd,1).decode(errors='ignore')
                    if not key:break
                    if key==' ':
                        grip['open']=not grip['open'];robot.command_gripper(1.0 if grip['open'] else 0.0)
                        print('GRIPPER','OPEN' if grip['open'] else 'CLOSE',flush=True)
                    elif key.lower()=='q':
                        if time.monotonic()-quit_at<2:break
                        quit_at=time.monotonic();print('再按一次 q 结束（先确认手臂已在支撑上）',flush=True)
            finally:
                if saved:termios.tcsetattr(fd,termios.TCSADRAIN,saved)
                done.set()
        threading.Thread(target=commands,daemon=True).start()
        begin=time.monotonic();last=0;skipped=0
        while not done.is_set():
            if not robot.is_running:raise RuntimeError('Manufacturer controller stopped')
            frame=pipe.wait_for_frames(2000).get_color_frame()
            if not frame:continue
            image=np.asanyarray(frame.get_data()).copy()
            st=robot.get_joint_state();fb=robot.gripper._motor.last_feedback
            if fb is None:continue
            q=np.append(np.asarray(st['pos'],dtype=np.float64),robot.gripper.get_feedback_norm());t=time.monotonic()
            # The policy commands nothing, so a bad sample is reported and skipped; only DISABLE ends the session.
            if not np.isfinite(q).all():
                skipped+=1
                if skipped%30==1:print('SHADOW_SKIP 关节读数无效',np.round(q,3).tolist(),flush=True)
                continue
            c=chunk(image,q);infer=time.monotonic()-t
            if not np.isfinite(c).all():
                skipped+=1
                if skipped%30==1:print('SHADOW_SKIP 模型输出无效 q=',np.round(q,3).tolist(),flush=True)
                continue
            step=c[0,:6]-q[:6];ahead=c[min(29,len(c)-1),:6]-q[:6]  # next frame and ~1 s ahead
            rows.append({'t_s':t-begin,'q':q.tolist(),'pred_next':c[0].tolist(),'pred_1s':c[min(29,len(c)-1)].tolist(),
                         'infer_s':infer,'frame':frame.get_frame_number()})
            if t-last>=a.print_every:
                last=t
                print('SHADOW',json.dumps({'t':round(t-begin,1),'skipped':skipped,'infer_ms':round(infer*1000),
                      'next_max_deg':round(float(np.degrees(np.abs(step).max())),2),
                      'ahead_1s_deg':[round(float(x),1) for x in np.degrees(ahead)],
                      'grip_now':round(float(q[6]),2),'grip_pred':round(float(c[0,6]),2)}),flush=True)
    except KeyboardInterrupt:pass
    finally:
        if robot is not None:robot.stop()
        if bus is not None:bus.shutdown()
        if pipe is not None:
            try:pipe.stop()
            except Exception:pass
        (out/'shadow.json').write_text(json.dumps({'metadata':meta,'samples':rows}))
        lock.close();print('SHADOW_SAVED',out,len(rows),flush=True)

if __name__=='__main__':main()
