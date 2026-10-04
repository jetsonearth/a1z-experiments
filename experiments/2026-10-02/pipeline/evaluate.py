#!/usr/bin/env python3
"""ACT evaluation. Default is offline; live execution requires --execute and READY.
Uses manufacturer position commands and defaults. No custom motion caps.
"""
import argparse
import json
import math
from pathlib import Path
import sys
import time


def finite_action(value,dimension=6):
    import numpy as np
    a=np.asarray(value,dtype=np.float64).reshape(-1)
    if a.shape!=(dimension,) or not np.isfinite(a).all():raise ValueError('Policy action dimension or values invalid')
    return a


def main():
    p=argparse.ArgumentParser();p.add_argument('--checkpoint',type=Path,required=True)
    p.add_argument('--dataset',type=Path,help='Offline inference check dataset')
    p.add_argument('--device',default='cpu');p.add_argument('--execute',action='store_true')
    p.add_argument('--seconds',type=float,default=10,help='Trial duration, not a velocity limit')
    p.add_argument('--output',type=Path);a=p.parse_args()
    if not math.isfinite(a.seconds) or a.seconds<=0:p.error('seconds must be positive')
    import torch
    import numpy as np
    from policy_runtime import load_checkpoint
    torch.set_num_threads(4)
    model,pre,post=load_checkpoint(a.checkpoint,a.device)
    dimension=model.config.output_features['action'].shape[0]
    if dimension not in [6,7]:raise ValueError('Expected six arm joints and optional gripper')
    def infer(image,q):
        obs={'observation.images.wrist':torch.from_numpy(image.copy()).permute(2,0,1).float()/255.,
             'observation.state':torch.tensor(q,dtype=torch.float32)}
        with torch.inference_mode():value=post(model.select_action(pre(obs)))
        return finite_action(value.detach().cpu().numpy(),dimension)
    if not a.execute:
        if a.dataset is None:p.error('Offline mode requires --dataset; no hardware connected')
        from lerobot.datasets.lerobot_dataset import LeRobotDataset
        ds=LeRobotDataset('local/a1z_usb_task',root=a.dataset)
        row=ds[0];start=time.monotonic()
        image=(row['observation.images.wrist'].permute(1,2,0).numpy()*255).round().astype(np.uint8)
        target=infer(image,row['observation.state'].numpy())
        print(json.dumps({'will_connect_hardware':False,'action_rad':target.tolist(),'inference_s':time.monotonic()-start}))
        return
    # A diagnostic-only checkpoint is never a physical task policy.
    report_path=a.checkpoint.parent/'training-report.json'
    if not report_path.exists():raise ValueError('Training provenance report required beside checkpoint')
    if json.loads(report_path.read_text()).get('smoke_only'):raise ValueError('Diagnostic-only checkpoint cannot run a physical task')
    import fcntl
    import pyrealsense2 as rs
    import cv2
    root=Path(__file__).resolve().parents[1];sys.path.insert(0,str(root))
    from teach import build_robot,preflight,verify_revision
    out=a.output or root/'evaluations'/str(time.monotonic_ns())
    out.mkdir(parents=True,exist_ok=False)
    lock=(root/'controller.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    meta={'checkpoint':str(a.checkpoint),'sdk_revision':verify_revision(),'seconds':a.seconds,
          'status':'not_started','outcome':'unlabeled','control':'manufacturer defaults','fps':30}
    robot=bus=pipe=None;started=False;rows=[];images=[]
    try:
        pipe=rs.pipeline();cfg=rs.config()
        cfg.enable_stream(rs.stream.color,640,480,rs.format.rgb8,30)
        cfg.enable_stream(rs.stream.depth,640,480,rs.format.z16,30)
        pipe.start(cfg);started=True;pipe.wait_for_frames(2000)
        print('七维模型会在启动时自动张开夹爪，请先移除夹持物。将在当前位置启用厂家位置保持。策略开始后执行绝对关节目标；结束继续保持。')
        if input('现场就绪，输入 READY：').strip()!='READY':meta['status']='cancelled';return
        robot,bus=build_robot(with_gripper=dimension==7);preflight(robot,bus);robot.zero_gravity_mode=False;robot.start()
        input('当前位置就是评测起点，手已离开运动区域，按回车开始本轮：')
        model.reset();begin=time.monotonic();meta['status']='running'
        try:
            while time.monotonic()-begin<a.seconds:
                if not robot.is_running:raise RuntimeError('Manufacturer controller stopped')
                frame=pipe.wait_for_frames(2000).get_color_frame()
                if not frame:continue
                image=np.asanyarray(frame.get_data()).copy()
                state=robot.get_joint_state();observed=time.monotonic();q=state['pos']
                if dimension==7:
                    if robot.gripper._motor.last_feedback is None:raise RuntimeError('No measured gripper state')
                    q=np.append(q,robot.gripper.get_feedback_norm())
                target=infer(image,q)
                # Same manufacturer's range acceptance used by command_joint_pos.
                accepted=robot._clip_joint_pos(target[:6].copy())
                if accepted is not None:
                    robot.command_joint_pos(accepted)
                    if dimension==7:robot.command_gripper(float(target[6]))
                rows.append({'t_s':time.monotonic()-begin,'observation_host_s':observed,
                             'q':q.tolist(),'requested_action':target.tolist(),
                             'sdk_accepted_action':None if accepted is None else accepted.tolist(),
                             'camera_frame_number':frame.get_frame_number(),'inference_s':time.monotonic()-observed})
                images.append(image)
            meta['status']='completed'
        except KeyboardInterrupt:meta['status']='operator_aborted'
        except Exception as e:meta.update(status='failed',error=repr(e))
        if robot.is_running:
            # Reset the SDK command to measured position, then keep its controller
            # running while operator labels the trial and prepares support.
            robot.estop();robot.release()
            print('本轮结束，当前位置保持。')
        else:print('控制器已停止；记录为失败。')
        outcome=input('现场结果 success / failure / abort：').strip().lower()
        meta['outcome']=outcome if outcome in ['success','failure','abort'] else 'unlabeled'
        if meta['status']!='completed' and meta['outcome']=='success':meta['outcome']='invalid_success_label'
        if robot.is_running:
            if input('需要辅助拖动回支撑时输入 FLOAT：').strip()=='FLOAT':robot.estop()
            while input('已放稳支撑，输入 DISABLE 退出：').strip()!='DISABLE':pass
    finally:
        # SDK shutdown on process exit, including unrecoverable hardware faults.
        if robot is not None:robot.stop()
        if bus is not None:bus.shutdown()
        if pipe is not None and started:pipe.stop()
        (out/'trial.json').write_text(json.dumps({'metadata':meta,'samples':rows},indent=2))
        if images:
            h,w,_=images[0].shape;writer=cv2.VideoWriter(str(out/'wrist.mp4'),cv2.VideoWriter_fourcc(*'mp4v'),30,(w,h))
            if not writer.isOpened():raise RuntimeError('Cannot save eval video; JSON already saved')
            for image in images:writer.write(cv2.cvtColor(image,cv2.COLOR_RGB2BGR))
            writer.release()
        lock.close();print('EVAL_RESULT',out)

if __name__=='__main__':main()
