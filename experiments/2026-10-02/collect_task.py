#!/usr/bin/env python3
"""Repeated kinesthetic episodes in one vendor-default gravity-assisted session."""
import fcntl
import builtins
import json
from pathlib import Path
import threading
import time
from teach import build_robot,preflight,verify_revision,persist_episode
from camera_session_control import CameraSession
from usb_watch import UsbWatcher


_PRINT_LOCK=threading.Lock()


def print(*args,sep=' ',end='\n',flush=False,**kw):
    """Console lines are parsed by the dashboard; emit each as a single locked write so threads cannot interleave."""
    if kw.get('file') is not None:return builtins.print(*args,sep=sep,end=end,flush=flush,**kw)
    with _PRINT_LOCK:
        __import__('sys').stdout.write(sep.join(str(a) for a in args)+end);__import__('sys').stdout.flush()


def recorded_setting(read):
    """Metadata extras must never abort a take: unreadable settings are stored as None."""
    try:return read()
    except Exception:return None


def episode_qc(rows,meta,j4_limit):
    """Quick per-take check printed for the console; never blocks saving."""
    duration=rows[-1]['t_monotonic_s']-rows[0]['t_monotonic_s']
    targets=[r['gripper_target'] for r in rows]
    changes=sum(1 for a,b in zip(targets,targets[1:]) if a!=b)
    j4_max=max(r['q'][3] for r in rows)
    flags=[]
    if duration<2.0:flags.append('过短')
    if changes:flags.append('录制中开合夹爪')
    if meta.get('usb_inserted') is False:flags.append('未插入')
    if j4_max>j4_limit-0.05:flags.append('J4 贴近限位')
    if meta.get('status')!='motion_complete':flags.append('状态 '+str(meta.get('status')))
    return {'frames':len(rows),'duration_s':round(duration,2),'usb_inserted':meta.get('usb_inserted'),
            'gripper_changes':changes,'j4_max':round(j4_max,3),'j4_limit':j4_limit,'flags':flags}


def gripper_feedback(robot):
    """Every scalar field of the gripper motor's last feedback (temperatures, error code, position...)."""
    fb=robot.gripper._motor.last_feedback
    if fb is None:return None
    try:
        import dataclasses
        fields=dataclasses.asdict(fb) if dataclasses.is_dataclass(fb) else fb._asdict() if hasattr(fb,'_asdict') else vars(fb)
    except Exception:return {'repr':repr(fb)[:200]}
    return {k:(round(float(v),3) if isinstance(v,float) else v) for k,v in fields.items() if isinstance(v,(int,float,bool))}


def gripper_observation(robot):
    g=robot.gripper
    fb=g._motor.last_feedback
    return {'gripper_target':float(g.get_pos()),
            'gripper_position':float(g.get_feedback_norm()) if fb is not None else None,
            'gripper_feedback_valid':fb is not None}


class EpisodeBuffer:
    """Thread-safe task boundaries; reset motion never enters a saved episode."""
    def __init__(self):
        self.lock=threading.Lock();self.rows=None
    def begin(self):
        with self.lock:
            if self.rows is not None:raise ValueError('Already recording')
            self.rows=[]
    def append(self,row):
        with self.lock:
            if self.rows is None:return None
            self.rows.append(row);return len(self.rows)
    def finish(self):
        with self.lock:
            if self.rows is None:return None
            rows=self.rows;self.rows=None;return rows


def main():
    root=Path(__file__).resolve().parent
    lock=(root/'controller.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    rev=verify_revision()
    print('连续示教：启动一次，BEGIN/END 连续记录；OPEN/CLOSE 控制夹爪；DISABLE 结束整场。',flush=True)
    print('厂家 SDK 启动时会自动张开夹爪。先移除夹持物，避开夹爪；臂体靠稳支撑。',flush=True)
    print('TASK_PHASE ready',flush=True)
    if input('输入 READY 启动厂家重力辅助和夹爪：').strip()!='READY':return
    robot=bus=camera=usb=None;done=threading.Event();worker=None
    buffer=EpisodeBuffer();episodes=[];faults=[];phase='positioning';current=None
    stamp='20261002-'+str(time.monotonic_ns())
    stage='camera_startup';controller_started=False
    def finish_episode(status='motion_complete'):
        nonlocal current
        rows=buffer.finish()
        if rows is None:return
        meta=current['metadata'];path=current['path']
        if usb is not None and rows:
            meta['usb_events']=usb.events_between(rows[0]['t_monotonic_s'],rows[-1]['t_monotonic_s'])
            meta['usb_inserted']=any(e['connected'] for e in meta['usb_events']) or rows[-1].get('usb_connected') is True
            meta['usb_watch_error']=usb.error
        meta['status']=status if len(rows)>1 and not faults else 'failed'
        if faults:meta['fault']='; '.join(faults)
        meta['finalization_complete']=False
        try:meta['qc']=episode_qc(rows,meta,robot._joint_limits[3][1]) if len(rows)>1 else None
        except Exception as e:meta['qc']={'error':repr(e)}
        persist_episode(path,meta,rows)
        episodes.append((path,meta,rows));current=None
        print('TASK_EPISODE',path,len(rows),flush=True)
        if meta.get('qc'):print('TASK_QC',json.dumps({'path':str(path),**meta['qc']},ensure_ascii=False),flush=True)
    try:
        usb=UsbWatcher().start();print('USB_WATCH',usb.describe(),flush=True)
        camera=CameraSession(root/'datasets'/stamp)
        ready=camera.prepare()
        if not ready.get('ready'):raise RuntimeError(ready.get('error') or 'Camera not ready')
        stage='robot_startup'
        robot,bus=build_robot(with_gripper=True);preflight(robot,bus);robot.start()
        controller_started=True;stage='positioning'
        print('TASK_GRIPPER', 'open' if robot.gripper.get_pos() >= 0.5 else 'closed',flush=True)
        def sample():
            tick=0
            while not done.wait(1/50):
                tick+=1
                if tick%50==0:
                    try:print('TASK_TEMP',json.dumps({'mos':[round(float(x),1) for x in robot._state.temp_mos],'rotor':[round(float(x),1) for x in robot._state.temp_rotor],
                                                      'gripper':gripper_feedback(robot),'gripper_feedback_age_s':None if bus.gripper_last is None else round(time.monotonic()-bus.gripper_last,3)}),flush=True)
                    except Exception:pass
                try:
                    if not robot.is_running:raise RuntimeError('Manufacturer controller stopped')
                    st=robot.get_joint_state()
                    row={'t_monotonic_s':time.monotonic(),'q':st['pos'].tolist(),
                         'dq':st['vel'].tolist(),'effort':st['eff'].tolist(),
                         'feedback_age_s':bus.ages(),'gripper_feedback_age_s':None if bus.gripper_last is None else time.monotonic()-bus.gripper_last,**gripper_observation(robot),
                         'usb_connected':usb.connected}
                    count=buffer.append(row)
                    if count and count%50==0:print('TASK_FRAMES',count,flush=True)
                except Exception as e:
                    faults.append(repr(e));print('TASK_FAULT',repr(e),flush=True);return
        worker=threading.Thread(target=sample,daemon=True);worker.start()
        print('TASK_PHASE positioning',flush=True)
        for line in __import__('sys').stdin:
            command=line.strip().upper()
            if command=='BEGIN' and phase in ['positioning','support']:
                if faults or not robot.is_running or not camera.is_running():raise RuntimeError('Controller/camera unavailable: '+str(faults))
                path=root/'episodes'/('20261002-'+str(time.monotonic_ns())+'-teaching.json');path.parent.mkdir(exist_ok=True)
                meta={'sdk_revision':rev,'source':'task_kinesthetic_vendor_defaults','status':'recording',
                      'units':{'q':'rad','dq':'rad/s','effort':'Nm','gripper':'normalized 0 closed, 1 open'},
                      'experiment_date':'2026-10-02','task':'USB-C manipulation demonstration',
                      'gripper_controlled':True,'gripper_max_torque_nm':recorded_setting(lambda:robot.gripper._i_des*11.0),'joint_limits':recorded_setting(lambda:[list(l) for l in robot._joint_limits]),'gravity_comp_factor':robot.gravity_comp_factor,
                      'controller_started':True,'stage':'recording','session_id':stamp,
                      'camera':{'dataset_dir':str(camera.output),'finalization_pending':True},
                      'velocity_limits':robot._vel_limit.tolist(),'torque_limits':robot._torque_clip.tolist(),
                      'action_label_policy':'next demonstrated arm joint target plus current commanded gripper target',
                      'usb_signal':usb.describe(),'usb_connected_at_begin':usb.connected}
                current={'path':path,'metadata':meta};buffer.begin();phase='recording'
                print('TASK_PHASE recording',flush=True)
            elif command=='END' and phase=='recording':
                finish_episode();phase='support';print('TASK_PHASE support',flush=True)
            elif command in ['OPEN','CLOSE'] and phase in ['positioning','recording','support']:
                if not robot.is_running:raise RuntimeError('Controller stopped')
                robot.command_gripper(1.0 if command=='OPEN' else 0.0)
                print('TASK_GRIPPER', 'open' if command=='OPEN' else 'closed',flush=True)
            elif command=='DISCARD' and phase=='support' and episodes:
                path,meta,rows=episodes[-1]
                meta['excluded']=True;meta['excluded_reason']='operator_discard'
                persist_episode(path,meta,rows);print('TASK_DISCARDED',path,flush=True)
            elif command=='DISABLE' and phase in ['positioning','support']:
                break
            else:print('TASK_IGNORED',command,phase,flush=True)
    except BaseException as exc:
        faults.append(repr(exc));finish_episode('failed')
        print('CAPTURE_ERROR',f'stage={stage} controller_started={controller_started}',repr(exc),flush=True)
    finally:
        finish_episode('operator_stopped');done.set()
        if worker:worker.join(timeout=1)
        if usb is not None:usb.stop()
        if robot is not None:robot.stop()
        if bus is not None:bus.shutdown()
        print('TASK_PHASE finalizing',flush=True)
        camera_result=camera.stop() if camera is not None else {'success':False,'error':'Camera not started'}
        for path,meta,rows in episodes:
            meta['camera']=camera_result;meta['camera_recorded']=camera_result.get('success') is True
            if not meta['camera_recorded']:meta['camera_error']=camera_result.get('error') or 'Camera did not finalize'
            meta['finalization_complete']=True
            persist_episode(path,meta,rows);print('SAVED',path,flush=True)
        print('TASK_SESSION_COMPLETE',len(episodes),flush=True);lock.close()

if __name__=='__main__':main()
