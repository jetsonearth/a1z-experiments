#!/usr/bin/env python3
"""Convert explicit task episodes to LeRobot ACT data; raw recordings stay intact."""
import argparse
import bisect
import hashlib
import json
from pathlib import Path
import sys
import cv2
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from unified_dataset import alignment_plan


def camera_dir(meta, camera_root=None):
    """Recorded dataset_dir is a Jetson path; camera_root holds local copies named by its basename."""
    remote=Path(meta['camera']['dataset_dir'])
    return Path(camera_root)/remote.name if camera_root else remote


def prepare_episode(path, fps=30, smoke=False, camera_root=None):
    path=Path(path)
    joint=json.loads(path.read_text());meta=joint['metadata']
    if meta.get('status')!='motion_complete' or meta.get('fault'):
        raise ValueError(f'Incomplete episode: {path}')
    if not smoke and meta.get('source')!='task_kinesthetic_vendor_defaults':
        raise ValueError('Select task demonstrations, not earlier hardware checks; --smoke only for software validation')
    if not meta.get('finalization_complete') or not meta.get('camera_recorded'):
        raise ValueError('Camera finalization not complete')
    media=path.with_name(path.stem+'-media')
    if (media/'timestamps.json').exists():
        timing=json.loads((media/'timestamps.json').read_text());video=media/'wrist.mp4'
        manifest=json.loads((media/'manifest.json').read_text())
        linked=manifest.get('remote_dataset',{}).get('dataset_dir')
        if linked != meta['camera']['dataset_dir']:raise ValueError('Camera source association mismatch')
    else:
        media=camera_dir(meta,camera_root)
        timing=json.loads((media/'raw-camera-timestamps.json').read_text())
        videos=list((media/'videos').rglob('*.mp4'))
        if len(videos)!=1:raise ValueError('Expected one camera video')
        video=videos[0]
    plan=alignment_plan(joint['samples'],timing['frames'],fps)
    if len(plan)<2:raise ValueError('Too few aligned frames')
    # The action is the next demonstrated absolute joint target (radians).
    # This is a kinematic imitation label, not the passive teach-mode motor command.
    with_gripper=bool(meta.get('gripper_controlled'))
    stamps=[x['t_monotonic_s'] for x in joint['samples']]
    rows=[]
    for current,nxt in zip(plan,plan[1:]):
        state=list(current['q']);action=list(nxt['q'])
        if with_gripper:
            idx=max(0,bisect.bisect_right(stamps,current['time'])-1)
            sample=joint['samples'][idx]
            if not sample.get('gripper_feedback_valid') or sample.get('gripper_position') is None:
                raise ValueError('Gripper feedback missing; cannot use commanded value as measured state')
            measured=float(sample['gripper_position']);target=float(sample['gripper_target'])
            if not np.isfinite([measured,target]).all():raise ValueError('Invalid gripper data')
            state.append(measured);action.append(target)
        rows.append({'observation.state':np.array(state,np.float32),
                     'camera_index':current['camera_index'],
                     'action':np.array(action,np.float32),'task':meta.get('task','USB-C manipulation')})
    report={'source':str(path),'sha256':hashlib.sha256(path.read_bytes()).hexdigest(),
            'frames':len(rows),'fps':fps,'max_camera_offset_s':max(abs(x['residual']) for x in plan),
            'action':'q(t+1/fps), absolute six-joint target in rad; derived from kinesthetic demonstration',
            'smoke_only':smoke,'gripper_controlled':with_gripper,'gripper_action':'current commanded opening, 0 closed / 1 open' if with_gripper else None}
    report['video']=str(video);report['video_frames']=len(timing['frames'])
    return rows,report


def stream_images(video, total, episodes):
    """Decode a session video once, yielding (episode, rows-with-images) in order; memory holds one episode."""
    pending=sorted(episodes,key=lambda e:e[0][0]['camera_index']);need={}
    for i,(rows,_) in enumerate(pending):
        for r in rows:need.setdefault(r['camera_index'],[]).append((i,r))
    cap=cv2.VideoCapture(str(video));decoded=0;done=0;filled=[0]*len(pending)
    try:
        while True:
            ok,im=cap.read()
            if not ok:break
            for i,r in need.pop(decoded,[]):
                r['observation.images.wrist']=cv2.cvtColor(im,cv2.COLOR_BGR2RGB);filled[i]+=1
            decoded+=1
            while done<len(pending) and filled[done]==len(pending[done][0]):
                yield pending[done];done+=1
    finally:cap.release()
    if decoded!=total:raise ValueError(f'Video/timestamp frame count mismatch: {video}')
    if done!=len(pending):raise ValueError(f'Frames missing for {len(pending)-done} episodes in {video}')


def main():
    p=argparse.ArgumentParser();p.add_argument('episodes',type=Path,nargs='+');p.add_argument('--output',type=Path,required=True)
    p.add_argument('--repo-id',default='local/a1z_usb_task');p.add_argument('--smoke',action='store_true');p.add_argument('--fps',type=int,default=30)
    p.add_argument('--camera-root',type=Path,help='local copies of Jetson camera dataset directories')
    p.add_argument('--images',action='store_true',help='store PNG frames instead of H.264 video (much larger)')
    a=p.parse_args()
    if a.output.exists():raise ValueError('Output already exists; use a new dataset directory')
    from lerobot.datasets.lerobot_dataset import LeRobotDataset
    ds=None;reports=[];sessions={}
    for path in a.episodes:
        rows,report=prepare_episode(path,a.fps,a.smoke,a.camera_root)
        sessions.setdefault((report['video'],report['video_frames']),[]).append((rows,report))
    for (video,total),episodes in sessions.items():
      for rows,report in stream_images(video,total,episodes):
        for r in rows:r.pop('camera_index')
        if ds is None:
            h,w,c=rows[0]['observation.images.wrist'].shape
            dimension=len(rows[0]['action']);names=[f'j{i}' for i in range(1,7)]+(['gripper'] if dimension==7 else [])
            features={'observation.state':{'dtype':'float32','shape':(dimension,),'names':names},
                      'action':{'dtype':'float32','shape':(dimension,),'names':names},
                      'observation.images.wrist':{'dtype':'image' if a.images else 'video','shape':(h,w,c),'names':['height','width','channels']}}
            ds=LeRobotDataset.create(repo_id=a.repo_id,root=a.output,fps=a.fps,robot_type='a1z_single',features=features,
                                     use_videos=not a.images,image_writer_threads=4,vcodec='h264')
        if len(rows[0]['action'])!=dimension:raise ValueError('Cannot mix arm-only and gripper demonstrations')
        for row in rows:ds.add_frame(row)
        ds.save_episode();reports.append(report)
        for r in rows:r.pop('observation.images.wrist',None)  # release this episode's frames
        print('EPISODE',len(reports),report['frames'],Path(report['source']).name,flush=True)
    ds.finalize()
    # torchcodec on macOS often mismatches Homebrew ffmpeg; PyAV decodes the same files.
    fresh=LeRobotDataset(repo_id=a.repo_id,root=a.output,video_backend='pyav' if sys.platform=='darwin' else None)
    count=sum(x['frames'] for x in reports)
    assert len(fresh)==count
    for i in [0,len(fresh)-1]:
        row=fresh[i]
        assert tuple(row['action'].shape)==(dimension,) and tuple(row['observation.state'].shape)==(dimension,)
        assert np.isfinite(row['action'].numpy()).all()
    (a.output/'source-report.json').write_text(json.dumps({'episodes':reports,'total_frames':count,'reopened':True},indent=2))
    print(json.dumps({'dataset':str(a.output),'episodes':len(reports),'frames':count,'smoke_only':a.smoke}))
if __name__=='__main__':main()
