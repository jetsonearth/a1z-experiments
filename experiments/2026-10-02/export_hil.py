#!/usr/bin/env python3
"""HG-DAgger data export: operator-controlled stretches of --hil trials -> LeRobot episodes, then D = D_BC ∪ D_1 ∪ ...

Faithful to Kelly et al. 2019 (Algorithm 1): only states visited while the expert holds uninterrupted control are
labelled, D_i = {(o_t, pi_H(x_t)) | g(x_t)=1}; D starts from the behavioral-cloning set and only ever grows; the
next novice is trained on the aggregate. Policy-controlled stretches are never labelled (they stay in the raw
session for evaluation and later RL).

Labels use the demonstrations' exact convention (pipeline/export_training.py): 30 Hz grid from the same
alignment_plan, state = measured joints + measured gripper opening, action = next aligned joint position +
commanded gripper opening. Each intervention becomes its own episode so ACT's action chunks never reach past
the handback into policy-driven motion (LeRobot pads and masks the chunk at the episode end).

One deliberate addition: leading frames where the arm has not yet moved since the takeover are trimmed
(--idle-rad). In the paper the expert takes over by acting (turning the wheel); with kinesthetic takeover the
first frames are the operator's reaction time, which would label the failure state with "stay still".
Raw files are never modified; set --idle-rad 0 to keep every frame.

Usage:
  export_hil.py hil-runs/<session> [...] --output <new dir>                   # D_i only
  export_hil.py hil-runs/<session> [...] --output <new dir> --aggregate <bc or previous aggregate dataset> --aggregate-output <new dir>
"""
import argparse
import json
from pathlib import Path
import sys

import numpy as np

ROOT=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT));sys.path.insert(0,str(ROOT/'pipeline'))
from unified_dataset import alignment_plan
from export_training import stream_images

TASK='USB-C manipulation demonstration'  # the demonstrations' task string (ACT ignores it; kept identical for merging)


def load_flight(session):
    rows=[]
    for line in (session/'flight.jsonl').read_text().splitlines():
        r=json.loads(line)
        if 'q' in r:rows.append(r)
    return rows


def segment_samples(flight,t0,t1):
    """Flight-recorder rows (50 Hz) inside [t0,t1] in the demonstration sample format."""
    out=[]
    for r in flight:
        if t0<=r['t']<=t1:
            if 'grip_cmd' not in r:raise ValueError('flight.jsonl has no commanded gripper value (recorded before the HIL update)')
            g=r['grip'];valid=g is not None and np.isfinite(g)
            out.append({'t_monotonic_s':r['t'],'q':r['q'],'dq':r['dq'],'effort':r['eff'],'mode':r['mode'],
                        'gripper_position':g if valid else None,'gripper_target':r['grip_cmd'],'gripper_feedback_valid':bool(valid)})
    return out


def label_rows(samples,camera_rows,fps):
    """Same state/action construction as export_training.prepare_episode."""
    plan=alignment_plan(samples,camera_rows,fps)
    stamps=[s['t_monotonic_s'] for s in samples];rows=[]
    import bisect
    for cur,nxt in zip(plan,plan[1:]):
        s=samples[max(0,bisect.bisect_right(stamps,cur['time'])-1)]
        if not s['gripper_feedback_valid']:raise ValueError('Gripper feedback missing; cannot use commanded value as measured state')
        rows.append({'observation.state':np.array(list(cur['q'])+[s['gripper_position']],np.float32),'camera_index':cur['camera_index'],
                     'action':np.array(list(nxt['q'])+[s['gripper_target']],np.float32),'task':TASK})
    return rows,plan


def trim_idle(rows,idle_rad,max_s,fps):
    """Drop leading frames until any joint has moved idle_rad from the takeover pose (at most max_s seconds)."""
    if idle_rad<=0 or not rows:return rows,0
    q0=rows[0]['observation.state'][:6];limit=int(max_s*fps)
    for i,r in enumerate(rows[:limit]):
        if np.abs(r['observation.state'][:6]-q0).max()>=idle_rad:return rows[i:],i
    return rows[limit:],min(limit,len(rows))


def collect(sessions,a):
    episodes=[];report={'kept':[],'skipped':[],'policy_s':0.0,'human_s':0.0,'trials':0,'interventions':0}
    for session in sessions:
        session=Path(session);flight=load_flight(session)
        for f in sorted(session.glob('trial-*.json')):
            t=json.loads(f.read_text())
            if not t.get('hil'):continue
            report['trials']+=1
            for x in t['segments']:report[('human' if x['controller']=='human' else 'policy')+'_s']+=x['t1']-x['t0']
            human=[(i,x) for i,x in enumerate(t['segments']) if x['controller']=='human']
            report['interventions']+=len(human)
            if not t.get('hil_keep',True):
                report['skipped'].append({'trial':f'{session.name}/{f.stem}','reason':'operator discarded','interventions':len(human)});continue
            if not t.get('success') and not a.include_unverified:
                # no USB enumeration: the correction may be incomplete, and anything after it (waiting, pulling the plug out)
                # would become labels. Only USB-verified corrections are exported unless asked.
                report['skipped'].append({'trial':f'{session.name}/{f.stem}','reason':f"no USB signal ({t['end_reason']})",'interventions':len(human)});continue
            if not human:continue
            timing=json.loads(f.with_suffix('.timestamps.json').read_text());video=f.with_suffix('.mp4')
            if timing.get('encoder_error'):raise ValueError(f'{f}: video encoder error {timing["encoder_error"]}')
            for i,x in human:
                key=f'{session.name}/{f.stem}/segment-{i}'
                samples=segment_samples(flight,x['t0'],x['t1'])
                if len(samples)<2:report['skipped'].append({'segment':key,'reason':'too short'});continue
                floating=sum(s['mode']==0 for s in samples)/len(samples)
                if floating<0.95:raise ValueError(f'{key}: arm was engaged during {1-floating:.0%} of an operator segment')
                try:rows,plan=label_rows(samples,timing['frames'],a.fps)
                except ValueError as e:  # e.g. a camera stall >50 ms inside the segment: reported, never silently relabelled
                    report['skipped'].append({'segment':key,'reason':str(e)});continue
                rows,trimmed=trim_idle(rows,a.idle_rad,a.idle_max_s,a.fps)
                if len(rows)<a.min_frames:report['skipped'].append({'segment':key,'reason':f'{len(rows)} frames after trimming {trimmed}'});continue
                info={'segment':key,'epoch':t.get('epoch'),'checkpoint':t.get('checkpoint'),'t0':x['t0'],'t1':x['t1'],'frames':len(rows),
                      'trimmed_idle_frames':trimmed,'max_camera_offset_s':max(abs(p['residual']) for p in plan),
                      'takeover_camera_index':plan[0]['camera_index'],  # observation at the takeover: input for an ensemble doubt d_N(o_t) later
                      'trial_usb_inserted':t['success'],'trial_end_reason':t['end_reason'],'video':str(video)}
                episodes.append(((video,len(timing['frames'])),rows,info))
    return episodes,report


def main():
    p=argparse.ArgumentParser();p.add_argument('sessions',nargs='+',type=Path);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--repo-id',default='local/a1z_usb_hil');p.add_argument('--fps',type=int,default=30)
    p.add_argument('--idle-rad',type=float,default=0.003,help='trim leading frames until a joint moves this far (0 = keep all)')
    p.add_argument('--idle-max-s',type=float,default=1.5);p.add_argument('--min-frames',type=int,default=10)
    p.add_argument('--aggregate',type=Path,help='existing dataset D (the BC set or the previous aggregate) to merge with')
    p.add_argument('--aggregate-output',type=Path);p.add_argument('--aggregate-repo-id',default='local/a1z_usb_aggregate')
    p.add_argument('--dry-run',action='store_true',help='label and report only; write nothing')
    p.add_argument('--include-unverified',action='store_true',help='also export trials that ended without a USB signal')
    a=p.parse_args()
    if a.aggregate and not a.aggregate_output:p.error('--aggregate needs --aggregate-output')
    for d in [a.output,a.aggregate_output]:
        if d and d.exists():p.error(f'{d} exists; use a new directory')
    episodes,report=collect(a.sessions,a)
    frames=sum(len(r) for _,r,_ in episodes)
    print(json.dumps({'trials':report['trials'],'interventions':report['interventions'],'segments_kept':len(episodes),'frames':frames,
                      'human_s':round(report['human_s'],1),'policy_s':round(report['policy_s'],1),'skipped':report['skipped']},ensure_ascii=False,indent=1))
    if a.dry_run or not episodes:return
    from lerobot.datasets.lerobot_dataset import LeRobotDataset
    names=[f'j{i}' for i in range(1,7)]+['gripper']
    features={'observation.state':{'dtype':'float32','shape':(7,),'names':names},'action':{'dtype':'float32','shape':(7,),'names':names},
              'observation.images.wrist':{'dtype':'video','shape':(480,640,3),'names':['height','width','channels']}}
    ds=LeRobotDataset.create(repo_id=a.repo_id,root=a.output,fps=a.fps,robot_type='a1z_single',features=features,
                             use_videos=True,image_writer_threads=4,vcodec='h264')
    by_video={}
    for v,rows,info in episodes:by_video.setdefault(v,[]).append((rows,info))
    kept=[]
    for (video,total),eps in by_video.items():
        for rows,info in stream_images(video,total,eps):
            for r in rows:r.pop('camera_index')
            for r in rows:ds.add_frame(r)
            ds.save_episode();kept.append(info)
            for r in rows:r.pop('observation.images.wrist',None)
            print('EPISODE',len(kept),info['frames'],info['segment'],flush=True)
    ds.finalize()
    fresh=LeRobotDataset(repo_id=a.repo_id,root=a.output,video_backend='pyav' if sys.platform=='darwin' else None)
    assert len(fresh)==frames,(len(fresh),frames)
    (a.output/'source-report.json').write_text(json.dumps({'kind':'hg-dagger intervention labels','episodes':kept,'total_frames':frames,
        'skipped':report['skipped'],'policy_s':report['policy_s'],'human_s':report['human_s'],'trials':report['trials'],
        'idle_rad':a.idle_rad,'label_convention':'identical to pipeline/export_training.py','reopened':True},indent=1))
    if a.aggregate:
        from lerobot.datasets.dataset_tools import merge_datasets
        base=LeRobotDataset(repo_id='local/base',root=a.aggregate,video_backend='pyav' if sys.platform=='darwin' else None)
        merged=merge_datasets([base,fresh],output_repo_id=a.aggregate_repo_id,output_dir=a.aggregate_output)
        assert len(merged)==len(base)+frames
        prev=json.loads((a.aggregate/'source-report.json').read_text()) if (a.aggregate/'source-report.json').exists() else {}
        (a.aggregate_output/'source-report.json').write_text(json.dumps({'kind':'hg-dagger aggregate D','components':
            prev.get('components',[{'dataset':str(a.aggregate),'episodes':base.meta.total_episodes,'frames':len(base)}])+
            [{'dataset':str(a.output),'episodes':len(kept),'frames':frames}],'total_frames':len(merged),
            'episodes':prev.get('episodes',[])+kept},indent=1))
        print('AGGREGATE',a.aggregate_output,merged.meta.total_episodes,'episodes',len(merged),'frames')

if __name__=='__main__':main()
