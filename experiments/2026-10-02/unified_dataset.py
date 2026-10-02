"""Offline, observation-only alignment. Never connects to robot or camera hardware."""
import bisect
import json
import math
import subprocess
from pathlib import Path


def alignment_plan(samples, camera_rows, fps=30):
    """Pure alignment planning; no image buffers or third-party dependencies."""
    if len(samples) < 2 or not camera_rows:
        raise ValueError('Need at least two joint samples and one camera frame')
    jt = [float(s['t_monotonic_s']) for s in samples]
    ct = [float(s['host_monotonic_s']) for s in camera_rows]
    for name, values in [('joint',jt),('camera',ct)]:
        if not all(math.isfinite(v) for v in values) or any(b <= a for a,b in zip(values,values[1:])):
            raise ValueError(f'{name} timestamps must be finite and strictly increasing')
    if ct[-1] < jt[0] or ct[0] > jt[-1]:
        raise ValueError('No temporal overlap between joints and camera')
    if ct[0] > jt[0]+.05 or ct[-1] < jt[-1]-.05:
        raise ValueError('Camera does not cover the complete joint interval within 50 ms')
    if any(b-a > .100000001 for a,b in zip(jt,jt[1:])):
        raise ValueError('Joint bracketing gap exceeds 100 ms')
    for s in samples:
        for key in ['q','dq','effort']:
            if len(s[key]) != 6 or not all(math.isfinite(float(v)) for v in s[key]):
                raise ValueError(f'Invalid six-joint {key}')
    plan=[]
    for i in range(int(math.floor((jt[-1]-jt[0])*fps+1e-7))+1):
        t=jt[0]+i/fps
        ci=bisect.bisect_left(ct,t)
        choices=[k for k in [ci-1,ci] if 0 <= k < len(ct)]
        cam=min(choices,key=lambda k:abs(ct[k]-t))
        if abs(ct[cam]-t) > .050000001:
            raise ValueError('Camera alignment residual exceeds 50 ms')
        right=min(max(1,bisect.bisect_left(jt,t)),len(jt)-1)
        left=right-1
        alpha=(t-jt[left])/(jt[right]-jt[left])
        if not -1e-7 <= alpha <= 1+1e-7:
            raise ValueError('Joint extrapolation would be required')
        row={'time':t,'camera_index':cam,'camera_time':ct[cam],
             'joint_left':jt[left],'joint_right':jt[right], 'residual':ct[cam]-t}
        for key in ['q','dq','effort']:
            row[key]=[(1-alpha)*float(a)+alpha*float(b) for a,b in zip(samples[left][key],samples[right][key])]
        plan.append(row)
    return plan


def _write(path, value):
    path.write_text(json.dumps(value,indent=2))


def export_episode(joint_json_path, camera_dir, output_dir):
    joint_json_path, camera_dir, output_dir = map(lambda p:Path(p).resolve(),
                                                  [joint_json_path,camera_dir,output_dir])
    result={'dataset_dir':str(output_dir),'frames':0,'video_checks':[],'success':False,'validated':False,'error':None}
    ds=cap=None
    created=False
    try:
        if output_dir.exists():
            raise ValueError('Output directory already exists; refusing to overwrite')
        joint=json.loads(joint_json_path.read_text())
        camera_result=json.loads((camera_dir/'result.json').read_text())
        source=json.loads((camera_dir/'raw-camera-timestamps.json').read_text())
        linked=joint.get('metadata',{}).get('camera',{}).get('dataset_dir')
        if not linked or Path(linked).resolve() != camera_dir:
            raise ValueError('Joint episode does not reference this camera session')
        if camera_result.get('success') is not True:
            raise ValueError('Source camera capture did not pass verification')
        units=joint.get('metadata',{}).get('units',{})
        if any(units.get(k)!=v for k,v in [('q','rad'),('dq','rad/s'),('effort','Nm')]):
            raise ValueError('Original joint units must explicitly be rad, rad/s, Nm')
        plan=alignment_plan(joint['samples'],source['frames'])
        videos=list((camera_dir/'videos').rglob('*.mp4'))
        if len(videos)!=1:
            raise ValueError('Expected exactly one source RGB video')
        if camera_result.get('frames')!=len(source['frames']):
            raise ValueError('Source camera frame count and timestamp count differ')
        import numpy as np
        import cv2
        import pyarrow.parquet as pq
        from lerobot.datasets.lerobot_dataset import LeRobotDataset
        names=[f'j{i}' for i in range(1,7)]
        features={
            'observation.state':{'dtype':'float32','shape':(6,),'names':names},
            'observation.velocity':{'dtype':'float32','shape':(6,),'names':names},
            'observation.effort':{'dtype':'float32','shape':(6,),'names':names},
            'observation.images.wrist':{'dtype':'video','shape':(480,640,3),'names':['height','width','channels']},
        }
        scalar_keys={'observation.source_camera_host_s':'camera_time',
                     'observation.source_joint_left_s':'joint_left',
                     'observation.source_joint_right_s':'joint_right',
                     'observation.alignment_residual_s':'residual',
                     'observation.grid_host_s':'time'}
        for name in scalar_keys:
            features[name]={'dtype':'float64','shape':(1,),'names':[name.rsplit('.',1)[1]]}
        cap=cv2.VideoCapture(str(videos[0]))
        if not cap.isOpened():raise RuntimeError('Cannot open source camera video')
        ds=LeRobotDataset.create(repo_id='local/a1z_unified_observations',root=output_dir,fps=30,
            robot_type='a1z_single_observation_only',features=features,use_videos=True,
            streaming_encoding=False,image_writer_processes=0,image_writer_threads=2,
            vcodec='h264',encoder_threads=2)
        created=True
        decoded_index=-1
        image=None
        for item in plan:
            while decoded_index < item['camera_index']:
                ok,image=cap.read()
                if not ok:raise RuntimeError('Source video ended before selected frame')
                decoded_index+=1
            if image.shape!=(480,640,3):raise ValueError('Unexpected RGB dimensions')
            frame={'observation.state':np.asarray(item['q'],dtype=np.float32),
                   'observation.velocity':np.asarray(item['dq'],dtype=np.float32),
                   'observation.effort':np.asarray(item['effort'],dtype=np.float32),
                   'observation.images.wrist':cv2.cvtColor(image,cv2.COLOR_BGR2RGB),
                   'task':('SYNTHETIC SCHEMA TEST ONLY - not robot demonstration'
                           if joint.get('metadata',{}).get('test_only') else
                           'A1Z kinesthetic observation; no policy action defined')}
            for key,source_key in scalar_keys.items():
                frame[key]=np.asarray([item[source_key]],dtype=np.float64)
            ds.add_frame(frame)
            result['frames']+=1
        # Complete sequential source decoding validates the sidecar-to-video index mapping.
        while True:
            ok,_=cap.read()
            if not ok:break
            decoded_index+=1
        if decoded_index+1!=len(source['frames']):
            raise ValueError('Decoded source video count does not match camera timestamps')
        cap.release();cap=None
        ds.save_episode();ds.finalize();ds.stop_image_writer();ds=None
        unique=len({r['camera_index'] for r in plan})
        report={'fps':30,'frames':len(plan),'joint_source':str(joint_json_path),'camera_source':str(camera_dir),
            'joint_first_host_s':joint['samples'][0]['t_monotonic_s'],
            'joint_last_host_s':joint['samples'][-1]['t_monotonic_s'],
            'grid_first_host_s':plan[0]['time'],'grid_last_host_s':plan[-1]['time'],
            'max_camera_offset_s':max(abs(r['residual']) for r in plan),
            'max_joint_bracket_gap_s':max(r['joint_right']-r['joint_left'] for r in plan),
            'repeated_camera_frames':len(plan)-unique,'unused_camera_frames':len(source['frames'])-unique,
            'clock_boundary':'Same host monotonic receipt clock, not calibrated camera exposure synchronization',
            'method':'30 Hz grid from first joint timestamp; nearest RGB <=50 ms; linear joint interpolation <=100 ms brackets',
            'action_defined':False,'units':units,'depth_video_saved':False}
        _write(output_dir/'alignment-report.json',report)
        row_count=sum(pq.ParquetFile(p).metadata.num_rows for p in (output_dir/'data').rglob('*.parquet'))
        if row_count!=len(plan):raise RuntimeError('Parquet row count mismatch')
        reopened=LeRobotDataset(repo_id='local/a1z_unified_observations',root=output_dir)
        if len(reopened)!=len(plan):raise RuntimeError('Reopened dataset length mismatch')
        for index in sorted({0,len(plan)//2,len(plan)-1}):
            item=reopened[index]
            for key in ['observation.state','observation.velocity','observation.effort']:
                if tuple(item[key].shape)!=(6,):raise RuntimeError(f'Reopen shape mismatch: {key}')
            if tuple(item['observation.images.wrist'].shape)!=(3,480,640):
                raise RuntimeError('Reopened RGB shape mismatch')
            if not np.allclose(item['observation.state'].numpy(),plan[index]['q'],rtol=1e-5,atol=1e-6):
                raise RuntimeError('Reopened joint values mismatch')
        for video in (output_dir/'videos').rglob('*.mp4'):
            check=subprocess.run(['ffprobe','-v','error','-select_streams','v:0','-count_frames',
                '-show_entries','stream=codec_name,width,height,nb_read_frames','-of','json',str(video)],
                capture_output=True,text=True,timeout=60,check=True)
            result['video_checks'].append({'file':str(video.relative_to(output_dir)),**json.loads(check.stdout)})
        frames_decoded=sum(int(s['nb_read_frames']) for c in result['video_checks'] for s in c['streams'])
        if frames_decoded!=len(plan):raise RuntimeError('Output video frame count mismatch')
        playback_samples=[{'t_monotonic_s':r['time'],
            **{key:np.asarray(r[key],dtype=np.float32).tolist() for key in ['q','dq','effort']}} for r in plan]
        _write(output_dir/'playback.json',{'metadata':{'source':'lerobot_derived_view','fps':30,'units':units},
                                         'samples':playback_samples})
        _write(output_dir/'frame-timestamps.json',{'metadata':{'source':'lerobot_derived_grid','fps':30},
            'frames':[{'host_monotonic_s':r['time'],'source_camera_host_monotonic_s':r['camera_time'],
                       'source_camera_index':r['camera_index'],'alignment_residual_s':r['residual'],
                       'source_joint_left_s':r['joint_left'],'source_joint_right_s':r['joint_right']} for r in plan]})
        result.update(success=True,validated=True,alignment_report='alignment-report.json',
            parquet_rows=row_count,action_defined=False,timestamps_file='frame-timestamps.json',playback_file='playback.json')
    except Exception as exc:
        result['error']=f'{type(exc).__name__}: {exc}'
    finally:
        if cap is not None:cap.release()
        if ds is not None:
            for operation in [ds.finalize,ds.stop_image_writer]:
                try:operation()
                except Exception as exc:result['error']=(result['error'] or '')+f'; cleanup: {exc}'
        if created:
            _write(output_dir/'result.json',result)
    return result


if __name__ == '__main__':
    import argparse
    parser=argparse.ArgumentParser(description='Optional offline observation-format experiment; never controls hardware')
    parser.add_argument('--joints','--joint-json',dest='joint_json',required=True)
    parser.add_argument('--camera','--camera-dir',dest='camera_dir',required=True)
    parser.add_argument('--output','--output-dir',dest='output_dir',required=True)
    args=parser.parse_args()
    result=export_episode(args.joint_json,args.camera_dir,args.output_dir)
    print(json.dumps(result,indent=2))
    raise SystemExit(0 if result['success'] and result['validated'] else 1)
