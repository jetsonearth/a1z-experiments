#!/usr/bin/env python3
"""Isolated RGB dataset capture. No robot/CAN imports or commands."""
import argparse
import json
import subprocess
import time
from pathlib import Path


def atomic_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + '.tmp')
    temporary.write_text(json.dumps(value, indent=2))
    temporary.replace(path)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--stop-file', type=Path, required=True)
    parser.add_argument('--ready-file', type=Path, required=True)
    parser.add_argument('--max-seconds', type=float, default=300)
    args = parser.parse_args()
    if not 1 <= args.max_seconds <= 300:
        parser.error('max-seconds must be between 1 and 300')
    if args.output.exists():
        parser.error('output must not already exist')
    rows, errors, checks = [], [], []
    ds = pipe = profile = None
    reason = 'error'
    last_status_write = 0.0
    metadata = {'dataset_dir': str(args.output), 'depth_video_saved': False,
                'action_defined': False,
                'timing_boundary': 'host receipt, not calibrated exposure; nominal video grid is 30 fps'}
    deadline = time.monotonic() + args.max_seconds
    try:
        import numpy as np
        import pyrealsense2 as rs
        from lerobot.datasets.lerobot_dataset import LeRobotDataset
        ds = LeRobotDataset.create(repo_id='local/a1z_camera_session', root=args.output, fps=30,
            robot_type='a1z_single_observation_only', features={
                'observation.images.wrist': {'dtype':'video','shape':(480,640,3),'names':['height','width','channels']},
                'observation.camera_host_monotonic_s': {'dtype':'float64','shape':(1,),'names':['host_receipt_s']},
            }, use_videos=True, streaming_encoding=False, image_writer_processes=0,
            image_writer_threads=2, vcodec='h264', encoder_threads=2)
        pipe = rs.pipeline()
        config = rs.config()
        config.enable_stream(rs.stream.color,640,480,rs.format.rgb8,30)
        config.enable_stream(rs.stream.depth,640,480,rs.format.z16,30)
        profile = pipe.start(config)
        intr = profile.get_stream(rs.stream.color).as_video_stream_profile().get_intrinsics()
        metadata.update(device=profile.get_device().get_info(rs.camera_info.name),
            depth_scale_m=profile.get_device().first_depth_sensor().get_depth_scale(),
            rgb_intrinsics={k:getattr(intr,k) for k in ['width','height','ppx','ppy','fx','fy','coeffs']},
            distortion_model=str(intr.model))
        while time.monotonic() < deadline:
            if args.stop_file.exists():
                reason = 'requested_stop'
                break
            frames = pipe.wait_for_frames(2000)
            host = time.monotonic()
            if host >= deadline:
                break
            rgb, depth = frames.get_color_frame(), frames.get_depth_frame()
            if not rgb or not depth:
                continue
            row = {'host_monotonic_s':host, 'rgb_device_timestamp_ms':rgb.get_timestamp(),
                   'rgb_frame_number':rgb.get_frame_number(),
                   'rgb_timestamp_domain':str(rgb.get_frame_timestamp_domain()),
                   'depth_device_timestamp_ms':depth.get_timestamp(),
                   'depth_frame_number':depth.get_frame_number(),
                   'depth_timestamp_domain':str(depth.get_frame_timestamp_domain())}
            ds.add_frame({'observation.images.wrist':np.asanyarray(rgb.get_data()).copy(),
                'observation.camera_host_monotonic_s':np.array([host],dtype=np.float64),
                'task':'A1Z kinesthetic demonstration camera observation'})
            rows.append(row)
            if len(rows) == 1 or host - last_status_write >= 0.5:
                atomic_json(args.ready_file, {'host_monotonic_s':host, 'state':'capturing',
                                             'dataset_dir':str(args.output)})
                last_status_write = host
        if reason != 'requested_stop':
            reason = 'max_seconds'
            errors.append('camera maximum duration reached before requested stop')
    except BaseException as exc:
        errors.append(f'capture: {type(exc).__name__}: {exc}')
    finally:
        if rows:
            try:
                atomic_json(args.ready_file, {'host_monotonic_s':rows[-1]['host_monotonic_s'],
                    'state':'stopped', 'dataset_dir':str(args.output)})
            except BaseException as exc:
                errors.append(f'status stop: {type(exc).__name__}: {exc}')
        if pipe is not None and profile is not None:
            try:
                pipe.stop()
            except BaseException as exc:
                errors.append(f'camera stop: {type(exc).__name__}: {exc}')
        gaps = [b['host_monotonic_s']-a['host_monotonic_s'] for a,b in zip(rows,rows[1:])]
        metadata.update(frames=len(rows), stop_reason=reason,
            receipt_hz=(len(gaps)/sum(gaps) if gaps and sum(gaps)>0 else 0),
            max_receipt_gap_s=max(gaps,default=0),
            skipped_rgb_frames=sum(max(0,b['rgb_frame_number']-a['rgb_frame_number']-1) for a,b in zip(rows,rows[1:])))
        try:
            atomic_json(args.output/'raw-camera-timestamps.json', {'metadata':metadata,'frames':rows})
        except BaseException as exc:
            errors.append(f'timestamps: {type(exc).__name__}: {exc}')
        if ds is not None:
            for name, operation in [('save_episode', ds.save_episode if rows else lambda:None),
                                    ('finalize',ds.finalize), ('stop_image_writer',ds.stop_image_writer)]:
                try:
                    operation()
                except BaseException as exc:
                    errors.append(f'{name}: {type(exc).__name__}: {exc}')
    try:
        videos = list(args.output.rglob('*.mp4'))
        if not videos:
            raise RuntimeError('No video produced')
        for video in videos:
            probe = subprocess.run(['ffprobe','-v','error','-select_streams','v:0','-count_frames',
                '-show_entries','stream=codec_name,width,height,nb_read_frames','-of','json',str(video)],
                capture_output=True,text=True,timeout=30)
            check = {'file':str(video.relative_to(args.output)), 'returncode':probe.returncode,
                     'stdout':probe.stdout,'stderr':probe.stderr}
            checks.append(check)
            if probe.returncode:
                raise RuntimeError(f'ffprobe failed: {probe.stderr}')
            check['decoded'] = json.loads(probe.stdout)
        decoded_frames = sum(int(s.get('nb_read_frames',0)) for c in checks for s in c['decoded'].get('streams',[]))
        if decoded_frames != len(rows) or not rows:
            raise RuntimeError(f'Decoded frame count {decoded_frames} != recorded {len(rows)}')
    except BaseException as exc:
        errors.append(f'verification: {type(exc).__name__}: {exc}')
    metadata.update(video_checks=checks, success=not errors, error='; '.join(errors) or None)
    atomic_json(args.output/'result.json',metadata)
    print(json.dumps(metadata),flush=True)
    return 1 if errors else 0


if __name__ == '__main__':
    raise SystemExit(main())
