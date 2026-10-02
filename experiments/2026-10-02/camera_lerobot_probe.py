#!/usr/bin/env python3
"""Camera-only recording through installed LeRobot 0.4.4. No arm imports."""
import argparse
import json
import time
from pathlib import Path


def main():
    import numpy as np
    import pyrealsense2 as rs
    from lerobot.datasets.lerobot_dataset import LeRobotDataset
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--seconds', type=float, default=4)
    args = parser.parse_args()
    if not 1 <= args.seconds <= 15:
        parser.error('probe duration must be 1–15 seconds')
    ds = LeRobotDataset.create(repo_id='local/a1z_camera_probe', root=args.output, fps=30,
        robot_type='a1z_single_observation_only', features={
            'observation.images.wrist': {'dtype':'video','shape':(480,640,3),'names':['height','width','channels']},
            'observation.camera_host_monotonic_s': {'dtype':'float64','shape':(1,),'names':['host_receipt_s']},
        },use_videos=True,streaming_encoding=False,image_writer_processes=0,image_writer_threads=2,
        vcodec='h264',encoder_threads=2)
    pipe=rs.pipeline();config=rs.config()
    config.enable_stream(rs.stream.color,640,480,rs.format.rgb8,30)
    config.enable_stream(rs.stream.depth,640,480,rs.format.z16,30)
    rows=[];profile=None;error=None
    try:
        profile=pipe.start(config)
        intr=profile.get_stream(rs.stream.color).as_video_stream_profile().get_intrinsics()
        metadata={'device':profile.get_device().get_info(rs.camera_info.name),
                  'depth_scale_m':profile.get_device().first_depth_sensor().get_depth_scale(),
                  'rgb_intrinsics':{k:getattr(intr,k) for k in ['width','height','ppx','ppy','fx','fy','coeffs']},
                  'distortion_model':str(intr.model),'depth_video_saved':False,
                  'action_defined':False,'timing_boundary':'host receipt, not calibrated exposure; nominal video grid is 30 fps'}
        deadline=time.monotonic()+args.seconds
        while time.monotonic()<deadline:
            frames=pipe.wait_for_frames(2000);host=time.monotonic();rgb=frames.get_color_frame();depth=frames.get_depth_frame()
            if not rgb or not depth:continue
            rows.append({'host_monotonic_s':host,'rgb_device_timestamp_ms':rgb.get_timestamp(),
                         'rgb_frame_number':rgb.get_frame_number(),'rgb_timestamp_domain':str(rgb.get_frame_timestamp_domain()),
                         'depth_device_timestamp_ms':depth.get_timestamp(),'depth_frame_number':depth.get_frame_number()})
            ds.add_frame({'observation.images.wrist':np.asanyarray(rgb.get_data()).copy(),
                          'observation.camera_host_monotonic_s':np.array([host],dtype=np.float64),
                          'task':'D405 camera-only acquisition check'})
    except Exception as exc:
        error=repr(exc)
        raise
    finally:
        if profile is not None:pipe.stop()
        metadata=locals().get('metadata',{})
        gaps=np.diff([r['host_monotonic_s'] for r in rows])
        metadata.update(frames=len(rows),error=error,receipt_hz=float(1/gaps.mean()) if len(gaps) else 0,
                        max_receipt_gap_s=float(gaps.max()) if len(gaps) else 0,
                        skipped_rgb_frames=sum(max(0,b['rgb_frame_number']-a['rgb_frame_number']-1) for a,b in zip(rows,rows[1:])))
        (args.output/'raw-camera-timestamps.json').write_text(json.dumps({'metadata':metadata,'frames':rows},indent=2))
        try:
            if rows:ds.save_episode()
            ds.finalize()
        finally:ds.stop_image_writer()
    # Decoder verification uses installed FFmpeg tooling, independent of the writer.
    import subprocess
    files=list(args.output.rglob('*.mp4'))
    if not files:raise RuntimeError('No video produced')
    checks=[]
    for p in files:
        result=subprocess.run(['ffprobe','-v','error','-select_streams','v:0','-count_frames','-show_entries',
              'stream=codec_name,width,height,nb_read_frames','-of','json',str(p)],check=True,capture_output=True,text=True)
        checks.append({'file':str(p.relative_to(args.output)),**json.loads(result.stdout)})
    metadata['video_checks']=checks
    (args.output/'probe-result.json').write_text(json.dumps(metadata,indent=2))
    print(json.dumps(metadata),flush=True)


if __name__=='__main__':main()
