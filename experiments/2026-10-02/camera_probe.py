#!/usr/bin/env python3
"""Camera-only D405 probe; never imports or commands the arm.
Preserve native RGB/depth in RealSense recording and receipt/device timestamps.
Receipt timing is not a calibrated exposure-time synchronization guarantee.
"""
import argparse
import json
import time
from pathlib import Path


def main():
    import cv2
    import numpy as np
    import pyrealsense2 as rs
    parser = argparse.ArgumentParser()
    parser.add_argument('--seconds', type=float, default=4)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--native-recording', action='store_true')
    args = parser.parse_args()
    if not 1 <= args.seconds <= 15:
        parser.error('probe duration must be 1–15 seconds')
    args.output.mkdir(parents=True, exist_ok=False)
    from wrist_camera import enable_wrist
    pipe = rs.pipeline()
    config = rs.config()
    enable_wrist(rs, config)
    config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
    config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
    if args.native_recording:
        config.enable_record_to_file(str(args.output / 'rgb-depth.db3'))
    rows, profile, error = [], None, None
    try:
        profile = pipe.start(config)
        intrinsics = profile.get_stream(rs.stream.color).as_video_stream_profile().get_intrinsics()
        dev = profile.get_device()
        meta = {'device': dev.get_info(rs.camera_info.name),
                'depth_scale_m': dev.first_depth_sensor().get_depth_scale(),
                'rgb_intrinsics': {k: getattr(intrinsics, k) for k in ['width','height','ppx','ppy','fx','fy','coeffs']},
                'distortion_model': str(intrinsics.model),
                'host_clock': 'time.monotonic; same Jetson clock as teach.py',
                'timing_boundary': 'host receipt after wait_for_frames, not exposure; device clock retained separately'}
        until = time.monotonic() + args.seconds
        while time.monotonic() < until:
            frames = pipe.wait_for_frames(2000)
            received = time.monotonic()
            color, depth = frames.get_color_frame(), frames.get_depth_frame()
            if not color or not depth:
                continue
            row = {'host_monotonic_s': received}
            for name, frame in [('rgb',color),('depth',depth)]:
                row[name] = {'frame_number': frame.get_frame_number(), 'device_timestamp_ms': frame.get_timestamp(),
                             'timestamp_domain': str(frame.get_frame_timestamp_domain())}
            rows.append(row)
            # Last saved still is useful for checking actual wrist-camera visibility.
            last_rgb = np.asanyarray(color.get_data()).copy()
            last_depth = np.asanyarray(depth.get_data()).copy()
        if rows:
            if not cv2.imwrite(str(args.output / 'preview.jpg'),last_rgb):
                raise RuntimeError('Preview write failed')
            meta['valid_depth_fraction'] = float(np.count_nonzero(last_depth) / last_depth.size)
            meta['rgb_mean'],meta['rgb_std'] = float(last_rgb.mean()),float(last_rgb.std())
    except Exception as exc:
        error = repr(exc)
        raise
    finally:
        if 'last_rgb' in locals():
            cv2.imwrite(str(args.output / 'preview.jpg'), last_rgb)
        if profile is not None:
            pipe.stop()
        meta = locals().get('meta', {})
        gaps = [b['host_monotonic_s']-a['host_monotonic_s'] for a,b in zip(rows,rows[1:])]
        meta.update(frames=len(rows), error=error,
                    duration_s=sum(gaps), receipt_hz=(len(gaps)/sum(gaps) if gaps else 0),
                    max_receipt_gap_s=max(gaps,default=0),
                    skipped_rgb_frames=sum(max(0,b['rgb']['frame_number']-a['rgb']['frame_number']-1) for a,b in zip(rows,rows[1:])),
                    recording_bytes=(args.output/'rgb-depth.db3').stat().st_size if (args.output/'rgb-depth.db3').exists() else 0)
        (args.output/'timestamps.json').write_text(json.dumps({'metadata':meta,'frames':rows},indent=2))
        print(json.dumps(meta),flush=True)


if __name__ == '__main__':
    main()
