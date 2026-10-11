#!/usr/bin/env python3
"""Stream camera RGB directly to video; no LeRobot or stop-time image encoding."""
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


def parse_args(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--stop-file', type=Path, required=True)
    parser.add_argument('--ready-file', type=Path, required=True)
    parser.add_argument('--max-seconds', type=float, default=None)
    args = parser.parse_args(argv)
    if args.max_seconds is not None and not args.max_seconds > 0:
        parser.error('max-seconds must be positive when supplied')
    if args.output.exists(): parser.error('output must not already exist')
    return args


class StreamingVideo:
    def __init__(self, output, width=640, height=480, fps=30):
        self.path=Path(output);self.path.parent.mkdir(parents=True,exist_ok=True)
        self.log=self.path.with_suffix('.encoder.log').open('wb')
        self.process=subprocess.Popen(['ffmpeg','-nostdin','-hide_banner','-loglevel','error',
            '-f','rawvideo','-pixel_format','rgb24','-video_size',f'{width}x{height}',
            '-framerate',str(fps),'-i','pipe:0','-an','-c:v','libx264','-preset','ultrafast',
            '-tune','zerolatency','-crf','23','-pix_fmt','yuv420p','-threads','2',
            '-movflags','+frag_keyframe+empty_moov+default_base_moof','-g',str(fps),str(self.path)],
            stdin=subprocess.PIPE,stdout=subprocess.DEVNULL,stderr=self.log)

    def write(self, frame):
        self.process.stdin.write(frame)

    def close(self):
        # Only flush already-encoded packets; never encode accumulated PNGs here.
        self.process.stdin.close()
        try:
            code=self.process.wait(timeout=20)
        except subprocess.TimeoutExpired:
            self.process.terminate()
            try:self.process.wait(timeout=3)
            except subprocess.TimeoutExpired:self.process.kill();self.process.wait(timeout=3)
            raise RuntimeError('Streaming encoder did not close; partial fragmented video retained')
        finally:self.log.close()
        if code:raise RuntimeError(f'Streaming encoder exited {code}')


class Preview:
    """Best-effort live preview: a side thread writes a small JPEG to /dev/shm. Never blocks or fails capture."""
    PATH=Path('/dev/shm/a1z-preview.jpg')

    def __init__(self, every=3):
        import threading
        self.every=every;self.count=0;self.frame=None;self.lock=threading.Lock();self.wake=threading.Event()
        self.alive=True;self.thread=threading.Thread(target=self.run,daemon=True);self.thread.start()

    def offer(self, rgb):
        self.count+=1
        if not self.alive or self.count%self.every:return
        with self.lock:self.frame=rgb.copy()  # librealsense reuses the buffer after the frame is released
        self.wake.set()

    def run(self):
        try:
            import cv2
            while self.alive:
                if not self.wake.wait(1):continue
                self.wake.clear()
                with self.lock:frame,self.frame=self.frame,None
                if frame is None:continue
                small=cv2.cvtColor(cv2.resize(frame,(320,240),interpolation=cv2.INTER_AREA),cv2.COLOR_RGB2BGR)
                ok,jpg=cv2.imencode('.jpg',small,[cv2.IMWRITE_JPEG_QUALITY,70])
                if ok:
                    tmp=self.PATH.with_suffix('.tmp');tmp.write_bytes(jpg.tobytes());tmp.replace(self.PATH)
        except BaseException:self.alive=False

    def close(self):
        self.alive=False;self.wake.set()


def probe_video(video, root, expected):
    # Count compressed packets, not full image decoding: bounded finalization work.
    p=subprocess.run(['ffprobe','-v','error','-select_streams','v:0','-count_packets',
        '-show_entries','stream=codec_name,width,height,nb_read_packets','-of','json',str(video)],
        capture_output=True,text=True,timeout=20)
    check={'file':str(Path(video).relative_to(root)),'returncode':p.returncode,'stdout':p.stdout,'stderr':p.stderr}
    if p.returncode:raise RuntimeError(p.stderr)
    check['decoded']=json.loads(p.stdout)
    count=sum(int(s.get('nb_read_packets',0)) for s in check['decoded'].get('streams',[]))
    check['verification']='H264 packet count; encoder uses zerolatency without B frames'
    if count!=expected or not expected:raise RuntimeError(f'Video packet count {count} != captured {expected}')
    return check


def main():
    args=parse_args();args.output.mkdir(parents=True)
    rows=[];errors=[];checks=[];pipe=profile=video=None;reason='error';last_status_write=0
    metadata={'dataset_dir':str(args.output),'source':'streaming_rgb','depth_video_saved':False,
        'action_defined':False,'timing_boundary':'host receipt, not calibrated exposure; nominal video grid is 30 fps'}
    deadline=time.monotonic()+args.max_seconds if args.max_seconds is not None else float('inf')
    journal=(args.output/'raw-camera-timestamps.jsonl').open('a',buffering=1);preview=None
    try:
        try:preview=Preview()
        except BaseException:preview=None
        import numpy as np
        import pyrealsense2 as rs
        video=StreamingVideo(args.output/'videos'/'wrist.mp4')
        from wrist_camera import enable_wrist
        pipe=rs.pipeline();config=rs.config();enable_wrist(rs,config)
        config.enable_stream(rs.stream.color,640,480,rs.format.rgb8,30)
        config.enable_stream(rs.stream.depth,640,480,rs.format.z16,30)
        profile=pipe.start(config)
        intr=profile.get_stream(rs.stream.color).as_video_stream_profile().get_intrinsics()
        metadata.update(device=profile.get_device().get_info(rs.camera_info.name),
            depth_scale_m=profile.get_device().first_depth_sensor().get_depth_scale(),
            rgb_intrinsics={k:getattr(intr,k) for k in ['width','height','ppx','ppy','fx','fy','coeffs']},distortion_model=str(intr.model))
        while time.monotonic()<deadline:
            if args.stop_file.exists():reason='requested_stop';break
            frames=pipe.wait_for_frames(2000);host=time.monotonic()
            rgb,depth=frames.get_color_frame(),frames.get_depth_frame()
            if not rgb or not depth:continue
            row={'host_monotonic_s':host,'rgb_device_timestamp_ms':rgb.get_timestamp(),
                'rgb_frame_number':rgb.get_frame_number(),'rgb_timestamp_domain':str(rgb.get_frame_timestamp_domain()),
                'depth_device_timestamp_ms':depth.get_timestamp(),'depth_frame_number':depth.get_frame_number(),
                'depth_timestamp_domain':str(depth.get_frame_timestamp_domain())}
            image=np.asanyarray(rgb.get_data());video.write(image.tobytes())
            if preview is not None:preview.offer(image)
            rows.append(row);journal.write(json.dumps(row)+'\n')
            if len(rows)==1 or host-last_status_write>=.5:
                atomic_json(args.ready_file,{'host_monotonic_s':host,'state':'capturing','dataset_dir':str(args.output),'frames':len(rows)})
                last_status_write=host
        if reason!='requested_stop':reason='max_seconds';errors.append('camera maximum duration reached before requested stop')
    except BaseException as exc:errors.append(f'capture: {type(exc).__name__}: {exc}')
    finally:
        if preview is not None:preview.close()
        journal.close()
        atomic_json(args.ready_file,{'host_monotonic_s':time.monotonic(),'state':'stopped','dataset_dir':str(args.output)})
        if pipe is not None and profile is not None:
            try:pipe.stop()
            except BaseException as exc:errors.append(f'camera stop: {exc}')
        gaps=[b['host_monotonic_s']-a['host_monotonic_s'] for a,b in zip(rows,rows[1:])]
        metadata.update(frames=len(rows),stop_reason=reason,receipt_hz=len(gaps)/sum(gaps) if gaps and sum(gaps)>0 else 0,
            max_receipt_gap_s=max(gaps,default=0),skipped_rgb_frames=sum(max(0,b['rgb_frame_number']-a['rgb_frame_number']-1) for a,b in zip(rows,rows[1:])))
        atomic_json(args.output/'raw-camera-timestamps.json',{'metadata':metadata,'frames':rows})
        if video is not None:
            try:video.close();checks.append(probe_video(video.path,args.output,len(rows)))
            except BaseException as exc:errors.append(f'video finalization: {type(exc).__name__}: {exc}')
    metadata.update(video_checks=checks,success=not errors,error='; '.join(errors) or None)
    atomic_json(args.output/'result.json',metadata);print(json.dumps(metadata),flush=True)
    return 1 if errors else 0

if __name__=='__main__':raise SystemExit(main())
