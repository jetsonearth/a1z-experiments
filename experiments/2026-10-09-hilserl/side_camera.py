#!/usr/bin/env python3
"""Fixed side camera (RealSense D435 or D455), colour only, opened by model name or serial so it can never be
confused with the wrist D405. Latest frame kept by a background thread, like the wrist camera in a1z_robot.py.
Whoever owns the camera (actor.py, or side_preview.py --headless started from the console's Cameras page) also writes a
JPEG to /dev/shm/a1z_side.jpg at 5 Hz, the same way a1z_robot.py publishes the wrist view for the console.

  python side_camera.py [out.jpg]      # one frame, for checking the mount angle
"""
import json
import os
import threading
import time

import numpy as np

WRIST_MODEL='D405'
LIVE_JPEG='/dev/shm/a1z_side.jpg'      # read by ../2026-10-02/console/server.py
LIVE_INFO='/dev/shm/a1z_side.json'
CAMERAS=os.path.join(os.path.dirname(os.path.abspath(__file__)),'cameras.json')   # shared with common.CAMERAS / the console
# Fixed exposure (2026-10-10): with auto exposure the white fin-ray finger entering the crop re-exposed the whole image,
# and the reward classifier trained on one session's brightness read 0.03 on a real insertion in the next. The first
# start lets auto exposure settle, then freezes exposure, gain and white balance and stores them in cameras.json
# ('side_exposure'); every later start applies the stored values. Delete that key to re-lock under new lighting.
LOCK_KEYS=('exposure','gain','white_balance')


def find(model=None,serial=None):
    import pyrealsense2 as rs
    devs=[(d.get_info(rs.camera_info.name),d.get_info(rs.camera_info.serial_number)) for d in rs.context().query_devices()]
    for name,sn in devs:
        if serial and sn==serial:return name,sn
        if not serial and WRIST_MODEL not in name and (model is None or model in name):return name,sn
    raise RuntimeError(f'side camera not found (model={model}, serial={serial}); connected: {devs}')


class SideCamera:
    def __init__(self,model=None,serial=None,width=640,height=480,fps=30,publish=True,lock_exposure=True):
        self.model=model;self.serial=serial;self.w=width;self.h=height;self.fps=fps;self.publish=publish;self._pub_t=0.0
        self.lock_exposure=lock_exposure;self.exposure=None;self._rec=None;self._rec_lock=threading.Lock()
        self._frame=None;self._t=0.0;self.lock=threading.Lock();self._stop=threading.Event();self._pipe=None;self.name=None

    def start(self):
        import pyrealsense2 as rs
        self.name,self.serial=find(self.model,self.serial)
        self._pipe=rs.pipeline();cfg=rs.config();cfg.enable_device(self.serial)
        cfg.enable_stream(rs.stream.color,self.w,self.h,rs.format.rgb8,self.fps)
        prof=self._pipe.start(cfg);self._pipe.wait_for_frames(3000)
        if self.lock_exposure:self._lock(prof.get_device().first_color_sensor(),rs)
        if self.publish:
            try:
                with open(LIVE_INFO+'.tmp','w') as f:json.dump({**self.describe(),'pid':os.getpid()},f)
                os.replace(LIVE_INFO+'.tmp',LIVE_INFO)
            except OSError:pass
        threading.Thread(target=self._loop,daemon=True).start()
        deadline=time.monotonic()+3
        while self._frame is None and time.monotonic()<deadline:time.sleep(0.02)
        if self._frame is None:raise RuntimeError('side camera gave no frames')
        return self

    def _lock(self,sensor,rs):
        opt={'exposure':rs.option.exposure,'gain':rs.option.gain,'white_balance':rs.option.white_balance}
        try:d=json.load(open(CAMERAS))
        except (OSError,ValueError):d={}
        saved=d.get('side_exposure')
        if not saved:
            t=time.monotonic()
            while time.monotonic()-t<2.5:self._pipe.wait_for_frames(1000)   # auto exposure / white balance settle
            saved={k:float(sensor.get_option(o)) for k,o in opt.items()};saved['locked_at']=time.strftime('%Y-%m-%dT%H:%M:%S')
            d['side_exposure']=saved
            with open(CAMERAS+'.tmp','w') as f:json.dump(d,f,indent=1)
            os.replace(CAMERAS+'.tmp',CAMERAS)
        sensor.set_option(rs.option.enable_auto_exposure,0);sensor.set_option(rs.option.enable_auto_white_balance,0)
        for k,o in opt.items():sensor.set_option(o,saved[k])
        self.exposure={k:saved[k] for k in LOCK_KEYS}

    def _loop(self):
        while not self._stop.is_set():
            try:f=self._pipe.wait_for_frames(1000)
            except RuntimeError:continue
            c=f.get_color_frame()
            if not c:continue
            img=np.asanyarray(c.get_data()).copy()
            with self.lock:self._frame=img;self._t=time.monotonic()
            if self.publish and self._t-self._pub_t>=0.2:self._pub_t=self._t;self._write(img)
            with self._rec_lock:
                if self._rec is not None:
                    try:self._rec[0].write(np.ascontiguousarray(img[:,:,::-1]));self._rec[1].append(round(time.monotonic(),4))
                    except Exception:pass

    def start_record(self,path):
        """Full-resolution side video (mp4v, camera fps) for reports; timestamps go to <path>.timestamps.json."""
        import cv2
        w=cv2.VideoWriter(str(path),cv2.VideoWriter_fourcc(*'mp4v'),self.fps,(self.w,self.h))
        with self._rec_lock:self._rec=(w,[],str(path))

    def stop_record(self):
        with self._rec_lock:rec,self._rec=self._rec,None
        if rec is None:return
        rec[0].release()
        with open(os.path.splitext(rec[2])[0]+'.timestamps.json','w') as f:json.dump({'video':os.path.basename(rec[2]),'t_monotonic_s':rec[1]},f)

    def _write(self,img):
        try:
            import cv2
            ok,buf=cv2.imencode('.jpg',img[:,:,::-1],[cv2.IMWRITE_JPEG_QUALITY,80])
            if ok:
                with open(LIVE_JPEG+'.tmp','wb') as f:f.write(buf.tobytes())
                os.replace(LIVE_JPEG+'.tmp',LIVE_JPEG)
        except Exception:pass   # the console view must never break the camera thread

    def latest(self,max_age=0.25):
        with self.lock:img=self._frame;age=time.monotonic()-self._t
        if img is None or age>max_age:raise RuntimeError(f'side camera stale ({age:.2f} s)')
        return img

    def describe(self):
        return {'name':self.name,'serial':self.serial,'size':[self.w,self.h],'fps':self.fps,'exposure':self.exposure}

    def stop(self):
        self._stop.set()
        if self._pipe is not None:
            try:self._pipe.stop()
            except Exception:pass


class FakeSideCamera:
    """Offline tests: a grey frame with a bright patch that brightens when the fake plug is fully inserted."""
    def __init__(self,usb=None):self.usb=usb;self.name='fake'
    def start(self):return self
    def latest(self,max_age=0.25):
        img=np.full((480,640,3),90,np.uint8)
        if self.usb is not None and self.usb.connected:img[200:280,280:360]=230
        return img
    def describe(self):return {'name':'fake'}
    def stop(self):pass


if __name__=='__main__':
    import sys,cv2
    cam=SideCamera().start();time.sleep(2.0);img=cam.latest();cam.stop()   # auto exposure / white balance settle first
    out=sys.argv[1] if len(sys.argv)>1 else '/tmp/side-view.jpg'
    cv2.imwrite(out,img[:,:,::-1]);print('saved',out,cam.describe())
