#!/usr/bin/env python3
"""Live side-camera view for aiming the mount: MJPEG over HTTP on 127.0.0.1 (reach it with ssh -L).
Overlay: centre cross and thirds grid. Opens only the non-D405 camera; stop it (Ctrl+C) before actor.py, which
needs the same camera.

  Jetson: ~/a1z-infer/run.sh side_preview.py [--port 8770]
  Mac:    ssh -N -L 8770:127.0.0.1:8770 jetson-direct   then open http://127.0.0.1:8770
  --headless: no HTTP server, only keep the camera open so it publishes /dev/shm/a1z_side.jpg for the A1Z Console's
  Cameras page (which starts and stops this in tmux session "sidecam"; actor.py ends that session before it opens
  the camera).
"""
import argparse
from http.server import BaseHTTPRequestHandler,ThreadingHTTPServer
import time

import cv2

from side_camera import SideCamera

PAGE=b'<html><body style="margin:0;background:#111"><img src="/stream" style="width:100%;max-width:960px"></body></html>'


def overlay(img):
    img=img.copy();h,w=img.shape[:2]
    for x in (w//3,2*w//3):cv2.line(img,(x,0),(x,h),(80,80,80),1)
    for y in (h//3,2*h//3):cv2.line(img,(0,y),(w,y),(80,80,80),1)
    cv2.drawMarker(img,(w//2,h//2),(255,60,60),cv2.MARKER_CROSS,30,2)
    return img


def main():
    p=argparse.ArgumentParser();p.add_argument('--port',type=int,default=8770);p.add_argument('--model',default=None)
    p.add_argument('--headless',action='store_true');a=p.parse_args()
    cam=SideCamera(model=a.model).start();print('side camera',cam.describe(),flush=True)
    if a.headless:
        try:
            while True:time.sleep(1)
        except KeyboardInterrupt:pass
        finally:cam.stop()
        return

    class H(BaseHTTPRequestHandler):
        def log_message(self,*x):pass
        def do_GET(self):
            if self.path!='/stream':
                self.send_response(200);self.send_header('Content-Type','text/html');self.end_headers();self.wfile.write(PAGE);return
            self.send_response(200);self.send_header('Content-Type','multipart/x-mixed-replace; boundary=f');self.end_headers()
            try:
                while True:
                    ok,buf=cv2.imencode('.jpg',overlay(cam.latest(1.0))[:,:,::-1],[cv2.IMWRITE_JPEG_QUALITY,75])
                    self.wfile.write(b'--f\r\nContent-Type: image/jpeg\r\n\r\n'+buf.tobytes()+b'\r\n');time.sleep(0.1)
            except (BrokenPipeError,ConnectionResetError):pass

    srv=ThreadingHTTPServer(('127.0.0.1',a.port),H);print(f'SIDE_PREVIEW http://127.0.0.1:{a.port}',flush=True)
    try:srv.serve_forever()
    except KeyboardInterrupt:pass
    finally:cam.stop()


if __name__=='__main__':main()
