#!/usr/bin/env python3
"""Read-only MJPEG view of /dev/shm/a1z-preview.jpg (written by camera_session.py), localhost only.
Reach it from the Mac through `ssh -L 8771:127.0.0.1:8771 jetson-direct`."""
import http.server
from pathlib import Path
import time

FRAME = Path('/dev/shm/a1z-preview.jpg')
PORT = 8771


class Handler(http.server.BaseHTTPRequestHandler):
    def log_message(self, *args): pass

    def do_GET(self):
        if self.path.startswith('/status'):
            age = time.time() - FRAME.stat().st_mtime if FRAME.exists() else None
            body = ('{"age_s": %s}' % ('null' if age is None else round(age, 2))).encode()
            self.send_response(200); self.send_header('Content-Type', 'application/json')
            self.send_header('Access-Control-Allow-Origin', '*'); self.send_header('Cache-Control', 'no-store')
            self.end_headers(); self.wfile.write(body); return
        if not self.path.startswith('/stream'):
            self.send_error(404); return
        self.send_response(200)
        self.send_header('Content-Type', 'multipart/x-mixed-replace; boundary=frame')
        self.send_header('Cache-Control', 'no-store'); self.end_headers()
        last = None
        try:
            while True:
                try:
                    mtime = FRAME.stat().st_mtime
                    if mtime != last and time.time() - mtime < 2:
                        jpg = FRAME.read_bytes(); last = mtime
                        self.wfile.write(b'--frame\r\nContent-Type: image/jpeg\r\nContent-Length: %d\r\n\r\n' % len(jpg) + jpg + b'\r\n')
                        self.wfile.flush()
                except FileNotFoundError:
                    pass
                time.sleep(0.05)
        except (BrokenPipeError, ConnectionResetError):
            pass


if __name__ == '__main__':
    server = http.server.ThreadingHTTPServer(('127.0.0.1', PORT), Handler)
    server.daemon_threads = True
    server.serve_forever()
