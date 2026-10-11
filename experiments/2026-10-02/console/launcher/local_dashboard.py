#!/usr/bin/env python3
"""Mac dashboard with background Jetson reconnect. Never queue or retry POSTs."""
import http.client
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
import mimetypes
from pathlib import Path
import re
import secrets
import subprocess
import threading
import time
from urllib.parse import urlsplit, unquote

STATIC = Path(__file__).resolve().parents[1] / 'static'
PORT, UPSTREAM_PORT = 8780, 18780
TOKEN = secrets.token_hex(32)
ONLINE = threading.Event()
SSH = ['/usr/bin/ssh', '-o', 'BatchMode=yes', '-o', 'ConnectTimeout=4',
       '-o', 'ConnectionAttempts=1', '-o', 'ServerAliveInterval=5', '-o', 'ServerAliveCountMax=2']
REMOTE_START = ('curl -fsS -m 2 -o /dev/null http://127.0.0.1:8780/api/session || '
                '(cd ~/a1z-experiments/2026-10-02 && setsid nohup ~/a1z-infer/run.sh '
                'console/server.py > /tmp/a1z_console.log 2>&1 < /dev/null &)')


def upstream(method, path, body=None, headers=None):
    conn = http.client.HTTPConnection('127.0.0.1', UPSTREAM_PORT, timeout=4)
    try:
        conn.request(method, path, body=body, headers=headers or {})
        return conn, conn.getresponse()
    except Exception:
        conn.close()
        raise


def probe():
    try:
        conn, response = upstream('GET', '/api/session')
        try:
            data = json.loads(response.read())
            return response.status == 200 and isinstance(data.get('state'), str)
        finally:
            conn.close()
    except (OSError, ValueError, http.client.HTTPException):
        return False


def reconnect():
    # Starts only the HTTP server, never a robot session or motor controller.
    while True:
        for host in ('jetson-direct', 'jetson'):
            proc = None
            try:
                result = subprocess.run(SSH + [host, REMOTE_START], timeout=12, stdin=subprocess.DEVNULL)
                if result.returncode:
                    continue
                proc = subprocess.Popen(SSH + ['-N', '-o', 'ExitOnForwardFailure=yes', '-L',
                    f'127.0.0.1:{UPSTREAM_PORT}:127.0.0.1:8780', host], stdin=subprocess.DEVNULL)
                for _ in range(20):
                    if proc.poll() is not None:
                        break
                    if probe():
                        ONLINE.set()
                        break
                    time.sleep(.5)
                while proc.poll() is None and ONLINE.is_set():
                    time.sleep(3)
                    if not probe():
                        ONLINE.clear()
            except (OSError, subprocess.TimeoutExpired) as exc:
                print(f'Background connection: {exc}', flush=True)
            finally:
                ONLINE.clear()
                if proc is not None and proc.poll() is None:
                    proc.terminate()
                    try:
                        proc.wait(timeout=3)
                    except subprocess.TimeoutExpired:
                        proc.kill()
                        proc.wait()
        time.sleep(5)


class Handler(BaseHTTPRequestHandler):
    def log_message(self, *args):
        pass

    def send(self, code, body, content_type='application/json; charset=utf-8'):
        if not isinstance(body, bytes):
            body = json.dumps(body).encode()
        self.send_response(code)
        self.send_header('Content-Type', content_type)
        self.send_header('Content-Length', str(len(body)))
        self.send_header('Cache-Control', 'no-store')
        self.end_headers()
        try:
            self.wfile.write(body)
        except (BrokenPipeError, ConnectionResetError):
            pass

    def allowed(self):
        return self.headers.get('Host', '') in (f'localhost:{PORT}', f'127.0.0.1:{PORT}')

    def unavailable(self):
        self.send(503, {'ok': False, 'error': 'Waiting for Jetson data',
                        'message': 'Controls are unavailable until Jetson reconnects.'})

    def do_GET(self):
        if not self.allowed():
            return self.send(403, {'error': 'host'})
        path = urlsplit(self.path).path
        if path == '/_local/health':
            return self.send(200, {'service': 'a1z-local-dashboard', 'online': ONLINE.is_set()})
        if path in ('/', '/index.html'):
            return self.send(200, (STATIC / 'index.html').read_text().replace(
                '__TOKEN__', TOKEN).encode(), 'text/html; charset=utf-8')
        if path.startswith('/static/'):
            file = (STATIC / unquote(path[len('/static/'):])).resolve()
            if STATIC.resolve() not in file.parents or not file.is_file():
                return self.send(404, {'error': 'not found'})
            return self.send(200, file.read_bytes(), mimetypes.guess_type(file.name)[0] or 'application/octet-stream')
        if path.startswith('/api/'):
            return self.forward()
        self.send(404, {'error': 'not found'})

    def do_POST(self):
        if not self.allowed() or not secrets.compare_digest(self.headers.get('X-A1Z-Token', ''), TOKEN):
            return self.send(403, {'error': 'token'})
        origin = self.headers.get('Origin')
        if origin and origin not in (f'http://localhost:{PORT}', f'http://127.0.0.1:{PORT}'):
            return self.send(403, {'error': 'origin'})
        if urlsplit(self.path).path not in ('/api/session/start', '/api/session/key', '/api/session/close', '/api/trial/edit',
                                             '/api/side/preview', '/api/side/crop'):
            return self.send(404, {'error': 'not found'})
        try:
            size = int(self.headers.get('Content-Length', '0'))
        except ValueError:
            return self.send(400, {'error': 'length'})
        if not 0 <= size <= 1048576:
            return self.send(413, {'error': 'body too large'})
        self.forward(self.rfile.read(size))

    def forward(self, body=None):
        if not ONLINE.is_set():
            return self.unavailable()
        conn, started = None, False
        try:
            headers = {}
            if self.headers.get('Range'):
                headers['Range'] = self.headers['Range']
            if self.command == 'POST':
                conn, response = upstream('GET', '/')
                html, status = response.read().decode(), response.status
                conn.close()
                match = re.search(r'name="a1z-token" content="([a-f0-9]+)"', html)
                if status != 200 or not match:
                    raise ValueError('Jetson dashboard token unavailable')
                headers.update({'X-A1Z-Token': match.group(1), 'Content-Type': 'application/json'})
            conn, response = upstream(self.command, self.path, body, headers)
            self.send_response(response.status)
            for key in ('Content-Type', 'Content-Length', 'Content-Range', 'Accept-Ranges', 'X-Frame-Age'):
                if response.getheader(key) is not None:
                    self.send_header(key, response.getheader(key))
            self.send_header('Cache-Control', 'no-store')
            self.end_headers()
            started = True
            while chunk := response.read(65536):
                self.wfile.write(chunk)
        except (BrokenPipeError, ConnectionResetError):
            pass
        except (OSError, ValueError, http.client.HTTPException):
            ONLINE.clear()
            if not started:
                self.unavailable()  # POST outcome may be unknown; never replay it.
        finally:
            if conn is not None:
                conn.close()


def main():
    server = ThreadingHTTPServer(('127.0.0.1', PORT), Handler)
    threading.Thread(target=reconnect, daemon=True).start()
    print(f'A1Z local dashboard http://localhost:{PORT}/#/run', flush=True)
    server.serve_forever()


if __name__ == '__main__':
    main()
