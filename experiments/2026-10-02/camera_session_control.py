"""Manage only a camera child process; caller must stop the arm before stop()."""
import json
import math
import os
import subprocess
import sys
import time
from pathlib import Path


class CameraSession:
    def __init__(self, output, max_seconds=300, worker_path=None):
        self.output = Path(output).resolve()
        self.max_seconds = max_seconds
        self.worker_path = Path(worker_path) if worker_path else Path(__file__).with_name('camera_session.py')
        self.ready_file = self.output.with_name(self.output.name+'.ready.json')
        self.stop_file = self.output.with_name(self.output.name+'.stop')
        self.log_file = self.output.with_name(self.output.name+'.camera.log')
        self.process = None
        self._log = None
        self.ready = None
        self.error = None

    def _metadata(self, **extra):
        return {'dataset_dir':str(self.output), 'ready':self.ready is not None,
                'error':self.error, 'log_file':str(self.log_file), **extra}

    def _close_log(self):
        if self._log:
            self._log.close()
            self._log = None

    def _terminate(self):
        if self.process and self.process.poll() is None:
            self.process.terminate()
            try:
                self.process.wait(timeout=3)
            except subprocess.TimeoutExpired:
                self.process.kill()
                self.process.wait(timeout=3)
        self._close_log()

    def prepare(self, timeout=30):
        if self.process is not None:
            return self._metadata(**(self.ready or {}))
        if any(p.exists() for p in [self.output,self.ready_file,self.stop_file,self.log_file]):
            self.error = 'Camera session path already exists; use a new session path'
            return self._metadata()
        try:
            self.output.parent.mkdir(parents=True,exist_ok=True)
            self._log = self.log_file.open('x')
            command = [sys.executable,str(self.worker_path),'--output',str(self.output),
                '--stop-file',str(self.stop_file),'--ready-file',str(self.ready_file),
                '--max-seconds',str(self.max_seconds)]
            self.process = subprocess.Popen(command,stdout=self._log,stderr=subprocess.STDOUT,
                preexec_fn=lambda:os.nice(10), start_new_session=True)
            deadline = time.monotonic()+min(30,max(0,timeout))
            while time.monotonic()<deadline:
                if self.process.poll() is not None:
                    self.error = f'Camera worker exited before ready (code {self.process.returncode})'
                    self._close_log()
                    return self._metadata()
                if self.ready_file.exists():
                    ready = json.loads(self.ready_file.read_text())
                    if not self._valid_capture_status(ready):
                        raise ValueError('Camera status is not fresh and capturing')
                    self.ready = ready
                    return self._metadata(**ready)
                time.sleep(.05)
            self.error = 'Camera startup timed out before ready'
            self._terminate()
        except Exception as exc:
            self.error = f'Camera startup: {type(exc).__name__}: {exc}'
            self._terminate()
        return self._metadata()

    @staticmethod
    def _valid_capture_status(status):
        if not isinstance(status, dict) or status.get('state') != 'capturing':
            return False
        stamp = status.get('host_monotonic_s')
        return (isinstance(stamp, (int, float)) and not isinstance(stamp, bool)
                and math.isfinite(stamp) and 0 <= time.monotonic() - stamp <= 3.0)

    def is_running(self):
        if self.process is None or self.process.poll() is not None:
            return False
        try:
            return self._valid_capture_status(json.loads(self.ready_file.read_text()))
        except (OSError, ValueError, TypeError):
            return False

    def check_running(self):
        """Nonblocking health check for the robot control loop."""
        running = self.is_running()
        if not running:
            self.error = self.error or "Camera worker stopped unexpectedly"
        return self._metadata(running=running)

    def stop(self, timeout=60):
        """Call after arm shutdown; encoding may block up to timeout seconds."""
        if self.process is None:
            self.error = self.error or 'Camera worker was not started'
            return self._metadata(success=False)
        try:
            self.stop_file.touch()
            self.process.wait(timeout=min(60,max(0,timeout)))
        except subprocess.TimeoutExpired:
            self.error = 'Camera stop/encoding timed out; partial data retained'
            self._terminate()
        except Exception as exc:
            self.error = f'Camera stop: {type(exc).__name__}: {exc}'
            self._terminate()
        finally:
            self._close_log()
        result = {}
        try:
            result = json.loads((self.output/'result.json').read_text())
        except Exception as exc:
            self.error = self.error or f'Camera result unavailable: {type(exc).__name__}: {exc}'
        if self.process.returncode != 0:
            self.error = self.error or result.get('error') or f'Camera worker failed (code {self.process.returncode})'
        if result.get('success') is not True:
            self.error = self.error or result.get('error') or 'Camera worker did not confirm success'
        return {**result, **self._metadata(success=not bool(self.error),returncode=self.process.returncode)}
