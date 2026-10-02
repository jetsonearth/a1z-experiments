#!/usr/bin/env python3
"""Foreground serial console. Never auto-answer the robot's readiness prompt."""
import fcntl
import glob
import re
import os
from pathlib import Path
import select
import signal
import sys
import termios
import time
import tty

HERE = Path(__file__).resolve().parent
LOG = HERE / 'console-latest.log'
END = b'__A1Z_TEACH_FINISHED__'
COMMAND = '''python3 -c "import subprocess,signal;signal.signal(signal.SIGINT,lambda *a:None);r=subprocess.call(['/home/dimos/dimos/.venv/bin/python','-u','/home/dimos/a1z-experiments/2026-10-02/teach.py']);print('__A1Z_TEACH_FINISHED__',r,flush=True)"\r'''


def main():
    if not sys.stdin.isatty():
        raise RuntimeError('Open this program in an interactive Terminal')
    lock = (HERE / 'console.lock').open('a')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    ports = glob.glob('/dev/cu.usbserial-*')
    if len(ports) != 1:
        raise RuntimeError('Expected exactly one Jetson serial device; check connection')
    fd = os.open(ports[0], os.O_RDWR | os.O_NOCTTY | os.O_NONBLOCK)
    fcntl.ioctl(fd, termios.TIOCEXCL)
    old_stdin = termios.tcgetattr(sys.stdin.fileno())
    old_serial = termios.tcgetattr(fd)
    attrs = termios.tcgetattr(fd)
    attrs[0] = attrs[1] = attrs[3] = 0
    attrs[2] = termios.CS8 | termios.CREAD | termios.CLOCAL
    attrs[4] = attrs[5] = termios.B115200
    attrs[6][termios.VMIN] = attrs[6][termios.VTIME] = 0
    termios.tcsetattr(fd, termios.TCSANOW, attrs)
    signal.signal(signal.SIGHUP, lambda *unused: (_ for _ in ()).throw(KeyboardInterrupt()))
    signal.signal(signal.SIGTERM, lambda *unused: (_ for _ in ()).throw(KeyboardInterrupt()))
    completed = False
    launched = False
    tail = b''
    try:
        tty.setraw(sys.stdin.fileno())
        os.write(sys.stdout.fileno(), '\r\nA1Z 首次短轨迹录制 | 等待启动前检查\r\n'.encode())
        with LOG.open('wb', buffering=0) as log:
            # Ctrl-L only redraws the shell; never press Enter into an unknown
            # foreground program (it could be a recorder readiness prompt).
            os.write(fd, b'\x0c')
            observed = b''
            until = time.monotonic() + 3
            while time.monotonic() < until:
                if select.select([fd], [], [], .15)[0]:
                    observed += os.read(fd, 65536)
                    clean = re.sub(r'\x1b\[[0-?]*[ -/]*[@-~]', '', observed.decode(errors='replace')).replace('\x00', '')
                    if re.search(r'dimos@jetson01[^\n]*%\s*$', clean):
                        break
            else:
                raise RuntimeError('Jetson idle shell not confirmed; no recording command sent')
            payload = COMMAND.encode()
            if os.environ.get('A1Z_CAMERA') == '1':
                payload = payload.replace(b"teach.py']", b"teach.py','--camera']")
            launched = True
            for offset in range(0, len(payload), 64):
                part = payload[offset:offset + 64]
                sent = 0
                while sent < len(part):
                    try:
                        sent += os.write(fd, part[sent:])
                    except BlockingIOError:
                        select.select([], [fd], [], 0.1)
                time.sleep(0.03)
            while True:
                readable, _, _ = select.select([fd, sys.stdin.fileno()], [], [], 0.25)
                if fd in readable:
                    data = os.read(fd, 65536)
                    if not data:
                        raise RuntimeError('Serial disconnected; keep supporting the arm and use its independent power switch if necessary.')
                    log.write(data)
                    os.write(sys.stdout.fileno(), data)
                    tail = (tail + data)[-1024:]
                    # Require the standalone result line, not the command echo.
                    if b'\n' + END + b' ' in tail:
                        completed = True
                        break
                if sys.stdin.fileno() in readable:
                    data = os.read(sys.stdin.fileno(), 1024)
                    if not data:
                        raise KeyboardInterrupt
                    sent = 0
                    while sent < len(data):
                        try:
                            sent += os.write(fd, data[sent:])
                        except BlockingIOError:
                            select.select([], [fd], [], 0.1)
    finally:
        if launched and not completed:
            try:
                os.write(fd, b'\x03')
                time.sleep(0.5)
            except OSError:
                pass
        termios.tcsetattr(sys.stdin.fileno(), termios.TCSANOW, old_stdin)
        try:
            termios.tcsetattr(fd, termios.TCSANOW, old_serial)
        except OSError:
            pass
        os.close(fd)
    print('\n本轮结束。请保持机械臂有支撑，回到 Codex 查看结果。')


if __name__ == '__main__':
    main()
