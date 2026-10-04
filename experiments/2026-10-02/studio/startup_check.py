"""Pre-enable checks executed on Jetson. Does not import or instantiate robot SDK."""
from pathlib import Path


def check_source():
    return Path(__file__).read_text() + '\nprint(json.dumps(run_checks(), ensure_ascii=False))\n'


import fcntl
import json
import shutil
import socket
import subprocess
import tempfile


def run_checks(root=Path('/home/dimos/a1z-experiments/2026-10-02')):
    checks = []
    result = {'ok': False, 'host': socket.gethostname(), 'checks': checks}
    lock = None
    def passed(label, detail):
        checks.append({'label': label, 'state': 'ok', 'detail': detail})
    def run(args):
        p = subprocess.run(args, capture_output=True, text=True, timeout=8)
        if p.returncode:
            raise RuntimeError(p.stderr.strip() or '命令执行失败：' + ' '.join(args[:3]))
        return p.stdout
    stage = '采集程序'
    try:
        required = ['collect_task.py', 'teach.py', 'camera_session.py', 'camera_session_control.py']
        missing = [name for name in required if not (root/name).is_file()]
        if missing: raise RuntimeError('缺少 ' + ', '.join(missing))
        if not Path('/home/dimos/dimos/.venv/bin/python').is_file():
            raise RuntimeError('Jetson 采集 Python 环境不存在。')
        if shutil.which('ffmpeg') is None: raise RuntimeError('相机编码器 ffmpeg 不存在。')
        passed(stage, '程序和运行环境入口存在')
        stage = '控制占用'
        lock = (root/'controller.lock').open('a')
        try: fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError: raise RuntimeError('另一采集或控制进程仍在运行；先结束原会话，不会自动打断它。')
        # The lock covers all changes to CAN; collect_task reacquires it before use.
        passed(stage, '控制锁可用')
        stage = '相机连接'
        usb = run(['lsusb'])
        if '8086:0b5b' not in usb.lower(): raise RuntimeError('未识别到 D405，请检查相机 USB 连接。')
        passed(stage, 'D405 已连接；启动后验证图像帧')
        stage = '存储'
        directory = root/'episodes'; directory.mkdir(exist_ok=True)
        with tempfile.TemporaryFile(dir=directory) as f:
            f.write(b'a1z-write-check'); f.flush()
        free = shutil.disk_usage(root).free
        passed(stage, f'可写，剩余 {free / 1024**3:.1f} GB；视频保留在 Jetson')
        stage = 'CAN'
        links = json.loads(run(['ip', '-details', '-json', 'link', 'show', 'can1']))
        if len(links) != 1: raise RuntimeError('未识别到 can1，请检查 USB-CAN 连接。')
        link = links[0]; detail = link.get('linkinfo', {}).get('info_data', {})
        repaired = False
        if 'UP' not in link.get('flags', []):
            run(['sudo', '-n', 'ip', 'link', 'set', 'can1', 'up', 'type', 'can', 'bitrate', '1000000'])
            repaired = True
            link = json.loads(run(['ip', '-details', '-json', 'link', 'show', 'can1']))[0]
            detail = link.get('linkinfo', {}).get('info_data', {})
        bitrate = detail.get('bittiming', {}).get('bitrate')
        if 'UP' not in link.get('flags', []) or bitrate != 1000000:
            raise RuntimeError('CAN 未处于厂家 1 Mbps 配置；未重置正在运行的接口。')
        if detail.get('state') not in {'ERROR-ACTIVE', 'ERROR-WARNING'}:
            raise RuntimeError('CAN 状态异常：' + str(detail.get('state')) + '；请检查机械臂供电和接线。')
        passed(stage, ('已自动恢复' if repaired else '已就绪') + ' · 1 Mbps')
        checks.append({'label': '机械臂', 'state': 'pending', 'detail': '尚未启用；现场确认后检查关节反馈'})
        result['ok'] = True
    except Exception as e:
        message = stage + '：' + str(e)
        checks.append({'label': stage, 'state': 'error', 'detail': str(e)})
        result['error'] = message
    finally:
        if lock is not None: lock.close()
    return result
