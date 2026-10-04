#!/usr/bin/env python3
"""Local A1Z recorder UI. Task recording via manufacturer SDK defaults."""
import base64
import codecs
import fcntl
import glob
import hashlib
import http.server
import json
import math
import os
from pathlib import Path
import pty
import re
import secrets
import select
import signal
import shutil
import subprocess
import sys
import termios
import threading
import time
import urllib.parse
import uuid
import zlib

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT.parent))
from serial_session import ensure_shell
from startup_check import check_source
EXPERIMENT = ROOT.parent
EPISODES = EXPERIMENT / 'episodes'
# Takes across all rounds of this collection; the per-round session list resets on every new round.
COLLECTION = EXPERIMENT / 'collection.json'


def load_collection():
    try:
        data = json.loads(COLLECTION.read_text())
        return [e for e in data.get('episodes', []) if isinstance(e, dict) and e.get('path')]
    except (OSError, ValueError):
        return []


def save_collection(episodes):
    tmp = COLLECTION.with_suffix('.json.tmp')
    tmp.write_text(json.dumps({'episodes': episodes}, ensure_ascii=False, indent=1))
    os.replace(tmp, COLLECTION)
STATIC = ROOT / 'static'
CAMERA_TEST = EXPERIMENT / 'diagnostics' / 'lerobot-camera-test' / 'file-000.mp4'
PORT = int(os.environ.get('A1Z_STUDIO_PORT', '8769'))
TOKEN = secrets.token_urlsafe(32)
HARDWARE_ENABLED = os.environ.get('A1Z_STUDIO_HARDWARE') == '1'
FETCH_VIDEO = os.environ.get('A1Z_STUDIO_FETCH_VIDEO') == '1'  # User prefers original videos on Jetson.
ANSI = re.compile(r'\x1b\[[0-?]*[ -/]*[@-~]')
BUSY = {'checking', 'starting', 'waiting', 'recording', 'stopping', 'finalizing', 'syncing'}
MEDIA_PAUSE = threading.Event()

class MediaTransferPaused(Exception):
    pass

def check_media_pause():
    if MEDIA_PAUSE.is_set(): raise MediaTransferPaused('视频传输已暂停，关节数据已保存；可继续采集，稍后点同步续传。')

REMOTE_DIR = '/home/dimos/a1z-experiments/2026-10-02/episodes/'


def episode_index():
    rows = []
    for p in sorted(EPISODES.glob('*-teaching.json'), key=lambda p: int(p.name.split('-')[1]) if p.name.split('-')[1].isdigit() else 0, reverse=True):
        try:
            x = episode_data(p.name)
            meta, samples = x['metadata'], x['samples']
            if not samples:
                continue
            t0 = samples[0]['t_monotonic_s']
            duration = samples[-1]['t_monotonic_s'] - t0
            rows.append({'id': p.name, 'date': meta.get('experiment_date', '未知日期'),
                         'duration': duration, 'frames': len(samples),
                         'hz': (len(samples)-1) / duration if duration else 0,
                         'playback_basis': x.get('playback_basis'), 'status': meta.get('status', 'unknown'), 'video': bool((local_media(p.name) or {}).get('video_url'))})
        except (ValueError, KeyError, TypeError):
            continue
    return rows


def serial_rpc(source, timeout=35):
    """Send only after observing an idle shell prompt; never interrupt a process."""
    lock = (EXPERIMENT / 'console.lock').open('a')
    fd = None
    old_attrs = None
    locked = False
    try:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        locked = True
        ports = glob.glob('/dev/cu.usbserial-*')
        device = os.environ.get('A1Z_SERIAL_DEVICE')
        if device and device not in ports:
            raise RuntimeError('指定的 Jetson 串口不存在。')
        if not device:
            device = ports[0] if len(ports) == 1 else None
        if not device:
            raise RuntimeError('Jetson 串口不存在或有多个候选，请明确设备。')
        fd = os.open(device, os.O_RDWR | os.O_NOCTTY | os.O_NONBLOCK)
        fcntl.ioctl(fd, termios.TIOCEXCL)
        old_attrs = termios.tcgetattr(fd)
        a = termios.tcgetattr(fd)
        a[0] = a[1] = a[3] = 0
        a[2] = termios.CS8 | termios.CREAD | termios.CLOCAL
        a[4] = a[5] = termios.B115200
        a[6][termios.VMIN] = a[6][termios.VTIME] = 0
        termios.tcsetattr(fd, termios.TCSANOW, a)
        ensure_shell(fd)
        marker = 'RPC_' + uuid.uuid4().hex
        wrapped = f"print('{marker}_BEGIN',flush=True)\n" + source + f"\nprint('{marker}_END',flush=True)\n"
        encoded = base64.b64encode(zlib.compress(wrapped.encode())).decode()
        payload = (f'python3 -c "import base64,zlib;exec(zlib.decompress(base64.b64decode(\'{encoded}\')))"\r').encode()
        output = b''
        for i in range(0, len(payload), 64):
            block = payload[i:i+64]
            while block:
                try:
                    n = os.write(fd, block); block = block[n:]
                except BlockingIOError:
                    select.select([], [fd], [], .1)
            time.sleep(.025)
            while select.select([fd], [], [], 0)[0]:
                part = os.read(fd, 65536)
                if not part: raise RuntimeError('Jetson 串口已断开。')
                output += part
        end = time.monotonic() + timeout
        while time.monotonic() < end:
            clean = ANSI.sub('', output.decode(errors='replace')).replace('\x00', '').replace('\r','')
            match = re.search(r'(?:^|\n)' + marker + r'_BEGIN\n(.*?)\n' + marker + r'_END(?:\n|$)', clean, re.S)
            if match:
                return match.group(1).strip()
            if select.select([fd], [], [], .2)[0]:
                part = os.read(fd, 65536)
                if not part: raise RuntimeError('Jetson 串口已断开。')
                output += part
        raise RuntimeError('Jetson 响应超时；没有收到完整结果，未中断远端进程。')
    finally:
        try:
            if fd is not None:
                try:
                    if old_attrs is not None: termios.tcsetattr(fd, termios.TCSANOW, old_attrs)
                finally:
                    os.close(fd)
        finally:
            try:
                if locked: fcntl.flock(lock, fcntl.LOCK_UN)
            finally:
                lock.close()

def settle_shell(attempts=3, wait=1.5):
    """Call only right after our own wrapper printed its finished marker (known foreground)."""
    for attempt in range(attempts):
        time.sleep(wait)
        try:
            return serial_rpc("print('settled')", 10)
        except RuntimeError as e:
            if 'idle shell not confirmed' not in str(e) or attempt == attempts - 1: raise

def media_dir(name):
    return EPISODES / (Path(name).stem + '-media')


def successful_dataset(metadata):
    dataset = metadata.get('lerobot_dataset')
    if not isinstance(dataset, dict) or not dataset.get('dataset_dir'):
        return None
    if dataset.get('success') is not True or dataset.get('validated') is not True or dataset.get('error'):
        return None
    return dataset


def local_media(name):
    folder = media_dir(name)
    manifest = folder / 'manifest.json'
    if not manifest.is_file(): return None
    info = json.loads(manifest.read_text())
    canonical = info.get('source') == 'lerobot' and info.get('complete') is True and successful_dataset({'lerobot_dataset':info.get('remote_dataset')}) is not None
    video = 'unified-wrist.mp4' if canonical else 'wrist.mp4'
    timestamps = 'unified-timestamps.json' if canonical else 'timestamps.json'
    if not (folder/video).is_file():
        return {'error': info.get('error', '视频尚未传回 Mac'), 'remote': True,
                'export_error': info.get('export_error')}
    if canonical and not all((folder/f).is_file() for f in [timestamps,'playback.json']):
        return {'error':'LeRobot 回放文件尚未完整传回 Mac', 'remote':True}
    base = '/api/media/' + urllib.parse.quote(name) + '/'
    return {'video_url': base+video,
            'timestamps_url': base+timestamps if (folder/timestamps).is_file() else None,
            'fps': info.get('fps',30), 'error': info.get('error'),
            'export_error': info.get('export_error'), 'source':'lerobot' if canonical else 'raw',
            'sync_basis': 'LeRobot 30 Hz 对齐观测' if canonical else 'host receipt; not calibrated exposure',
            'info_url':base+'info.json' if canonical and (folder/'info.json').is_file() else None,
            'alignment_report_url':base+'alignment-report.json' if canonical and (folder/'alignment-report.json').is_file() else None}


def episode_data(name):
    data=json.loads((EPISODES/name).read_text())
    media=local_media(name)
    data['local_media']=media
    data['playback_basis']='原始关节记录'
    if media and media.get('source')=='lerobot':
        playback=json.loads((media_dir(name)/'playback.json').read_text())
        raw_samples = data['samples']
        data['raw_sample_count']=len(raw_samples)
        index = 0
        for sample in playback['samples']:
            while index + 1 < len(raw_samples) and raw_samples[index+1]['t_monotonic_s'] <= sample['t_monotonic_s']: index += 1
            if raw_samples:
                for key in ('gripper_position','gripper_target','gripper_feedback_valid'):
                    if key in raw_samples[index]: sample.setdefault(key,raw_samples[index][key])
        data['samples']=playback['samples']
        data['playback_basis']='LeRobot 30 Hz 对齐观测'
    return data


def fetch_remote_file(remote, dest, progress=None):
    # All callers validate and constrain remote paths before arriving here.
    check_media_pause()
    out = serial_rpc("from pathlib import Path\nimport json,hashlib\n" +
                     f"p=Path({remote!r})\n" +
                     "print(json.dumps({'bytes':p.stat().st_size,'sha256':hashlib.sha256(p.read_bytes()).hexdigest()}))", 45)
    info = json.loads(out)
    size = info['bytes']
    if not isinstance(size,int) or not 0 < size <= 128*1024*1024:
        raise RuntimeError('媒体文件为空或超过 128 MB 单文件传输上限。')
    checksum = info['sha256']
    if not re.fullmatch(r'[a-f0-9]{64}', checksum):
        raise RuntimeError('远端文件校验信息无效。')
    cache = EPISODES / '.media-cache' / checksum
    for candidate in (dest, cache):
        if candidate.is_file() and candidate.stat().st_size == size and hashlib.sha256(candidate.read_bytes()).hexdigest() == checksum:
            if candidate != dest: shutil.copyfile(candidate, dest)
            return info
    tmp = dest.with_suffix(dest.suffix+'.incoming')
    sidecar = tmp.with_suffix(tmp.suffix+'.sha256')
    resumable = tmp.is_file() and tmp.stat().st_size <= size and (not sidecar.is_file() or sidecar.read_text() == checksum)
    offset_start = tmp.stat().st_size if resumable else 0
    digest = hashlib.sha256(tmp.read_bytes() if resumable else b'')
    sidecar.write_text(checksum)
    with tmp.open('ab' if resumable else 'wb') as writer:
        for offset in range(offset_start,size,65536):
            check_media_pause()
            code = ("import base64\n" + f"with open({remote!r},'rb') as f:\n f.seek({offset})\n print(base64.b64encode(f.read({min(65536,size-offset)})).decode())")
            block = base64.b64decode(serial_rpc(code,45),validate=True)
            if len(block) != min(65536,size-offset): raise RuntimeError('媒体传输长度不符，远端原文件保留。')
            writer.write(block); digest.update(block)
            if progress: progress(f'录制已结束，正在传回 {dest.name}：{min(offset+len(block),size)*100//size}%')
    if digest.hexdigest() != info['sha256']:
        tmp.replace(tmp.with_suffix(tmp.suffix+'.checksum-failed'))
        raise RuntimeError('媒体 SHA256 校验不符，远端原文件保留；可重新同步。')
    tmp.replace(dest)
    sidecar.unlink(missing_ok=True)
    cache.parent.mkdir(exist_ok=True)
    shutil.copyfile(dest, cache)
    return info


def sync_media(name, metadata, progress=None):
    if not FETCH_VIDEO:
        if progress: progress('关节数据已同步；视频保留在 Jetson，不下载到 Mac。')
        return
    canonical=successful_dataset(metadata)
    camera=metadata.get('camera')
    if not canonical and (not isinstance(camera,dict) or not camera.get('dataset_dir')): return
    folder=media_dir(name); folder.mkdir(exist_ok=True)
    export_info=metadata.get('lerobot_dataset')
    export_error=metadata.get('lerobot_error') or metadata.get('lerobot_export_error')
    if isinstance(export_info,dict): export_error=export_info.get('error') or export_error
    manifest={'fps':30, 'export_error':export_error}

    def download_source(source, unified):
        remote_root=Path(source['dataset_dir'])
        allowed=Path('/home/dimos/a1z-experiments/2026-10-02/datasets')
        if remote_root.parent != allowed or not re.fullmatch(r'[A-Za-z0-9_-]+',remote_root.name):
            raise RuntimeError('数据集路径不符合预期。')
        checks=source.get('video_checks') or []
        videos=[x.get('file') for x in checks if isinstance(x,dict) and str(x.get('file','')).endswith('.mp4')]
        if len(videos)!=1: raise RuntimeError('未找到唯一相机视频；视频仍保留在 Jetson。')
        rel=Path(videos[0])
        if rel.is_absolute() or '..' in rel.parts or not rel.parts or rel.parts[0]!='videos':
            raise RuntimeError('相机视频相对路径不符合预期。')
        if unified and (source.get('timestamps_file','frame-timestamps.json')!='frame-timestamps.json' or source.get('playback_file','playback.json')!='playback.json'):
            raise RuntimeError('LeRobot 回放伴随文件路径不符合预期。')
        video_file='unified-wrist.mp4' if unified else 'wrist.mp4'
        timestamp_file='unified-timestamps.json' if unified else 'timestamps.json'
        result={'source':'lerobot' if unified else 'raw','remote_dataset':source,'fps':30}
        result['video']=fetch_remote_file(str(remote_root/rel),folder/video_file,progress)
        result['timestamps']=fetch_remote_file(str(remote_root/('frame-timestamps.json' if unified else 'raw-camera-timestamps.json')),folder/timestamp_file,progress)
        if unified:
            result['playback']=fetch_remote_file(str(remote_root/'playback.json'),folder/'playback.json',progress)
            playback=json.loads((folder/'playback.json').read_text())
            frames=json.loads((folder/timestamp_file).read_text())['frames']
            samples=playback.get('samples',[])
            if not samples or len(samples)!=len(frames) or (source.get('frames') and len(samples)!=source['frames']):
                raise RuntimeError('LeRobot 视频时间轴与观测帧数不一致。')
            for index,(sample,frame) in enumerate(zip(samples,frames)):
                if len(sample.get('q',[]))!=6 or not all(isinstance(v,(int,float)) and math.isfinite(v) for v in sample['q']):
                    raise RuntimeError('LeRobot 回放关节观测无效。')
                t=sample.get('t_monotonic_s'); ft=frame.get('host_monotonic_s')
                if not isinstance(t,(int,float)) or not math.isfinite(t) or not isinstance(ft,(int,float)) or not math.isfinite(ft) or abs(t-ft)>1e-6:
                    raise RuntimeError('LeRobot 回放目标时钟不一致。')
                if index and abs(t-samples[index-1]['t_monotonic_s']-1/30)>1e-4:
                    raise RuntimeError('LeRobot 回放不符合 30 Hz 时间网格。')
            for remote_file,local_file in [('meta/info.json','info.json'),('alignment-report.json','alignment-report.json')]:
                try: result[local_file]=fetch_remote_file(str(remote_root/remote_file),folder/local_file,progress)
                except MediaTransferPaused: raise
                except Exception as e: result.setdefault('inspection_warnings',[]).append(str(e))
        result['complete']=True
        return result

    if canonical:
        try: manifest.update(download_source(canonical,True))
        except MediaTransferPaused: raise
        except Exception as e:
            manifest['export_error']='LeRobot 本地回放整理未完成：'+str(e)
            if progress: progress(manifest['export_error']+'；尝试保留原始视频回放。')
    if not manifest.get('complete') and isinstance(camera,dict) and camera.get('dataset_dir'):
        try: manifest.update(download_source(camera,False))
        except MediaTransferPaused: raise
        except Exception as e:
            manifest['error']=str(e)
            if progress: progress('关节记录已保存；视频传输未完整完成：'+str(e))
    if not manifest.get('complete'): manifest.setdefault('error',manifest.get('export_error') or '媒体尚未完整传回 Mac')
    temp=folder/'manifest.incoming';temp.write_text(json.dumps(manifest,ensure_ascii=False,indent=2));temp.replace(folder/'manifest.json')


def remote_episode(path, progress=None, on_json_saved=None):
    if not re.fullmatch(re.escape(REMOTE_DIR) + r'[A-Za-z0-9_-]+-teaching\.json', path):
        raise RuntimeError('录制文件路径不符合预期。')
    source = ("from pathlib import Path\nimport base64,zlib,hashlib\n"
              f"raw=Path({path!r}).read_bytes()\n"
              "print('SHA',hashlib.sha256(raw).hexdigest())\n"
              "print('DATA',base64.b64encode(zlib.compress(raw)).decode())")
    out = serial_rpc(source, 90)
    digest = re.search(r'^SHA ([a-f0-9]{64})', out, re.M).group(1)
    encoded = re.search(r'^DATA ([A-Za-z0-9+/=]+)', out, re.M).group(1)
    raw = zlib.decompress(base64.b64decode(encoded))
    if hashlib.sha256(raw).hexdigest() != digest:
        raise RuntimeError('文件传输校验失败，Jetson 原文件保留。')
    data = json.loads(raw)
    if not isinstance(data.get('samples'), list) or not isinstance(data.get('metadata'), dict):
        raise RuntimeError('无效的轨迹文件。')
    dest = EPISODES / Path(path).name
    tmp = dest.with_suffix('.incoming')
    tmp.write_bytes(raw); tmp.replace(dest)
    if on_json_saved: on_json_saved(dest.name, len(data['samples']))
    sync_media(dest.name, data['metadata'], progress)
    return dest.name, len(data['samples']), data['metadata'].get('status')


def failure_message(metadata, fallback=''):
    cause = metadata.get('fault') or metadata.get('camera_error') or fallback or '请查看运行日志。'
    if metadata.get('stage') == 'camera_startup' and metadata.get('controller_started') is False:
        return '相机启动失败，机械臂尚未启用。' + str(cause)
    if metadata.get('status') == 'controller_stopped_early' or str(fallback).startswith(('Control loop error:', 'CHECK_FAILED')):
        return '厂家控制器停止，已保留数据。' + str(cause)
    return '采集未完成，已保留记录。' + str(cause)


def console_failure(tail):
    clean = ANSI.sub('', tail).replace('\r', '\n').replace('\x00', '')
    lines = [line.strip() for line in clean.splitlines() if line.strip()]
    if 'Jetson idle shell not confirmed' in clean:
        return '采集未启动：未识别到 Jetson 空闲命令提示符。未发送采集启动命令。请完成 Jetson 登录并退出登录窗口，再点重新准备。'
    errors = [line for line in lines if re.match(r'(?:[\w.]*Error|[\w.]*Exception):', line) or line.startswith(('CHECK_FAILED', 'CAPTURE_ERROR'))]
    reason = errors[-1] if errors else '未收到进程完成标记'
    return '采集连接异常：' + reason[:600] + '。请查看下方诊断详情；机械臂状态需现场确认。'


class Recorder:
    def __init__(self):
        self.lock = threading.RLock()
        self.active = False
        self.state = 'idle'
        self.connected = False
        self.device = '未检查'
        self.message = '先连接 Jetson，或回看已有示范。'
        self.frames = 0
        self.record_started = None
        self.started = None
        self.last_episode = None
        self.remote_saved = None
        self.session_episodes = []
        self.ack_events = []
        self.ack_sequence = 0
        self.gripper_state = None
        self.gripper_target = None
        self.gripper_pending = False
        self.motor_temp = None
        self.usb_connected = None
        self.recording_usb = None  # None outside recording; True once USB_CONNECTED seen during the take.
        self.motor_temps = None; self.motor_temps_at = None  # TASK_TEMP: per-joint MOS/rotor °C, monotonic receipt time
        self.enabled_at = None; self.enabled_until = None  # controller enabled span (first positioning -> finalizing)
        self.discard_pending = False; self.pending_qc = {}
        self.collection = load_collection() if HARDWARE_ENABLED else []
        self.session_complete = None
        self.proc = None
        self.master = None
        self.stop_requested = False
        self.stop_signal_sent = False
        self.finalizing = False
        self.log = []
        self.checks = []
        self.cancelled_prepare = False

    def snapshot(self):
        with self.lock:
            # A lost confirmation must not lock X forever; the Jetson file is the record of truth.
            if self.discard_pending and time.monotonic() - getattr(self, 'discard_sent_at', 0) > 5:
                self.discard_pending = False
                self.message = '未收到排除确认（Jetson 可能已排除）；请告诉助手核对这一条。'
            return {'state': self.state, 'active': self.active, 'connected': self.connected, 'device': self.device,
                    'message': self.message, 'frames': self.frames,
                    'elapsed': max(0,time.monotonic()-self.record_started) if self.record_started else 0,
                    'last_episode': self.last_episode, 'remote_saved': self.remote_saved,
                    'session_episodes': [dict(e) for e in self.session_episodes], 'discard_pending': self.discard_pending,
                    'motor_temps': dict(self.motor_temps, age_s=round(time.monotonic()-self.motor_temps_at,1)) if self.motor_temps else None,
                    'enabled_s': round((self.enabled_until or time.monotonic())-self.enabled_at,1) if self.enabled_at else None, 'ack_events': self.ack_events[-12:], 'gripper_state': self.gripper_state, 'gripper_target': self.gripper_target, 'gripper_pending': self.gripper_pending, 'motor_temp': self.motor_temp, 'collection': [dict(e) for e in self.collection], 'usb_connected': self.usb_connected,
                    'camera_test_available': CAMERA_TEST.is_file(), 'hardware_enabled': HARDWARE_ENABLED, 'video_download_enabled': FETCH_VIDEO, 'camera': 'not_recorded', 'serial_available': bool(glob.glob('/dev/cu.usbserial-*')),
                    'checks': list(self.checks), 'log': self.log[-40:]}

    def update(self, state=None, message=None):
        with self.lock:
            if state: self.state = state
            if message: self.message = message

    def connect(self):
        self.start()

    def start(self):
        with self.lock:
            if self.active or self.state in BUSY:
                return  # Duplicate clicks/reloads cannot start a second controller.
            self.active = True
            self.connected = False
            self.checks = []
            self.frames = 0; self.record_started = None; self.last_episode = None
            self.session_episodes = []; self.gripper_state = None; self.gripper_target = None; self.gripper_pending = False
            self.motor_temp = None; self.usb_connected = None; self.recording_usb = None
            self.motor_temps = None; self.motor_temps_at = None; self.enabled_at = None; self.enabled_until = None
            self.discard_pending = False; self.pending_qc = {}; self.session_complete = None
            self.remote_saved = None; self.stop_requested = False; self.log = []; self.cancelled_prepare = False
            self.stop_signal_sent = False; self.finalizing = False
            self.state = 'checking'; self.message = '自动准备：检查连接、相机、存储和控制接口…'
        threading.Thread(target=self._prepare, daemon=True).start()

    def _prepare(self):
        try:
            # Wait for a powered-off/booting board to enumerate without sending input.
            deadline = time.monotonic() + 180
            while not glob.glob('/dev/cu.usbserial-*'):
                self.update('checking', '等待 Jetson 上电或 USB-C 连接；检测到设备后自动继续…')
                if time.monotonic() >= deadline:
                    raise RuntimeError('仍未检测到 Jetson 串口，请检查电源和 USB-C 线后点重新准备。')
                time.sleep(2)
            info = json.loads(serial_rpc(check_source(), 25))
            with self.lock:
                self.connected = True; self.device = info.get('host', 'jetson01')
                self.checks = info.get('checks', [])
                self.log.extend(item['label'] + '：' + item['detail'] for item in self.checks)
            if not info.get('ok'):
                raise RuntimeError(info.get('error', '准备检查未通过。'))
            self.update('starting', '基础检查通过，正在打开采集入口；电机尚未启用…')
            self._run()
        except Exception as e:
            with self.lock:
                self.log.append(str(e)); self.active = False
            self.update('error', console_failure(str(e)) if 'idle shell not confirmed' in str(e) else str(e))

    def cancel_prepare(self):
        with self.lock:
            self.command('ready', b'CANCEL\r', 'finalizing', '已取消本轮准备，未启用电机。')
            self.cancelled_prepare = True

    def command(self, expected, payload, state, message):
        with self.lock:
            if self.state not in ({expected} if isinstance(expected, str) else expected) or self.master is None:
                raise RuntimeError('当前阶段不接受此操作，请等待状态更新。')
            os.write(self.master, payload)
            self.state = state
            self.message = message

    def ready(self):
        self.command('ready', b'READY\r', 'starting', '已确认臂体有支撑、夹爪已清空，正在启用厂家重力辅助并张开夹爪…')

    def begin(self):
        self.command({'positioning','support'}, b'BEGIN\r', 'starting_recording', '正在开始任务录制…')

    def stop(self):
        with self.lock:
            if self.state in {'stopping','support','finalizing','syncing','saved'}: return
        self.command('recording', b'END\r', 'stopping', '正在结束记录；厂家重力辅助保持启用。')

    def discard(self):
        with self.lock:
            if self.discard_pending: raise RuntimeError('正在等待排除确认。')
            if not self.session_episodes or self.session_episodes[-1].get('excluded'): raise RuntimeError('没有可排除的上一条。')
            self.command('support', b'DISCARD\r', 'support', '正在排除上一条…')
            self.discard_pending = True; self.discard_sent_at = time.monotonic()

    def disable(self):
        self.command({'positioning','support'}, b'DISABLE\r', 'finalizing', '已确认臂体有支撑，正在失能并保存视频。')
        self.finalizing = True

    def gripper(self, opened):
        with self.lock:
            if self.gripper_pending: raise RuntimeError('正在等待夹爪命令确认。')
            self.command({'positioning','recording','support'}, b'OPEN\r' if opened else b'CLOSE\r', self.state, '正在张开夹爪…' if opened else '正在闭合夹爪…')
            self.gripper_target = 'open' if opened else 'closed'
            self.gripper_pending = True

    def toggle_gripper(self):
        with self.lock:
            if self.gripper_target not in {'open','closed'}: raise RuntimeError('尚未收到夹爪初始状态，请等待就绪。')
            self.gripper(self.gripper_target == 'closed')

    def json_saved(self, name, count):
        with self.lock:
            self.last_episode = name
            self.frames = count

    @staticmethod
    def json_after(tag, line):
        at = line.find(tag + ' ')
        if at < 0: return None
        brace = line.find('{', at)
        if brace < 0: return None
        try: return json.JSONDecoder().raw_decode(line, brace)[0]  # tolerate text glued after the object
        except ValueError: return None

    def save_collection(self):
        if not HARDWARE_ENABLED: return  # unit tests must never touch the real collection file
        try: save_collection(self.collection)
        except OSError as e: self.log.append(f'collection.json not saved: {e}')

    def attach_qc(self, episode, qc):
        episode['qc'] = {k: v for k, v in qc.items() if k != 'path'}
        if isinstance(qc.get('usb_inserted'), bool): episode['usb_inserted'] = qc['usb_inserted']

    def acknowledge(self, text):
        self.ack_sequence += 1
        self.ack_events.append({'id':self.ack_sequence, 'text':text})
        self.ack_events = self.ack_events[-12:]

    def observe_line(self, line):
        phases = {
            'ready': ('ready', '启动时夹爪会自动张开，请先移除夹持物并避开夹爪。确认臂体已支撑后再启用。'),
            'positioning': ('positioning', '厂家重力辅助已启用。到起点按 R 开始；空格切换夹爪开合。'),
            'recording': ('recording', '正在录制任务。完成后点击「结束录制」，此时不会失能。'),
            'support': ('support', '本条关节数据已保存，重力辅助和相机继续运行。手动复位，按 R 立即录下一条；全部完成后支撑并结束本轮。'),
            'finalizing': ('finalizing', '控制已结束，正在整理关节数据和视频。'),
        }
        temp = re.search(r'joint(\d) MOS=([\d.]+)', line)
        if temp:
            with self.lock: self.motor_temp = {'joint': int(temp.group(1)), 'mos_c': float(temp.group(2)), 'at': time.time()}
        if line.startswith('USB_WATCH') and 'new_since_startup' in line:
            with self.lock:
                if self.usb_connected is None: self.usb_connected = False  # baseline watcher starts disconnected by definition
        usb = re.search(r'\bUSB_(CONNECTED|REMOVED)\b', line)
        if usb:
            with self.lock:
                self.usb_connected = usb.group(1) == 'CONNECTED'
                if self.usb_connected and self.recording_usb is not None: self.recording_usb = True
        marker = re.search(r'TASK_PHASE (ready|positioning|recording|support|finalizing)', line)
        if marker:
            phase = marker.group(1)
            if phase in phases:
                with self.lock:
                    previous = self.state
                    self.state, self.message = phases[phase]
                    if previous != phase and phase in {'positioning','recording','support'}:
                        self.acknowledge({'positioning':'已就绪','recording':'开始录制','support':'本条已结束，可以复位'}[phase])
                    if phase == 'recording': self.frames = 0; self.recording_usb = False
                    if phase == 'positioning' and self.enabled_at is None: self.enabled_at = time.monotonic()
                    if phase == 'finalizing' and self.enabled_at and not self.enabled_until: self.enabled_until = time.monotonic()
                    self.record_started = time.monotonic() if phase == 'recording' else None
                    self.finalizing = phase == 'finalizing'
        marker = re.search(r'TASK_EPISODE (\S+) (\d+)', line)
        if marker:
            parts = ['TASK_EPISODE', marker.group(1), marker.group(2)]
            if len(parts) == 3:
                with self.lock:
                    episode = {'path':parts[1], 'frames':int(parts[2])}
                    # Mirrors collect_task's usb_inserted: connected at any point during the take or at its end; unknown without a watcher.
                    if self.recording_usb or self.usb_connected is True: episode['usb_inserted'] = True
                    elif self.usb_connected is not None: episode['usb_inserted'] = False
                    self.recording_usb = None
                    if not any(e['path'] == parts[1] for e in self.session_episodes):
                        self.session_episodes.append(episode)
                        if parts[1] in self.pending_qc: self.attach_qc(episode, self.pending_qc.pop(parts[1]))
                        # Same dict object, so later QC / discard updates land in both lists.
                        self.collection = [e for e in self.collection if e['path'] != parts[1]] + [episode]
                        self.save_collection()
        qc = self.json_after('TASK_QC', line)
        if isinstance(qc, dict) and qc.get('path'):
            with self.lock:
                episode = next((e for e in self.session_episodes if e['path'] == qc['path']), None)
                if episode: self.attach_qc(episode, qc); self.save_collection()
                else: self.pending_qc[qc['path']] = qc
        temps = self.json_after('TASK_TEMP', line)
        if isinstance(temps, dict) and isinstance(temps.get('mos'), list):
            with self.lock:
                self.motor_temps = {'mos': temps['mos'], 'rotor': temps.get('rotor') if isinstance(temps.get('rotor'), list) else None,
                                    'gripper': temps.get('gripper') if isinstance(temps.get('gripper'), dict) else None}
                self.motor_temps_at = time.monotonic()
        marker = re.search(r'TASK_DISCARDED (\S+)', line)
        if marker:
            with self.lock:
                episode = next((e for e in reversed(self.session_episodes) if marker.group(1).startswith(e['path'])), None)
                if episode: episode['excluded'] = True; self.save_collection()
                self.discard_pending = False
                if self.state == 'support': self.message = '已排除上一条（Jetson 保留原始文件）。'
        if re.search(r'TASK_IGNORED DISCARD', line):
            with self.lock:
                self.discard_pending = False
                if self.state == 'support': self.message = 'Jetson 未接受排除指令。'
        marker = re.search(r'TASK_GRIPPER (open|closed)', line)
        if marker:
            state = marker.group(1)
            if state in {'open','closed'}:
                with self.lock:
                    self.gripper_state = state
                    self.gripper_target = state
                    self.gripper_pending = False
                    self.acknowledge('夹爪已张开' if state == 'open' else '夹爪已闭合')
        marker = re.search(r'TASK_SESSION_COMPLETE (\d+)', line)
        if marker:
            with self.lock: self.session_complete = int(marker.group(1))
        marker = re.search(r'TASK_FRAMES (\d+)', line)
        if marker:
            with self.lock: self.frames = int(marker.group(1))

    def _run(self):
        remote = None; remotes = []; failed = None; answered = False; done = False; linebuf = ''; tail = ''
        master = slave = None
        decoder = codecs.getincrementaldecoder('utf-8')(errors='replace')
        try:
            with self.lock:
                if self.stop_requested:
                    self.update('idle', '已取消启动，没有开始采集。')
                    return
            master, slave = pty.openpty()
            process = subprocess.Popen([sys.executable, str(EXPERIMENT/'mac_console.py')],
                                       stdin=slave, stdout=slave, stderr=slave, start_new_session=True,
                                       env={**os.environ, 'A1Z_MODE':'task'})
            os.close(slave); slave = None
            with self.lock: self.master = master; self.proc = process
            while True:
                if select.select([master], [], [], .15)[0]:
                    try: chunk = os.read(master, 65536)
                    except OSError: break
                    if not chunk: break
                    text = decoder.decode(chunk)
                    tail = (tail + text)[-12000:]
                    clean = ANSI.sub('', text).replace('\x00','').replace('\r','\n')
                    linebuf += clean
                    lines = linebuf.split('\n'); linebuf = lines.pop()
                    for line in lines:
                        line = line.strip()
                        if not line: continue
                        self.observe_line(line)
                        if any(k in line for k in ['PREFLIGHT_OK','ARMED','RECORDING','SAVED','error:','CHECK_FAILED','STOP','静止','FINALIZING:','TASK_PHASE','TASK_FRAMES','TASK_EPISODE','TASK_GRIPPER','CAPTURE_ERROR','Traceback','Error:','Exception:']):
                            with self.lock: self.log.append(line[:500])
                        if line.startswith('SAVED '):
                            remote = line[6:].strip()
                            if remote not in remotes: remotes.append(remote)
                        if line.startswith(('Control loop error:', 'CHECK_FAILED', 'CAPTURE_ERROR')): failed = line
                        if line.startswith('__A1Z_TEACH_FINISHED__ '):
                            done = True
                            if line.split()[-1] != '0' and not failed: failed = '远端录制进程异常退出：' + line
                if done or process.poll() is not None: break
            if done and process.poll() is None:
                process.wait(timeout=8)
            if not done:
                with self.lock:
                    self.log.extend(ANSI.sub('', tail).replace('\r', '\n').replace('\x00','').strip().splitlines()[-30:])
                raise RuntimeError(console_failure(tail))
            process.wait(timeout=5)
            with self.lock:
                self.master = None; self.proc = None; self.remote_saved = remote
            os.close(master); master = None
            # Our wrapper printed its finished marker, so the foreground is our own shell. The first redraw
            # after it exits is often swallowed; only here (known foreground) is a retried handshake allowed.
            if remote:
                settle_shell()
                MEDIA_PAUSE.clear()
                self.update('syncing', '机械臂已结束控制，正在把数据传回 Mac…')
                imported = []
                for saved_path in remotes:
                    imported.append(remote_episode(saved_path, lambda message:self.update(message=message), self.json_saved))
                name, count, status = imported[-1]
                if any(item[2] in {'failed','controller_stopped_early'} for item in imported): failed = failed or '本轮部分记录未完成。'
                with self.lock: self.last_episode = name; self.frames = count; self.record_started = None
                if failed or status in {'failed','controller_stopped_early'}:
                    metadata=json.loads((EPISODES/name).read_text()).get('metadata',{})
                    self.update('error', failure_message(metadata, failed or ''))
                elif count:
                    media=local_media(name)
                    camera_error=json.loads((EPISODES/name).read_text()).get('metadata',{}).get('camera_error')
                    suffix='；视频传输未完成，可稍后同步。' if media and media.get('error') else '，可以回看。'
                    if not FETCH_VIDEO: suffix='；视频保留在 Jetson，不下载到 Mac。'
                    if camera_error: suffix='；关节数据已保存，相机整理失败：'+str(camera_error)
                    self.update('saved', f'已保存 {len(imported)} 条，共 {sum(item[1] for item in imported)} 帧'+suffix)
                else:
                    self.update('idle', '没有检测到动作，本轮没有轨迹。')
            elif self.cancelled_prepare:
                self.update('idle', '已取消准备；电机未启用。')
            elif not failed and self.session_complete == 0:
                self.update('idle', '本轮没有录制，已正常结束；电机已失能。')
            else:
                self.update('error', failed or '未收到保存路径，请检查 Jetson。')
        except MediaTransferPaused as e:
            self.update('saved', str(e))
        except Exception as e:
            with self.lock:
                if 'Jetson idle shell not confirmed' not in tail: self.connected = False
                self.log.append(str(e))
            self.update('error', str(e))
        finally:
            if slave is not None: os.close(slave)
            with self.lock:
                proc = self.proc
                if self.master is not None: master = self.master
            if proc is not None and proc.poll() is None:
                # Keep live controller intact; UI cleanup must not send SIGINT/disable.
                self.update('error', '控制进程仍在运行，连接保持锁定；请现场确认状态。未发送失能或中断。')
                return
            if master is not None:
                try: os.close(master)
                except OSError: pass
            with self.lock: self.master = None; self.proc = None; self.record_started = None; self.active = False

    def pause_sync(self):
        with self.lock:
            if self.state != 'syncing': raise RuntimeError('当前没有视频传输。')
            MEDIA_PAUSE.set()
            self.message = '完成当前传输块后暂停；关节数据保留，不会中断远端。'

    def sync(self):
        with self.lock:
            if self.active or self.state in BUSY: raise RuntimeError('采集期间不能同步。')
            self.active = True
            MEDIA_PAUSE.clear()
            self.state = 'syncing'; self.message = '正在同步 Jetson 上的示范…'
        threading.Thread(target=self._sync, daemon=True).start()

    def _sync(self):
        try:
            out = serial_rpc("import json\nfrom pathlib import Path\np=Path('/home/dimos/a1z-experiments/2026-10-02/episodes')\nprint(json.dumps([str(f) for f in sorted(p.glob('*-teaching.json'))][-20:]))", 10)
            imported = 0
            for path in json.loads(out):
                check_media_pause()
                if not (EPISODES/Path(path).name).exists():
                    remote_episode(path, lambda message:self.update(message=message)); imported += 1
                else:
                    original=json.loads((EPISODES/Path(path).name).read_text())
                    if (original['metadata'].get('camera') or successful_dataset(original['metadata'])) and (not local_media(Path(path).name) or local_media(Path(path).name).get('error') or (successful_dataset(original['metadata']) and local_media(Path(path).name).get('source')!='lerobot')):
                        sync_media(Path(path).name,original['metadata'],lambda message:self.update(message=message))
            self.update('idle', f'同步完成，新增 {imported} 条记录。')
        except MediaTransferPaused as e:
            self.update('saved', str(e))
        except Exception as e:
            with self.lock: self.connected = False
            self.update('error', str(e))
        finally:
            with self.lock: self.active = False


RECORDER = Recorder()


class Handler(http.server.BaseHTTPRequestHandler):
    def log_message(self, *args): pass

    def send(self, data, content_type='application/json', code=200):
        if isinstance(data, (dict,list)): data = json.dumps(data, ensure_ascii=False, allow_nan=False).encode()
        elif isinstance(data,str): data=data.encode()
        self.send_response(code); self.send_header('Content-Type',content_type)
        self.send_header('Content-Length',str(len(data))); self.send_header('Cache-Control','no-store')
        self.send_header('X-Content-Type-Options','nosniff'); self.end_headers(); self.wfile.write(data)

    def valid_host(self):
        return self.headers.get('Host') in {f'127.0.0.1:{PORT}',f'localhost:{PORT}'}

    def do_GET(self):
        if not self.valid_host(): return self.send({'error':'Invalid host'},code=403)
        path = urllib.parse.unquote(urllib.parse.urlparse(self.path).path)
        try:
            if path in {'/api/camera-test', '/api/camera-test-video'}:
                if not CAMERA_TEST.is_file(): return self.send({'error':'No local camera test'},code=404)
                return self.send(CAMERA_TEST.read_bytes(),'video/mp4')
            if path.startswith('/audio/'):
                if path not in {'/audio/'+name+'.wav' for name in ['ready','recording','ended','opening','closing']}: return self.send({'error':'Not found'},code=404)
                audio_file = STATIC / path.lstrip('/')
                if not audio_file.is_file(): return self.send({'error':'Audio not ready'},code=404)
                return self.send(audio_file.read_bytes(),'audio/wav')
            if path=='/api/status': return self.send(RECORDER.snapshot())
            if path=='/api/episodes': return self.send(episode_index())
            if path.startswith('/api/media/'):
                parts=path.removeprefix('/api/media/').split('/')
                if len(parts)!=2 or parts[0] not in {x['id'] for x in episode_index()} or parts[1] not in {'wrist.mp4','timestamps.json','unified-wrist.mp4','unified-timestamps.json','playback.json','info.json','alignment-report.json'}:
                    return self.send({'error':'Not found'},code=404)
                media=media_dir(parts[0])/parts[1]
                if not media.is_file(): return self.send({'error':'Not found'},code=404)
                return self.send(media.read_bytes(),'video/mp4' if parts[1].endswith('.mp4') else 'application/json')
            if path.startswith('/api/episode/'):
                name=path.removeprefix('/api/episode/')
                if name not in {x['id'] for x in episode_index()}: return self.send({'error':'Not found'},code=404)
                if urllib.parse.parse_qs(urllib.parse.urlparse(self.path).query).get('raw')==['1']:
                    return self.send((EPISODES/name).read_bytes())
                data=episode_data(name)
                return self.send(data)
            if path=='/': return self.send((STATIC/'index.html').read_text().replace('__TOKEN__',TOKEN),'text/html; charset=utf-8')
            name=path.lstrip('/')
            types={'audio.js':'text/javascript','app.js':'text/javascript','capture-controls.js':'text/javascript','style.css':'text/css','kinematics.json':'application/json'}
            if name in types: return self.send((STATIC/name).read_bytes(),types[name]+'; charset=utf-8')
            return self.send({'error':'Not found'},code=404)
        except Exception as e: self.send({'error':str(e)},code=500)

    def do_POST(self):
        if not self.valid_host() or self.headers.get('X-A1Z-Token')!=TOKEN:
            return self.send({'error':'Forbidden'},code=403)
        origin=self.headers.get('Origin')
        if origin and origin not in {f'http://127.0.0.1:{PORT}',f'http://localhost:{PORT}'}:
            return self.send({'error':'Forbidden origin'},code=403)
        actions={'/api/connect':RECORDER.connect,'/api/cancel-prepare':RECORDER.cancel_prepare,'/api/start':RECORDER.start,'/api/stop':RECORDER.stop,'/api/ready':RECORDER.ready,'/api/begin':RECORDER.begin,'/api/disable':RECORDER.disable,'/api/discard':RECORDER.discard,'/api/sync':RECORDER.sync,'/api/sync/pause':RECORDER.pause_sync,'/api/gripper/toggle':RECORDER.toggle_gripper,'/api/gripper/open':lambda:RECORDER.gripper(True),'/api/gripper/close':lambda:RECORDER.gripper(False)}
        try:
            if self.path not in actions: return self.send({'error':'Not found'},code=404)
            if not HARDWARE_ENABLED: return self.send({'error':'硬件采集入口尚未启用；当前可回看本地记录。'},code=409)
            actions[self.path](); self.send({'ok':True})
        except Exception as e: self.send({'error':str(e)},code=409)


PREVIEW_PORT = 8771


def preview_tunnel():
    """Keep the Jetson preview server running and forwarded to 127.0.0.1:8771 over the direct Ethernet link.
    Read-only camera view; it never touches the serial console or the robot."""
    remote = '/home/dimos/a1z-experiments/2026-10-02/preview_server.py'
    ssh = ['ssh', '-o', 'BatchMode=yes', '-o', 'ConnectTimeout=5']
    while True:
        try:
            subprocess.run(ssh + ['jetson-direct', f'curl -s -m 1 http://127.0.0.1:{PREVIEW_PORT}/status >/dev/null || (nohup python3 {remote} >/dev/null 2>&1 &)'],
                           timeout=20, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            subprocess.run(ssh + ['-o', 'ExitOnForwardFailure=yes', '-o', 'ServerAliveInterval=5', '-o', 'ServerAliveCountMax=2',
                                  '-N', '-L', f'{PREVIEW_PORT}:127.0.0.1:{PREVIEW_PORT}', 'jetson-direct'],
                           stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        except Exception:
            pass
        time.sleep(3)


if __name__=='__main__':
    EPISODES.mkdir(exist_ok=True)
    if HARDWARE_ENABLED and os.environ.get('A1Z_PREVIEW', '1') == '1':
        threading.Thread(target=preview_tunnel, daemon=True).start()
    print(f'A1Z Studio http://127.0.0.1:{PORT}',flush=True)
    httpd = http.server.ThreadingHTTPServer(('127.0.0.1',PORT),Handler)
    if HARDWARE_ENABLED: RECORDER.start()
    httpd.serve_forever()
