#!/usr/bin/env python3
"""Local A1Z recorder UI. Reuses the verified CLI controller; never replays motors."""
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
import subprocess
import sys
import termios
import threading
import time
import urllib.parse
import uuid
import zlib

ROOT = Path(__file__).resolve().parent
EXPERIMENT = ROOT.parent
EPISODES = EXPERIMENT / 'episodes'
STATIC = ROOT / 'static'
CAMERA_TEST = EXPERIMENT / 'diagnostics' / 'lerobot-camera-test' / 'file-000.mp4'
PORT = int(os.environ.get('A1Z_STUDIO_PORT', '8769'))
TOKEN = secrets.token_urlsafe(32)
HARDWARE_ENABLED = os.environ.get('A1Z_STUDIO_HARDWARE') == '1'
ANSI = re.compile(r'\x1b\[[0-?]*[ -/]*[@-~]')
BUSY = {'checking', 'starting', 'waiting', 'recording', 'stopping', 'finalizing', 'syncing'}
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
        # A local lock does not prove the remote foreground process is idle.
        # Ctrl-L requests a shell redraw without sending Enter or Ctrl-C.
        # A foreground input prompt cannot accept this as a start command.
        while select.select([fd], [], [], 0)[0]:
            if not os.read(fd, 65536): raise RuntimeError('Jetson 串口已断开。')
        os.write(fd, b'\x0c')
        observed = b''
        deadline = time.monotonic() + 2
        while time.monotonic() < deadline:
            if select.select([fd], [], [], .1)[0]:
                chunk = os.read(fd, 65536)
                if not chunk:
                    raise RuntimeError('Jetson 串口已断开。')
                observed = (observed + chunk)[-8192:]
                clean = ANSI.sub('', observed.decode(errors='replace')).replace('\r', '').replace('\x00', '')
                if re.search(r'(?:^|\n)[ \t]*dimos@jetson01[^\n]*[\$%] $', clean):
                    break
        else:
            raise RuntimeError('没有观察到 Jetson 空闲 shell 提示符；为避免打断前台任务，本次未发送命令或回车。')
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
        data['raw_sample_count']=len(data['samples'])
        data['samples']=playback['samples']
        data['playback_basis']='LeRobot 30 Hz 对齐观测'
    return data


def fetch_remote_file(remote, dest, progress=None):
    # All callers validate and constrain remote paths before arriving here.
    out = serial_rpc("from pathlib import Path\nimport json,hashlib\n" +
                     f"p=Path({remote!r})\n" +
                     "print(json.dumps({'bytes':p.stat().st_size,'sha256':hashlib.sha256(p.read_bytes()).hexdigest()}))", 45)
    info = json.loads(out)
    size = info['bytes']
    if not isinstance(size,int) or not 0 < size <= 128*1024*1024:
        raise RuntimeError('媒体文件为空或超过 128 MB 单文件传输上限。')
    tmp = dest.with_suffix(dest.suffix+'.incoming')
    digest = hashlib.sha256()
    with tmp.open('wb') as writer:
        for offset in range(0,size,65536):
            code = ("import base64\n" + f"with open({remote!r},'rb') as f:\n f.seek({offset})\n print(base64.b64encode(f.read({min(65536,size-offset)})).decode())")
            block = base64.b64decode(serial_rpc(code,45),validate=True)
            if len(block) != min(65536,size-offset): raise RuntimeError('媒体传输长度不符，远端原文件保留。')
            writer.write(block); digest.update(block)
            if progress: progress(f'录制已结束，正在传回 {dest.name}：{min(offset+len(block),size)*100//size}%')
    if digest.hexdigest() != info['sha256']:
        raise RuntimeError('媒体 SHA256 校验不符，远端原文件保留。')
    tmp.replace(dest)
    return info


def sync_media(name, metadata, progress=None):
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
                except Exception as e: result.setdefault('inspection_warnings',[]).append(str(e))
        result['complete']=True
        return result

    if canonical:
        try: manifest.update(download_source(canonical,True))
        except Exception as e:
            manifest['export_error']='LeRobot 本地回放整理未完成：'+str(e)
            if progress: progress(manifest['export_error']+'；尝试保留原始视频回放。')
    if not manifest.get('complete') and isinstance(camera,dict) and camera.get('dataset_dir'):
        try: manifest.update(download_source(camera,False))
        except Exception as e:
            manifest['error']=str(e)
            if progress: progress('关节记录已保存；视频传输未完整完成：'+str(e))
    if not manifest.get('complete'): manifest.setdefault('error',manifest.get('export_error') or '媒体尚未完整传回 Mac')
    temp=folder/'manifest.incoming';temp.write_text(json.dumps(manifest,ensure_ascii=False,indent=2));temp.replace(folder/'manifest.json')


def remote_episode(path, progress=None):
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
    sync_media(dest.name, data['metadata'], progress)
    return dest.name, len(data['samples']), data['metadata'].get('status')


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
        self.proc = None
        self.master = None
        self.stop_requested = False
        self.stop_signal_sent = False
        self.finalizing = False
        self.log = []

    def snapshot(self):
        with self.lock:
            return {'state': self.state, 'active': self.active, 'connected': self.connected, 'device': self.device,
                    'message': self.message, 'frames': self.frames,
                    'elapsed': max(0,time.monotonic()-self.record_started) if self.record_started else 0,
                    'last_episode': self.last_episode, 'remote_saved': self.remote_saved,
                    'camera_test_available': CAMERA_TEST.is_file(), 'hardware_enabled': HARDWARE_ENABLED, 'camera': 'not_recorded', 'serial_available': bool(glob.glob('/dev/cu.usbserial-*')),
                    'log': self.log[-8:]}

    def update(self, state=None, message=None):
        with self.lock:
            if state: self.state = state
            if message: self.message = message

    def connect(self):
        with self.lock:
            if self.active or self.state in BUSY: raise RuntimeError('当前任务尚未结束。')
            self.active = True
            self.state = 'checking'
            self.message = '正在检查 Jetson 与录制入口…'
        threading.Thread(target=self._connect, daemon=True).start()

    def _connect(self):
        try:
            out = serial_rpc("import json,socket\nfrom pathlib import Path\nprint(json.dumps({'host':socket.gethostname(),'can':Path('/sys/class/net/can1').exists(),'recorder':Path('/home/dimos/a1z-experiments/2026-10-02/teach.py').exists()}))", 10)
            info = json.loads(out)
            if not (info['can'] and info['recorder']):
                raise RuntimeError('Jetson 已连接，但 CAN 或录制入口不可用。')
            with self.lock:
                self.connected = True; self.device = info['host']
            self.update('idle', '已连接。开始后等待动作，静止 2 秒自动保存。')
        except Exception as e:
            with self.lock: self.connected = False
            self.update('error', str(e))
        finally:
            with self.lock: self.active = False

    def start(self):
        with self.lock:
            if self.active or self.state in BUSY: raise RuntimeError('已经在采集中，请勿重复开始。')
            if not self.connected: raise RuntimeError('请先连接 Jetson。')
            self.active = True
            self.frames = 0; self.record_started = None; self.last_episode = None
            self.remote_saved = None; self.stop_requested = False; self.log = []
            self.stop_signal_sent = False; self.finalizing = False
            self.state = 'starting'; self.message = '检查六关节反馈…'
        threading.Thread(target=self._run, daemon=True).start()

    def _interrupt_acquisition(self):
        """Caller holds self.lock. Never send twice or interrupt finalization."""
        if self.finalizing or self.stop_signal_sent or self.master is None:
            return
        self.stop_signal_sent = True
        os.write(self.master, b'\x03')

    def stop(self):
        with self.lock:
            if self.finalizing or self.state in {'finalizing','syncing','saved'} or self.stop_requested:
                return
            if self.state not in {'starting','waiting','recording'}:
                raise RuntimeError('当前没有正在运行的采集。')
            self.stop_requested = True
            self.state = 'stopping'; self.message = '正在结束采集，随后保存数据与视频。'
            self._interrupt_acquisition()

    def observe_finalization(self, line):
        if line.startswith('FINALIZING:') or ('连续静止' in line and '保存并停用' in line):
            with self.lock:
                self.finalizing = True
                self.state = 'finalizing'
                self.message = '动作采集已结束，正在保存关节数据并整理视频，请等待完成。'
                self.record_started = None
            return True
        return False

    def _run(self):
        remote = None; failed = None; answered = False; done = False; linebuf = ''; tail = ''
        master = slave = None
        decoder = codecs.getincrementaldecoder('utf-8')(errors='replace')
        try:
            with self.lock:
                if self.stop_requested:
                    self.update('idle', '已取消启动，没有开始采集。')
                    return
            master, slave = pty.openpty()
            process = subprocess.Popen([sys.executable, str(EXPERIMENT/'mac_console.py')],
                                       stdin=slave, stdout=slave, stderr=slave, start_new_session=True)
            os.close(slave); slave = None
            with self.lock: self.master = master; self.proc = process
            end = time.monotonic() + 450
            timeout_stop = False
            while True:
                if time.monotonic() >= end:
                    if self.finalizing:
                        raise RuntimeError('视频整理超过等待时限；未发送中断，后台任务仍锁定，请检查保存进度。')
                    if timeout_stop:
                        raise RuntimeError('采集超时且停止未确认；未重复发送中断，请检查设备状态。')
                    self.stop()
                    timeout_stop = True
                    end = time.monotonic() + 90
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
                        self.observe_finalization(line)
                        if any(k in line for k in ['PREFLIGHT_OK','ARMED','RECORDING','SAVED','error:','CHECK_FAILED','STOP','静止','FINALIZING:']):
                            with self.lock: self.log.append(line[:500])
                        if line.startswith('ARMED') and not self.stop_requested and not self.finalizing:
                            self.update('waiting', '等待动作。开始拖动就会自动录制。')
                        if line.startswith('RECORDING') and not self.stop_requested and not self.finalizing:
                            with self.lock: self.record_started = time.monotonic()
                            self.update('recording', '正在录制。结束时放回支撑，静止 2 秒即保存。')
                        count = re.search(r'已录 (\d+) 帧', line)
                        if count:
                            with self.lock: self.frames = int(count.group(1))
                        if line.startswith('SAVED '): remote = line[6:].strip()
                        if line.startswith(('Control loop error:', 'CHECK_FAILED')): failed = line
                        if line.startswith('__A1Z_TEACH_FINISHED__ '):
                            done = True
                            if line.split()[-1] != '0' and not failed: failed = '远端录制进程异常退出：' + line
                    if not answered and '按回车开始监听' in tail:
                        # Clicking Start is the user's explicit action; no second terminal prompt.
                        with self.lock:
                            if self.stop_requested: self._interrupt_acquisition()
                            elif not self.finalizing: os.write(master,b'\r')
                            answered = True
                if done or process.poll() is not None: break
            if done and process.poll() is None:
                process.wait(timeout=8)
            if not done:
                raise RuntimeError('录制连接提前结束；请保持机械臂有支撑。可重新连接后同步文件。')
            process.wait(timeout=5)
            with self.lock:
                self.master = None; self.proc = None; self.remote_saved = remote
            os.close(master); master = None
            if remote:
                self.update('syncing', '机械臂已结束控制，正在把数据传回 Mac…')
                name, count, status = remote_episode(remote, lambda message:self.update(message=message))
                with self.lock: self.last_episode = name; self.frames = count; self.record_started = None
                if failed or status in {'failed','controller_stopped_early'}:
                    self.update('error', '录制被保护中止，已保留数据。' + (failed or ''))
                elif count:
                    media=local_media(name)
                    camera_error=json.loads((EPISODES/name).read_text()).get('metadata',{}).get('camera_error')
                    suffix='；视频传输未完成，可稍后同步。' if media and media.get('error') else '，可以回看。'
                    if camera_error: suffix='；关节数据已保存，相机整理失败：'+str(camera_error)
                    self.update('saved', f'已保存 {count} 帧'+suffix)
                else:
                    self.update('idle', '没有检测到动作，本轮没有轨迹。')
            else:
                self.update('error', failed or '未收到保存路径，请检查 Jetson。')
        except Exception as e:
            with self.lock: self.connected = False
            self.update('error', str(e))
        finally:
            if slave is not None: os.close(slave)
            with self.lock:
                proc = self.proc
                if self.master is not None: master = self.master
            if proc is not None and proc.poll() is None:
                try:
                    with self.lock:
                        if not done: self._interrupt_acquisition()
                    proc.wait(timeout=8)
                except Exception:
                    # Terminating mac_console forwards another SIGINT remotely.
                    # Preserve process/PTY and the lifecycle gate for inspection.
                    self.update('error', '后台结束尚未确认；未重复发送中断，采集保持锁定，请检查保存进度。')
                    return
            if master is not None:
                try: os.close(master)
                except OSError: pass
            with self.lock: self.master = None; self.proc = None; self.record_started = None; self.active = False

    def sync(self):
        with self.lock:
            if self.active or self.state in BUSY: raise RuntimeError('采集期间不能同步。')
            self.active = True
            self.state = 'syncing'; self.message = '正在同步 Jetson 上的示范…'
        threading.Thread(target=self._sync, daemon=True).start()

    def _sync(self):
        try:
            out = serial_rpc("import json\nfrom pathlib import Path\np=Path('/home/dimos/a1z-experiments/2026-10-02/episodes')\nprint(json.dumps([str(f) for f in sorted(p.glob('*-teaching.json'))][-20:]))", 10)
            imported = 0
            for path in json.loads(out):
                if not (EPISODES/Path(path).name).exists():
                    remote_episode(path, lambda message:self.update(message=message)); imported += 1
                else:
                    original=json.loads((EPISODES/Path(path).name).read_text())
                    if (original['metadata'].get('camera') or successful_dataset(original['metadata'])) and (not local_media(Path(path).name) or local_media(Path(path).name).get('error') or (successful_dataset(original['metadata']) and local_media(Path(path).name).get('source')!='lerobot')):
                        sync_media(Path(path).name,original['metadata'],lambda message:self.update(message=message))
            self.update('idle', f'同步完成，新增 {imported} 条记录。')
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
            types={'app.js':'text/javascript','style.css':'text/css','kinematics.json':'application/json'}
            if name in types: return self.send((STATIC/name).read_bytes(),types[name]+'; charset=utf-8')
            return self.send({'error':'Not found'},code=404)
        except Exception as e: self.send({'error':str(e)},code=500)

    def do_POST(self):
        if not self.valid_host() or self.headers.get('X-A1Z-Token')!=TOKEN:
            return self.send({'error':'Forbidden'},code=403)
        origin=self.headers.get('Origin')
        if origin and origin not in {f'http://127.0.0.1:{PORT}',f'http://localhost:{PORT}'}:
            return self.send({'error':'Forbidden origin'},code=403)
        actions={'/api/connect':RECORDER.connect,'/api/start':RECORDER.start,'/api/stop':RECORDER.stop,'/api/sync':RECORDER.sync}
        try:
            if self.path not in actions: return self.send({'error':'Not found'},code=404)
            if not HARDWARE_ENABLED: return self.send({'error':'硬件采集入口尚未启用；当前可回看本地记录。'},code=409)
            actions[self.path](); self.send({'ok':True})
        except Exception as e: self.send({'error':str(e)},code=409)


if __name__=='__main__':
    EPISODES.mkdir(exist_ok=True)
    print(f'A1Z Studio http://127.0.0.1:{PORT}',flush=True)
    http.server.ThreadingHTTPServer(('127.0.0.1',PORT),Handler).serve_forever()
