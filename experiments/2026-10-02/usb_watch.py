#!/usr/bin/env python3
"""USB insertion success signal: watch for the target device enumerating on the Jetson.

Event time is kernel enumeration time, which lags physical insertion by a few hundred ms.
Target: A1Z_USB_TARGET="vvvv" or "vvvv:pppp" (hex, comma-separated), else usb_target.txt beside
this file. Neither means any device absent from the startup baseline, which requires the plug
to be out of the socket at startup.

Probe: python3 usb_watch.py [seconds]  -> lists devices, then prints insert/remove events.
"""
import os
from pathlib import Path
import sys
import threading
import time

SYSFS=Path('/sys/bus/usb/devices')


def _read(path):
    try:return path.read_text().strip()
    except OSError:return ''


def list_devices(root=SYSFS):
    """{sysfs name: 'vvvv:pppp product'} for real devices (interfaces have no idVendor)."""
    found={}
    for d in root.glob('*'):
        vendor=_read(d/'idVendor')
        if vendor:found[d.name]=f"{vendor}:{_read(d/'idProduct')} {_read(d/'product')}".strip()
    return found


def parse_targets(spec):
    return [t.strip().lower() for t in (spec or '').split(',') if t.strip()]


class UsbWatcher:
    def __init__(self,targets=None,period_s=0.05,root=SYSFS):
        if targets is None:
            spec=os.environ.get('A1Z_USB_TARGET') or _read(Path(__file__).resolve().parent/'usb_target.txt')
            targets=parse_targets(spec)
        self.targets=targets
        self.period_s=period_s;self.root=root
        self.baseline=set() if self.targets else set(list_devices(root))
        self.lock=threading.Lock();self.events=[];self.connected=False;self.error=None
        self._stop=threading.Event();self._thread=None
        self.connected=self._match(list_devices(root))
    def _match(self,devices):
        if self.targets:
            return any(desc.split(' ')[0].startswith(t) for desc in devices.values() for t in self.targets)
        return any(name not in self.baseline for name in devices)
    def describe(self):
        return {'targets':self.targets or None,'mode':'vendor_product' if self.targets else 'new_since_startup',
                'baseline':sorted(self.baseline),'signal':'kernel enumeration; lags physical insertion'}
    def start(self):
        self._thread=threading.Thread(target=self._run,daemon=True);self._thread.start();return self
    def _run(self):
        while not self._stop.wait(self.period_s):
            try:now=self._match(list_devices(self.root))
            except Exception as e:
                self.error=repr(e);return
            if now!=self.connected:
                event={'t_monotonic_s':time.monotonic(),'connected':now}
                with self.lock:self.connected=now;self.events.append(event)
                print('USB_CONNECTED' if now else 'USB_REMOVED',flush=True)
    def events_between(self,t0,t1):
        with self.lock:return [e for e in self.events if t0<=e['t_monotonic_s']<=t1]
    def stop(self):
        self._stop.set()
        if self._thread:self._thread.join(timeout=1)


if __name__=='__main__':
    for name,desc in sorted(list_devices().items()):print(name,desc)
    w=UsbWatcher().start();print('USB_WATCH',w.describe(),'connected' if w.connected else 'not connected',flush=True)
    try:time.sleep(float(sys.argv[1]) if len(sys.argv)>1 else 60)
    except KeyboardInterrupt:pass
    w.stop()
