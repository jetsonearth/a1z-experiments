#!/usr/bin/env python3
"""USB referee for ONE physical port: connected while a device enumerates at exactly this sysfs path (e.g. 1-2.4.2.1,
the target port of the black hub on 2026-10-09). usb_watch.UsbWatcher's port: targets count any device anywhere under
a hub, so a plug pushed into a neighbouring port of the same inner hub would also count as success.

Same interface as UsbWatcher (connected, events, describe, start, stop). Kernel enumeration lags physical insertion.
"""
import os
import threading
import time

SYSFS='/sys/bus/usb/devices'


class ExactUsb:
    def __init__(self,path,period_s=0.05,root=SYSFS):
        self.path=path;self.period_s=period_s;self.root=root
        self.lock=threading.Lock();self.events=[];self.error=None;self._stop=threading.Event();self._thread=None
        self.connected=self._present()

    def _present(self):
        return os.path.exists(os.path.join(self.root,self.path,'idVendor'))

    def describe(self):
        return {'mode':'exact_port','path':self.path,'signal':'kernel enumeration at this port only; lags physical insertion'}

    def start(self):
        self._thread=threading.Thread(target=self._run,daemon=True);self._thread.start();return self

    def _run(self):
        while not self._stop.wait(self.period_s):
            try:now=self._present()
            except Exception as e:self.error=repr(e);return
            if now!=self.connected:
                with self.lock:self.connected=now;self.events.append({'t_monotonic_s':time.monotonic(),'connected':now})
                print('USB_CONNECTED' if now else 'USB_REMOVED',flush=True)

    def events_between(self,t0,t1):
        with self.lock:return [e for e in self.events if t0<=e['t_monotonic_s']<=t1]

    def stop(self):
        self._stop.set()
        if self._thread:self._thread.join(timeout=1)
