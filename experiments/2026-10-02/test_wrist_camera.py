#!/usr/bin/env python3
"""Offline check of wrist_camera.pick_wrist: the wrist D405 is chosen even with a side D455/D435 attached."""
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parent))
from wrist_camera import pick_wrist,WRIST_SERIAL

def main():
    side=('Intel RealSense D455','261822303805');wrist=('Intel RealSense D405',WRIST_SERIAL)
    assert pick_wrist([side,wrist])==WRIST_SERIAL and pick_wrist([wrist,side])==WRIST_SERIAL  # order does not matter
    assert pick_wrist([('Intel RealSense D435','111'),('Intel RealSense D405','999')])=='999'  # another D405 alone
    assert pick_wrist([side,('Intel RealSense D405','999'),wrist])==WRIST_SERIAL               # two D405: the known one
    for devs in ([side],[],[side,('Intel RealSense D405','1'),('Intel RealSense D405','2')]):
        try:pick_wrist(devs);raise AssertionError(devs)
        except RuntimeError:pass
    assert pick_wrist([side,wrist],override='261822303805')=='261822303805'
    try:pick_wrist([wrist],override='nope');raise AssertionError('override')
    except RuntimeError:pass
    print('wrist camera OK')

if __name__=='__main__':main()
