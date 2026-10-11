#!/usr/bin/env python3
"""30-second recalibration after a regrasp: only key 3 (fully inserted, USB connected) in calib.py, then this script
derives the mouth and surface samples from the hub geometry measured once in a full calibration.

  ~/a1z-infer/eval.sh --calib            # press 3 (once or twice), q q
  ~/a1z-infer/run.sh quick_calib.py      # newest calibration file -> port-<time>-quick.json beside it

Geometry (full calibration 2026-10-09 13:5x, port-20261009-133432.json): mouth 5.45 mm above full insertion, surface
3.64 mm above the mouth. Those are properties of the hub, not of the grasp. The port position and the insertion
orientation come from the key-3 samples, i.e. from the current grasp. Port() still validates the result.
Why key 3 is acceptable now: if the plug slides in the fingers while being pushed home, the slid grasp is the one that
persists afterwards, so the inserted finger pose is the reference the episodes will actually see.
"""
import argparse
import json
from pathlib import Path
import sys

import numpy as np

ROOT=Path(__file__).resolve().parent
CALIB=ROOT.parent/'2026-10-02'/'calibration'
MOUTH_ABOVE_INSERTED=5.45
SURFACE_ABOVE_MOUTH=3.64
SIDE_MM=6.5   # synthetic surface samples beside the port, as in the full calibration (6.4 / 6.9 mm)


def main():
    p=argparse.ArgumentParser();p.add_argument('src',nargs='?',help='calibration file with key-3 samples (default: newest)');a=p.parse_args()
    src=Path(a.src) if a.src else max((f for f in CALIB.glob('port-*.json') if not f.stem.endswith('-quick')),key=lambda f:f.stat().st_mtime)
    d=json.loads(src.read_text())
    ins=[x for x in d['samples'] if x['label']=='inserted' and x.get('usb_connected') and max(x.get('std_mm',[0]))<=1.0]
    if not ins:sys.exit(f'{src}: no steady key-3 sample with USB connected')
    P=np.array([x['p_mm'] for x in ins]);pi=P.mean(0)
    if len(P)>1 and np.max(np.linalg.norm(P-pi,axis=1))>1.5:sys.exit(f'{src}: key-3 samples {np.round(P,1).tolist()} differ by more than 1.5 mm: grasp moved between them')
    q=ins[-1]['q'];zm=pi[2]+MOUTH_ABOVE_INSERTED;zs=zm+SURFACE_ABOVE_MOUTH
    def sample(label,p):return {'label':label,'p_mm':[round(float(v),2) for v in p],'std_mm':[0.0,0.0,0.0],'q':q,'derived':True}
    out={'frame':d.get('frame'),'derived_from':str(src),'geometry_mm':{'mouth_above_inserted':MOUTH_ABOVE_INSERTED,'surface_above_mouth':SURFACE_ABOVE_MOUTH},
         'samples':[sample('surface',[pi[0],pi[1]+SIDE_MM,zs]),sample('surface',[pi[0],pi[1]-SIDE_MM,zs]),sample('mouth',[pi[0],pi[1],zm])]+ins}
    dst=src.with_name(src.stem+'-quick.json');dst.write_text(json.dumps(out,indent=1))
    print(f'{dst}\n  port xy ({pi[0]:.1f}, {pi[1]:.1f})  mouth z {zm:.1f}  surface z {zs:.1f}  from {len(ins)} key-3 sample(s)')
    print('  use it: hilserl.sh <mode> --calib',dst)


if __name__=='__main__':main()
