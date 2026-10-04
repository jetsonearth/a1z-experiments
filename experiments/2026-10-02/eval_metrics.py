#!/usr/bin/env python3
"""Two-level success for live trials: localization (plug entered the port mouth) and insertion (USB enumerated).

Localization = the plug's lowest height went more than --margin mm below the calibrated hub-surface height.
Uses the newest calibration/port-*.json; without one, falls back to estimates from the demos (port = mean plug
pose at USB-connect, surface = +24 mm) and says so. Sessions and exclusions come from eval-runs/labels.json.
Usage: eval_metrics.py [session ...]   (default: every session marked use_for_eval)
"""
import argparse
import glob
import json
from pathlib import Path
import sys

import numpy as np

ROOT=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT))


def references(kin):
    cal=sorted((ROOT/'calibration').glob('port-*.json'))
    if cal:
        s=json.loads(cal[-1].read_text())['samples'];by={}
        for x in s:by.setdefault(x['label'],[]).append(x['p_mm'])
        if 'inserted' in by and 'surface' in by:
            ins=np.mean(by['inserted'],0)
            return {'source':cal[-1].name,'port_xy':ins[:2],'inserted_z':ins[2],'surface_z':np.mean(by['surface'],0)[2]}
    ports=[]
    for p in json.loads((ROOT/'collection.json').read_text())['episodes']:
        if p.get('excluded') or not p.get('usb_inserted'):continue
        f=ROOT/'episodes'/Path(p['path']).name
        if not f.exists():continue
        smp=json.loads(f.read_text())['samples'];i=next((k for k,x in enumerate(smp) if x.get('usb_connected')),None)
        if i is not None:ports.append(kin.fk(smp[i]['q'])[:3,3]*1000)
    port=np.mean(ports,0)
    return {'source':f'ESTIMATE from {len(ports)} demos (no calibration yet)','port_xy':port[:2],'inserted_z':port[2],'surface_z':port[2]+24.0}


def main():
    a=argparse.ArgumentParser();a.add_argument('sessions',nargs='*');a.add_argument('--margin',type=float,default=2.0);a=a.parse_args()
    from insertion import Kin
    kin=Kin();ref=references(kin)
    labels=json.loads((ROOT/'eval-runs'/'labels.json').read_text())['sessions']
    sessions=a.sessions or [k for k,v in labels.items() if v.get('use_for_eval')]
    print(f"reference: {ref['source']}  surface {ref['surface_z']-ref['inserted_z']:+.1f} mm above inserted; localized = lowest < surface - {a.margin} mm")
    for sess in sessions:
        lab=labels.get(sess,{});excl=lab.get('exclude_trials',{});loc=ins=n=0;rows=[]
        for f in sorted(glob.glob(str(ROOT/'eval-runs'/sess/'trial-*.json'))):
            name=Path(f).stem;t=json.loads(Path(f).read_text())
            if name in excl or not t['log']:continue
            P=np.array([kin.fk(e['q'])[:3,3]*1000 for e in t['log']]);i=int(np.argmin(P[:,2]))
            z=P[i,2]-ref['inserted_z'];xy=float(np.hypot(*(P[i,:2]-ref['port_xy'])))
            localized=P[i,2]<ref['surface_z']-a.margin
            n+=1;loc+=localized;ins+=bool(t['success'])
            rows.append(f"  {name}  lowest z {z:+6.1f} mm  xy {xy:5.1f} mm  localized {'Y' if localized else '-'}  inserted {'Y' if t['success'] else '-'}  {t.get('end_reason','')}"
                        +(f"  stages {[s['stage'] for s in t.get('stages',[])]}" if t.get('stages') else ''))
        print(f"{sess}: localization {loc}/{n}  insertion {ins}/{n}"+(f"  (excluded {', '.join(excl)})" if excl else ''))
        print('\n'.join(rows))

if __name__=='__main__':main()
