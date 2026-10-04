#!/usr/bin/env python3
"""Offline test of the HG-DAgger export: a synthetic --hil session (flight log, trial json, H.264 video from the
real StreamingVideo encoder) must yield exactly the operator stretches, labelled like the demonstrations."""
import json
from pathlib import Path
import subprocess
import sys
import tempfile

import numpy as np

ROOT=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT))


def make_session(d):
    from camera_session import StreamingVideo
    d.mkdir(parents=True);t0=1000.0;fps=30;T=6.0
    # policy 0-2 s, human 2-4 s (still for the first 0.3 s), policy 4-5 s, human 5-6 s (discarded trial? no: kept)
    segs=[('policy',0,2),('human',2,4),('policy',4,5),('human',5,6)]
    with open(d/'flight.jsonl','w') as f:
        for k in range(int(T*50)+1):
            t=t0+k*0.02;s=k*0.02
            ctrl=next(c for c,a,b in segs if a<=s<=b)
            moving=s-2>=0.3 if 2<=s<=4 else True
            q=[0.1+0.05*s if moving else 0.1+0.05*2.0]+[0.5]*5
            if 2<=s<2.3:q=[0.1+0.05*2.0]+[0.5]*5
            f.write(json.dumps({'t':round(t,4),'mode':0 if ctrl=='human' else 1,'q':q,'dq':[0]*6,'eff':[0]*6,'grip':0.01,'grip_cmd':0.0})+'\n')
        f.write(json.dumps({'t':t0+1,'sdk':'WARNING','msg':'x'})+'\n')
    v=StreamingVideo(d/'trial-01.mp4');frames=[]
    for k in range(int(T*fps)+3):
        img=np.full((480,640,3),k%255,np.uint8);v.write(img.tobytes());frames.append({'host_monotonic_s':t0-0.03+k/fps})
    v.close()
    (d/'trial-01.timestamps.json').write_text(json.dumps({'frames':frames,'encoder_error':None}))
    trial={'n':1,'hil':True,'epoch':1,'checkpoint':'ck','success':True,'end_reason':'usb_inserted','hil_keep':True,
           'segments':[{'controller':c,'t0':t0+a,'t1':t0+b} for c,a,b in segs]}
    (d/'trial-01.json').write_text(json.dumps(trial))
    t2=dict(trial,n=2,hil_keep=False);(d/'trial-02.json').write_text(json.dumps(t2))  # discarded: must be skipped


def main():
    import export_hil
    with tempfile.TemporaryDirectory() as tmp:
        s=Path(tmp)/'hil-runs'/'S1';make_session(s)
        class A:fps=30;idle_rad=0.003;idle_max_s=1.5;min_frames=10
        eps,rep=export_hil.collect([s],A)
        assert rep['trials']==2 and rep['interventions']==4,rep
        assert len(eps)==2,len(eps)  # trial 2 discarded
        assert rep['skipped'][0]['reason']=='operator discarded'
        (_,r1,i1),(_,r2,i2)=eps
        assert 50<=len(r1)<=53 and i1['trimmed_idle_frames']>=8,(len(r1),i1)  # 2 s minus ~0.3 s still
        assert 28<=len(r2)<=30,len(r2)
        for rows in (r1,r2):
            st=np.stack([r['observation.state'] for r in rows]);ac=np.stack([r['action'] for r in rows])
            assert np.allclose(ac[:-1,:6],st[1:,:6],atol=1e-6)  # action = next aligned joint position
            assert np.all(ac[:,6]==0.0) and np.allclose(st[:,6],0.01)
        out=Path(tmp)/'ds'
        r=subprocess.run([sys.executable,str(ROOT/'export_hil.py'),str(s),'--output',str(out)],capture_output=True,text=True)
        assert r.returncode==0,r.stderr[-2000:]
        rep=json.loads((out/'source-report.json').read_text());assert len(rep['episodes'])==2
        print('export_hil OK',len(r1),len(r2),'frames; trimmed',i1['trimmed_idle_frames'])

if __name__=='__main__':main()
