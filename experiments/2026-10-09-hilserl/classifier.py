#!/usr/bin/env python3
"""Binary success classifier for the reward (paper appendix B): pretrained ResNet-10 feature extractor (frozen) + a
two-layer MLP, cross-entropy, Adam 3e-4. Default view: the side camera only (the paper's USB task used side_1).

Data: actor.py classify sessions (runs/*-classify/clf-*.npz): cropped 128x128 frames per view, the operator's label
(key s toggles "fully inserted"), and the USB signal recorded alongside but never used for training, so the USB
signal stays an independent referee.

  python classifier.py train runs/<session> [more sessions] --out classifier.pt [--views side]
Validation is the last 20 % of every session in time (neighbouring frames are near-duplicates: a random split would
overstate accuracy). Reported: accuracy, false-positive and false-negative rates, and agreement with USB.

--agree-only (default on): frames where the operator's label and USB disagree are left out of training and of the
label-based validation. They are toggle slips (a key press that went to another window, or the frames between
switching the label off and pulling the plug out). USB never creates a positive label: positives still come only from
the operator. The vs-USB numbers are computed on all validation frames.
"""
import argparse
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

import common
from common import FRAME_KEYS


class Head(nn.Module):
    """Per view: 1x1 conv 512->32 on the 4x4 ResNet-10 map (keeps where things are), flattened; views concatenated;
    then the two-layer MLP."""
    def __init__(self,n_views,hidden=256,dropout=0.1):
        super().__init__()
        self.reduce=nn.ModuleList([nn.Sequential(nn.Conv2d(512,32,1),nn.ReLU()) for _ in range(n_views)])
        self.mlp=nn.Sequential(nn.Linear(n_views*32*16,hidden),nn.LayerNorm(hidden),nn.ReLU(),nn.Dropout(dropout),nn.Linear(hidden,1))

    def forward(self,feats):
        return self.mlp(torch.cat([r(f).flatten(1) for r,f in zip(self.reduce,feats)],1)).squeeze(1)


class RewardClassifier:
    def __init__(self,views=('side',),device='cpu',threshold=0.5):
        import rl_agent
        self.views=tuple(views);self.device=device;self.threshold=threshold
        self.enc=rl_agent.FrozenResNet10().to(device).eval()
        for p in self.enc.parameters():p.requires_grad_(False)
        self.head=Head(len(self.views)).to(device);self.meta={}

    @torch.no_grad()
    def features(self,imgs):
        """imgs: list per view of float (B,3,128,128) in [0,1] on device."""
        return [self.enc(x) for x in imgs]

    @torch.no_grad()
    def prob(self,obs):
        """obs: env observation dict ('image' wrist, 'side' side), uint8 HWC. Returns P(success)."""
        self.head.eval();key={'wrist':'image','side':'side'}
        imgs=[torch.from_numpy(obs[key[v]]).to(self.device).permute(2,0,1).unsqueeze(0).float().div_(255.0) for v in self.views]
        return float(torch.sigmoid(self.head(self.features(imgs)))[0])

    def save(self,path):
        torch.save({'head':self.head.state_dict(),'views':self.views,'threshold':self.threshold,'meta':self.meta},path)

    @classmethod
    def load(cls,path,device='cpu'):
        d=torch.load(path,map_location=device)
        c=cls(d['views'],device,d.get('threshold',0.5));c.head.load_state_dict(d['head']);c.meta=d.get('meta',{});c.head.eval()
        return c


def load_sessions(dirs,views):
    """-> list of (frames dict view->(N,128,128,3), labels (N,), usb (N,)) per recording file."""
    out=[]
    for d in dirs:
        for f in sorted(Path(d).glob('clf-*.npz')):
            z=np.load(f);out.append(({v:z[FRAME_KEYS[v]] for v in views},z['labels'].astype(np.float32),z['usb'].astype(np.float32),str(f)))
    if not out:raise SystemExit(f'no clf-*.npz in {dirs}')
    return out


def split(sessions,val_frac=0.2,agree_only=True):
    tr={'x':[],'y':[]};va={'x':[],'y':[],'usb':[],'agree':[]}
    for frames,y,usb,_ in sessions:
        n=len(y);cut=int(n*(1-val_frac));ok=(y>0.5)==(usb>0.5) if agree_only else np.ones(n,bool)
        m=ok[:cut]
        tr['x'].append({v:a[:cut][m] for v,a in frames.items()});tr['y'].append(y[:cut][m])
        va['x'].append({v:a[cut:] for v,a in frames.items()});va['y'].append(y[cut:]);va['usb'].append(usb[cut:]);va['agree'].append(ok[cut:])
    cat=lambda parts:{v:np.concatenate([p[v] for p in parts]) for v in parts[0]}
    return (cat(tr['x']),np.concatenate(tr['y'])),(cat(va['x']),np.concatenate(va['y']),np.concatenate(va['usb']),np.concatenate(va['agree']))


def _encode_all(clf,frames,views,chunk=128):
    out=[]
    for v in views:
        fs=[]
        for i in range(0,len(frames[v]),chunk):
            x=common.to_chw(frames[v][i:i+chunk]).to(clf.device).float().div_(255.0)
            fs.append(clf.features([x])[0].cpu())
        out.append(torch.cat(fs))
    return out


def rates(p,y,thr):
    pred=p>=thr;y=y>0.5;n_pos=max(1,int(y.sum()));n_neg=max(1,int((~y).sum()))
    return {'n':int(len(y)),'positives':int(y.sum()),'accuracy':round(float((pred==y).mean()),4),
            'false_positive_rate':round(float((pred&~y).sum()/n_neg),4),'false_negative_rate':round(float((~pred&y).sum()/n_pos),4)}


def photometric(x):
    """Per-image brightness, contrast and colour-balance jitter (x float [0,1], B,3,H,W): the side camera's auto exposure
    and white balance drift during a session (2026-10-09: one insertion was 15-20 % brighter and the classifier missed it)."""
    B=x.shape[0];d=x.device
    gain=torch.empty(B,1,1,1,device=d).uniform_(0.75,1.25)
    tint=torch.empty(B,3,1,1,device=d).uniform_(0.9,1.1)
    mean=x.mean(dim=(1,2,3),keepdim=True);contrast=torch.empty(B,1,1,1,device=d).uniform_(0.8,1.2)
    return ((x-mean)*contrast+mean).mul(gain*tint).clamp_(0,1)


def train(dirs,out,views=('side',),steps=2000,batch=64,lr=3e-4,device=None,seed=0,log=print,agree_only=True,shift_pad=4):
    from lerobot.rl.buffer import random_shift
    torch.manual_seed(seed);np.random.seed(seed)
    device=device or ('cuda' if torch.cuda.is_available() else 'cpu')
    clf=RewardClassifier(views,device)
    sessions=load_sessions(dirs,views)
    total=sum(len(y) for _,y,_,_ in sessions);dis={'usb_on_label_off':int(sum(((u>0.5)&(y<0.5)).sum() for _,y,u,_ in sessions)),
                                                  'label_on_usb_off':int(sum(((u<0.5)&(y>0.5)).sum() for _,y,u,_ in sessions))}
    log(f'frames {total}, label/USB disagreements {dis} ({"left out" if agree_only else "kept"})')
    (xtr,ytr),(xva,yva,uva,agree)=split(sessions,agree_only=agree_only)
    if ytr.sum()==0 or (1-ytr).sum()==0:raise SystemExit('training split needs both labels')
    pos_w=torch.tensor(float((1-ytr).sum()/max(1,ytr.sum())),device=device)
    opt=torch.optim.Adam(clf.head.parameters(),lr=lr);lossf=nn.BCEWithLogitsLoss(pos_weight=pos_w)
    xs={v:common.to_chw(xtr[v]) for v in views};yt=torch.from_numpy(ytr)
    for s in range(steps):
        clf.head.train();idx=torch.randint(0,len(yt),(batch,))
        imgs=[photometric(random_shift(xs[v][idx].to(device).float().div_(255.0),pad=shift_pad)) for v in views]   # shift + lighting jitter
        loss=lossf(clf.head(clf.features(imgs)),yt[idx].to(device))
        opt.zero_grad();loss.backward();opt.step()
        if (s+1)%500==0:log(f'step {s+1}/{steps} loss {loss.item():.4f}')
    clf.head.eval()
    with torch.no_grad():p=torch.sigmoid(clf.head([f.to(device) for f in _encode_all(clf,xva,views)])).cpu().numpy()
    m=rates(p[agree],yva[agree],clf.threshold)
    m['vs_usb']=rates(p,uva,clf.threshold)          # how well it predicts the electrical signal, all validation frames
    m['labels_vs_usb_agreement']=round(float(((yva>0.5)==(uva>0.5)).mean()),4)   # operator labels vs USB (latency shows up here)
    m.update({'shift_pad':shift_pad,'disagreements':dis,'agree_only':agree_only,'train_frames':int(len(ytr)),'train_positives':int(ytr.sum()),'views':list(views),'sessions':[str(d) for d in dirs],'steps':steps})
    clf.meta=m;clf.save(out);log(json.dumps(m,indent=1,ensure_ascii=False))
    return clf,m


def cross_validate(dirs,views=('side',),steps=600,batch=64,lr=3e-4,device=None,shift_pad=4,log=print,seed=0):
    """Leave-one-insertion-out: each USB-on stretch (+-15 s) is held out in turn and the head is trained on the rest
    (agreeing frames only). USB only groups the frames into insertions here; labels are still the operator's.
    Shows whether the classifier recognises an insertion it has never seen, which a time split cannot."""
    from lerobot.rl.buffer import random_shift
    torch.manual_seed(seed);np.random.seed(seed)
    device=device or ('cuda' if torch.cuda.is_available() else 'cpu')
    S=load_sessions(dirs,views)
    X={v:common.to_chw(np.concatenate([s[0][v] for s in S])) for v in views}
    y=np.concatenate([s[1] for s in S]);u=np.concatenate([s[2] for s in S])
    t=np.concatenate([np.load(f)['t']+1e4*i for i,f in enumerate(s[3] for s in S)])   # files never overlap in time
    agree=(y>0.5)==(u>0.5)
    e=np.flatnonzero(np.diff(np.r_[0,u.astype(int),0]));ins=[(t[a],t[b-1]) for a,b in zip(e[::2],e[1::2])]
    gid=-np.ones(len(t),int)
    for i,(a,b) in enumerate(ins):gid[(t>=a-15)&(t<=b+15)]=i
    yt=torch.from_numpy(y);res=[]
    for k in range(len(ins)):
        va=np.flatnonzero(agree&(gid==k))
        if (y[va]>0.5).sum()==0:continue
        tr=np.flatnonzero(agree&(gid!=k));clf=RewardClassifier(views,device);h=clf.head
        opt=torch.optim.Adam(h.parameters(),lr=lr)
        lossf=nn.BCEWithLogitsLoss(pos_weight=torch.tensor(float((y[tr]<0.5).sum()/max(1,(y[tr]>0.5).sum())),device=device))
        for _ in range(steps):
            b=np.random.choice(tr,batch);h.train()
            imgs=[photometric(random_shift(X[v][b].to(device).float().div_(255.0),pad=shift_pad)) for v in views]
            loss=lossf(h(clf.features(imgs)),yt[b].to(device));opt.zero_grad();loss.backward();opt.step()
        h.eval();p=[]
        with torch.no_grad():
            for i in range(0,len(va),128):
                j=va[i:i+128];p.append(torch.sigmoid(h(clf.features([X[v][j].to(device).float().div_(255.0) for v in views]))).cpu().numpy())
        p=np.concatenate(p);r=rates(p,y[va],clf.threshold);r.update({'held_out':k,'t':f'{ins[k][0]%1e4:.0f}-{ins[k][1]%1e4:.0f}s'})
        # threshold / consecutive-frame sweep: the episode ends on clf_hold consecutive frames >= threshold, so what
        # matters for a false success is a run of hold false positives among the negatives (va is time-ordered)
        neg=y[va]<0.5;sw={}
        for thr in (0.5,0.7,0.9,0.95):
            fp=(p>=thr)&neg;runs={h:int(np.convolve(fp.astype(int),np.ones(h,int),'valid').__ge__(h).sum()>0) for h in (1,2,3)}
            sw[thr]={'fpr':round(float(fp.sum()/max(1,neg.sum())),3),'fnr':round(float(((p<thr)&~neg).sum()/max(1,(~neg).sum())),3),'false_trigger_hold':runs}
        fpi=va[(p>=0.9)&neg];up=np.flatnonzero(u>0.5)
        d=[float(np.min(np.abs(t[up]-t[i]))) if len(up) else None for i in fpi]   # seconds to the nearest USB-connected frame
        r['fp_s_to_usb']=sorted(round(x,1) for x in d if x is not None)
        r['sweep']=sw;res.append(r)
        log(json.dumps(r,ensure_ascii=False))
    return res


def from_episodes(ep_dir,out_dir,margin_s=1.0,hz=10.0,skip=()):
    """Demo / training episodes -> classifier recordings labelled by the USB referee (the data flywheel, 2026-10-10).
    Frame k>=1 follows action k-1, so its USB state is usb[k-1]; frame 0 is the reset (USB checked off). Positive =
    USB connected. The last margin_s before the USB comes on (or before a classifier-ended success, where the plug was
    visibly home but not yet enumerated) is ambiguous: written as label 1 / usb 0, which training drops (agree-only).
    Everything earlier is negative."""
    out=Path(out_dir);out.mkdir(parents=True,exist_ok=True);m=int(round(margin_s*hz));n=0;stats={'pos':0,'neg':0,'ambiguous':0}
    meta={}
    jl=Path(ep_dir)/'episodes.jsonl'
    for line in (jl.read_text().splitlines() if jl.exists() else []):
        x=json.loads(line)
        if 'success' in x:meta[x['n']]=x
    for f in sorted(Path(ep_dir).glob('ep-*.npz')):
        idx=int(f.stem.split('-')[1])
        if idx in skip:continue
        z=np.load(f);u=np.r_[False,z['usb'].astype(bool)];T=len(u)
        lab=u.astype(np.float32);usb=u.astype(np.float32)
        ended=bool(len(z['terminals']) and z['terminals'][-1])
        fooled=ended and not u.any() and idx in meta and not meta[idx].get('usb_success',True)   # classifier fired, USB never came on
        on=int(np.argmax(u)) if u.any() else (T if (ended and not fooled) else None)
        if on is not None:
            a=max(0,on-m);lab[a:on]=1.0;usb[a:on]=0.0
        t=1e3*idx+np.arange(T)/hz
        np.savez_compressed(out/f'clf-{idx:04d}.npz',frames=z['frames'],frames_side=z['frames_side'],labels=lab,usb=usb,t=t);n+=1
        stats['pos']+=int((u).sum());stats['ambiguous']+=int(((lab>0.5)&(usb<0.5)).sum());stats['neg']+=int(((lab<0.5)&(usb<0.5)).sum())
    print(json.dumps({'episodes':n,'out':str(out),**stats}),flush=True)


def evaluate(model,dirs,device=None):
    """Per-recording rates of a trained classifier on labelled recordings (agree frames only)."""
    device=device or ('cuda' if torch.cuda.is_available() else 'cpu');clf=RewardClassifier.load(model,device)
    for frames,y,usb,name in load_sessions(dirs,clf.views):
        ok=(y>0.5)==(usb>0.5);p=[]
        with torch.no_grad():
            clf.head.eval()
            for i in range(0,len(y),128):
                imgs=[common.to_chw(frames[v][i:i+128]).to(device).float().div_(255.0) for v in clf.views]
                p.append(torch.sigmoid(clf.head(clf.features(imgs))).reshape(-1).cpu().numpy())
        p=np.concatenate(p)
        print(json.dumps({'file':Path(name).name,**rates(p[ok],y[ok],clf.threshold),'p_pos_median':round(float(np.median(p[ok&(y>0.5)])),3) if (ok&(y>0.5)).any() else None}),flush=True)


def main():
    p=argparse.ArgumentParser()
    p.add_argument('cmd',choices=['train','cv','from-episodes','eval'])
    p.add_argument('dirs',nargs='+')
    p.add_argument('--out')
    p.add_argument('--views',default='side')
    p.add_argument('--steps',type=int,default=2000)
    p.add_argument('--device',default=None)
    p.add_argument('--keep-disagreements',action='store_true',help='train on every frame, even where the label and USB disagree')
    p.add_argument('--shift-pad',type=int,default=4,help='random shift augmentation, pixels of the 128x128 crop')
    p.add_argument('--margin-s',type=float,default=1.0,help='from-episodes: ambiguous seconds before the USB comes on')
    p.add_argument('--skip',type=int,nargs='*',default=[],help='from-episodes: episode numbers to leave out')
    p.add_argument('--model',help='eval: classifier .pt')
    a=p.parse_args()
    if a.cmd=='from-episodes':
        if not a.out:p.error('from-episodes needs --out')
        from_episodes(a.dirs[0],a.out,a.margin_s,skip=set(a.skip));return
    if a.cmd=='eval':evaluate(a.model,a.dirs,a.device);return
    if a.cmd=='cv':cross_validate(a.dirs,common.parse_views(a.views),steps=min(a.steps,600),device=a.device,shift_pad=a.shift_pad);return
    if not a.out:p.error('train needs --out')
    train(a.dirs,a.out,common.parse_views(a.views),steps=a.steps,device=a.device,agree_only=not a.keep_disagreements,shift_pad=a.shift_pad)


if __name__=='__main__':main()
