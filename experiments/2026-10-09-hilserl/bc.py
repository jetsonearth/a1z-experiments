#!/usr/bin/env python3
"""ACT baselines in the HIL-SERL environment (P-END-200 DoD 4): plain BC on the demos, then HG-DAgger.

Same environment as the RL policy: 10 Hz, 3-D delta actions in [-1,1] (x 1.5 mm/step), the 9-D relative state, the
128x128 wrist crop, the same starts, compliance layer and USB referee. Only the learner differs. Inputs: wrist view only
by decision 2026-10-11 (Jetson); the RL policy also sees the side camera, which the write-up states.

Labels: every step of a successful demo; in an HG-DAgger session only the operator-driven steps (Kelly et al. 2019,
Algorithm 1). An action chunk is cut at the first non-operator step (padded and masked), so a label never contains the
policy's own motion after a handback.

  python bc.py train --demos runs/...-demo [--dagger runs/...-dagger ...] --out runs/act-bc [--init ckpt.pt --steps 5000]
"""
import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch

import common

# State dims the ACT sees: displacement from the start only. 2026-10-11 first ACT eval: 0/30, the arm barely moved.
# Dims 3-5 are the tracking error (p_cmd - p_meas)/5; in kinesthetic demos the command follows the hand, so their std is
# 0.02-0.04, while under policy control gravity and the press give -0.1 to -0.5, i.e. 10+ sigma out of distribution, and
# the policy output ~0. Dims 6-8 (previous action) are dropped too: human vs policy actions differ in the same way and
# action history invites copycat behaviour in BC. The RL policy keeps all 9 (it learns from its own rollouts).
STATE_KEEP=(0,1,2)
# Operator no-op steps are dropped from the labels (as export_hil.py's --idle-rad in the 2026-10-04 pipeline). In the 25
# demos 34% of steps move < 0.05 x 1.5 mm, and the operator waits a median ~1 s before dragging; ACT learnt "stand
# still at the start", and in closed loop the image never changes, so it never leaves the start. Dropped steps are
# removed from the sequence (the next kept frame follows), so chunks run on through them.
IDLE=0.05
# Contact relabel for operator corrections (deviation from plain HG-DAgger, 2026-10-11). Kinesthetic labels are the measured
# motion; while the operator slides the plug on the hub or pushes it into the port, the hub blocks the motion, so the
# vertical label is ~0 although the hand presses down (round-1 labels 0-5 mm above the surface: mean z -0.03, demos
# -0.45). In this action space pressing = commanding below the contact, so round 2 found the port and stopped pressing.
# For operator steps with the plug at or below CONTACT_MM above the calibrated surface, z = min(z, PRESS_Z) unless the
# operator is lifting (z > LIFT_Z). Demos are kept as recorded (the RL offline buffer).
CONTACT_MM=3.0;PRESS_Z=-0.5;LIFT_Z=0.1   # 3 mm: round-1 sliding labels sit 1-3 mm above the calibrated surface (grasp offset)
CHUNK=10   # 1 s at 10 Hz (the 2026-10-04 ACT used 100 steps at 30 Hz from home; here episodes are 25-65 steps)
IMNET_MEAN=torch.tensor([0.485,0.456,0.406]).view(1,3,1,1);IMNET_STD=torch.tensor([0.229,0.224,0.225]).view(1,3,1,1)


def make_act(device,pretrained=True,chunk=CHUNK,ensemble=0.01):
    from lerobot.configs.types import FeatureType,PolicyFeature
    from lerobot.policies.act.configuration_act import ACTConfig
    from lerobot.policies.act.modeling_act import ACTPolicy
    cfg=ACTConfig(input_features={common.IMG:PolicyFeature(type=FeatureType.VISUAL,shape=(3,common.IMG_SIZE,common.IMG_SIZE)),
                                  common.STATE:PolicyFeature(type=FeatureType.STATE,shape=(common.STATE_DIM,))},
                  output_features={common.ACT:PolicyFeature(type=FeatureType.ACTION,shape=(common.ACTION_DIM,))},
                  chunk_size=chunk,n_action_steps=1,temporal_ensemble_coeff=ensemble,device=device,
                  pretrained_backbone_weights='ResNet18_Weights.IMAGENET1K_V1' if pretrained else None)
    return ACTPolicy(cfg).to(device)


def drop_idle(ep,lab,idle=IDLE):
    """Remove operator steps with |a| < idle; policy steps stay (they cut chunks). -> frames, states, actions, labels."""
    T=len(ep['actions']);keep=~(lab&(np.linalg.norm(ep['actions'],axis=1)<idle))
    idx=np.flatnonzero(keep);fi=np.r_[idx,T]
    return ep['frames'][fi],ep['states'][fi],ep['actions'][idx],lab[idx]


def relabel_contact(ep,lab,z_start,z_surface):
    """-> actions with the contact relabel applied to operator steps, and how many were changed."""
    a=ep['actions'].astype(np.float32).copy();z=z_start+ep['states'][:-1,2]*10.0
    m=lab&(z<=z_surface+CONTACT_MM)&(a[:,2]<=LIFT_Z)&(a[:,2]>PRESS_Z)
    a[m,2]=PRESS_Z;return a,int(m.sum())


def load_sessions(demos,dagger,contact_relabel=True):
    """-> list of (frames uint8 (T+1,H,W,3), states (T+1,9), actions (T,3), label mask (T,)) and counts."""
    out=[];cnt={'demo_eps':0,'demo_steps':0,'dagger_eps':0,'dagger_steps':0}
    def eps(d):
        meta={}
        jl=Path(d)/'episodes.jsonl'
        for line in (jl.read_text().splitlines() if jl.exists() else []):
            m=json.loads(line)
            if 'n' in m and 'success' in m:meta[m['n']]=m
        for f in sorted(Path(d).glob('ep-*.npz')):yield common.load_episode(f),meta.get(int(f.stem.split('-')[1]),{})
    for d in demos or []:
        for ep,m in eps(d):
            if not (len(ep['terminals']) and ep['terminals'][-1]):continue   # successful demos only, as the RL offline buffer
            lab=np.ones(len(ep['actions']),bool);x=drop_idle(ep,lab);out.append(x)
            cnt['demo_eps']+=1;cnt['demo_steps']+=int(x[3].sum());cnt['demo_idle_dropped']=cnt.get('demo_idle_dropped',0)+len(lab)-len(x[3])
    for d in dagger or []:
        zs=json.loads((Path(d)/'session.json').read_text())['port']['z_surface_mm']
        for ep,m in eps(d):
            lab=ep['human'].astype(bool)
            if not lab.any():continue
            if contact_relabel:
                ep=dict(ep);ep['actions'],k=relabel_contact(ep,lab,m['p_start'][2],zs);cnt['contact_relabelled']=cnt.get('contact_relabelled',0)+k
            x=drop_idle(ep,lab);out.append(x);cnt['dagger_eps']+=1;cnt['dagger_steps']+=int(x[3].sum())
            cnt['dagger_idle_dropped']=cnt.get('dagger_idle_dropped',0)+int(lab.sum())-int(x[3].sum())
    return out,cnt


class ChunkData:
    def __init__(self,episodes,chunk=CHUNK):
        self.F=[];self.S=[];self.idx=[];self.A=[];self.L=[]
        for e,(fr,st,ac,lab) in enumerate(episodes):
            self.F.append(fr);self.S.append(st.astype(np.float32));self.A.append(ac.astype(np.float32));self.L.append(lab)
            self.idx+= [(e,t) for t in np.flatnonzero(lab)]
        self.chunk=chunk
        lab_a=np.concatenate([a[l] for a,l in zip(self.A,self.L)]);lab_s=np.concatenate([s[:-1][l] for s,l in zip(self.S,self.L)])
        self.stats={'state_mean':lab_s.mean(0),'state_std':lab_s.std(0)+1e-3,'action_mean':lab_a.mean(0),'action_std':lab_a.std(0)+1e-3}

    def __len__(self):return len(self.idx)

    def batch(self,ids):
        K=self.chunk;img=[];st=[];act=np.zeros((len(ids),K,common.ACTION_DIM),np.float32);pad=np.ones((len(ids),K),bool)
        for i,j in enumerate(ids):
            e,t=self.idx[j];img.append(self.F[e][t]);st.append(self.S[e][t])
            for k in range(K):
                if t+k>=len(self.A[e]) or not self.L[e][t+k]:break   # cut at the episode end or the first policy step
                act[i,k]=self.A[e][t+k];pad[i,k]=False
        return np.stack(img),np.stack(st),act,pad


def norm_batch(img,st,stats,device,shift=0,keep=STATE_KEEP):
    x=torch.from_numpy(img).to(device).permute(0,3,1,2).float().div_(255.0)
    if shift:
        from lerobot.rl.buffer import random_shift
        x=random_shift(x,pad=shift)   # DrQ shift, as in the RL learner
    x=(x-IMNET_MEAN.to(device))/IMNET_STD.to(device)
    s=(torch.from_numpy(st).to(device)-torch.as_tensor(stats['state_mean'],device=device))/torch.as_tensor(stats['state_std'],device=device)
    if keep is not None:
        m=torch.zeros(s.shape[-1],device=device);m[list(keep)]=1.0;s=s*m   # dropped dims -> 0 (their normalised mean)
    return {common.IMG:x,common.STATE:s.float()}


def train(a):
    device=a.device or ('cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu')
    episodes,cnt=load_sessions(a.demos,a.dagger,contact_relabel=not a.no_contact_relabel)
    rng=np.random.default_rng(0);order=rng.permutation(len(episodes));nval=max(1,round(len(episodes)*a.val_frac)) if a.val_frac>0 else 0
    val=[episodes[i] for i in order[:nval]];tr=[episodes[i] for i in order[nval:]]
    data=ChunkData(tr);vdata=ChunkData(val) if val else None
    pol=make_act(device,pretrained=a.init is None)
    stats=data.stats
    if a.init:   # HG-DAgger fine-tune: keep the previous novice's weights; normalisation from the aggregate (as export_hil)
        d=torch.load(a.init,map_location=device,weights_only=False);pol.load_state_dict(d['state_dict'])
    st_t={k:torch.as_tensor(v,device=device,dtype=torch.float32) for k,v in stats.items()}
    groups=[{'params':[p for n,p in pol.named_parameters() if not n.startswith('model.backbone') and p.requires_grad]},
            {'params':[p for n,p in pol.named_parameters() if n.startswith('model.backbone') and p.requires_grad],'lr':a.lr_backbone}]
    opt=torch.optim.AdamW(groups,lr=a.lr,weight_decay=1e-4)
    out=Path(a.out);out.mkdir(parents=True,exist_ok=True)
    meta={'kind':'act','chunk':CHUNK,'views':['wrist'],'state_keep':list(STATE_KEEP),'counts':cnt,'train_samples':len(data),'val_samples':len(vdata) if vdata else 0,
          'args':{k:(v if not isinstance(v,Path) else str(v)) for k,v in vars(a).items()},'t':time.time()}
    print(json.dumps(meta,ensure_ascii=False),flush=True)
    log=open(out/'train_log.jsonl','a',buffering=1);t0=time.time();pol.train();best=None
    def save(path,step,vl):
        torch.save({'kind':'act','chunk':CHUNK,'views':['wrist'],'state_keep':list(STATE_KEEP),'idle':IDLE,'state_dict':pol.state_dict(),
                    'stats':{k:np.asarray(v) for k,v in stats.items()},'step':step,'val_l1':vl,'meta':meta},path)
    for step in range(1,a.steps+1):
        ids=rng.integers(0,len(data),a.batch);img,st,act,pad=data.batch(ids)
        b=norm_batch(img,st,stats,device,shift=4)
        b[common.ACT]=(torch.from_numpy(act).to(device)-st_t['action_mean'])/st_t['action_std'];b['action_is_pad']=torch.from_numpy(pad).to(device)
        loss,info=pol.forward(b);opt.zero_grad();loss.backward();torch.nn.utils.clip_grad_norm_(pol.parameters(),10.0);opt.step()
        if step%a.log_every==0 or step==a.steps:
            vl=None
            if vdata:
                pol.eval()
                with torch.no_grad():
                    ls=[]
                    for i in range(0,len(vdata),256):
                        ids=np.arange(i,min(i+256,len(vdata)));img,st,act,pad=vdata.batch(ids);b=norm_batch(img,st,stats,device)
                        b[common.ACT]=(torch.from_numpy(act).to(device)-st_t['action_mean'])/st_t['action_std'];b['action_is_pad']=torch.from_numpy(pad).to(device)
                        b['observation.images']=[b[common.IMG]];ah,_=pol.model(b)   # eval mode: latent = 0, as at test time
                        m=~b['action_is_pad'];ls.append(float((ah-b[common.ACT]).abs().mean(-1)[m].mean())*len(ids))
                    vl=float(sum(ls)/len(vdata))
                pol.train()
            r={'step':step,'loss':float(loss),'l1':float(info.get('l1_loss',float('nan'))),'kld':float(info.get('kld_loss',float('nan'))),'val_l1':vl,'s':round(time.time()-t0,1)}
            log.write(json.dumps(r)+'\n');print(r,flush=True)
            save(out/'act-last.pt',step,vl)
            if vl is not None and (best is None or vl<best):best=vl;save(out/'act-best-val.pt',step,vl)
    print(f'saved {out}/act-last.pt',flush=True)


class ActAgent:
    """Drop-in for actor.RemoteAgent in eval and HG-DAgger sessions. reset() at every episode start and handback
    clears ACT's temporal-ensembling queue."""
    def __init__(self,path,device):
        d=torch.load(path,map_location=device,weights_only=False)
        self.device=device;self.policy=make_act(device,pretrained=False,chunk=d.get('chunk',CHUNK));self.policy.load_state_dict(d['state_dict']);self.policy.eval()
        self.stats={k:np.asarray(v,np.float32) for k,v in d['stats'].items()};self.info={'step':d.get('step'),'val_l1':d.get('val_l1')};self.version=1
        self.views=tuple(d.get('views',['wrist']));self.keep=d.get('state_keep')   # None: checkpoints before the fix saw all 9

    def reset(self):self.policy.reset()

    def act(self,obs,deterministic=True):
        with torch.no_grad():
            b=norm_batch(obs['image'][None],obs['state'][None].astype(np.float32),self.stats,self.device,keep=self.keep)
            a=self.policy.select_action(b)[0].float().cpu().numpy()
        return np.clip(a*self.stats['action_std']+self.stats['action_mean'],-1,1).astype(np.float32)

    def warmup(self):
        z=np.zeros((common.IMG_SIZE,common.IMG_SIZE,3),np.uint8);ob={'image':z,'state':np.zeros(common.STATE_DIM,np.float32)}
        for _ in range(3):self.act(ob)
        self.reset()


def is_act_ckpt(path):
    try:return torch.load(path,map_location='cpu',weights_only=False).get('kind')=='act'
    except Exception:return False


def main():
    p=argparse.ArgumentParser();sp=p.add_subparsers(dest='cmd',required=True)
    t=sp.add_parser('train')
    t.add_argument('--demos',nargs='*',default=[]);t.add_argument('--dagger',nargs='*',default=[])
    t.add_argument('--out',required=True);t.add_argument('--init',default=None,help='HG-DAgger: fine-tune from the previous novice')
    t.add_argument('--steps',type=int,default=20000);t.add_argument('--batch',type=int,default=64)
    t.add_argument('--lr',type=float,default=1e-5);t.add_argument('--lr-backbone',type=float,default=1e-5)   # ACT defaults
    t.add_argument('--no-contact-relabel',action='store_true',help='plain HG-DAgger labels (measured motion only)')
    t.add_argument('--val-frac',type=float,default=0.1);t.add_argument('--log-every',type=int,default=500);t.add_argument('--device',default=None)
    a=p.parse_args()
    if a.cmd=='train':train(a)


if __name__=='__main__':main()
