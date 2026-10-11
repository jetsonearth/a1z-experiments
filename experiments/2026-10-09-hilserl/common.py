#!/usr/bin/env python3
"""Shared by the Jetson actor and the GPU learner: one SAC policy definition, the transition format, the replay buffer.

Policy: LeRobot 0.4.4 SACPolicy with ../2026-10-02/rl_agent's substitutions (HIL-SERL's frozen ResNet-10 through the
torch-only port, ImageNet normalisation inside the encoder, the cusolver-free tanh-Gaussian). Image views are
configurable (`--views`, default wrist + side, as the paper's USB task: wrist cameras + side_1); actor and learner must
use the same list. Each view gets its own learned spatial embedding on top of the shared frozen backbone.

Episodes travel as one message each (encode_episode: JPEG frames, each frame once); the learner turns them into LeRobot
Transition dicts with uint8 images.
LeRobot's ReplayBuffer is not used: it stores everything as float32 (4x the memory for images) and in 0.4.4 add() and
sample() race without a lock (fixed upstream in #3372). ImageBuffer below keeps uint8 frames under a lock and applies
LeRobot's DrQ random_shift when sampling.
"""
import json
import sys
import threading
from pathlib import Path

import numpy as np
import torch

ROOT=Path(__file__).resolve().parent
PREV=ROOT.parent/'2026-10-02'
if str(PREV) not in sys.path:sys.path.insert(0,str(PREV))

VIEW_KEYS={'wrist':'observation.images.wrist','side':'observation.images.side'}
FRAME_KEYS={'wrist':'frames','side':'frames_side'}      # episode npz arrays
IMG=VIEW_KEYS['wrist']
STATE='observation.state'
ACT='action'
IMG_SIZE=128;STATE_DIM=9;ACTION_DIM=3
DEFAULT_VIEWS=('wrist','side')
IS_INTERVENTION='is_intervention'   # same key as lerobot.teleoperators.utils.TeleopEvents.IS_INTERVENTION.value
IS_DEMO='is_demo'
FROZEN_PREFIX='encoder.image_encoder.'   # actor.state_dict() keys of the frozen backbone: never sent over the wire

# Paper values (Luo et al. 2024, appendix tables) where LeRobot's defaults differ.
POLICY_KW=dict(discount=0.97,temperature_init=0.01,num_critics=2)


CAMERAS=ROOT/'cameras.json'   # crops saved from the A1Z Console's Cameras page; lives on the Jetson, not synced


def saved_crops():
    """{'crop': (x,y,w,h), 'crop_side': (x,y,w,h)} from cameras.json, only the keys that are set."""
    try:d=json.loads(CAMERAS.read_text())
    except (OSError,ValueError):return {}
    return {k:tuple(int(v) for v in d[k]) for k in ('crop','crop_side') if d.get(k)}


def parse_views(s):
    v=tuple(x.strip() for x in (s.split(',') if isinstance(s,str) else s) if x.strip())
    bad=[x for x in v if x not in VIEW_KEYS]
    if bad or not v:raise ValueError(f'views must be from {list(VIEW_KEYS)}, got {s!r}')
    return v


def make_policy(device,views=DEFAULT_VIEWS):
    import rl_agent
    from lerobot.configs.types import FeatureType,PolicyFeature
    from lerobot.policies.sac import modeling_sac
    from lerobot.policies.sac.configuration_sac import SACConfig
    modeling_sac.PretrainedImageEncoder=rl_agent.FrozenResNet10
    modeling_sac.TanhMultivariateNormalDiag=rl_agent.TanhDiagNormal
    feats={VIEW_KEYS[v]:PolicyFeature(type=FeatureType.VISUAL,shape=(3,IMG_SIZE,IMG_SIZE)) for v in views}
    feats[STATE]=PolicyFeature(type=FeatureType.STATE,shape=(STATE_DIM,))
    cfg=SACConfig(input_features=feats,output_features={ACT:PolicyFeature(type=FeatureType.ACTION,shape=(ACTION_DIM,))},
                  device=device,storage_device='cpu',vision_encoder_name='local:resnet10',freeze_vision_encoder=True,
                  shared_encoder=True,use_torch_compile=False,**POLICY_KW)
    return modeling_sac.SACPolicy(cfg).to(device)


def actor_params(policy):
    """Trainable actor weights only (the frozen ResNet-10 is identical on both sides), on CPU."""
    return {k:v.detach().to('cpu',copy=True) for k,v in policy.actor.state_dict().items() if not k.startswith(FROZEN_PREFIX)}


def load_actor_params(policy,params):
    sd={k:v for k,v in params.items() if not k.startswith('info.')}
    missing,unexpected=policy.actor.load_state_dict(sd,strict=False)
    missing=[k for k in missing if not k.startswith(FROZEN_PREFIX)]
    if missing or unexpected:raise RuntimeError(f'actor weights do not match (same --views on both sides?): missing {missing[:5]} unexpected {unexpected[:5]}')


def to_chw(frames):
    return torch.from_numpy(np.ascontiguousarray(frames)).permute(0,3,1,2).contiguous()   # (N,3,H,W) uint8


def episode_transitions(ep,views=DEFAULT_VIEWS,demo=False):
    """Runner episode dict -> list of LeRobot Transition dicts. A demo episode is all human; in a training episode
    only the operator-driven steps are flagged as interventions (the learner adds those to both buffers)."""
    T=len(ep['actions']);out=[]
    imgs={VIEW_KEYS[v]:to_chw(ep[FRAME_KEYS[v]]) for v in views}
    st=torch.from_numpy(ep['states'].astype(np.float32))
    for t in range(T):
        human=bool(demo or ep['human'][t])
        s={k:x[t:t+1] for k,x in imgs.items()};s[STATE]=st[t:t+1]
        n={k:x[t+1:t+2] for k,x in imgs.items()};n[STATE]=st[t+1:t+2]
        out.append({'state':s,'action':torch.from_numpy(ep['actions'][t:t+1].astype(np.float32)),'reward':float(ep['rewards'][t]),
                    'next_state':n,'done':bool(ep['terminals'][t]),'truncated':bool(t==T-1 and not ep['terminals'][t]),
                    'complementary_info':{IS_INTERVENTION:torch.tensor([float(human)]),IS_DEMO:torch.tensor([float(demo)])}})
    return out


def encode_episode(ep,views=DEFAULT_VIEWS,demo=False,quality=95):
    """One episode as bytes for the actor -> learner link (2026-10-10): each frame once (not again as next_state), as
    JPEG. Raw uint8 transitions were ~8 MB per 40-step episode; the Mac uplink to the GPU machine moved <1 MB/s."""
    import cv2,pickle
    jpg=lambda a:cv2.imencode('.jpg',np.ascontiguousarray(a[:,:,::-1]),[cv2.IMWRITE_JPEG_QUALITY,quality])[1].tobytes()
    d={'views':list(views),'demo':bool(demo),'frames':{v:[jpg(x) for x in ep[FRAME_KEYS[v]]] for v in views},
       **{k:np.asarray(ep[k]) for k in ('states','actions','rewards','terminals','human')}}
    return pickle.dumps(d,protocol=4)


def decode_episode(b):
    """-> (episode dict as episode_transitions expects, views, demo)."""
    import cv2,pickle
    d=pickle.loads(b);ep={k:d[k] for k in ('states','actions','rewards','terminals','human')}
    for v in d['views']:ep[FRAME_KEYS[v]]=np.stack([cv2.imdecode(np.frombuffer(x,np.uint8),cv2.IMREAD_COLOR)[:,:,::-1] for x in d['frames'][v]])
    return ep,tuple(d['views']),d['demo']


def load_episode(path):
    z=np.load(path);return {k:z[k] for k in z.files}


class ImageBuffer:
    """Ring buffer of transitions with uint8 images per view. sample() returns float images in [0,1], DrQ-shifted (pad 4)."""
    def __init__(self,capacity,views=DEFAULT_VIEWS):
        self.cap=capacity;self.size=0;self.ptr=0;self.added=0;self.lock=threading.Lock()
        self.keys=[VIEW_KEYS[v] for v in views];H=W=IMG_SIZE
        self.img={k:np.zeros((capacity,3,H,W),np.uint8) for k in self.keys};self.nimg={k:np.zeros((capacity,3,H,W),np.uint8) for k in self.keys}
        self.st=np.zeros((capacity,STATE_DIM),np.float32);self.nst=np.zeros_like(self.st)
        self.act=np.zeros((capacity,ACTION_DIM),np.float32);self.rew=np.zeros(capacity,np.float32);self.done=np.zeros(capacity,np.float32)

    def __len__(self):return self.size

    def add(self,tr):
        s,n=tr['state'],tr['next_state']
        with self.lock:
            i=self.ptr
            for k in self.keys:self.img[k][i]=s[k][0].numpy();self.nimg[k][i]=n[k][0].numpy()
            self.st[i]=s[STATE][0].numpy();self.nst[i]=n[STATE][0].numpy()
            self.act[i]=tr['action'][0].numpy();self.rew[i]=tr['reward'];self.done[i]=float(tr['done'])
            self.ptr=(self.ptr+1)%self.cap;self.size=min(self.size+1,self.cap);self.added+=1

    def sample(self,n,device,drq=True):
        from lerobot.rl.buffer import random_shift
        with self.lock:
            idx=np.random.randint(0,self.size,n)
            img={k:torch.from_numpy(self.img[k][idx]) for k in self.keys};nimg={k:torch.from_numpy(self.nimg[k][idx]) for k in self.keys}
            st=torch.from_numpy(self.st[idx]);nst=torch.from_numpy(self.nst[idx])
            act=torch.from_numpy(self.act[idx]);rew=torch.from_numpy(self.rew[idx]);done=torch.from_numpy(self.done[idx])
        d=lambda x:x.to(device,non_blocking=True)
        s={STATE:d(st)};ns={STATE:d(nst)}
        for k in self.keys:
            both=torch.cat([d(img[k]),d(nimg[k])],0).float().div_(255.0)
            if drq:both=random_shift(both,pad=4)
            s[k]=both[:n];ns[k]=both[n:]
        return {'state':s,'next_state':ns,ACT:d(act),'reward':d(rew),'done':d(done)}


def cat_batches(a,b):
    out={}
    for k in a:out[k]=cat_batches(a[k],b[k]) if isinstance(a[k],dict) else torch.cat([a[k],b[k]],0)
    return out


def wilson(k,n,z=1.96):
    import math
    if n==0:return (float('nan'),float('nan'))
    p=k/n;d=1+z*z/n;c=(p+z*z/(2*n))/d;h=z*math.sqrt(p*(1-p)/n+z*z/(4*n*n))/d
    return (round(c-h,3),round(c+h,3))
