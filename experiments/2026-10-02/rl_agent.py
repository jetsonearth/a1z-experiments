#!/usr/bin/env python3
"""HIL-SERL agent on LeRobot's SAC (lerobot.policies.sac, the implementation behind LeRobot's HIL-SERL).

Reused from LeRobot as is: SACPolicy (critic ensemble, tanh-Gaussian actor, temperature, losses, target update),
the SpatialLearnedEmbeddings image head, and DrQ random_shift augmentation. The update step below follows
lerobot/rl/learner.py (UTD critic updates with a target update after each, then one actor and one temperature update,
grad-norm clipping 40) and samples half of each batch from the online buffer and half from the offline
(demonstrations + interventions) buffer, as learner.py does when a dataset is given (RLPD-style symmetric sampling).

Local substitutions, because the Jetson has no internet and no transformers: the frozen image backbone is
HIL-SERL's ResNet-10 (helper2424/resnet10) loaded through a torch-only port with the Hub weights copied from the Mac
(resnet10_hilserl.py); --encoder resnet18 uses torchvision's ImageNet ResNet-18 instead (same 512x4x4 output at 128x128).
The replay buffer stores the frozen encoder's features instead of images (see Buffer), because Orin NX cannot afford
the backbone in every update.
"""
import copy
import os
import threading
import time

import numpy as np
import torch
import torch.nn as nn

IMG='observation.images.wrist'
STATE='observation.state'
ACT='action'
R18_WEIGHTS=os.path.expanduser('~/.cache/torch/hub/checkpoints/resnet18-f37072fd.pth')


class FrozenResNet18(nn.Module):
    """Stand-in for modeling_sac.PretrainedImageEncoder. Input float [0,1] (B,3,H,W); ImageNet normalisation inside."""
    def __init__(self,config=None):
        super().__init__()
        import torchvision
        m=torchvision.models.resnet18()
        m.load_state_dict(torch.load(R18_WEIGHTS,map_location='cpu'))
        self.image_enc_layers=nn.Sequential(m.conv1,m.bn1,m.relu,m.maxpool,m.layer1,m.layer2,m.layer3,m.layer4)
        self.image_enc_out_shape=512
        self.register_buffer('mean',torch.tensor([0.485,0.456,0.406]).view(1,3,1,1))
        self.register_buffer('std',torch.tensor([0.229,0.224,0.225]).view(1,3,1,1))
        self.eval()

    def train(self,mode=True):
        return super().train(False)   # frozen: BatchNorm statistics stay ImageNet's

    def forward(self,x):
        return self.image_enc_layers((x-self.mean)/self.std)


class TanhDiagNormal(torch.distributions.TransformedDistribution):
    """Same distribution as modeling_sac.TanhMultivariateNormalDiag (diagonal Gaussian squashed by tanh), built from
    Independent(Normal) instead of MultivariateNormal: the Jetson torch build fails to load cusolver for the Cholesky
    that MultivariateNormal runs (libtorch_cuda_linalg.so: undefined symbol cusolverDnXsyevBatched_bufferSize)."""
    def __init__(self,loc,scale_diag,low=None,high=None):
        from torch.distributions import Independent,Normal,TanhTransform
        super().__init__(Independent(Normal(loc,scale_diag),1),[TanhTransform(cache_size=1)])

    def mode(self):
        return torch.tanh(self.base_dist.mean)


class FrozenResNet10(nn.Module):
    """HIL-SERL's ImageNet ResNet-10 (helper2424/resnet10, ported in resnet10_hilserl.py), frozen; GroupNorm, so there
    are no batch statistics to go stale. Input float [0,1]; ImageNet normalisation inside, as LeRobot's processor does."""
    def __init__(self,config=None):
        super().__init__()
        import resnet10_hilserl as r
        self.image_enc_layers=r.load(os.path.join(os.path.dirname(os.path.abspath(__file__)),r.WEIGHTS))
        self.image_enc_out_shape=512
        self.register_buffer('mean',torch.tensor([0.485,0.456,0.406]).view(1,3,1,1))
        self.register_buffer('std',torch.tensor([0.229,0.224,0.225]).view(1,3,1,1))
        self.eval()

    def train(self,mode=True):
        return super().train(False)

    def forward(self,x):
        return self.image_enc_layers((x-self.mean)/self.std)


ENCODERS={'resnet10':FrozenResNet10,'resnet18':FrozenResNet18}


def make_policy(device,img=128,state_dim=9,action_dim=3,discount=0.97,temperature_init=0.01,num_critics=2,encoder='resnet10'):
    from lerobot.configs.types import FeatureType,PolicyFeature
    from lerobot.policies.sac import modeling_sac
    from lerobot.policies.sac.configuration_sac import SACConfig
    modeling_sac.PretrainedImageEncoder=ENCODERS[encoder]
    modeling_sac.TanhMultivariateNormalDiag=TanhDiagNormal
    cfg=SACConfig(
        input_features={IMG:PolicyFeature(type=FeatureType.VISUAL,shape=(3,img,img)),STATE:PolicyFeature(type=FeatureType.STATE,shape=(state_dim,))},
        output_features={ACT:PolicyFeature(type=FeatureType.ACTION,shape=(action_dim,))},
        device=device,storage_device='cpu',vision_encoder_name=f'local:{encoder}',freeze_vision_encoder=True,
        shared_encoder=True,discount=discount,temperature_init=temperature_init,num_critics=num_critics,use_torch_compile=False)
    policy=modeling_sac.SACPolicy(cfg).to(device)
    return policy


class Buffer:
    """Ring buffer of transitions holding frozen-encoder features (fp16, K augmented versions per frame), not images.
    Orin NX needs ~0.8 s per update when 1024 images pass through ResNet-18 every update; with cached features an
    update only runs the trainable heads. DrQ is kept in a cheaper form: each frame is encoded unshifted and with
    K-1 random 4-pixel shifts at collection time, and sampling picks one version per transition."""
    def __init__(self,capacity,K=2,C=512,H=4,W=4,state_dim=9,action_dim=3):
        self.cap=capacity;self.K=K;self.size=0;self.ptr=0;self.lock=threading.Lock();self.added=0
        self.f=np.zeros((capacity,K,C,H,W),np.float16);self.nf=np.zeros_like(self.f)
        self.st=np.zeros((capacity,state_dim),np.float32);self.nst=np.zeros_like(self.st)
        self.act=np.zeros((capacity,action_dim),np.float32);self.rew=np.zeros(capacity,np.float32);self.done=np.zeros(capacity,np.float32)

    def add_episode(self,ep,mask=None):
        """ep: feats (T+1,K,C,H,W) fp16, states (T+1,S), actions (T,A), rewards (T), terminals (T). mask: which t to add."""
        T=len(ep['actions'])
        with self.lock:
            for t in range(T):
                if mask is not None and not mask[t]:continue
                i=self.ptr
                self.f[i]=ep['feats'][t];self.nf[i]=ep['feats'][t+1];self.st[i]=ep['states'][t];self.nst[i]=ep['states'][t+1]
                self.act[i]=ep['actions'][t];self.rew[i]=ep['rewards'][t];self.done[i]=ep['terminals'][t]
                self.ptr=(self.ptr+1)%self.cap;self.size=min(self.size+1,self.cap);self.added+=1

    def sample(self,n,device):
        with self.lock:
            idx=np.random.randint(0,self.size,n);k=np.random.randint(0,self.K,n);nk=np.random.randint(0,self.K,n)
            f=torch.from_numpy(self.f[idx,k]);nf=torch.from_numpy(self.nf[idx,nk])
            st=torch.from_numpy(self.st[idx]);nst=torch.from_numpy(self.nst[idx])
            act=torch.from_numpy(self.act[idx]);rew=torch.from_numpy(self.rew[idx]);done=torch.from_numpy(self.done[idx])
        d=lambda x:x.to(device,non_blocking=True)
        return {'state':{STATE:d(st)},'next_state':{STATE:d(nst)},'feat':{IMG:d(f).float()},'next_feat':{IMG:d(nf).float()},
                ACT:d(act),'reward':d(rew),'done':d(done)}


def _cat(a,b):
    if b is None:return a
    out={}
    for k in a:
        out[k]=_cat(a[k],b[k]) if isinstance(a[k],dict) else torch.cat([a[k],b[k]],0)
    return out


class Agent:
    def __init__(self,device='cuda',batch=256,utd=2,lr=3e-4,clip=40.0,**kw):
        self.device=device;self.batch=batch;self.utd=utd;self.clip=clip;self.encoder=kw.get('encoder','resnet10')
        self.policy=make_policy(device,**kw);self.policy.train()
        self.infer=copy.deepcopy(self.policy);self.infer.eval();self.infer_lock=threading.Lock()
        p=self.policy.get_optim_params()
        self.opt={'critic':torch.optim.Adam([x for x in p['critic'] if x.requires_grad],lr=lr),
                  'actor':torch.optim.Adam([x for x in p['actor'] if x.requires_grad],lr=lr),
                  'temperature':torch.optim.Adam([p['temperature']],lr=lr)}
        self.updates=0;self.last_info={}

    # ---------- acting ----------
    @torch.no_grad()
    def act(self,obs,deterministic=False):
        img=torch.from_numpy(obs['image']).to(self.device).permute(2,0,1).unsqueeze(0).float().div_(255.0)
        b={STATE:torch.from_numpy(obs['state']).to(self.device).unsqueeze(0)}
        with self.infer_lock:
            with torch.autocast('cuda',dtype=torch.float16,enabled=self.device.startswith('cuda')):
                f=self.infer.actor.encoder.image_encoder(img)
            a,_,mean=self.infer.actor(b,{IMG:f.float()})
        out=torch.tanh(mean) if deterministic else a
        return out[0].float().cpu().numpy()

    def warmup(self):
        """Pay the first-call CUDA costs now (no weight change: gradients are computed and discarded)."""
        ob={'image':np.zeros((128,128,3),np.uint8),'state':np.zeros(9,np.float32)}
        for _ in range(3):self.act(ob);self.act(ob,deterministic=True)
        f=self.encode(np.zeros((8,128,128,3),np.uint8))
        b=Buffer(16);T=7
        b.add_episode({'feats':f,'states':np.zeros((T+1,9),np.float32),'actions':np.zeros((T,3),np.float32),
                       'rewards':np.zeros(T,np.float32),'terminals':np.zeros(T,np.float32)})
        bt=b.sample(self.batch,self.device)
        fb={ACT:bt[ACT],'reward':bt['reward'],'state':bt['state'],'next_state':bt['next_state'],'done':bt['done'],
            'observation_feature':bt['feat'],'next_observation_feature':bt['next_feat']}
        for m in ('critic','actor','temperature'):
            list(self.policy.forward(fb,model=m).values())[0].backward()
        for o in self.opt.values():o.zero_grad(set_to_none=True)
        if self.device.startswith('cuda'):torch.cuda.synchronize()

    def sync_infer(self):
        """Copy the trainable weights (the frozen backbone is identical in both copies) to the acting policy."""
        self._gpu_wait()
        src=dict(self.policy.named_parameters())
        with self.infer_lock,torch.no_grad():
            for n,p in self.infer.named_parameters():
                if src[n].requires_grad:p.copy_(src[n])
            self.infer.log_alpha.copy_(self.policy.log_alpha)
        self._gpu_wait()

    # ---------- learning (follows lerobot/rl/learner.py) ----------
    def _batch(self,online,offline):
        if offline is not None and offline.size>0:
            h=self.batch//2;return _cat(online.sample(h,self.device),offline.sample(h,self.device))
        return online.sample(self.batch,self.device)

    def _features(self,b):
        return b['feat'],b['next_feat']

    @torch.no_grad()
    def encode(self,frames,K=2,chunk=64):
        """frames (N,H,W,3) uint8 -> (N,K,512,4,4) fp16 features: version 0 unshifted, the rest DrQ random shifts (pad 4)."""
        from lerobot.rl.buffer import random_shift
        enc=self.policy.actor.encoder.image_encoder;out=[]
        for i in range(0,len(frames),chunk):
            x=torch.from_numpy(np.ascontiguousarray(frames[i:i+chunk])).to(self.device).permute(0,3,1,2).float().div_(255.0)
            vs=[x]+[random_shift(x,pad=4) for _ in range(K-1)]
            with torch.autocast('cuda',dtype=torch.float16,enabled=self.device.startswith('cuda')):
                fs=[enc(v) for v in vs]
            out.append(torch.stack(fs,1).half().cpu())
        return torch.cat(out,0).numpy()

    def _gpu_wait(self):
        """Wait for queued GPU work with the GIL released. SACPolicy.temperature calls .item() inside the losses, and a
        .item() that has to wait for the GPU holds the GIL meanwhile: in the dry run that stalled the 30 Hz control
        thread for up to 387 ms (robot_proc floats the arm after 500 ms without a heartbeat)."""
        if self.device.startswith('cuda'):torch.cuda.synchronize()

    def update(self,online,offline=None):
        pol=self.policy
        for i in range(self.utd):
            b=self._batch(online,offline);f,nf=self._features(b)
            fb={ACT:b[ACT],'reward':b['reward'],'state':b['state'],'next_state':b['next_state'],'done':b['done'],
                'observation_feature':f,'next_observation_feature':nf}
            self._gpu_wait()
            loss_c=pol.forward(fb,model='critic')['loss_critic']
            self.opt['critic'].zero_grad();loss_c.backward()
            torch.nn.utils.clip_grad_norm_(pol.critic_ensemble.parameters(),self.clip);self.opt['critic'].step()
            pol.update_target_networks()
        self._gpu_wait()
        loss_a=pol.forward(fb,model='actor')['loss_actor']
        self.opt['actor'].zero_grad();loss_a.backward()
        torch.nn.utils.clip_grad_norm_(pol.actor.parameters(),self.clip);self.opt['actor'].step()
        self._gpu_wait()
        loss_t=pol.forward(fb,model='temperature')['loss_temperature']
        self.opt['temperature'].zero_grad();loss_t.backward();self.opt['temperature'].step()
        self._gpu_wait();self.updates+=1
        self.last_info={'loss_critic':round(loss_c.item(),4),'loss_actor':round(loss_a.item(),4),'temperature':round(pol.temperature,5),'updates':self.updates}
        return self.last_info

    def save(self,path,extra=None):
        torch.save({'policy':self.policy.state_dict(),'opt':{k:v.state_dict() for k,v in self.opt.items()},'updates':self.updates,
                    'encoder':self.encoder,'extra':extra or {}},path)

    def load(self,path,optim=True):
        d=torch.load(path,map_location=self.device)
        if d.get('encoder','resnet18')!=self.encoder:raise ValueError(f"{path} was trained with {d.get('encoder','resnet18')}, not {self.encoder}: pass --encoder {d.get('encoder','resnet18')}")
        self.policy.load_state_dict(d['policy'])
        if optim:
            for k,v in d['opt'].items():self.opt[k].load_state_dict(v)
        self.updates=d.get('updates',0);self.sync_infer()
        return d.get('extra',{})


class LearnerThread(threading.Thread):
    """Background updates while the robot runs. Update count is capped at max_ratio x online transitions added
    (plus a warm-up budget on the demonstrations), so the critic never trains far ahead of the data."""
    def __init__(self,agent,online,offline,min_online=100,max_ratio=4.0,warmup=500,sync_every=50):
        super().__init__(daemon=True)
        self.agent=agent;self.online=online;self.offline=offline;self.min_online=min_online
        self.max_ratio=max_ratio;self.warmup=warmup;self.sync_every=sync_every
        self.stop=threading.Event();self.paused=threading.Event();self.error=None;self.dt_ms=None;self.trace=[]

    def run(self):
        try:
            while not self.stop.is_set():
                budget=self.warmup+self.max_ratio*self.online.added
                if self.paused.is_set() or self.online.size<self.min_online or self.agent.updates>=budget:
                    time.sleep(0.05);continue
                t=time.monotonic();self.agent.update(self.online,self.offline)
                self.dt_ms=round((time.monotonic()-t)*1000,1);self.trace.append((round(t,3),self.dt_ms,'update'))
                if self.agent.updates%self.sync_every==0:
                    t=time.monotonic();self.agent.sync_infer();self.trace.append((round(t,3),round((time.monotonic()-t)*1000,1),'sync'))
                time.sleep(0.002)   # let the 30 Hz control thread take the GIL
        except Exception as e:
            self.error=repr(e)
