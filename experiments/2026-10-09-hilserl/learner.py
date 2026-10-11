#!/usr/bin/env python3
"""HIL-SERL learner (runs on the GPU machine, e.g. Vast). The actor on the Jetson connects over gRPC.

Reused from LeRobot 0.4.4 as is: the gRPC LearnerService and transport (lerobot.rl.learner_service,
lerobot.transport), SACPolicy and its losses, DrQ random_shift. The update step is LeRobot's learner.py loop
(UTD critic updates with a target update after each, then one actor and one temperature update, grad clip 40),
via ../2026-10-02/rl_agent.Agent.update.

Data routing follows the paper (sec. 3.4): demonstrations -> offline buffer only; operator-driven steps during
training -> both buffers; policy steps -> online buffer only. Each batch is half online, half offline (RLPD).
LeRobot 0.4.4's own actor drops the intervention flag, so its learner never fills the offline buffer from
interventions; here the flag travels in complementary_info and is honoured.

  python learner.py --out runs/<name>                 # listens on 127.0.0.1:50051 (reach it through an SSH tunnel)
  python learner.py --out runs/<name> --resume runs/<name>/ckpt-last.pt
"""
import argparse
from concurrent.futures import ThreadPoolExecutor
import json
import logging
from pathlib import Path
from queue import Empty
import threading
import time

import torch
from torch.multiprocessing import Queue

import common
from common import IS_DEMO,IS_INTERVENTION,ImageBuffer,cat_batches


class Learner:
    def __init__(self,a):
        import rl_agent
        self.a=a;self.out=Path(a.out);self.out.mkdir(parents=True,exist_ok=True)
        self.agent=rl_agent.Agent.__new__(rl_agent.Agent)   # reuse Agent.update/save without its feature-cache buffer
        ag=self.agent;ag.device=a.device;ag.batch=a.batch;ag.utd=a.utd;ag.clip=40.0;ag.encoder='resnet10';ag.updates=0;ag.last_info={}
        self.views=common.parse_views(a.views)
        ag.policy=common.make_policy(a.device,self.views);ag.policy.train()
        p=ag.policy.get_optim_params()
        ag.opt={'critic':torch.optim.Adam([x for x in p['critic'] if x.requires_grad],lr=a.lr),
                'actor':torch.optim.Adam([x for x in p['actor'] if x.requires_grad],lr=a.lr),
                'temperature':torch.optim.Adam([p['temperature']],lr=a.lr)}
        ag._batch=self._batch;ag._features=self._features
        self.online=ImageBuffer(a.online_cap,self.views);self.offline=ImageBuffer(a.offline_cap,self.views)
        self.counts={'policy':0,'intervention':0,'demo':0}
        if a.resume:
            d=torch.load(a.resume,map_location=a.device)
            ag.policy.load_state_dict(d['policy'])
            for k,v in d['opt'].items():ag.opt[k].load_state_dict(v)
            ag.updates=d.get('updates',0);print(f'resumed {a.resume} at update {ag.updates}; the actor resends episodes to refill the buffers',flush=True)
        self.log=open(self.out/'learner.jsonl','a',buffering=1)

    # ---------- batches: images go through the frozen encoder here (one pass for state + next_state) ----------
    def _batch(self,online,offline):
        h=self.a.batch//2
        if len(self.offline)>0:return cat_batches(self.online.sample(h,self.a.device),self.offline.sample(h,self.a.device))
        return self.online.sample(self.a.batch,self.a.device)

    @torch.no_grad()
    def _features(self,b):
        enc=self.agent.policy.actor.encoder
        f=enc.get_cached_image_features(b['state']);nf=enc.get_cached_image_features(b['next_state'])
        return f,nf

    # ---------- incoming data ----------
    def route(self,tr):
        ci=tr.get('complementary_info') or {}
        demo=bool(float(ci.get(IS_DEMO,torch.zeros(1))[0]));human=bool(float(ci.get(IS_INTERVENTION,torch.zeros(1))[0]))
        if demo:self.offline.add(tr);self.counts['demo']+=1;return
        self.online.add(tr)
        if human:self.offline.add(tr);self.counts['intervention']+=1
        else:self.counts['policy']+=1

    def drain(self,tq,iq):
        from lerobot.transport.utils import bytes_to_python_object
        n=0
        while True:
            try:buf=tq.get_nowait()
            except Empty:break
            ep,views,demo=common.decode_episode(buf)   # one JPEG-compressed episode per message (common.encode_episode)
            for tr in common.episode_transitions(ep,views,demo=demo):self.route(tr);n+=1
        while True:
            try:msg=bytes_to_python_object(iq.get_nowait())
            except Empty:break
            msg.update({'t':time.time(),'updates':self.agent.updates,'online':len(self.online),'offline':len(self.offline),**self.counts,**self.agent.last_info})
            self.log.write(json.dumps(msg)+'\n')
            print(json.dumps(msg,ensure_ascii=False),flush=True)
        return n

    def params_payload(self):
        sd=common.actor_params(self.agent.policy)
        sd['info.updates']=torch.tensor(self.agent.updates)
        for k in ('loss_critic','loss_actor','temperature'):
            if k in self.agent.last_info:sd['info.'+k]=torch.tensor(float(self.agent.last_info[k]))
        return sd

    def save(self,name):
        self.agent.save(self.out/name,{'counts':self.counts,'args':vars(self.a)})

    # ---------- main loop ----------
    def run(self):
        import grpc
        from lerobot.rl.learner_service import MAX_WORKERS,SHUTDOWN_TIMEOUT,LearnerService
        from lerobot.transport import services_pb2_grpc
        from lerobot.transport.utils import MAX_MESSAGE_SIZE,state_to_bytes
        import signal
        a=self.a;stop=threading.Event();pq=Queue();tq=Queue();iq=Queue()
        signal.signal(signal.SIGTERM,lambda *_:stop.set())   # tmux kill / terminate(): leave the loop and save
        service=LearnerService(shutdown_event=stop,parameters_queue=pq,seconds_between_pushes=a.push_s,transition_queue=tq,
                               interaction_message_queue=iq,queue_get_timeout=0.5)
        server=grpc.server(ThreadPoolExecutor(max_workers=MAX_WORKERS),options=[('grpc.max_receive_message_length',MAX_MESSAGE_SIZE),
                                                                              ('grpc.max_send_message_length',MAX_MESSAGE_SIZE)])
        services_pb2_grpc.add_LearnerServiceServicer_to_server(service,server)
        server.add_insecure_port(f'{a.host}:{a.port}');server.start()
        print(f'LEARNER_READY {a.host}:{a.port}  views {",".join(self.views)}  batch {a.batch} utd {a.utd}  discount {common.POLICY_KW["discount"]}  temperature_init {common.POLICY_KW["temperature_init"]}',flush=True)
        pq.put(state_to_bytes(self.params_payload()))
        last_push=time.monotonic();last_save=time.monotonic();last_print=0.0;dts=[]
        try:
            while not stop.is_set():
                self.drain(tq,iq)
                budget=a.warmup+a.max_ratio*self.online.added
                if len(self.online)<a.min_online or self.agent.updates>=budget or (a.updates and self.agent.updates>=a.updates):
                    time.sleep(0.02)
                else:
                    t=time.monotonic();self.agent.update(self.online,self.offline);dts.append(time.monotonic()-t)
                if time.monotonic()-last_push>=a.push_s:
                    pq.put(state_to_bytes(self.params_payload()));last_push=time.monotonic()
                if time.monotonic()-last_save>=a.save_s:self.save('ckpt-last.pt');last_save=time.monotonic()
                if time.monotonic()-last_print>=30 and dts:
                    print(f"updates {self.agent.updates}  {1000*sum(dts)/len(dts):.0f} ms/update  online {len(self.online)} offline {len(self.offline)}  {self.counts}  {self.agent.last_info}",flush=True)
                    last_print=time.monotonic();dts=[]
                if a.exit_after_s and time.monotonic()-self._t0>a.exit_after_s:break
        except KeyboardInterrupt:pass
        finally:
            stop.set();server.stop(SHUTDOWN_TIMEOUT);self.save('ckpt-last.pt')
            print(f'LEARNER_SAVED {self.out}/ckpt-last.pt  updates {self.agent.updates}',flush=True)


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--out',required=True)
    p.add_argument('--host',default='127.0.0.1');p.add_argument('--port',type=int,default=50051)
    p.add_argument('--device',default='cuda')
    p.add_argument('--views',default='wrist,side',help='image views; must match the actor')
    p.add_argument('--batch',type=int,default=256,help='half online, half offline once the offline buffer has data')
    p.add_argument('--utd',type=int,default=2,help='critic updates per actor update')
    p.add_argument('--lr',type=float,default=3e-4)
    p.add_argument('--min-online',type=int,default=100,help='online transitions before the first update')
    p.add_argument('--max-ratio',type=float,default=8.0,help='updates <= warmup + max_ratio x online transitions (no runaway while the robot waits)')
    p.add_argument('--warmup',type=int,default=500)
    p.add_argument('--online-cap',type=int,default=60000);p.add_argument('--offline-cap',type=int,default=20000)
    p.add_argument('--push-s',type=float,default=2.0,help='seconds between weight pushes to the actor (LeRobot default 4; docs advise 1-2)')
    p.add_argument('--save-s',type=float,default=120.0)
    p.add_argument('--updates',type=int,default=0,help='stop updating after this many (0 = no limit)')
    p.add_argument('--resume',help='ckpt-last.pt from an earlier run (weights and optimisers; buffers are refilled by the actor)')
    p.add_argument('--exit-after-s',type=float,default=0.0,help='tests only')
    a=p.parse_args()
    logging.basicConfig(level=logging.WARNING)
    L=Learner(a);L._t0=time.monotonic();L.run()


if __name__=='__main__':main()
