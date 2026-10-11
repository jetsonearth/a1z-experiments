#!/usr/bin/env python3
"""Same-conditions comparison: ACT vs HG-DAgger vs HG-DAgger -> HIL-SERL (rl_run.py chain, see RL-RECOVERY.md).

ACT checkpoints run in-process with the same observation pipeline as lerobot.async_inference's policy server
(raw_observation_to_observation -> saved preprocessor -> predict_action_chunk -> saved postprocessor), executing
30-step chunks at 30 Hz and replanning every 10 actions, blending overlapping timesteps 30 % old / 70 % new (the
client's weighted_average). This is not byte-identical to eval.sh's async client, so ACT / HG-DAgger numbers from
this mode are comparable with each other and with the chained group, not with the afternoon eval.sh sessions.

Every trial starts from one of the stored start poses (chain_starts.json: first joint pose of 10-05 eval trials),
reached by a lift and a cosine joint move. Success = the plug enters the port mouth (rl_env.InsertEnv.in_port,
calibrated height), the same test for all groups; the trial ends there, without pushing to full depth.
"""
import json
import time

import numpy as np

JOINTS=[f'j{i}' for i in range(1,7)]


class ActRunner:
    def __init__(self,robot,path,device='cuda',actions_per_chunk=30,replan_every=10,blend_new=0.7):
        import torch
        from lerobot.async_inference.helpers import map_robot_keys_to_lerobot_features
        from lerobot.policies.factory import get_policy_class,make_pre_post_processors
        self.torch=torch;self.path=path;self.device=device
        self.policy=get_policy_class('act').from_pretrained(path);self.policy.to(device);self.policy.eval()
        dev={'device':device}
        self.pre,self.post=make_pre_post_processors(self.policy.config,pretrained_path=path,
            preprocessor_overrides={'device_processor':dev,'rename_observations_processor':{'rename_map':{}}},
            postprocessor_overrides={'device_processor':dev})
        self.features=map_robot_keys_to_lerobot_features(robot)
        self.apc=actions_per_chunk;self.replan=replan_every;self.blend=blend_new
        self.reset()

    def reset(self):
        self.queue=[];self.since=10**9;self.policy.reset()

    def _chunk(self,raw):
        from lerobot.async_inference.helpers import raw_observation_to_observation
        torch=self.torch
        obs=raw_observation_to_observation(raw,self.features,self.policy.config.image_features)
        obs=self.pre(obs)
        with torch.no_grad():
            a=self.policy.predict_action_chunk(obs)
        if a.ndim!=3:a=a.unsqueeze(0)
        a=a[:,:self.apc,:]
        out=[self.post(a[:,i,:]) for i in range(a.shape[1])]
        return torch.stack(out,1).squeeze(0).detach().cpu().numpy()

    def warmup(self,raw):
        for _ in range(2):self._chunk(raw)

    def step(self,raw):
        """One 30 Hz tick: returns the 6 joint targets to send (gripper output ignored, as in eval)."""
        if self.since>=self.replan or not self.queue:
            new=list(self._chunk(raw))
            for i in range(min(len(self.queue),len(new))):
                new[i]=(1-self.blend)*np.asarray(self.queue[i])+self.blend*np.asarray(new[i])
            self.queue=new;self.since=0
        a=self.queue.pop(0);self.since+=1
        return np.asarray(a[:6],dtype=float)


def cosine_joint_move(robot,clock,q1,rate=0.3):
    """Cosine-eased joint interpolation from the commanded pose to q1 at <= rate rad/s (control_test.smooth_plan)."""
    q0=np.asarray(robot.commanded() if np.isfinite(robot.commanded()).all() else robot.measured()[0],dtype=float)
    q1=np.asarray(q1,dtype=float);T=max(1.5,float(np.abs(q1-q0).max())/rate);n=int(T*30)+1
    for i in range(1,n+1):
        s=i/n;robot.send_target(q0+(q1-q0)*(1-np.cos(np.pi*s))/2,'start_move');clock.tick()
    for _ in range(15):robot.send_target(q1,'start_move');clock.tick()


def stalled(hist,z_surface,window=45,move_mm=1.0,height_mm=20.0):
    """Last 1.5 s: plug within height_mm above the hub surface and moved less than move_mm."""
    if len(hist)<window:return False
    w=np.array(hist[-window:])
    return bool(w[-1,2]<z_surface+height_mm and np.linalg.norm(w.max(0)-w.min(0))<move_mm)


def run_trial(runner,env,group,start_q,acts,agent,cfg):
    """One trial. group in {'act','dagger','dagger+rl'}. Returns a result dict."""
    robot=runner.robot;clock=runner.clock;port=env.port
    env.engage()
    pm,_=env.p_meas()
    if pm[2]<port.z_surface+20:env.goto([env.p_cmd[0],env.p_cmd[1],port.z_surface+30],15.0)   # clear the hub first
    cosine_joint_move(robot,clock,start_q)
    act=acts['act' if group=='act' else 'dagger'];act.reset()
    t0=time.monotonic();hist=[];res={'group':group,'success':False,'phase':None,'handoff_s':None,'t_success_s':None}
    budget=cfg['act_s']+cfg['rl_s']
    while time.monotonic()-t0<(cfg['act_s'] if group=='dagger+rl' else budget):
        raw=robot.get_observation();target=act.step(raw)
        robot.send_target(target,'act');clock.tick()
        pm,_=env.p_meas();hist.append(pm.copy())
        if env.in_port(pm):
            res.update(success=True,phase='act',t_success_s=round(time.monotonic()-t0,2));break
        if group=='dagger+rl' and time.monotonic()-t0>cfg['min_act_s'] and stalled(hist,port.z_surface):break
    res['act_end']=env.p_meas()[0].round(2).tolist()
    if group=='dagger+rl' and not res['success']:
        res['handoff_s']=round(time.monotonic()-t0,2)
        env._sync_cmd()
        env.goto(env.p_cmd,10.0,R_goal=port.R_ref,settle_ticks=3)   # rotate to the RL orientation in place
        obs=env.begin_episode()
        steps=0
        while time.monotonic()-t0<budget and steps<cfg['rl_steps']:
            obs,r,term,trunc,info=env.step(agent.act(obs,deterministic=True));steps+=1
            if term:res.update(success=True,phase='rl',t_success_s=round(time.monotonic()-t0,2));break
        res['rl_steps']=steps
    res['end']=env.p_meas()[0].round(2).tolist()
    res['end_xy_from_port_mm']=round(float(np.hypot(*(np.array(res['end'][:2])-port.p_port[:2]))),2)
    res['end_dz_surface_mm']=round(res['end'][2]-port.z_surface,2)
    res['duration_s']=round(time.monotonic()-t0,2)
    env._sync_cmd();env.lift(15.0)
    return res
