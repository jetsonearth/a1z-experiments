#!/usr/bin/env python3
"""A1Z ACT training using LeRobot 0.4.4 policy, dataset and processors."""
import argparse
import json
from pathlib import Path
import time
import torch
from lerobot.configs.types import FeatureType
from lerobot.datasets.lerobot_dataset import LeRobotDataset,LeRobotDatasetMetadata
from lerobot.datasets.utils import dataset_to_policy_features
from lerobot.policies.act.configuration_act import ACTConfig
from lerobot.policies.act.modeling_act import ACTPolicy
from lerobot.policies.act.processor_act import make_act_pre_post_processors


def main():
    p=argparse.ArgumentParser();p.add_argument('--dataset',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--repo-id',default='local/a1z_usb_task');p.add_argument('--steps',type=int,default=2000)
    p.add_argument('--batch-size',type=int,default=8);p.add_argument('--device',default='cuda');p.add_argument('--smoke',action='store_true')
    a=p.parse_args()
    if a.output.exists():raise ValueError('Output exists; choose a new run directory')
    if a.steps<1:raise ValueError('steps must be positive')
    report=json.loads((a.dataset/'source-report.json').read_text())
    if any(x['smoke_only'] for x in report['episodes']) and not a.smoke:raise ValueError('Diagnostic dataset requires --smoke')
    torch.manual_seed(42);torch.set_num_threads(4)
    m=LeRobotDatasetMetadata(a.repo_id,root=a.dataset)
    ft=dataset_to_policy_features(m.features)
    ins={k:v for k,v in ft.items() if k in ['observation.state','observation.images.wrist']}
    outs={k:v for k,v in ft.items() if v.type is FeatureType.ACTION}
    cfg=ACTConfig(input_features=ins,output_features=outs,device=a.device)
    if a.smoke:cfg.pretrained_backbone_weights=None
    ds=LeRobotDataset(a.repo_id,root=a.dataset,delta_timestamps={'action':[i/m.fps for i in cfg.action_delta_indices]})
    policy=ACTPolicy(cfg).to(a.device);pre,post=make_act_pre_post_processors(cfg,dataset_stats=m.stats)
    optimizer=cfg.get_optimizer_preset().build(policy.parameters())
    loader=torch.utils.data.DataLoader(ds,batch_size=min(a.batch_size,len(ds)),shuffle=True,drop_last=False,num_workers=0)
    a.output.mkdir(parents=True)
    iterator=iter(loader);losses=[];start=time.monotonic();policy.train()
    initial=next(policy.parameters()).detach().cpu().clone()
    for step in range(a.steps):
        try:batch=next(iterator)
        except StopIteration:iterator=iter(loader);batch=next(iterator)
        loss,metrics=policy.forward(pre(batch))
        if not torch.isfinite(loss):raise RuntimeError('Nonfinite loss')
        loss.backward();optimizer.step();optimizer.zero_grad(set_to_none=True)
        losses.append(float(loss.detach()))
        if step%20==0 or step==a.steps-1:print(json.dumps({'step':step+1,'loss':losses[-1],'elapsed_s':time.monotonic()-start}),flush=True)
    changed=not torch.equal(initial,next(policy.parameters()).detach().cpu())
    if not changed:raise RuntimeError('Parameters did not update')
    checkpoint=a.output/'checkpoint';policy.save_pretrained(checkpoint);pre.save_pretrained(checkpoint);post.save_pretrained(checkpoint)
    from policy_runtime import load_checkpoint
    loaded,reloaded_pre,reloaded_post=load_checkpoint(checkpoint,a.device)
    observation={k:ds[0][k] for k in ins}
    with torch.inference_mode():action=reloaded_post(loaded.select_action(reloaded_pre(observation)))
    if tuple(action.shape[-1:])!=tuple(cfg.output_features['action'].shape) or not torch.isfinite(action).all():raise RuntimeError('Invalid reloaded action')
    result={'smoke_only':a.smoke,'steps':a.steps,'frames':len(ds),'episodes':len(report['episodes']),
            'parameter_update_verified':changed,'checkpoint_reload_inference_verified':True,
            'elapsed_s':time.monotonic()-start,'first_loss':losses[0],'last_loss':losses[-1],
            'physical_eval_completed':False,'source_dataset':str(a.dataset),'losses':losses}
    (a.output/'training-report.json').write_text(json.dumps(result,indent=2))
    print('TRAINING_RESULT',a.output/'training-report.json',flush=True)
if __name__=='__main__':main()
