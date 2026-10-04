"""Shared offline/live ACT checkpoint loader; no robot connections."""
from pathlib import Path
import torch
from lerobot.configs.policies import PreTrainedConfig
from lerobot.policies.act.modeling_act import ACTPolicy
from lerobot.policies.factory import make_pre_post_processors
from safetensors.torch import load_model


def load_checkpoint(path,device='cpu'):
    path=Path(path)
    cfg=PreTrainedConfig.from_pretrained(path)
    if cfg.type!='act':raise ValueError('Expected ACT checkpoint')
    cfg.device=device
    # Full checkpoint weights are loaded below; no backbone download needed.
    cfg.pretrained_backbone_weights=None
    model=ACTPolicy(cfg).to(device)
    load_model(model,str(path/'model.safetensors'))
    model.eval();model.reset()
    pre,post=make_pre_post_processors(cfg,pretrained_path=str(path),
        preprocessor_overrides={'device_processor':{'device':device},'normalizer_processor':{'device':device}},
        postprocessor_overrides={'device_processor':{'device':'cpu'},'unnormalizer_processor':{'device':device}})
    return model,pre,post
