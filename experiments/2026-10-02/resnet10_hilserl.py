#!/usr/bin/env python3
# Port of helper2424/resnet10 (Hugging Face, Apache-2.0): the ImageNet ResNet-10 that LeRobot's HIL-SERL loads as
# vision_encoder_name="helper2424/resnet10" (itself a PyTorch port of the JAX encoder used by HIL-SERL / SERL).
# Copied from modeling_resnet.py at the Hub's main revision on 2026-10-05, with the transformers wrappers removed
# (PreTrainedModel, ACT2FN, output dataclasses) so it runs on the offline Jetson without transformers. Module
# names are unchanged, so the Hub's model.safetensors loads with strict=True.
import math

import torch
import torch.nn as nn

WEIGHTS='resnet10_hilserl.safetensors'   # helper2424/resnet10 model.safetensors, beside this file


class MaxPool2dJax(nn.Module):
    """JAX MaxPool with padding='SAME'."""
    def __init__(self,kernel_size,stride=2):
        super().__init__()
        self.kernel_size=kernel_size if isinstance(kernel_size,tuple) else (kernel_size,kernel_size)
        self.stride=stride if isinstance(stride,tuple) else (stride,stride)

    def _compute_padding(self,h,w):
        ph=max(0,(math.ceil(h/self.stride[0])-1)*self.stride[0]+self.kernel_size[0]-h)
        pw=max(0,(math.ceil(w/self.stride[1])-1)*self.stride[1]+self.kernel_size[1]-w)
        return (pw//2,pw-pw//2,ph//2,ph-ph//2)

    def forward(self,x):
        x=nn.functional.pad(x,self._compute_padding(*x.shape[-2:]),value=-float('inf'))
        return nn.functional.max_pool2d(x,kernel_size=3,stride=2,padding=0)


class Conv2dJax(nn.Module):
    """JAX Conv2D with padding='SAME'."""
    def __init__(self,in_channels,out_channels,kernel_size,stride=1,bias=False):
        super().__init__()
        self.kernel_size=kernel_size if isinstance(kernel_size,tuple) else (kernel_size,kernel_size)
        self.stride=stride if isinstance(stride,tuple) else (stride,stride)
        self.conv=nn.Conv2d(in_channels,out_channels,kernel_size=self.kernel_size,stride=self.stride,padding=0,bias=bias)

    def _compute_padding(self,h,w):
        ph=max(0,(math.ceil(h/self.stride[0])-1)*self.stride[0]+self.kernel_size[0]-h)
        pw=max(0,(math.ceil(w/self.stride[1])-1)*self.stride[1]+self.kernel_size[1]-w)
        return (pw//2,pw-pw//2,ph//2,ph-ph//2)

    def forward(self,x):
        return self.conv(nn.functional.pad(x,self._compute_padding(*x.shape[-2:])))


class MyGroupNorm(nn.Module):
    def __init__(self,num_groups,num_channels,eps=1e-5,affine=True):
        super().__init__()
        self.group_norm=nn.GroupNorm(num_groups,num_channels,eps,affine)

    def forward(self,x):
        return self.group_norm(x)


class BasicBlock(nn.Module):
    def __init__(self,in_channels,out_channels,stride=1,norm_groups=4):
        super().__init__()
        self.conv1=Conv2dJax(in_channels,out_channels,kernel_size=3,stride=stride,bias=False)
        self.norm1=MyGroupNorm(norm_groups,out_channels)
        self.conv2=Conv2dJax(out_channels,out_channels,kernel_size=3,stride=1,bias=False)
        self.norm2=MyGroupNorm(norm_groups,out_channels)
        self.shortcut=None
        if in_channels!=out_channels:
            self.shortcut=nn.Sequential(Conv2dJax(in_channels,out_channels,kernel_size=1,stride=stride,bias=False),
                                        MyGroupNorm(norm_groups,out_channels))

    def forward(self,x):
        out=torch.relu(self.norm1(self.conv1(x)))
        out=self.norm2(self.conv2(out))
        identity=self.shortcut(x) if self.shortcut is not None else x
        return torch.relu(out+identity)


class Encoder(nn.Module):
    def __init__(self,embedding_size=64,hidden_sizes=(64,128,256,512)):
        super().__init__()
        self.stages=nn.ModuleList([BasicBlock(embedding_size if i==0 else hidden_sizes[i-1],s,stride=1 if i==0 else 2)
                                   for i,s in enumerate(hidden_sizes)])

    def forward(self,x):
        for stage in self.stages:x=stage(x)
        return x


class ResNet10(nn.Module):
    """forward(x) -> last hidden state (B,512,H/32,W/32); x must already be ImageNet-normalised."""
    def __init__(self,num_channels=3,embedding_size=64,hidden_sizes=(64,128,256,512)):
        super().__init__()
        self.embedder=nn.Sequential(nn.Conv2d(num_channels,embedding_size,kernel_size=7,stride=2,padding=3,bias=False),
                                    MyGroupNorm(4,embedding_size),nn.ReLU(),MaxPool2dJax(kernel_size=3,stride=2))
        self.encoder=Encoder(embedding_size,hidden_sizes)

    def forward(self,x):
        return self.encoder(self.embedder(x))


def load(path):
    from safetensors.torch import load_file
    m=ResNet10();m.load_state_dict(load_file(str(path)),strict=True);return m
