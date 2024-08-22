'''
Author: wenjun-VCC
Date: 2024-08-22 16:11:17
LastEditors: wenjun-VCC
LastEditTime: 2024-08-22 17:30:39
Description: Just can run on Linux Platform and on GPU device.
Email: wenjun.9707@gmail.com
Copyright (c) 2024 by wenjun/VCC, All Rights Reserved. 
'''

import torch
import torch.nn as nn
import numpy as np
import math
from einops import rearrange, repeat
from torchtyping import TensorType
from beartype import beartype
from typing import Optional

from mamba_ssm import Mamba as mamba1
from mamba_ssm import Mamba2 as mamba2



# implementation from paper "Mamba: Linear-Time Sequence Modeling with Selective State Spaces"
# https://arxiv.org/abs/2312.00752

# Mamaba is fast
# Mamba2 is slow


@beartype
def modulate(
    x: TensorType["batch", "seq_len", "dim", float],
    *,
    shift: TensorType["batch", 1, "dim", float]=None,
    scale: TensorType["batch", 1, "dim", float]=None,
):
    
    if scale is not None:
        x = x * (1. + scale)
    if shift is not None:
        x = x + shift
    
    return x



class FeedForward(nn.Module):
    
    def __init__(
        self,
        dim: int,
        out_dim: int=None,
        hidden_dim: int=2048,
        ac_func=nn.GELU,
        dropout: float=None,
    ) -> None:
        super(FeedForward, self).__init__()
        
        self.out_dim = dim if out_dim is None else out_dim
        
        self.fc1 = nn.Linear(in_features=dim, out_features=hidden_dim)
        self.ac_func = ac_func()
        self.dropout = nn.Identity() if dropout is None else nn.Dropout(dropout)
        self.fc2 = nn.Linear(in_features=hidden_dim, out_features=self.out_dim)
        
    
    @beartype   
    def forward(
        self,
        x: TensorType['bs', 'sl', 'dim', float],
    ):
        
        out = self.fc1(x)
        out = self.ac_func(out)
        out = self.dropout(out)
        out = self.fc2(out)
        
        return out
        
    
    
class AdaLNDiMBlock(nn.Module):
    
    def __init__(
        self,
        dim: int,
        d_state: int=64,
        d_conv: int=4,
        mamba_expand: int=2,
        mlp_hidden_dim: int=2048,
        ffd_dropout: float=None,
        block_type: str='mamba1',  # 'mamba1' or 'mamba2'
    ):
        super(AdaLNDiMBlock, self).__init__()
        
        self.norm1 = nn.LayerNorm(dim, elementwise_affine=False, eps=1e-6)
        self.norm2 = nn.LayerNorm(dim, elementwise_affine=False, eps=1e-6)
        mb = mamba1 if block_type == 'mamba1' else mamba2
        self.mb = mb(
            d_model=dim,
            d_state=d_state,
            d_conv=d_conv,
            expand=mamba_expand,
        )
        self.mlp = FeedForward(
            dim=dim,
            hidden_dim=mlp_hidden_dim,
            dropout=ffd_dropout,
        )
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(dim, 6 * dim, bias=True)
        )
    
    
    @beartype
    def forward(
        self,
        x: TensorType['bs', 'sl', 'dim', float],
        cond: TensorType['bs', 1, 'dim', float],  # conclude t and other conditions
    ):
        
        # all params shape [bs, dim]
        gama1, beta1, alpha1, gama2, beta2, alpha2 = self.adaLN_modulation(cond).chunk(6, dim=-1)
        residual = x
        x = modulate(self.norm1(x), shift=gama1, scale=beta1)
        x = self.mb(x)
        x = residual + alpha1 * x
        residual = x
        x = modulate(self.norm2(x), shift=gama2, scale=beta2)
        x = residual + alpha2 * self.mlp(x)
        
        return x

          

class FinalLayer(nn.Module):
    """
    The final layer of DiT.
    """
    def __init__(
        self,
        dim,
        out_dim:int,
    ):
        super(FinalLayer, self).__init__()
        
        self.out_dim = dim if out_dim is None else out_dim
        
        self.norm_final = nn.LayerNorm(dim, elementwise_affine=False, eps=1e-6)
        
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(dim, 2 * dim, bias=True)
        )
        
        self.linear = nn.Linear(dim, self.out_dim, bias=True)


    @beartype
    def forward(
        self,
        x: TensorType['bs', 'sl', 'dim', float],
        cond: TensorType['bs', 'cl', 'dim', float],
    ):
        
        if cond.shape[1] != 1:
            cond = torch.sum(cond, dim=1, keepdim=True)
            
        shift, scale = self.adaLN_modulation(cond).chunk(2, dim=-1)
        x = modulate(self.norm_final(x), shift=shift, scale=scale)
        
        x = self.linear(x)
        
        return x



class DiM(nn.Module):
    
    def __init__(
        self,
        dim: int,
        depth: int=12,
        d_state: int=64,
        d_conv: int=4,
        mamba_expand: int=2,
        ffd_dropout: float=None,
        mlp_hidden_dim: int=2048,
        block_type: str='mamba1',  # 'mamba1' or 'mamba2'
        learn_sigma: bool=False,
    ):
        super(DiM, self).__init__()
        
        self.dim = dim
        self.out_dim = 2 * dim if learn_sigma else dim
        self.learn_sigma = learn_sigma
        
        self.blocks = nn.ModuleList([AdaLNDiMBlock(
            dim=self.dim,
            d_state=d_state,
            d_conv=d_conv,
            mamba_expand=mamba_expand,
            mlp_hidden_dim=mlp_hidden_dim,
            ffd_dropout=ffd_dropout,
            block_type=block_type,
        ) for _ in range(depth)])

        self.final_layer = FinalLayer(
            dim=self.dim,
            out_dim=self.out_dim,
        )
        
        self.initialize_weights()
        
    
    def initialize_weights(self):
        # Initialize transformer layers:
        def _basic_init(module):
            if isinstance(module, nn.Linear):
                torch.nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
        self.apply(_basic_init)

        # Zero-out adaLN modulation layers in DiT blocks:
        for block in self.blocks:
            if hasattr(block, 'adaLN_modulation'):
                nn.init.constant_(block.adaLN_modulation[-1].weight, 0)
                nn.init.constant_(block.adaLN_modulation[-1].bias, 0)
            else:
                continue

        # Zero-out output layers:
        nn.init.constant_(self.final_layer.adaLN_modulation[-1].weight, 0)
        nn.init.constant_(self.final_layer.adaLN_modulation[-1].bias, 0)
        nn.init.constant_(self.final_layer.linear.weight, 0)
        nn.init.constant_(self.final_layer.linear.bias, 0)
        
    
    @beartype
    def forward(
        self,
        x: TensorType['bs', 'sl', 'dim', float],
        t_embed: TensorType['bs', 1, 'dim', float],
        context: Optional[TensorType['bs', 1, 'dim', float]]=None,
    ):
        
        if context is None:
            cond = t_embed
        else:
            cond = t_embed + context

        for block in self.blocks:
            x = block(x, cond)
        
        out = self.final_layer(x, cond)
        # if learn_sigma: output: [bs, sl, 2*dim] {noise, sigma}
        # else: output: [bs, sl, dim] {noise}
        if self.learn_sigma:
            noise, sigma = out.chunk(2, dim=-1)
            return noise, sigma
        
        return out



def DiM_1d_AdaLNDiMBlock(
    dim: int,
    depth: int=12,
    d_state: int=64,
    d_conv: int=4,
    mamba_expand: int=2,
    ffd_dropout: float=None,
    mlp_hidden_dim: int=2048,
    block_type: str='mamba1',  # 'mamba1' or 'mamba2'
    learn_sigma: bool=False,
):
    
    return DiM(
        dim=dim,
        depth=depth,
        d_state=d_state,
        d_conv=d_conv,
        mamba_expand=mamba_expand,
        ffd_dropout=ffd_dropout,
        mlp_hidden_dim=mlp_hidden_dim,
        block_type=block_type,
        learn_sigma=learn_sigma,
    )

