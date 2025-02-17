'''
Author: wenjun-VCC
Date: 2024-06-13 17:31:17
LastEditors: wenjun-VCC
<<<<<<< HEAD
LastEditTime: 2024-10-09 15:16:46
=======
LastEditTime: 2024-10-10 16:53:06
>>>>>>> 88b750596451efcd6486a2ff3b826889dd35420b
Description: __discription:__
Email: wenjun.9707@gmail.com
Copyright (c) 2024 by wenjun/VCC, All Rights Reserved. 
'''

import torch
import torch.nn as nn
from torchtyping import TensorType
from beartype import beartype
from typing import Optional


# implementation from paper "Scalable Diffusion Models with Transformers"
# https://arxiv.org/pdf/2212.09748
# Diffusion Transformer (DiT) architecture
# Three types of DiT blocks are implemented:
#       1,  DiT Block with adaLN-Zero
#       2,  DiT Block with Cross-Attention
#       3,  DiT Block with In-Context Conditioning




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
        dropout: float=0.0,
    ) -> None:
        super(FeedForward, self).__init__()
        
        self.out_dim = dim if out_dim is None else out_dim
        
        self.fc1 = nn.Linear(in_features=dim, out_features=hidden_dim)
        self.ac_func = ac_func()
        self.dropout = nn.Dropout(dropout)
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

    

class AdaLNDiTBlock(nn.Module):
    
    def __init__(
        self,
        dim: int,
        nheads: int=16,
        qkv_bias: bool=True,
        mlp_hidden_dim: int=2048,
        ffd_dropout: float=0.0,
        attn_dropout: float=0.0,
    ):
        super(AdaLNDiTBlock, self).__init__()
        
        self.norm1 = nn.LayerNorm(dim, elementwise_affine=False, eps=1e-6)
        self.norm2 = nn.LayerNorm(dim, elementwise_affine=False, eps=1e-6)
        self.attn = nn.MultiheadAttention(
            embed_dim=dim,
            num_heads=nheads,
            add_bias_kv=qkv_bias,
            dropout=attn_dropout,
            batch_first=True,
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
        x = residual + alpha1 * self.attn(query=x, key=x, value=x, need_weights=False)[0]

        x = x + alpha2 * self.mlp(modulate(self.norm2(x), shift=gama2, scale=beta2))
        
        return x



class CroAttnDitBlock(nn.Module):
    
    def __init__(
        self,
        dim: int,
        nheads: int=16,
        qkv_bias: bool=True,
        mlp_hidden_dim: int=2048,
        ffd_dropout: float=0.0,
        attn_dropout: float=0.0,
    ) -> None:
        super(CroAttnDitBlock, self).__init__()
        
        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)
        self.norm3 = nn.LayerNorm(dim)
        
        self.self_attn = nn.MultiheadAttention(
            embed_dim=dim,
            num_heads=nheads,
            add_bias_kv=qkv_bias,
            dropout=attn_dropout,
            batch_first=True,
        )
        self.cross_attn = nn.MultiheadAttention(
            embed_dim=dim,
            num_heads=nheads,
            add_bias_kv=qkv_bias,
            dropout=attn_dropout,
            batch_first=True,
        )
        self.mlp = FeedForward(
            dim=dim,
            hidden_dim=mlp_hidden_dim,
            dropout=ffd_dropout,
        )
    
    
    @beartype
    def forward(
        self,
        x: TensorType['bs', 'sl', 'dim', float],
        cond: TensorType['bs', 'c', 'dim', float],
    ):
        
        residual = x
        x = self.norm1(x)
        x = residual + self.self_attn(query=x, key=x, value=x, need_weights=False)[0]
        
        residual = x
        x = self.norm2(x)
        x = residual + self.cross_attn(query=x, key=cond, value=cond, need_weights=False)[0]

        x = x + self.mlp(self.norm3(x))
        
        return x



class InContextDiTBlock(nn.Module):
    
    def __init__(
        self,
        dim: int,
        nheads: int=16,
        qkv_bias: bool=True,
        mlp_hidden_dim: int=2048,
        ffd_dropout: float=0.0,
        attn_dropout: float=0.0,
    ):
        super(InContextDiTBlock, self).__init__()
        
        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)
        
        self.self_attn = nn.MultiheadAttention(
            embed_dim=dim,
            num_heads=nheads,
            add_bias_kv=qkv_bias,
            dropout=attn_dropout,
            batch_first=True,
        )
        self.mlp = FeedForward(
            dim=dim,
            hidden_dim=mlp_hidden_dim,
            dropout=ffd_dropout,
        )
        
    
    @beartype
    def forward(
        self,
        x: TensorType['bs', 'sl', 'dim', float],
        cond: TensorType['bs', 'c', 'dim', float],
    ):
       
        sl = x.shape[1]  # valid sequence length
        
        # calculate attention scores with context
        x = torch.cat([x, cond], dim=1)  # x->[bs, sl+c, dim]
        
        residual = x
        x = self.norm1(x)
        x = residual + self.self_attn(query=x, key=x, value=x, need_weights=False)[0]
        
        x = x + self.mlp(self.norm2(x))
        
        return x[:, :sl, :]
  


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



class DiT(nn.Module):
    
    def __init__(
        self,
        dim: int,
        depth: int=12,
        nheads: int=16,
        ffd_dropout: float=0.0,
        attn_dropout: float=0.0,
        mlp_hidden_dim: int=2048,
        qkv_bias: bool=False,
        block = AdaLNDiTBlock,
    ):
        super(DiT, self).__init__()
        
        self.dim = dim
        self.block = block
        
        self.blocks = nn.ModuleList([block(
            dim=self.dim,
            nheads=nheads,
            qkv_bias=qkv_bias,
            mlp_hidden_dim=mlp_hidden_dim,
            ffd_dropout=ffd_dropout,
            attn_dropout=attn_dropout,
        ) for _ in range(depth)])

        self.final_layer = FinalLayer(
            dim=self.dim,
            out_dim=dim,
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
        
        if self.block == AdaLNDiTBlock:
            if context is None:
                cond = t_embed
            else:
                cond = t_embed + context
        else:
            if context is None:
                cond = t_embed
            else:
                cond = torch.cat([t_embed, context], dim=1)
        
        for block in self.blocks:
            x = block(x, cond)
        
        out = self.final_layer(x, cond)
        
        return out



def DiT_1d_AdaLNDiTBlock(
    dim: int,
    depth: int=12,
    nheads: int=16,
    ffd_dropout: float=0.0,
    attn_dropout: float=0.0,
    mlp_hidden_dim: int=2048,
    qkv_bias: bool=False,
):
    
    return DiT(
        dim=dim,
        depth=depth,
        nheads=nheads,
        ffd_dropout=ffd_dropout,
        attn_dropout=attn_dropout,
        mlp_hidden_dim=mlp_hidden_dim,
        qkv_bias=qkv_bias,
        block=AdaLNDiTBlock,
    )


def DiT_1d_CroAttnDitBlock(
    dim: int,
    depth: int=12,
    nheads: int=16,
    ffd_dropout: float=0.0,
    attn_dropout: float=0.0,
    mlp_hidden_dim: int=2048,
    qkv_bias: bool=False,
):
    
    return DiT(
        dim=dim,
        depth=depth,
        nheads=nheads,
        ffd_dropout=ffd_dropout,
        attn_dropout=attn_dropout,
        mlp_hidden_dim=mlp_hidden_dim,
        qkv_bias=qkv_bias,
        block=CroAttnDitBlock,
    )


def DiT_1d_InContextDiTBlock(
    dim: int,
    depth: int=12,
    nheads: int=16,
    ffd_dropout: float=0.0,
    attn_dropout: float=0.0,
    mlp_hidden_dim: int=2048,
    qkv_bias: bool=False,
):
    
    return DiT(
        dim=dim,
        depth=depth,
        nheads=nheads,
        ffd_dropout=ffd_dropout,
        attn_dropout=attn_dropout,
        mlp_hidden_dim=mlp_hidden_dim,
        qkv_bias=qkv_bias,
        block=InContextDiTBlock,
    )