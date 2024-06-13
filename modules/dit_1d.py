'''
Author: wenjun-VCC
Date: 2024-06-13 17:31:17
LastEditors: wenjun-VCC
LastEditTime: 2024-06-14 03:59:36
FilePath: dit_1d.py
Description: __discription:__
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
    shift: TensorType["batch", 1, "dim", float]=None,
    scale: TensorType["batch", 1, "dim", float]=None,
):
    
    if scale is not None:
        x = x * (1. + scale)
    if shift is not None:
        x = x + shift
    
    return x



class MLP(nn.Module):
    
    def __init__(
        self,
        dim: int,
        out_dim: int=None,
        hidden_dim: int=2048,
        ac_func=nn.GELU,
        dropout: float=None,
    ) -> None:
        super(MLP, self).__init__()
        
        self.out_dim = dim if out_dim is None else out_dim
        
        self.fc1 = nn.Linear(in_features=dim, out_features=hidden_dim)
        self.ac_func = ac_func()
        self.dropout = nn.Identity() if dropout is None else nn.Dropout(dropout)
        self.fc2 = nn.Linear(in_features=hidden_dim, out_features=self.out_dim)
        
    
    @beartype   
    def forward(
        self,
        x: TensorType['bs','sl', 'dim', float],
    ):
        
        out = self.fc1(x)
        out = self.ac_func(out)
        out = self.dropout(out)
        out = self.fc2(out)
        
        return out
        


class MultiHeadAttention(nn.Module):
    
    def __init__(
        self,
        d_model: int,
        n_heads: int=8,
        qkv_bias: bool=False,
        is_causal: bool=False,
        atten_dropout: float=None,
    ) -> None:
        super(MultiHeadAttention, self).__init__()
        
        assert d_model % n_heads == 0, "d_model should divisible by n_heads != 0!"
        
        self.d_k = d_model // n_heads
        self.d_model = d_model
        self.n_heads = n_heads
        self.is_causal = is_causal
        self.scale = 1./math.sqrt(self.d_k)
        self.dropout = nn.Identity() if atten_dropout is None else nn.Dropout(atten_dropout)
        
        self.Qw = nn.Linear(d_model, d_model, bias=qkv_bias)  # Wq
        self.Kw = nn.Linear(d_model, d_model, bias=qkv_bias)  # Wk
        self.Vw = nn.Linear(d_model, d_model, bias=qkv_bias)  # Wv
        self.Ow = nn.Linear(d_model, d_model, bias=True)
        
    
    @beartype
    def attention(
        self,
        query: TensorType['b', 'ql', 'dim', float],
        key: TensorType['b', 'kl', 'dim', float],
        value: TensorType['b', 'vl', 'dim', float],
        *,  # force to use keyword arguments
        key_padding_mask: Optional[TensorType['bs', 1, 1, 'kl', bool]]=None,
        causal_mask: Optional[TensorType[1, 1, 'ql', 'kl', bool]]=None,
    ):
        ''' 
            Params:
                query: [bs, q_len, dk, float]
                key: [bs, k_len, dk, float]
                value: [bs, v_len, dk, float]
                key_padding_mask: [bs, 1, 1, k_len, bool]
                causal_mask: [1, 1, q_len, k_len, bool]
            Return:
                output: [bs, q_len, (nh*dk)]
                atten_scores: [bs, nh, q_len, k_len]
        '''
        # reshape (Q, K, V) to multi-head vector
        # query : ->[bs, query_len, heads, dk]
        query = rearrange(query, 'b q (h dk) -> b q h dk', dk=self.d_k)
        # key   : ->[bs, key_len, heads, dk]
        key = rearrange(key, 'b k (h dk) -> b k h dk', dk=self.d_k)
        # value : ->[bs, value_len, heads, dk]
        value = rearrange(value, 'b v (h dk) -> b v h dk', dk=self.d_k)
        
        # calculate attention scores: [batch_size, heads, query_len, key_len]
        attention_scores = torch.einsum('nqhd, nkhd -> nhqk', [query, key]) * self.scale
        
        # apply masks
        # apply key padding mask
        attention_scores = self._apply_mask(
            score=attention_scores,
            mask=key_padding_mask,
        )
        # apply causal mask
        attention_scores = self._apply_mask(
            score=attention_scores,
            mask=causal_mask,
        )

        # calculate attention distribution
        attention_scores = attention_scores.softmax(dim=-1)
        attention_scores = self.dropout(attention_scores)
        
        # attention_scores = [batch_size, heads, query_len, key_len]
        # value = [batch_size, value_len, heads, d_k]
        # key_len = value_len [batch_size, query_len, heads, d_k]
        output = torch.einsum('nhql, nlhd -> nqhd', [attention_scores, value])
        output = rearrange(output, 'b s h d -> b s (h d)')
        
        return output, attention_scores
    
    
    @beartype
    def _apply_mask(
        self,
        *,  # force to use keyword arguments
        score: TensorType['bs', 'h', 'ql', 'kl', float],
        mask: Optional[TensorType['bs', 1, 1, 'kl', bool]]=None,
    ):
        
        if mask is not None:
            
            score = score.masked_fill(~mask, float(-1e10))
            
            return score
        
        else:
            
            return score
        
        
    @beartype
    def forward(
        self,
        query: TensorType['bs', 'ql', 'dim', float],
        key: TensorType['bs', 'kl', 'dim', float],
        value: TensorType['bs', 'vl', 'dim', float],
        *,  # force to use keyword arguments
        return_scores: bool=False,
        key_padding_mask: Optional[TensorType['bs', 'kl', bool]]=None,  # key padding
    ):
        """ MultiheadAttention forward
            include kv_cache

        Args:
            query (TensorType[bs, ql, dim, float]): in self-atten q=k=v
            key (TensorType[bs, kl, dim, float]): kl=vl
            value (TensorType[bs, vl, dim, float]):
            key_padding_mask (TensorType[bs, kl, bool], optional): mask the padding locations. Defaults to None.
            return_scores (bool, optional): return attention scores. Defaults to False.
        Returns:
            tuple(output, atten_score) or (output)
        """
        
        query = self.Qw(query)
        key = self.Kw(key)
        value = self.Vw(value)
        
        causal_mask = None
        if self.is_causal:
            # usually we just use causal mask in decoder self attention
            # ql = kl = vl
            causal_mask = self._make_causal_mask(query)
            causal_mask = rearrange(causal_mask, '... -> 1 1 ...')  # [1, 1, sl, sl]
        if key_padding_mask is not None:
            key_padding_mask = rearrange(key_padding_mask, 'bs kl -> bs 1 1 kl')
        
        output, attention_scores = self.attention(
            query, key, value,
            key_padding_mask=key_padding_mask,
            causal_mask=causal_mask,
        )
        
        output = self.Ow(output)
        
        if return_scores:
            return output, attention_scores

        return output
        

    @beartype
    def _make_causal_mask(
        self,
        tgt: TensorType['bs', 'sl', 'dim' ,float],
    ):
        """causal mask for autoregressive task

        Args:
            tgt : query

        Returns:
            bool_mask[1, sl, sl] : [[true, false, false],
                                    [ture, true,  false],
                                    [true, true,  true ]]
        """
        
        length = tgt.shape[1]
        mask = torch.tril(torch.ones((length, length), dtype=torch.bool, device=tgt.device))
        
        return mask
    
    

class AdaLNDiTBlock(nn.Module):
    
    def __init__(
        self,
        dim: int,
        nheads: int=8,
        qkv_bias: bool=True,
        mlp_hidden_dim: int=2048,
        mlp_dropout: float=None,
        atten_dropout: float=None,
    ):
        super(AdaLNDiTBlock, self).__init__()
        
        self.norm1 = nn.LayerNorm(dim, elementwise_affine=False, eps=1e-6)
        self.norm2 = nn.LayerNorm(dim, elementwise_affine=False, eps=1e-6)
        self.attn = MultiHeadAttention(
            d_model=dim,
            n_heads=nheads,
            qkv_bias=qkv_bias,
            atten_dropout=atten_dropout,
        )
        self.mlp = MLP(
            dim=dim,
            hidden_dim=mlp_hidden_dim,
            dropout=mlp_dropout,
        )
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(dim, 6 * dim, bias=True)
        )
    
    
    @beartype
    def forward(
        self,
        x: TensorType['bs', 'sl', 'dim', float],
        cond: TensorType['bs', 1, 'dim', float],
    ):
        
        # all params shape [bs, dim]
        gama1, beta1, alpha1, gama2, beta2, alpha2 = self.adaLN_modulation(cond).chunk(6, dim=-1)
        residual = x
        x = modulate(self.norm1(x), gama1, beta1)
        x = self.attn(x, x, x)
        x = residual + alpha1 * x
        residual = x
        x = modulate(self.norm2(x), gama2, beta2)
        x = residual + alpha2 * self.mlp(x)
        
        return x



class CroAttnDitBlock(nn.Module):
    
    def __init__(
        self,
        dim: int,
        nheads: int=8,
        qkv_bias: bool=True,
        mlp_hidden_dim: int=2048,
        mlp_dropout: float=None,
        atten_dropout: float=None,
    ) -> None:
        super(CroAttnDitBlock, self).__init__()
        
        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)
        self.norm3 = nn.LayerNorm(dim)
        
        self.self_attn = MultiHeadAttention(
            d_model=dim,
            n_heads=nheads,
            qkv_bias=qkv_bias,
            atten_dropout=atten_dropout,
        )
        self.cross_attn = MultiHeadAttention(
            d_model=dim,
            n_heads=nheads,
            qkv_bias=qkv_bias,
            atten_dropout=atten_dropout,
        )
        self.mlp = MLP(
            dim=dim,
            hidden_dim=mlp_hidden_dim,
            dropout=mlp_dropout,
        )
    
    
    @beartype
    def forward(
        self,
        x: TensorType['bs', 'sl', 'dim', float],
        cond: TensorType['bs', 2, 'dim', float],
    ):
        
        residual = x
        x = self.norm1(x)
        x = residual + self.self_attn(x, x, x)
        residual = x
        x = self.norm2(x)
        x = residual + self.cross_attn(x, cond, cond)
        residual = x
        x = self.norm3(x)
        x = residual + self.mlp(x)
        
        return x



class InContextDiTBlock(nn.Module):
    
    def __init__(
        self,
        dim: int,
        nheads: int=8,
        qkv_bias: bool=True,
        mlp_hidden_dim: int=2048,
        mlp_dropout: float=None,
        atten_dropout: float=None,
    ):
        super(InContextDiTBlock, self).__init__()
        
        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)
        
        self.self_attn = MultiHeadAttention(
            d_model=dim,
            n_heads=nheads,
            qkv_bias=qkv_bias,
            atten_dropout=atten_dropout,
        )
        self.mlp = MLP(
            dim=dim,
            hidden_dim=mlp_hidden_dim,
            dropout=mlp_dropout,
        )
        
    
    @beartype
    def forward(
        self,
        x: TensorType['bs', 'sl', 'dim', float],
        cond: TensorType['bs', 2, 'dim', float],
    ):
       
        sl = x.shape[1]  # valid sequence length
        
        x = torch.cat([x, cond], dim=1)
        residual = x
        x = self.norm1(x)
        x = self.self_attn(x, x, x)
        x = residual + x
        residual = x
        x = self.norm2(x)
        x = residual + self.mlp(x)
        
        return x[:, :-2, :]
        
        

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
        
        self.out_linear = nn.Linear(dim, self.out_dim, bias=True)


    def forward(
        self,
        x: TensorType['bs', 'sl', 'dim', float],
        cond: TensorType['bs', 'cl', 'dim', float],
    ):
        
        if cond.shape[1] != 1:
            cond = torch.sum(cond, dim=1, keepdim=True)
            
        shift, scale = self.adaLN_modulation(cond).chunk(2, dim=-1)
        x = modulate(self.norm_final(x), shift, scale)
        
        x = self.out_linear(x)
        
        return x



class DiT(nn.Module):
    
    def __init__(
        self,
        dim: int,
        depth: int=12,
        nheads: int=8,
        mlp_dropout: float=None,
        atten_dropout: float=None,
        mlp_hidden_dim: int=2048,
        qkv_bias: bool=True,
        learn_sigma: bool=False,
        block = AdaLNDiTBlock,
    ):
        super(DiT, self).__init__()
        
        self.dim = dim
        self.out_dim = 2 * dim if learn_sigma else dim
        self.block = block
        self.learn_sigma = learn_sigma
        
        self.blocks = nn.ModuleList([block(
            dim=self.dim,
            nheads=nheads,
            qkv_bias=qkv_bias,
            mlp_hidden_dim=mlp_hidden_dim,
            mlp_dropout=mlp_dropout,
            atten_dropout=atten_dropout,
        ) for _ in range(depth)])

        self.final_layer = FinalLayer(
            dim=self.dim,
            out_dim=self.out_dim,
        )
        
    
    @beartype
    def forward(
        self,
        x: TensorType['bs', 'sl', 'dim', float],
        t_embed: TensorType['bs', 1, 'dim', float],
        context: TensorType['bs', 1, 'dim', float],
    ):
        
        if self.block == AdaLNDiTBlock:
            cond = t_embed + context
        else:
            cond = torch.cat([t_embed, context], dim=1)
        
        for block in self.blocks:
            x = block(x, cond)
        
        out = self.final_layer(x, cond)
        # if learn_sigma: output: [bs, sl, 2*dim] {noise, sigma}
        # else: output: [bs, sl, dim] {noise}
        if self.learn_sigma:
            noise, sigma = out.chunk(2, dim=-1)
            return noise, sigma
        
        return out



def DiT_1d_AdaLNDiTBlock(
    dim: int,
    depth: int=12,
    nheads: int=8,
    mlp_dropout: float=None,
    atten_dropout: float=None,
    mlp_hidden_dim: int=2048,
    qkv_bias: bool=True,
    learn_sigma: bool=False,
):
    
    return DiT(
        dim=dim,
        depth=depth,
        nheads=nheads,
        mlp_dropout=mlp_dropout,
        atten_dropout=atten_dropout,
        mlp_hidden_dim=mlp_hidden_dim,
        qkv_bias=qkv_bias,
        learn_sigma=learn_sigma,
        block=AdaLNDiTBlock,
    )


def DiT_1d_CroAttnDitBlock(
    dim: int,
    depth: int=12,
    nheads: int=8,
    mlp_dropout: float=None,
    atten_dropout: float=None,
    mlp_hidden_dim: int=2048,
    qkv_bias: bool=True,
    learn_sigma: bool=False,
):
    
    return DiT(
        dim=dim,
        depth=depth,
        nheads=nheads,
        mlp_dropout=mlp_dropout,
        atten_dropout=atten_dropout,
        mlp_hidden_dim=mlp_hidden_dim,
        qkv_bias=qkv_bias,
        learn_sigma=learn_sigma,
        block=CroAttnDitBlock,
    )


def DiT_1d_InContextDiTBlock(
    dim: int,
    depth: int=12,
    nheads: int=8,
    mlp_dropout: float=None,
    atten_dropout: float=None,
    mlp_hidden_dim: int=2048,
    qkv_bias: bool=True,
    learn_sigma: bool=False,
):
    
    return DiT(
        dim=dim,
        depth=depth,
        nheads=nheads,
        mlp_dropout=mlp_dropout,
        atten_dropout=atten_dropout,
        mlp_hidden_dim=mlp_hidden_dim,
        qkv_bias=qkv_bias,
        learn_sigma=learn_sigma,
        block=InContextDiTBlock,
    )