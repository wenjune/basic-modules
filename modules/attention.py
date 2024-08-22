'''
Author: wenjun-VCC
Date: 2024-07-30 23:27:22
LastEditors: wenjun-VCC
LastEditTime: 2024-08-17 16:59:25
Description: __discription:__
Email: wenjun.9707@gmail.com
Copyright (c) 2024 by wenjun/VCC, All Rights Reserved. 
'''
import math
import torch
from torch import nn, einsum
import torch.nn.functional as F

from beartype import beartype
from typing import Optional
from torchtyping import TensorType

from einops import rearrange, reduce, repeat


def conv_nd(ndim):
    
    if ndim == 1:
        return nn.Conv1d
    elif ndim == 2:
        return nn.Conv2d
    elif ndim == 3:
        return nn.Conv3d
    else:
        raise ValueError("Unsupported ndim: {}".format(ndim))



class ConvAttention(nn.Module):
    
    def __init__(self, dim, ndim, nheads = 8, dim_head = 32):
        super(ConvAttention, self).__init__()
        
        self.scale = dim_head ** -0.5
        self.heads = nheads
        self.ndim = ndim
        hidden_dim = dim_head * nheads

        self.to_qkv = conv_nd(ndim)(dim, hidden_dim * 3, 1, bias = False)
        self.to_out = conv_nd(ndim)(hidden_dim, dim, 1)


    def forward(
        self,
        x,
        *,
        return_atten_score: bool=False,
    ):
        
        # for convolutional data, the second dimension is the channel(feature) dimension.
        qkv = self.to_qkv(x).chunk(3, dim = 1)
        if self.ndim == 1:
            # if conv1d, datashape is [b, d, seq_len]
            q, k, v = map(lambda t: rearrange(t, 'b (h c) l -> b h c l', h = self.heads), qkv)
        elif self.ndim == 2:
            # if conv2d, datashape is [b, c, h, w]
            q, k, v = map(lambda t: rearrange(t, 'b (h c) x y -> b h c (x y)', h = self.heads), qkv)
        elif self.ndim == 3:
            # if conv3d, datashape is [b, c, d, h, w]
            q, k, v = map(lambda t: rearrange(t, 'b (h c) x y z -> b h c (x y z)', h = self.heads), qkv)
        else:
            raise ValueError("Unsupported ndim: {}".format(self.ndim))

        # query:[bs, heads, dim, ql] x key:[bs, heads, dim, kl] -> sim:[bs, heads, ql, kl]
        sim = einsum('b h d i, b h d j -> b h i j', q, k) * self.scale
        attn = sim.softmax(dim = -1)
        # attn:[bs, heads, ql, kl] x value:[bs, heads, dim, kl] -> out:[bs, heads, ql, dim]
        out = einsum('b h i j, b h d j -> b h i d', attn, v)

        if self.ndim == 1:
            out = rearrange(out, 'b h l d -> b (h d) l')
        elif self.ndim == 2:
            out = rearrange(out, 'b h (x y) d -> b (h d) x y', x = x.shape[-2], y = x.shape[-1])
        elif self.ndim == 3:
            out = rearrange(out, 'b h (x y z) d -> b (h d) x y z', x = x.shape[-3], y = x.shape[-2], z = x.shape[-1])
        else:
            raise ValueError("Unsupported ndim: {}".format(self.ndim))
        out = self.to_out(out)
        
        if return_atten_score:
            return out, attn
        
        else:
            return out
    
    
class MHA(nn.Module):
    
    def __init__(
        self,
        dim: int,
        nheads: int=8,
        dim_head: int=None,
        atten_dropout: float=None,
    ) -> None:
        super(MHA, self).__init__()
        
        self.dim = dim
        self.dim_head = dim//nheads if dim_head is None else dim_head
        self.hidden_dim = dim if dim_head is None else dim_head * nheads
        self.nheads = nheads
        self.scale = self.dim_head ** -0.5
        
        self.dropout = nn.Identity() if atten_dropout is None else nn.Dropout(atten_dropout)
        
        self.to_qkv = nn.Linear(dim, self.hidden_dim*3, bias=False)
        self.Ow = nn.Linear(self.hidden_dim, dim)
        
    
    @beartype
    def attention(
        self,
        query: TensorType['b', 'ql', 'dim', float],
        key: TensorType['b', 'ql', 'dim', float],
        value: TensorType['b', 'ql', 'dim', float],
        *,  # force to use keyword arguments
        padding_mask: Optional[TensorType['bs', 1, 1, 'kl', bool]]=None,  # just for 1d attention
    ):
        ''' 
        Args:
            q, k, v: [bs, q_len, dk, float]
            padding_mask: [bs, 1, 1, k_len, bool]
        Returns:
            output: [bs, q_len, (nh*dk)]
            atten_scores: [bs, nh, q_len, k_len]
        '''
        # reshape (Q, K, V) to multi-head vector
        # query, key, value : ->[bs, query_len, heads, dk]
        query = rearrange(query, 'b q (h dk) -> b q h dk', dk=self.dim_head)
        key = rearrange(key, 'b k (h dk) -> b k h dk', dk=self.dim_head)
        value = rearrange(value, 'b v (h dk) -> b v h dk', dk=self.dim_head)

        # calculate attention scores: [batch_size, heads, query_len, key_len]
        attention_scores = torch.einsum('nqhd, nkhd -> nhqk', [query, key]) * self.scale
        
        # apply masks
        # apply padding mask
        attention_scores = self._apply_mask(
            score=attention_scores,
            mask=padding_mask,
        )

        # calculate attention distribution
        # attention_scores = [batch_size, heads, query_len, key_len]
        attention_scores = attention_scores.softmax(dim=-1)
        attention_scores = self.dropout(attention_scores)
        
        # value = [batch_size, value_len, heads, d_k]
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
        x: TensorType['bs', 'ql', 'dim', float],
        *,  # force to use keyword arguments
        padding_mask: Optional[TensorType['bs', 'kl', bool]]=None,  # key padding
        return_atten_score: bool=False,
    ):
        """ MultiheadAttention forward
            include kv_cache

        Args:
            x (TensorType[bs, ql, dim, float]): in self-atten q=k=v
            padding_mask (TensorType[bs, kl, bool], optional): mask the padding locations. Defaults to None.

        Returns:
            tuple(out_put, new_cache, atten_score): if not use_cache->new_cache is None
        """
        
        query, key, value = self.to_qkv(x).chunk(3, dim=-1)
        
        
        if padding_mask is not None:   
            padding_mask = rearrange(padding_mask, 'bs kl -> bs 1 1 kl')
        
        output, attention_scores = self.attention(
            query, key, value,
            padding_mask=padding_mask,
        )
        
        output = self.Ow(output)
        
        if return_atten_score:
            return output, attention_scores
        
        return output


    