'''
Author: wenjun-VCC
Date: 2024-07-30 23:27:22
LastEditors: wenjun-VCC
LastEditTime: 2024-09-08 04:05:41
Description: __discription:__
Email: wenjun.9707@gmail.com
Copyright (c) 2024 by wenjun/VCC, All Rights Reserved. 
'''
import torch
from torch import nn, einsum

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
        nheads: int=16,
        dim_head: int=64,
        qkv_bias: bool=True,
        attn_drop: float=0.,
        proj_drop: float=0.,
    ) -> None:
        super(MHA, self).__init__()
        
        self.dim = dim
        self.dim_head = dim//nheads if dim_head is None else dim_head
        self.hidden_dim = dim if dim_head is None else dim_head * nheads
        self.nheads = nheads
        self.scale = self.dim_head ** -0.5
        
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj_drop = nn.Dropout(proj_drop)
        
        self.to_q = nn.Linear(dim, self.hidden_dim, bias=qkv_bias)
        self.to_kv = nn.Linear(dim, self.hidden_dim*2, bias=qkv_bias)
        self.Ow = nn.Linear(self.hidden_dim, dim)
        
        self.apply(self.weight_init)
        
    
    def weight_init(self, m):
        
        if isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        
    
    @beartype
    def attention(
        self,
        query: TensorType['b', 'ql', 'dim', float],
        key: TensorType['b', 'ql', 'dim', float],
        value: TensorType['b', 'ql', 'dim', float],
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

        # calculate attention distribution
        # attention_scores = [batch_size, heads, query_len, key_len]
        attention_scores = attention_scores.softmax(dim=-1)
        attention_scores = self.attn_drop(attention_scores)
        
        # value = [batch_size, value_len, heads, d_k]
        output = torch.einsum('nhql, nlhd -> nqhd', [attention_scores, value])
        output = rearrange(output, 'b s h d -> b s (h d)')
        
        return output, attention_scores
        
        
    @beartype
    def forward(
        self,
        q: TensorType['bs', 'ql', 'dim', float],
        *,  # force to use keyword arguments
        kv: Optional[TensorType['bs', 'kl', 'dim', float]]=None,
        return_atten_score: bool=False,
    ):
        """ MultiheadAttention forward
            include kv_cache

        Args:
            x (TensorType[bs, ql, dim, float]): in self-atten q=k=v

        Returns:
            tuple(out_put, new_cache, atten_score): if not use_cache->new_cache is None
        """
        
        query = self.to_q(q)
        if kv is None:
            key, value = self.to_kv(q).chunk(2, dim=-1)
        else:
            key, value = self.to_kv(kv).chunk(2, dim=-1)
        
        output, attention_scores = self.attention(
            query, key, value,
        )
        
        output = self.Ow(output)
        output = self.proj_drop(output)
        
        if return_atten_score:
            return output, attention_scores
        
        return output




# Agent Attention, borrowed from https://github.com/LeapLabTHU/Agent-Attention/tree/master
# Simple implementation of Agent Attention for 1D sequence data
# reference: https://arxiv.org/pdf/2312.08874
class AgentAttention(nn.Module):
    
    def __init__(
        self,
        dim,
        nheads=16,
        nagents=64,
        seq_len=384,
        qkv_bias=False,
        attn_drop=0.,
        proj_drop=0.,
    ):
        super(AgentAttention, self).__init__()
        
        # Out = Proj(Attn(Q, A, Attn(A, K, V)))
        
        assert dim % nheads == 0, f"dim {dim} should be divided by num_heads {nheads}."

        self.nheads = nheads
        head_dim = dim // nheads
        self.scale = head_dim ** -0.5
        
        # learnable parameters
        # using learnable parameters to represent the agent tokens
        self.agent_token = nn.Parameter(torch.zeros(1, nagents, dim))
        # bias
        # (1, nh, agent_num, sl) like position embedding
        self.position_bias = nn.Parameter(torch.zeros(1, nheads, nagents, seq_len))
        # (1, nh, sl, agent_num)
        self.agent_bias = nn.Parameter(torch.zeros(1, nheads, seq_len, nagents))

        self.softmax = nn.Softmax(dim=-1)
        
        self.to_q = nn.Linear(dim, dim, bias=qkv_bias)
        self.to_kv = nn.Linear(dim, dim * 2, bias=qkv_bias)
        self.out_proj = nn.Linear(dim, dim)
        
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj_drop = nn.Dropout(proj_drop)
        
        nn.init.trunc_normal_(self.agent_token, std=.02)
        nn.init.trunc_normal_(self.position_bias, std=.02)
        nn.init.trunc_normal_(self.agent_bias, std=.02)
        
        self.apply(self.weight_init)
        
    
    def weight_init(self, m):
        
        if isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)


    @beartype
    def forward(
        self,
        x: TensorType['bs', 'sl', 'dim', float],
    ):
        
        q = self.to_q(x)  # query (bs, sl, dim)
        k, v = self.to_kv(x).chunk(2, dim=-1)  # k,v (bs, sl, dim)

        # (bs, sl, dim) -> (bs, nh, sl, dk)
        rearrange_qkv = lambda x: rearrange(x, 'b l (h d) -> b h l d', h=self.nheads)
        q, k, v = map(rearrange_qkv, (q, k, v))
        
        # agent token
        # (1, agent_num, dim) -> (1, nh, agent_num, dk)
        agent_tokens = rearrange_qkv(self.agent_token)

        # agent attention 
        # (1, nh, agent_num, dk) x (bs, nh, dk, sl) -> (bs, nh, agent_num, sl)
        agent_attn = self.softmax((agent_tokens * self.scale) @ k.transpose(-2, -1) + self.position_bias)
        agent_attn = self.attn_drop(agent_attn)
        # (bs, nh, agent_num, sl) x (bs, nh, sl, dk) -> (bs, nh, agent_num, dk)
        agent_v = agent_attn @ v  # Attn(Softmax((agents * scale) @ k + position_bias) @ v)
        
        # query attention
        # (bs, nh, sl, dk) x (bs, nh, dk, agent_num) -> (bs, nh, sl, agent_num)
        q_attn = self.softmax((q * self.scale) @ agent_tokens.transpose(-2, -1) + self.agent_bias)
        q_attn = self.attn_drop(q_attn)
        # (bs, nh, sl, agent_num) x (bs, nh, agent_num, dk) -> (bs, nh, sl, dk)
        x = q_attn @ agent_v  # Attn(Softmax(q * scale @ agents + agent_bias) @ agent_v)
        
        x = rearrange(x, 'b h l d -> b l (h d)')

        x = self.out_proj(x)
        x = self.proj_drop(x)
        
        return x
    


# Simple implementation of Mediator Attention for 1D sequence data
# reference: https://arxiv.org/pdf/2408.05710v1
class MediatorAttention(nn.Module):
    
    def __init__(
        self,
        dim: int,
        nheads: int=16,
        nmediators: int=64,
        qkv_bias: bool=False,
        attn_drop: float=0.,
        proj_drop: float=0.,
    ):
        super(MediatorAttention, self).__init__()
        
        # Out = Proj(Softmax(Q @ Meditors.T * scale) * Softmax(Mediators @ K * scale) @ V)
        
        assert dim % nheads == 0, f"dim {dim} should be divided by num_heads {nheads}."
        
        self.dim = dim
        self.nheads = nheads
        self.scale = (dim // nheads) ** -0.5
        
        # learnable parameters
        # mediator tokens
        self.mediator_token = nn.Parameter(torch.zeros(1, nmediators, dim))
        
        self.to_q = nn.Linear(dim, dim, bias=qkv_bias)
        self.to_kv = nn.Linear(dim, dim*2, bias=qkv_bias)
        self.out_proj = nn.Linear(dim, dim)
        
        self.soft_max = nn.Softmax(dim=-1)
        
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj_drop = nn.Dropout(proj_drop)
        
        nn.init.trunc_normal_(self.mediator_token, std=.02)
        
        self.apply(self.weight_init)
        
    
    def weight_init(self, m):
        
        if isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        
    
    @beartype
    def forward(
        self,
        x: TensorType['bs', 'sl', 'dim', float],
    ):
        
        q = self.to_q(x)
        k, v = self.to_kv(x).chunk(2, dim=-1)
        
        # (bs, sl, dim) -> (bs, nh, sl, dk)
        rearrange_qkv = lambda x: rearrange(x, 'b l (h d) -> b h l d', h=self.nheads)
        q, k, v = map(rearrange_qkv, (q, k, v))
        
        # agent token
        # (1, agent_num, dim) -> (1, nh, agent_num, dk)
        mediator_tokens = rearrange_qkv(self.mediator_token)
        
        # mk attention
        # (1, nh, mediator_num, dk) x (bs, nh, dk, sl) -> (bs, nh, mediator_num, sl)
        mk_attn = self.soft_max(mediator_tokens @ k.transpose(-2, -1) * self.scale)
        mk_attn = self.attn_drop(mk_attn)
        # (bs, nh, mediator_num, sl) x (bs, nh, sl, dk) -> (bs, nh, mediator_num, dk)
        mk_v = mk_attn @ v  # Attn(Softmax(mediators @ k) * scale @ v)
        
        # qm attention
        # (bs, nh, sl, dk) x (bs, nh, dk, mediator_num) -> (bs, nh, sl, mediator_num)
        qm_attn = self.soft_max(q @ mediator_tokens.transpose(-2, -1) * self.scale)
        qm_attn = self.attn_drop(qm_attn)
        # (bs, nh, sl, mediator_num) x (bs, nh, mediator_num, dk) -> (bs, nh, sl, dk)
        x = qm_attn @ mk_v  # Attn(Softmax(q @ mediators) * scale @ mk_v)
        
        x = rearrange(x, 'b h l d -> b l (h d)')
        
        x = self.out_proj(x)
        x = self.proj_drop(x)
        
        return x
        
        
        