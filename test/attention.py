'''
Author: wenjun-VCC
Date: 2024-07-31 00:07:00
LastEditors: wenjun-VCC
LastEditTime: 2024-09-05 00:17:35
Description: __discription:__
Email: wenjun.9707@gmail.com
Copyright (c): 2024 by wenjun-VCC, All Rights Reserved.
'''
import torch

import os
import sys
ROOT_PATH = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(ROOT_PATH)

from modules.attention import MHA, ConvAttention, AgentAttention, MediatorAttention


def test_conv1d_attn():
    
    x = torch.randn(4, 128, 16)
    net = ConvAttention(
        dim=128,
        ndim=1,
        nheads=8,
        dim_head=32,
    )
    out = net(x)
    print(out.shape)  # [bs, sl, dim]
    print('Conv1d attention test passed!')


def test_conv2d_attn():
    
    x = torch.randn(4, 128, 16, 16)
    net = ConvAttention(
        dim=128,
        ndim=2,
        nheads=8,
        dim_head=32,
    )
    out = net(x)
    print(out.shape)  # [bs, sl, dim]
    print('Conv2d attention test passed!')
    
    
def test_conv3d_attn():
    
    x = torch.randn(4, 128, 8, 8, 8)
    net = ConvAttention(
        dim=128,
        ndim=3,
        nheads=8,
        dim_head=32,
    )
    out = net(x)
    print(out.shape)  # [bs, sl, dim]
    print('Conv3d attention test passed!')
    

def test_attn():
    
    x = torch.randn(4, 41, 192)
    net = MHA(
        dim=192,
        nheads=8,
        dim_head=32,
    )
    out = net(x)
    print(out.shape)  # [bs, sl, dim]
    print('Linear attention test passed!')
    

def test_agent_attn():
    
    x = torch.randn(16, 41, 192)
    net = AgentAttention(
        dim=192,
        nheads=8,
        seq_len=41,
    )
    out = net(x)
    print(out.shape)  # [bs, sl, dim]
    print('Agent attention test passed!')
    

def test_mediator_attn():
    
    x = torch.randn(16, 41, 192)
    net = MediatorAttention(
        dim=192,
        nheads=8,
    )
    out = net(x)
    print(out.shape)  # [bs, sl, dim]
    print('Mediator attention test passed!')
    
    
if __name__ == '__main__':
        
    test_conv1d_attn()
    test_conv2d_attn()
    test_conv3d_attn()
    test_attn()
    test_agent_attn()
    test_mediator_attn()
    
    print('All attention tests passed!')