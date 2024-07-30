import torch
import torch.nn as nn

import os
import sys
ROOT_PATH = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(ROOT_PATH)

from modules.attention import LinearAttention, ConvAttention


def test_conv1d_atten():
    
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


def test_conv2d_atten():
    
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
    
    
def test_conv3d_atten():
    
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
    

def test_linear_atten():
    
    x = torch.randn(4, 41, 192)
    net = LinearAttention(
        dim=192,
        nheads=8,
        dim_head=32,
    )
    out = net(x)
    print(out.shape)  # [bs, sl, dim]
    print('Linear attention test passed!')
    
    
if __name__ == '__main__':
        
    test_conv1d_atten()
    test_conv2d_atten()
    test_conv3d_atten()
    test_linear_atten()
    
    print('All attention tests passed!')