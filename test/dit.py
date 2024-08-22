'''
Author: wenjun-VCC
Date: 2024-06-14 02:26:47
LastEditors: wenjun-VCC
LastEditTime: 2024-06-14 04:05:00
Description: __discription:__
Email: wenjun.9707@gmail.com
Copyright (c) 2024 by wenjun/VCC, All Rights Reserved. 
'''
import torch

import os
import sys
ROOT_PATH = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(ROOT_PATH)

from modules.dit_1d import AdaLNDiTBlock, CroAttnDitBlock, InContextDiTBlock, DiT
from modules.dit_1d import DiT_1d_AdaLNDiTBlock, DiT_1d_CroAttnDitBlock, DiT_1d_InContextDiTBlock



def test_AdaLN():
    
    x = torch.randn(4, 512, 128)  # [bs, sl, dim]
    cond = torch.randn(4, 1, 128)  # [bs, 1, dim]

    model = AdaLNDiTBlock(
        dim=128,
        nheads=8,
    )
    
    out = model(x, cond)
    print(out.shape)  # [bs, sl, dim]
    print('AdaLN test passed!')
    
    
def test_CroAttn():
    
    x = torch.randn(4, 512, 128)  # [bs, sl, dim]
    cond = torch.randn(4, 1, 128)  # [bs, 1, dim]

    model = CroAttnDitBlock(
        dim=128,
        nheads=8,
    )
    
    out = model(x, cond)
    print(out.shape)  # [bs, sl, dim]
    print('CroAttn test passed!')
    

def test_InContext():
    
    x = torch.randn(4, 512, 128)  # [bs, sl, dim]
    cond = torch.randn(4, 1, 128)  # [bs, 1, dim]

    model = InContextDiTBlock(
        dim=128,
        nheads=8,
    )
    
    out = model(x, cond)
    print(out.shape)  # [bs, sl, dim]
    print('InContext test passed!')
    

def test_dit():
    
    x = torch.randn(4, 512, 128)  # [bs, sl, dim]
    t_embed = torch.randn(4, 1, 128)  # [bs, 1, dim]
    context = torch.randn(4, 1, 128)  # [bs, 1, dim]

    model = DiT(
        dim=128,
        nheads=8,
        learn_sigma=True,
        block=CroAttnDitBlock,
    )
    
    noise, sigma = model(x, t_embed, context)
    print(noise.shape)  # [bs, sl, dim]
    print(sigma.shape)  # [bs, sl, dim]
    print('DiT test passed!')


def test_DiT_1d():
    
    x = torch.randn(4, 512, 128)  # [bs, sl, dim]
    t_embed = torch.randn(4, 1, 128)  # [bs, 1, dim]
    context = torch.randn(4, 1, 128)  # [bs, 1, dim]

    model1 = DiT_1d_AdaLNDiTBlock(
        dim=128,
        learn_sigma=False,
    )
    noise = model1(x, t_embed, context)
    print(noise.shape)  # [bs, sl, dim]
    print('DiT_1d_AdaLNDiTBlock test passed!')
    
    model2 = DiT_1d_CroAttnDitBlock(
        dim=128,
        learn_sigma=True,
    )
    noise, sigma = model2(x, t_embed, context)
    print(noise.shape)  # [bs, sl, dim]
    print(sigma.shape)  # [bs, sl, dim]
    print('DiT_1d_CroAttnDitBlock test passed!')
    
    model3 = DiT_1d_InContextDiTBlock(
        dim=128,
        learn_sigma=True,
    )
    noise, sigma = model3(x, t_embed, context)
    print(noise.shape)  # [bs, sl, dim]
    print(sigma.shape)  # [bs, sl, dim]
    print('DiT_1d_InContextDiTBlock test passed!')



if __name__ == '__main__':
    
    test_DiT_1d()
    
    print('All tests passed!')