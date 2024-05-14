'''
Author: wenjun-VCC
Date: 2024-05-13 22:43:44
LastEditors: wenjun-VCC
LastEditTime: 2024-05-14 09:56:48
FilePath: resnet_1d.py
Description: __discription:__
Email: wenjun.9707@gmail.com
Copyright (c) 2024 by wenjun/VCC, All Rights Reserved. 
'''
import torch
import torch.nn as nn

import os
import sys
ROOT_PATH = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(ROOT_PATH)

from modules.resnet_1d import resnet18_1d, resnet34_1d, resnet50_1d, resnet101_1d, resnet152_1d


# if project_out_dims is not specified,
# it will be set to dims[-1] by default
# using GroupNorm instead of BatchNorm

Resnet18 = resnet18_1d(
    in_dims=128,
    proj_out_dims=1024,
    ac_func=nn.ReLU
)

Resnet34 = resnet34_1d(
    in_dims=128,
    proj_out_dims=1024,
)

Resnet50 = resnet50_1d(
    in_dims=128,
    proj_out_dims=1024,
)

Resnet101 = resnet101_1d(
    in_dims=128,
    proj_out_dims=1024,
)

Resnet152 = resnet152_1d(
    in_dims=128,
    proj_out_dims=1024,
)

if __name__ == '__main__':
    
    x = torch.randn(1, 128, 100)  # [bs, in_dims, seq_len]
    module = Resnet18
    out = module(x)
    print(out.shape)  # [bs, proj_out_dims, seq_len]
    # [1, 1024, 100]
    