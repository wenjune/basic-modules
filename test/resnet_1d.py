'''
Author: wenjun-VCC
Date: 2024-05-13 22:43:44
LastEditors: wenjun-VCC
LastEditTime: 2024-05-15 21:46:18
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

# for default module
from modules.resnet_1d import resnet18_1d, resnet34_1d, resnet50_1d, resnet101_1d, resnet152_1d
# for custom module
from modules.resnet_1d import ShallowResnet1d, DeepResnet1d

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


# custom module
# ShallowResnet1d 2 blocks in each layer
# DeepResnet1d 3 blocks in each layer
ResnetCustom = ShallowResnet1d(
    in_dims=128,
    basic_out_dims=16,
    proj_out_dims=512,
    blocks=[2, 2, 2, 2, 2],  # 5 layers in total
    dims=[32, 64, 128, 256, 384],
    ac_func=nn.ReLU,
    norm=nn.GroupNorm,
)


if __name__ == '__main__':
    
    # default
    x = torch.randn(1, 128, 100)  # [bs, in_dims, seq_len]
    module = Resnet18
    out = module(x)
    print(out.shape)  # [bs, proj_out_dims, seq_len]
    # [1, 1024, 100]
    
    # custom
    x = torch.randn(1, 128, 100)  # [bs, in_dims, seq_len]
    module = ResnetCustom
    out = module(x)
    print(out.shape)  # [bs, proj_out_dims, seq_len]
    # [1, 512, 100]
    