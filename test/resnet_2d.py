'''
Author: wenjun-VCC
Date: 2024-05-13 22:43:52
LastEditors: wenjun-VCC
LastEditTime: 2024-08-15 20:32:54
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
from modules.resnet_2d import resnet18_2d, resnet34_2d, resnet50_2d, resnet101_2d, resnet152_2d
# for custom module
from modules.resnet_2d import ShallowResnet2d, DeepResnet2d


# if project_out_dims is not specified,
# it will be set to dims[-1] by default
# 4 blocks in each resnet module
# 4 downsample layers in each resnet module

Resnet18 = resnet18_2d(
    in_dims=3,
    proj_out_dims=1024,
    ac_func=nn.ReLU,
    downsample_way='max_pool'  # 'max_pool' or 'conv' or 'avg_pool'
)

Resnet34 = resnet34_2d(
    in_dims=3,
    proj_out_dims=1024,
    ac_func=nn.ReLU,
    downsample_way='max_pool'
)

Resnet50 = resnet50_2d(
    in_dims=3,
    proj_out_dims=1024,
    ac_func=nn.ReLU,
    downsample_way='max_pool'
)

# Resnet101 = resnet101_2d(
#     in_dims=3,
#     proj_out_dims=1024,
#     ac_func=nn.ReLU,
#     downsample_way='max_pool'
# )

# Resnet152 = resnet152_2d(
#     in_dims=3,
#     proj_out_dims=1024,
#     ac_func=nn.ReLU,
#     downsample_way='max_pool'
# )


# custom module
# ShallowResnet2d 2 blocks in each layer
# DeepResnet2d 3 blocks in each layer
ResnetCustom = ShallowResnet2d(
    in_dims=3,
    basic_out_dims=16,
    proj_out_dims=512,
    blocks=[2, 2, 2, 2, 2],  # 5 layers in total, dowmsample in each layer
    dims=[32, 64, 128, 256, 384],
    ac_func=nn.ReLU,
    downsample_way='conv',
)


if __name__ == '__main__':
    
    # default
    x = torch.randn(1, 3, 128, 128)  # [bs, channels, H, W]
    module = Resnet18
    out = module(x)
    print(out.shape)  # [bs, proj_out_dims, H//16, W//16]
    # [1, 1024, 16, 16]
    
    # custom
    x = torch.randn(1, 3, 128, 128)  # [bs, channels, H, W]
    module = ResnetCustom
    out = module(x)
    print(out.shape)  # [bs, proj_out_dims, H//32, W//32]
    # [1, 512, 8, 8]