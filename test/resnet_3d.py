import torch
import torch.nn as nn

import os
import sys
ROOT_PATH = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(ROOT_PATH)

from modules.resnet_3d import resnet18_3d, resnet34_3d, resnet50_3d, resnet101_3d, resnet152_3d

# if project_out_dims is not specified,
# it will be set to dims[-1] by default
# 4 blocks in each resnet module
# 4 downsample layers in each resnet module

Resnet18 = resnet18_3d(
    in_dims=3,
    proj_out_dims=1024,
    ac_func=nn.ReLU,
    downsample_way='max_pool'  # 'max_pool' or 'conv' or 'avg_pool'
)

Resnet34 = resnet34_3d(
    in_dims=3,
    proj_out_dims=1024,
    ac_func=nn.ReLU,
    downsample_way='max_pool'
)

Resnet50 = resnet50_3d(
    in_dims=3,
    proj_out_dims=1024,
    ac_func=nn.ReLU,
    downsample_way='max_pool'
)

Resnet101 = resnet101_3d(
    in_dims=3,
    proj_out_dims=1024,
    ac_func=nn.ReLU,
    downsample_way='max_pool'
)

Resnet152 = resnet152_3d(
    in_dims=3,
    proj_out_dims=1024,
    ac_func=nn.ReLU,
    downsample_way='max_pool'
)


if __name__ == '__main__':
    
    x = torch.randn(1, 3, 128, 128, 128)  # [bs, channels, H, W, D]
    module = Resnet18
    out = module(x)
    print(out.shape)  # [bs, proj_out_dims, H//4, W//4, D//4]
    # [1, 1024, 8, 8, 8]