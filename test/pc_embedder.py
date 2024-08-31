'''
Author: wenjun-VCC
Date: 2024-08-31 17:48:01
LastEditors: wenjun-VCC
LastEditTime: 2024-08-31 17:52:24
Description: __discription:__
Email: wenjun.9707@gmail.com
Copyright (c): 2024 by wenjun-VCC, All Rights Reserved.
'''
import torch
import torch.nn as nn

import os
import sys
ROOT_PATH = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(ROOT_PATH)

from modules.pc_embedder import point_cloud_feature_extrator


if __name__ == '__main__':
    
    x = torch.randn(4, 128, 3).to('cuda')  # (bs, npoints, in_dims)
    net = point_cloud_feature_extrator(
        in_dim=3,
        out_dim=1024,
    ).to('cuda')
    out = net(x)
    print(out.shape)  # [bs, sl, dim]
    print('Point cloud feature extraction test passed!')