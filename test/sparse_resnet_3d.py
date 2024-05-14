'''
Author: wenjun-VCC
Date: 2024-05-14 19:26:33
LastEditors: wenjun-VCC
LastEditTime: 2024-05-14 20:43:36
FilePath: sparse_resnet_3d.py
Description: __discription:__
Email: wenjun.9707@gmail.com
Copyright (c) 2024 by wenjun/VCC, All Rights Reserved. 
'''
import torch
import torch.nn as nn
from spconv.pytorch import SparseConvTensor

import os
import sys
ROOT_PATH = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(ROOT_PATH)

from modules.sparse_resnet_3d import spresnet18_3d, spresnet34_3d, spresnet50_3d, spresnet101_3d, spresnet152_3d


if __name__ == '__main__':
    
    # init a sparse tensor
    features = torch.randn(6, 3).to('cuda')
    indices = torch.tensor(
        [[0, 14, 64, 32],
        [0, 32, 11, 23],
        [0, 43, 21, 32],
        [0, 47, 45, 23],
        [0, 26, 15, 19],
        [0, 7, 29, 53]]
    ).to(torch.int32).to('cuda')
    sptensor = SparseConvTensor(
        features=features,
        indices=indices,
        spatial_shape=(64, 64, 64),
        batch_size=1,
    )
    
    # init module
    model = spresnet50_3d(3, 1000, downsample_way='conv').to('cuda')
    out = model(sptensor)
    
    print(out.dense().shape)
    # torch.Size([1, 1000])
    # spatial_shape: (4, 4, 4)