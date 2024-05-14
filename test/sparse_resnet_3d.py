import torch
import torch.nn as nn
from spconv.pytorch import SparseConvTensor

import os
import sys
ROOT_PATH = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(ROOT_PATH)

from modules.sparse_resnet_3d import spresnet18_3d, spresnet34_3d, spresnet50_3d, spresnet101_3d, spresnet152_3d


if __name__ == '__main__':
    
    features = torch.randn(6, 3).to('cuda')

    indices = torch.tensor(
        [[0, 14, 64, 32],
        [0, 32, 11, 23],
        [0, 43, 21, 32],
        [0, 47, 45, 23],
        [0, 26, 15, 19],
        [0, 7, 29, 53]]
    ).to(torch.int32).to('cuda')

    t = SparseConvTensor(
        features=features,
        indices=indices,
        spatial_shape=(64, 64, 64),
        batch_size=1,
    )

    model = spresnet50_3d(3, 1000, downsample_way='conv').to('cuda')
    out = model(t)
    
    print(out.dense().shape)  # [6, 1000]