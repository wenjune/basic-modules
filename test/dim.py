import torch

import os
import sys
ROOT_PATH = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(ROOT_PATH)

from modules.dim_1d import AdaLNDiMBlock, DiM
from modules.dim_1d import DiM_1d_AdaLNDiMBlock



def test_AdaLN():
    
    x = torch.randn(4, 512, 512).to('cuda')  # [bs, sl, dim]
    cond = torch.randn(4, 1, 512).to('cuda')  # [bs, 1, dim]

    model = AdaLNDiMBlock(
        dim=512,
    ).to('cuda')
    
    out = model(x, cond)
    print(out.shape)  # [bs, sl, dim]
    print('AdaLN test passed!')
    


def test_DiM_1d():
    
    x = torch.randn(4, 512, 1024).to('cuda')  # [bs, sl, dim]
    t_embed = torch.randn(4, 1, 1024).to('cuda')  # [bs, 1, dim]
    context = torch.randn(4, 1, 1024).to('cuda')  # [bs, 1, dim]

    model1 = DiM_1d_AdaLNDiMBlock(
        dim=1024,
        learn_sigma=False,
    ).to('cuda')
    noise = model1(x, t_embed, context)
    print(noise.shape)  # [bs, sl, dim]
    print('DiT_1d_AdaLNDiTBlock test passed!')
    



if __name__ == '__main__':
    
    # test_AdaLN()
    test_DiM_1d()
    
    print('All tests passed!')