'''
Author: wenjun-VCC
Date: 2024-05-13 22:44:07
LastEditors: wenjun-VCC
LastEditTime: 2024-08-22 07:59:26
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

from modules.transformer import TransformerEncoder, TransformerDecoder


# test transformer encoder
def encoder_test():
    
    encoder = TransformerEncoder(
        d_model=128,
        depth=8,
        nheads=8,
        ac_func=nn.GELU,
    )
    src = torch.rand(4, 64, 128)  # [bs, sl, d_model]
    out, scores = encoder(src, return_scores=True)
    print(out.shape)  # [bs, ql, d_model]
    print(scores.shape)  # [bs, n_heads, sl, sl]


# test transformer decoder
def decoder_test():
    
    decoder = TransformerDecoder(
        d_model=128,
        depth=4,
        nheads=8,
        ac_func=nn.ReLU,
        is_cross_attn=True,
    )
    tgt = torch.rand(4, 64, 128)
    memory = torch.rand(4, 16, 128)
    out, cache = decoder(tgt, encoder_output=memory)
    print(out.shape)  # [bs, ql, d_model]


def autoregressive():
    
    # sos:0, eos:1, pad:10
    seq_01 = torch.tensor([0, 2, 3, 4, 2, 5, 3, 4, 5, 4, 1, 9, 9, 9, 9])
    mask_01 = torch.tensor([1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0]).bool()
    seq_02 = torch.tensor([0, 2, 3, 4, 2, 5, 3, 4, 1, 9, 9, 9, 9, 9, 9])
    mask_02 = torch.tensor([1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0]).bool()
    input = torch.stack([seq_01[:-1], seq_02[:-1]], dim=0).to('cuda')
    target = torch.stack([seq_01[1:], seq_02[1:]], dim=0).to('cuda')
    mask = torch.stack([mask_01[:-1], mask_02[:-1]], dim=0).to('cuda')
    
    class Model(nn.Module):
        
        def __init__(
            self,
        ):
            super(Model, self).__init__()
            self.embedding = nn.Embedding(10, 128, padding_idx=9)
            self.decoder = TransformerDecoder(
                d_model=128,
                depth=4,
                nheads=4,
                ac_func=nn.ReLU,
                is_cross_attn=False,
            )
            self.to_logits = nn.Linear(128, 10)
            
        def forward(
            self,
            input,
            tgt_mask,
        ):
            input = self.embedding(input)
            out, _ = self.decoder(input, tgt_mask=tgt_mask)
            logits = self.to_logits(out)
            
            return logits
    
    model = Model().to('cuda')
    
    for i in range(1000):  # epoch
        logits = model(input, mask)
        loss = nn.CrossEntropyLoss(ignore_index=9)(logits.view(-1, 10), target.view(-1))
        if i % 100 == 0:
            print(loss.item())
        loss.backward()
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
        optimizer.step()
        model.zero_grad()
    
    # torch.save(model.state_dict(), os.path.join(ROOT_PATH, 'model.pth'))
    
    def sample(
        model,
        input,
    ):
        model.eval()
        seq = []
        with torch.no_grad():
            for i in range(15):
                logits = model(input, tgt_mask=None)
                pred = torch.argmax(logits, dim=-1)
                seq.append(pred[:, -1].item())
                input = torch.cat([input, pred[:, -1].unsqueeze(1)], dim=1)

            print(seq)
    
    sample_input = torch.tensor([[0]]).to('cuda')
    sample(model, sample_input)


if __name__ == '__main__':
    
    encoder_test()
    decoder_test()
    
    # autoregressive()
    
    
    