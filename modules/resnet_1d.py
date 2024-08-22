'''
Author: wenjun-VCC
Date: 2024-05-13 22:39:33
LastEditors: wenjun-VCC
LastEditTime: 2024-07-31 15:06:49
Description: __discription:__
Email: wenjun.9707@gmail.com
Copyright (c) 2024 by wenjun/VCC, All Rights Reserved. 
'''
import torch.nn as nn
from torchtyping import TensorType
from typing import Optional, List
from beartype import beartype




"""
    ResNet1d Framework

    This framework implements a one-dimensional version of the Residual Network (ResNet),
    typically used for processing one-dimensional sequential data. ResNet1d effectively
    captures temporal dependencies within sequences, making it applicable to a variety
    of one-dimensional signal processing tasks including but not limited to audio processing,
    time series analysis, and biomedical signal processing.

    Key features include:
    - Utilizing residual connections to alleviate the vanishing gradient problem in deep network training,
      enabling the extension of network depth.
    - Suitable for any form of one-dimensional input data, such as audio waveforms,
      financial market price series, or healthcare monitoring data.

    Input:
    - features: The input feature sequence, expected to be in the shape [batch_size, channels, sequence_length], where
    - batch_size is the size of the batch,
    - channels is the number of input channels,
    - sequence_length is the length of the sequence.

    Output:
    - output: The output after processing by ResNet1d, retaining the shape [batch_size, channels, sequence_length].
      The number of output channels depends on the network design and can be used for subsequent classification,
      regression, or other tasks.

    Usage Example:
    ```python
    model = ResNet1d(input_channels=10, num_blocks=[3, 4, 6], num_classes=5)
    input = torch.randn(32, 10, 256)  # 32 samples, each with 10 channels, sequence length 256
    output = model(input)
"""

class BasicBlock1d(nn.Module):
    
    def __init__(
        self,
        in_dims: int,
        out_dims: int,
        kernel_size: int=7,
        ac_func=nn.ReLU,
        norm=nn.GroupNorm,
        groups: int=16,
    ) -> None:
        super(BasicBlock1d, self).__init__()

        assert kernel_size % 2 != 0, 'The kernel_size should be odd.'
        self.padding = kernel_size//2
        
        self.out_dims = in_dims if out_dims is None else out_dims
        
        self.ln = False
        if norm == nn.LayerNorm:
            self.ln = True
        
        if norm == nn.GroupNorm:
            self.norm = norm(groups, self.out_dims)
        else:
            self.norm = norm(self.out_dims)
        
        self.basic_conv = nn.Conv1d(in_dims, self.out_dims, kernel_size=kernel_size, padding=self.padding)
        self.ac_func = ac_func()
        self.out_fc = nn.Conv1d(self.out_dims, self.out_dims, kernel_size=1)
    
    
    @beartype
    def forward(
        self,
        x: TensorType['bs', 'dims', 'seq_len', float],
        mask: Optional[TensorType['bs', '1', 'seq_len', bool]]=None,
    ):
            
        x = self.basic_conv(x)
        
        if mask is not None:
            x = x.masked_fill(~mask, 0.)
        if self.ln:
            x = self.norm(x.permute(0, 2, 1)).permute(0, 2, 1)
        else:
            x = self.norm(x)
        x = self.ac_func(x)
        x = self.out_fc(x)
        
        if mask is not None:
            x = x.masked_fill(~mask, 0.)
        
        return x



class Block1d(nn.Module):
    
    def __init__(
        self,
        dims: int,
        out_dims: int=None,
        kernel_size: int=3,
        norm=nn.GroupNorm,
        groups: int=16,
    ) -> None:
        super(Block1d, self).__init__()
        
        assert kernel_size % 2 != 0, 'The kernel_size should be odd.'
        self.padding = 0 if kernel_size==1 else kernel_size // 2
        
        self.dims = dims
        self.out_dims = dims if out_dims is None else out_dims
        
        self.ln = False
        if norm == nn.LayerNorm:
            self.ln = True
        if norm == nn.GroupNorm:
            self.norm = norm(groups, self.out_dims)
        else:
            self.norm = norm(self.out_dims)
        
        self.conv = nn.Conv1d(self.dims, self.out_dims, kernel_size=kernel_size, padding=self.padding)
       
    
    @beartype
    def forward(
        self,
        x: TensorType['bs', 'dims', 'seq_len', float],
        mask: Optional[TensorType['bs', '1', 'seq_len', bool]]=None,
    ):
        
        x = self.conv(x)
        if self.ln:
            x = self.norm(x.permute(0, 2, 1)).permute(0, 2, 1)
        else:
            x = self.norm(x)
        if mask is not None:
            x = x.masked_fill(~mask, 0.)
        
        return x



class ShallowResnetBlock1d(nn.Module):
    
    def __init__(
        self,
        dims: int,
        out_dims: int=None,
        ac_func=nn.ReLU,
        norm=nn.GroupNorm,
        groups: int=16,
    ) -> None:
        super(ShallowResnetBlock1d, self).__init__()
        
        self.dims = dims
        self.out_dims = dims if out_dims is None else out_dims
        
        self.short_cut = nn.Identity() if out_dims is None else nn.Conv1d(dims, self.out_dims, kernel_size=1, bias=False)
        
        self.block1 = Block1d(
            dims=dims,
            out_dims=dims,
            kernel_size=3,
            norm=norm,
            groups=groups,
        )
        
        self.block2 = Block1d(
            dims=dims,
            out_dims=self.out_dims,
            kernel_size=3,
            norm=norm,
            groups=groups,
        )
        
        self.ac_func = ac_func()
        
    
    @beartype
    def forward(
        self,
        x: TensorType['bs', 'dims', 'seq_len', float],
        mask: Optional[TensorType['bs', '1', 'seq_len', bool]]=None,
    ):
            
        residual = self.short_cut(x)
        
        if mask is not None:
            residual = residual.masked_fill(~mask, 0.)
            
        x = self.block1(x, mask)
        x = self.ac_func(x)
        x = self.block2(x, mask)
        
        out = self.ac_func(residual + x)
        
        return out
        


class DeepResnetBlock1d(nn.Module):
    
    def __init__(
        self,
        dims: int,
        out_dims: int=None,
        ac_func=nn.ReLU,
        norm=nn.GroupNorm,
        groups: int=16,
    ) -> None:
        super(DeepResnetBlock1d, self).__init__()
        
        self.dims = dims
        self.out_dims = dims if out_dims is None else out_dims
        
        self.short_cut = nn.Identity() if out_dims is None else nn.Conv1d(dims, self.out_dims, kernel_size=1, bias=False)
        
        self.block1 = Block1d(
            dims=dims,
            out_dims=dims,
            kernel_size=3,
            norm=norm,
            groups=groups,
        )
        self.block2 = Block1d(
            dims=dims,
            out_dims=dims,
            kernel_size=3,
            norm=norm,
            groups=groups,
        )
        self.block3 = Block1d(
            dims=dims,
            out_dims=self.out_dims,
            kernel_size=3,
            norm=norm,
            groups=groups,
        )
        
        self.ac_func = ac_func()
        
    
    @beartype
    def forward(
        self,
        x: TensorType['bs', 'dims', 'seq_len', float],
        mask: Optional[TensorType['bs', '1', 'seq_len', bool]]=None,
    ):
            
        residual = self.short_cut(x)
        
        if mask is not None:
            residual = residual.masked_fill(~mask, 0.)
            
        x = self.block1(x, mask)
        x = self.ac_func(x)
        x = self.block2(x, mask)
        x = self.ac_func(x)
        x = self.block3(x, mask)
        
        out = self.ac_func(residual + x)
        
        return out
        
    

class ShallowResnet1d(nn.Module):
    
    def __init__(
        self,
        in_dims: int,
        basic_out_dims: int,
        proj_out_dims: int=None,
        blocks: List[int]=[2, 2, 2, 2],
        dims: List[int]=[128, 192, 256, 384],
        ac_func=nn.ReLU,
        norm=nn.GroupNorm,
        groups: int=16,
    ) -> None:
        super(ShallowResnet1d, self).__init__()
        
        self.blocks = blocks
        self.dims = dims
        
        self.basic_conv = BasicBlock1d(
            in_dims,
            basic_out_dims,
            ac_func=ac_func,
            norm=norm,
            groups=groups,
        )
        self.proj_out_conv = nn.Identity() if proj_out_dims is None else nn.Conv1d(dims[-1], proj_out_dims, kernel_size=1, bias=False)
        
        self.resnet_module_list = nn.ModuleList([])
            
        curr_dim = basic_out_dims
        for idx, _ in enumerate(blocks):
            for i in range(_):
                self.resnet_module_list.append(
                    ShallowResnetBlock1d(
                    curr_dim,
                    dims[idx],
                    ac_func=ac_func,
                    norm=norm,
                    groups=groups,
                    )
                )
                curr_dim = dims[idx]
    
    
    @beartype       
    def forward(
        self,
        x: TensorType['bs', 'dims', 'seq_len', float],
        mask: Optional[TensorType['bs', '1', 'seq_len', bool]]=None,
    ):
        
        if mask is not None:
            x = x.masked_fill(~mask, 0.)
        x = self.basic_conv(x, mask)
        
        for module in self.resnet_module_list:
            
            if mask is not None:
                x = x.masked_fill(~mask, 0.)
            x = module(x, mask)
        
        x = self.proj_out_conv(x)
        
        return x
        
        
        
class DeepResnet1d(nn.Module):
    
    def __init__(
        self,
        in_dims: int,
        basic_out_dims: int,
        proj_out_dims: int=None,
        blocks: List[int]=[3, 4, 6, 3],
        dims: List[int]=[128, 256, 384, 512],
        ac_func=nn.ReLU,
        norm=nn.GroupNorm,
        groups: int=16,
    ) -> None:
        super().__init__()
        
        self.blocks = blocks
        self.dims = dims
        
        self.basic_conv = BasicBlock1d(
            in_dims,
            basic_out_dims,
            ac_func=ac_func,
            norm=norm,
            groups=groups,
        )
        self.proj_out_conv = nn.Identity() if proj_out_dims is None else nn.Conv1d(dims[-1], proj_out_dims, kernel_size=1, bias=False)
        
        self.resnet_module_list = nn.ModuleList([])
        curr_dim = basic_out_dims
        for idx, _ in enumerate(blocks):
            for i in range(_):
                self.resnet_module_list.append(
                    DeepResnetBlock1d(
                    curr_dim,
                    dims[idx],
                    ac_func=ac_func,
                    norm=norm,
                    groups=groups,
                    )
                )
                curr_dim = dims[idx]
            
    
    @beartype
    def forward(
        self,
        x: TensorType['bs', 'dims', 'seq_len', float],
        mask: Optional[TensorType['bs', '1', 'seq_len', bool]]=None,
    ):
        
        if mask is not None:
            x = x.masked_fill(~mask, 0.)
        x = self.basic_conv(x, mask)
        
        for module in self.resnet_module_list:
            
            if mask is not None:
                x = x.masked_fill(~mask, 0.)
            x = module(x, mask)
            
        x = self.proj_out_conv(x)
        
        return x
        



# for nn.Conv1d:
# considering the information between the sequence (local word)
# usually don't change the sequence length


# for resnet_1d model:
# using to sequence processing
# input : [bs, dims, seq_len]
# output: [bs, proj_out_dims, seq_len]

def resnet18_1d(
    in_dims: int,
    proj_out_dims: int=1000,
    basic_block_dims: int=128,
    dims: List[int]=[128, 256, 384, 512],
    ac_func=nn.ReLU,
    norm=nn.GroupNorm,
    groups: int=16,
):
    
    module = ShallowResnet1d(
        in_dims=in_dims,
        basic_out_dims=basic_block_dims,
        proj_out_dims=proj_out_dims,
        blocks=[2, 2, 2, 2],
        dims=dims,
        ac_func=ac_func,
        norm=norm,
        groups=groups,
    )
    
    return module


def resnet34_1d(
    in_dims: int,
    proj_out_dims: int=1000,
    basic_block_dims: int=128,
    dims: List[int]=[128, 256, 384, 512],
    ac_func=nn.ReLU,
    norm=nn.GroupNorm,
    groups: int=16,
):
    
    module = ShallowResnet1d(
        in_dims=in_dims,
        basic_out_dims=basic_block_dims,
        proj_out_dims=proj_out_dims,
        blocks=[3, 4, 6, 3],
        dims=dims,
        ac_func=ac_func,
        norm=norm,
        groups=groups,
    )
    
    return module


def resnet50_1d(
    in_dims: int,
    proj_out_dims: int=1000,
    basic_block_dims: int=128,
    dims: List[int]=[128, 256, 384, 512],
    ac_func=nn.ReLU,
    norm=nn.GroupNorm,
    groups: int=16,
):
    
    module = DeepResnet1d(
        in_dims=in_dims,
        basic_out_dims=basic_block_dims,
        proj_out_dims=proj_out_dims,
        blocks=[3, 4, 6, 3],
        dims=dims,
        ac_func=ac_func,
        norm=norm,
        groups=groups,
    )
    
    return module


def resnet101_1d(
    in_dims: int,
    proj_out_dims: int=1000,
    basic_block_dims: int=128,
    dims: List[int]=[128, 256, 384, 512],
    ac_func=nn.ReLU,
    norm=nn.GroupNorm,
    groups: int=32,
):
    
    module = DeepResnet1d(
        in_dims=in_dims,
        basic_out_dims=basic_block_dims,
        proj_out_dims=proj_out_dims,
        blocks=[3, 4, 23, 3],
        dims=dims,
        ac_func=ac_func,
        norm=norm,
        groups=groups,
    )
    
    return module


def resnet152_1d(
    in_dims: int,
    proj_out_dims: int=1000,
    basic_block_dims: int=128,
    dims: List[int]=[128, 256, 384, 512],
    ac_func=nn.ReLU,
    norm=nn.GroupNorm,
    groups: int=32,
):
    
    module = DeepResnet1d(
        in_dims=in_dims,
        basic_out_dims=basic_block_dims,
        proj_out_dims=proj_out_dims,
        blocks=[3, 8, 36, 3],
        dims=dims,
        ac_func=ac_func,
        norm=norm,
        groups=groups,
    )
    
    return module

