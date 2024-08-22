'''
Author: wenjun-VCC
Date: 2024-05-14 18:41:59
LastEditors: wenjun-VCC
LastEditTime: 2024-05-14 19:28:02
Description: Just can run on Linux Platform and on GPU device.
Email: wenjun.9707@gmail.com
Copyright (c) 2024 by wenjun/VCC, All Rights Reserved. 
'''
import torch
import torch.nn as nn
from typing import Optional, List
import spconv.pytorch as spconv
from spconv.pytorch import SparseConvTensor




class SparseBasicBlock3d(nn.Module):
    
    def __init__(
        self,
        in_dims: int,
        out_dims: int,
        kernel_size: int=7,
        ac_func=nn.ReLU
    ) -> None:
        super(SparseBasicBlock3d, self).__init__()

        assert kernel_size % 2 != 0, 'The kernel_size should be odd.'
        self.conv = spconv.SparseSequential(
            spconv.SubMConv3d(in_channels=in_dims, out_channels=out_dims, kernel_size=kernel_size, indice_key='basic'),
            nn.BatchNorm1d(out_dims),
            ac_func(),
        )
        
        self.out_conv = spconv.SubMConv3d(in_channels=out_dims, out_channels=out_dims, kernel_size=1, indice_key='basic')
    
    
    def forward(
        self,
        x: SparseConvTensor,
    ):
            
        x = self.conv(x)
        x = self.out_conv(x)
        
        return x



class SparseBlock3d(nn.Module):
    
    def __init__(
        self,
        dims: int,
        out_dims: int=None,
        key='sparseblock3d',
        ac_func=nn.ReLU,
    ):
        super(SparseBlock3d, self).__init__()
        
        self.out_dims = dims if out_dims is None else out_dims
        
        self.block = spconv.SparseSequential(
            spconv.SparseConv3d(dims, dims, 3, 1, padding=1, indice_key=f'{key}sc01'),
            nn.BatchNorm1d(dims),
            ac_func(),
            spconv.SubMConv3d(dims, self.out_dims, 1, indice_key=f'{key}smc02'),
            nn.BatchNorm1d(self.out_dims),
            ac_func(),
            spconv.SparseInverseConv3d(self.out_dims, self.out_dims, 3, indice_key=f'{key}sc01'),
            nn.BatchNorm1d(self.out_dims),
            ac_func(),
        )
        
    
    def forward(
        self,
        x: SparseConvTensor,
    ):
        
        x = self.block(x)
        
        return x
        


class ShallowSparseResnetBlock3d(nn.Module):
    
    def __init__(
        self,
        dims :int,
        out_dims: int=None,
        key: str='shallow_sparse_resnet_block3d',
        downsample: bool=False,
        ac_func=nn.ReLU,
        downsample_way: str='maxpool',
    ):
        super(ShallowSparseResnetBlock3d, self).__init__()
        
        assert downsample_way in ['maxpool', 'conv'], 'The downsample_way should be in [maxpool, conv].'
        
        self.dims = dims
        self.out_dims = dims if out_dims is None else out_dims
        
        if downsample:
            self.downsample = spconv.SparseMaxPool3d(2, 2)
            if downsample_way == 'conv':
                self.downsample = spconv.SparseConv3d(dims, dims, kernel_size=2, stride=2, indice_key=f'{key}sc00')
        else:
            self.downsample = spconv.Identity()
        
        self.dimchange_layer = spconv.SubMConv3d(in_channels=dims, out_channels=self.out_dims, kernel_size=1, indice_key=f'{key}smc01')
        self.downsample_layer = spconv.SparseMaxPool3d(2, 2) if downsample else spconv.Identity()
        
        self.short_cut = spconv.SparseSequential(
            self.downsample_layer,
            self.dimchange_layer,
        )
        
        self.block1 = SparseBlock3d(dims=dims, out_dims=self.out_dims, key=f'{key}block1', ac_func=ac_func)
        
        self.block2 = SparseBlock3d(dims=self.out_dims, key=f'{key}block2', ac_func=ac_func)
        
        self.ac_func = spconv.SparseSequential(
            ac_func()
        )
    
    
    def forward(
        self,
        x: SparseConvTensor,
    ) -> SparseConvTensor:
        
        residual = self.short_cut(x)
        out = self.block1(x)
        out = self.downsample(out)
        out = self.block2(out)
        out = self.ac_func(out+residual)
        
        return out



class DeepSparseResnetBlock3d(nn.Module):
    
    def __init__(
        self,
        dims :int,
        out_dims: int=None,
        key: str='deep_sparse_resnet_block3d',
        downsample: bool=False,
        ac_func=nn.ReLU,
        downsample_way: str='maxpool',
    ) -> None:
        super(DeepSparseResnetBlock3d, self).__init__()
        
        assert downsample_way in ['maxpool', 'conv'], 'The downsample_way should be in [maxpool, conv].'
        
        self.dims = dims
        self.out_dims = dims if out_dims is None else out_dims
        
        if downsample:
            self.downsample = spconv.SparseMaxPool3d(2, 2)
            if downsample_way == 'conv':
                self.downsample = spconv.SparseConv3d(dims, dims, kernel_size=2, stride=2, indice_key=f'{key}sc00')
        else:
            self.downsample = spconv.Identity()
        
        self.dimchange_layer = spconv.SubMConv3d(in_channels=dims, out_channels=self.out_dims, kernel_size=1, indice_key=f'{key}smc01')
        self.downsample_layer = spconv.SparseMaxPool3d(2, 2) if downsample else spconv.Identity()
        
        self.short_cut = spconv.SparseSequential(
            self.downsample_layer,
            self.dimchange_layer,
        )
        
        self.block1 = SparseBlock3d(dims=dims, out_dims=self.out_dims, key=f'{key}block1', ac_func=ac_func)
        self.block2 = SparseBlock3d(dims=self.out_dims, key=f'{key}block2', ac_func=ac_func)
        self.block3 = SparseBlock3d(dims=self.out_dims, key=f'{key}block3', ac_func=ac_func)
        
        self.ac_func = spconv.SparseSequential(
            ac_func()
        )
        
    
    def forward(
        self,
        x: SparseConvTensor,
    ) -> SparseConvTensor:
        
        residual = self.short_cut(x)
        out = self.block1(x)
        out = self.block2(out)
        out = self.downsample(out)
        out = self.block3(out)
        out = self.ac_func(out+residual)
        
        return out



class ShallowSparseResnet3d(nn.Module):
    
    def __init__(
        self,
        in_dims: int,
        basic_out_dims: int,
        proj_out_dims: int=None,
        blocks: List[int]=[2, 2, 2, 2],
        dims: List[int]=[128, 256, 384, 512],
        ac_func=nn.ReLU,
        downsample_way: str='maxpool',
    ) -> None:
        super(ShallowSparseResnet3d, self).__init__()
        
        self.blocks = blocks
        self.dims = dims
        
        self.basic_conv = SparseBasicBlock3d(in_dims, basic_out_dims, ac_func=ac_func)
        self.proj_out_conv = nn.Identity() if proj_out_dims is None else spconv.SubMConv3d(dims[-1], proj_out_dims, kernel_size=1, indice_key='out')
        
        self.resnet_module_list = nn.ModuleList([])
        curr_dim = basic_out_dims
        for idx, _ in enumerate(blocks):
            downsample = False
            for i in range(_):
                if i == _-1:
                    downsample = True
                self.resnet_module_list.append(ShallowSparseResnetBlock3d(
                    curr_dim, dims[idx],
                    key=f'shallow_sparse_resnet_block3d_layer{idx}_block{i}',
                    ac_func=ac_func,
                    downsample=downsample,
                    downsample_way=downsample_way,
                ))
                curr_dim = dims[idx]
    
      
    def forward(
        self,
        x: spconv.SparseConvTensor,
    ):
        
        x = self.basic_conv(x)
        
        for module in self.resnet_module_list:
            x = module(x)
        
        x = self.proj_out_conv(x)
        
        return x
        
        
        
class DeepSparseResnet3d(nn.Module):
    
    def __init__(
        self,
        in_dims: int,
        basic_out_dims: int,
        proj_out_dims: int=None,
        blocks: List[int]=[3, 4, 6, 3],
        dims: List[int]=[128, 256, 384, 512],
        ac_func=nn.ReLU,
        downsample_way: str='maxpool',
    ) -> None:
        super().__init__()
        
        self.blocks = blocks
        self.dims = dims
        
        self.basic_conv = SparseBasicBlock3d(in_dims, basic_out_dims, ac_func=ac_func)
        self.proj_out_conv = nn.Identity() if proj_out_dims is None else spconv.SubMConv3d(dims[-1], proj_out_dims, kernel_size=1, indice_key='out')
        
        self.resnet_module_list = nn.ModuleList([])
        curr_dim = basic_out_dims
        for idx, _ in enumerate(blocks):
            downsample = False
            for i in range(_):
                if i == _-1:
                    downsample = True
                self.resnet_module_list.append(DeepSparseResnetBlock3d(
                    curr_dim, dims[idx],
                    key=f'deep_sparse_resnet_block3d_layer{idx}_block{i}',
                    downsample=downsample,
                    ac_func=ac_func,
                    downsample_way=downsample_way,
                ))
                curr_dim = dims[idx]
            
    
    def forward(
        self,
        x: spconv.SparseConvTensor,
    ):
        
        x = self.basic_conv(x)
        
        for module in self.resnet_module_list:
            x = module(x)
            
        x = self.proj_out_conv(x)
        
        return x
        



def spresnet18_3d(
    in_dims: int,
    proj_out_dims: int=1000,
    basic_block_dims: int=32,
    dims: List[int]=[64, 128, 156, 384],
    ac_func=nn.ReLU,
    downsample_way: str='maxpool',
):
    
    module = ShallowSparseResnet3d(
        in_dims=in_dims,
        basic_out_dims=basic_block_dims,
        proj_out_dims=proj_out_dims,
        blocks=[2, 2, 2, 2],
        dims=dims,
        ac_func=ac_func,
        downsample_way=downsample_way,
    )
    
    return module


def spresnet34_3d(
    in_dims: int,
    proj_out_dims: int=1000,
    basic_block_dims: int=32,
    dims: List[int]=[64, 128, 256, 384],
    ac_func=nn.ReLU,
    downsample_way: str='maxpool',
):
    
    module = ShallowSparseResnet3d(
        in_dims=in_dims,
        basic_out_dims=basic_block_dims,
        proj_out_dims=proj_out_dims,
        blocks=[3, 4, 6, 3],
        dims=dims,
        ac_func=ac_func,
        downsample_way=downsample_way,
    )
    
    return module


def spresnet50_3d(
    in_dims: int,
    proj_out_dims: int=1000,
    basic_block_dims: int=32,
    dims: List[int]=[64, 128, 256, 384],
    ac_func=nn.ReLU,
    downsample_way: str='maxpool',
):
    
    module = DeepSparseResnet3d(
        in_dims=in_dims,
        basic_out_dims=basic_block_dims,
        proj_out_dims=proj_out_dims,
        blocks=[3, 4, 6, 3],
        dims=dims,
        ac_func=ac_func,
        downsample_way=downsample_way,
    )
    
    return module


def spresnet101_3d(
    in_dims: int,
    proj_out_dims: int=1000,
    basic_block_dims: int=32,
    dims: List[int]=[64, 128, 256, 384],
    ac_func=nn.ReLU,
    downsample_way: str='maxpool',
):
    
    module = DeepSparseResnet3d(
        in_dims=in_dims,
        basic_out_dims=basic_block_dims,
        proj_out_dims=proj_out_dims,
        blocks=[3, 4, 23, 3],
        dims=dims,
        ac_func=ac_func,
        downsample_way=downsample_way,
    )
    
    return module


def spresnet152_3d(
    in_dims: int,
    proj_out_dims: int=1000,
    basic_block_dims: int=32,
    dims: List[int]=[64, 128, 256, 384],
    ac_func=nn.ReLU,
    downsample_way: str='maxpool',
):
    
    module = DeepSparseResnet3d(
        in_dims=in_dims,
        basic_out_dims=basic_block_dims,
        proj_out_dims=proj_out_dims,
        blocks=[3, 8, 36, 3],
        dims=dims,
        ac_func=ac_func,
        downsample_way=downsample_way,
    )
    
    return module



