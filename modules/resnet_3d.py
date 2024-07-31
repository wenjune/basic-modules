'''
Author: wenjun-VCC
Date: 2024-05-13 22:41:19
LastEditors: wenjun-VCC
LastEditTime: 2024-07-31 15:07:36
FilePath: resnet_3d.py
Description: __discription:__
Email: wenjun.9707@gmail.com
Copyright (c) 2024 by wenjun/VCC, All Rights Reserved. 
'''
import torch.nn as nn
from torchtyping import TensorType
from typing import Optional, List
from beartype import beartype

import os, sys
ROOT_PATH = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(ROOT_PATH)

from modules.attention import ConvAttention



class BasicBlock3d(nn.Module):
    
    def __init__(
        self,
        in_dims: int,
        out_dims: int,
        kernel_size: int=7,
        ac_func=nn.ReLU
    ) -> None:
        super(BasicBlock3d, self).__init__()

        assert kernel_size % 2 != 0, 'The kernel_size should be odd.'
        self.padding = kernel_size//2
        
        self.basic_conv = nn.Conv3d(in_dims, out_dims, kernel_size=kernel_size, padding=self.padding)
        self.norm1 = nn.BatchNorm3d(out_dims)
        self.ac_func = ac_func()
        self.out_fc = nn.Conv3d(out_dims, out_dims, kernel_size=1)
    
    
    @beartype
    def forward(
        self,
        x: TensorType['bs','c','x','y', 'z', float]
    ):
            
        x = self.basic_conv(x)
        x = self.norm1(x)
        x = self.ac_func(x)
        x = self.out_fc(x)
        
        return x




class Block3d(nn.Module):
    
    def __init__(
        self,
        dims: int,
        out_dims: int=None,
        kernel_size: int=3
    ) -> None:
        super(Block3d, self).__init__()
        
        assert kernel_size % 2 != 0, 'The kernel_size should be odd.'
        self.padding = 0 if kernel_size==1 else kernel_size // 2
        
        self.dims = dims
        self.out_dims = dims if out_dims is None else out_dims
        
        self.conv = nn.Conv3d(self.dims, self.out_dims, kernel_size=kernel_size, padding=self.padding)
        self.norm = nn.BatchNorm3d(self.out_dims)
    
    
    @beartype
    def forward(
        self,
        x: TensorType['bs','c','x','y', 'z', float],
    ):
        
        x = self.conv(x)
        x = self.norm(x)
        
        return x



class ShallowResnetBlock3d(nn.Module):
    
    def __init__(
        self,
        dims: int,
        out_dims: int=None,
        downsample: bool=False,
        downsample_way: str='maxpool',
        ac_fun=nn.ReLU
    ) -> None:
        super(ShallowResnetBlock3d, self).__init__()
        
        self.dims = dims
        self.out_dims = dims if out_dims is None else out_dims
        
        if downsample:
            self.downsample = nn.MaxPool3d(2, 2)
            if downsample_way == 'avgpool':
                self.downsample = nn.AvgPool3d(2, 2)
            if downsample_way == 'conv':
                self.downsample = nn.Conv3d(dims, dims, kernel_size=2, stride=2)
        else:
            self.downsample = nn.Identity()
        
        downsample_layer = nn.MaxPool3d(2, 2) if downsample else nn.Identity()
        dimchange_layer = nn.Identity() if dims==out_dims else nn.Conv3d(dims, self.out_dims, kernel_size=1, bias=False)
        
        self.short_cut = nn.Sequential(
            downsample_layer,
            dimchange_layer,
        )
        
        self.block1 = Block3d(dims=dims, out_dims=dims, kernel_size=3)
        self.block2 = Block3d(dims=dims, out_dims=self.out_dims, kernel_size=3)
        
        self.ac_func = ac_fun()
        
    
    @beartype
    def forward(
        self,
        x: TensorType['bs','c','x','y', 'z', float],
    ):
            
        residual = self.short_cut(x)

        x = self.block1(x)
        x = self.ac_func(x)
        x = self.downsample(x)
        x = self.block2(x)
        
        out = self.ac_func(residual + x)
        
        return out
        


class DeepResnetBlock3d(nn.Module):
    
    def __init__(
        self,
        dims: int,
        out_dims: int=None,
        downsample: bool=False,
        downsample_way: str='maxpool',
        ac_fun=nn.ReLU
    ) -> None:
        super(DeepResnetBlock3d, self).__init__()
        
        self.dims = dims
        self.out_dims = dims if out_dims is None else out_dims
        
        if downsample:
            self.downsample = nn.MaxPool3d(2, 2)
            if downsample_way == 'avgpool':
                self.downsample = nn.AvgPool3d(2, 2)
            if downsample_way == 'conv':
                self.downsample = nn.Conv3d(dims, dims, kernel_size=2, stride=2)
        else:
            self.downsample = nn.Identity()

        downsample_layer = nn.MaxPool3d(2, 2) if downsample else nn.Identity()
        dimchange_layer = nn.Identity() if dims==out_dims else nn.Conv3d(dims, self.out_dims, kernel_size=1, bias=False)
        
        self.short_cut = nn.Sequential(
            downsample_layer,
            dimchange_layer,
        )
        
        self.block1 = Block3d(dims=dims, out_dims=dims, kernel_size=3)
        self.block2 = Block3d(dims=dims, out_dims=dims, kernel_size=3)
        self.block3 = Block3d(dims=dims, out_dims=self.out_dims, kernel_size=3)
        
        self.ac_func = ac_fun()
        
    
    @beartype
    def forward(
        self,
        x: TensorType['bs','c','x','y', 'z', float],
    ):
            
        residual = self.short_cut(x)
            
        x = self.block1(x)
        x = self.ac_func(x)
        x = self.downsample(x)
        x = self.block2(x)
        x = self.ac_func(x)
        x = self.block3(x)
        
        out = self.ac_func(residual + x)
        
        return out
        
    

class ShallowResnet3d(nn.Module):
    
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
        super(ShallowResnet3d, self).__init__()
        
        self.blocks = blocks
        self.dims = dims
        
        self.basic_conv = BasicBlock3d(in_dims, basic_out_dims)
        self.proj_out_conv = nn.Identity() if proj_out_dims is None else nn.Conv3d(dims[-1], proj_out_dims, kernel_size=1, bias=False)
        
        self.resnet_module_list = nn.ModuleList([])
        curr_dim = basic_out_dims
        for idx, _ in enumerate(blocks):
            downsample = False
            for i in range(_):
                if i == _-1:
                    downsample = True
                self.resnet_module_list.append(ShallowResnetBlock3d(
                    curr_dim, dims[idx],
                    ac_fun=ac_func,
                    downsample=downsample,
                    downsample_way=downsample_way,
                ))
                curr_dim = dims[idx]
    
    
    @beartype  
    def forward(
        self,
        x: TensorType['bs','c','x','y', 'z', float],
    ):
        
        x = self.basic_conv(x)
        
        for module in self.resnet_module_list:
            x = module(x)
        
        x = self.proj_out_conv(x)
        
        return x
        
        
        
class DeepResnet3d(nn.Module):
    
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
        
        self.basic_conv = BasicBlock3d(in_dims, basic_out_dims)
        self.proj_out_conv = nn.Identity() if proj_out_dims is None else nn.Conv3d(dims[-1], proj_out_dims, kernel_size=1, bias=False)
        
        self.resnet_module_list = nn.ModuleList([])
        curr_dim = basic_out_dims
        for idx, _ in enumerate(blocks):
            downsample = False
            for i in range(_):
                if i == _-1:
                    downsample = True
                self.resnet_module_list.append(DeepResnetBlock3d(
                    curr_dim, dims[idx],
                    downsample=downsample,
                    ac_fun=ac_func,
                    downsample_way=downsample_way,
                ))
                curr_dim = dims[idx]
            
    
    @beartype
    def forward(
        self,
        x: TensorType['bs','c','x','y', 'z', float],
    ):
        
        x = self.basic_conv(x)
        
        for module in self.resnet_module_list:
            x = module(x)
            
        x = self.proj_out_conv(x)
        
        return x
        



def resnet18_3d(
    in_dims: int,
    proj_out_dims: int=1000,
    basic_block_dims: int=32,
    dims: List[int]=[64, 128, 156, 384],
    ac_func=nn.ReLU,
    downsample_way: str='maxpool',
):
    
    module = ShallowResnet3d(
        in_dims=in_dims,
        basic_out_dims=basic_block_dims,
        proj_out_dims=proj_out_dims,
        blocks=[2, 2, 2, 2],
        dims=dims,
        ac_func=ac_func,
        downsample_way=downsample_way,
    )
    
    return module


def resnet34_3d(
    in_dims: int,
    proj_out_dims: int=1000,
    basic_block_dims: int=32,
    dims: List[int]=[64, 128, 256, 384],
    ac_func=nn.ReLU,
    downsample_way: str='maxpool',
):
    
    module = ShallowResnet3d(
        in_dims=in_dims,
        basic_out_dims=basic_block_dims,
        proj_out_dims=proj_out_dims,
        blocks=[3, 4, 6, 3],
        dims=dims,
        ac_func=ac_func,
        downsample_way=downsample_way,
    )
    
    return module


def resnet50_3d(
    in_dims: int,
    proj_out_dims: int=1000,
    basic_block_dims: int=32,
    dims: List[int]=[64, 128, 256, 384],
    ac_func=nn.ReLU,
    downsample_way: str='maxpool',
):
    
    module = DeepResnet3d(
        in_dims=in_dims,
        basic_out_dims=basic_block_dims,
        proj_out_dims=proj_out_dims,
        blocks=[3, 4, 6, 3],
        dims=dims,
        ac_func=ac_func,
        downsample_way=downsample_way,
    )
    
    return module


def resnet101_3d(
    in_dims: int,
    proj_out_dims: int=1000,
    basic_block_dims: int=32,
    dims: List[int]=[64, 128, 256, 384],
    ac_func=nn.ReLU,
    downsample_way: str='maxpool',
):
    
    module = DeepResnet3d(
        in_dims=in_dims,
        basic_out_dims=basic_block_dims,
        proj_out_dims=proj_out_dims,
        blocks=[3, 4, 23, 3],
        dims=dims,
        ac_func=ac_func,
        downsample_way=downsample_way,
    )
    
    return module


def resnet152_3d(
    in_dims: int,
    proj_out_dims: int=1000,
    basic_block_dims: int=32,
    dims: List[int]=[64, 128, 256, 384],
    ac_func=nn.ReLU,
    downsample_way: str='maxpool',
):
    
    module = DeepResnet3d(
        in_dims=in_dims,
        basic_out_dims=basic_block_dims,
        proj_out_dims=proj_out_dims,
        blocks=[3, 8, 36, 3],
        dims=dims,
        ac_func=ac_func,
        downsample_way=downsample_way,
    )
    
    return module

