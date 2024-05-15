'''
Author: wenjun-VCC
Date: 2024-05-13 22:40:45
LastEditors: wenjun-VCC
LastEditTime: 2024-05-15 22:05:15
FilePath: resnet_2d.py
Description: __discription:__
Email: wenjun.9707@gmail.com
Copyright (c) 2024 by wenjun/VCC, All Rights Reserved. 
'''
import torch
import torch.nn as nn
from torchtyping import TensorType
from typing import Optional, List
from beartype import beartype



class BasicBlock2d(nn.Module):
    
    def __init__(
        self,
        in_dims: int,
        out_dims: int=None,
        kernel_size: int=7,
        ac_func=nn.ReLU,
        norm=nn.BatchNorm2d,
        groups: int=8,
    ) -> None:
        super(BasicBlock2d, self).__init__()

        assert kernel_size % 2 != 0, 'The kernel_size should be odd.'
        self.padding = kernel_size//2
        
        self.out_dims = in_dims if out_dims is None else out_dims
        
        if norm == nn.GroupNorm:
            self.norm = norm(groups, self.out_dims)
        else:
            self.norm = norm(self.out_dims)
        
        self.basic_conv = nn.Conv2d(in_dims, self.out_dims, kernel_size=kernel_size, padding=self.padding)
        self.ac_func = ac_func()
        self.out_fc = nn.Conv2d(out_dims, out_dims, kernel_size=1)
    
    
    @beartype
    def forward(
        self,
        x: TensorType['bs','c','h','w', float]
    ):
            
        x = self.basic_conv(x)
        x = self.norm(x)
        x = self.ac_func(x)
        x = self.out_fc(x)
        
        return x




class Block2d(nn.Module):
    
    def __init__(
        self,
        dims: int,
        out_dims: int=None,
        kernel_size: int=3,
        norm=nn.BatchNorm2d,
        groups: int=8,
    ) -> None:
        super(Block2d, self).__init__()
        
        assert kernel_size % 2 != 0, 'The kernel_size should be odd.'
        self.padding = 0 if kernel_size==1 else kernel_size // 2
        
        self.dims = dims
        self.out_dims = dims if out_dims is None else out_dims
        
        if norm == nn.GroupNorm:
            self.norm = norm(groups, self.out_dims)
        else:
            self.norm = norm(self.out_dims)
        
        self.conv = nn.Conv2d(self.dims, self.out_dims, kernel_size=kernel_size, padding=self.padding)
    
    
    @beartype
    def forward(
        self,
        x: TensorType['bs','c','h','w', float],
    ):
        
        x = self.conv(x)
        x = self.norm(x)
        
        return x



class ShallowResnetBlock2d(nn.Module):
    
    def __init__(
        self,
        dims: int,
        out_dims: int=None,
        downsample: bool=False,
        downsample_way: str='maxpool',
        ac_fun=nn.ReLU,
        norm=nn.BatchNorm2d,
        groups: int=8,
    ) -> None:
        super(ShallowResnetBlock2d, self).__init__()
        
        self.dims = dims
        self.out_dims = dims if out_dims is None else out_dims
        
        self.downsample_way = nn.MaxPool2d(2, 2)
        if downsample_way == 'avgpool':
            self.downsample_way = nn.AvgPool2d(2, 2)
        if downsample_way == 'conv':
            self.downsample_way = nn.Conv2d(dims, dims, kernel_size=2, stride=2)
        
        downsample_layer = nn.MaxPool2d(2, 2) if downsample else nn.Identity()
        dimchange_layer = nn.Identity() if dims==out_dims else nn.Conv2d(dims, self.out_dims, kernel_size=1, bias=False)
        
        self.short_cut = nn.Sequential(
            downsample_layer,
            dimchange_layer,
        )
        
        self.block1 = Block2d(
            dims=dims,
            out_dims=dims,
            kernel_size=3,
            norm=norm,
            groups=groups,
        )
        self.downsample = self.downsample_way if downsample else nn.Identity()
        self.block2 = Block2d(
            dims=dims,
            out_dims=self.out_dims,
            kernel_size=3,
            norm=norm,
            groups=groups,
        )
        
        self.ac_func = ac_fun()
        
    
    @beartype
    def forward(
        self,
        x: TensorType['bs','c','h','w', float],
    ):
            
        residual = self.short_cut(x)

        x = self.block1(x)
        x = self.ac_func(x)
        x = self.downsample(x)
        x = self.block2(x)
        
        out = self.ac_func(residual + x)
        
        return out
        


class DeepResnetBlock2d(nn.Module):
    
    def __init__(
        self,
        dims: int,
        out_dims: int=None,
        downsample: bool=False,
        downsample_way: str='maxpool',
        ac_fun=nn.ReLU,
        norm=nn.BatchNorm2d,
        groups: int=8,
    ) -> None:
        super(DeepResnetBlock2d, self).__init__()
        
        self.dims = dims
        self.out_dims = dims if out_dims is None else out_dims
        
        self.downsample_way = nn.MaxPool2d(2, 2)
        if downsample_way == 'avgpool':
            self.downsample_way = nn.AvgPool2d(2, 2)
        if downsample_way == 'conv':
            self.downsample_way = nn.Conv2d(dims, dims, kernel_size=2, stride=2)

        downsample_layer = nn.MaxPool2d(2, 2) if downsample else nn.Identity()
        dimchange_layer = nn.Identity() if dims==out_dims else nn.Conv2d(dims, self.out_dims, kernel_size=1, bias=False)
        
        self.short_cut = nn.Sequential(
            downsample_layer,
            dimchange_layer,
        )
        
        self.block1 = Block2d(
            dims=dims,
            out_dims=dims,
            kernel_size=3,
            norm=norm,
            groups=groups,
        )
        self.block2 = Block2d(
            dims=dims,
            out_dims=dims,
            kernel_size=3,
            norm=norm,
            groups=groups,
        )
        self.downsample = self.downsample_way if downsample else nn.Identity()
        self.block3 = Block2d(
            dims=dims,
            out_dims=self.out_dims,
            kernel_size=3,
            norm=norm,
            groups=groups,
        )
        
        self.ac_func = ac_fun()
        
    
    @beartype
    def forward(
        self,
        x: TensorType['bs','c','h','w', float],
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
        
    

class ShallowResnet2d(nn.Module):
    
    def __init__(
        self,
        in_dims: int,
        basic_out_dims: int,
        proj_out_dims: int=None,
        blocks: List[int]=[2, 2, 2, 2],
        dims: List[int]=[128, 256, 384, 512],
        ac_func=nn.ReLU,
        norm=nn.BatchNorm2d,
        groups: int=8,
        downsample_way: str='maxpool'
    ) -> None:
        super(ShallowResnet2d, self).__init__()
        
        self.blocks = blocks
        self.dims = dims
        
        self.basic_conv = BasicBlock2d(
            in_dims,
            basic_out_dims,
            ac_func=ac_func,
            norm=norm,
            groups=groups,
        )
        self.proj_out_conv = nn.Identity() if proj_out_dims is None else nn.Conv2d(dims[-1], proj_out_dims, kernel_size=1, bias=False)
        
        self.resnet_module_list = nn.ModuleList([])
        curr_dim = basic_out_dims
        for idx, _ in enumerate(blocks):
            downsample = False
            for i in range(_):
                if i == _-1:
                    downsample = True
                self.resnet_module_list.append(ShallowResnetBlock2d(
                    curr_dim, dims[idx],
                    ac_fun=ac_func,
                    downsample=downsample,
                    downsample_way=downsample_way,
                    norm=norm,
                    groups=groups,
                ))
                curr_dim = dims[idx]
    
    
    @beartype       
    def forward(
        self,
        x: TensorType['bs','c','h','w', float],
    ):
        
        x = self.basic_conv(x)
        
        for module in self.resnet_module_list:
            x = module(x)
        
        x = self.proj_out_conv(x)
        
        return x
        
        
        
class DeepResnet2d(nn.Module):
    
    def __init__(
        self,
        in_dims: int,
        basic_out_dims: int,
        proj_out_dims: int=None,
        blocks: List[int]=[3, 4, 6, 3],
        dims: List[int]=[128, 256, 384, 512],
        ac_func=nn.ReLU,
        norm=nn.BatchNorm2d,
        groups: int=8,
        downsample_way: str='maxpool',
    ) -> None:
        super().__init__()
        
        self.blocks = blocks
        self.dims = dims
        
        self.basic_conv = BasicBlock2d(
            in_dims,
            basic_out_dims,
            ac_func=ac_func,
            norm=norm,
            groups=groups,
        )
        self.proj_out_conv = nn.Identity() if proj_out_dims is None else nn.Conv2d(dims[-1], proj_out_dims, kernel_size=1, bias=False)
        
        self.resnet_module_list = nn.ModuleList([])
        curr_dim = basic_out_dims
        for idx, _ in enumerate(blocks):
            downsample = False
            for i in range(_):
                if i == _-1:
                    downsample = True
                self.resnet_module_list.append(DeepResnetBlock2d(
                    curr_dim, dims[idx],
                    downsample=downsample,
                    ac_fun=ac_func,
                    downsample_way=downsample_way,
                    norm=norm,
                    groups=groups,
                ))
                curr_dim = dims[idx]
            
    
    @beartype
    def forward(
        self,
        x: TensorType['bs','c','h','w', float],
    ):
        
        x = self.basic_conv(x)
        
        for module in self.resnet_module_list:
            x = module(x)
            
        x = self.proj_out_conv(x)
        
        return x
        



def resnet18_2d(
    in_dims: int,
    proj_out_dims: int=1000,
    basic_block_dims: int=32,
    dims: List[int]=[64, 128, 156, 384],
    ac_func=nn.ReLU,
    norm=nn.BatchNorm2d,
    groups: int=8,
    downsample_way: str='maxpool',
):
    
    module = ShallowResnet2d(
        in_dims=in_dims,
        basic_out_dims=basic_block_dims,
        proj_out_dims=proj_out_dims,
        blocks=[2, 2, 2, 2],
        dims=dims,
        ac_func=ac_func,
        downsample_way=downsample_way,
        norm=norm,
        groups=groups,
    )
    
    return module


def resnet34_2d(
    in_dims: int,
    proj_out_dims: int=1000,
    basic_block_dims: int=32,
    dims: List[int]=[64, 128, 256, 384],
    ac_func=nn.ReLU,
    norm=nn.BatchNorm2d,
    groups: int=8,
    downsample_way: str='maxpool',
):
    
    module = ShallowResnet2d(
        in_dims=in_dims,
        basic_out_dims=basic_block_dims,
        proj_out_dims=proj_out_dims,
        blocks=[3, 4, 6, 3],
        dims=dims,
        ac_func=ac_func,
        downsample_way=downsample_way,
        norm=norm,
        groups=groups,
    )
    
    return module


def resnet50_2d(
    in_dims: int,
    proj_out_dims: int=1000,
    basic_block_dims: int=32,
    dims: List[int]=[64, 128, 256, 384],
    ac_func=nn.ReLU,
    norm=nn.BatchNorm2d,
    groups: int=8,
    downsample_way: str='maxpool',
):
    
    module = DeepResnet2d(
        in_dims=in_dims,
        basic_out_dims=basic_block_dims,
        proj_out_dims=proj_out_dims,
        blocks=[3, 4, 6, 3],
        dims=dims,
        ac_func=ac_func,
        downsample_way=downsample_way,
        norm=norm,
        groups=groups,
    )
    
    return module


def resnet101_2d(
    in_dims: int,
    proj_out_dims: int=1000,
    basic_block_dims: int=32,
    dims: List[int]=[64, 128, 256, 384],
    ac_func=nn.ReLU,
    norm=nn.BatchNorm2d,
    groups: int=8,
    downsample_way: str='maxpool',
):
    
    module = DeepResnet2d(
        in_dims=in_dims,
        basic_out_dims=basic_block_dims,
        proj_out_dims=proj_out_dims,
        blocks=[3, 4, 23, 3],
        dims=dims,
        ac_func=ac_func,
        downsample_way=downsample_way,
        norm=norm,
        groups=groups,
    )
    
    return module


def resnet152_2d(
    in_dims: int,
    proj_out_dims: int=1000,
    basic_block_dims: int=32,
    dims: List[int]=[64, 128, 256, 384],
    ac_func=nn.ReLU,
    norm=nn.BatchNorm2d,
    groups: int=8,
    downsample_way: str='maxpool',
):
    
    module = DeepResnet2d(
        in_dims=in_dims,
        basic_out_dims=basic_block_dims,
        proj_out_dims=proj_out_dims,
        blocks=[3, 8, 36, 3],
        dims=dims,
        ac_func=ac_func,
        downsample_way=downsample_way,
        norm=norm,
        groups=groups,
    )
    
    return module
