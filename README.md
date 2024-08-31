# Wenjun-Modules

Welcome to the Wenjun-Modules repository! This project houses a collection of basic neural network modules that I have developed for use in various machine learning projects. These modules are designed to be modular, easy to integrate, and adaptable for different tasks.

## Modules

This repository includes the following modules:

- `resnet_1d.py`: Implementation of the ResNet architecture for 1D data.
- `resnet_2d.py`: Implementation of the ResNet architecture for 2D data (commonly used for image processing).
- `resnet_3d.py`: Implementation of the ResNet architecture for 3D data.
- `sparse_resnet_3d.py`: Implementation of the ResNet architecture for sparse 3D data using spconv.
- `transformer.py`: Basic implementation of the Transformer architecture, adaptable for various sequence-to-sequence tasks.
- `dit_1d.py`: Basic implementation of the Scalable Diffusion Models with Transformers, this module is suitable for 1d sequence date, and you can do some change for 2d image data based on the basic blocks.
- `attention.py`: Implementation attention mechanism for different data types, you can use it directly and add it in your owm modules.
- `pc_embedder.py`: Implementation PointNet++ (pure python) as a point cloud feature extrator, and borrowed form https://github.com/yanx27/Pointnet_Pointnet2_pytorch.

## Usage

These modules are designed to be straightforward to integrate into your projects. You can clone this repository and import the necessary module into your Python scripts and find the usage in test folder.


