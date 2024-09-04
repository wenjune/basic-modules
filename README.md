<!--
 * @Author: wenjun-VCC
 * @Date: 2024-05-14 20:36:41
 * @LastEditors: wenjun-VCC
 * @LastEditTime: 2024-09-05 01:17:03
 * @Description: __discription:__
 * @Email: wenjun.9707@gmail.com
 * @Copyright (c): 2024 by wenjun-VCC, All Rights Reserved.
-->
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

## References

- [Deep Residual Learning for Image Recognition](https://arxiv.org/abs/1512.03385) 
```
@misc{he2015deepresiduallearningimage,
    title={Deep Residual Learning for Image Recognition},
    author={Kaiming He and Xiangyu Zhang and Shaoqing Ren and Jian Sun},
    year={2015},
    eprint={1512.03385},
    archivePrefix={arXiv},
    primaryClass={cs.CV},
    url={https://arxiv.org/abs/1512.03385},
}
```

- [Attention Is All You Need](https://arxiv.org/abs/1706.03762)
```
@misc{vaswani2023attentionneed,
    title={Attention Is All You Need}, 
    author={Ashish Vaswani and Noam Shazeer and Niki Parmar and Jakob Uszkoreit and Llion Jones and Aidan N. Gomez and Lukasz Kaiser and Illia Polosukhin},
    year={2023},
    eprint={1706.03762},
    archivePrefix={arXiv},
    primaryClass={cs.CL},
    url={https://arxiv.org/abs/1706.03762}, 
}
```

- [spconv](https://github.com/traveller59/spconv/tree/master)
```
@misc{spconv2022,
    title={Spconv: Spatially Sparse Convolution Library},
    author={Spconv Contributors},
    howpublished = {\url{https://github.com/traveller59/spconv}},
    year={2022}
}
```

- [Scalable Diffusion Models with Transformers](https://arxiv.org/abs/2212.09748)
```
@misc{peebles2023scalablediffusionmodelstransformers,
    title={Scalable Diffusion Models with Transformers}, 
    author={William Peebles and Saining Xie},
    year={2023},
    eprint={2212.09748},
    archivePrefix={arXiv},
    primaryClass={cs.CV},
    url={https://arxiv.org/abs/2212.09748}, 
}
```

- [Efficient Diffusion Transformer with Step-wise Dynamic Attention Mediators](https://www.arxiv.org/abs/2408.05710)
```
@misc{pu2024efficientdiffusiontransformerstepwise,
    title={Efficient Diffusion Transformer with Step-wise Dynamic Attention Mediators}, 
    author={Yifan Pu and Zhuofan Xia and Jiayi Guo and Dongchen Han and Qixiu Li and Duo Li and Yuhui Yuan and Ji Li and Yizeng Han and Shiji Song and Gao Huang and Xiu Li},
    year={2024},
    eprint={2408.05710},
    archivePrefix={arXiv},
    primaryClass={cs.CV},
    url={https://arxiv.org/abs/2408.05710}, 
}
```

- [Agent Attention: On the Integration of Softmax and Linear Attention](https://arxiv.org/abs/2312.08874)
```
@misc{han2024agentattentionintegrationsoftmax,
    title={Agent Attention: On the Integration of Softmax and Linear Attention}, 
    author={Dongchen Han and Tianzhu Ye and Yizeng Han and Zhuofan Xia and Siyuan Pan and Pengfei Wan and Shiji Song and Gao Huang},
    year={2024},
    eprint={2312.08874},
    archivePrefix={arXiv},
    primaryClass={cs.CV},
    url={https://arxiv.org/abs/2312.08874}, 
}
```

- [PointNet: Deep Learning on Point Sets for 3D Classification and Segmentation](https://arxiv.org/abs/1612.00593)
```
@misc{qi2017pointnetdeeplearningpoint,
    title={PointNet: Deep Learning on Point Sets for 3D Classification and Segmentation}, 
    author={Charles R. Qi and Hao Su and Kaichun Mo and Leonidas J. Guibas},
    year={2017},
    eprint={1612.00593},
    archivePrefix={arXiv},
    primaryClass={cs.CV},
    url={https://arxiv.org/abs/1612.00593}, 
}
```

- [PointNet++: Deep Hierarchical Feature Learning on Point Sets in a Metric Space](https://arxiv.org/abs/1706.02413)
```
@misc{qi2017pointnetdeephierarchicalfeature,
    title={PointNet++: Deep Hierarchical Feature Learning on Point Sets in a Metric Space}, 
    author={Charles R. Qi and Li Yi and Hao Su and Leonidas J. Guibas},
    year={2017},
    eprint={1706.02413},
    archivePrefix={arXiv},
    primaryClass={cs.CV},
    url={https://arxiv.org/abs/1706.02413}, 
}
```