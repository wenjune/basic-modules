import torch
import torch.nn as nn
import numpy as np



def first_layer_sine_init(m):
    with torch.no_grad():
        if isinstance(m, nn.Linear):
            num_input = m.weight.size(-1)
            m.weight.uniform_(-1 / num_input, 1 / num_input)


def frequency_init(freq):
    def init(m):
        with torch.no_grad():
            if isinstance(m, nn.Linear):
                num_input = m.weight.size(-1)
                m.weight.uniform_(-np.sqrt(6 / num_input) / freq, np.sqrt(6 / num_input) / freq)
    return init


class Sine(nn.Module):
    """Sine Activation Function."""
    
    def __init__(self, w0=30.):
        super().__init__()
        self.w0 = w0
        
    def forward(self, x):
        return torch.sin(self.w0 * x)
    

class Siren(nn.Module):
    def __init__(
        self,
        input_dim,
        output_dim,
        hidden_dim=256,
        num_hidden_layers=2,
        freq=30,
    ):
        super(Siren, self).__init__()
        
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            Sine(freq),
            *[nn.Sequential(nn.Linear(hidden_dim, hidden_dim), Sine(freq)) for _ in range(num_hidden_layers)],
            nn.Linear(hidden_dim, output_dim)
        )
        
        self.net.apply(frequency_init(freq))
        self.net[0].apply(first_layer_sine_init)
        
        
    def forward(self, input):
        return self.net(input)
    
    
    

class FourierFeatureTransform(nn.Module):
    def __init__(self, num_input_channels, mapping_size, scale=10):
        super().__init__()

        self._num_input_channels = num_input_channels
        self._mapping_size = mapping_size
        self._B = nn.Parameter(torch.randn((num_input_channels, mapping_size)) * scale, requires_grad=False)

    def forward(self, x):
        B, N, C = x.shape
        x = (x.reshape(B*N, C) @ self._B).reshape(B, N, -1)
        x = 2 * np.pi * x
        return torch.cat([torch.sin(x), torch.cos(x)], dim=-1)
    
    

class CoordEmbedder:
    def __init__(self, input_dims, max_freq_log2, num_freqs, include_input=True, log_sampling=True, periodic_fns=[torch.sin, torch.cos]):
        """
        初始化 Embedder
        :param input_dims: 输入的维度 (例如: x, y, z 的维度为 3)
        :param max_freq_log2: 最大频率对数 (2^max_freq_log2)
        :param num_freqs: 频率数量
        :param include_input: 是否包含原始输入
        :param log_sampling: 是否在对数空间中均匀采样频率
        :param periodic_fns: 周期性函数 (正弦和余弦)
        """
        self.input_dims = input_dims
        self.max_freq_log2 = max_freq_log2
        self.num_freqs = num_freqs
        self.include_input = include_input
        self.log_sampling = log_sampling
        self.periodic_fns = periodic_fns

        # 创建位置编码函数
        self.create_embedding_fn()

    def create_embedding_fn(self):
        """
        创建位置编码函数，将输入映射到高维空间
        """
        embed_fns = []  # 存储所有编码函数
        out_dim = 0     # 输出的总维度

        # 可选：包含原始输入
        if self.include_input:
            embed_fns.append(lambda x: x)  # 恒等函数
            out_dim += self.input_dims

        # 计算频率范围
        if self.log_sampling:
            freq_bands = 2.**torch.linspace(0., self.max_freq_log2, steps=self.num_freqs)
        else:
            freq_bands = torch.linspace(2.**0., 2.**self.max_freq_log2, steps=self.num_freqs)

        # 为每个频率添加正弦和余弦函数
        for freq in freq_bands:
            for p_fn in self.periodic_fns:
                embed_fns.append(lambda x, p_fn=p_fn, freq=freq: p_fn(x * freq))
                out_dim += self.input_dims

        self.embed_fns = embed_fns  # 保存所有编码函数
        self.out_dim = out_dim      # 保存总输出维度

    def embed(self, inputs):
        """
        将输入数据编码为高维特征
        :param inputs: 输入的低维数据 (形状: [N, input_dims])
        :return: 编码后的高维特征 (形状: [N, out_dim])
        """
        return torch.cat([fn(inputs) for fn in self.embed_fns], dim=-1)
    
    


class PointEmbed(nn.Module):
    def __init__(self, hidden_dim=48, dim=128):
        super().__init__()

        assert hidden_dim % 6 == 0

        self.embedding_dim = hidden_dim
        e = torch.pow(2, torch.arange(self.embedding_dim // 6)).float() * np.pi
        e = torch.stack([
            torch.cat([e, torch.zeros(self.embedding_dim // 6),
                        torch.zeros(self.embedding_dim // 6)]),
            torch.cat([torch.zeros(self.embedding_dim // 6), e,
                        torch.zeros(self.embedding_dim // 6)]),
            torch.cat([torch.zeros(self.embedding_dim // 6),
                        torch.zeros(self.embedding_dim // 6), e]),
        ])
        self.register_buffer('basis', e)  # 3 x 16

        self.mlp = nn.Linear(self.embedding_dim+3, dim)

    @staticmethod
    def embed(input, basis):
        projections = torch.einsum(
            'bnd,de->bne', input, basis)
        embeddings = torch.cat([projections.sin(), projections.cos()], dim=2)
        return embeddings
    
    def forward(self, input):
        # input: B x N x 3
        embed = self.mlp(torch.cat([self.embed(input, self.basis), input], dim=2)) # B x N x C
        return embed


