import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import os

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class Embedder:
    def __init__(self, **kwargs):
        self.kwargs = kwargs
        self.create_embedding_fn()

    def create_embedding_fn(self):
        embed_fns = []
        d = self.kwargs['input_dims']
        out_dim = 0

        max_freq = self.kwargs['max_freq_log2']
        N_freqs = self.kwargs['num_freqs']

        if self.kwargs['log_sampling']:  # 对数间距取样
            freq_bands = 2. ** torch.linspace(0., max_freq, steps=N_freqs)  # 得到 [2^0, 2^1, ... ,2^(L-1)]
        else:
            freq_bands = torch.linspace(2. ** 0., 2. ** max_freq, steps=N_freqs)  # 得到 [2^0,...,2^(L-1)] 的等差数列

        for freq in freq_bands:
            for p_fn in self.kwargs['periodic_fns']:  # 这个数组里只有cos和sin
                embed_fns.append(lambda x, p_fn_=p_fn, freq_=freq: p_fn_(x * freq_))
                out_dim += d  # 每使用子编码公式一次就要把输出维度加上原始维度，因为每个待编码的位置维度是自定义input_dims

        self.embed_fns = embed_fns  # 相当于是一个编码公式列表[sin(2^0*x),cos(2^0*x),...]
        self.out_dim = out_dim

    def embed(self, inputs):
        # 对各个输入进行编码，给定一个输入，使用编码列表中的公式分别对他编码
        return torch.cat([fn(inputs) for fn in self.embed_fns], -1)


def get_embedder(multires, input_dims):
    embed_kwargs = {
        'input_dims': input_dims,  # 输入给编码器的数据的维度
        'max_freq_log2': multires - 1,
        'num_freqs': multires,  # 论文中的L
        'log_sampling': True,
        'periodic_fns': [torch.sin, torch.cos],
    }

    embedder_obj = (Embedder(**embed_kwargs))
    # embed 现在相当于一个编码器，具体的编码公式与论文中的一致。
    embed = lambda x, eo=embedder_obj: eo.embed(x)
    return embed, embedder_obj.out_dim


def sample_pdf(bins, weights, N_samples, det):
    # 逆变换采样
    # 根据权重 weight 判断这个点在物体表面附近的概率，重新采样
    # bins z坐标的中值[chunk, N_samples]
    weights = weights + 1e-5  # prevent nans [chunk, N_samples]
    pdf = weights / torch.sum(weights, -1, keepdim=True)  # 把每个值变为占总体占比
    cdf = torch.cumsum(pdf, -1)  # 第一列加到第二列，第二列加到第三列，...
    # cdf就是随机变量的分布函数
    cdf = torch.cat([torch.zeros_like(cdf[..., :1]), cdf], -1)  # [chunk, N_samples+1]在cdf每列权重前加一个0

    # 是否添加随机噪声
    if det:
        u = torch.linspace(0., 1., steps=N_samples)
        u = u.expand(list(cdf.shape[:-1]) + [N_samples])
    else:
        u = torch.rand(list(cdf.shape[:-1]) + [N_samples])

    # Invert CDF
    u = u.contiguous()  # 深拷贝
    inds = torch.searchsorted(cdf, u, right=True)
    below = torch.max(torch.zeros_like(inds - 1), inds - 1)
    above = torch.min((cdf.shape[-1] - 1) * torch.ones_like(inds), inds)
    inds_g = torch.stack([below, above], -1)  # (batch, N_samples, 2)

    # cdf_g = tf.gather(cdf, inds_g, axis=-1, batch_dims=len(inds_g.shape)-2)
    # bins_g = tf.gather(bins, inds_g, axis=-1, batch_dims=len(inds_g.shape)-2)
    matched_shape = [inds_g.shape[0], inds_g.shape[1], cdf.shape[-1]]
    cdf_g = torch.gather(cdf.unsqueeze(1).expand(matched_shape), 2, inds_g)
    bins_g = torch.gather(bins.unsqueeze(1).expand(matched_shape), 2, inds_g)

    denom = (cdf_g[..., 1] - cdf_g[..., 0])
    denom = torch.where(denom < 1e-5, torch.ones_like(denom), denom)
    t = (u - cdf_g[..., 0]) / denom
    samples = bins_g[..., 0] + t * (bins_g[..., 1] - bins_g[..., 0])

    return samples


def get_rays(H, W, K, c2w):
    # 返回tenser
    i, j = torch.meshgrid(torch.linspace(0, W-1, W), torch.linspace(0, H-1, H))  # pytorch's meshgrid has indexing='ij'
    i = i.t()
    j = j.t()
    dirs = torch.stack([(i-K[0][2])/K[0][0], -(j-K[1][2])/K[1][1], -torch.ones_like(i)], -1)
    # Rotate ray directions from camera frame to the world frame
    rays_d = torch.sum(dirs[..., np.newaxis, :] * c2w[:3,:3], -1)  # dot product, equals to: [c2w.dot(dir) for dir in dirs]
    rays_d = rays_d / torch.sqrt(torch.sum(rays_d ** 2, dim=-1))[..., None]
    # Translate camera frame's origin to the world frame. It is the origin of all rays.
    rays_o = c2w[:3, -1].expand(rays_d.shape)
    return rays_o, rays_d


def get_rays_np(H, W, K, c2w):
    # 返回ndarray
    # 功能同matlab meshgrid
    i, j = np.meshgrid(np.arange(W, dtype=np.float32), np.arange(H, dtype=np.float32), indexing='xy')
    # 计算图片上每一像素点坐标相对于光心的单位方向
    dirs = np.stack([(i-K[0][2])/K[0][0], -(j-K[1][2])/K[1][1], -np.ones_like(i)], -1)  # dirs的形状是H*W*3
    # 利用相机外参转置矩阵将相机坐标转换为世界坐标 rays_d的形状是H*W*3
    # dot product, equals to: [c2w.dot(dir) for dir in dirs]
    rays_d = np.sum(dirs[..., np.newaxis, :] * c2w[:3, :3], -1)
    rays_d = rays_d / np.sqrt(np.sum(rays_d**2, axis=-1))[..., None]
    # 每条光线的坐标原点的坐标 rays_o的形状是H*W*3
    rays_o = np.broadcast_to(c2w[:3, -1], np.shape(rays_d))
    return rays_o, rays_d


# Misc
def img2mse(x, y):
    return torch.mean((x - y) ** 2)


def mse2psnr(x):
    return -10. * torch.log(x) / torch.log(torch.Tensor([10.]))


def to8b(x):
    return (255*np.clip(x, 0, 1)).astype(np.uint8)


def grays_to_rgbs(grays):
    return np.expand_dims(grays, axis=3).repeat(3, axis=3)
