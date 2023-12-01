# 将一条光线上离散的点合成一个像素点
from helper import *


def raw2outputs(raw, z_vals, rays_d, raw_noise_std):
    # raw[chunk, N, 2]
    # F.relu = max(0,x), _raw值越大越透明
    def raw2alpha(_raw, dists, act_fn=F.relu):
        return 1.-torch.exp(-act_fn(_raw)*dists)

    dists = z_vals[..., 1:] - z_vals[..., :-1]  # 计算两点Z轴之间的距离
    # 在最后加一个无穷大（1e10）的距离
    dists = torch.cat([dists, torch.Tensor([1e10]).expand(dists[..., :1].shape)], -1)  # [N_rays, N_samples]
    # 得到num_rays*1，每一个元素是（x^2+y^2+z^2）^0.5
    dists = dists * torch.norm(rays_d[..., None, :], dim=-1)

    gray_each_point = torch.sigmoid(raw[..., :1])  # [N_rays, N_samples, 1]
    noise = 0.
    if raw_noise_std > 0.:
        noise = torch.randn(raw[..., 1].shape) * raw_noise_std

    alpha = raw2alpha(raw[..., 1] + noise, dists)  # [N_rays, N_samples] 即透明度
    # 这里alpha乘的是Ti，公式等价于（1-alpha）的累乘
    # t = torch.cat([torch.ones(alpha.shape[0], 1), 1. - alpha + 1e-10], -1) 得到N_rays*（N_samples+1）
    # 每一行的内容是 [1 1-a0 1-a2 ,..., 1-a_(N_samples-1)]
    # T = torch.cumprod(t, -1) 公式中的累乘(保持第一列不变，后面的列依次累乘前列)
    # weights = alpha * T[:, :-1] 又因为公式中计算的是前 i-1 列的累积结果，所以舍去最后一列
    weights = alpha * torch.cumprod(torch.cat([torch.ones((alpha.shape[0], 1)), 1. - alpha + 1e-10], -1), -1)[:, :-1]
    gray_map = torch.sum(weights[..., None] * gray_each_point, -2)  # [N_rays, 1] 在倒数第二个维度上累加

    depth_map = torch.sum(weights * z_vals, -1)
    disp_map = 1. / torch.max(1e-10 * torch.ones_like(depth_map), depth_map / torch.sum(weights, -1))
    acc_map = torch.sum(weights, -1)

    return gray_map, disp_map, acc_map, weights, depth_map


def render_rays(ray_batch, retraw, network_fn, network_query_fn, N_samples, perturb,
                N_importance, network_fine, raw_noise_std):

    #  ray_batch [chunk, 11]
    N_rays = ray_batch.shape[0]  # 光线数量
    rays_o, rays_d = ray_batch[:, 0:3], ray_batch[:, 3:6]  # rays_o[chunk, 3], rays_d[chunk, 3]
    viewdirs = ray_batch[:, -3:]   # viewdirs[chunk, 3] 单位向量
    bounds = torch.reshape(ray_batch[..., 6:8], [-1, 1, 2])  # [chunk, 1, 2]
    near, far = bounds[..., 0], bounds[..., 1]  # [chunk, 1], [chunk, 1]
    d_near, d_far = near[0], far[0]
    #  确定空间中一个坐标的Z轴位置
    t_vals = torch.linspace(0., 1., steps=N_samples)  # 在 0-1 内生成 N_samples 个等差点
    z_vals = near * (1. - t_vals) + far * t_vals  # 方法是，a*(1 - i/(n-1)) + b*(i/(n-1)) 确定z轴坐标值[chunk, N_samples]
    z_vals = z_vals.expand([N_rays, N_samples])  # [chunk, N_samples](应该多余)

    #  随机化间距取样
    if perturb > 0.:
        mids = .5 * (z_vals[..., 1:] + z_vals[..., :-1])
        upper = torch.cat([mids, z_vals[..., -1:]], -1)
        lower = torch.cat([z_vals[..., :1], mids], -1)
        # 生成和取样点坐标数一样多的随机点
        t_rand = torch.rand(z_vals.shape)
        z_vals = lower + (upper - lower) * t_rand

    # 坐标等于光线原点加光线向量乘上z轴长度（之所以不是光线长度是因为这样比较方便）
    pts = rays_o[..., None, :] + rays_d[..., None, :] * z_vals[..., :, None]  # [chunk, N_samples, 3]
    # 计算每个坐标与光心的距离并（归一化）(将距离统一除以far-near)
    temp_rays_o = rays_o[:, None].expand(pts.shape)
    ds = torch.sqrt(torch.sum((pts - temp_rays_o)**2, dim=-1))
    #  ds = (ds - d_near) / (d_far - d_near)
    # 将光线上的每个点投入到 MLP 网络 network_fn 中前向传播得到每个点对应的 （gray，A）raw [chunk, N_samples, 2]
    raw = network_query_fn(pts, viewdirs, ds, network_fn)
    # 对这些离散点进行体积渲染，即进行积分操作
    gray_map, disp_map, acc_map, weights, depth_map = raw2outputs(raw, z_vals, rays_d, raw_noise_std)
    # gray_map [chunk, 1]

    # 分层采样的细采样
    if N_importance > 0:
        gray_map_0, disp_map_0, acc_map_0 = gray_map, disp_map, acc_map

        z_vals_mid = .5 * (z_vals[..., 1:] + z_vals[..., :-1])
        # 根据权重 weight 判断这个点在物体表面附近的概率，重新采样
        z_samples = sample_pdf(z_vals_mid, weights[..., 1:-1], N_importance, det=(perturb == 0.))
        z_samples = z_samples.detach()

        z_vals, _ = torch.sort(torch.cat([z_vals, z_samples], -1), -1)
        # [N_rays, N_samples + N_importance, 3]  生成新的采样点坐标
        pts = rays_o[..., None, :] + rays_d[..., None, :] * z_vals[..., :, None]
        # 计算每个坐标与光心的距离并归一化
        temp_rays_o = rays_o[:, None].expand(pts.shape)
        ds = torch.sqrt(torch.sum((pts - temp_rays_o) ** 2, dim=-1))
        ds = (ds - d_near) / (d_far - d_near)
        # 生成新采样点的颜色密度
        raw = network_query_fn(pts, viewdirs, ds, network_fine)
        gray_map, disp_map, acc_map, weights, depth_map = raw2outputs(raw, z_vals, rays_d, raw_noise_std)

    ret = {'gray_map': gray_map, 'disp_map': disp_map, 'acc_map': acc_map}
    if retraw:
        ret['raw'] = raw
    if N_importance > 0:
        ret['gray_map_0'] = gray_map_0
        ret['disp0'] = disp_map_0
        ret['acc0'] = acc_map_0
        ret['z_std'] = torch.std(z_samples, dim=-1, unbiased=False)

    # DEBUG
    for k in ret:
        if torch.isnan(ret[k]).any() or torch.isinf(ret[k]).any():
            print(f"! [Numerical Error] {k} contains nan or inf.")

    return ret


def batchify_rays(rays_flat, chunk, **kwargs):
    # 用较小批量的光线送入模型
    # kwargs有七个参数（加上near far有九个）
    all_ret = {}
    for i in range(0, rays_flat.shape[0], chunk):
        ret = render_rays(rays_flat[i:i+chunk], **kwargs)
        for k in ret:
            if k not in all_ret:
                all_ret[k] = []
            all_ret[k].append(ret[k])

    all_ret = {k: torch.cat(all_ret[k], 0) for k in all_ret}
    return all_ret

def render(H, W, K, chunk, rays, near, far, c2w=None, **kwargs):
    # 给定相机参数和光线，返回渲染结果
    # H, W, K, c2w 用于在只给定了渲染坐标时渲染图片时
    # rays [2, N*H*W, 3]
    if c2w is not None:
        # special case to render full image
        rays_o, rays_d = get_rays(H, W, K, c2w)
        rays_o = torch.reshape(rays_o, [-1, 3]).float()
        rays_d = torch.reshape(rays_d, [-1, 3]).float()
    else:
        rays_o, rays_d = rays  # rays_o光线的原点坐标[N*H*W, 3] , rays_d光线的方向向量[N*H*W, 3]

    viewdirs = rays_d  # viewdirs光线的方向向量[N*H*W, 3] 以下用n代N*H*W
    # 下面两句的意思是，将每一个向量坐标单位化 最后得到[n, 3]
    # viewdirs = viewdirs / torch.norm(viewdirs, dim=-1, keepdim=True)
    # viewdirs = torch.reshape(viewdirs, [-1, 3]).float()

    shape_dirs = rays_d.shape
    near, far = near * torch.ones_like(rays_d[..., :1]), far * torch.ones_like(rays_d[..., :1])  # [n]
    rays = torch.cat([rays_o, rays_d, near, far], -1)  # [n, 8]
    rays = torch.cat([rays, viewdirs], -1)  # [n, 11]

    # 开始并行计算光线属性
    all_ret = batchify_rays(rays, chunk, **kwargs)
    for k in all_ret:
        k_sh = list(shape_dirs[:-1]) + list(all_ret[k].shape[1:])
        all_ret[k] = torch.reshape(all_ret[k], k_sh)

    k_extract = ['gray_map', 'disp_map', 'acc_map']
    ret_list = [all_ret[k] for k in k_extract]
    ret_dict = {k: all_ret[k] for k in all_ret if k not in k_extract}
    return ret_list + [ret_dict]