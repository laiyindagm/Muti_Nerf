import imageio.v2 as imageio
from render import batchify_rays
from helper import *
import time
from tqdm import tqdm


class Pts_Block(nn.Module):
    def __init__(self, D, W, input_ch, skip):
        super(Pts_Block, self).__init__()
        self.D = D
        self.W = W
        self.input_ch = input_ch
        self.skip = skip
        # 在网络的第（skips+1）层，额外添加input_ch维的输入数据
        Linears_list = [nn.Linear(input_ch, W)]
        for i in range(D-1):
            Linears_input_ch = W
            if i == skip:
                Linears_input_ch += input_ch
            Linears_list.append(nn.Linear(Linears_input_ch, W))
        self.pts_linears = nn.ModuleList(Linears_list)

    def forward(self, input_pts):
        assert input_pts.shape[-1] == self.input_ch
        h = input_pts
        for i, l in enumerate(self.pts_linears):
            h = self.pts_linears[i](h)
            h = F.relu_(h)
            if i == self.skip:
                h = torch.cat([input_pts, h], -1)
        return h


class IR_Block(nn.Module):
    def __init__(self, feature_D,  W, input_views_ch, input_ds_ch):
        super(IR_Block, self).__init__()
        self.W = W
        self.feature_D = feature_D
        self.input_views_ch = input_views_ch
        self.input_ds_ch = input_ds_ch

        self.views_linear = nn.Linear(input_views_ch + W, W)
        self.ds_linear = nn.Linear(input_ds_ch + W, W // 2)

        feature_list = []
        for i in range(feature_D):
            feature_list.append(nn.Linear(W, W))
        self.feature_linears = nn.ModuleList(feature_list)

        self.alpha_linear = nn.Linear(W, 1)  # 输出透明度值的全连接层
        self.gray_linear = nn.Linear(W // 2, 1)

    def forward(self, h, input_views, input_ds):
        assert input_views.shape[-1] == self.input_views_ch
        assert input_ds.shape[-1] == self.input_ds_ch
        alpha = self.alpha_linear(h)

        for i in range(self.feature_D):
            h = self.feature_linears[i](h)
            h = F.relu_(h)

        h = torch.cat([h, input_views], -1)
        h = self.views_linear(h)
        h = F.relu(h)
        h = torch.cat([h, input_ds], -1)
        h = self.ds_linear(h)
        h = F.relu(h)

        gray = self.gray_linear(h)
        outputs = torch.cat([gray, alpha], -1)
        return outputs

class RGB_Block(nn.Module):
    def __init__(self, feature_D, W, input_views_ch):
        super(RGB_Block, self).__init__()
        self.W = W
        self.feature_D = feature_D
        self.input_views_ch = input_views_ch

        self.views_linear = nn.Linear(input_views_ch + W, W // 2)

        feature_list = []
        for i in range(feature_D):
            feature_list.append(nn.Linear(W, W))
        self.feature_linears = nn.ModuleList(feature_list)

        self.alpha_linear = nn.Linear(W, 1)  # 输出透明度值的全连接层
        self.rgb_linear = nn.Linear(W // 2, 3)

    def forward(self, h, input_views):
        assert input_views.shape[-1] == self.input_views_ch
        alpha = self.alpha_linear(h)

        for i in range(self.feature_D):
            h = self.feature_linears[i](h)
            h = F.relu_(h)

        h = torch.cat([h, input_views], -1)
        h = self.views_linear(h)
        h = F.relu(h)

        rgb = self.rgb_linear(h)
        outputs = torch.cat([rgb, alpha], -1)
        return outputs

class Blend_Block(nn.Module):
    def __init__(self, D, W, input_views_ch, input_ds_ch, skip_alpha=0):
        super(Blend_Block, self).__init__()
        self.W = W
        self.D = D
        self.input_views_ch = input_views_ch
        self.input_ds_ch = input_ds_ch
        self.skip_alpha = skip_alpha

        assert skip_alpha < D

        self.views_linear = nn.Linear(input_views_ch + W, W)
        self.ds_linear = nn.Linear(input_ds_ch + W, W // 2)
        linear_list = []
        for i in range(D):
            linear_list.append(nn.Linear(W, W))
        self.linears = nn.ModuleList(linear_list)
        self.alpha_linear = nn.Linear(W, 1)  # 输出透明度值的全连接层
        self.blend_linear = nn.Linear(W // 2, 3)

    def forward(self, h, input_views, input_ds):
        assert input_views.shape[-1] == self.input_views_ch
        assert input_ds.shape[-1] == self.input_ds_ch

        alpha = self.alpha_linear(h)

        for i in range(self.D):
            if i == self.skip_alpha:
                alpha = self.alpha_linear(h)
            h = self.linears[i](h)
            h = F.relu_(h)

        h = torch.cat([h, input_views], -1)
        h = self.views_linear(h)
        h = F.relu(h)
        h = torch.cat([h, input_ds], -1)
        h = self.ds_linear(h)
        h = F.relu(h)

        blend = self.blend_linear(h)
        outputs = torch.cat([blend, alpha], -1)
        return outputs


class Blend_Nerf(nn.Module):
    rgb_flag = 0
    ir_flag = 1

    def __init__(self, D, W, input_ch, input_views_ch, input_ds_ch, skip, feature_D=1, skip_alpha=0):
        super(Blend_Nerf, self).__init__()
        self.pts_block = Pts_Block(D, W, input_ch, skip)
        self.ir_block = IR_Block(feature_D,  W, input_views_ch, input_ds_ch)
        self.rgb_block = RGB_Block(feature_D, W, input_views_ch)
        self.blend_block = Blend_Block(D, W, input_views_ch, input_ds_ch, skip_alpha)

    def forward(self, x, data_flag, blend_flag=False):
        # 如果blend_flag==True, 那么将输出融合后rgb
        input_pts, input_views, input_ds = torch.split(x, [self.input_ch, self.input_views_ch, self.input_d_ch], dim=-1)
        h = self.pts_block(input_pts)

        if blend_flag:
            outputs = self.blend_block(h, input_views, input_ds)
            return outputs

        if data_flag == self.rgb_flag:
            outputs = self.rgb_block(h, input_views)
        else:
            outputs = self.ir_block(h, input_views, input_ds)

        return outputs


def create_blend_nerf(basedir, expname):
    # 获得编码器 以及编码后数据（坐标）维度
    L_input = 7
    L_views = 4
    L_d = 4
    embed_fn, input_ch = get_embedder(L_input, 3)
    embed_views_fn, input_views_ch = get_embedder(L_views, 3)  # 视角数据转换成了三维的单位向量
    embed_d_fn, input_ds_ch = get_embedder(L_d, 1)

    # 模型参数设定
    netdepth = 6
    netwidth = 256
    skip = 3  # 第skip+1层网络加入位置信息
    feature_D = 1
    skip_alpha = 0

    # 训练参数设定
    netchunk = 1024 * 4  # 实际每次送入模型的数据量
    lrate = 5e-4  # 学习率
    start = 0  # 训练起点轮数
    perturb = True  # 是否加入随机扰动
    raw_noise_std = 1e0  # 为了正规化输出sigma_a而添加的噪声的标准差（也可为0）

    # 采样点数量
    N_samples = 64
    N_importance = 64

    # 初始化模型参数
    model = Blend_Nerf(netdepth, netwidth, input_ch, input_views_ch, input_ds_ch, skip, feature_D,
                       skip_alpha).to(device)
    model_fine = Blend_Nerf(netdepth, netwidth, input_ch, input_views_ch, input_ds_ch, skip, feature_D,
                            skip_alpha).to(device)
    grad_vars = list(model.parameters()) + list(model_fine.parameters())

    # 给定点坐标，方向，距离，查询网络，以及其他参数,得到该点在该网络下的输出（[gray,alpha]）
    # 这样写使得其他模型可以复用run_network
    def sub_query_fn(data_flag, blend_flag):
        def network_query_fn(inputs, viewdirs, ds, network_fn):
            def fn(x):
                return network_fn(x, data_flag, blend_flag)
            return run_network(inputs, viewdirs, ds, fn, embed_fn, embed_views_fn, embed_d_fn, netchunk)
        return network_query_fn

    # 创建网络优化器
    optimizer = torch.optim.Adam(params=grad_vars, lr=lrate, betas=(0.9, 0.999))

    # 加载检查点
    check_points = [os.path.join(basedir, expname, f) for f in sorted(os.listdir(os.path.join(basedir, expname)))
                    if 'tar' in f]
    if len(check_points) > 0:
        print('Found ckpts', check_points)
        check_point_path = check_points[-1]
        print('Reloading from', check_point_path)
        check_point = torch.load(check_point_path)
        # 加载训练数据
        start = check_point['global_step']
        optimizer.load_state_dict(check_point['optimizer_state_dict'])
        # 加载模型
        model.load_state_dict(check_point['network_fn_state_dict'])
        model_fine.load_state_dict(check_point['network_fine_state_dict'])

    # 所有参数以字典形式返回
    render_kwargs_train = {
        'sub_query_fn': sub_query_fn,
        'perturb': perturb,
        'raw_noise_std': raw_noise_std,
        'N_samples': N_samples, 'N_importance': N_importance,
        'network_fn': model, 'network_fine': model_fine
    }

    render_kwargs_test = {k: render_kwargs_train[k] for k in render_kwargs_train}
    render_kwargs_test['perturb'] = False
    render_kwargs_test['raw_noise_std'] = 0.0

    return render_kwargs_train, render_kwargs_test, start, grad_vars, optimizer


def train_blend_nerf(data_flag, blend_flag, chunk, rays, target_s, near, far, sub_query_fn, **kwargs):
    # 返回loss, psnr
    rays_o, rays_d = rays
    viewdirs = rays_d  # 在生成时就已经是单位向量
    shape_dirs = rays_d.shape
    near, far = near * torch.ones_like(rays_d[..., :1]), far * torch.ones_like(rays_d[..., :1])  # [n]
    rays = torch.cat([rays_o, rays_d, near, far], -1)  # [n, 8]
    rays = torch.cat([rays, viewdirs], -1)  # [n, 11]
    kwargs['network_query_fn'] = sub_query_fn(data_flag, blend_flag)
    # 开始并行计算光线属性
    all_ret = batchify_rays(rays, chunk, **kwargs)
    for k in all_ret:
        k_sh = list(shape_dirs[:-1]) + list(all_ret[k].shape[1:])
        all_ret[k] = torch.reshape(all_ret[k], k_sh)

    color = all_ret['color_map']
    color_0 = all_ret['color_map_0']

    if blend_flag:
        if data_flag == Blend_Nerf.rgb_flag:
            # rgb是真实传入的target_s
            kwargs['network_query_fn'] = sub_query_fn(Blend_Nerf.ir_flag, False)
            with torch.no_grad():
                all_ret = batchify_rays(rays, chunk, **kwargs)
            for k in all_ret:
                k_sh = list(shape_dirs[:-1]) + list(all_ret[k].shape[1:])
                all_ret[k] = torch.reshape(all_ret[k], k_sh)
            temp = all_ret['color_map'].detach().expand_as(target_s)
        else:
            # ir是真实传入的target_s
            kwargs['network_query_fn'] = sub_query_fn(Blend_Nerf.rgb_flag, False)
            with torch.no_grad():
                all_ret = batchify_rays(rays, chunk, **kwargs)
            for k in all_ret:
                k_sh = list(shape_dirs[:-1]) + list(all_ret[k].shape[1:])
                all_ret[k] = torch.reshape(all_ret[k], k_sh)
            temp = all_ret['color_map'].detach()
            target_s = target_s.expand_as(temp)
        target_s = (temp + target_s) / 2

    img_loss = img2mse(color, target_s)  # 差值平方的均值
    loss = img_loss
    psnr = mse2psnr(img_loss)  # 将损失转换为 PSNR 指标
    img_loss0 = img2mse(color_0, target_s)
    loss = loss + img_loss0

    return loss, psnr


def apply_blend_nerf(data_flag, blend_flag, poses, H, W, K, c2w, near, far, chunk, sub_query_fn, savedir, **kwargs):
    rays_o, rays_d = get_rays(H, W, K, c2w)
    rays_o = torch.reshape(rays_o, [-1, 3]).float()
    rays_d = torch.reshape(rays_d, [-1, 3]).float()
    viewdirs = rays_d
    shape_dirs = rays_d.shape
    near, far = near * torch.ones_like(rays_d[..., :1]), far * torch.ones_like(rays_d[..., :1])  # [n]
    rays = torch.cat([rays_o, rays_d, near, far], -1)  # [n, 8]
    rays = torch.cat([rays, viewdirs], -1)  # [n, 11]
    kwargs['network_query_fn'] = sub_query_fn(data_flag, blend_flag)
    colors = []

    t = time.time()
    for i, c2w in enumerate(tqdm(poses)):
        print(i, time.time() - t)  # 打印渲染时间
        t = time.time()

        all_ret = batchify_rays(rays, chunk, **kwargs)
        for k in all_ret:
            k_sh = list(shape_dirs[:-1]) + list(all_ret[k].shape[1:])
            all_ret[k] = torch.reshape(all_ret[k], k_sh)
        color = all_ret['color_map']
        if blend_flag is False and data_flag == Blend_Nerf.ir_flag:
            color = color[..., None].expand([H, W, 3])
        colors.append(color.cpu().numpy())
        if i == 0:
            print(color.shape)

        color8 = to8b(colors[-1])
        filename = os.path.join(savedir, '{:03d}.png'.format(i))
        imageio.imwrite(filename, color8)

    colors = np.stack(colors, 0)
    return colors



