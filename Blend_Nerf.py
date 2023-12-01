from helper import *


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
        # 如果blend_flag==True, 输出[n, 1或3 + 1 + 3 + 1]
        # 否则 [n, 1或3 + 1]
        input_pts, input_views, input_ds = torch.split(x, [self.input_ch, self.input_views_ch, self.input_d_ch], dim=-1)
        h = self.pts_block(input_pts)
        if data_flag == self.rgb_flag:
            outputs = self.rgb_block(h, input_views)
        else:
            outputs = self.ir_block(h, input_views, input_ds)

        if blend_flag:
            outputs_blend = self.blend_block(h, input_views, input_ds)
            outputs = torch.cat([outputs, outputs_blend], -1)
            return outputs

        return outputs


def create_blend_nerf():
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


