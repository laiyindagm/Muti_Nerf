from torch.utils.tensorboard import SummaryWriter
import os
import numpy as np
from load_data import load_test_data
from Blend_Nerf import create_blend_nerf
import torch
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"


def train():
    # 测试变量
    i_print = 2000  # 打印测试信息的轮数
    i_testset = 10000  # 印出测试集图像的轮数
    i_weights = 5000  # 保存训练信息的轮数
    i_video = 200000  # 导出视频的轮数
    L_blend = 100000  # 开始训练合成的轮数
    basedir = './logs'  # 训练数据保存文件夹
    expname = 'plane_000'  # 实验名
    writer = SummaryWriter(os.path.join(basedir, expname))
    imgs, poses, hwfs, bds, render_poses, i_train, i_test = load_test_data(basedir)
    ir_imgs, rgb_imgs = imgs
    ir_hwf, rgb_hwf = hwfs
    ir_H, ir_W, ir_focal = ir_hwf
    ir_H, ir_W = int(ir_H), int(ir_W)
    rgb_H, rgb_W, rgb_focal = rgb_hwf
    rgb_H, rgb_W = int(rgb_H), int(rgb_W)
    near = np.percentile(sorted(bds.flat), 1)
    far = np.percentile(sorted(bds.flat), 99)
    ir_K = np.array([
        [ir_focal, 0, 0.5 * ir_W],
        [0, ir_focal, 0.5 * ir_H],
        [0, 0, 1]
    ])
    rgb_K = np.array([
        [rgb_focal, 0, 0.5 * rgb_W],
        [0, rgb_focal, 0.5 * rgb_H],
        [0, 0, 1]
    ])

    render_kwargs_train, render_kwargs_test, start, grad_vars, optimizer = create_blend_nerf(basedir, expname)

    global_step = start
    bds_dict = {
        'near': near,
        'far': far,
    }
    render_kwargs_train.update(bds_dict)
    render_kwargs_test.update(bds_dict)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    render_poses = torch.Tensor(render_poses).to(device)

    N_rand = 1024  # 随机抽取N_rand条光线

