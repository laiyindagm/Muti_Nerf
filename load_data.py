import os
import imageio.v2 as imageio
import numpy as np
import random


def _load_camera_focal(basedir):
    focals = []
    with open(os.path.join(basedir, 'cameras.txt'), "r") as f:
        data = f.readlines()
        for line in data:
            if line[0] != '#':
                token = line.split(' ')
                focals.append(float(token[4]))
    f.close()
    return np.mean(focals[np.percentile(sorted(focals), 1) <= focals <= np.percentile(sorted(focals), 99)])


def _load_data(basedir, dataname):
    basedir = os.path.join(basedir, dataname)

    poses_arr = np.load(os.path.join(basedir, 'poses_bounds.npy'))
    poses = poses_arr[:, :-2].reshape([-1, 3, 5]).transpose([1, 2, 0])
    bds = poses_arr[:, -2:].transpose([1, 0])

    imgdir = os.path.join(basedir, 'images')

    imgfiles = [os.path.join(imgdir, f) for f in sorted(os.listdir(imgdir)) if
                f.endswith('JPG') or f.endswith('jpg') or f.endswith('png')]

    if poses.shape[-1] != len(imgfiles):
        print('Mismatch between imgs {} and poses {} !!!!'.format(len(imgfiles), poses.shape[-1]))
        return

    sh = imageio.imread(imgfiles[0]).shape
    poses[:2, 4, :] = np.array(sh[:2]).reshape([2, 1])
    poses[2, 4, :] = poses[2, 4, :] * 1.

    def imread(f):
        if f.endswith('png'):
            return imageio.imread(f)
        else:
            return imageio.imread(f)

    imgs = [imread(f)[..., :3] / 255. for f in imgfiles]
    imgs = np.stack(imgs, -1)  # [H, W, 3, num]

    print('Loaded image data:' + dataname, imgs.shape, poses[:, -1, 0])
    return poses, bds, imgs


def normalize(x):
    return x / np.linalg.norm(x)


def viewmatrix(z, up, pos):
    vec2 = normalize(z)
    vec1_avg = up
    vec0 = normalize(np.cross(vec1_avg, vec2))
    vec1 = normalize(np.cross(vec2, vec0))
    m = np.stack([vec0, vec1, vec2, pos], 1)
    return m


def poses_avg(poses):
    hwf = poses[0, :3, -1:]

    center = poses[:, :3, 3].mean(0)
    vec2 = normalize(poses[:, :3, 2].sum(0))
    up = poses[:, :3, 1].sum(0)
    c2w = np.concatenate([viewmatrix(vec2, up, center), hwf], 1)

    return c2w


def recenter_poses(poses):

    poses_ = poses+0
    bottom = np.reshape([0, 0, 0, 1.], [1, 4])
    c2w = poses_avg(poses)
    c2w = np.concatenate([c2w[:3, :4], bottom], -2)
    bottom = np.tile(np.reshape(bottom, [1, 1, 4]), [poses.shape[0], 1, 1])
    poses = np.concatenate([poses[:, :3, :4], bottom], -2)

    poses = np.linalg.inv(c2w) @ poses
    poses_[:, :3, :4] = poses[:, :3, :4]
    poses = poses_
    return poses


def load_test_data(basedir, recenter=True):
    poses, bds, ir_imgs = _load_data(basedir, "IR")
    poses, bds, rgb_imgs = _load_data(basedir, "RGB")

    poses = np.concatenate([poses[:, 1:2, :], -poses[:, 0:1, :], poses[:, 2:, :]], 1)
    poses = np.moveaxis(poses, -1, 0).astype(np.float32)
    ir_imgs = np.moveaxis(ir_imgs, -1, 0).astype(np.float32)
    rgb_imgs = np.moveaxis(rgb_imgs, -1, 0).astype(np.float32)
    bds = np.moveaxis(bds, -1, 0).astype(np.float32)

    if recenter:
        poses = recenter_poses(poses)

    i_train = poses.shape[0]
    ir_H = ir_imgs.shape[1]
    ir_W = ir_imgs.shape[2]
    rgb_H = rgb_imgs.shape[1]
    rgb_W = rgb_imgs.shape[2]

    print('IR Data:')
    print(poses.shape, ir_imgs.shape, bds.shape)
    print('RGB Data:')
    print(poses.shape, rgb_imgs.shape, bds.shape)

    ir_imgs = ir_imgs.astype(np.float32)
    rgb_imgs = rgb_imgs.astype(np.float32)
    poses = poses.astype(np.float32)
    trains = range(i_train)
    i_test = random.sample(trains, 0)  # 随机抽取k张用于测试
    i_train = [i for i in trains if i not in i_test]
    print('HOLDOUT view is', i_test)

    ir_focal = _load_camera_focal(os.path.join(basedir, "IR"))
    rgb_focal = _load_camera_focal(os.path.join(basedir, "RGB"))
    hwfs = [[ir_H, ir_W, ir_focal], [rgb_H, rgb_W, rgb_focal]]
    imgs = [ir_imgs, rgb_imgs]

    render_poses = (poses[i_train[4:124]] + poses[i_train[5:125]]) / 2

    return imgs, poses, hwfs, bds, render_poses, i_train, i_test

