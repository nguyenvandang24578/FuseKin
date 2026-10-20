"""
Mau cho LIFTER (cfg.DATASET.lift_only) tu bo 2D-only co pseudo-GT 3D NeuralAnnot (MSCOCO, CrowdPose).
  2D dau vao : Halpe-26 (doi H36M-17), lat + affine giong Human36M._halpe_input
  GT 3D      : SMPL NeuralAnnot -> mesh (met, he camera) -> H36M regressor -> root-relative -> xoay aug
  cam_bbox   : neu cfg.DATASET.lift_cam_bbox (camera = cam_param cua NeuralAnnot, la camera UOC LUONG)
Tra ve (inputs, targets, meta) cung bo key voi nhanh lift_only cua Human36M / MuCo.
"""
import numpy as np
import torch

from core.config import cfg
from utils.preprocessing import augmentation_noimg
from utils.cam_bbox import cam_bbox_feat

H36M_FLIP_PAIRS = ((1, 4), (2, 5), (3, 6), (14, 11), (15, 12), (16, 13))


def halpe_to_crop(k, do_flip, img_shape, img2bb_trans, thr):
    """k: (17, 3) Halpe da doi H36M-17, pixel anh goc + score. -> xy (17, 2) khung hm, mask (17, 1)."""
    k = k.copy()
    if do_flip:
        k[:, 0] = img_shape[1] - 1 - k[:, 0]
        for a, b in H36M_FLIP_PAIRS:
            k[a], k[b] = k[b].copy(), k[a].copy()
    xy1 = np.concatenate((k[:, :2], np.ones_like(k[:, :1])), 1)
    xy = np.dot(img2bb_trans, xy1.transpose(1, 0)).transpose(1, 0)[:, :2]
    xy[:, 0] = xy[:, 0] / cfg.input_img_shape[1] * cfg.output_hm_shape[2]
    xy[:, 1] = xy[:, 1] / cfg.input_img_shape[0] * cfg.output_hm_shape[1]
    inside = (xy[:, 0] >= 0) * (xy[:, 0] < cfg.output_hm_shape[2]) * (xy[:, 1] >= 0) * (xy[:, 1] < cfg.output_hm_shape[1])
    mask = ((k[:, 2] > thr) * inside).reshape(-1, 1).astype(np.float32)
    if cfg.DATASET.lift_soft_conf:
        mask = mask * np.clip(k[:, 2:3], 0.0, 1.0).astype(np.float32)
    return xy.astype(np.float32), mask


def lift_pseudo3d_sample(ds, data, smpl_param, cam_param):
    """
    ds: dataset (MSCOCO / CrowdPose) co: data_split, get_smpl_coord, h36m_joint_regressor, halpe26_conf_thr
    data: 1 phan tu datalist (can 'img_shape', 'bbox', 'halpe_h36m')
    smpl_param, cam_param: NeuralAnnot cua nguoi do
    """
    img_shape, bbox = data['img_shape'], data['bbox']
    img2bb_trans, _, rot, do_flip = augmentation_noimg(bbox, ds.data_split, img_shape)

    xy, mask = halpe_to_crop(data['halpe_h36m'], do_flip, img_shape, img2bb_trans, ds.halpe26_conf_thr)

    # Pseudo-GT 3D: get_smpl_coord da xu ly lat (pose + tinh tien) -> mesh khop anh da lat
    mesh, _, _, _ = ds.get_smpl_coord(smpl_param, cam_param, do_flip, img_shape)    # (6890, 3) met
    j3d = np.dot(ds.h36m_joint_regressor, mesh).astype(np.float32)                    # (17, 3)
    j3d = j3d - j3d[0:1]                                                                # root = Pelvis
    rot_aug_mat = np.array([[np.cos(np.deg2rad(-rot)), -np.sin(np.deg2rad(-rot)), 0],
                            [np.sin(np.deg2rad(-rot)), np.cos(np.deg2rad(-rot)), 0],
                            [0, 0, 1]], dtype=np.float32)
    j3d = np.dot(rot_aug_mat, j3d.transpose(1, 0)).transpose(1, 0).astype(np.float32)

    inputs = {'img': torch.zeros(1), 'joints': xy, 'joints_mask': mask}
    if cfg.DATASET.lift_cam_bbox:
        inputs['cam_bbox'] = cam_bbox_feat(bbox, cam_param, img_shape, do_flip, rot)
    targets = {'orig_joint_img': np.zeros((17, 3), dtype=np.float32), 'orig_joint_cam': j3d}
    meta = {'orig_joint_valid': np.ones((17, 1), dtype=np.float32),
            'orig_joint_trunc': np.ones((17, 1), dtype=np.float32),
            'is_3D': float(True)}
    return inputs, targets, meta


def load_halpe_into_datalist(ds, name, datalist, param_key, data_root):
    """
    Che do lifter: doc <data_root>/<name>/<name>_train_halpe26.npz, gan 'halpe_h36m' cho tung nguoi,
    CHI giu nguoi co ca NeuralAnnot (datalist[i][param_key] khac None) va Halpe.
    """
    import os.path as osp
    from utils.h36m_adapter import halpe26_to_human36m
    path = osp.join(data_root, name, f'{name}_train_halpe26.npz')
    if not osp.isfile(path):
        raise FileNotFoundError(f'{path} khong ton tai. Chay tools/extract_halpe26_coco.py --dataset {name.lower()}')
    z = np.load(path)
    index = {int(a): i for i, a in enumerate(z['ann_id'])}
    kpts = z['kpts']
    out, n_noparam, n_nohalpe = [], 0, 0
    for d in datalist:
        if d.get(param_key) is None:
            n_noparam += 1
            continue
        hi = index.get(int(d['ann_id']))
        if hi is None:
            n_nohalpe += 1
            continue
        d['halpe_h36m'] = halpe26_to_human36m(kpts[hi])
        out.append(d)
    print(f'[{name} lift_only] Load Halpe-26: {path} | giu {len(out)} nguoi '
          f'(bo {n_noparam} khong co NeuralAnnot, {n_nohalpe} khong co Halpe)')
    return out
