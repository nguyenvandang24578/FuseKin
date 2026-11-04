"""
lib/utils/dense2d.py -- Nhan 2D DAY (2026-11-04): chieu mot tap CO DINH dinh mesh SMPL len anh crop.

Tap dinh: file json (cfg.DATASET.dense2d_vertex_file, vd config/dense48_fps.json) do tools/select_dense_vertices.py tao
(farthest point sampling tren mesh mau SMPL neutral). Nhan tinh ngay khi doc du lieu tu mesh GT + camera, giong cach
tinh 17 khop H36M tu mesh: khong tao bo du lieu rieng. Lat anh: mesh duoc tao tu pose SMPL da lat (doi trai / phai)
nen dinh i van dung vi tri giai phau tren nguoi trong anh lat -> khong can doi cap dinh.
"""
import json
import os.path as osp

import numpy as np

from core.config import cfg
from utils.transforms import cam2pixel

_CACHE = {}


def dense_vertex_ids():
    """Danh sach chi so dinh (np.int64) hoac None neu tat (cfg.DATASET.dense2d_vertex_file rong)."""
    path = cfg.DATASET.get('dense2d_vertex_file', '')
    if not path:
        return None
    if not osp.isabs(path):
        path = osp.join(cfg.root_dir, path)
    if path not in _CACHE:
        with open(path, 'r', encoding='utf-8') as f:
            _CACHE[path] = np.asarray(json.load(f)['vertex_ids'], dtype=np.int64)
    return _CACHE[path]


def project_dense(mesh_cam, cam_param, img2bb_trans):
    """mesh_cam (6890, 3) toa do camera (met hoac mm, chieu khong phu thuoc don vi), cam_param co focal / princpt,
    img2bb_trans (2, 3). Tra (N, 2) khung output_hm_shape va trunc (N, 1) = diem nam trong khung crop."""
    return project_points(np.asarray(mesh_cam, dtype=np.float32)[dense_vertex_ids()], cam_param, img2bb_trans)


def project_points(pts, cam_param, img2bb_trans):
    """Chieu diem 3D toa do camera (M, 3) vao crop: (M, 2) khung output_hm_shape + trunc (M, 1)."""
    pts = np.asarray(pts, dtype=np.float32)
    img = cam2pixel(pts, cam_param['focal'], cam_param['princpt'])
    xy1 = np.concatenate((img[:, :2], np.ones_like(img[:, :1])), 1)
    xy = np.dot(img2bb_trans, xy1.transpose(1, 0)).transpose(1, 0)[:, :2]
    xy[:, 0] = xy[:, 0] / cfg.input_img_shape[1] * cfg.output_hm_shape[2]
    xy[:, 1] = xy[:, 1] / cfg.input_img_shape[0] * cfg.output_hm_shape[1]
    trunc = ((xy[:, 0] >= 0) * (xy[:, 0] < cfg.output_hm_shape[2]) *
             (xy[:, 1] >= 0) * (xy[:, 1] < cfg.output_hm_shape[1])).reshape(-1, 1).astype(np.float32)
    return xy.astype(np.float32), trunc
