"""
Thong tin vi tri bbox trong anh goc so voi camera (y tuong CLIFF, Li et al. ECCV 2022).
Lifter chi thay 2D da chuan hoa theo crop -> mat thong tin nguoi nam o dau so voi truc quang hoc camera
-> khong suy ra duoc goc nghieng nguoi trong he toa do camera (camera cam tay chuc len/xuong o 3DPW test).
Dac trung 3 so: [(cx - px) / f, (cy - py) / f, kich thuoc bbox / f].

MSCOCO / CrowdPose khong co camera that: cam_param cua NeuralAnnot la camera AO dat tai tam bbox tung nguoi
(princpt = tam bbox, focal ~ ti le kich thuoc bbox) -> dac trung tren ~ hang so, va GT 3D bi "nhin thang" vao
nguoi. Hai bo nay dung est_cam (camera uoc luong kieu CLIFF: f = duong cheo anh, princpt = tam anh) va xoay
GT 3D bang view_rot de dua ve cung quy uoc camera voi H36M / MuCo / 3DPW.
"""
import numpy as np


def cam_bbox_feat(bbox, cam_param, img_shape, do_flip=False, rot=0.0):
    """
    bbox: xywh (pixel anh goc, bbox da mo rong giong bbox dung de crop)
    cam_param: dict co 'focal' (fx, fy hoac 1 so) va 'princpt' (px, py); thieu princpt -> tam anh
    img_shape: (h, w)
    do_flip: anh bi lat ngang (GT 3D x da doi dau) -> doi dau dx
    rot: goc xoay augmentation (do), giong rot dung cho rot_aug_mat cua GT 3D
         -> xoay (dx, dy) cung ma tran (xoay camera quanh truc z lam xoay toa do chuan hoa X/Z, Y/Z y het)
    Tra ve (3,) float32.
    """
    x, y, w, h = [float(v) for v in bbox]
    cx, cy = x + w / 2.0, y + h / 2.0
    f = float(np.mean(np.asarray(cam_param['focal'], dtype=np.float64)))
    if 'princpt' in cam_param and cam_param['princpt'] is not None:
        px, py = [float(v) for v in np.asarray(cam_param['princpt']).reshape(-1)[:2]]
    else:
        px, py = (img_shape[1] - 1) / 2.0, (img_shape[0] - 1) / 2.0
    dx, dy = (cx - px) / f, (cy - py) / f
    if do_flip:
        dx = -dx
    if rot:
        a = np.deg2rad(-rot)
        dx, dy = np.cos(a) * dx - np.sin(a) * dy, np.sin(a) * dx + np.cos(a) * dy
    return np.array([dx, dy, max(w, h) / f], dtype=np.float32)


def est_cam(img_shape):
    """Camera uoc luong cho anh khong co camera (CLIFF): f = duong cheo anh, princpt = tam anh. img_shape: (h, w)."""
    h, w = float(img_shape[0]), float(img_shape[1])
    f = float(np.sqrt(h * h + w * w))
    return {'focal': (f, f), 'princpt': ((w - 1) / 2.0, (h - 1) / 2.0)}


def view_rot(dx, dy):
    """
    Ma tran xoay (3, 3) dua truc z cua camera AO "nhin thang vao nguoi" thanh tia tu camera that toi nguoi
    d = (dx, dy, 1) (dx, dy = toa do chuan hoa cua tam bbox). Xoay nho nhat (Rodrigues quanh truc z x d).
    Dung cho pseudo-GT NeuralAnnot: pose_camera_that = view_rot @ pose_camera_ao (root-relative).
    """
    d = np.array([dx, dy, 1.0], dtype=np.float64)
    d /= np.linalg.norm(d)
    v = np.array([-d[1], d[0], 0.0])        # z x d
    s, c = np.linalg.norm(v), d[2]
    if s < 1e-8:
        return np.eye(3, dtype=np.float32)
    k = np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])
    return (np.eye(3) + k + k @ k * ((1 - c) / (s * s))).astype(np.float32)
