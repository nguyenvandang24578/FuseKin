"""
Thong tin vi tri bbox trong anh goc so voi camera (y tuong CLIFF, Li et al. ECCV 2022).
Lifter chi thay 2D da chuan hoa theo crop -> mat thong tin nguoi nam o dau so voi truc quang hoc camera
-> khong suy ra duoc goc nghieng nguoi trong he toa do camera (camera cam tay chuc len/xuong o 3DPW test).
Dac trung 3 so: [(cx - px) / f, (cy - py) / f, kich thuoc bbox / f].
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
