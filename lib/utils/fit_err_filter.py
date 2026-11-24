"""
lib/utils/fit_err_filter.py -- Loc nguoi MSCOCO / CrowdPose co fit NeuralAnnot lech nhan 2D tay (kieu ARTS / I2L-MeshNet).
cfg.DATASET.fit_err_thr > 0 (mac dinh 0 = TAT): bo khoi datalist nguoi co sai so fit > nguong (px luoi output_hm_shape 64).
Sai so doc tu file tinh san boi tools/fit_err_stats.py: <fit_err_dir>/fit_err_<MSCOCO|CrowdPose>.npz (ann_id, err),
KHONG tinh lai luc train (COCO mat ~5.5h CPU). Nguoi khong co trong file (hoac err = nan) -> GIU, in so luong.
"""
import os.path as osp

import numpy as np

from core.config import cfg


def filter_by_fit_err(name, datalist):
    thr = float(cfg.DATASET.get('fit_err_thr', 0.0))
    if thr <= 0:
        return datalist
    d = str(cfg.DATASET.get('fit_err_dir', 'experiment/_fit_err_stats'))
    path = osp.join(d if osp.isabs(d) else osp.join(cfg.root_dir, d), f'fit_err_{name}.npz')
    if not osp.isfile(path):
        raise FileNotFoundError(f'[fit_err_thr] khong thay {path}. Chay tools/fit_err_stats.py --sets {name} truoc.')
    z = np.load(path)
    ids, err = z['ann_id'].astype(np.int64), z['err'].astype(np.float64)
    ok = (ids >= 0) & ~np.isnan(err)
    m = dict(zip(ids[ok].tolist(), err[ok].tolist()))
    out, n_drop, n_miss = [], 0, 0
    for p in datalist:
        e = m.get(int(p.get('ann_id', -1)))
        if e is None:
            n_miss += 1                                  # khong co sai so -> giu nguyen
            out.append(p)
        elif e > thr:
            n_drop += 1                                  # fit lech nhan tay qua nguong -> bo
        else:
            out.append(p)
    print(f'[{name} fit_err_thr = {thr:g} px] giu {len(out)} / {len(datalist)} nguoi '
          f'(bo {n_drop}, khong co trong file {n_miss}) | {path}')
    return out
