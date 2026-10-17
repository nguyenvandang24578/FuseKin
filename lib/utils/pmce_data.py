"""Du lieu cho finetune / danh gia PMCE (GraphormerNet) tren dataset FuseKin, dung DAC TRUNG ANH TINH SAN.

Y tuong (giong PMCE goc: dac trung anh tinh san, khong augment):
  * Moi tap duoc dung o che do lift_only (khong doc anh) va KHONG augment -> moi chi so i luon cho cung 1 crop.
  * tools/extract_img_feat.py doc anh, warpAffine bang chinh img2bb_trans cua mau i (= crop 256 ma dataset tao),
    chay SPIN ResNet-50 -> luu feat[i] (2048) + hw[i] (kich thuoc anh goc) vao <feat_dir>/<ten>_spin256.npz.
  * PMCESet doc lai mau i: 2D dau vao (Halpe, khung crop [-1, 1]) -> nghich dao affine -> pixel ANH GOC ->
    chuan hoa kieu PMCE X/w*2 - [1, h/w] (KHONG tru root, giong PMCE/Human36M/dataset.py). GT 3D: met, tru root.

Ten tap ho tro (FEAT_SETS):
  h36m-train  : Human36M S1/5/6/7/8 (lift_only, TAT augment)    muco-train : MuCo (lift_only, TAT augment)
  3dpw-train  : 3dpw-train-noaug                                 3dpw-val / 3dpw-test / h36m-test : tap danh gia
"""
import numpy as np
import torch
from torch.utils.data import Dataset

from core.config import cfg

FEAT_SETS = ('h36m-train', 'muco-train', '3dpw-train', '3dpw-val', '3dpw-test', 'h36m-test')


def _patch_noaug(module):
    """Thay augmentation_noimg trong module dataset bang ban luon dung split 'test' (scale 1, xoay 0, khong lat)."""
    from utils import preprocessing as pp
    module.augmentation_noimg = lambda bbox, split, shape, exclude_flip=False: \
        pp.augmentation_noimg(bbox, 'test', shape, exclude_flip)


def build_dataset(name):
    """Dataset FuseKin (wrapper Human36M17Dataset) o che do lift_only, khong augment, giu img_path + img2bb_trans."""
    from utils.jotr_dataset import get_train_dataset, get_test_dataset
    assert name in FEAT_SETS, f'{name} khong thuoc {FEAT_SETS}'
    cfg.DATASET.lift_only = True
    if name == 'h36m-train':
        import data_final.Human36M.dataset as m
        _patch_noaug(m)
        ds = get_train_dataset('Human36M', None)
    elif name == 'muco-train':
        import data_final.MuCo.dataset as m
        _patch_noaug(m)
        ds = get_train_dataset('MuCo', None)
    else:
        ds = get_test_dataset({'3dpw-train': '3dpw-train-noaug', '3dpw-val': '3dpw-val',
                               '3dpw-test': '3dpw', 'h36m-test': 'Human36M'}[name], None)
    ds.keep_vis_info = True
    return ds


def feat_path(feat_dir, name):
    return f'{feat_dir}/{name}_spin256.npz'


def crop_norm_to_raw(xy_norm, img2bb):
    """2D khung crop chuan hoa [-1, 1] (J, 2) -> pixel ANH GOC, dung nghich dao affine img2bb (2, 3)."""
    W, H = cfg.input_img_shape[1], cfg.input_img_shape[0]
    px = np.empty_like(xy_norm, dtype=np.float64)
    px[:, 0] = (xy_norm[:, 0] + 1) / 2 * W
    px[:, 1] = (xy_norm[:, 1] + 1) / 2 * H
    A = np.concatenate([np.asarray(img2bb, np.float64), [[0, 0, 1]]], axis=0)
    Ainv = np.linalg.inv(A)[:2]
    return (np.concatenate([px, np.ones_like(px[:, :1])], axis=1) @ Ainv.T).astype(np.float32)


class PMCESet(Dataset):
    """Tra ve: x (17, 2) 2D chuan hoa kieu PMCE, mask (17, 1), feat (2048,), gt3d (17, 3) met root-rel, valid (17, 1)."""

    def __init__(self, name, feat_dir):
        self.name = name
        self.ds = build_dataset(name)
        z = np.load(feat_path(feat_dir, name))
        self.feat, self.hw, self.ok = z['feat'], z['hw'], z['ok']
        if len(self.feat) != len(self.ds):
            raise ValueError(f'{feat_path(feat_dir, name)}: {len(self.feat)} dac trung != {len(self.ds)} mau dataset '
                             f'(cau hinh 2D source / dataset da doi? Trich lai bang tools/extract_img_feat.py)')
        if 'check_idx' in z:   # doi chieu thu tu mau giua luc trich dac trung va bay gio
            for ci, cp in zip(z['check_idx'], z['check_path']):
                now = self.ds[int(ci)][2]['img_path']
                if str(now) != str(cp):
                    raise ValueError(f'{name}: mau {int(ci)} lech thu tu ({now} != {cp}) -> trich lai dac trung')
        self.reg = np.asarray(self.ds.dataset.h36m_joint_regressor, dtype=np.float32)
        print(f'[PMCESet] {name}: {len(self.ds)} mau | dac trung loi (anh khong doc duoc): {int((self.ok == 0).sum())}')

    def __len__(self):
        return len(self.ds)

    def __getitem__(self, i):
        inp, tgt, meta = self.ds[i]
        ok = float(self.ok[i])
        mask = np.asarray(inp['joints_mask'], np.float32).reshape(-1, 1) * ok
        raw = crop_norm_to_raw(np.asarray(inp['joints'], np.float32)[:, :2], meta['img2bb_trans'])
        h, w = float(self.hw[i][0]), float(self.hw[i][1])
        x = raw / max(w, 1.0) * 2.0 - np.array([1.0, h / max(w, 1.0)], np.float32)
        x = (x * (mask > 0)).astype(np.float32)

        if 'orig_joint_cam' in tgt:
            gt = np.asarray(tgt['orig_joint_cam'], np.float32)
        else:
            gt = self.reg @ np.asarray(tgt['smpl_mesh_cam'], np.float32)
        gt = gt - gt[:1]
        valid = np.asarray(meta['orig_joint_valid'], np.float32).reshape(-1, 1) if 'orig_joint_valid' in meta \
            else np.ones((len(gt), 1), np.float32)
        valid = valid * ok
        return (torch.from_numpy(x), torch.from_numpy(mask), torch.from_numpy(self.feat[i].astype(np.float32)),
                torch.from_numpy(gt.astype(np.float32)), torch.from_numpy(valid))
