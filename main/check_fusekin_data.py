"""
main/check_fusekin_data.py
Kiem tra mau train FuseKin thong nhat (DATASET.fusekin_unified) TRUOC khi train Teacher / Student tren
Human36M / MuCo / MSCOCO / CrowdPose. KHONG train, chi doc vai mau moi bo (co anh, co augmentation).

Kiem tra moi bo:
  1. Bo key inputs / targets / meta giong nhau giua cac bo (tron chung duoc) + shape
  2. Don vi: GT 3D met (do dai xuong TB ~0.2-0.3 m), root = 0; mesh khop voi GT 3D (H36M regressor x mesh)
  3. 2D: lech giua GT 2D (chieu tu mesh) va 2D Halpe dau vao (pixel anh crop 256) -> lon bat thuong = sai lat / chieu
  4. Ghi anh <out_dir>/check_<bo>.jpg: crop + GT 2D (xanh la) + Halpe (do), vai mau -> nhin bang mat

Chay:
  python main/check_fusekin_data.py --cfg ./config/train_teacher_k16_v2_jotr.yml [--n 64] [--vis 6]
"""
import os, sys
sys.path.append('./lib')
sys.path.append('./')
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--cfg', type=str, required=True)
parser.add_argument('--n', type=int, default=64, help='so mau kiem tra moi bo')
parser.add_argument('--vis', type=int, default=6, help='so mau ve moi bo')
parser.add_argument('--out_dir', type=str, default='experiment/_check_fusekin_data')
args = parser.parse_args()

import cv2
import numpy as np
from core.config import cfg, update_config
update_config(args.cfg)

import __init_path  # noqa: F401
from utils.jotr_dataset import get_train_dataset

BONES = ((0, 1), (1, 2), (2, 3), (0, 4), (4, 5), (5, 6), (0, 7), (7, 8), (8, 9), (9, 10),
         (8, 11), (11, 12), (12, 13), (8, 14), (14, 15), (15, 16))


def keyset(inp, tgt, meta):
    return (tuple(sorted(inp)), tuple(sorted(tgt)), tuple(sorted(meta)))


def to_px(xy_hm):
    """khung output_hm_shape -> pixel anh crop input_img_shape."""
    return np.stack([xy_hm[:, 0] / cfg.output_hm_shape[2] * cfg.input_img_shape[1],
                     xy_hm[:, 1] / cfg.output_hm_shape[1] * cfg.input_img_shape[0]], 1)


def draw(img_t, gt_px, in_px, in_mask):
    img = (img_t.numpy().transpose(1, 2, 0) * 255).clip(0, 255).astype(np.uint8)[:, :, ::-1].copy()
    for pts, col, valid in ((gt_px, (0, 255, 0), np.ones(17, bool)), (in_px, (0, 0, 255), in_mask > 0)):
        for a, b in BONES:
            if valid[a] and valid[b]:
                cv2.line(img, tuple(int(v) for v in pts[a]), tuple(int(v) for v in pts[b]), col, 1)
        for j in range(17):
            if valid[j]:
                cv2.circle(img, tuple(int(v) for v in pts[j]), 2, col, -1)
    return img


def main():
    assert cfg.DATASET.fusekin_unified, 'yml can DATASET.fusekin_unified: True'
    os.makedirs(args.out_dir, exist_ok=True)
    rng = np.random.default_rng(0)
    ref_keys, all_ok = None, True
    for name in cfg.DATASET.train_list:
        print('\n' + '=' * 70 + f'\n[{name}]')
        ds = get_train_dataset(name, None)
        idxs = rng.choice(len(ds), min(args.n, len(ds)), replace=False)
        bone, root_abs, mesh_gap, err2d, zmax, vis = [], [], [], [], [], []
        for k, i in enumerate(idxs):
            inp, tgt, meta = ds[int(i)]
            ks = keyset(inp, tgt, meta)
            if ref_keys is None:
                ref_keys = ks
                print(f'  key: inputs={ks[0]}\n       targets={ks[1]}\n       meta={ks[2]}')
            elif ks != ref_keys:
                all_ok = False
                print(f'  [LOI] bo key khac bo dau tien: {ks}')
                break
            j3 = np.asarray(tgt['orig_joint_cam'])
            mesh = np.asarray(tgt['smpl_mesh_cam'])
            reg = np.asarray(ds.h36m_joint_regressor)
            bone.append(np.mean([np.linalg.norm(j3[a] - j3[b]) for a, b in BONES]))
            root_abs.append(np.abs(j3[0]).max())
            zmax.append(np.abs(j3[:, 2]).max())
            jm = np.dot(reg, mesh)
            mesh_gap.append(np.linalg.norm((jm - jm[0:1]) - j3, axis=-1).mean())
            # 2D: dau vao (wrapper da chuan hoa [-1, 1]) va GT (khung hm)
            jin = np.asarray(inp['joints'])
            m = np.asarray(inp['joints_mask']).reshape(-1)
            in_px = np.stack([(jin[:, 0] + 1) / 2 * cfg.input_img_shape[1], (jin[:, 1] + 1) / 2 * cfg.input_img_shape[0]], 1)
            gt_px = to_px(np.asarray(tgt['orig_joint_img'])[:, :2])
            if (m > 0).any():
                err2d.append(np.median(np.linalg.norm(gt_px - in_px, axis=-1)[m > 0]))
            if k < args.vis:
                vis.append(draw(inp['img'], gt_px, in_px, m))
        if not bone:
            continue
        f = lambda v: f'{np.median(v):.3f} (max {np.max(v):.3f})'
        print(f'  n kiem tra = {len(bone)} / {len(ds)}')
        print(f'  do dai xuong TB (m)            : {f(bone)}   [ky vong ~0.2 - 0.3]')
        print(f'  |root| (m)                     : {f(root_abs)}   [ky vong 0]')
        print(f'  |z| lon nhat (m)               : {f(zmax)}   [ky vong < ~1]')
        print(f'  lech mesh -> GT 3D (m)         : {f(mesh_gap)}   [ky vong ~0]')
        print(f'  lech GT 2D vs Halpe (px, 256)  : {f(err2d)}   [ky vong vai px; lon -> sai lat / chieu]')
        ok = (0.15 < np.median(bone) < 0.4) and np.max(root_abs) < 1e-4 and np.median(mesh_gap) < 1e-3 \
            and np.median(err2d) < 25
        all_ok &= ok
        print('  => ' + ('OK' if ok else '[CAN XEM LAI] (xem so lieu + anh)'))
        if vis:
            fp = os.path.join(args.out_dir, f'check_{name}.jpg')
            cv2.imwrite(fp, np.concatenate(vis, axis=1))
            print(f'  anh: {fp}  (xanh la = GT 2D chieu tu mesh, do = Halpe dau vao)')
    print('\n' + ('TAT CA OK' if all_ok else 'CO BO CAN XEM LAI'))


if __name__ == '__main__':
    main()
