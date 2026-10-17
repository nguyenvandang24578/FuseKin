"""Kiem chung 2D Halpe-26 cua Human36M / MuCo QUA CHINH DATASET FuseKin (sau crop, augment, lat anh).

Moi mau ve 1 hang (anh crop 256 sau augment):
  [GT 2D] | [Halpe-26 (dau vao)] | [PoseFix (dau vao cu, chi train)] | [GT (xanh) + Halpe (do) chong len]
Cung 1 mau, Halpe va PoseFix duoc lay voi CUNG seed -> cung phep augment (scale / xoay / lat), nen moi khac biet
la do nguon 2D. Neu lat anh / affine sai, Halpe se lech han GT (nhat la khop trai-phai) -> loi px rat lon.

In them thong ke tren --stats_num mau: lech px (trong crop 256) giua Halpe va GT theo tung khop,
ti le khop co mat (mask), va so sanh voi PoseFix (train).

Chay (tu thu muc goc repo, moi truong FuseKin; can Human36M_{split}_halpe26.npz / MuCo_train_halpe26.npz da trich):
  python tools/vis_h36m_halpe26.py --cfg ./config/train_teacher_hint_k16_v2.yml --dataset MuCo --split train --num 12 --stats_num 1000
  (MuCo: GT 2D che do co anh khong co Torso/Neck/Nose -> 3 khop do de trong, binh thuong)
  python tools/vis_h36m_halpe26.py --cfg ./config/train_teacher_hint_k16_v2.yml --split test --num 12
  python tools/vis_h36m_halpe26.py --cfg ./config/train_teacher_hint_k16_v2.yml --split train --num 12 --stats_num 1000
Ket qua: <out_dir>/{split}_XXXX.jpg + <out_dir>/{split}_stats.json
"""
import os, sys
sys.path.append('./lib')
sys.path.append('./')
import argparse
import json
import random

import cv2
import numpy as np
import torch

from core.config import cfg, update_config
from utils.jotr_dataset import get_train_dataset, get_test_dataset
from utils.h36m_adapter import HUMAN36M_JOINTS

GREEN, RED, BLUE = (0, 200, 0), (0, 0, 255), (255, 128, 0)
# Khung xuong H36M-17 (giong Human36M.h36m_skeleton)
H36M_SKELETON = ((0, 7), (7, 8), (8, 9), (9, 10), (8, 11), (11, 12), (12, 13), (8, 14), (14, 15), (15, 16),
                 (0, 1), (1, 2), (2, 3), (0, 4), (4, 5), (5, 6))
SRC_ATTR = {'Human36M': 'h36m_2d_source', 'MuCo': 'muco_2d_source'}   # thuoc tinh nguon 2D cua tung dataset


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--cfg', required=True, help='yml bat ky (chi lay DATASET.*); script tu dat h36m_2d_source=halpe26')
    p.add_argument('--dataset', default='Human36M', choices=['Human36M', 'MuCo'])
    p.add_argument('--split', default='test', choices=['train', 'test'], help='MuCo chi co train')
    p.add_argument('--num', type=int, default=12, help='so mau ve')
    p.add_argument('--stats_num', type=int, default=500, help='so mau tinh thong ke (gom ca mau ve)')
    p.add_argument('--seed', type=int, default=0)
    p.add_argument('--out_dir', default='debug_vis_h36m_halpe26')
    return p.parse_args()


def seed_all(s):
    random.seed(s)
    np.random.seed(s)
    torch.manual_seed(s)


def get_sample(ds, idx, source):
    """Lay mau idx voi nguon 2D cho truoc, seed co dinh theo idx -> cung augment giua cac nguon."""
    base = ds.dataset
    attr = SRC_ATTR[type(base).__name__]
    old = getattr(base, attr)
    setattr(base, attr, source)
    seed_all((100003 * idx + 7) % (2 ** 32 - 1))   # numpy chi nhan seed < 2^32
    try:
        return ds[idx]
    finally:
        setattr(base, attr, old)


def to_px_input(joints):
    """inputs['joints'] da chuan hoa [-1, 1] (wrapper) -> pixel crop."""
    xy = np.asarray(joints, dtype=np.float32)[:, :2].copy()
    xy[:, 0] = (xy[:, 0] + 1) / 2 * cfg.input_img_shape[1]
    xy[:, 1] = (xy[:, 1] + 1) / 2 * cfg.input_img_shape[0]
    return xy


def to_px_hm(joint_img):
    """khung output_hm_shape -> pixel crop."""
    xy = np.asarray(joint_img, dtype=np.float32)[:, :2].copy()
    xy[:, 0] = xy[:, 0] / cfg.output_hm_shape[2] * cfg.input_img_shape[1]
    xy[:, 1] = xy[:, 1] / cfg.output_hm_shape[1] * cfg.input_img_shape[0]
    return xy


def to_bgr(img_t):
    img = img_t.numpy().transpose(1, 2, 0) if torch.is_tensor(img_t) else np.asarray(img_t)
    img = np.clip(img * 255, 0, 255).astype(np.uint8)
    return np.ascontiguousarray(img[:, :, ::-1])


def draw_pose(img, xy, valid, skeleton, color, label=True):
    for a, b in skeleton:
        if valid[a] and valid[b]:
            cv2.line(img, (int(xy[a, 0]), int(xy[a, 1])), (int(xy[b, 0]), int(xy[b, 1])), color, 1, cv2.LINE_AA)
    for j in range(len(xy)):
        if not valid[j]:
            continue
        cv2.circle(img, (int(xy[j, 0]), int(xy[j, 1])), 3, color, -1, cv2.LINE_AA)
        if label:
            cv2.putText(img, str(j), (int(xy[j, 0]) + 3, int(xy[j, 1]) - 3), cv2.FONT_HERSHEY_SIMPLEX, 0.3, color, 1)
    return img


def title(img, text):
    cv2.putText(img, text, (4, 14), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (0, 0, 0), 3, cv2.LINE_AA)
    cv2.putText(img, text, (4, 14), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (255, 255, 255), 1, cv2.LINE_AA)
    return img


def main():
    args = parse_args()
    update_config(args.cfg)
    if args.dataset == 'MuCo':
        args.split = 'train'                  # MuCo chi co split train
        cfg.DATASET.muco_2d_source = 'halpe26'
    else:
        cfg.DATASET.h36m_2d_source = 'halpe26'   # nap file npz; tung mau doi nguon qua thuoc tinh cua dataset
    cfg.DATASET.lift_only = False             # can anh that de ve
    cfg.DATASET.workers = 0
    os.makedirs(args.out_dir, exist_ok=True)

    ds = get_train_dataset(args.dataset, None) if args.split == 'train' else get_test_dataset(args.dataset, None)
    base = ds.dataset
    skeleton = H36M_SKELETON
    is_train = args.split == 'train'
    tag = f'{args.dataset}_{args.split}'
    print(f'[data] {args.dataset} {args.split}: {len(ds)} mau | conf_thr = {base.halpe26_conf_thr}')

    rng = np.random.RandomState(args.seed)
    n_stats = min(max(args.stats_num, args.num), len(ds))
    idxs = rng.choice(len(ds), n_stats, replace=False)

    err = {'halpe': [], 'posefix': []}      # moi phan tu (17,) px, nan = khong tinh
    cover = {'halpe': [], 'posefix': []}
    for i, idx in enumerate(idxs):
        inp_h, tgt, meta = get_sample(ds, int(idx), 'halpe26')
        gt = to_px_hm(tgt['orig_joint_img'])
        gt_ok = (np.asarray(meta['orig_joint_valid']).reshape(-1) > 0) & (np.asarray(meta['orig_joint_trunc']).reshape(-1) > 0)
        rows = {'halpe': inp_h}
        if is_train:
            rows['posefix'], _, _ = get_sample(ds, int(idx), 'gt_posefix')

        px = {}
        for k, inp in rows.items():
            xy = to_px_input(inp['joints'])
            m = np.asarray(inp['joints_mask']).reshape(-1) > 0
            ok = m & gt_ok
            e = np.full(17, np.nan, dtype=np.float32)
            e[ok] = np.linalg.norm(xy[ok] - gt[ok], axis=1)
            err[k].append(e)
            cover[k].append(m.astype(np.float32))
            px[k] = (xy, m, np.nanmean(e) if ok.any() else float('nan'))

        if i < args.num:
            img = to_bgr(inp_h['img'])
            panels = [title(draw_pose(img.copy(), gt, gt_ok, skeleton, GREEN), 'GT 2D')]
            xy, m, e = px['halpe']
            panels.append(title(draw_pose(img.copy(), xy, m, skeleton, RED), f'Halpe-26 | lech {e:.1f}px'))
            if is_train:
                xyp, mp, ep = px['posefix']
                panels.append(title(draw_pose(img.copy(), xyp, mp, skeleton, BLUE), f'PoseFix | lech {ep:.1f}px'))
            ov = draw_pose(img.copy(), gt, gt_ok, skeleton, GREEN, label=False)
            ov = draw_pose(ov, xy, m, skeleton, RED, label=False)
            panels.append(title(ov, 'GT (xanh) + Halpe (do)'))
            path = os.path.join(args.out_dir, f'{tag}_{i:04d}_idx{int(idx)}.jpg')
            cv2.imwrite(path, np.concatenate(panels, axis=1))

    # ---------------- Thong ke ----------------
    print('\n' + '=' * 72)
    print(f'{args.dataset} {args.split} | {n_stats} mau | lech 2D so voi GT (px trong crop {cfg.input_img_shape[1]})')
    print('=' * 72)
    keys = ['halpe'] + (['posefix'] if is_train else [])
    E = {k: np.stack(err[k]) for k in keys}
    C = {k: np.stack(cover[k]) for k in keys}
    hdr = f'{"khop":<12}' + ''.join(f'{k + " (px)":>14}{k + " co mat":>14}' for k in keys)
    print(hdr)
    for j, jn in enumerate(HUMAN36M_JOINTS):
        line = f'{jn:<12}'
        for k in keys:
            line += f'{np.nanmean(E[k][:, j]):>14.1f}{C[k][:, j].mean() * 100:>13.1f}%'
        print(line)
    stats = {'dataset': args.dataset, 'split': args.split, 'n': int(n_stats), 'conf_thr': float(base.halpe26_conf_thr)}
    for k in keys:
        per_sample = np.nanmean(E[k], axis=1)
        stats[k] = {
            'mean_px': float(np.nanmean(E[k])),
            'median_sample_px': float(np.nanmedian(per_sample)),
            'p90_sample_px': float(np.nanpercentile(per_sample, 90)),
            'frac_sample_gt15px': float(np.nanmean(per_sample > 15)),
            'coverage': float(C[k].mean()),
            'per_joint_px': np.nanmean(E[k], axis=0).tolist(),
        }
        print(f'\n{k}: TB {stats[k]["mean_px"]:.2f}px | trung vi/mau {stats[k]["median_sample_px"]:.2f}px | '
              f'p90/mau {stats[k]["p90_sample_px"]:.2f}px | mau lech TB > 15px: {stats[k]["frac_sample_gt15px"] * 100:.1f}% | '
              f'khop co mat {stats[k]["coverage"] * 100:.1f}%')
    print('\nCach doc:')
    print('  * Halpe dung: lech TB vai px (Head_top / Torso / hong lech hon do khac dinh nghia khop, khong phai loi).')
    print('  * Nghi loi lat / affine: lech TB > 15-20px o NHIEU mau, hoac khop trai-phai lech ~ be rong vai/hong.')
    if is_train:
        print('  * PoseFix la nhieu tong hop (co ca nhay sang khop khac) -> thuong lech hon Halpe tren H36M.')
    path = os.path.join(args.out_dir, f'{tag}_stats.json')
    with open(path, 'w') as f:
        json.dump(stats, f, indent=2)
    print(f'\nDa luu {min(args.num, n_stats)} anh + {path}')


if __name__ == '__main__':
    main()
