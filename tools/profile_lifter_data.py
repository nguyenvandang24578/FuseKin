"""
tools/profile_lifter_data.py -- DO thoi gian sinh 1 mau train cua LIFTER theo tung buoc, tung bo (2026-11-01).
Boc (monkeypatch) cac ham trong module dataset, KHONG sua code dataset:
  bbox   : process_bbox (update_bbox, scale ngau nhien)
  aug    : augmentation_noimg (crop / xoay / lat, khong doc anh)
  posefix: replace_joint_img (PoseFix; H36M / MuCo / COCO / CrowdPose che do gt_posefix)
  smpl   : get_smpl_coord (SMPL -> mesh; COCO / CrowdPose lay nhan 3D, MuCo neu co)
  khac   : tong __getitem__ - cac buoc tren (doi khop, nhan 3D, ...)
In ms / mau tung bo + uoc tinh thoi gian 1 epoch (samples_per_epoch, ti le tron cua YAML, so workers).
Chay (moi truong arts, 1 tien trinh, khong GPU):
  python tools/profile_lifter_data.py --cfg config/finetune_difflifter_d512_e15_jotr_pf05.yaml --n 200
"""
import os, sys
sys.path.insert(0, './lib')
sys.path.insert(0, './')
sys.path.insert(0, './main')
os.environ["OMP_NUM_THREADS"] = "1"
import argparse
import random
import time
from collections import defaultdict

import numpy as np

parser = argparse.ArgumentParser()
parser.add_argument('--cfg', required=True)
parser.add_argument('--n', type=int, default=200, help='so mau ngau nhien moi bo')
parser.add_argument('--sets', default='Human36M,MuCo,MSCOCO,CrowdPose')
parser.add_argument('--seed', type=int, default=0)
args = parser.parse_args()

from core.config import cfg, update_config
update_config(args.cfg)
import __init_path  # noqa: F401
from torchvision import transforms
import utils.lift_pseudo3d as lp3d
from utils.jotr_dataset import TRAIN_DATASETS

T = defaultdict(float)        # tong giay theo buoc (dat lai moi bo)
C = defaultdict(int)          # so lan goi


def wrap(fn, key):
    def f(*a, **k):
        t = time.perf_counter()
        try:
            return fn(*a, **k)
        finally:
            T[key] += time.perf_counter() - t
            C[key] += 1
    return f


STEPS = ('bbox', 'aug', 'posefix', 'smpl')


def patch_module(mod):
    for name, key in (('process_bbox', 'bbox'), ('augmentation_noimg', 'aug'), ('replace_joint_img', 'posefix')):
        if hasattr(mod, name):
            setattr(mod, name, wrap(getattr(mod, name), key))


patch_module(lp3d)            # lift_pseudo3d_sample dung augmentation_noimg cua module nay
rows = []
for name in [s for s in args.sets.split(',') if s]:
    t0 = time.perf_counter()
    ds = TRAIN_DATASETS[name](transforms.ToTensor(), 'train')
    t_init = time.perf_counter() - t0
    patch_module(sys.modules[type(ds).__module__])
    ds.get_smpl_coord = wrap(ds.get_smpl_coord, 'smpl')
    rng = np.random.default_rng(args.seed)
    idxs = rng.choice(len(ds), min(args.n, len(ds)), replace=False).tolist()
    T.clear(); C.clear()
    n_overlap = []
    t_all = 0.0
    for k, i in enumerate(idxs):
        np.random.seed(args.seed * 100000 + k); random.seed(args.seed * 100000 + k)
        n_overlap.append(int(ds.datalist[i].get('num_overlap', 0) or 0))
        t = time.perf_counter()
        ds[i]
        t_all += time.perf_counter() - t
    n = len(idxs)
    ms = {s: 1000 * T[s] / n for s in STEPS}
    tot = 1000 * t_all / n
    ms['khac'] = tot - sum(ms.values())
    rows.append((name, len(ds), tot, ms, float(np.mean(n_overlap)), t_init))
    print(f'[{name}] xong ({n} mau, khoi tao {t_init:.0f} s)', flush=True)

print(f'\n===== ms / mau (1 tien trinh, {args.n} mau / bo) | YAML {args.cfg} =====')
print(f'  {"bo":<10} | {"so mau":>8} | {"TONG":>7} | ' + ' | '.join(f'{s:>7}' for s in STEPS + ('khac',)) + ' | nguoi gan TB')
for name, L, tot, ms, ov, _ in rows:
    print(f'  {name:<10} | {L:>8} | {tot:>7.1f} | ' + ' | '.join(f'{ms[s]:>7.1f}' for s in STEPS + ('khac',))
          + f' | {ov:>5.2f}')

# Uoc tinh 1 epoch theo cach tron cua YAML (chi tinh phan sinh du lieu, chua tinh GPU)
dl = cfg.DIFFLIFT
names = [r[0] for r in rows]
lens = {r[0]: r[1] for r in rows}
tot_ms = {r[0]: r[2] for r in rows}
if getattr(dl, 'mix_mode', 'ratio') == 'jotr':
    g3 = [n for n in names if n in ('Human36M', 'MuCo')]
    g2 = [n for n in names if n not in ('Human36M', 'MuCo')]
    w = {n: 0.5 * lens[n] / sum(lens[m] for m in (g3 if n in g3 else g2)) for n in names}
elif list(getattr(dl, 'dataset_ratio', [])):
    r = list(dl.dataset_ratio)
    w = {n: r[k] / sum(r) for k, n in enumerate(names)}
else:
    w = {n: lens[n] / sum(lens.values()) for n in names}
spe = int(getattr(dl, 'samples_per_epoch', 0)) or sum(lens.values())
per = sum(w[n] * tot_ms[n] for n in names)
nw = max(int(cfg.DATASET.workers), 1)
print(f'\nTi le tron: ' + ', '.join(f'{n} {w[n]:.3f}' for n in names))
print(f'TB {per:.1f} ms / mau -> {spe} mau / epoch, {nw} workers: ~{spe * per / 1000 / nw / 60:.1f} phut / epoch '
      f'(chi sinh du lieu, gia su {nw} nhan CPU ranh)')
for s in STEPS + ('khac',):
    share = sum(w[n] * rows[k][3][s] for k, n in enumerate(names)) / per * 100
    print(f'  {s:<8}: {share:5.1f}% thoi gian sinh du lieu')
