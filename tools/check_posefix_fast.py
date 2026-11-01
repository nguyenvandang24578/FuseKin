"""
tools/check_posefix_fast.py -- KIEM TRA synthesize_pose_fast GIONG TUNG BIT synthesize_pose (ban goc JOTR) + do toc do.
Thu cac lan goi PoseFix THAT tu 4 bo train (YAML lifter kieu JOTR), roi voi moi lan goi: dat cung seed (np.random +
random), chay ban goc va ban nhanh, so output bang np.array_equal. Lap lai voi posefix_scale 1.0 va 0.5.
Chay (moi truong arts):
  python tools/check_posefix_fast.py --cfg config/finetune_difflifter_d512_e15_jotr_pf05.yaml --n 300
"""
import os, sys
sys.path.insert(0, './lib')
sys.path.insert(0, './')
sys.path.insert(0, './main')
os.environ["OMP_NUM_THREADS"] = "1"
import argparse
import copy
import random
import time

import numpy as np

parser = argparse.ArgumentParser()
parser.add_argument('--cfg', required=True)
parser.add_argument('--n', type=int, default=300, help='so mau / bo de thu lan goi PoseFix')
parser.add_argument('--sets', default='Human36M,MuCo,MSCOCO,CrowdPose')
args = parser.parse_args()

from core.config import cfg, update_config
update_config(args.cfg)
cfg.DATASET.posefix_fast = False            # thu bang ban goc
import __init_path  # noqa: F401
from torchvision import transforms
import utils.posefix as pf
from utils.jotr_dataset import TRAIN_DATASETS

calls = {}
orig = pf.synthesize_pose


def rec(name):
    def f(joints, near_joints, area, num_overlap):
        calls[name].append((joints.copy(), near_joints.copy(), float(area), int(num_overlap)))
        return orig(joints, near_joints, area, num_overlap)
    return f


for name in [s for s in args.sets.split(',') if s]:
    calls[name] = []
    pf.synthesize_pose = rec(name)
    ds = TRAIN_DATASETS[name](transforms.ToTensor(), 'train')
    idxs = np.random.default_rng(0).choice(len(ds), min(args.n, len(ds)), replace=False)
    for k, i in enumerate(idxs):
        np.random.seed(k); random.seed(k)
        ds[int(i)]
    print(f'[{name}] thu {len(calls[name])} lan goi PoseFix', flush=True)
pf.synthesize_pose = orig

ok_all = True
for scale in (1.0, 0.5):
    cfg.DATASET.posefix_scale = scale
    print(f'\n===== posefix_scale = {scale} =====')
    for name, lst in calls.items():
        bad, t_o, t_f = 0, 0.0, 0.0
        for k, (j, nj, area, no) in enumerate(lst):
            np.random.seed(1000 + k); random.seed(1000 + k)
            t = time.perf_counter(); a = orig(j.copy(), nj, area, no); t_o += time.perf_counter() - t
            np.random.seed(1000 + k); random.seed(1000 + k)
            t = time.perf_counter(); b = pf.synthesize_pose_fast(j.copy(), nj, area, no); t_f += time.perf_counter() - t
            # trang thai RNG sau khi goi cung phai trung (cung so lan rut)
            if not np.array_equal(a, b):
                bad += 1
        # kiem tra them: RNG con lai giong nhau (1 lan goi cuoi)
        n = max(len(lst), 1)
        ok = bad == 0
        ok_all &= ok
        print(f'  [{"OK " if ok else "LECH"}] {name:<10} | {len(lst)} lan goi | lech {bad} | goc {1000 * t_o / n:.2f} ms'
              f' -> nhanh {1000 * t_f / n:.2f} ms (x{t_o / max(t_f, 1e-9):.2f})')
# RNG: sau 1 lan goi, so ngau nhien tiep theo phai trung
j, nj, area, no = next(iter(calls.values()))[0]
np.random.seed(7); random.seed(7); orig(j.copy(), nj, area, no); ra = (np.random.rand(), random.random())
np.random.seed(7); random.seed(7); pf.synthesize_pose_fast(j.copy(), nj, area, no); rb = (np.random.rand(), random.random())
ok_all &= ra == rb
print(f'\nTrang thai RNG sau goi: {"TRUNG" if ra == rb else "KHAC"}')
print('KET LUAN: ' + ('GIONG TUNG BIT -> bat DATASET.posefix_fast: True duoc' if ok_all else 'CON LECH -> KHONG dung ban nhanh'))
