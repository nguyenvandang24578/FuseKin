"""
tools/fit_h36m_to_smpl29.py -- (2026-11-08) Khop anh xa tuyen tinh M (29 x 17): H36M-17 -> 29 diem HybrIK
(24 khop SMPL + 5 diem dau chi), dung cho IK (main/ik_upper_bound.py, Student v6 sau nay).

Du lieu: mesh GT tu tham so SMPL (pose_param, shape_param) cua tap TRAIN Student (mac dinh Human36M, MuCo),
KHONG dung 3DPW. Bo mau is_valid_fit = 0.
  H36M-17  = h36m_regressor x mesh            (cung dinh nghia muc tieu cua lifter)
  SMPL-29  = 24 khop dong hoc smplpytorch (jtr) + 5 dinh dau chi
  Ca hai tru goc Pelvis H36M. Giai binh phuong toi thieu (ridge nho), cung he so cho x, y, z:
      SMPL29[j] = sum_k M[j, k] * H36M17[k]
In loi khop lai tung diem (mm) tren 10% mau giu lai (val), luu M (.npy).

Chay (cfg = yml Student de co duong dan du lieu):
  python tools/fit_h36m_to_smpl29.py --cfg config/train_student_v5_bonequery.yml \
      --sets Human36M,MuCo --max_samples 20000 --out data_final/h36m17_to_smpl29.npy --gpu 0
"""
import os, sys
sys.path.append('./lib')
sys.path.append('./')
os.environ["OMP_NUM_THREADS"] = "1"
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--cfg', type=str, required=True)
parser.add_argument('--sets', type=str, default='Human36M,MuCo')
parser.add_argument('--max_samples', type=int, default=20000, help='so mau toi da moi bo')
parser.add_argument('--ridge', type=float, default=1e-6)
parser.add_argument('--out', type=str, default='data_final/h36m17_to_smpl29.npy')
parser.add_argument('--batch_size', type=int, default=64)
parser.add_argument('--seed', type=int, default=0)
parser.add_argument('--gpu', type=str, default='0')
args = parser.parse_args()
os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu

import copy
import random
import numpy as np
import torch
from torch.utils.data import DataLoader, Subset

from core.config import cfg, update_config
update_config(args.cfg)

import __init_path  # noqa: F401
from utils.jotr_dataset import get_train_dataset
from models.hybrik_smpl import HybrIKSMPL, LEAF_VERTS

NAMES29 = ['pelvis', 'l_hip', 'r_hip', 'spine1', 'l_knee', 'r_knee', 'spine2', 'l_ankle', 'r_ankle', 'spine3',
           'l_foot', 'r_foot', 'neck', 'l_collar', 'r_collar', 'head', 'l_shoulder', 'r_shoulder', 'l_elbow',
           'r_elbow', 'l_wrist', 'r_wrist', 'l_hand', 'r_hand', 'head_top', 'l_middle', 'r_middle',
           'l_bigtoe', 'r_bigtoe']


@torch.no_grad()
def collect(name, dev):
    ds = get_train_dataset(name, None)
    n = len(ds)
    if args.max_samples and args.max_samples < n:
        ds = Subset(ds, sorted(random.Random(args.seed).sample(range(n), args.max_samples)))
    loader = DataLoader(ds, batch_size=args.batch_size, shuffle=False, num_workers=cfg.DATASET.workers)
    base = ds.dataset if isinstance(ds, Subset) else ds
    # ban sao rieng tren GPU: KHONG .to() lop cua dataset (worker DataLoader con dung lop do tren CPU)
    layer = copy.deepcopy(base.mesh_model.layer['neutral']).to(dev)
    reg = torch.as_tensor(base.joint_regressor_human36, dtype=torch.float32, device=dev)   # (17, V)
    hyb = HybrIKSMPL(layer, base.joint_regressor_human36).to(dev)   # tao som de kiem tra ten buffer SMPL
    X, Y = [], []
    for inputs, targets, meta in loader:
        assert 'pose_param' in targets and 'shape_param' in targets, \
            f'{name}: targets khong co pose_param/shape_param (keys: {list(targets.keys())})'
        keep = meta['is_valid_fit'].float().reshape(-1) > 0
        if not bool(keep.any()):
            continue
        pose = targets['pose_param'].float().to(dev)[keep].reshape(-1, 72)
        shape = targets['shape_param'].float().to(dev)[keep].reshape(-1, 10)
        verts, jtr = layer(pose, shape)                     # met (smplpytorch)
        j17 = torch.einsum('jv,bvc->bjc', reg, verts)
        j29 = torch.cat([jtr[:, :24], verts[:, LEAF_VERTS]], dim=1)
        root = j17[:, 0:1]
        X.append((j17 - root).cpu().numpy())
        Y.append((j29 - root).cpu().numpy())
    X, Y = np.concatenate(X), np.concatenate(Y)
    print(f'[{name}] lay {len(X)} mau hop le / {len(ds)}')
    return X, Y, hyb


def main():
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    dev = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    Xs, Ys = [], []
    for name in [s.strip() for s in args.sets.split(',') if s.strip()]:
        X, Y, _ = collect(name, dev)
        Xs.append(X)
        Ys.append(Y)
    X, Y = np.concatenate(Xs).astype(np.float64), np.concatenate(Ys).astype(np.float64)   # (N,17,3), (N,29,3)
    N = len(X)
    perm = np.random.RandomState(args.seed).permutation(N)
    nv = max(N // 10, 1)
    va, tr = perm[:nv], perm[nv:]

    def flat(A):                                             # (n, J, 3) -> (n*3, J): cung he so cho x, y, z
        return A.transpose(0, 2, 1).reshape(-1, A.shape[1])
    A, Bm = flat(X[tr]), flat(Y[tr])
    # ridge: (A^T A + l I) M^T = A^T B
    G = A.T @ A + args.ridge * np.eye(A.shape[1])
    M = np.linalg.solve(G, A.T @ Bm).T                       # (29, 17)

    pred = np.einsum('jk,nkc->njc', M, X[va])
    err = np.linalg.norm(pred - Y[va], axis=-1) * 1000       # (nv, 29) mm
    print(f'===> M (29 x 17) | train {len(tr)} | val {nv} | loi TB {err.mean():.2f} mm')
    for j in range(29):
        print(f'   {j:2d} {NAMES29[j]:11s} {err[:, j].mean():7.2f} mm  (p90 {np.percentile(err[:, j], 90):6.2f})')
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    np.save(args.out, M.astype(np.float32))
    print(f'===> Luu {args.out}')


if __name__ == '__main__':
    main()
