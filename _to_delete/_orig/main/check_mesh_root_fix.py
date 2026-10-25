"""
check_mesh_root_fix.py -- Kiem tra sua loi goc mesh trong loss (code cu: pred tru pelvis SMPL; code moi: _mesh_for_loss), dung mesh GT lam "pred hoan hao".
  target (GT trong loss)  = mesh GT - pelvis H36M          (nhu fusekin_sample.py / PW3D train)
  1) TRUOC sua: pred = mesh GT - pelvis SMPL (nhu get_coord, dua thang vao loss) -> so voi target
  2) SAU  sua: cung pred do  -> _mesh_for_loss (code hien tai, = mesh GT - pelvis H36M) -> so voi target
Dung DUNG ham _mesh_for_loss trong lib/core/base.py (truyen mesh tuyet doi nhu smpl_mesh_cam_render cua model).
Ky vong: (1) ~19 mm L1/toa do, ~38 mm/dinh; (2) ~0 (chi sai so float).
Chay: python main/check_mesh_root_fix.py --cfg config/train_student_kd_e1_v2_diff_jotr_l2last.yml --set 3dpw --max_samples 2000
"""
import os, sys
sys.path.append('./lib'); sys.path.append('./')
import argparse
import numpy as np
import torch
from torch.utils.data import DataLoader, Subset

ap = argparse.ArgumentParser()
ap.add_argument('--cfg', required=True)
ap.add_argument('--set', default='3dpw')
ap.add_argument('--max_samples', type=int, default=2000)
ap.add_argument('--workers', type=int, default=4)
args = ap.parse_args()
from core.config import cfg, update_config
update_config(args.cfg)
import __init_path  # noqa
from utils.jotr_dataset import get_test_dataset
from core.base import _mesh_for_loss

ds = get_test_dataset(args.set, None)
Rh = torch.as_tensor(ds.h36m_joint_regressor, dtype=torch.float32)   # = self.J_regressor cua trainer (human36)
Rs = torch.as_tensor(ds.joint_regressor, dtype=torch.float32)        # SMPL (= joint_regressor_t cua model), hang 0 = Pelvis
n = len(ds)
sub = ds if args.max_samples <= 0 or args.max_samples >= n else \
    Subset(ds, np.sort(np.random.default_rng(0).choice(n, args.max_samples, replace=False)).tolist())
dl = DataLoader(sub, batch_size=64, shuffle=False, num_workers=args.workers)

res = {False: {'l1': [], 'v': []}, True: {'l1': [], 'v': []}}
with torch.no_grad():
    for _, targets, _ in dl:
        mesh = targets['smpl_mesh_cam'].float()                                       # (B, 6890, 3) met
        target = mesh - torch.matmul(Rh[None, 0:1], mesh)                             # GT - pelvis H36M
        pred = mesh - torch.matmul(Rs[None, 0:1], mesh)                               # GT - pelvis SMPL (nhu model)
        for flag in (False, True):
            p = _mesh_for_loss(Rh, pred, mesh) if flag else pred                     # True: code hien tai (mesh tuyet doi)
            d = p - target
            res[flag]['l1'].append(d.abs().mean((1, 2)))                              # nhu coordLoss (L1 TB)
            res[flag]['v'].append(torch.norm(d, dim=-1).mean(-1))                     # TB khoang cach dinh
print(f'[{args.set}] n={sum(x.shape[0] for x in res[False]["l1"])} | target = mesh GT - pelvis H36M | pred = mesh GT (hoan hao)')
for flag, lab in ((False, '1) TRUOC sua: pred - pelvis SMPL'), (True, '2) SAU sua : pred - pelvis H36M')):
    l1 = torch.cat(res[flag]['l1']) * 1000; v = torch.cat(res[flag]['v']) * 1000
    print(f'{lab:36s} | L1/toa do TB {l1.mean():7.3f} mm (max {l1.max():7.3f}) | sai so dinh TB {v.mean():7.3f} mm '
          f'(p95 {v.quantile(0.95):7.3f}, max {v.max():7.3f})')
