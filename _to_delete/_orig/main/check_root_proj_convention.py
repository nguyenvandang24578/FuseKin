"""
check_root_proj_convention.py -- Do do lech quy uoc khop giua GT va prediction trong loss (CHI DO, khong train).
 (1) Loss mesh: GT mesh tru pelvis H36M (J_regressor_h36m), pred mesh tru pelvis SMPL (th_J_regressor) -> lech tinh tien d.
     In |d| (mm) va "san" cua loss mesh L1 (khi pred = GT hoan hao): mean |d_xyz| (mm).
 (2) Loss chieu 2D: GT 2D = chieu H36M-17 (h36m regressor x mesh); pred = khop SMPL-30 chon theo TEN.
     In lech 3D tung khop cung ten ||SMPL30[name] - H36M[name]|| (mm) tren mesh GT.
Chay: python main/check_root_proj_convention.py --cfg config/train_student_kd_e1_v2_diff_jotr_l2last.yml --set 3dpw --max_samples 2000
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

H36M = ('Pelvis', 'R_Hip', 'R_Knee', 'R_Ankle', 'L_Hip', 'L_Knee', 'L_Ankle', 'Torso', 'Neck', 'Nose', 'Head_top',
        'L_Shoulder', 'L_Elbow', 'L_Wrist', 'R_Shoulder', 'R_Elbow', 'R_Wrist')
ds = get_test_dataset(args.set, None)
smpl_names = ds.smpl.joints_name if hasattr(ds, 'smpl') else ds.mesh_model.joints_name
idx30 = [smpl_names.index(n) for n in H36M]
Rh = torch.as_tensor(ds.h36m_joint_regressor, dtype=torch.float32)
Rs = torch.as_tensor(ds.joint_regressor, dtype=torch.float32)       # SMPL 30 khop (giong joint_regressor_t cua model)
assert Rs.shape[0] == len(smpl_names), (Rs.shape, len(smpl_names))
n = len(ds)
sub = ds if args.max_samples <= 0 or args.max_samples >= n else \
    Subset(ds, np.sort(np.random.default_rng(0).choice(n, args.max_samples, replace=False)).tolist())
dl = DataLoader(sub, batch_size=64, shuffle=False, num_workers=args.workers)
offs, perj, swp = [], [], []
SW = {'R_Hip': 'L_Hip', 'L_Hip': 'R_Hip', 'R_Knee': 'L_Knee', 'L_Knee': 'R_Knee'}
sw_idx = [smpl_names.index(SW[n]) for n in SW]
sw_h = [H36M.index(n) for n in SW]
with torch.no_grad():
    for _, targets, _ in dl:
        mesh = targets['smpl_mesh_cam'].float()                              # GT, da tru pelvis H36M
        jh = torch.einsum('jk,bkc->bjc', Rh, mesh)
        js = torch.einsum('jk,bkc->bjc', Rs, mesh)
        offs.append((js[:, 0] - jh[:, 0]))                                   # pelvis SMPL - pelvis H36M
        perj.append(torch.norm(js[:, idx30] - jh, dim=-1))
        swp.append(torch.norm(js[:, sw_idx] - jh[:, sw_h], dim=-1))
        hw_h = torch.norm(jh[:, 1] - jh[:, 4], dim=-1); hw_s = torch.norm(js[:, smpl_names.index('R_Hip')] - js[:, smpl_names.index('L_Hip')], dim=-1)
        offs_hw = getattr(sys.modules[__name__], '_hw', []); offs_hw.append(torch.stack([hw_h, hw_s], -1)); setattr(sys.modules[__name__], '_hw', offs_hw)
offs = torch.cat(offs) * 1000; perj = torch.cat(perj) * 1000
print(f'[{args.set}] n={offs.shape[0]}')
print(f'(0) tong hang regressor (=1 -> lech khong phu thuoc tinh tien): H36M {Rh.sum(1).min():.4f}..{Rh.sum(1).max():.4f}, '
      f'SMPL {Rs.sum(1).min():.4f}..{Rs.sum(1).max():.4f}')
print(f'(0) pelvis H36M cua mesh GT tap test (z TB): {torch.einsum("jk,bkc->bjc", Rh, mesh)[:,0,2].mean().item()*1000:.0f} mm '
      f'-> tap test: mesh tuyet doi (khong tru goc); tap train unified: tru pelvis H36M')
print(f'(1) lech tinh tien d = pelvis_SMPL - pelvis_H36M: |d| TB {offs.norm(dim=-1).mean():.1f} mm '
      f'(median {offs.norm(dim=-1).median():.1f}, p95 {offs.norm(dim=-1).quantile(0.95):.1f}); '
      f'TB theo truc x/y/z = {offs.mean(0).tolist()} mm')
print(f'    -> san loss mesh L1 khi pred = GT hoan hao (mean |d_xyz|): {offs.abs().mean():.1f} mm / toa do')
print('(2) lech 3D khop cung ten SMPL30 vs H36M (mm, TB):')
print('    ' + ', '.join(f'{nm} {perj[:, j].mean():.0f}' for j, nm in enumerate(H36M)))
print(f'    TB 17 khop: {perj.mean():.1f} mm')
swp = torch.cat(swp) * 1000
print('(3) kiem tra dao trai/phai (SMPL ben kia vs H36M): ' + ', '.join(f'SMPL {SW[n]} vs H36M {n} {swp[:, i].mean():.0f}' for i, n in enumerate(SW)))
hw = torch.cat(_hw) * 1000
print(f'(3) do rong hong R_Hip-L_Hip: H36M {hw[:, 0].mean():.0f} mm, SMPL {hw[:, 1].mean():.0f} mm')
jh0 = torch.einsum('jk,bkc->bjc', Rh, mesh); js0 = torch.einsum('jk,bkc->bjc', Rs, mesh)
print('(4) lech theo truc (SMPL - H36M, mm, batch cuoi) hip/torso: ' + ', '.join(
    f"{nm} {((js0[:, smpl_names.index(nm)] - jh0[:, H36M.index(nm)]).mean(0) * 1000).round().tolist()}" for nm in ('R_Hip', 'L_Hip', 'Torso')))
