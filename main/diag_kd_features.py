"""
main/diag_kd_features.py -- Chan doan (2026-11-03): Student co HOC tu Teacher (KD feature) khong?
Chi danh gia. Teacher nhan dau vao GIONG luc train KD (core/base.py Student_Trainer.train): GT 3D root-relative (met,
H36M regressor x mesh GT), cung anh, cung 2D, cung cam_bbox. Moi Student nhan 2D test (OpenPose / HHRNet o YAML JOTR).
Do tren 3 nhanh KD (feat_joint_in = 'proj', feat_joint = 'joint', feat_global = 'global'), TRUC TIEP (khong projector):
  cos   : cosine Student vs Teacher (TB theo khop, roi theo mau)
  l2n   : || s/|s| - t/|t| || (L2 sau chuan hoa)
  rho   : Spearman theo mau giua (1 - cos) va MPJPE14 cua Student (> 0: mau cang giong Teacher cang it loi)
So nhieu Student (--students ten=duong_dan,...), vd KD vs khong KD (v3 scratch). LUU Y: Student train khac du lieu (Halpe)
-> khac Teacher mot phan do du lieu, khong chi do KD -> chi la DAU HIEU; ket luan chac can ablation tat KD cung du lieu.
Chay:
  python main/diag_kd_features.py --cfg ./config/train_student_v3_kd_jotr_pf05lift3dpw.yml \\
    --teacher ./experiment/teacher_v3_jotr_pf05/checkpoint/best.pth.tar \\
    --students kd_t2=./experiment/student_v3_kd_jotr_pf05lift3dpw_t2/checkpoint/best_3dpw.pth.tar,scratch=./experiment/student_v3_scratch_jotr/checkpoint/best.pth.tar \\
    --sets 3dpw,3dpw-crowd --max_samples 3000
"""
import os, sys
sys.path.append('./lib')
sys.path.append('./')
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
import argparse
import json

parser = argparse.ArgumentParser(description='Chan doan KD: dac trung Student vs Teacher')
parser.add_argument('--cfg', type=str, required=True)
parser.add_argument('--teacher', type=str, required=True)
parser.add_argument('--students', type=str, required=True, help='ten=duong_dan,ten2=duong_dan2')
parser.add_argument('--sets', type=str, default='3dpw,3dpw-crowd')
parser.add_argument('--max_samples', type=int, default=3000, help='so mau dau moi tap (0 = toan bo)')
parser.add_argument('--batch_size', type=int, default=0)
parser.add_argument('--out_dir', type=str, default='experiment/diag_kd_features')
parser.add_argument('--gpu', type=str, default='0')
args = parser.parse_args()
os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm

from core.config import cfg, update_config
update_config(args.cfg)

import __init_path  # noqa: F401
from utils.ckpt_tools import build_and_load
from utils.jotr_dataset import get_test_dataset

BRANCHES = (('proj', 'feat_joint_in'), ('joint', 'feat_joint'), ('global', 'feat_global'))


def spearman(a, b):
    ra = np.argsort(np.argsort(a)).astype(np.float64)
    rb = np.argsort(np.argsort(b)).astype(np.float64)
    ra -= ra.mean(); rb -= rb.mean()
    return float((ra * rb).sum() / max(np.sqrt((ra ** 2).sum() * (rb ** 2).sum()), 1e-12))


def feat_stats(s, t):
    """s, t: (B, J, D) hoac (B, D) -> cos (B,), l2n (B,) (TB theo khop neu co)."""
    s, t = s.float(), t.float()
    cos = F.cosine_similarity(s, t, dim=-1)
    l2n = torch.norm(F.normalize(s, dim=-1) - F.normalize(t, dim=-1), dim=-1)
    if cos.dim() == 2:
        cos, l2n = cos.mean(1), l2n.mean(1)
    return cos, l2n


@torch.no_grad()
def run_set(name, teacher, students, dev, bs):
    ds = get_test_dataset(name, None)
    sub = ds if not args.max_samples else Subset(ds, list(range(min(args.max_samples, len(ds)))))
    loader = DataLoader(sub, batch_size=bs, shuffle=False, num_workers=cfg.DATASET.workers)
    reg = torch.as_tensor(ds.h36m_joint_regressor, dtype=torch.float32, device=dev)
    acc = {sn: {'mpjpe': [], **{f'{b}_{m}': [] for b, _ in BRANCHES for m in ('cos', 'l2n')}} for sn in students}
    t_mpjpe, seen = [], 0
    for inputs, targets, meta in tqdm(loader, desc=name, leave=False):
        img = inputs['img'].to(dev).float()
        B = img.shape[0]
        j2 = inputs['joints'].to(dev).float()
        jm = inputs['joints_mask'].to(dev).float().reshape(B, -1, 1)
        cam = inputs['cam_bbox'].to(dev).float()
        gt_mesh = targets['smpl_mesh_cam'].to(dev).float()
        gt17 = torch.einsum('jk,bkc->bjc', reg, gt_mesh)
        gt17 = gt17 - gt17[:, 0:1]                                   # GT 3D root-relative, met (giong orig_joint_cam - goc)
        tgt_np = gt_mesh.cpu().numpy()
        t_out = teacher(img, gt17, is_train=False, pose_2d=j2, joints_mask=jm, cam_bbox=cam)
        res = ds.evaluate([{'smpl_mesh_cam': t_out['smpl_mesh_cam'][k].float().cpu().numpy(),
                            'smpl_mesh_cam_target': tgt_np[k]} for k in range(B)], seen)
        t_mpjpe.extend(res['mpjpe'])
        for sn, sm in students.items():
            s_out = sm(img, j2, is_train=False, joints_mask=jm, cam_bbox=cam)
            res = ds.evaluate([{'smpl_mesh_cam': s_out['smpl_mesh_cam'][k].float().cpu().numpy(),
                                'smpl_mesh_cam_target': tgt_np[k]} for k in range(B)], seen)
            acc[sn]['mpjpe'].extend(res['mpjpe'])
            for b, key in BRANCHES:
                if key not in s_out or key not in t_out:
                    continue
                c, l = feat_stats(s_out[key], t_out[key])
                acc[sn][f'{b}_cos'].append(c.cpu().numpy())
                acc[sn][f'{b}_l2n'].append(l.cpu().numpy())
        seen += B
    r = {'n': int(seen), 'teacher_mpjpe': float(np.mean(t_mpjpe)), 'students': {}}
    for sn, a in acc.items():
        mp = np.asarray(a['mpjpe'])
        d = {'mpjpe': float(mp.mean())}
        for b, _ in BRANCHES:
            if not a[f'{b}_cos']:
                continue
            c = np.concatenate(a[f'{b}_cos']); l = np.concatenate(a[f'{b}_l2n'])
            d[b] = {'cos': float(c.mean()), 'l2n': float(l.mean()), 'rho': spearman(1 - c, mp)}
        r['students'][sn] = d
    return r


def main():
    dev = torch.device('cuda')
    bs = args.batch_size or cfg.TEST.batch_size
    teacher = build_and_load('teacher', args.teacher, dev)[0]
    students = {}
    for item in [s for s in args.students.split(',') if s.strip()]:
        sn, p = item.split('=', 1)
        students[sn.strip()] = build_and_load('student', p.strip(), dev)[0]
    summary = {}
    for name in [s.strip() for s in args.sets.split(',') if s.strip()]:
        r = run_set(name, teacher, students, dev, bs)
        summary[name] = r
        print(f'\n===== [{name}] n = {r["n"]} | Teacher (GT 3D vao) MPJPE14 {r["teacher_mpjpe"]:.2f} =====')
        print(f'  {"student":<10} | {"MPJPE14":>7} | ' + ' | '.join(f'{b:^22}' for b, _ in BRANCHES))
        print(f'  {"":<10} | {"":>7} | ' + ' | '.join(f'{"cos":>6} {"l2n":>6} {"rho":>6}  ' for _ in BRANCHES))
        for sn, d in r['students'].items():
            cells = []
            for b, _ in BRANCHES:
                x = d.get(b)
                cells.append(f'{x["cos"]:>6.3f} {x["l2n"]:>6.3f} {x["rho"]:>+6.2f}  ' if x else f'{"-":>22}')
            print(f'  {sn:<10} | {d["mpjpe"]:>7.2f} | ' + ' | '.join(cells))
        print('  (cos cao / l2n thap = giong Teacher; rho > 0 = mau giong Teacher hon thi it loi hon)')
    os.makedirs(args.out_dir, exist_ok=True)
    out = os.path.join(args.out_dir, 'summary.json')
    with open(out, 'w', encoding='utf-8') as f:
        json.dump({'cfg': args.cfg, 'teacher': args.teacher, 'students': args.students, 'results': summary},
                  f, indent=2, ensure_ascii=False)
    print(f'\nDa luu: {out}')


if __name__ == '__main__':
    main()
