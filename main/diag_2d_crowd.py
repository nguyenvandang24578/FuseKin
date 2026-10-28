"""
main/diag_2d_crowd.py -- Chan doan: sai so PC / Crowd co phai do 2D dau vao (Halpe) khong?

D1 (chat luong 2D Halpe, so voi GT 2D = chieu H36M-17 tu mesh GT vao cung khung crop 256 x 256):
   - loi trung binh / trung vi (px) tren khop Halpe phat hien (mask > 0) VA GT nam trong khung crop
   - % khop xa (> --far_px), % khop xa MA tu tin cao (conf >= --conf_hi) -> nghi "bat nham nguoi"
   - % khop GT nam trong khung nhung Halpe bo (mask 0); % mau co >= 1 khop xa tu tin
D2 (oracle 2D): chay Student voi 3 loai 2D; MOI nhanh (DiffLifter, heatmap [A], feature tai khop [B]/[C], PE,
   tinh chinh [D]) deu nhan cung 2D do:
   - halpe  : nhu test that (phai trung so test)
   - fixfar : chi thay cac khop xa (> far_px) bang GT (conf 1), giu nguyen khop con lai
              -> tach phan loi do khop sai tho (nham nguoi) voi phan do nhieu nho
   - gt     : toan bo GT 2D (conf 1; khop GT ngoai khung -> mask 0) -> tran tren khi 2D hoan hao
   Moi loai: lifter MPJPE17 (so voi GT H36M-17 tu mesh) + mesh dataset.evaluate (MPJPE14 / PA / MPVPE, mm),
   va mesh MPJPE14 tach theo mau CO / KHONG co khop xa tu tin (xac dinh tren Halpe).
Gioi han: crop va cam_bbox van tinh tu 2D Halpe (nhu test that) -> neu Halpe bat nham nguoi, crop co the lech.

Chi danh gia, khong train. OC / PC mac dinh TOAN BO (giao thuc JOTR); --ocpc_mode test = chi seq test.
In them so mau tung tap (n) de doi chieu dinh nghia tap voi JOTR / DPMesh.
Chay:
  python main/diag_2d_crowd.py --cfg ./config/train_student_v3_scratch_jotr.yml \
      --ckpt ./experiment/student_v3_scratch_jotr/checkpoint/best.pth.tar --gpu 0
  (chay thu nhanh: them --max_batches 20)
"""
import os, sys
sys.path.append('./lib')
sys.path.append('./')
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
import argparse
import json

parser = argparse.ArgumentParser(description='Chan doan 2D Halpe tren 3DPW / OC / PC / Crowd')
parser.add_argument('--cfg', type=str, required=True, help='yml DA DUNG de train checkpoint (dung kien truc v3)')
parser.add_argument('--ckpt', type=str, required=True, help='checkpoint Student (best.pth.tar | last.pth.tar)')
parser.add_argument('--sets', type=str, default='3dpw,3dpw-oc,3dpw-pc,3dpw-crowd')
parser.add_argument('--ocpc_mode', type=str, default='all', choices=['all', 'test'])
parser.add_argument('--far_px', type=float, default=40.0, help='nguong khop "xa" (px trong crop 256)')
parser.add_argument('--conf_hi', type=float, default=0.5, help='nguong conf "tu tin" cua Halpe')
parser.add_argument('--batch_size', type=int, default=0, help='0 = cfg.TEST.batch_size')
parser.add_argument('--max_batches', type=int, default=0, help='0 = toan bo tap')
parser.add_argument('--gpu', type=str, default='0')
args = parser.parse_args()
os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from core.config import cfg, update_config
update_config(args.cfg)
cfg.DATASET.pw3d_test_gt2d = True          # dataset tra them GT 2D (targets['orig_joint_img'], meta orig_joint_trunc)

import __init_path  # noqa: F401
from utils.ckpt_tools import build_and_load
from utils.jotr_dataset import get_test_dataset

J = 17
H36M_NAMES = ('Pelvis', 'R_Hip', 'R_Knee', 'R_Ankle', 'L_Hip', 'L_Knee', 'L_Ankle', 'Torso', 'Neck', 'Nose',
              'Head_top', 'L_Shoulder', 'L_Elbow', 'L_Wrist', 'R_Shoulder', 'R_Elbow', 'R_Wrist')
VARIANTS = ('halpe', 'fixfar', 'gt')
PX = cfg.input_img_shape[1] / 2.0          # [-1, 1] -> pixel trong crop


@torch.no_grad()
def run_set(model, ds, loader, dev):
    reg = torch.as_tensor(ds.h36m_joint_regressor, dtype=torch.float32, device=dev)
    hm = torch.tensor([cfg.output_hm_shape[2], cfg.output_hm_shape[1]], dtype=torch.float32, device=dev)
    acc = {k: np.zeros(J) for k in ('err', 'det', 'far', 'hi', 'far_hi', 'gin', 'miss')}
    err_list, sample_bad = [], []
    mesh = {v: {'mpjpe': [], 'pa_mpjpe': [], 'mpvpe': []} for v in VARIANTS}
    lift = {v: [] for v in VARIANTS}
    seen = 0
    for bi, (inputs, targets, meta) in enumerate(tqdm(loader, leave=False)):
        if args.max_batches and bi >= args.max_batches:
            break
        img = inputs['img'].to(dev).float()
        B = img.shape[0]
        j2 = inputs['joints'].to(dev).float()                                     # (B,17,3) [-1,1] + cot 3
        jm = inputs['joints_mask'].to(dev).float().reshape(B, J, 1)               # conf (0 = bo)
        cam = inputs['cam_bbox'].to(dev).float()
        gt_mesh = targets['smpl_mesh_cam'].to(dev).float()
        g2 = targets['orig_joint_img'].to(dev).float()[..., :2] / hm * 2 - 1     # GT 2D (B,17,2) [-1,1]
        gin = meta['orig_joint_trunc'].to(dev).float().reshape(B, J, 1)          # 1 = khop GT trong khung crop

        # ---------- D1: chat luong 2D Halpe ----------
        det = (jm > 0).float()
        valid = det * gin
        e = torch.norm(j2[..., :2] - g2, dim=-1, keepdim=True) * PX              # px
        far = (e > args.far_px).float() * valid
        hi = (jm >= args.conf_hi).float() * valid
        for k, t in (('err', e * valid), ('det', valid), ('far', far), ('hi', hi), ('far_hi', far * hi),
                     ('gin', gin), ('miss', (1 - det) * gin)):
            acc[k] += t.sum(0).reshape(-1).cpu().numpy()
        err_list.append(e[valid > 0].cpu().numpy())
        sample_bad.append(((far * hi).sum((1, 2)) > 0).cpu().numpy())

        # ---------- D2: Student voi 3 loai 2D ----------
        gt17 = torch.einsum('jk,bkc->bjc', reg, gt_mesh)
        gt17 = gt17 - gt17[:, 0:1]
        gt_in = torch.cat([g2 * gin, gin], -1)
        fix_conf = torch.where(far > 0, torch.ones_like(jm), jm)
        fix_in = torch.cat([j2[..., :2] * (1 - far) + g2 * far, fix_conf], -1)
        tgt_np = gt_mesh.cpu().numpy()
        for v, (jj, mm) in (('halpe', (j2, jm)), ('fixfar', (fix_in, fix_conf)), ('gt', (gt_in, gin))):
            out = model(img, jj, is_train=False, joints_mask=mm, cam_bbox=cam)
            lift[v].append((torch.norm(out['lifted_joints_3d'] - gt17, dim=-1).mean(-1) * 1000).cpu().numpy())
            pm = out['smpl_mesh_cam'].float().cpu().numpy()
            res = ds.evaluate([{'smpl_mesh_cam': pm[k], 'smpl_mesh_cam_target': tgt_np[k]} for k in range(B)], seen)
            for k_ in mesh[v]:
                mesh[v][k_].extend(res[k_])
        seen += B

    bad = np.concatenate(sample_bad)
    errs = np.concatenate(err_list) if err_list else np.zeros(0)
    d = lambda a, b: float(a.sum() / max(b.sum(), 1))
    r = {'n_eval': int(seen),
         'd1': {'err_px_mean': d(acc['err'], acc['det']), 'err_px_median': float(np.median(errs)) if errs.size else 0.0,
                'far_pct': 100 * d(acc['far'], acc['det']), 'far_conf_hi_pct': 100 * d(acc['far_hi'], acc['det']),
                'miss_pct': 100 * d(acc['miss'], acc['gin']), 'samples_with_far_conf_hi_pct': 100 * float(bad.mean()),
                'per_joint_far_pct': {H36M_NAMES[j]: 100 * float(acc['far'][j] / max(acc['det'][j], 1)) for j in range(J)}},
         'd2': {}}
    for v in VARIANTS:
        mp = np.asarray(mesh[v]['mpjpe'])
        r['d2'][v] = {'lift_mpjpe17': float(np.concatenate(lift[v]).mean()),
                      **{k_: float(np.mean(x)) for k_, x in mesh[v].items()},
                      'mpjpe_bad_samples': float(mp[bad].mean()) if bad.any() else float('nan'),
                      'mpjpe_clean_samples': float(mp[~bad].mean()) if (~bad).any() else float('nan')}
    return r


def show(label, n, r):
    a = r['d1']
    print(f'\n===== [{label}] n tap = {n} | n danh gia = {r["n_eval"]} =====')
    print(f'D1 2D Halpe: loi TB {a["err_px_mean"]:.1f} px (trung vi {a["err_px_median"]:.1f}) | khop xa (>{args.far_px:.0f}px) '
          f'{a["far_pct"]:.1f}% | xa & conf>={args.conf_hi} {a["far_conf_hi_pct"]:.1f}% | Halpe bo khop GT trong khung '
          f'{a["miss_pct"]:.1f}% | mau co >=1 khop xa tu tin {a["samples_with_far_conf_hi_pct"]:.1f}%')
    top = sorted(a['per_joint_far_pct'].items(), key=lambda x: -x[1])[:5]
    print('   khop xa nhieu nhat: ' + ', '.join(f'{k} {v:.1f}%' for k, v in top))
    print(f'D2 {"2D":<7} | {"lift17":>7} | {"MPJPE":>7} | {"PA":>6} | {"MPVPE":>7} | {"MPJPE mau xau":>13} | {"MPJPE mau sach":>14}')
    for v in VARIANTS:
        m = r['d2'][v]
        print(f'   {v:<7} | {m["lift_mpjpe17"]:>7.2f} | {m["mpjpe"]:>7.2f} | {m["pa_mpjpe"]:>6.2f} | {m["mpvpe"]:>7.2f} | '
              f'{m["mpjpe_bad_samples"]:>13.2f} | {m["mpjpe_clean_samples"]:>14.2f}')


def main():
    dev = torch.device('cuda')
    model = build_and_load('student', args.ckpt, dev)[0]
    bs = args.batch_size or cfg.TEST.batch_size
    summary = {}
    for name in [s.strip() for s in args.sets.split(',') if s.strip()]:
        cfg.DATASET.pw3d_ocpc_test_only = (args.ocpc_mode == 'test') and name in ('3dpw-oc', '3dpw-pc')
        label = name + (' (chi seq test)' if cfg.DATASET.pw3d_ocpc_test_only else '')
        ds = get_test_dataset(name, None)
        loader = DataLoader(ds, batch_size=bs, shuffle=False, num_workers=cfg.DATASET.workers)
        r = run_set(model, ds, loader, dev)
        r['n_set'] = len(ds)
        show(label, len(ds), r)
        summary[label] = r
    os.makedirs(cfg.output_dir, exist_ok=True)
    out = os.path.join(cfg.output_dir, 'diag_2d_crowd.json')
    with open(out, 'w', encoding='utf-8') as f:
        json.dump({'ckpt': args.ckpt, 'cfg': args.cfg, 'far_px': args.far_px, 'conf_hi': args.conf_hi,
                   'results': summary}, f, indent=2, ensure_ascii=False)
    print('\n' + '=' * 100)
    print(f'TOM TAT ({args.ckpt}) | mesh MPJPE14: halpe -> fixfar -> gt | lift17 halpe -> gt | 2D xa tu tin %')
    for label, r in summary.items():
        m = r['d2']
        print(f'  {label:<26} n={r["n_set"]:>6} | {m["halpe"]["mpjpe"]:6.2f} -> {m["fixfar"]["mpjpe"]:6.2f} -> '
              f'{m["gt"]["mpjpe"]:6.2f} | {m["halpe"]["lift_mpjpe17"]:6.2f} -> {m["gt"]["lift_mpjpe17"]:6.2f} | '
              f'{r["d1"]["far_conf_hi_pct"]:5.1f}%')
    print(f'Da luu: {out}')


if __name__ == '__main__':
    main()
