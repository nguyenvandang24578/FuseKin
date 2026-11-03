"""
main/diag_student_vs_lifter.py -- Chan doan (2026-11-03): Student co DUNG ANH de sua 2D sai so voi DiffLifter khong?
Chi danh gia, khong train. 2D dau vao = nguon test cua YAML (OpenPose / HHRNet o che do JOTR).

(1) Theo TUNG KHOP (12 khop tay chan, chi khop GT nam trong crop): phan nhom theo 2D dau vao so voi GT 2D
    (GT 2D = H36M-17 chieu tu mesh GT; kich thuoc nguoi = canh lon khung bao GT 2D trong crop):
      tot   : 2D phat hien, lech <= --far x kich thuoc nguoi
      xa    : 2D phat hien, lech >  --far x kich thuoc nguoi (sai tho / bat nham nguoi)
      thieu : 2D bi bo (mask 0)
    Moi nhom: loi 3D TB (mm, root-relative pelvis) cua LIFTER (lifted_joints_3d) va STUDENT (H36M regressor x mesh),
    % khop Student SUA (loi giam > --gain mm so voi lifter) / LAM HONG (loi tang > --gain mm).
(2) ANH XAM: chay lai Student voi anh = mau trung binh tung kenh cua chinh anh do (giu nguyen 2D) -> neu MPJPE gan
    nhu khong doi thi Student khong dung anh.
(3) ORACLE fixfar: thay khop 2D "xa" bang GT 2D (conf 1), giu cac khop khac -> tran tren neu sua duoc moi khop sai tho.
Moi bien the: mesh MPJPE14 / PA / MPVPE (dataset.evaluate) + lift17.
Checkpoint: --ckpt_dir chua best_<tap>.pth.tar (giong JOTR / DPMesh: 1 ckpt / tap), thieu -> best.pth.tar; hoac --ckpt.
Chay:
  python main/diag_student_vs_lifter.py --cfg ./config/train_student_v3_kd_jotr_pf05lift3dpw.yml \\
      --ckpt_dir ./experiment/student_v3_kd_jotr_pf05lift3dpw_t2/checkpoint --sets 3dpw,3dpw-oc,3dpw-pc,3dpw-crowd
  (thu nhanh: --max_batches 20)
"""
import os, sys
sys.path.append('./lib')
sys.path.append('./')
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
import argparse
import json

parser = argparse.ArgumentParser(description='Chan doan Student vs DiffLifter (sua 2D sai bang anh?)')
parser.add_argument('--cfg', type=str, required=True)
parser.add_argument('--ckpt_dir', type=str, default='', help='thu muc co best_<tap>.pth.tar')
parser.add_argument('--ckpt', type=str, default='', help='1 checkpoint cho moi tap (bo qua --ckpt_dir)')
parser.add_argument('--sets', type=str, default='3dpw,3dpw-oc,3dpw-pc,3dpw-crowd')
parser.add_argument('--far', type=float, default=0.2, help='khop "xa" = lech > far x kich thuoc nguoi (giong compare_2d_sources)')
parser.add_argument('--gain', type=float, default=30.0, help='nguong mm de tinh SUA / LAM HONG')
parser.add_argument('--variants', type=str, default='real,grey,fixfar')
parser.add_argument('--batch_size', type=int, default=0)
parser.add_argument('--max_batches', type=int, default=0)
parser.add_argument('--out_dir', type=str, default='experiment/diag_student_vs_lifter')
parser.add_argument('--gpu', type=str, default='0')
args = parser.parse_args()
os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from core.config import cfg, update_config
update_config(args.cfg)
cfg.DATASET.pw3d_test_gt2d = True          # dataset tra them GT 2D (targets orig_joint_img, meta orig_joint_trunc)

import __init_path  # noqa: F401
from utils.ckpt_tools import build_and_load
from utils.jotr_dataset import get_test_dataset

J = 17
LIMB = (1, 2, 3, 4, 5, 6, 11, 12, 13, 14, 15, 16)        # hong, goi, co chan, vai, khuyu, co tay (H36M-17)
GROUPS = ('tot', 'xa', 'thieu')
VARIANTS = tuple(v.strip() for v in args.variants.split(',') if v.strip())
assert set(VARIANTS) <= {'real', 'grey', 'fixfar'} and 'real' in VARIANTS, VARIANTS


@torch.no_grad()
def run_set(model, ds, loader, dev):
    reg = torch.as_tensor(ds.h36m_joint_regressor, dtype=torch.float32, device=dev)
    hm = torch.tensor([cfg.output_hm_shape[2], cfg.output_hm_shape[1]], dtype=torch.float32, device=dev)
    mesh = {v: {'mpjpe': [], 'pa_mpjpe': [], 'mpvpe': []} for v in VARIANTS}
    lift = {v: [] for v in VARIANTS}
    jl, js, jg = [], [], []                 # loi tung khop lifter / student (mm), nhom (0 tot, 1 xa, 2 thieu, -1 bo qua)
    seen = 0
    for bi, (inputs, targets, meta) in enumerate(tqdm(loader, leave=False)):
        if args.max_batches and bi >= args.max_batches:
            break
        img = inputs['img'].to(dev).float()
        B = img.shape[0]
        j2 = inputs['joints'].to(dev).float()
        jm = inputs['joints_mask'].to(dev).float().reshape(B, J, 1)
        cam = inputs['cam_bbox'].to(dev).float()
        gt_mesh = targets['smpl_mesh_cam'].to(dev).float()
        g2 = targets['orig_joint_img'].to(dev).float()[..., :2] / hm * 2 - 1      # GT 2D [-1, 1]
        gin = meta['orig_joint_trunc'].to(dev).float().reshape(B, J, 1)

        # kich thuoc nguoi (don vi [-1, 1]) tu GT 2D trong crop
        size = torch.maximum(g2[..., 0].amax(1) - g2[..., 0].amin(1), g2[..., 1].amax(1) - g2[..., 1].amin(1))
        size = size.clamp_min(1e-3).reshape(B, 1, 1)
        det = (jm > 0).float()
        e2 = torch.norm(j2[..., :2] - g2, dim=-1, keepdim=True)
        far = (e2 > args.far * size).float() * det
        grp = torch.full((B, J), -1, dtype=torch.long, device=dev)
        g_in = gin[..., 0] > 0
        grp[g_in & (det[..., 0] > 0) & (far[..., 0] == 0)] = 0
        grp[g_in & (far[..., 0] > 0)] = 1
        grp[g_in & (det[..., 0] == 0)] = 2

        gt17 = torch.einsum('jk,bkc->bjc', reg, gt_mesh)
        gt17 = gt17 - gt17[:, 0:1]
        tgt_np = gt_mesh.cpu().numpy()
        fix_conf = torch.where(far > 0, torch.ones_like(jm), jm)
        fix_in = torch.cat([j2[..., :2] * (1 - far) + g2 * far, fix_conf, j2[..., 3:]], -1) if j2.shape[-1] > 3 else \
            torch.cat([j2[..., :2] * (1 - far) + g2 * far, fix_conf], -1)
        grey = img.mean(dim=(2, 3), keepdim=True).expand_as(img).contiguous()
        cand = {'real': (img, j2, jm), 'grey': (grey, j2, jm), 'fixfar': (img, fix_in, fix_conf)}
        for v in VARIANTS:
            im_, jj, mm = cand[v]
            out = model(im_, jj, is_train=False, joints_mask=mm, cam_bbox=cam)
            lj = out['lifted_joints_3d'].float()
            lift[v].append((torch.norm(lj - gt17, dim=-1).mean(-1) * 1000).cpu().numpy())
            pm = out['smpl_mesh_cam'].float()
            res = ds.evaluate([{'smpl_mesh_cam': pm[k].cpu().numpy(), 'smpl_mesh_cam_target': tgt_np[k]}
                               for k in range(B)], seen)
            for k_ in mesh[v]:
                mesh[v][k_].extend(res[k_])
            if v == 'real':
                s17 = torch.einsum('jk,bkc->bjc', reg, pm)
                s17 = s17 - s17[:, 0:1]
                jl.append((torch.norm(lj - gt17, dim=-1) * 1000)[:, LIMB].cpu().numpy())
                js.append((torch.norm(s17 - gt17, dim=-1) * 1000)[:, LIMB].cpu().numpy())
                jg.append(grp[:, LIMB].cpu().numpy())
        seen += B

    jl, js, jg = np.concatenate(jl), np.concatenate(js), np.concatenate(jg)
    r = {'n_eval': int(seen), 'variants': {}, 'joint_groups': {}}
    for v in VARIANTS:
        r['variants'][v] = {'lift_mpjpe17': float(np.concatenate(lift[v]).mean()),
                            **{k_: float(np.mean(x)) for k_, x in mesh[v].items()}}
    n_all = int((jg >= 0).sum())
    for gi, gname in enumerate(GROUPS):
        m = jg == gi
        if not m.any():
            r['joint_groups'][gname] = {'n': 0}
            continue
        d = js[m] - jl[m]
        r['joint_groups'][gname] = {'n': int(m.sum()), 'pct': 100 * float(m.sum() / max(n_all, 1)),
                                    'lift_mm': float(jl[m].mean()), 'student_mm': float(js[m].mean()),
                                    'sua_pct': 100 * float((d < -args.gain).mean()),
                                    'hong_pct': 100 * float((d > args.gain).mean())}
    return r


def show(label, n, r):
    print(f'\n===== [{label}] n = {n} (danh gia {r["n_eval"]}) =====')
    print(f'  {"bien the":<8} | {"MPJPE14":>7} | {"PA":>6} | {"MPVPE":>7} | {"lift17":>7}')
    for v, m in r['variants'].items():
        print(f'  {v:<8} | {m["mpjpe"]:>7.2f} | {m["pa_mpjpe"]:>6.2f} | {m["mpvpe"]:>7.2f} | {m["lift_mpjpe17"]:>7.2f}')
    if 'grey' in r['variants']:
        d = r['variants']['grey']['mpjpe'] - r['variants']['real']['mpjpe']
        print(f'  -> anh xam lam MPJPE14 thay doi {d:+.2f} mm (gan 0 = Student khong dung anh)')
    print(f'  khop tay chan theo 2D dau vao (xa = lech > {args.far} x kich thuoc nguoi; SUA / HONG = Student vs lifter '
          f'> {args.gain:.0f} mm):')
    print(f'  {"nhom 2D":<6} | {"n khop":>8} | {"%":>5} | {"lifter mm":>9} | {"Student mm":>10} | {"SUA %":>6} | {"HONG %":>6}')
    for g, m in r['joint_groups'].items():
        if m['n'] == 0:
            print(f'  {g:<6} | {0:>8} |')
            continue
        print(f'  {g:<6} | {m["n"]:>8} | {m["pct"]:>5.1f} | {m["lift_mm"]:>9.1f} | {m["student_mm"]:>10.1f} | '
              f'{m["sua_pct"]:>6.1f} | {m["hong_pct"]:>6.1f}')


def ckpt_for(name):
    if args.ckpt:
        return args.ckpt
    p = os.path.join(args.ckpt_dir, f'best_{name}.pth.tar')
    return p if os.path.isfile(p) else os.path.join(args.ckpt_dir, 'best.pth.tar')


def main():
    assert args.ckpt or args.ckpt_dir, 'can --ckpt hoac --ckpt_dir'
    dev = torch.device('cuda')
    bs = args.batch_size or cfg.TEST.batch_size
    summary, cache = {}, {}
    for name in [s.strip() for s in args.sets.split(',') if s.strip()]:
        ck = ckpt_for(name)
        if ck not in cache:
            cache.clear()
            cache[ck] = build_and_load('student', ck, dev)[0]
        model = cache[ck]
        print(f'\n[{name}] checkpoint: {ck}')
        ds = get_test_dataset(name, None)
        loader = DataLoader(ds, batch_size=bs, shuffle=False, num_workers=cfg.DATASET.workers)
        r = run_set(model, ds, loader, dev)
        r['n_set'], r['ckpt'] = len(ds), ck
        show(name, len(ds), r)
        summary[name] = r
    os.makedirs(args.out_dir, exist_ok=True)
    out = os.path.join(args.out_dir, 'summary.json')
    with open(out, 'w', encoding='utf-8') as f:
        json.dump({'cfg': args.cfg, 'far': args.far, 'gain': args.gain, 'results': summary}, f, indent=2, ensure_ascii=False)
    print('\n' + '=' * 90)
    print('TOM TAT MPJPE14: ' + ' / '.join(VARIANTS) + ' | lift17 real | khop 2D xa: lifter -> Student mm (SUA% / HONG%)')
    for name, r in summary.items():
        v = r['variants']
        x = r['joint_groups'].get('xa', {})
        xs = (f'{x["lift_mm"]:.1f} -> {x["student_mm"]:.1f} ({x["sua_pct"]:.1f} / {x["hong_pct"]:.1f})'
              if x.get('n') else '-')
        print(f'  {name:<11} | ' + ' / '.join(f'{v[k]["mpjpe"]:.2f}' for k in VARIANTS) +
              f' | {v["real"]["lift_mpjpe17"]:.2f} | {xs}')
    print(f'Da luu: {out}')


if __name__ == '__main__':
    main()
