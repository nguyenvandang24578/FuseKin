"""So sanh dau ra MotionBERT khi dau vao 2D la GT sach hay 2D detector (Halpe-26).

Cung checkpoint, cung mau, cung giao thuc (17 khop H36M, root-relative, F = 1). Chi khac 2D dau vao:
  halpe          : 2D detector cua dataset (inputs['joints'], mask = inputs['joints_mask'])
  gt2d           : GT 2D sach = targets['orig_joint_img'] (phep chieu cua khop 3D dich), mask = orig_joint_valid * orig_joint_trunc
  gt2d+<N>px     : GT 2D + nhieu Gauss N pixel (trong crop 256 px) -> uoc luong nhieu tuong duong cua detector

Chay (tu thu muc goc repo):
  python main/compare_motionbert_gt2d.py --cfg ./config/finetune_motionbert_halpe26.yaml \
      --ckpt experiment/finetune_motionbert_halpe26_3dpw/best_epoch.bin \
      --split 3dpw-train-noaug --max_samples 4000

Luu y:
  * GT 2D: tap train cua 3DPW tra san; val/test tra khi cfg.DATASET.pw3d_test_gt2d = True (script tu bat).
    Vd test: --split 3dpw | 3dpw-oc | 3dpw-pc. (3dpw-crowd + halpe26 hien bi loi ValueError, xem PLAN_FULL_DATA V3)
  * GT 2D la phep chieu CHINH XAC cua 3D dich -> day la can tren (oracle), khong phai kich ban thuc.
  * Neu checkpoint da finetune tren 3dpw-train thi ca hai dong deu da thay mau nay. Chay them voi checkpoint
    chua finetune (MB_release.bin / MB_ft_h36m.bin) de co so sanh khong bi hoc thuoc.
"""
import os, sys, json
sys.path.append('./lib')
sys.path.append('./')
import argparse
import numpy as np
import torch
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm

from core.config import cfg, update_config
from utils.jotr_dataset import get_test_dataset
from utils.h36m_adapter import h36m_joint_indices
# Dung chung ham voi finetune_motionbert.py de dau vao/giao thuc dung y het luc finetune
from finetune_motionbert import build_model, make_mb_input, normalize_hm_xy, batch_pa_errors

JOINTS = ['Pelvis', 'R_Hip', 'R_Knee', 'R_Ankle', 'L_Hip', 'L_Knee', 'L_Ankle', 'Torso', 'Neck', 'Nose',
          'Head_top', 'L_Shoulder', 'L_Elbow', 'L_Wrist', 'R_Shoulder', 'R_Elbow', 'R_Wrist']


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--cfg', default='config/finetune_motionbert_halpe26.yaml')
    ap.add_argument('--ckpt', required=True, help="checkpoint MotionBERT (khoa 'model_pos')")
    ap.add_argument('--split', default='3dpw-train-noaug')
    ap.add_argument('--max_samples', type=int, default=4000, help='<=0: toan bo; lay mau trai deu')
    ap.add_argument('--batch_size', type=int, default=64)
    ap.add_argument('--workers', type=int, default=8)
    ap.add_argument('--noise_px', type=float, nargs='*', default=[2.0, 5.0, 10.0],
                    help='muc nhieu Gauss (px trong crop 256) them vao GT 2D')
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--gpu', default='0')
    args, _ = ap.parse_known_args()

    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
    update_config(args.cfg)
    cfg.DATASET.workers = args.workers
    cfg.DATASET.pw3d_test_gt2d = True   # val/test cua 3DPW tra them GT 2D (targets['orig_joint_img'])
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = torch.device('cuda')

    mb_cfg = cfg.MOTIONBERT
    rootrel = cfg.MODEL.motionbert_2d_rootrel
    drop_names = list(cfg.MODEL.motionbert_drop_joints)
    drop_idx = h36m_joint_indices(drop_names)
    print(f'[cfg] pw3d_2d_source={cfg.DATASET.pw3d_2d_source} | rootrel_2d={rootrel} | drop_joints={drop_names}')
    if cfg.DATASET.pw3d_2d_source != 'halpe26':
        print("  CANH BAO: pw3d_2d_source khong phai 'halpe26' -> dong 'halpe' dang dung 2D OpenPose.")

    # ---------------- Model ----------------
    model = build_model(mb_cfg)
    ckpt = torch.load(args.ckpt, map_location=lambda s, l: s)
    model.load_state_dict(ckpt['model_pos'], strict=True)
    model.eval()
    meta_ck = ckpt.get('fusekin_meta', {})
    print(f"[ckpt] {args.ckpt} | epoch {ckpt.get('epoch', '?')} | meta: {meta_ck if meta_ck else '(khong co)'}")
    if args.split.startswith('3dpw-train') and '3dpw-train' in meta_ck.get('train_list', []):
        print('  LUU Y: checkpoint da finetune tren 3dpw-train -> ca halpe va gt2d deu la mau da thay luc train. '
              'Chay them voi checkpoint chua finetune de so sanh khong bi hoc thuoc.')

    # ---------------- Data ----------------
    dataset = get_test_dataset(args.split, None)
    n_total = len(dataset)
    if args.max_samples <= 0 or args.max_samples >= n_total:
        idxs = np.arange(n_total)
    else:
        idxs = np.linspace(0, n_total - 1, args.max_samples).astype(int)
    loader = DataLoader(Subset(dataset, idxs.tolist()), batch_size=args.batch_size, shuffle=False,
                        num_workers=cfg.DATASET.workers, pin_memory=True)
    reg = torch.as_tensor(dataset.dataset.h36m_joint_regressor, dtype=torch.float32, device=device)
    print(f'[data] {args.split}: dung {len(idxs)}/{n_total} mau')

    half = cfg.input_img_shape[1] / 2.0          # 1 don vi toa do chuan hoa = 128 px trong crop 256
    names = ['halpe', 'gt2d'] + [f'gt2d+{int(p) if float(p).is_integer() else p}px' for p in args.noise_px]
    err_pj = {n: [] for n in names}               # moi phan tu (B, 17) mm
    pa_err = {n: [] for n in names}               # moi phan tu (B,) mm
    mask_sum = {n: np.zeros(17) for n in names}
    px_sum, px_cnt = np.zeros(17), np.zeros(17)
    n_seen = 0

    with torch.no_grad():
        for bi, (inputs, targets, meta) in enumerate(tqdm(loader, desc='MotionBERT')):
            if bi == 0:
                need_t = [k for k in ('orig_joint_img',) if k not in targets]
                need_m = [k for k in ('orig_joint_valid', 'orig_joint_trunc') if k not in meta]
                if need_t or need_m:
                    raise SystemExit(f'Tap {args.split} khong tra GT 2D (thieu targets{need_t} meta{need_m}). '
                                     f'targets keys: {list(targets.keys())}. Kiem tra cfg.DATASET.pw3d_test_gt2d / PW3D dataset.')
            xy_h = inputs['joints'][..., :2].to(device).float()
            conf_h = inputs['joints_mask'].to(device).float()
            gt_xy = normalize_hm_xy(targets['orig_joint_img'][..., :2].to(device).float())
            gt_mask = (meta['orig_joint_valid'] * meta['orig_joint_trunc']).to(device).float()
            gt_xy = gt_xy * (gt_mask > 0)

            if 'orig_joint_cam' in targets:
                gt3d = targets['orig_joint_cam'].to(device).float()
            else:
                gt3d = torch.matmul(reg[None], targets['smpl_mesh_cam'].to(device).float())
            gt3d = gt3d - gt3d[:, 0:1, :]

            variants = {'halpe': (xy_h, conf_h), 'gt2d': (gt_xy, gt_mask)}
            for p, n in zip(args.noise_px, names[2:]):
                sigma = p / half                                   # px -> don vi chuan hoa
                noisy = (gt_xy + torch.randn_like(gt_xy) * sigma) * (gt_mask > 0)
                variants[n] = (noisy, gt_mask)

            for n, (xy, conf) in variants.items():
                pred = model(make_mb_input(xy, conf, rootrel, drop_idx))[:, 0]
                pred = pred - pred[:, 0:1, :]
                err_pj[n].append((torch.norm(pred - gt3d, dim=-1) * 1000).cpu().numpy())
                pa_err[n].append((batch_pa_errors(pred, gt3d) * 1000).cpu().numpy())
                mask_sum[n] += (conf[..., 0] > 0).float().sum(0).cpu().numpy()

            both = (conf_h[..., 0] > 0) & (gt_mask[..., 0] > 0)
            d_px = ((xy_h - gt_xy).norm(dim=-1) * half) * both
            px_sum += d_px.sum(0).cpu().numpy()
            px_cnt += both.float().sum(0).cpu().numpy()
            n_seen += gt3d.shape[0]

    err_pj = {n: np.concatenate(v) for n, v in err_pj.items()}
    pa_err = {n: np.concatenate(v) for n, v in pa_err.items()}
    mpjpe = {n: err_pj[n].mean(1) for n in names}                  # (N,) moi mau

    # ---------------- In ket qua ----------------
    print('\n' + '=' * 78)
    print(f'MotionBERT lift | {args.split} | n={n_seen} | 17 khop, root-relative (mm)')
    print('=' * 78)
    print(f'{"dau vao 2D":<16}{"MPJPE":>10}{"PA-MPJPE":>11}{"khop co mat":>14}')
    for n in names:
        print(f'{n:<16}{mpjpe[n].mean():>10.2f}{pa_err[n].mean():>11.2f}{mask_sum[n].mean() / n_seen * 100:>13.1f}%')

    d = mpjpe['halpe'] - mpjpe['gt2d']
    se = d.std(ddof=1) / np.sqrt(len(d))
    print(f'\nhalpe - gt2d (MPJPE): {d.mean():+.2f} +/- {1.96 * se:.2f} mm | '
          f'gt2d tot hon o {(d > 0).mean() * 100:.1f}% mau | GT 2D giam loi {d.mean() / mpjpe["halpe"].mean() * 100:.1f}%')
    print('  (khoang tin cay tinh theo mau doc lap -> lac quan: cac frame 3DPW lien tiep tuong quan)')

    xs = [0.0] + list(args.noise_px)
    ys = [mpjpe['gt2d'].mean()] + [mpjpe[n].mean() for n in names[2:]]
    h = mpjpe['halpe'].mean()
    if all(b >= a for a, b in zip(ys, ys[1:])) and ys[0] <= h <= ys[-1]:
        print(f'Loi cua 2D Halpe tuong duong GT 2D + nhieu Gauss ~ {np.interp(h, ys, xs):.1f} px (noi suy tuyen tinh)')
    else:
        print('Loi cua 2D Halpe nam NGOAI khoang cac muc nhieu da thu (hoac khong don dieu) -> doi --noise_px')

    print(f'\n{"khop":<12}{"halpe (mm)":>12}{"gt2d (mm)":>12}{"2D lech (px)":>14}')
    for j, jn in enumerate(JOINTS):
        px = px_sum[j] / px_cnt[j] if px_cnt[j] > 0 else float('nan')
        print(f'{jn:<12}{err_pj["halpe"][:, j].mean():>12.1f}{err_pj["gt2d"][:, j].mean():>12.1f}{px:>14.1f}')
    print('  "2D lech" = khoang cach giua 2D Halpe va GT 2D (px trong crop 256), chi tinh khop ca hai deu hop le.')
    print('  Phan lech nay gom CA nhieu detector LAN khac dinh nghia khop (Halpe nam tren be mat, GT 2D = khop trong co the).')

    out = {'ckpt': args.ckpt, 'split': args.split, 'n': int(n_seen), 'cfg': args.cfg,
           'mpjpe_mm': {n: float(mpjpe[n].mean()) for n in names},
           'pa_mpjpe_mm': {n: float(pa_err[n].mean()) for n in names},
           'halpe_minus_gt2d_mm': float(d.mean()),
           'per_joint_mm': {n: err_pj[n].mean(0).tolist() for n in names}}
    os.makedirs(cfg.res_dir, exist_ok=True)
    path = os.path.join(cfg.res_dir, 'compare_motionbert_gt2d.json')
    with open(path, 'w') as f:
        json.dump(out, f, indent=2)
    print(f'\nDa luu: {path}')


if __name__ == '__main__':
    main()
