"""
main/eval_difflifter_3dpw.py
Do MPJPE (17 khop H36M, root-relative, mm) cua 1 checkpoint DiffLifter tren cac tap 3DPW. Chi in MPJPE.

  3dpw-train-noaug : 3DPW train, TAT augmentation (lifter KHONG train tren tap nay -> cung la du lieu chua thay)
  3dpw-val         : 3DPW validation (tap chon best)
  3dpw             : 3DPW test

GT: H36M regressor x mesh SMPL GT (giong finetune_difflifter.evaluate). Mau khong co GT (toan 0) bi bo qua.
Sampler mac dinh = sampler luu trong checkpoint (S, K); doi bang --S / --K. K > 1: gop trung binh K gia thuyet.

Chay:
  python main/eval_difflifter_3dpw.py --cfg config/finetune_difflifter_r25_cam.yaml --gpu 0 \
      --ckpt experiment/finetune_difflifter_r25_cam/best_epoch.bin
"""
import os
import sys
sys.path.append('./lib')
sys.path.append('./')

import argparse

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from core.config import update_config, cfg
from utils.jotr_dataset import get_test_dataset
from diag_difflifter import build_model, make_cond, get_gt


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--cfg', required=True)
    p.add_argument('--ckpt', required=True)
    p.add_argument('--gpu', default='0')
    p.add_argument('--sets', default='3dpw-train-noaug,3dpw-val,3dpw')
    p.add_argument('--batch_size', type=int, default=512)
    p.add_argument('--S', type=int, default=0, help='>0: doi so buoc DDIM (mac dinh theo checkpoint)')
    p.add_argument('--K', type=int, default=0, help='>0: doi so gia thuyet (mac dinh theo checkpoint)')
    p.add_argument('--seed', type=int, default=0)
    return p.parse_args()


@torch.no_grad()
def eval_mpjpe(model, name, args, device):
    ds = get_test_dataset(name, None)
    loader = DataLoader(ds, batch_size=args.batch_size, shuffle=False, num_workers=cfg.DATASET.workers,
                        pin_memory=True)
    reg = torch.as_tensor(ds.dataset.h36m_joint_regressor, dtype=torch.float32, device=device)
    gen = torch.Generator(device=device)
    gen.manual_seed(args.seed)
    err_sum, n, n_skip = 0.0, 0, 0
    for inp, tgt, meta in tqdm(loader, desc=name):
        cond = make_cond(inp, device)                         # (B, 17, 6)
        gt = get_gt(tgt, reg, device)                         # (B, 17, 3) met, root-relative
        valid = gt.abs().flatten(1).sum(1) > 0                # mau khong co GT -> bo
        n_skip += int((~valid).sum())
        if valid.sum() == 0:
            continue
        mean_pose, _, _ = model(cond[valid], generator=gen)
        mean_pose = mean_pose - mean_pose[:, 0:1]
        err_sum += torch.norm(mean_pose - gt[valid], dim=-1).mean(-1).sum().item()
        n += int(valid.sum())
    return err_sum / max(n, 1) * 1000, n, n_skip


def main():
    args = parse_args()
    if args.gpu:
        os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
    update_config(args.cfg)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    ckpt = torch.load(args.ckpt, map_location='cpu')
    model = build_model(ckpt, device)
    if args.S > 0:
        model.sampling_timesteps = args.S
    if args.K > 0:
        model.num_proposals = args.K

    results = []
    for name in [s.strip() for s in args.sets.split(',') if s.strip()]:
        try:
            results.append((name, *eval_mpjpe(model, name, args, device)))
        except Exception as e:
            results.append((name, None, 0, 0))
            print(f'[{name}] LOI: {type(e).__name__}: {e}')

    print('\n' + '=' * 60)
    print(f'ckpt = {args.ckpt} | epoch = {ckpt.get("epoch", "?")} | S = {model.sampling_timesteps} | K = {model.num_proposals}')
    print('=' * 60)
    for name, mpjpe, n, n_skip in results:
        if mpjpe is None:
            print(f'  {name:<18} LOI (xem phia tren)')
        else:
            skip = f' | bo {n_skip} mau khong co GT' if n_skip else ''
            print(f'  {name:<18} MPJPE = {mpjpe:6.2f} mm  (n = {n}{skip})')


if __name__ == '__main__':
    main()
