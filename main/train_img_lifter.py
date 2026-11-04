"""
main/train_img_lifter.py -- Train ImgLifter (lib/models/img_lifter.py): diffusion 2D -> 3D CO ANH, thay DiffLifter cu.

Moi batch: anh + 2D (GT + PoseFix kieu JOTR, giong luc train Refiner) -> Refiner 2D DONG BANG -> xy, log_b, f8, feat_j
-> ImgLifter hoc khu nhieu x0. Muc tieu = orig_joint_cam cua mau fusekin_unified = H36M-17 tinh tu mesh (met,
root-relative) cho ca 4 bo -> cung dinh nghia khop mesh voi GT 3DPW. Khong dung 3DPW-train (giao thuc JOTR).
  - IMGLIFT.input_drop_prob: bo ngau nhien khop dau vao Refiner (det = 0) giong detector bo sot luc test.
  - IMGLIFT.train_xy_noise: cong nhieu Gauss std = train_xy_noise x b vao xy Refiner (Refiner chay tren anh no
    da hoc -> loi nho hon test; nhieu bu khoang cach).
EMA (IMGLIFT.ema_decay): 'model_state' trong checkpoint = trong so EMA (dung danh gia); 'raw_model_state' de resume.
Moi epoch: danh gia nhanh tren IMGLIFT.eval_sets (chi LOG). Bao ket qua theo EPOCH CUOI (lich co dinh).

Chay:
  python main/train_img_lifter.py --cfg config/train_img_lifter_jotr.yaml --exp_name img_lifter_jotr --gpu 0
"""
import os, sys
sys.path.append('./lib')
sys.path.append('./')
os.environ["OMP_NUM_THREADS"] = "1"
import argparse
import copy
import json
import random

parser = argparse.ArgumentParser(description='Train ImgLifter')
parser.add_argument('--cfg', type=str, required=True)
parser.add_argument('--gpu', type=str, default='0')
parser.add_argument('--exp_name', type=str, required=True, help='experiment/<exp_name> (doc trong core/config.py)')
parser.add_argument('--seed', type=int, default=123)
parser.add_argument('--resume_training', action='store_true', help='tiep tuc tu checkpoint/last.pth.tar')
args = parser.parse_args()
os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm

from core.config import cfg, update_config
update_config(args.cfg)
assert cfg.DATASET.fusekin_unified, 'ImgLifter dung mau fusekin_unified (anh + 2D + GT 3D tu mesh)'
cfg.DATASET.pw3d_test_gt2d = True          # tap danh gia 3DPW tra GT 2D -> log lech 2D (reproj)
assert cfg.IMGLIFT.refiner_ckpt, 'Can IMGLIFT.refiner_ckpt (checkpoint Refiner 2D target mesh)'

import __init_path  # noqa: F401
from core.base import get_dataloader
from models.img_lifter import ImgLifter
from utils.jotr_dataset import get_test_dataset
from utils.img_lifter_tools import (build_img_lifter, imglift_arch, load_frozen_refiner, refiner_inputs,
                                    lifter_det, eval_img_lifter, format_eval)

random.seed(args.seed)
np.random.seed(args.seed)
torch.manual_seed(args.seed)
DEVICE = torch.device('cuda')
L = cfg.IMGLIFT


def main():
    log_path = os.path.join(cfg.output_dir, 'log.txt')

    def log(msg):
        print(msg)
        with open(log_path, 'a', encoding='utf-8') as f:
            f.write(msg + '\n')

    log(f'cfg={args.cfg} | IMGLIFT={json.dumps(dict(L), default=str)} | TRAIN lr={cfg.TRAIN.lr} '
        f'lr_step={cfg.TRAIN.lr_step} end_epoch={cfg.TRAIN.end_epoch} batch={cfg.TRAIN.batch_size}')
    refiner = load_frozen_refiner(L.refiner_ckpt, DEVICE)
    model = build_img_lifter().to(DEVICE)
    ema = copy.deepcopy(model).eval()
    for p in ema.parameters():
        p.requires_grad = False
    log(f'ImgLifter: {sum(p.numel() for p in model.parameters()) / 1e6:.2f}M tham so')
    opt = torch.optim.AdamW(model.parameters(), lr=cfg.TRAIN.lr, weight_decay=float(L.weight_decay))
    sched = torch.optim.lr_scheduler.MultiStepLR(opt, milestones=list(cfg.TRAIN.lr_step), gamma=cfg.TRAIN.lr_factor)

    _, train_loader = get_dataloader(args, cfg.DATASET.train_list, is_train=True)
    eval_loaders = {}
    for name in list(L.eval_sets or []):
        ds = get_test_dataset(name, None)
        reg = ds.h36m_joint_regressor
        if L.eval_max_samples and L.eval_max_samples < len(ds):
            ds = Subset(ds, list(range(int(L.eval_max_samples))))
        eval_loaders[name] = (DataLoader(ds, batch_size=cfg.TEST.batch_size, shuffle=False,
                                         num_workers=cfg.DATASET.workers), reg)

    begin = 1
    last_path = os.path.join(cfg.checkpoint_dir, 'last.pth.tar')
    if args.resume_training:
        ck = torch.load(last_path, map_location='cpu', weights_only=False)
        model.load_state_dict(ck['raw_model_state'], strict=True)
        ema.load_state_dict(ck['model_state'], strict=True)
        opt.load_state_dict(ck['optim_state'])
        sched.load_state_dict(ck['sched_state'])
        begin = ck['epoch'] + 1
        log(f'===> resume tu epoch {ck["epoch"]}')

    decay = float(L.ema_decay)
    for epoch in range(begin, cfg.TRAIN.end_epoch + 1):
        model.train()
        run, nb = {'loss_mm': 0.0, 'xy_err_px': 0.0, 'bone_mm': 0.0, 'angle': 0.0}, 0
        pbar = tqdm(train_loader, desc=f'Epoch {epoch}')
        for i, (inputs, targets, meta) in enumerate(pbar):
            img = inputs['img'].to(DEVICE).float()
            B = img.shape[0]
            j2 = inputs['joints'].to(DEVICE).float()[..., :2]                     # GT + PoseFix [-1,1]
            jm = inputs['joints_mask'].to(DEVICE).float().reshape(B, 17)
            cam = inputs['cam_bbox'].to(DEVICE).float()
            gt3d = targets['orig_joint_cam'].to(DEVICE).float()                   # (B,17,3) met, dinh nghia mesh
            valid = meta['is_valid_fit'].to(DEVICE).float().reshape(B)
            drop = (torch.rand_like(jm) < float(L.input_drop_prob)).float() * (jm > 0).float()
            jm_in = jm * (1 - drop)                                               # detector "bo sot"
            NC = int(L.num_ctx)                                                   # >0: them diem day lam ngu canh
            xy, log_b, f8, fj = refiner_inputs(refiner, img, j2, jm_in, 17 + NC)
            if float(L.train_xy_noise) > 0:                                       # b: o luoi 64 -> [-1,1] = / 32
                b = torch.exp(log_b).clamp(max=8.0)                               # chan 8 o (32 px): log_b toi da 6
                xy = xy + torch.randn_like(xy) * (b / 32.0 * float(L.train_xy_noise)).unsqueeze(-1)
            cond7 = ImgLifter.make_cond7(xy, log_b, lifter_det(jm_in, NC), cam)
            loss, _ = model.loss(gt3d, cond7, fj, f8, valid,
                                 w_bone=float(L.w_bone), w_angle=float(L.w_angle))   # 0 = tat (chi MPJPE)
            opt.zero_grad()
            loss.backward()
            opt.step()
            with torch.no_grad():
                for pe, pm in zip(ema.parameters(), model.parameters()):
                    pe.mul_(decay).add_(pm.detach(), alpha=1 - decay)
                for be, bm in zip(ema.buffers(), model.buffers()):
                    be.copy_(bm)
                g2 = targets['orig_joint_img'].to(DEVICE).float()[..., :2]
                g2 = g2 / torch.tensor([cfg.output_hm_shape[2], cfg.output_hm_shape[1]], device=DEVICE) * 2 - 1
                gv = meta['orig_joint_trunc'].to(DEVICE).float().reshape(B, 17)
                xy_err = (torch.norm(xy[:, :17] - g2, dim=-1) * gv).sum() / gv.sum().clamp_min(1) * 128.0
            tm = model.last_terms                                                 # tung thanh phan, chua nhan trong so
            run['loss_mm'] += tm['mpjpe'] * 1000
            run['bone_mm'] += tm.get('bone', 0.0) * 1000
            run['angle'] += tm.get('angle', 0.0)
            run['xy_err_px'] += float(xy_err)
            nb += 1
            if i % cfg.TRAIN.print_freq == 0:
                pbar.set_postfix_str(f'MPJPE x0 {run["loss_mm"] / nb:.1f} mm | loi 2D Refiner (+nhieu) '
                                     f'{run["xy_err_px"] / nb:.1f} px')
        sched.step()
        log(f'Epoch {epoch} | lr {opt.param_groups[0]["lr"]:.2e} | MPJPE x0 (train, moi t) '
            f'{run["loss_mm"] / max(nb, 1):.2f} mm | loi 2D Refiner {run["xy_err_px"] / max(nb, 1):.2f} px'
            + (f' | bone {run["bone_mm"] / max(nb, 1):.2f} mm | angle (cos^2) {run["angle"] / max(nb, 1):.4f}'
               if float(L.w_bone) > 0 or float(L.w_angle) > 0 else ''))

        ck = {'epoch': epoch, 'model_state': ema.state_dict(), 'imglift_arch': imglift_arch(),
              'refiner_ckpt': L.refiner_ckpt, 'cfg_file': args.cfg}
        torch.save(ck, os.path.join(cfg.checkpoint_dir, f'epoch_{epoch:02d}.pth.tar'))
        torch.save({**ck, 'raw_model_state': model.state_dict(), 'optim_state': opt.state_dict(),
                    'sched_state': sched.state_dict()}, last_path)

        for name, (loader, reg) in eval_loaders.items():
            r = eval_img_lifter(ema, refiner, loader, DEVICE, reg, K=int(L.eval_K), S=int(L.eval_S),
                                eta=float(L.eta), jpma_max_b=float(L.jpma_max_b))
            log(format_eval(f'{name} ep{epoch} (danh gia nhanh, EMA)', r))
    log('Xong. Bao ket qua bang main/eval_img_lifter.py voi checkpoint/last.pth.tar (epoch cuoi).')


if __name__ == '__main__':
    main()
