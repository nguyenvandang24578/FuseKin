"""
main/train_refiner2d.py -- Train Refiner 2D (lib/models/refiner2d.py) doc lap, KHONG dung lifter / Teacher / Student.

Du lieu: DATASET.train_list (giao thuc JOTR: Human36M + MuCo + MSCOCO + CrowdPose, khong 3DPW), mau fusekin_unified
voi 2D dau vao = GT + PoseFix kieu JOTR (*_2d_source 'gt_posefix'). Muc tieu theo REFINE2D.target:
  'clean' : GT 2D annotation TRUOC PoseFix (targets['joint_img_clean'], can DATASET.return_clean_2d), mask = khop GT
            trong khung (= joints_mask)
  'mesh'  : 2D H36M-17 tinh tu mesh GT (H36M regressor x mesh, chieu vao crop; targets['orig_joint_img']), mask =
            meta orig_joint_trunc -> CUNG dinh nghia khop voi GT 3D cua benchmark (2026-11-03)
Luc train bo ngau nhien khop dau vao (REFINE2D.input_drop_prob) de Refiner hoc dien khop detector bo sot.
Lay mau / batch / lich lr theo cfg.TRAIN (base.get_dataloader). lr backbone = TRAIN.lr * REFINE2D.bb_lr_scale.
Moi epoch: danh gia nhanh 2D tren REFINE2D.eval_sets (OpenPose / HHRNet nhu benchmark; chi LOG, khong chon ckpt).
Luu checkpoint/last.pth.tar (day du, de resume) + checkpoint/epoch_XX.pth.tar (chi trong so) moi epoch.
Bao ket qua theo EPOCH CUOI voi lich co dinh (3dpw-val chua 2 sequence Crowd -> khong chon theo val / test).

Chay:
  python main/train_refiner2d.py --cfg config/train_refiner2d_jotr.yaml --exp_name refiner2d_jotr --gpu 0
"""
import os, sys
sys.path.append('./lib')
sys.path.append('./')
os.environ["OMP_NUM_THREADS"] = "1"
import argparse
import json
import random

parser = argparse.ArgumentParser(description='Train Refiner 2D')
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
TARGET = str(cfg.REFINE2D.get('target', 'clean'))
assert TARGET in ('clean', 'mesh'), f"REFINE2D.target khong hop le: {TARGET} (chon 'clean' | 'mesh')"
if TARGET == 'clean':
    assert cfg.DATASET.get('return_clean_2d', False), "target 'clean' can DATASET.return_clean_2d: True"
assert cfg.DATASET.fusekin_unified, 'Refiner 2D dung mau fusekin_unified (anh + 2D + GT)'
cfg.DATASET.pw3d_test_gt2d = True          # tap danh gia 3DPW tra GT 2D

import __init_path  # noqa: F401
from core.base import get_dataloader
from models.refiner2d import refiner_losses
from utils.jotr_dataset import get_test_dataset
from utils.refiner2d_tools import build_refiner, refine2d_arch, eval_refiner, format_result

random.seed(args.seed)
np.random.seed(args.seed)
torch.manual_seed(args.seed)
DEVICE = torch.device('cuda')
R = cfg.REFINE2D


def main():
    log_path = os.path.join(cfg.output_dir, 'log.txt')

    def log(msg):
        print(msg)
        with open(log_path, 'a', encoding='utf-8') as f:
            f.write(msg + '\n')

    log(f'cfg={args.cfg} | REFINE2D={json.dumps(dict(R), default=str)} | TRAIN lr={cfg.TRAIN.lr} '
        f'lr_step={cfg.TRAIN.lr_step} end_epoch={cfg.TRAIN.end_epoch} batch={cfg.TRAIN.batch_size}')
    model = build_refiner(pretrained=R.backbone_pretrained).to(DEVICE)
    log(f'Refiner 2D: {sum(p.numel() for p in model.parameters()) / 1e6:.2f}M tham so')
    opt = torch.optim.AdamW([{'params': model.head_parameters(), 'lr': cfg.TRAIN.lr},
                             {'params': model.backbone_parameters(), 'lr': cfg.TRAIN.lr * float(R.bb_lr_scale)}],
                            weight_decay=float(R.weight_decay))
    sched = torch.optim.lr_scheduler.MultiStepLR(opt, milestones=list(cfg.TRAIN.lr_step), gamma=cfg.TRAIN.lr_factor)

    _, train_loader = get_dataloader(args, cfg.DATASET.train_list, is_train=True)
    eval_loaders = {}
    for name in list(R.eval_sets or []):
        ds = get_test_dataset(name, None)
        n = len(ds)
        if R.eval_max_samples and R.eval_max_samples < n:
            ds = Subset(ds, list(range(int(R.eval_max_samples))))     # cac mau dau, co dinh giua cac epoch
        eval_loaders[name] = DataLoader(ds, batch_size=cfg.TEST.batch_size, shuffle=False,
                                        num_workers=cfg.DATASET.workers)

    begin = 1
    last_path = os.path.join(cfg.checkpoint_dir, 'last.pth.tar')
    if args.resume_training:
        ck = torch.load(last_path, map_location='cpu', weights_only=False)
        model.load_state_dict(ck['model_state'], strict=True)
        opt.load_state_dict(ck['optim_state'])
        sched.load_state_dict(ck['sched_state'])
        begin = ck['epoch'] + 1
        log(f'===> resume tu epoch {ck["epoch"]}')

    hm = torch.tensor([cfg.output_hm_shape[2], cfg.output_hm_shape[1]], dtype=torch.float32, device=DEVICE)
    for epoch in range(begin, cfg.TRAIN.end_epoch + 1):
        model.train()
        run = {k: 0.0 for k in ('loss', 'coord', 'heat', 'nll', 'err_px', 'in_err_px')}
        nb = 0
        pbar = tqdm(train_loader, desc=f'Epoch {epoch}')
        for i, (inputs, targets, meta) in enumerate(pbar):
            img = inputs['img'].to(DEVICE).float()
            B = img.shape[0]
            j2 = inputs['joints'].to(DEVICE).float()[..., :2]                     # GT + PoseFix, [-1,1]
            jm = inputs['joints_mask'].to(DEVICE).float().reshape(B, 17)          # = khop GT (annotation) trong khung
            if TARGET == 'mesh':
                # 2D H36M-17 tu mesh GT (cung dinh nghia khop GT 3D benchmark); mask = khop do trong khung crop
                tgt = targets['orig_joint_img'].to(DEVICE).float()[..., :2] / hm * 2 - 1
                valid = (meta['orig_joint_trunc'].to(DEVICE).float().reshape(B, 17) > 0).float()
            else:
                tgt = targets['joint_img_clean'].to(DEVICE).float() / hm * 2 - 1  # GT 2D annotation sach [-1,1]
                valid = (jm > 0).float()
            drop = (torch.rand_like(jm) < float(R.input_drop_prob)).float() * (jm > 0).float()
            jm_in = jm * (1 - drop)                                               # khop bi bo: van giam sat
            out = model(img, j2 * (jm_in > 0).unsqueeze(-1), jm_in.unsqueeze(-1))
            loss, parts = refiner_losses(out, tgt, valid, sigma=float(R.heat_sigma), w_coord=float(R.w_coord),
                                         w_heat=float(R.w_heat), w_nll=float(R.w_nll))
            opt.zero_grad()
            loss.backward()
            opt.step()

            with torch.no_grad():
                keep = valid * (jm_in > 0).float()                                # loi dau vao so voi CUNG muc tieu
                in_err = (torch.norm(j2 - tgt, dim=-1) * keep).sum() / keep.sum().clamp_min(1) * 128.0
            run['loss'] += float(loss)
            for k in ('coord', 'heat', 'nll', 'err_px'):
                run[k] += parts[k]
            run['in_err_px'] += float(in_err)
            nb += 1
            if i % cfg.TRAIN.print_freq == 0:
                pbar.set_postfix_str(f'loss {run["loss"] / nb:.3f} | loi 2D vao / ra '
                                     f'{run["in_err_px"] / nb:.1f} / {run["err_px"] / nb:.1f} px (crop 256)')
        sched.step()
        log(f'Epoch {epoch} | lr head {opt.param_groups[0]["lr"]:.2e} bb {opt.param_groups[1]["lr"]:.2e} | '
            + ' | '.join(f'{k} {v / max(nb, 1):.4f}' for k, v in run.items()))

        ck = {'epoch': epoch, 'model_state': model.state_dict(), 'refine2d_arch': refine2d_arch(),
              'cfg_file': args.cfg}
        torch.save(ck, os.path.join(cfg.checkpoint_dir, f'epoch_{epoch:02d}.pth.tar'))
        torch.save({**ck, 'optim_state': opt.state_dict(), 'sched_state': sched.state_dict()}, last_path)

        for name, loader in eval_loaders.items():
            r = eval_refiner(model, loader, DEVICE, far_ratio=float(R.far_ratio))
            log(format_result(f'{name} ep{epoch} (danh gia nhanh, {len(loader.dataset)} mau)', r))
    log('Xong. Bao ket qua bang main/eval_refiner2d.py voi checkpoint/last.pth.tar (epoch cuoi).')


if __name__ == '__main__':
    main()
