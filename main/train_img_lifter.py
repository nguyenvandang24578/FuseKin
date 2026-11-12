"""
main/train_img_lifter.py -- Train ImgLifter (lib/models/img_lifter.py): diffusion 2D -> 3D CO ANH, thay DiffLifter cu.

Moi batch: anh + 2D (GT + PoseFix kieu JOTR, giong luc train Refiner) -> Refiner 2D DONG BANG -> xy, log_b, f8, feat_j
-> ImgLifter hoc khu nhieu x0. Muc tieu = orig_joint_cam cua mau fusekin_unified = H36M-17 tinh tu mesh (met,
root-relative) -> cung dinh nghia khop mesh voi GT 3DPW. Tap train theo DATASET.train_list; co the them 3dpw-train.
  - IMGLIFT.input_drop_prob: bo ngau nhien khop dau vao Refiner (det = 0) giong detector bo sot luc test.
  - IMGLIFT.train_xy_noise: cong nhieu Gauss std = train_xy_noise x b vao xy Refiner (Refiner chay tren anh no
    da hoc -> loi nho hon test; nhieu bu khoang cach).
EMA (IMGLIFT.ema_decay): 'model_state' trong checkpoint = trong so EMA (dung danh gia); 'raw_model_state' de resume.
IMGLIFT.refiner_finetune (2026-11-12, mac dinh TAT): mo khoa Refiner, train cung lifter (loss 3D qua f8 + loss 2D Refiner);
  Refiner EMA luu moi epoch thanh checkpoint/refiner_epoch_XX.pth.tar va refiner_last.pth.tar, ck['refiner_ckpt'] tro toi do.
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
                                    refiner_inputs_train, lifter_det, eval_img_lifter, format_eval)
from models.refiner2d import refiner_losses

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
    FT_REF = bool(L.get('refiner_finetune', False))
    ema_ref, ref_arch, NUM_DENSE_REF = None, None, 0
    if FT_REF:
        # Mo khoa Refiner: train() giu BN backbone o eval (bb_bn_eval trong ckpt), head BN train binh thuong
        ref_arch = torch.load(L.refiner_ckpt, map_location='cpu', weights_only=False)['refine2d_arch']
        NUM_DENSE_REF = int(ref_arch.get('num_dense', 0))
        for p in refiner.parameters():
            p.requires_grad = True
        refiner.train()
        ema_ref = copy.deepcopy(refiner).eval()
        for p in ema_ref.parameters():
            p.requires_grad = False
        log(f'===> Fine-tune Refiner cung lifter: {sum(p.numel() for p in refiner.parameters()) / 1e6:.2f}M tham so, '
            f'arch {ref_arch}, loss 2D x {float(L.refiner_loss_w)}')
    model = build_img_lifter().to(DEVICE)
    ema = copy.deepcopy(model).eval()
    for p in ema.parameters():
        p.requires_grad = False
    log(f'ImgLifter: {sum(p.numel() for p in model.parameters()) / 1e6:.2f}M tham so')
    if FT_REF:
        # Nhom tham so: lifter (lr TRAIN.lr) | head Refiner | backbone Refiner (lr nho, giu dac trung pretrain)
        lr_h = cfg.TRAIN.lr * float(L.refiner_lr_scale)
        lr_b = lr_h * float(cfg.REFINE2D.bb_lr_scale)
        wd_r = float(cfg.REFINE2D.weight_decay)
        opt = torch.optim.AdamW([{'params': list(model.parameters())},
                                 {'params': refiner.head_parameters(), 'lr': lr_h, 'weight_decay': wd_r},
                                 {'params': refiner.backbone_parameters(), 'lr': lr_b, 'weight_decay': wd_r}],
                                lr=cfg.TRAIN.lr, weight_decay=float(L.weight_decay))
        log(f'===> lr lifter {cfg.TRAIN.lr:.1e} | head Refiner {lr_h:.1e} | backbone Refiner {lr_b:.1e}')
    else:
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
        # Cac co moi khong doi shape tham so: phai kiem tra metadata, strict=True chua du.
        saved_arch = dict(ck['imglift_arch'])
        for key, default in (('num_ctx', 0), ('markov', False), ('markov_gate_b', False),
                             ('markov_iters', 0), ('markov_gate_positive', False), ('bone_paper', False),
                             ('markov_msg_norm', False)):
            saved_arch.setdefault(key, default)
        if saved_arch != imglift_arch():
            raise ValueError('YAML khac kien truc/quy uoc Markov cua checkpoint; hay tao run moi')
        model.load_state_dict(ck['raw_model_state'], strict=True)
        ema.load_state_dict(ck['model_state'], strict=True)
        if FT_REF:
            assert 'refiner_raw_state' in ck, 'checkpoint khong co Refiner fine-tune -> hay tao run moi'
            refiner.load_state_dict(ck['refiner_raw_state'], strict=True)
            ema_ref.load_state_dict(ck['refiner_ema_state'], strict=True)
        opt.load_state_dict(ck['optim_state'])
        sched.load_state_dict(ck['sched_state'])
        begin = ck['epoch'] + 1
        log(f'===> resume tu epoch {ck["epoch"]}')

    decay = float(L.ema_decay)
    for epoch in range(begin, cfg.TRAIN.end_epoch + 1):
        model.train()
        if FT_REF:
            refiner.train()
        run, nb = {'loss_mm': 0.0, 'xy_err_px': 0.0, 'bone': 0.0, 'angle': 0.0, 'total': 0.0,
                   'ref_loss': 0.0, 'ref_px': 0.0}, 0
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
            if FT_REF:
                xy, log_b, f8, fj, ref_out = refiner_inputs_train(refiner, img, j2, jm_in, 17 + NC)
            else:
                xy, log_b, f8, fj = refiner_inputs(refiner, img, j2, jm_in, 17 + NC)
            if float(L.train_xy_noise) > 0:                                       # b: o luoi 64 -> [-1,1] = / 32
                b = torch.exp(log_b).clamp(max=8.0)                               # chan 8 o (32 px): log_b toi da 6
                xy = xy + torch.randn_like(xy) * (b / 32.0 * float(L.train_xy_noise)).unsqueeze(-1)
            cond7 = ImgLifter.make_cond7(xy, log_b, lifter_det(jm_in, NC), cam)
            loss, _ = model.loss(gt3d, cond7, fj, f8, valid,
                                 w_bone=float(L.w_bone), w_angle=float(L.w_angle))   # 0 = tat (chi MPJPE)
            if FT_REF:
                # Loss 2D Refiner giong train_refiner2d.py (target 'mesh'): 17 khop tu mesh GT (+ diem day);
                # khop bi bo o dau vao (input_drop) van duoc giam sat
                hm2 = torch.tensor([cfg.output_hm_shape[2], cfg.output_hm_shape[1]], device=DEVICE, dtype=torch.float32)
                rtg = targets['orig_joint_img'].to(DEVICE).float()[..., :2] / hm2 * 2 - 1
                rva = (meta['orig_joint_trunc'].to(DEVICE).float().reshape(B, 17) > 0).float()
                if NUM_DENSE_REF:
                    assert 'dense_img' in targets, 'refiner_finetune voi Refiner day can DATASET.dense2d_vertex_file'
                    rtg = torch.cat([rtg, targets['dense_img'].to(DEVICE).float() / hm2 * 2 - 1], 1)
                    rva = torch.cat([rva, (meta['dense_trunc'].to(DEVICE).float().reshape(B, -1) > 0).float()], 1)
                R = cfg.REFINE2D
                ref_loss, ref_parts = refiner_losses(ref_out, rtg, rva, sigma=float(R.heat_sigma),
                                                     w_coord=float(R.w_coord), w_heat=float(R.w_heat),
                                                     w_nll=float(R.w_nll))
                loss = loss + float(L.refiner_loss_w) * ref_loss
                run['ref_loss'] += float(ref_loss)
                run['ref_px'] += ref_parts['err_px']
            opt.zero_grad()
            loss.backward()
            opt.step()
            with torch.no_grad():
                for pe, pm in zip(ema.parameters(), model.parameters()):
                    pe.mul_(decay).add_(pm.detach(), alpha=1 - decay)
                for be, bm in zip(ema.buffers(), model.buffers()):
                    be.copy_(bm)
                if FT_REF:
                    for pe, pm in zip(ema_ref.parameters(), refiner.parameters()):
                        pe.mul_(decay).add_(pm.detach(), alpha=1 - decay)
                    for be, bm in zip(ema_ref.buffers(), refiner.buffers()):
                        be.copy_(bm)
                g2 = targets['orig_joint_img'].to(DEVICE).float()[..., :2]
                g2 = g2 / torch.tensor([cfg.output_hm_shape[2], cfg.output_hm_shape[1]], device=DEVICE) * 2 - 1
                gv = meta['orig_joint_trunc'].to(DEVICE).float().reshape(B, 17)
                xy_err = (torch.norm(xy[:, :17] - g2, dim=-1) * gv).sum() / gv.sum().clamp_min(1) * 128.0
            tm = model.last_terms                                                 # tung thanh phan, chua nhan trong so
            run['loss_mm'] += tm['mpjpe'] * 1000
            run['bone'] += tm.get('bone', 0.0)
            run['angle'] += tm.get('angle', 0.0)
            run['total'] += tm['total']
            run['xy_err_px'] += float(xy_err)
            nb += 1
            if i % cfg.TRAIN.print_freq == 0:
                pbar.set_postfix_str(f'MPJPE x0 {run["loss_mm"] / nb:.1f} mm | loi 2D Refiner (+nhieu) '
                                     f'{run["xy_err_px"] / nb:.1f} px')
        sched.step()
        count = max(nb, 1)
        bone_log = (f' | bone paper {run["bone"] / count:.4f} (khong don vi)' if model.bone_paper else
                    f' | bone {run["bone"] / count * 1000:.2f} mm')
        log(f'Epoch {epoch} | lr {opt.param_groups[0]["lr"]:.2e} | MPJPE x0 (train, moi t) '
            f'{run["loss_mm"] / count:.2f} mm | loi 2D Refiner {run["xy_err_px"] / count:.2f} px'
            + (bone_log + f' | angle (cos^2) {run["angle"] / count:.4f}'
               f' | w*bone {float(L.w_bone) * run["bone"] / count:.5f}'
               f' | w*angle {float(L.w_angle) * run["angle"] / count:.5f} | total {run["total"] / count:.5f}'
               if float(L.w_bone) > 0 or float(L.w_angle) > 0 else '')
            + (f' | Refiner: loss 2D {run["ref_loss"] / count:.4f}, loi {run["ref_px"] / count:.2f} px (target mesh)'
               if FT_REF else ''))

        ref_ep_path = ref_last_path = L.refiner_ckpt
        if FT_REF:
            # Refiner EMA, CUNG dinh dang train_refiner2d.py -> load_frozen_refiner(ck['refiner_ckpt']) nap duoc.
            # Duong dan tuong doi theo goc repo (giong IMGLIFT.refiner_ckpt trong YAML)
            ref_save = {'epoch': epoch, 'model_state': ema_ref.state_dict(), 'refine2d_arch': ref_arch,
                        'cfg_file': args.cfg, 'init_refiner_ckpt': L.refiner_ckpt}
            ref_ep_abs = os.path.join(cfg.checkpoint_dir, f'refiner_epoch_{epoch:02d}.pth.tar')
            ref_last_abs = os.path.join(cfg.checkpoint_dir, 'refiner_last.pth.tar')
            torch.save(ref_save, ref_ep_abs)
            torch.save(ref_save, ref_last_abs)
            ref_ep_path = os.path.relpath(ref_ep_abs, cfg.root_dir)
            ref_last_path = os.path.relpath(ref_last_abs, cfg.root_dir)
        ck = {'epoch': epoch, 'model_state': ema.state_dict(), 'imglift_arch': imglift_arch(),
              'refiner_ckpt': ref_ep_path, 'cfg_file': args.cfg}
        torch.save(ck, os.path.join(cfg.checkpoint_dir, f'epoch_{epoch:02d}.pth.tar'))
        last_ck = {**ck, 'refiner_ckpt': ref_last_path, 'raw_model_state': model.state_dict(),
                   'optim_state': opt.state_dict(), 'sched_state': sched.state_dict()}
        if FT_REF:
            last_ck['refiner_raw_state'] = refiner.state_dict()
            last_ck['refiner_ema_state'] = ema_ref.state_dict()
        torch.save(last_ck, last_path)

        for name, (loader, reg) in eval_loaders.items():
            r = eval_img_lifter(ema, ema_ref if FT_REF else refiner, loader, DEVICE, reg, K=int(L.eval_K), S=int(L.eval_S),
                                eta=float(L.eta), jpma_max_b=float(L.jpma_max_b))
            log(format_eval(f'{name} ep{epoch} (danh gia nhanh, EMA)', r))
    log('Xong. Bao ket qua bang main/eval_img_lifter.py voi checkpoint/last.pth.tar (epoch cuoi).')


if __name__ == '__main__':
    main()
