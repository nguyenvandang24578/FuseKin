"""
main/train_vit_lifter.py (2026-11-18, nguoi dung duyet) -- Train VitLifter (lib/models/vit_lifter.py): diffusion
2D ViTPose + anh -> 3D, KHONG Refiner. Rut gon tu main/train_img_lifter.py (bo Refiner / KD / SD), KHONG sua file do.

Moi batch: anh crop + 2D ViTPose (x, y, conf) -> ResNet-50 SPIN dong bang (f8, feat_j) -> VitLifter hoc khu nhieu x0.
Muc tieu = orig_joint_cam (H36M-17 tu mesh GT, met, root-relative). Dau vao chinh cua diffusion = GT 3D cong nhieu.
  - IMGLIFT.input_drop_prob: bo ngau nhien khop (det = 0, conf = 0, xy = 0) giong detector bo sot.
EMA (IMGLIFT.ema_decay): 'model_state' = trong so EMA (dung danh gia); 'raw_model_state' de resume.
Moi epoch: danh gia nhanh tren IMGLIFT.eval_sets (OC / PC CHI sequence test vi train co 3dpw-train) -> chi LOG.
Bao ket qua theo EPOCH CUOI (last.pth.tar) bang main/eval_vit_lifter.py.

Chay:
  python main/train_vit_lifter.py --cfg config/train_vit_lifter_3dpw.yaml --exp_name vit_lifter_3dpw --gpu 0
"""
import os, sys
sys.path.append('./lib')
sys.path.append('./')
os.environ["OMP_NUM_THREADS"] = "1"
import argparse
import json
import random

parser = argparse.ArgumentParser(description='Train VitLifter')
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
cfg.DATASET.pw3d_test_gt2d = True          # tap danh gia tra GT 2D -> log loi 2D ViTPose + lech 2D (reproj)
assert cfg.DATASET.pw3d_2d_source == 'halpe26', 'VitLifter: 2D 3DPW phai la file Halpe-26 ViTPose (pw3d_2d_source halpe26)'
assert not cfg.DATASET.get('input_2d_gt', False), 'VitLifter dung 2D ViTPose that, khong oracle GT 2D'

import __init_path  # noqa: F401
from core.base import get_dataloader
from utils.jotr_dataset import get_test_dataset
from models.vit_lifter import vit_inputs
from utils.vit_lifter_tools import vitlift_arch, build_vit_lifter, build_encoder, eval_vit_lifter, format_vit_eval

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

    log(f'cfg={args.cfg} | IMGLIFT={json.dumps(dict(L), default=str)} | DATASET train_list={cfg.DATASET.train_list} '
        f'halpe test {cfg.DATASET.pw3d_halpe_suffix} train {cfg.DATASET.pw3d_halpe_train_suffix} | TRAIN lr={cfg.TRAIN.lr} '
        f'lr_step={cfg.TRAIN.lr_step} end_epoch={cfg.TRAIN.end_epoch} batch={cfg.TRAIN.batch_size}')
    enc = build_encoder(L.vit_enc_pretrained, DEVICE)
    model = build_vit_lifter().to(DEVICE)
    ema = build_vit_lifter().to(DEVICE)
    ema.load_state_dict(model.state_dict())
    ema.eval()
    for p in ema.parameters():
        p.requires_grad = False
    log(f'===> VitLifter {sum(p.numel() for p in model.parameters()) / 1e6:.2f}M tham so | arch {vitlift_arch()} '
        f'| encoder ResNet-50 {L.vit_enc_pretrained} DONG BANG')

    opt = torch.optim.AdamW(model.parameters(), lr=cfg.TRAIN.lr, weight_decay=float(L.weight_decay))
    sched = torch.optim.lr_scheduler.MultiStepLR(opt, milestones=list(cfg.TRAIN.lr_step), gamma=cfg.TRAIN.lr_factor)

    _, train_loader = get_dataloader(args, cfg.DATASET.train_list, is_train=True)
    eval_loaders = {}
    for name in list(L.eval_sets or []):
        # train co 3dpw-train -> OC / PC chi sequence test (tranh ro ri)
        cfg.DATASET.pw3d_ocpc_test_only = name in ('3dpw-oc', '3dpw-pc')
        ds = get_test_dataset(name, None)
        reg = ds.h36m_joint_regressor
        if L.eval_max_samples and L.eval_max_samples < len(ds):
            ds = Subset(ds, list(range(int(L.eval_max_samples))))
        eval_loaders[name] = (DataLoader(ds, batch_size=cfg.TEST.batch_size, shuffle=False,
                                         num_workers=cfg.DATASET.workers), reg)
    cfg.DATASET.pw3d_ocpc_test_only = False

    begin = 1
    last_path = os.path.join(cfg.checkpoint_dir, 'last.pth.tar')
    if args.resume_training:
        ck = torch.load(last_path, map_location='cpu', weights_only=False)
        if dict(ck['vitlift_arch']) != vitlift_arch():
            raise ValueError('YAML khac kien truc checkpoint; hay tao run moi')
        model.load_state_dict(ck['raw_model_state'], strict=True)
        ema.load_state_dict(ck['model_state'], strict=True)
        opt.load_state_dict(ck['optim_state'])
        sched.load_state_dict(ck['sched_state'])
        begin = int(ck['epoch']) + 1
        log(f'===> Resume tu {last_path} (epoch {ck["epoch"]})')

    decay = float(L.ema_decay)
    eps = float(L.vit_logb_eps)
    for epoch in range(begin, cfg.TRAIN.end_epoch + 1):
        model.train()
        run, nb = {'loss_mm': 0.0, 'xy_err_px': 0.0, 'bone': 0.0, 'angle': 0.0, 'total': 0.0}, 0
        pbar = tqdm(train_loader, desc=f'Epoch {epoch}')
        for i, (inputs, targets, meta) in enumerate(pbar):
            img = inputs['img'].to(DEVICE).float()
            B = img.shape[0]
            joints = inputs['joints'].to(DEVICE).float()                         # (B,17,3) [x, y, conf] ViTPose
            jm = inputs['joints_mask'].to(DEVICE).float().reshape(B, 17)
            cam = inputs['cam_bbox'].to(DEVICE).float()
            gt3d = targets['orig_joint_cam'].to(DEVICE).float()                   # (B,17,3) met, dinh nghia mesh
            valid = meta['is_valid_fit'].to(DEVICE).float().reshape(B)
            drop = (torch.rand_like(jm) < float(L.input_drop_prob)).float() * (jm > 0).float()
            jm_in = jm * (1 - drop)                                               # detector "bo sot"
            cond7, fj, f8, xy, _ = vit_inputs(enc, img, joints, jm_in, cam, eps)
            loss, _ = model.loss(gt3d, cond7, fj, f8, valid, w_bone=float(L.w_bone), w_angle=float(L.w_angle))
            opt.zero_grad()
            loss.backward()
            opt.step()
            with torch.no_grad():
                for pe, pm in zip(ema.parameters(), model.parameters()):
                    pe.mul_(decay).add_(pm.detach(), alpha=1 - decay)
                for be, bm in zip(ema.buffers(), model.buffers()):
                    be.copy_(bm)
                # loi 2D ViTPose (dau vao) so GT 2D mesh, chi khop co mat -> theo doi chat luong 2D train
                g2 = targets['orig_joint_img'].to(DEVICE).float()[..., :2]
                g2 = g2 / torch.tensor([cfg.output_hm_shape[2], cfg.output_hm_shape[1]], device=DEVICE) * 2 - 1
                gv = meta['orig_joint_trunc'].to(DEVICE).float().reshape(B, 17) * (jm_in > 0).float()
                xy_err = (torch.norm(xy - g2, dim=-1) * gv).sum() / gv.sum().clamp_min(1) * 128.0
            tm = model.last_terms
            run['loss_mm'] += tm['mpjpe'] * 1000
            run['bone'] += tm.get('bone', 0.0)
            run['angle'] += tm.get('angle', 0.0)
            run['total'] += tm['total']
            run['xy_err_px'] += float(xy_err)
            nb += 1
            if i % cfg.TRAIN.print_freq == 0:
                pbar.set_postfix_str(f'MPJPE x0 {run["loss_mm"] / nb:.1f} mm | loi 2D ViTPose {run["xy_err_px"] / nb:.1f} px')
        sched.step()
        c = max(nb, 1)
        log(f'Epoch {epoch} | lr {opt.param_groups[0]["lr"]:.2e} | MPJPE x0 (train, moi t) {run["loss_mm"] / c:.2f} mm'
            f' | loi 2D ViTPose {run["xy_err_px"] / c:.2f} px | bone {run["bone"] / c * 1000:.2f} mm'
            f' | angle (cos^2) {run["angle"] / c:.4f} | total {run["total"] / c:.5f}')

        ck = {'epoch': epoch, 'model_state': ema.state_dict(), 'vitlift_arch': vitlift_arch(),
              'vit_enc_pretrained': L.vit_enc_pretrained, 'vit_logb_eps': eps, 'cfg_file': args.cfg}
        torch.save(ck, os.path.join(cfg.checkpoint_dir, f'epoch_{epoch:02d}.pth.tar'))
        torch.save({**ck, 'raw_model_state': model.state_dict(), 'optim_state': opt.state_dict(),
                    'sched_state': sched.state_dict()}, last_path)

        for name, (loader, reg) in eval_loaders.items():
            r = eval_vit_lifter(ema, enc, loader, DEVICE, reg, K=int(L.eval_K), S=int(L.eval_S), eta=float(L.eta),
                                logb_eps=eps)
            label = name + (' (chi seq test)' if name in ('3dpw-oc', '3dpw-pc') else '')
            log(format_vit_eval(f'{label} ep{epoch} (danh gia nhanh, EMA)', r))
    log('Xong. Bao ket qua bang main/eval_vit_lifter.py voi checkpoint/last.pth.tar (epoch cuoi).')


if __name__ == '__main__':
    main()
