"""
main/train_selector.py -- Train Selector chon gia thuyet DiffLifter theo nhom khop bang anh (lib/models/selector.py).
Chi Selector duoc train; backbone SPIN va DiffLifter dong bang; KHONG can Teacher.
Du lieu train: DATASET.train_list (giao thuc JOTR: Human36M + MuCo + MSCOCO + CrowdPose, khong 3DPW), lay mau theo
TRAIN.dataset_ratio / samples_per_epoch, batch TRAIN.batch_size. GT = H36M-17 tu mesh (orig_joint_cam).
Moi epoch: log loss / top1 / loi nhom; danh gia nhanh pose MPJPE17 tren SELECTOR.val_set (SELECTOR.val_max_batches).
Luu checkpoint/last.pth.tar moi epoch (+ best.pth.tar theo pose MPJPE17 tren val_set, chi de tham khao).

Chay:
  python main/train_selector.py --cfg config/train_selector_l2last.yml --exp_name selector_l2last --gpu 0
Danh gia (co Teacher giai ma mesh):
  python main/eval_selector.py --cfg config/train_selector_l2last.yml \
      --ckpt experiment/selector_l2last/checkpoint/last.pth.tar --teacher_ckpt <teacher best.pth.tar> --gpu 0
"""
import os, sys
sys.path.append('./lib')
sys.path.append('./')
os.environ["OMP_NUM_THREADS"] = "1"
import argparse
import json
import random

parser = argparse.ArgumentParser(description='Train Selector gia thuyet DiffLifter')
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

import __init_path  # noqa: F401
from core.base import get_dataloader
from models.selector import SelectorNet, build_options, group_errors, soft_ce, expected_err, GROUP_NAMES
from utils.jotr_dataset import get_test_dataset
from utils.selector_tools import build_frozen_parts, make_candidates, gt_and_valid, evaluate, format_eval

random.seed(args.seed)
np.random.seed(args.seed)
torch.manual_seed(args.seed)
DEVICE = torch.device('cuda')
S = cfg.SELECTOR


def main():
    log_path = os.path.join(cfg.output_dir, 'log.txt')

    def log(msg):
        print(msg)
        with open(log_path, 'a', encoding='utf-8') as f:
            f.write(msg + '\n')

    log(f'cfg={args.cfg} | SELECTOR={json.dumps(dict(S), default=str)}')
    backbone, lifter = build_frozen_parts(DEVICE)
    model = SelectorNet(dim=S.dim, heads=S.heads, enc_layers=S.enc_layers, dec_layers=S.dec_layers,
                        use_flip=S.use_flip, dropout=S.dropout, base_prior_init=S.base_prior_init).to(DEVICE)
    log(f'Selector: {sum(p.numel() for p in model.parameters()) / 1e6:.2f}M tham so')
    opt = torch.optim.AdamW(model.parameters(), lr=S.lr, weight_decay=S.weight_decay)
    sched = torch.optim.lr_scheduler.MultiStepLR(opt, milestones=list(S.lr_step), gamma=S.lr_factor)

    _, train_loader = get_dataloader(args, cfg.DATASET.train_list, is_train=True)
    val_ds = get_test_dataset(S.val_set, None) if S.val_set else None
    val_loader = None
    if val_ds is not None:
        n = len(val_ds)
        if S.val_max_samples and S.val_max_samples < n:
            idx = np.sort(np.random.default_rng(0).choice(n, S.val_max_samples, replace=False)).tolist()
            val_loader = DataLoader(Subset(val_ds, idx), batch_size=cfg.TEST.batch_size, shuffle=False,
                                    num_workers=cfg.DATASET.workers)
        else:
            val_loader = DataLoader(val_ds, batch_size=cfg.TEST.batch_size, shuffle=False,
                                    num_workers=cfg.DATASET.workers)

    begin, best = 1, float('inf')
    last_path = os.path.join(cfg.checkpoint_dir, 'last.pth.tar')
    if args.resume_training:
        ck = torch.load(last_path, map_location='cpu', weights_only=False)
        model.load_state_dict(ck['selector_state'])
        opt.load_state_dict(ck['optim_state'])
        sched.load_state_dict(ck['sched_state'])
        begin, best = ck['epoch'] + 1, ck.get('best', best)
        log(f'===> resume tu epoch {ck["epoch"]}')

    for epoch in range(begin, S.epochs + 1):
        model.train()
        run = {'loss': 0.0, 'n': 0}
        g_hit, g_n, g_sel, g_base, g_ora, g_keep = (np.zeros(5) for _ in range(6))
        pbar = tqdm(train_loader, desc=f'Epoch {epoch}')
        for i, (inputs, targets, meta) in enumerate(pbar):
            img = inputs['img'].to(DEVICE).float()
            j2 = inputs['joints'].to(DEVICE).float()
            jm = inputs['joints_mask'].to(DEVICE).float()
            cam = inputs['cam_bbox'].to(DEVICE).float()
            gt, valid = gt_and_valid(targets, meta, DEVICE)
            with torch.no_grad():
                feat = backbone(img)[0]
                cands = make_candidates(lifter, j2, jm, cam, random_noise=True)
                groups = build_options(cands, S.use_flip)
                errs = group_errors(groups, gt, valid)
            logits = model(feat, j2, jm, groups)
            loss_ce = sum(soft_ce(lg, e, v, S.tau) for lg, (e, v) in zip(logits, errs)) / len(logits)
            loss_ee = sum(expected_err(lg, e, v) for lg, (e, v) in zip(logits, errs)) / len(logits)
            loss = S.ce_w * loss_ce + S.exp_err_w * loss_ee
            opt.zero_grad()
            loss.backward()
            opt.step()

            run['loss'] += loss.item() * img.shape[0]
            run['n'] += img.shape[0]
            with torch.no_grad():
                for g, (lg, (e, v)) in enumerate(zip(logits, errs)):
                    ar = torch.arange(e.shape[0], device=DEVICE)
                    cs, co = lg.argmax(-1), e.argmin(-1)
                    vf = v.float()
                    g_hit[g] += float(((cs == co).float() * vf).sum())
                    g_n[g] += float(vf.sum())
                    g_sel[g] += float((e[ar, cs] * vf).sum()) * 1000
                    g_base[g] += float((e[:, 0] * vf).sum()) * 1000
                    g_ora[g] += float((e[ar, co] * vf).sum()) * 1000
                    g_keep[g] += float(((cs == 0).float() * vf).sum())
            if i % cfg.TRAIN.print_freq == 0:
                tot = g_n.sum().clip(1)
                pbar.set_postfix_str(f'loss {run["loss"] / run["n"]:.3f} | nhom L0/sel/oracle '
                                     f'{g_base.sum() / tot:.1f}/{g_sel.sum() / tot:.1f}/{g_ora.sum() / tot:.1f} mm '
                                     f'| top1 {100 * g_hit.sum() / tot:.1f}% | giu L0 {100 * g_keep.sum() / tot:.0f}%')
        sched.step()
        log(f'Epoch {epoch} | lr {opt.param_groups[0]["lr"]:.2e} | loss {run["loss"] / max(run["n"], 1):.4f}')
        log('  train tung nhom (mm, L0 / Selector / Oracle, top1, giu L0): ' + ' | '.join(
            f'{GROUP_NAMES[g]} {g_base[g] / max(g_n[g], 1):.0f}/{g_sel[g] / max(g_n[g], 1):.0f}/'
            f'{g_ora[g] / max(g_n[g], 1):.0f} ({100 * g_hit[g] / max(g_n[g], 1):.0f}%, '
            f'{100 * g_keep[g] / max(g_n[g], 1):.0f}%)' for g in range(5)))

        ck = {'epoch': epoch, 'selector_state': model.state_dict(), 'optim_state': opt.state_dict(),
              'sched_state': sched.state_dict(), 'selector_cfg': dict(S), 'best': best,
              'difflift_checkpoint': cfg.MODEL.difflift_checkpoint}
        if val_loader is not None:
            r = evaluate(model, backbone, lifter, val_loader, val_ds, DEVICE)
            log(format_eval(S.val_set, r))
            if r['pose_mpjpe17']['Selector'] < best:
                best = r['pose_mpjpe17']['Selector']
                ck['best'] = best
                torch.save(ck, os.path.join(cfg.checkpoint_dir, 'best.pth.tar'))
                log(f'  -> best (pose MPJPE17 {S.val_set} {best:.2f})')
        torch.save(ck, last_path)
    log(f'Xong. Checkpoint: {cfg.checkpoint_dir}')


if __name__ == '__main__':
    main()
