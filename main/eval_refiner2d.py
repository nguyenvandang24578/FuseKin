"""
main/eval_refiner2d.py -- Danh gia Refiner 2D tren 3DPW / OC / PC / Crowd voi 2D detector chuan benchmark
(OpenPose cho 3DPW / OC / PC, Higher-HRNet cho Crowd; DATASET.pw3d_2d_source = 'openpose').

  (a) Chat luong 2D tren 12 khop tay chan so voi GT 2D: op (detector) vs refine vs refine_fill
      -> loi / kich thuoc nguoi, % khop xa, % khop bo, % mau co khop xa, so khop sua / lam hong.
  (b) --lifter_ckpt: DiffLifter dong bang (S1 K1, seed 0) voi tung bien the 2D -> MPJPE17 (mm, root-relative,
      GT = H36M regressor x mesh GT). Mau lam moc: cung lifter voi 2D 'op'.
Dieu kien di tiep (ke hoach 2026-11-03): % khop xa tren PC / Crowd thap hon op VA lifter MPJPE17 voi refine /
refine_fill thap hon op tren 3DPW va Crowd.
OC / PC mac dinh TOAN BO (giao thuc JOTR); --ocpc_mode test = chi sequence test.

Chay:
  python main/eval_refiner2d.py --cfg config/train_refiner2d_jotr.yaml \
    --ckpt experiment/refiner2d_jotr/checkpoint/last.pth.tar \
    --lifter_ckpt experiment/finetune_difflifter_d512_e15_jotr_pf05_3dpw/best_epoch.bin \
    --sets 3dpw,3dpw-oc,3dpw-pc,3dpw-crowd --gpu 0
  (thu nhanh: --max_samples 2000)
"""
import os, sys
sys.path.append('./lib')
sys.path.append('./')
os.environ["OMP_NUM_THREADS"] = "1"
import argparse
import json

parser = argparse.ArgumentParser(description='Danh gia Refiner 2D')
parser.add_argument('--cfg', type=str, required=True, help='yml da dung de train Refiner')
parser.add_argument('--ckpt', type=str, required=True, help='checkpoint Refiner (last.pth.tar | epoch_XX.pth.tar)')
parser.add_argument('--lifter_ckpt', type=str, default='', help='DiffLifter .bin (rong = chi danh gia 2D)')
parser.add_argument('--sets', type=str, default='3dpw,3dpw-oc,3dpw-pc,3dpw-crowd')
parser.add_argument('--ocpc_mode', type=str, default='all', choices=['all', 'test'])
parser.add_argument('--fill_max_b', type=float, default=3.0,
                    help='refine_fill: dien khop tay chan detector bo sot khi b <= nguong (o luoi 64; 1 o = 4 px)')
parser.add_argument('--max_samples', type=int, default=0, help='0 = toan bo tap')
parser.add_argument('--batch_size', type=int, default=0, help='0 = cfg.TEST.batch_size')
parser.add_argument('--gpu', type=str, default='0')
args = parser.parse_args()
os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu

import torch
from torch.utils.data import DataLoader, Subset

from core.config import cfg, update_config
update_config(args.cfg)
cfg.DATASET.pw3d_test_gt2d = True          # dataset tra GT 2D (targets['orig_joint_img'], meta orig_joint_trunc)
assert cfg.DATASET.pw3d_2d_source == 'openpose', 'Benchmark: dau vao 3DPW la OpenPose / HHRNet (pw3d_2d_source openpose)'

import __init_path  # noqa: F401
from utils.jotr_dataset import get_test_dataset
from utils.refiner2d_tools import load_refiner, eval_refiner, format_result


def main():
    dev = torch.device('cuda')
    model, ck = load_refiner(args.ckpt, dev)
    lifter = None
    if args.lifter_ckpt:
        from models.ARTS import build_diff_lifter
        lifter = build_diff_lifter(args.lifter_ckpt, 1, 1).to(dev).eval()
    bs = args.batch_size or cfg.TEST.batch_size
    summary = {}
    for name in [s.strip() for s in args.sets.split(',') if s.strip()]:
        cfg.DATASET.pw3d_ocpc_test_only = (args.ocpc_mode == 'test') and name in ('3dpw-oc', '3dpw-pc')
        label = name + (' (chi seq test)' if cfg.DATASET.pw3d_ocpc_test_only else '')
        ds = get_test_dataset(name, None)
        reg = ds.h36m_joint_regressor
        if args.max_samples and args.max_samples < len(ds):
            ds = Subset(ds, list(range(args.max_samples)))
        loader = DataLoader(ds, batch_size=bs, shuffle=False, num_workers=cfg.DATASET.workers)
        r = eval_refiner(model, loader, dev, far_ratio=float(cfg.REFINE2D.far_ratio), lifter=lifter,
                         regressor=reg, fill_max_b=args.fill_max_b)
        print(format_result(label, r))
        summary[label] = r
    out = os.path.join(cfg.output_dir, 'eval_refiner2d.json')
    with open(out, 'w', encoding='utf-8') as f:
        json.dump({'ckpt': args.ckpt, 'epoch': ck.get('epoch'), 'lifter_ckpt': args.lifter_ckpt, 'cfg': args.cfg,
                   'fill_max_b': args.fill_max_b, 'results': summary}, f, indent=2, ensure_ascii=False)
    print(f'===> Luu {out}')


if __name__ == '__main__':
    main()
