"""
main/eval_img_lifter.py -- Danh gia ImgLifter tren 3DPW / OC / PC / Crowd, 2D detector chuan benchmark
(OpenPose 3DPW / OC / PC, Higher-HRNet Crowd): OpenPose -> Refiner 2D -> ImgLifter (K gia thuyet, S buoc DDIM).
Bao MPJPE17 / PA-MPJPE17 (mm, GT = H36M regressor x mesh) cho gop mean / jpma, oracle min-K (chi do tiem nang)
va (--old_lifter_ckpt) DiffLifter cu voi OpenPose tren CUNG mau de so truc tiep.
--ablate_img: dac trung anh cua lifter = 0 (giu 2D Refiner) -> lifter co dung anh khong.
OC / PC mac dinh TOAN BO (giao thuc JOTR); --ocpc_mode test = chi sequence test.

Chay:
  python main/eval_img_lifter.py --cfg config/train_img_lifter_jotr.yaml \
    --ckpt experiment/img_lifter_jotr/checkpoint/last.pth.tar \
    --old_lifter_ckpt experiment/finetune_difflifter_d512_e15_jotr_pf05_3dpw/best_epoch.bin \
    --sets 3dpw,3dpw-oc,3dpw-pc,3dpw-crowd --gpu 0
  (thu nhanh: --max_samples 2000)
"""
import os, sys
sys.path.append('./lib')
sys.path.append('./')
os.environ["OMP_NUM_THREADS"] = "1"
import argparse
import json

parser = argparse.ArgumentParser(description='Danh gia ImgLifter')
parser.add_argument('--cfg', type=str, required=True, help='yml da dung de train ImgLifter')
parser.add_argument('--ckpt', type=str, required=True, help='checkpoint ImgLifter (last.pth.tar | epoch_XX.pth.tar)')
parser.add_argument('--refiner_ckpt', type=str, default='', help="rong = refiner luu trong ckpt ImgLifter")
parser.add_argument('--old_lifter_ckpt', type=str, default='', help='DiffLifter cu (.bin) de so tren cung mau')
parser.add_argument('--sets', type=str, default='3dpw,3dpw-oc,3dpw-pc,3dpw-crowd')
parser.add_argument('--ocpc_mode', type=str, default='all', choices=['all', 'test'])
parser.add_argument('--K', type=int, default=10)
parser.add_argument('--S', type=int, default=4)
parser.add_argument('--eta', type=float, default=1.0)
parser.add_argument('--jpma_max_b', type=float, default=3.0, help='jpma: chi chon theo 2D o khop co b <= nguong')
parser.add_argument('--ablate_img', action='store_true')
parser.add_argument('--max_samples', type=int, default=0, help='0 = toan bo tap')
parser.add_argument('--batch_size', type=int, default=0, help='0 = cfg.TEST.batch_size')
parser.add_argument('--seed', type=int, default=0)
parser.add_argument('--gpu', type=str, default='0')
args = parser.parse_args()
os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu

import torch
from torch.utils.data import DataLoader, Subset

from core.config import cfg, update_config
update_config(args.cfg)
assert cfg.DATASET.pw3d_2d_source == 'openpose', 'Benchmark: dau vao 3DPW la OpenPose / HHRNet (pw3d_2d_source openpose)'

import __init_path  # noqa: F401
from utils.jotr_dataset import get_test_dataset
from utils.img_lifter_tools import load_img_lifter, load_frozen_refiner, eval_img_lifter, format_eval


def main():
    dev = torch.device('cuda')
    lifter, ck = load_img_lifter(args.ckpt, dev)
    refiner = load_frozen_refiner(args.refiner_ckpt or ck['refiner_ckpt'], dev)
    old = None
    if args.old_lifter_ckpt:
        from models.ARTS import build_diff_lifter
        old = build_diff_lifter(args.old_lifter_ckpt, 1, 1).to(dev).eval()
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
        r = eval_img_lifter(lifter, refiner, loader, dev, reg, K=args.K, S=args.S, eta=args.eta,
                            jpma_max_b=args.jpma_max_b, ablate_img=args.ablate_img, old_lifter=old, seed=args.seed)
        print(format_eval(label, r))
        summary[label] = r
    out = os.path.join(cfg.output_dir, 'eval_img_lifter.json')
    with open(out, 'w', encoding='utf-8') as f:
        json.dump({'ckpt': args.ckpt, 'epoch': ck.get('epoch'), 'args': vars(args), 'results': summary}, f,
                  indent=2, ensure_ascii=False)
    print(f'===> Luu {out}')


if __name__ == '__main__':
    main()
