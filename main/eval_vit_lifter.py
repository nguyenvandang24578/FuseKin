"""
main/eval_vit_lifter.py (2026-11-18) -- Danh gia VitLifter (main/train_vit_lifter.py) tren 3DPW / OC / PC (/ Crowd),
2D dau vao = ViTPose (file Halpe-26 theo DATASET.pw3d_halpe_suffix cua YAML). Bao MPJPE17 / PA-MPJPE17 (mm),
GT = H36M regressor x mesh; oracle min-K chi do tiem nang. Lifter -> MPJPE17, KHONG so truc tiep voi mesh 14 khop.
--ocpc_mode test (mac dinh, vi lifter train co 3dpw-train) = OC / PC chi sequence test.

Chay:
  python main/eval_vit_lifter.py --cfg config/train_vit_lifter_3dpw.yaml \
    --ckpt experiment/vit_lifter_3dpw/checkpoint/last.pth.tar --sets 3dpw,3dpw-oc,3dpw-pc --gpu 0
"""
import os, sys
sys.path.append('./lib')
sys.path.append('./')
os.environ["OMP_NUM_THREADS"] = "1"
import argparse
import json

parser = argparse.ArgumentParser(description='Danh gia VitLifter')
parser.add_argument('--cfg', type=str, required=True, help='yml da dung de train VitLifter')
parser.add_argument('--ckpt', type=str, required=True, help='checkpoint VitLifter (last.pth.tar)')
parser.add_argument('--sets', type=str, default='3dpw,3dpw-oc,3dpw-pc')
parser.add_argument('--ocpc_mode', type=str, default='test', choices=['all', 'test'])
parser.add_argument('--K', type=int, default=10)
parser.add_argument('--S', type=int, default=4)
parser.add_argument('--eta', type=float, default=1.0)
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
cfg.DATASET.pw3d_test_gt2d = True
assert cfg.DATASET.pw3d_2d_source == 'halpe26', 'VitLifter: dau vao 3DPW la file Halpe-26 ViTPose'

import __init_path  # noqa: F401
from utils.jotr_dataset import get_test_dataset
from utils.vit_lifter_tools import load_vit_lifter, eval_vit_lifter, format_vit_eval


def main():
    dev = torch.device('cuda')
    lifter, enc, ck = load_vit_lifter(args.ckpt, dev)
    eps = float(ck.get('vit_logb_eps', cfg.IMGLIFT.vit_logb_eps))
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
        r = eval_vit_lifter(lifter, enc, loader, dev, reg, K=args.K, S=args.S, eta=args.eta, logb_eps=eps,
                            seed=args.seed)
        print(format_vit_eval(label, r))
        summary[label] = r
    out = os.path.join(cfg.output_dir, 'eval_vit_lifter.json')
    with open(out, 'w', encoding='utf-8') as f:
        json.dump({'ckpt': args.ckpt, 'epoch': ck.get('epoch'), 'args': vars(args), 'results': summary}, f,
                  indent=2, ensure_ascii=False)
    print(f'===> Luu {out}')


if __name__ == '__main__':
    main()
