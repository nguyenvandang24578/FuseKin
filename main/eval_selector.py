"""
main/eval_selector.py -- Danh gia Selector tren cac tap 3DPW: pose MPJPE17 (L0 / Selector / Oracle) va, neu co
--teacher_ckpt, mesh qua Teacher (dataset.evaluate, MPJPE14 / PA / MPVPE) + % khoang cach L0 -> Oracle lay lai.
OC / PC: --ocpc_mode test (chi seq test, mac dinh) | all | both. Khong dung GT de chon (tru dong Oracle).

Chay:
  python main/eval_selector.py --cfg config/train_selector_l2last.yml \
      --ckpt experiment/selector_l2last/checkpoint/last.pth.tar \
      --sets 3dpw,3dpw-oc,3dpw-pc,3dpw-crowd --gpu 0
"""
import os, sys
sys.path.append('./lib')
sys.path.append('./')
os.environ["OMP_NUM_THREADS"] = "1"
import argparse
import json

parser = argparse.ArgumentParser(description='Danh gia Selector')
parser.add_argument('--cfg', type=str, required=True)
parser.add_argument('--ckpt', type=str, required=True, help='checkpoint Selector (last.pth.tar | best.pth.tar)')
parser.add_argument('--teacher_ckpt', type=str, default='',
                    help="Teacher GT de giai ma mesh (bo trong = MODEL.TEACHER trong YAML; 'none' = chi pose)")
parser.add_argument('--sets', type=str, default='3dpw,3dpw-oc,3dpw-pc,3dpw-crowd')
parser.add_argument('--ocpc_mode', type=str, default='test', choices=['test', 'all', 'both'])
parser.add_argument('--max_batches', type=int, default=0, help='0 = toan bo')
parser.add_argument('--gpu', type=str, default='0')
args = parser.parse_args()
os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu

import torch
from torch.utils.data import DataLoader

from core.config import cfg, update_config
update_config(args.cfg)

import __init_path  # noqa: F401
from models.selector import SelectorNet
from utils.ckpt_tools import build_and_load
from utils.jotr_dataset import get_test_dataset
from utils.selector_tools import build_frozen_parts, evaluate, format_eval

DEVICE = torch.device('cuda')


def main():
    ck = torch.load(args.ckpt, map_location='cpu', weights_only=False)
    sc = ck.get('selector_cfg', dict(cfg.SELECTOR))
    for k in ('K', 'steps', 'use_flip'):
        if k in sc and sc[k] != cfg.SELECTOR[k]:
            print(f'[CANH BAO] SELECTOR.{k} YAML={cfg.SELECTOR[k]} khac checkpoint={sc[k]} -> dung gia tri checkpoint')
            cfg.SELECTOR[k] = sc[k]
    backbone, lifter = build_frozen_parts(DEVICE)
    model = SelectorNet(dim=sc['dim'], heads=sc['heads'], enc_layers=sc['enc_layers'], dec_layers=sc['dec_layers'],
                        use_flip=sc['use_flip'], dropout=sc['dropout'],
                        base_prior_init=sc.get('base_prior_init', 0.0)).to(DEVICE)
    model.load_state_dict(ck['selector_state'])
    print(f'===> Selector {args.ckpt} (epoch {ck.get("epoch")})')
    t_path = args.teacher_ckpt or cfg.MODEL.get('TEACHER', '')
    if t_path == 'none':
        t_path = ''
    elif t_path and not os.path.isfile(t_path):
        print(f'[CANH BAO] Khong thay Teacher {t_path} -> chi danh gia pose')
        t_path = ''
    teacher = build_and_load('teacher', t_path, DEVICE)[0] if t_path else None
    print(f'===> Teacher giai ma: {t_path or "khong"}')

    runs = []
    for name in [s.strip() for s in args.sets.split(',') if s.strip()]:
        if name in ('3dpw-oc', '3dpw-pc'):
            if args.ocpc_mode in ('test', 'both'):
                runs.append((f'{name} (chi seq test)', name, True))
            if args.ocpc_mode in ('all', 'both'):
                runs.append((f'{name} (toan bo, ro ri)', name, False))
        else:
            runs.append((name, name, False))
    summary = {}
    for label, name, test_only in runs:
        cfg.DATASET.pw3d_ocpc_test_only = test_only
        ds = get_test_dataset(name, None)
        loader = DataLoader(ds, batch_size=cfg.TEST.batch_size, shuffle=False, num_workers=cfg.DATASET.workers)
        r = evaluate(model, backbone, lifter, loader, ds, DEVICE, teacher=teacher, max_batches=args.max_batches)
        print(format_eval(label, r))
        summary[label] = r
    out = os.path.join(cfg.output_dir, 'summary.json')
    with open(out, 'w', encoding='utf-8') as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    print(f'Da luu: {out}')


if __name__ == '__main__':
    main()
