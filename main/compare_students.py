"""
compare_students.py — Danh gia NHIEU checkpoint Student tren CUNG tap test bang Student_Tester (base.py)
=======================================================================================================

Dung dung Student_Tester.test() nhu luc train (evaluate_3dpw_subset -> PW3D.evaluate: MPJPE 14 khop,
PA-MPJPE, MPVPE) tren TOAN BO tap test (cfg.DATASET.test_list), roi in bang so sanh + chenh lech so
voi checkpoint dau tien (moc, vd Student scratch).

Cach chay (tu thu muc goc repo, can CUDA):
    python main/compare_students.py --cfg ./config/train_student_kd_e1.yml \
        --ckpts <scratch>/checkpoint/best.pth.tar <e1>/checkpoint/best.pth.tar <e2>/checkpoint/best.pth.tar \
        --names scratch e1 e2 [--gpu 0] [--test_list 3dpw 3dpw-oc]

- --cfg: config Student bat ky (chi dung de dung kien truc + MotionBERT + tap test). Kien truc Student
  giong nhau o moi ban nen dung chung 1 cfg duoc. motionbert_pretrained / motionbert_2d_rootrel PHAI
  dung ban da dung luc train cac checkpoint nay.
- Checkpoint phai la checkpoint STUDENT (Teacher se bi bao loi ro rang trong load_model_weights).
"""
import os, sys
sys.path.append('./lib')
sys.path.append('./')
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"

import argparse
import warnings
warnings.filterwarnings("ignore")

parser = argparse.ArgumentParser(description='So sanh nhieu checkpoint Student bang Student_Tester')
parser.add_argument('--cfg', type=str, required=True, help='config Student (vd ./config/train_student_kd_e1.yml)')
parser.add_argument('--ckpts', type=str, nargs='+', required=True, help='cac checkpoint Student; cai DAU TIEN la moc')
parser.add_argument('--names', type=str, nargs='*', default=None, help='ten hien thi, cung so luong voi --ckpts')
parser.add_argument('--gpu', type=str, default='0')
parser.add_argument('--test_list', type=str, nargs='*', default=None,
                    help="ghi de cfg.DATASET.test_list, vd: 3dpw 3dpw-oc 3dpw-pc")
parser.add_argument('--batch_size', type=int, default=0, help='ghi de cfg.TEST.batch_size (0 = giu nguyen)')
args = parser.parse_args()
args.resume_training = False   # prepare_network co doc thuoc tinh nay

from core.config import cfg, update_config
update_config(args.cfg)
os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpu)

import torch
import __init_path  # noqa: F401
from core.base import Student_Tester

if cfg.MODEL.name != 'student':
    print(f"[LOI] cfg.MODEL.name = '{cfg.MODEL.name}'. Dung config cua STUDENT.")
    sys.exit(1)
if args.test_list:
    cfg.DATASET.test_list = list(args.test_list)
if args.batch_size > 0:
    cfg.TEST.batch_size = args.batch_size

names = args.names if args.names else [os.path.basename(os.path.dirname(os.path.dirname(c))) or c for c in args.ckpts]
if len(names) != len(args.ckpts):
    print(f"[LOI] --names co {len(names)} ten nhung --ckpts co {len(args.ckpts)} checkpoint.")
    sys.exit(1)
for c in args.ckpts:
    if not os.path.isfile(c):
        print(f"[LOI] Khong thay checkpoint: {c}")
        sys.exit(1)

print(f"Config   : {args.cfg}")
print(f"MotionBERT: {cfg.MODEL.motionbert_pretrained} (2d_rootrel={cfg.MODEL.motionbert_2d_rootrel})")
print(f"Tap test : {cfg.DATASET.test_list}")

all_results = {}
for name, ckpt in zip(names, args.ckpts):
    print('\n' + '=' * 78)
    print(f"[{name}] {ckpt}")
    print('=' * 78)
    tester = Student_Tester(args, load_dir=ckpt)
    with torch.no_grad():
        all_results[name] = tester.test(epoch=0)
    del tester
    torch.cuda.empty_cache()

base = names[0]
for ds in cfg.DATASET.test_list:
    print('\n' + '=' * 78)
    print(f"KET QUA tren {ds} (mm) — chenh lech so voi '{base}' (am = tot hon)")
    print('=' * 78)
    print(f"  {'ban':<16}{'MPJPE':>9}{'PA-MPJPE':>10}{'MPVPE':>9}   {'dMPJPE':>8}{'dPA':>8}{'dMPVPE':>8}")
    b = all_results[base][ds]
    for name in names:
        r = all_results[name][ds]
        print(f"  {name:<16}{r['mpjpe']:9.2f}{r['pa_mpjpe']:10.2f}{r['mpvpe']:9.2f}   "
              f"{r['mpjpe'] - b['mpjpe']:+8.2f}{r['pa_mpjpe'] - b['pa_mpjpe']:+8.2f}{r['mpvpe'] - b['mpvpe']:+8.2f}")
