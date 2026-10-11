"""
test_ckpt.py — Test MOT checkpoint, y het buoc test sau moi epoch luc train
==========================================================================

Dung dung Teacher_Tester / Student_Tester trong core/base.py (tuy cfg.MODEL.name) -> evaluate_3dpw_subset
-> PW3D.evaluate (MPJPE 14 khop, PA-MPJPE, MPVPE) tren tap test cfg.DATASET.test_list.
In ra dong '3dpw: MPJPE=..., PA-MPJPE=..., MPVPE=...' giong log train.

Cach chay (tu thu muc goc repo, can CUDA):
    python main/test_ckpt.py --cfg ./config/train_student_kd_e1.yml \
        --ckpt experiment/student_kd_e1/checkpoint/best.pth.tar [--gpu 0] [--test_list 3dpw 3dpw-oc 3dpw-pc]

- --cfg: config DA DUNG de train checkpoint do (dung kien truc, MotionBERT, chuan hoa anh...).
  Teacher -> config teacher; Student -> config student.
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

parser = argparse.ArgumentParser(description='Test 1 checkpoint (giong test sau moi epoch)')
parser.add_argument('--cfg', type=str, required=True, help='config da dung de train checkpoint')
parser.add_argument('--ckpt', type=str, required=True, help='duong dan checkpoint (.pth.tar)')
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
from core.base import Teacher_Tester, Student_Tester

if not os.path.isfile(args.ckpt):
    print(f"[LOI] Khong thay checkpoint: {args.ckpt}")
    sys.exit(1)
if args.test_list:
    cfg.DATASET.test_list = list(args.test_list)
if args.batch_size > 0:
    cfg.TEST.batch_size = args.batch_size

TESTERS = {'teacher': Teacher_Tester, 'student': Student_Tester}
if cfg.MODEL.name not in TESTERS:
    print(f"[LOI] cfg.MODEL.name = '{cfg.MODEL.name}', chi ho tro: {list(TESTERS)}")
    sys.exit(1)

print(f"Model    : {cfg.MODEL.name}")
print(f"Ckpt     : {args.ckpt}")
print(f"Config   : {args.cfg}")
print(f"Tap test : {cfg.DATASET.test_list}")

ckpt_epoch = None
try:
    from core.base import _PickleShim
    _c = torch.load(args.ckpt, map_location='cpu', pickle_module=_PickleShim, weights_only=False)
    ckpt_epoch = _c.get('epoch') if isinstance(_c, dict) else None
    del _c
except Exception:
    pass
if ckpt_epoch is not None:
    print(f"Epoch luu trong checkpoint: {ckpt_epoch}")

tester = TESTERS[cfg.MODEL.name](args, load_dir=args.ckpt)
with torch.no_grad():
    tester.test(epoch=ckpt_epoch or 0)
