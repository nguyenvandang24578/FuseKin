"""
main/test_student_3dpw_sets.py
Danh gia 1 checkpoint Student / Teacher FuseKin (best.pth.tar cua main/train.py) tren nhieu tap 3DPW:
3dpw (test), 3dpw-oc, 3dpw-pc. Dung DUNG ham danh gia luc train (utils.jotr_evaluation.evaluate_3dpw_subset):
MPJPE / PA-MPJPE (14 khop) va MPVPE, mm.

RO RI OC / PC: 3dpw-oc va 3dpw-pc gom ca sequence cua 3dpw-train / val. Model FuseKin train tren 3dpw-train
-> so tren TOAN BO OC / PC bi ro ri (da thay sequence luc train). --ocpc_mode:
  test : chi sequence thuoc 3dpw test (DATASET.pw3d_ocpc_test_only = True) -> so cong bang
  all  : toan bo OC / PC (giong giao thuc JOTR, nhung JOTR KHONG train 3dpw -> khong so truc tiep)
  both : (mac dinh) in ca hai

Chay:
  python main/test_student_3dpw_sets.py --cfg ./config/train_student_kd_e1_v2_diff.yml --gpu 0 \
      --ckpt ./experiment/student_kd_e1_v2_diff/checkpoint/best.pth.tar
"""
import os, sys
sys.path.append('./lib')
sys.path.append('./')
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--cfg', type=str, required=True, help='yml dung de train checkpoint (cung kien truc, cung lifter)')
parser.add_argument('--ckpt', type=str, required=True, help='best.pth.tar')
parser.add_argument('--gpu', type=str, default='0')
parser.add_argument('--sets', type=str, default='3dpw,3dpw-oc,3dpw-pc')
parser.add_argument('--ocpc_mode', type=str, default='both', choices=['test', 'all', 'both'])
parser.add_argument('--batch_size', type=int, default=0, help='0 = cfg.TEST.batch_size')
args = parser.parse_args()
os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu

import torch
from torch.utils.data import DataLoader
from core.config import cfg, update_config
update_config(args.cfg)

import __init_path  # noqa: F401
from models.ARTS import ARTS
from utils.jotr_dataset import get_test_dataset
from utils.jotr_evaluation import evaluate_3dpw_subset
from funcs_utils import check_data_pararell


def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = ARTS(num_joint=17, embed_dim=cfg.MODEL.hpe_dim).to(device)
    ck = torch.load(args.ckpt, map_location='cpu', weights_only=False)
    missing, unexpected = model.load_state_dict(check_data_pararell(ck['model_state_dict']), strict=False)
    print(f'===> nap {args.ckpt} (epoch {ck.get("epoch")}) | thieu {len(missing)} | thua {len(unexpected)} key')
    if missing:
        print(f'     thieu (vai key dau): {missing[:5]}')
    model.eval()
    bs = args.batch_size or cfg.TEST.batch_size

    runs = []   # (ten hien thi, ten tap, ocpc_test_only)
    for name in [s.strip() for s in args.sets.split(',') if s.strip()]:
        if name in ('3dpw-oc', '3dpw-pc'):
            if args.ocpc_mode in ('test', 'both'):
                runs.append((f'{name} (chi seq test)', name, True))
            if args.ocpc_mode in ('all', 'both'):
                runs.append((f'{name} (toan bo, ro ri)', name, False))
        else:
            runs.append((name, name, False))

    results = []
    for label, name, test_only in runs:
        cfg.DATASET.pw3d_ocpc_test_only = test_only      # PW3D doc co nay luc tao dataset
        try:
            ds = get_test_dataset(name, None)
            loader = DataLoader(ds, batch_size=bs, shuffle=False, num_workers=cfg.DATASET.workers)
            with torch.no_grad():
                m = evaluate_3dpw_subset(model, ds, loader, device=device)
            results.append((label, len(ds), m))
            print(f'[{label}] n={len(ds)} | MPJPE {m["mpjpe"]:.2f} | PA {m["pa_mpjpe"]:.2f} | MPVPE {m["mpvpe"]:.2f}')
        except Exception as e:
            results.append((label, 0, None))
            print(f'[{label}] LOI: {type(e).__name__}: {e}')

    print('\n' + '=' * 78)
    print(f'{args.ckpt} | epoch {ck.get("epoch")} | lifter = {cfg.MODEL.get("pose_lifter", "motionbert")}')
    print(f'{"tap":<28} | {"n":>6} | {"MPJPE":>7} | {"PA-MPJPE":>8} | {"MPVPE":>7}   (mm, 14 khop)')
    print('-' * 78)
    for label, n, m in results:
        if m is None:
            print(f'{label:<28} | LOI (xem phia tren)')
        else:
            print(f'{label:<28} | {n:>6} | {m["mpjpe"]:>7.2f} | {m["pa_mpjpe"]:>8.2f} | {m["mpvpe"]:>7.2f}')
    print('=' * 78)


if __name__ == '__main__':
    main()
