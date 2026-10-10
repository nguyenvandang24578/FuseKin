"""
eval_teacher_train_subset.py — Teacher sai 46.7 mm tren val (san decoder <= 27 mm): do OVERFIT hay do THIEU KHA NANG?
=====================================================================================================================

Danh gia CUNG mot checkpoint, CUNG giao thuc (evaluate_3dpw_subset -> PW3D.evaluate, 14 khop H36M,
PA-MPJPE, MPVPE) tren:
    [train] tap con ngau nhien cua 3DPW-train
    [val]   tap con ngau nhien cua 3DPW test (cfg.DATASET.test_list[0])

Tap train duoc nap voi data_name='3dpw-train' (dung datalist train) nhung ep data_split='test' sau khi
nap -> __getitem__ di nhanh TEST: KHONG augmentation (khong xoay/scale/mau), bbox tinh tu OpenPose giong
het val, target = smpl_mesh_cam tuyet doi -> evaluate() tinh y het val. Chi khac nhau o DU LIEU.

Doc ket qua:
    train ~ val (vd ca hai ~45 mm)   -> THIEU KHA NANG BIEU DIEN (duong fusion -> HyperGCN -> head -> VPoser
                                        khong tai tao du pose). Huong: pose 6D tung khop / bo VPoser / loss joint.
    train << val (vd ~25 vs ~47 mm)  -> OVERFIT / thieu du lieu (3DPW-train nho). Huong: them H36M, MuCo...
    O giua                          -> ca hai cung gop phan.

Cach chay:
    python main/eval_teacher_train_subset.py --cfg ./config/train_teacher_gt.yml \
        --teacher_ckpt <best.pth.tar> [--num 3000] [--batch_size 32] [--workers 4]

Can CUDA. Khong train gi.
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

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset
from torchvision import transforms

parser = argparse.ArgumentParser(description='Danh gia Teacher tren tap con 3DPW-train vs val (cung giao thuc)')
parser.add_argument('--cfg', type=str, required=True, help='config cua Teacher (vd ./config/train_teacher_gt.yml)')
parser.add_argument('--teacher_ckpt', type=str, default='', help='mac dinh: cfg.MODEL.TEACHER')
parser.add_argument('--num', type=int, default=3000, help='so mau moi tap (<=0: toan bo)')
parser.add_argument('--batch_size', type=int, default=32)
parser.add_argument('--workers', type=int, default=4)
parser.add_argument('--seed', type=int, default=0)
parser.add_argument('--skip_val', action='store_true', help='chi danh gia train')
args = parser.parse_args()

from core.config import cfg, update_config
update_config(args.cfg)

import __init_path  # noqa: F401
from models.ARTS import ARTS
from core.base import load_model_weights
from utils.jotr_dataset import Human36M17Dataset, get_test_dataset
from utils.jotr_evaluation import evaluate_3dpw_subset
from data_final.PW3D.dataset import PW3D

if not torch.cuda.is_available():
    print("\n[LOI] Can CUDA de chay script nay.\n")
    sys.exit(1)
DEVICE = torch.device('cuda')


def hr(title=''):
    print('\n' + '=' * 78)
    if title:
        print(title)
        print('=' * 78)


def build_teacher(ckpt_path):
    if cfg.MODEL.name != 'teacher':
        print(f"[LOI] cfg.MODEL.name = '{cfg.MODEL.name}', script nay chi danh gia Teacher. Dung config cua Teacher.")
        sys.exit(1)
    if not os.path.isfile(ckpt_path):
        print(f"[LOI] Khong thay checkpoint: {ckpt_path}")
        sys.exit(1)
    model = ARTS(num_joint=17, embed_dim=cfg.MODEL.hpe_dim).to(DEVICE)
    model = load_model_weights(model, ckpt_path)
    model.eval()
    for p in model.parameters():
        p.requires_grad = False
    return model


def build_train_eval_dataset():
    """Datalist 3DPW-train nhung di nhanh TEST cua __getitem__ (khong augmentation, target mesh tuyet doi)."""
    base = PW3D(transforms.ToTensor(), data_name='3dpw-train')
    base.data_split = 'test'   # ep nhanh test: augmentation(..., 'test'), tra {'smpl_mesh_cam'} nhu val
    return Human36M17Dataset(base)


def make_loader(dataset, num, seed):
    n = len(dataset)
    if num > 0 and num < n:
        rng = np.random.RandomState(seed)
        idx = np.sort(rng.choice(n, size=num, replace=False))
        sub = Subset(dataset, idx.tolist())
    else:
        sub = dataset
    loader = DataLoader(sub, batch_size=args.batch_size, shuffle=False,
                        num_workers=args.workers, pin_memory=True)
    return loader, len(sub)


def run(name, dataset, model):
    loader, n = make_loader(dataset, args.num, args.seed)
    print(f"  [{name}] {n}/{len(dataset)} mau")
    # Truyen dataset GOC (khong phai Subset): evaluate_3dpw_subset can dataset.h36m_joint_regressor va
    # dataset.evaluate; chi so mau trong evaluate() chi dung cho render/vis (dang tat) nen khong anh huong.
    res = evaluate_3dpw_subset(model, dataset, loader, device=DEVICE)
    print(f"  [{name}] MPJPE={res['mpjpe']:.2f}  PA-MPJPE={res['pa_mpjpe']:.2f}  MPVPE={res['mpvpe']:.2f}")
    return res, n


def main():
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    ckpt = args.teacher_ckpt or cfg.MODEL.get('TEACHER', '')
    if not ckpt:
        print("[LOI] Thieu checkpoint: dung --teacher_ckpt hoac dat cfg.MODEL.TEACHER")
        sys.exit(1)
    alpha_max = cfg.MODEL.get('teacher_lift_alpha_max', 0.0)
    print(f"Teacher ckpt : {ckpt}")
    print(f"teacher_lift_alpha_max = {alpha_max}" +
          ("" if alpha_max == 0 else f" -> eval tron lift voi a = {cfg.MODEL.get('teacher_lift_alpha_eval', alpha_max)}"
                                     " (CANH BAO: tren train MotionBERT da thuoc long, lift gan dung -> khong so sanh cong bang)"))

    hr("Build va nap Teacher")
    model = build_teacher(ckpt)

    hr("Danh gia (cung giao thuc evaluate_3dpw_subset)")
    results = {}
    results['train'] = run('train', build_train_eval_dataset(), model)
    if not args.skip_val:
        results['val'] = run('val', get_test_dataset(cfg.DATASET.test_list[0], None), model)

    hr("TOM TAT (mm)")
    print(f"  {'':8s} {'n':>6s} {'MPJPE':>9s} {'PA-MPJPE':>9s} {'MPVPE':>9s}")
    for k, (r, n) in results.items():
        print(f"  {k:8s} {n:6d} {r['mpjpe']:9.2f} {r['pa_mpjpe']:9.2f} {r['mpvpe']:9.2f}")
    print(f"  {'san':8s} {'':>6s} {'<=27.3':>9s} {'~16':>9s} {'~30':>9s}   (vposer_floor_test, shape=0)")

    if 'val' in results:
        tr, va = results['train'][0]['mpjpe'], results['val'][0]['mpjpe']
        gap = va - tr
        print(f"\n  val - train = {gap:+.2f} mm")
        if gap < 5:
            print("  -> train ~ val: THIEU KHA NANG BIEU DIEN (duong joint -> pose -> VPoser), khong phai overfit.")
        elif tr < 32:
            print("  -> train gan san, val cao hon nhieu: chu yeu OVERFIT / THIEU DU LIEU.")
        else:
            print("  -> ca hai: train con cach san va val con cach train -> vua thieu kha nang vua overfit.")


if __name__ == '__main__':
    main()
