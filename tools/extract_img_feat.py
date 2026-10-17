"""Tinh san dac trung anh SPIN ResNet-50 (global-average-pool, 2048-d) cho finetune / danh gia PMCE.

Moi mau i cua tap (lib/utils/pmce_data.build_dataset: lift_only, KHONG augment) -> doc anh goc, warpAffine bang
img2bb_trans cua chinh mau do (= crop 256 ma dataset tao, giong augmentation() khi khong augment) -> SPIN -> feat[i].
Luu <feat_dir>/<ten>_spin256.npz: feat (N, 2048) float16, hw (N, 2) float32 [h, w anh goc], ok (N,) uint8,
  check_idx / check_path (moi 10000 mau) de doi chieu thu tu khi nap lai.

Can dung CUNG yml (cung nguon 2D: h36m/muco/pw3d_2d_source) voi luc finetune, vi thu tu / so mau phu thuoc vao do.
Chay (moi truong FuseKin, tu thu muc goc repo), co the chia tap cho nhieu GPU:
  CUDA_VISIBLE_DEVICES=0 python tools/extract_img_feat.py --cfg ./config/finetune_pmce_halpe26.yaml --names h36m-train &
  CUDA_VISIBLE_DEVICES=1 python tools/extract_img_feat.py --cfg ./config/finetune_pmce_halpe26.yaml --names muco-train &
  CUDA_VISIBLE_DEVICES=2 python tools/extract_img_feat.py --cfg ./config/finetune_pmce_halpe26.yaml --names 3dpw-train 3dpw-val &
  CUDA_VISIBLE_DEVICES=3 python tools/extract_img_feat.py --cfg ./config/finetune_pmce_halpe26.yaml --names 3dpw-test h36m-test &
  wait
"""
import os, sys
sys.path.append('./lib')
sys.path.append('./')
import argparse

import cv2
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from core.config import cfg, update_config
from utils.pmce_data import FEAT_SETS, build_dataset, feat_path


class CropSet(Dataset):
    def __init__(self, ds):
        self.ds = ds

    def __len__(self):
        return len(self.ds)

    def __getitem__(self, i):
        _, _, meta = self.ds[i]
        W, H = int(cfg.input_img_shape[1]), int(cfg.input_img_shape[0])
        img = cv2.imread(meta['img_path'], cv2.IMREAD_COLOR | cv2.IMREAD_IGNORE_ORIENTATION)
        if img is None:
            return torch.zeros(3, H, W), np.zeros(2, np.float32), 0
        h, w = img.shape[:2]
        crop = cv2.warpAffine(img[:, :, ::-1].copy(), np.asarray(meta['img2bb_trans'], np.float64), (W, H),
                              flags=cv2.INTER_LINEAR)                 # RGB, giong load_img + generate_patch_image
        t = torch.from_numpy(crop.astype(np.float32).transpose(2, 0, 1).copy()) / 255.
        return t, np.array([h, w], np.float32), 1


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--cfg', required=True, help='yml co DATASET.* (cung voi luc finetune PMCE)')
    ap.add_argument('--names', nargs='+', required=True, choices=FEAT_SETS)
    ap.add_argument('--feat_dir', default='data_final/feat_cache')
    ap.add_argument('--spin_ckpt', default='data_final/base_data/spin_model_checkpoint.pth.tar')
    ap.add_argument('--batch_size', type=int, default=128)
    ap.add_argument('--workers', type=int, default=16)
    ap.add_argument('--overwrite', action='store_true')
    args = ap.parse_args()

    update_config(args.cfg)
    os.makedirs(args.feat_dir, exist_ok=True)
    device = torch.device('cuda')

    from models.backbones.resnet import ResNetBackbone
    backbone = ResNetBackbone(resnet_type=50)
    backbone.load_pretrained(source='spin', spin_checkpoint=args.spin_ckpt)
    backbone = backbone.to(device).eval()

    for name in args.names:
        out = feat_path(args.feat_dir, name)
        if os.path.isfile(out) and not args.overwrite:
            print(f'[{name}] da co {out} -> bo qua (--overwrite de tinh lai)')
            continue
        ds = build_dataset(name)
        n = len(ds)
        loader = DataLoader(CropSet(ds), batch_size=args.batch_size, shuffle=False,
                            num_workers=args.workers, pin_memory=True)
        feat = np.zeros((n, 2048), np.float16)
        hw = np.zeros((n, 2), np.float32)
        ok = np.zeros((n,), np.uint8)
        k = 0
        with torch.no_grad():
            for img, hw_b, ok_b in tqdm(loader, desc=name):
                _, tok = backbone(img.to(device, non_blocking=True))
                b = img.shape[0]
                feat[k:k + b] = tok.view(b, -1).cpu().numpy().astype(np.float16)
                hw[k:k + b] = hw_b.numpy()
                ok[k:k + b] = ok_b.numpy()
                k += b
        check_idx = np.arange(0, n, 10000)
        check_path = np.array([ds[int(i)][2]['img_path'] for i in check_idx])
        np.savez(out, feat=feat, hw=hw, ok=ok, check_idx=check_idx, check_path=check_path)
        print(f'[{name}] luu {n} dac trung -> {out} | anh loi: {int((ok == 0).sum())}')


if __name__ == '__main__':
    main()
