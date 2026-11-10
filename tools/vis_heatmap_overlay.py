"""
tools/vis_heatmap_overlay.py -- Ve heatmap 2D chong len anh RGB (kieu hinh minh hoa paper: nen xanh, nguoi do/vang).

Moi mau ve 1 hang 4 o (crop 256, phong to x2):
  (1) anh RGB crop
  (2) heatmap DAU VAO cua Refiner (Gauss sigma tu 2D detector, 17 khop, giong make_heatmaps) chong len anh
  (3) heatmap DAU RA cua Refiner (softmax 65 kenh = 17 khop + 48 dinh day) chong len anh
  (4) heatmap dau ra dang "thuan" (colormap JET, khong co anh) -- giong o minh hoa trong paper
Gop kenh: moi kenh chuan hoa ve [0,1] theo dinh cua no roi lay max qua cac kenh (--merge max) hoac tong (--merge sum).

Chay (Refiner day da train, yml cua Refiner):
  python tools/vis_heatmap_overlay.py --cfg config/train_refiner2d_jotr_mesh_dense48_scratch.yaml --ckpt experiment/refiner2d_dense48_scratch/checkpoint/last.pth.tar --sets 3dpw,3dpw-crowd --n 6 --gpu 0
Ra: <--out hoac cfg.output_dir>/vis_heatmap_<tap>.png
"""
import os, sys
sys.path.insert(0, './lib')
sys.path.insert(0, './')
sys.path.insert(0, './main')   # __init_path.py (them smplpytorch vao sys.path)
import argparse

parser = argparse.ArgumentParser(description='Ve heatmap 2D chong len anh RGB')
parser.add_argument('--cfg', type=str, required=True)
parser.add_argument('--ckpt', type=str, required=True, help='checkpoint Refiner 2D (day)')
parser.add_argument('--sets', type=str, default='3dpw,3dpw-crowd')
parser.add_argument('--n', type=int, default=6)
parser.add_argument('--seed', type=int, default=0)
parser.add_argument('--alpha', type=float, default=0.55, help='do dam heatmap khi chong len anh')
parser.add_argument('--gamma', type=float, default=0.5, help='<1: lam heatmap loang rong hon (de nhin)')
parser.add_argument('--merge', type=str, default='max', choices=['max', 'sum'])
parser.add_argument('--joints_only', action='store_true', help='chi ve 17 khop (bo 48 dinh day) o dau ra')
parser.add_argument('--out', type=str, default='', help='thu muc ra (mac dinh cfg.output_dir)')
parser.add_argument('--gpu', type=str, default='0')
args = parser.parse_args()
os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu

import cv2
import numpy as np
import torch

from core.config import cfg, update_config
update_config(args.cfg)

import __init_path  # noqa: F401
from models.ARTS import make_heatmaps
from utils.jotr_dataset import get_train_dataset, get_test_dataset
from utils.refiner2d_tools import load_refiner

S = 2   # phong to crop 256 -> 512 cho de nhin
TRAIN_SETS = ('Human36M', 'MuCo', 'MSCOCO', 'CrowdPose')


def merge_maps(m):
    """m (J, h, w) >= 0 -> (h, w) trong [0,1]: chuan hoa tung kenh theo dinh, gop max / sum, ap gamma."""
    m = m / np.maximum(m.reshape(m.shape[0], -1).max(1), 1e-8)[:, None, None]
    g = m.max(0) if args.merge == 'max' else m.sum(0)
    g = g / max(g.max(), 1e-8)
    return g ** args.gamma


def colorize(g, size):
    """(h, w) [0,1] -> anh mau JET (size, size, 3) BGR; noi suy bicubic de min."""
    g = cv2.resize(g.astype(np.float32), (size, size), interpolation=cv2.INTER_CUBIC).clip(0, 1)
    return cv2.applyColorMap((g * 255).astype(np.uint8), cv2.COLORMAP_JET), g


def overlay(img, g):
    """Chong heatmap len anh; vung heatmap ~0 giu anh goc (alpha ty le voi gia tri)."""
    col, gs = colorize(g, img.shape[0])
    a = (args.alpha * np.clip(gs * 1.5, 0, 1))[..., None]
    return (img * (1 - a) + col * a).astype(np.uint8)


def label(img, text):
    cv2.putText(img, text, (6, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
    cv2.putText(img, text, (6, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
    return img


@torch.no_grad()
def run_sample(model, sample, title):
    inp = sample[0]
    img_t = inp['img'][None].cuda().float()                                         # (1,3,256,256) RGB [0,1]
    p2 = torch.as_tensor(inp['joints'])[None, :, :2].cuda().float()                 # (1,17,2) [-1,1]
    m = torch.as_tensor(inp['joints_mask']).reshape(1, 17, 1).cuda().float()
    out = model(img_t, p2, m)
    # Heatmap dau ra: softmax tren luoi 64x64 cho tung kenh (giong soft-argmax trong Refiner)
    lg = out['logits'][0].float()
    J, H, W = lg.shape
    prob = torch.softmax(lg.reshape(J, -1), -1).reshape(J, H, W).cpu().numpy()
    if args.joints_only:
        prob = prob[:17]
    # Heatmap dau vao: dung lai make_heatmaps cua ARTS (cung sigma voi Refiner)
    hin = make_heatmaps(p2, m, size=img_t.shape[-1] // 4, sigma=model.heat_sigma)[0].cpu().numpy()

    img = (inp['img'].numpy().transpose(1, 2, 0) * 255).clip(0, 255).astype(np.uint8)[:, :, ::-1]
    img = cv2.resize(img, None, fx=S, fy=S, interpolation=cv2.INTER_LINEAR)
    g_out = merge_maps(prob)
    tiles = [label(img.copy(), title),
             label(overlay(img, merge_maps(hin)), 'input heatmap (detector)'),
             label(overlay(img, g_out), 'Refiner heatmap'),
             label(colorize(g_out, img.shape[0])[0], 'Refiner heatmap (raw)')]
    return cv2.hconcat(tiles)


def main():
    model, _ = load_refiner(args.ckpt, torch.device('cuda'))
    out_dir = args.out or cfg.output_dir
    os.makedirs(out_dir, exist_ok=True)
    rng = np.random.default_rng(args.seed)
    for name in [s.strip() for s in args.sets.split(',') if s.strip()]:
        ds = get_train_dataset(name, None) if name in TRAIN_SETS else get_test_dataset(name, None)
        idx = rng.choice(len(ds), size=min(args.n, len(ds)), replace=False)
        rows = [run_sample(model, ds[int(i)], f'{name} #{int(i)}') for i in idx]
        path = os.path.join(out_dir, f'vis_heatmap_{name}.png')
        cv2.imwrite(path, cv2.vconcat(rows))
        print(f'===> {name}: {len(rows)} mau -> {path}')


if __name__ == '__main__':
    main()
