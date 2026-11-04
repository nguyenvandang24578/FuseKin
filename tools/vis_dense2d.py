"""
tools/vis_dense2d.py -- Ve nhan 2D DAY (dinh mesh co dinh, lib/utils/dense2d.py) de KIEM TRA truoc khi train Refiner day.

Moi tap: N mau ngau nhien (seed co dinh), ve tren anh crop 256:
  xanh la nho  = 2D dau vao (train: GT + PoseFix; 3DPW: OpenPose / HHRNet)
  do + xuong   = 17 khop H36M tinh tu mesh GT (muc tieu Refiner hien tai)
  xanh duong   = N diem day (tron dac = trong khung, vong rong = ngoai khung / trunc 0), kem chi so diem
  (--ckpt) cam = 17 khop Refiner du doan, vang = diem day Refiner du doan
Kiem tra: diem day phai nam TREN nguoi dung (khong lech / khong lat trai phai), ca o mau bi lat ngang; ty le / vi tri
khop voi 17 khop do.
Ra: <cfg.output_dir>/vis_dense2d_<tap>.png

Chay (yml co DATASET.dense2d_vertex_file, vd config/train_refiner2d_jotr_mesh_dense48.yaml):
  python tools/vis_dense2d.py --cfg config/train_refiner2d_jotr_mesh_dense48.yaml --n 8
  (them Refiner day da train: --ckpt experiment/refiner2d_dense48/checkpoint/last.pth.tar --gpu 0)
"""
import os, sys
sys.path.append('./lib')
sys.path.append('./')
import argparse

parser = argparse.ArgumentParser(description='Ve nhan 2D day')
parser.add_argument('--cfg', type=str, required=True)
parser.add_argument('--sets', type=str, default='Human36M,MuCo,MSCOCO,CrowdPose,3dpw,3dpw-crowd')
parser.add_argument('--n', type=int, default=8)
parser.add_argument('--seed', type=int, default=0)
parser.add_argument('--ckpt', type=str, default='', help='Refiner (day) de ve them du doan')
parser.add_argument('--gpu', type=str, default='0')
args = parser.parse_args()
os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu

import cv2
import numpy as np
import torch

from core.config import cfg, update_config
update_config(args.cfg)
cfg.DATASET.pw3d_test_gt2d = True
assert cfg.DATASET.get('dense2d_vertex_file', ''), 'yml can DATASET.dense2d_vertex_file'

import __init_path  # noqa: F401
from models.diff_lifter import H36M_PARENTS
from utils.jotr_dataset import get_train_dataset, get_test_dataset

HM = cfg.output_hm_shape[2]
PX = cfg.input_img_shape[1] / HM                   # o luoi 64 -> px crop 256


def to_px_hm(p):
    return np.asarray(p, dtype=np.float32)[:, :2] * PX


def to_px_norm(p):
    return (np.asarray(p, dtype=np.float32)[:, :2] + 1) * 0.5 * cfg.input_img_shape[1]


def draw(sample, pred=None, title=''):
    inputs, targets, meta = sample
    img = (inputs['img'].numpy().transpose(1, 2, 0) * 255).clip(0, 255).astype(np.uint8)[:, :, ::-1].copy()
    img = cv2.resize(img, None, fx=2, fy=2, interpolation=cv2.INTER_LINEAR)    # 512 cho de nhin
    s = 2.0
    j_in, m_in = to_px_norm(inputs['joints']), np.asarray(inputs['joints_mask']).reshape(-1)
    for (x, y), m in zip(j_in, m_in):
        if m > 0:
            cv2.circle(img, (int(x * s), int(y * s)), 3, (0, 200, 0), -1)
    g17 = to_px_hm(targets['orig_joint_img'])
    for j, p in enumerate(H36M_PARENTS):
        if p >= 0:
            cv2.line(img, tuple(int(v * s) for v in g17[j]), tuple(int(v * s) for v in g17[p]), (0, 0, 220), 2)
    for x, y in g17:
        cv2.circle(img, (int(x * s), int(y * s)), 4, (0, 0, 255), -1)
    d = to_px_hm(targets['dense_img'])
    tr = np.asarray(meta['dense_trunc']).reshape(-1)
    for k, ((x, y), t) in enumerate(zip(d, tr)):
        cv2.circle(img, (int(x * s), int(y * s)), 4, (255, 80, 0), -1 if t > 0 else 1)
        cv2.putText(img, str(k), (int(x * s) + 4, int(y * s) - 3), cv2.FONT_HERSHEY_SIMPLEX, 0.3, (255, 80, 0), 1)
    if pred is not None:
        pxy = to_px_norm(pred)
        for x, y in pxy[:17]:
            cv2.circle(img, (int(x * s), int(y * s)), 3, (0, 140, 255), -1)
        for x, y in pxy[17:]:
            cv2.drawMarker(img, (int(x * s), int(y * s)), (0, 230, 230), cv2.MARKER_CROSS, 7, 1)
    cv2.putText(img, title, (6, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
    cv2.putText(img, title, (6, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
    return img


def main():
    model = None
    if args.ckpt:
        from utils.refiner2d_tools import load_refiner
        model, _ = load_refiner(args.ckpt, torch.device('cuda'))
    rng = np.random.default_rng(args.seed)
    for name in [s.strip() for s in args.sets.split(',') if s.strip()]:
        ds = get_train_dataset(name, None) if name in ('Human36M', 'MuCo', 'MSCOCO', 'CrowdPose') else \
            get_test_dataset(name, None)
        idx = rng.choice(len(ds), size=min(args.n, len(ds)), replace=False)
        tiles = []
        for i in idx:
            sample = ds[int(i)]
            pred = None
            if model is not None:
                inp = sample[0]
                with torch.no_grad():
                    out = model(inp['img'][None].cuda().float(),
                                torch.as_tensor(inp['joints'])[None, :, :2].cuda().float(),
                                torch.as_tensor(inp['joints_mask']).reshape(1, 17, 1).cuda().float())
                pred = out['xy'][0].cpu().numpy()
            tiles.append(draw(sample, pred, f'{name} #{int(i)}'))
        while len(tiles) % 4:
            tiles.append(np.zeros_like(tiles[0]))
        grid = cv2.vconcat([cv2.hconcat(tiles[r:r + 4]) for r in range(0, len(tiles), 4)])
        out = os.path.join(cfg.output_dir, f'vis_dense2d_{name}.png')
        cv2.imwrite(out, grid)
        print(f'===> {name}: {len(idx)} mau -> {out}')
    print('Chu thich: xanh la = 2D dau vao | do = 17 khop tu mesh | xanh duong = diem day (vong rong = ngoai khung)'
          + (' | cam = 17 khop du doan | vang (+) = diem day du doan' if model is not None else ''))


if __name__ == '__main__':
    main()
