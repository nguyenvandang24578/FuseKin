"""
tools/select_dense_vertices.py -- Chon N dinh CO DINH tren mesh mau SMPL neutral bang farthest point sampling
(FPS, khoang cach Euclid) cho nhan 2D day (lib/utils/dense2d.py). Chay 1 lan, ket qua dung chung cho moi mau.
Diem dau = dinh gan trong tam mesh nhat (tat dinh, lap lai duoc). FPS tren toa do -> phan bo deu theo khong gian,
khong bi don vao tay / mat (noi mesh co mat do dinh cao).

Ra: <out> (json: vertex_ids, n, method) + <out>.png (hinh truoc / ngang: dinh mesh xam, diem chon xanh kem chi so).
Chay (moi truong arts, can SMPL trong smplpytorch/smplpytorch/native/models):
  python tools/select_dense_vertices.py --n 48 --out config/dense48_fps.json
"""
import os, sys
sys.path.insert(0, './lib')
sys.path.insert(0, './')
sys.path.insert(0, './main')   # __init_path.py (them smplpytorch vao sys.path)
import argparse
import json

parser = argparse.ArgumentParser(description='Chon dinh SMPL cho nhan 2D day')
parser.add_argument('--n', type=int, default=48)
parser.add_argument('--out', type=str, default='config/dense48_fps.json')
args = parser.parse_args()

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from core.config import cfg  # noqa: F401  (cfg.smpl_path)
import __init_path  # noqa: F401
from utils.smpl import SMPL


def fps(points, n, start):
    chosen = [int(start)]
    d = np.linalg.norm(points - points[start], axis=1)
    for _ in range(n - 1):
        i = int(np.argmax(d))
        chosen.append(i)
        d = np.minimum(d, np.linalg.norm(points - points[i], axis=1))
    return chosen


def main():
    smpl = SMPL()
    v = smpl.layer['neutral'].th_v_template.reshape(-1, 3).cpu().numpy().astype(np.float64)   # (6890, 3) met
    start = int(np.argmin(np.linalg.norm(v - v.mean(0), axis=1)))
    ids = fps(v, args.n, start)
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    with open(args.out, 'w', encoding='utf-8') as f:
        json.dump({'vertex_ids': ids, 'n': args.n, 'method': 'fps_euclid_smpl_neutral_template', 'start': start}, f)
    print(f'===> {args.n} dinh: {ids}\n     luu {args.out}')

    joints = np.dot(smpl.joint_regressor[:24], v)                                          # khop SMPL de doi chieu
    fig, axes = plt.subplots(1, 2, figsize=(12, 9))
    for ax, (a, b, title) in zip(axes, [(0, 1, 'Nhin truoc (x, y)'), (2, 1, 'Nhin ngang (z, y)')]):
        ax.scatter(v[:, a], v[:, b], s=0.3, c='lightgray')
        ax.scatter(joints[:, a], joints[:, b], s=25, c='red', marker='x', label='khop SMPL')
        ax.scatter(v[ids, a], v[ids, b], s=30, c='blue', label=f'{args.n} dinh chon')
        for k, i in enumerate(ids):
            ax.annotate(str(k), (v[i, a], v[i, b]), fontsize=7, color='blue')
        ax.set_title(title)
        ax.set_aspect('equal')                                  # mesh mau SMPL: truc y huong len (dau o tren)
        ax.legend(loc='lower right')
    plt.tight_layout()
    png = os.path.splitext(args.out)[0] + '.png'
    plt.savefig(png, dpi=120)
    print(f'     hinh {png}')


if __name__ == '__main__':
    main()
