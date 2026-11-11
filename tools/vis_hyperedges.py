"""
tools/vis_hyperedges.py -- (2026-11-12) Ve sieu canh HyperGCN len khung xuong T-pose SMPL de kiem tra bang mat.

Import DUNG danh sach sieu canh model dang dung:
  H36M-17 : V7_PARTS / V7_BODY (lib/models/teacher_student.py, = nhom khop trong hypergraph.py 'h36m')
  SMPL-24 : SMPL24_PARTS / SMPL24_BODY (lib/models/hypergraph_smpl.py, HyperGCN tren token query, cfg.MODEL.smpl_hgcn)
Toa do khop: T-pose SMPL neutral (beta = 0): SMPL-24 = J_regressor x v_template, H36M-17 = J_regressor_h36m x v_template.
Nhin tu PHIA TRUOC nguoi: ben TRAI nguoi (L_*) nam ben PHAI hinh.

Ra (thu muc --out):
  hyperedges_overview.png : 2 x 2 (H36M / SMPL) x (10 part / 5 body), moi sieu canh 1 mau chong len xuong
  hyperedges_h36m17.png   : moi sieu canh 1 o nho (khop thuoc sieu canh = do, con lai = xam)
  hyperedges_smpl24.png   : nhu tren cho SMPL-24
In ra console: bang khop -> danh sach part / body, kiem tra khop nao cung thuoc it nhat 1 part va 1 body.

Chay:
  python tools/vis_hyperedges.py --cfg config/train_student.yml --out output/vis_hyperedges
"""
import os, sys
sys.path.insert(0, './lib')
sys.path.insert(0, './')
sys.path.insert(0, './main')   # __init_path.py
import argparse

parser = argparse.ArgumentParser(description='Ve sieu canh HyperGCN len xuong T-pose')
parser.add_argument('--cfg', type=str, default='config/train_student.yml')
parser.add_argument('--out', type=str, default='output/vis_hyperedges')
args = parser.parse_args()

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from core.config import cfg, update_config
update_config(args.cfg)
import __init_path  # noqa: F401
from utils.smpl import SMPL
from models.teacher_student import V7_PARTS, V7_BODY
from models.hypergraph_smpl import SMPL24_PARTS, SMPL24_BODY, SMPL24_PARENTS, SMPL24_NAMES

H36M_PARENTS = (-1, 0, 1, 2, 0, 4, 5, 0, 7, 8, 9, 8, 11, 12, 8, 14, 15)
H36M_NAMES = ('Pelvis', 'R_Hip', 'R_Knee', 'R_Ankle', 'L_Hip', 'L_Knee', 'L_Ankle', 'Torso', 'Neck', 'Nose', 'Head',
              'L_Shoulder', 'L_Elbow', 'L_Wrist', 'R_Shoulder', 'R_Elbow', 'R_Wrist')
PART_NAMES = ('than', 'co-dau', 'dui P', 'cang chan P', 'dui T', 'cang chan T', 'vai-canh tay P', 'vai-canh tay T',
              'cang tay P', 'cang tay T')
BODY_NAMES = ('than', 'tay P', 'tay T', 'chan P', 'chan T')


def tpose_joints():
    """T-pose SMPL neutral -> (J24 (24, 2), J17 (17, 2)) toa do (x, y) met, nhin tu phia truoc."""
    layer = SMPL().layer['neutral']
    v = layer.th_v_template.detach().cpu().numpy().reshape(-1, 3)                 # (6890, 3)
    j24 = layer.th_J_regressor.detach().cpu().numpy() @ v                          # (24, 3)
    reg17 = np.load(os.path.join(cfg.DATASET.jotr_data_root, 'Human36M', 'J_regressor_h36m_correct.npy'))
    j17 = reg17 @ v                                                                # (17, 3)
    return j24[:, :2], j17[:, :2]


def draw_skeleton(ax, J, parents, names, color='0.55', label=True):
    for j, p in enumerate(parents):
        if p >= 0:
            ax.plot([J[j, 0], J[p, 0]], [J[j, 1], J[p, 1]], '-', color=color, lw=2, zorder=1)
    ax.scatter(J[:, 0], J[:, 1], s=18, c='k', zorder=3)
    if label:
        for j in range(len(J)):
            ax.text(J[j, 0] + 0.012, J[j, 1] + 0.012, str(j), fontsize=6, zorder=4)
    ax.set_aspect('equal')
    ax.axis('off')


def draw_group(ax, J, members, color, text=None):
    """1 sieu canh: duong day noi cac khop thanh vien (theo thu tu trong danh sach) + diem to mau."""
    P = J[members]
    ax.plot(P[:, 0], P[:, 1], '-', color=color, lw=12, alpha=0.30, solid_capstyle='round', zorder=2)
    ax.scatter(P[:, 0], P[:, 1], s=60, color=color, alpha=0.85, zorder=2)
    if text:
        c = P.mean(0)
        ax.text(c[0], c[1] - 0.04, text, fontsize=7, color=color, weight='bold', ha='center', zorder=5)


def check(groups_part, groups_body, n, names, tag):
    print(f'===== {tag}: {len(groups_part)} part, {len(groups_body)} body =====')
    ok = True
    for j in range(n):
        ps = [i + 1 for i, g in enumerate(groups_part) if j in g]
        bs = [i + 1 for i, g in enumerate(groups_body) if j in g]
        flag = '' if (ps and bs) else '   <-- THIEU'
        ok = ok and bool(ps and bs)
        print(f'  {j:2d} {names[j]:11s} part {str(ps):12s} body {str(bs):8s}{flag}')
    print(f'  -> {"DU" if ok else "THIEU"}: moi khop thuoc it nhat 1 part va 1 body')


def overview(J24, J17, path):
    cmap = plt.get_cmap('tab10')
    fig, axes = plt.subplots(2, 2, figsize=(14, 12))
    for r, (J, par, names, parts, body, tag) in enumerate((
            (J17, H36M_PARENTS, H36M_NAMES, V7_PARTS, V7_BODY, 'H36M-17 (HyperGCN cu, hypergraph.py)'),
            (J24, SMPL24_PARENTS, SMPL24_NAMES, SMPL24_PARTS, SMPL24_BODY, 'SMPL-24 (smpl_hgcn, moi)'))):
        for c, (groups, gnames, kind) in enumerate(((parts, PART_NAMES, 'part'), (body, BODY_NAMES, 'body'))):
            ax = axes[r, c]
            draw_skeleton(ax, J, par, names)
            for i, g in enumerate(groups):
                draw_group(ax, J, g, cmap(i % 10), f'{kind[0].upper()}{i + 1} {gnames[i]}')
            ax.set_title(f'{tag} - {len(groups)} {kind}', fontsize=10)
    fig.suptitle('Sieu canh tren T-pose SMPL (nhin tu truoc: ben TRAI nguoi = ben PHAI hinh)', fontsize=12)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def small_multiples(J, par, names, parts, body, tag, path):
    groups = [('P', i, g, PART_NAMES[i]) for i, g in enumerate(parts)] + \
             [('B', i, g, BODY_NAMES[i]) for i, g in enumerate(body)]
    ncol = 5
    nrow = int(np.ceil(len(groups) / ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=(3.2 * ncol, 3.6 * nrow))
    for ax, (k, i, g, nm) in zip(axes.flat, groups):
        draw_skeleton(ax, J, par, names, label=False)
        P = J[g]
        ax.scatter(P[:, 0], P[:, 1], s=70, c='red', zorder=4)
        for j in g:
            ax.text(J[j, 0] + 0.015, J[j, 1] + 0.015, f'{j} {names[j]}', fontsize=6, color='red', zorder=5)
        ax.set_title(f'{k}{i + 1} {nm}: {g}', fontsize=8)
    for ax in list(axes.flat)[len(groups):]:
        ax.axis('off')
    fig.suptitle(f'{tag}: moi o 1 sieu canh (do = khop thanh vien)', fontsize=11)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def main():
    os.makedirs(args.out, exist_ok=True)
    J24, J17 = tpose_joints()
    check(V7_PARTS, V7_BODY, 17, H36M_NAMES, 'H36M-17')
    check(SMPL24_PARTS, SMPL24_BODY, 24, SMPL24_NAMES, 'SMPL-24')
    overview(J24, J17, os.path.join(args.out, 'hyperedges_overview.png'))
    small_multiples(J17, H36M_PARENTS, H36M_NAMES, V7_PARTS, V7_BODY, 'H36M-17',
                    os.path.join(args.out, 'hyperedges_h36m17.png'))
    small_multiples(J24, SMPL24_PARENTS, SMPL24_NAMES, SMPL24_PARTS, SMPL24_BODY, 'SMPL-24',
                    os.path.join(args.out, 'hyperedges_smpl24.png'))
    print(f'===> Da luu 3 hinh vao {args.out}')


if __name__ == '__main__':
    main()
