"""
tools/vis_vit_lifter.py (2026-11-18, nguoi dung duyet) -- Visual VitLifter (main/train_vit_lifter.py) tren mau train
va test: moi mau 1 hang 3 o
  [1] anh crop 256 + 2D ViTPose dau vao (mau theo conf: do < 0.3, cam < 0.6, xanh) + GT 2D mesh (x trang)
  [2] pose 3D tren truc 3D (matplotlib), GT (xanh la) + du doan (do), goc nhin camera
  [3] cung pose 3D, goc nhin cheo tu ben canh (thay loi do sau)
  Khop ViTPose thieu (det = 0) -> vong tron rong tren pose du doan. Tieu de: MPJPE17 / PA (mm), so khop thieu.
Moi tap: lay ngau nhien --max_samples mau (seed), chay lifter K / S, gop mean, ve --num mau ngau nhien va --num mau
MPJPE cao nhat. Toa do 3D: he camera, root-relative (pelvis = 0), met; ve X = x, Y = z (do sau), Z = -y (len tren).
GT 3D: tap train = orig_joint_cam; tap test = H36M regressor x mesh GT (giong eval_vit_lifter).

Chay (tu goc repo, can matplotlib):
  python tools/vis_vit_lifter.py --cfg config/train_vit_lifter_3dpw.yaml \
    --ckpt experiment/vit_lifter_3dpw/checkpoint/last.pth.tar \
    --sets 3dpw-train-noaug,3dpw-oc,3dpw-pc,3dpw-crowd --gpu 0
Ket qua: <out_dir>/<set>_random.jpg, <set>_worst.jpg
"""
import os, sys
sys.path.append('./lib')
sys.path.append('./')
sys.path.append('./main')
os.environ["OMP_NUM_THREADS"] = "1"
import argparse

parser = argparse.ArgumentParser(description='Visual VitLifter')
parser.add_argument('--cfg', type=str, required=True)
parser.add_argument('--ckpt', type=str, required=True)
parser.add_argument('--sets', type=str, default='3dpw-train-noaug,3dpw-oc,3dpw-pc,3dpw-crowd')
parser.add_argument('--ocpc_mode', type=str, default='test', choices=['all', 'test'])
parser.add_argument('--num', type=int, default=8, help='so mau moi nhom (random / worst)')
parser.add_argument('--max_samples', type=int, default=2000, help='so mau chay lifter de chon (moi tap)')
parser.add_argument('--K', type=int, default=10)
parser.add_argument('--S', type=int, default=4)
parser.add_argument('--eta', type=float, default=1.0)
parser.add_argument('--seed', type=int, default=0)
parser.add_argument('--batch_size', type=int, default=64)
parser.add_argument('--out_dir', type=str, default='output/vis_vit_lifter')
parser.add_argument('--gpu', type=str, default='0')
args = parser.parse_args()
os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from core.config import cfg, update_config
update_config(args.cfg)
cfg.DATASET.pw3d_test_gt2d = True          # tap test tra GT 2D mesh de ve
assert cfg.DATASET.pw3d_2d_source == 'halpe26', 'VitLifter: 2D 3DPW la file Halpe-26 ViTPose'

import __init_path  # noqa: F401
from utils.jotr_dataset import get_test_dataset
from models.img_lifter import aggregate
from models.vit_lifter import vit_inputs
from utils.img_lifter_tools import pa_errors
from utils.vit_lifter_tools import load_vit_lifter

# Khung xuong H36M-17 (Pelvis, R_Hip, R_Knee, R_Ankle, L_Hip, L_Knee, L_Ankle, Torso, Neck, Nose, Head_top,
# L_Shoulder, L_Elbow, L_Wrist, R_Shoulder, R_Elbow, R_Wrist)
SKEL = ((0, 7), (7, 8), (8, 9), (9, 10), (8, 11), (11, 12), (12, 13), (8, 14), (14, 15), (15, 16),
        (0, 1), (1, 2), (2, 3), (0, 4), (4, 5), (5, 6))


@torch.no_grad()
def run_set(name, lifter, enc, eps, dev):
    """Chay lifter tren max_samples mau ngau nhien -> dict mang numpy + chi so mau trong dataset."""
    cfg.DATASET.pw3d_ocpc_test_only = (args.ocpc_mode == 'test') and name in ('3dpw-oc', '3dpw-pc')
    ds = get_test_dataset(name, None)
    reg = torch.as_tensor(ds.h36m_joint_regressor, dtype=torch.float32, device=dev)
    rng = np.random.RandomState(args.seed)
    idx = np.arange(len(ds))
    if args.max_samples and args.max_samples < len(ds):
        idx = np.sort(rng.choice(len(ds), args.max_samples, replace=False))
    sub = Subset(ds, idx.tolist())
    loader = DataLoader(sub, batch_size=args.batch_size, shuffle=False, num_workers=cfg.DATASET.workers)
    hm = torch.tensor([cfg.output_hm_shape[2], cfg.output_hm_shape[1]], dtype=torch.float32, device=dev)
    out = {k: [] for k in ('pred', 'gt', 'xy', 'conf', 'jm', 'g2', 'gin', 'mpjpe', 'pa')}
    for bi, (inputs, targets, meta) in enumerate(loader):
        img = inputs['img'].to(dev).float()
        B = img.shape[0]
        joints = inputs['joints'].to(dev).float()
        jm = inputs['joints_mask'].to(dev).float().reshape(B, 17)
        cam = inputs['cam_bbox'].to(dev).float()
        if 'orig_joint_cam' in targets:                                     # tap train
            gt = targets['orig_joint_cam'].to(dev).float()
        else:
            gt = torch.einsum('jk,bkc->bjc', reg, targets['smpl_mesh_cam'].to(dev).float())
        gt = gt - gt[:, 0:1]
        cond7, fj, f8, xy, log_b = vit_inputs(enc, img, joints, jm, cam, eps)
        gen = torch.Generator(device=dev)
        gen.manual_seed(int(args.seed) + bi)
        hyp = lifter.sample(cond7, fj, f8, K=args.K, S=args.S, eta=args.eta, generator=gen)
        pred = aggregate(hyp, xy, log_b, 'mean')
        pred = pred - pred[:, 0:1]
        g2 = targets['orig_joint_img'].to(dev).float()[..., :2] / hm * 2 - 1
        gin = meta['orig_joint_trunc'].to(dev).float().reshape(B, 17)
        out['pred'].append(pred.cpu().numpy()); out['gt'].append(gt.cpu().numpy())
        out['xy'].append(xy.cpu().numpy()); out['conf'].append(joints[..., 2].cpu().numpy())
        out['jm'].append(jm.cpu().numpy()); out['g2'].append(g2.cpu().numpy()); out['gin'].append(gin.cpu().numpy())
        out['mpjpe'].append((torch.norm(pred - gt, dim=-1).mean(-1) * 1000).cpu().numpy())
        out['pa'].append(pa_errors(pred, gt).cpu().numpy())
    out = {k: np.concatenate(v) for k, v in out.items()}
    out['ds'], out['idx'] = ds, idx
    print(f'[{name}] {len(idx)} mau | MPJPE17 {out["mpjpe"].mean():.2f} | PA {out["pa"].mean():.2f} (K{args.K} S{args.S})')
    return out


def get_crop(ds, i):
    """Anh crop 256 (RGB [0,1]) cua mau i (dataset test / train-noaug khong ngau nhien -> trung luc chay lifter)."""
    inputs, _, _ = ds[int(i)]
    img = inputs['img']
    img = img.numpy() if torch.is_tensor(img) else np.asarray(img)
    return np.clip(img.transpose(1, 2, 0), 0, 1)


def to_plot(p):
    """He camera (x phai, y xuong, z truoc) -> truc ve (X = x, Y = z, Z = -y)."""
    return p[:, 0], p[:, 2], -p[:, 1]


def draw_3d(ax, gt, pred, jm, elev, azim, title):
    for (a, b) in SKEL:
        for p, c in ((gt, 'tab:green'), (pred, 'tab:red')):
            X, Y, Z = to_plot(p[[a, b]])
            ax.plot(X, Y, Z, color=c, lw=2)
    X, Y, Z = to_plot(gt)
    ax.scatter(X, Y, Z, color='tab:green', s=10)
    X, Y, Z = to_plot(pred)
    ok = jm > 0
    ax.scatter(X[ok], Y[ok], Z[ok], color='tab:red', s=10)
    ax.scatter(X[~ok], Y[~ok], Z[~ok], facecolors='none', edgecolors='black', s=60, lw=1.5)   # khop ViTPose thieu
    allp = np.concatenate([gt, pred], 0)
    X, Y, Z = to_plot(allp)
    c = np.array([X.mean(), Y.mean(), Z.mean()])
    r = max(np.ptp(X), np.ptp(Y), np.ptp(Z)) / 2 + 0.05
    ax.set_xlim(c[0] - r, c[0] + r); ax.set_ylim(c[1] - r, c[1] + r); ax.set_zlim(c[2] - r, c[2] + r)
    try:
        ax.set_box_aspect((1, 1, 1))
    except Exception:
        pass
    ax.view_init(elev=elev, azim=azim)
    ax.set_xlabel('x'); ax.set_ylabel('z (sau)'); ax.set_zlabel('-y')
    ax.tick_params(labelsize=6)
    ax.set_title(title, fontsize=8)


def draw_2d(ax, img, xy, conf, jm, g2, gin):
    ax.imshow(img)
    px = (xy + 1) * 128.0
    gp = (g2 + 1) * 128.0
    for (a, b) in SKEL:
        if jm[a] > 0 and jm[b] > 0:
            ax.plot(px[[a, b], 0], px[[a, b], 1], color='yellow', lw=1)
    for j in range(17):
        if gin[j] > 0:
            ax.scatter(gp[j, 0], gp[j, 1], marker='x', color='white', s=18, lw=1)
        if jm[j] > 0:
            col = 'red' if conf[j] < 0.3 else ('orange' if conf[j] < 0.6 else 'lime')
            ax.scatter(px[j, 0], px[j, 1], color=col, s=14)
    ax.set_xlim(0, 256); ax.set_ylim(256, 0)
    ax.axis('off')


def make_figure(name, r, sel, tag, path):
    n = len(sel)
    fig = plt.figure(figsize=(13, 4.2 * n))
    for row, k in enumerate(sel):
        i = r['idx'][k]
        img = get_crop(r['ds'], i)
        jm = r['jm'][k]
        title = (f'{name} #{i} | MPJPE17 {r["mpjpe"][k]:.1f} PA {r["pa"][k]:.1f} mm | '
                 f'khop thieu {int((jm <= 0).sum())}/17')
        ax = fig.add_subplot(n, 3, row * 3 + 1)
        draw_2d(ax, img, r['xy'][k], r['conf'][k], jm, r['g2'][k], r['gin'][k])
        ax.set_title(title + '\nViTPose (mau theo conf) | GT 2D (x trang)', fontsize=8)
        ax = fig.add_subplot(n, 3, row * 3 + 2, projection='3d')
        draw_3d(ax, r['gt'][k], r['pred'][k], jm, 10, -90, 'goc camera | GT xanh la, pred do')
        ax = fig.add_subplot(n, 3, row * 3 + 3, projection='3d')
        draw_3d(ax, r['gt'][k], r['pred'][k], jm, 15, -20, 'goc ben canh (thay do sau)')
    fig.suptitle(f'{name} - {tag} ({n} mau)', fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.995))
    fig.savefig(path, dpi=70)
    plt.close(fig)
    print('luu', path)


def main():
    dev = torch.device('cuda')
    lifter, enc, ck = load_vit_lifter(args.ckpt, dev)
    eps = float(ck.get('vit_logb_eps', cfg.IMGLIFT.vit_logb_eps))
    os.makedirs(args.out_dir, exist_ok=True)
    for name in [s.strip() for s in args.sets.split(',') if s.strip()]:
        r = run_set(name, lifter, enc, eps, dev)
        n = min(args.num, len(r['idx']))
        rng = np.random.RandomState(args.seed + 1)
        make_figure(name, r, rng.choice(len(r['idx']), n, replace=False), 'ngau nhien',
                    os.path.join(args.out_dir, f'{name}_random.jpg'))
        make_figure(name, r, np.argsort(-r['mpjpe'])[:n], 'MPJPE cao nhat',
                    os.path.join(args.out_dir, f'{name}_worst.jpg'))


if __name__ == '__main__':
    main()
