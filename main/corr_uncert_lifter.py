"""
main/corr_uncert_lifter.py -- (2026-11-08) Do tuong quan giua DO BAT DINH tung khop va LOI THAT tung khop.
Muc dich: kiem tra tien de cua y tuong "hypergraph dan dat boi do bat dinh" truoc khi code Student moi.
CHI DANH GIA (no_grad), khong train, khong sua code cu.

Pipeline giong Student v4/v5: OpenPose / HHRNet -> Refiner 2D day (dong bang) -> ImgLifter (K gia thuyet, S buoc,
gop mean). Moi (mau, khop) thu:
  - Tin hieu bat dinh:  log_b (Refiner, 2D) | spread = trung binh khoang cach K gia thuyet toi mean (mm, 3D)
                        | miss = detector sot khop (0/1) | comb = trung binh hang (rank) cua log_b va spread
  - Loi that:           e3d = loi 3D khop, root-relative (mm) | e3d_pa = loi 3D khop sau Procrustes (mm)
                        | e2d = loi 2D Refiner so GT 2D mesh (px crop 256)
Bao cao (moi tap):
  [A] Spearman gop toan bo (mau, khop)               -> bi lan voi loai khop (co tay luon kho hon hong)
  [B] Spearman TRONG TUNG MAU (xep hang 17 khop), TB  -> quan trong nhat: trong 1 nguoi, khop nao dang tin?
  [C] Spearman TRONG TUNG LOAI KHOP (qua cac mau), TB -> cung 1 khop, mau nao kho hon?
  [D] AUROC phat hien khop xau (e3d_pa >= phan vi 80% cua tap)
  [E] Loi TB theo 5 nhom (quintile) cua tin hieu     -> nhom bat dinh cao co loi cao that khong
Doc ket qua: [B] >= ~0.4 va AUROC >= ~0.7 -> tin hieu du tot de dan duong truyen tin trong hypergraph.

Chay (K, S mac dinh = dung nhu Student v4/v5):
  python main/corr_uncert_lifter.py --cfg config/train_img_lifter_dense48_3dpw_markov.yaml \
    --ckpt experiment/img_lifter_dense48_3dpw_markov/checkpoint/last.pth.tar \
    --sets 3dpw-oc,3dpw-crowd,3dpw-pc,3dpw --ocpc_mode test --gpu 0
  (thu nhanh: --max_samples 2000)
"""
import os, sys
sys.path.append('./lib')
sys.path.append('./')
os.environ["OMP_NUM_THREADS"] = "1"
import argparse

parser = argparse.ArgumentParser(description='Tuong quan do bat dinh - loi tung khop (ImgLifter)')
parser.add_argument('--cfg', type=str, required=True, help='yml da dung de train ImgLifter')
parser.add_argument('--ckpt', type=str, required=True, help='checkpoint ImgLifter')
parser.add_argument('--refiner_ckpt', type=str, default='', help='rong = refiner luu trong ckpt ImgLifter')
parser.add_argument('--sets', type=str, default='3dpw-oc,3dpw-crowd,3dpw-pc,3dpw')
parser.add_argument('--ocpc_mode', type=str, default='test', choices=['all', 'test'])
parser.add_argument('--K', type=int, default=5)
parser.add_argument('--S', type=int, default=2)
parser.add_argument('--eta', type=float, default=1.0)
parser.add_argument('--bad_q', type=float, default=0.8, help='khop xau = e3d_pa >= phan vi nay cua tap')
parser.add_argument('--max_samples', type=int, default=0, help='0 = toan bo tap')
parser.add_argument('--batch_size', type=int, default=0, help='0 = cfg.TEST.batch_size')
parser.add_argument('--seed', type=int, default=0)
parser.add_argument('--gpu', type=str, default='0')
args = parser.parse_args()
os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu

import torch
import numpy as np
from torch.utils.data import DataLoader, Subset

from core.config import cfg, update_config
update_config(args.cfg)
cfg.DATASET.pw3d_test_gt2d = True          # can GT 2D mesh de do loi 2D Refiner
assert cfg.DATASET.pw3d_2d_source == 'openpose', 'Benchmark: dau vao 3DPW la OpenPose / HHRNet'

import __init_path  # noqa: F401
from utils.jotr_dataset import get_test_dataset
from utils.img_lifter_tools import load_img_lifter, load_frozen_refiner, refiner_inputs, lifter_det
from models.img_lifter import ImgLifter, aggregate

J17 = ['Pelvis', 'RHip', 'RKnee', 'RAnkle', 'LHip', 'LKnee', 'LAnkle', 'Torso', 'Neck', 'Nose', 'Head',
       'LShoulder', 'LElbow', 'LWrist', 'RShoulder', 'RElbow', 'RWrist']


def pa_joint_errors(pred, gt):
    """Procrustes tung mau (giong pa_errors) nhung tra loi TUNG KHOP. (B,J,3) met -> (B,J) mm."""
    X = pred.double() - pred.double().mean(1, keepdim=True)
    Y = gt.double() - gt.double().mean(1, keepdim=True)
    var_x = (X ** 2).sum((1, 2)).clamp_min(1e-12)
    Km = X.transpose(1, 2) @ Y
    U, _, Vh = torch.linalg.svd(Km)
    V = Vh.transpose(1, 2)
    Z = torch.eye(3, dtype=X.dtype, device=X.device).repeat(X.shape[0], 1, 1)
    Z[:, -1, -1] = torch.sign(torch.det(U @ V.transpose(1, 2)))
    R = V @ Z @ U.transpose(1, 2)
    s = torch.einsum('bii->b', R @ Km) / var_x
    aligned = s[:, None, None] * (X @ R.transpose(1, 2)) + gt.double().mean(1, keepdim=True)
    return (torch.norm(aligned - gt.double(), dim=-1) * 1000).float()


# ---------------- thong ke (numpy, khong can scipy) ----------------
def rankdata(x, axis=-1):
    """Hang theo truc. 1D: tie lay hang trung binh (dung cho tin hieu 0/1 nhu 'miss').
    Nhieu chieu (xep hang trong tung mau): tin hieu lien tuc, tie hiem -> dung argsort."""
    x = np.asarray(x)
    if x.ndim == 1:
        _, inv, cnt = np.unique(x, return_inverse=True, return_counts=True)
        start = np.concatenate([[0], np.cumsum(cnt)[:-1]])
        return (start + (cnt - 1) / 2.0)[inv].astype(np.float64)        # hang TB cho nhom bang nhau
    return np.argsort(np.argsort(x, axis=axis, kind='stable'), axis=axis, kind='stable').astype(np.float64)


def pearson_rows(a, b):
    """Pearson theo tung hang (N, n) -> (N,), hang co phuong sai 0 -> nan."""
    a = a - a.mean(-1, keepdims=True)
    b = b - b.mean(-1, keepdims=True)
    den = np.sqrt((a ** 2).sum(-1) * (b ** 2).sum(-1))
    with np.errstate(invalid='ignore', divide='ignore'):
        return (a * b).sum(-1) / den


def spearman(x, y):
    """Spearman 1 vector."""
    return float(pearson_rows(rankdata(x)[None], rankdata(y)[None])[0])


def auroc(score, label):
    """AUROC qua Mann-Whitney. label 0/1."""
    label = label.astype(bool)
    npos, nneg = int(label.sum()), int((~label).sum())
    if npos == 0 or nneg == 0:
        return float('nan')
    r = rankdata(score) + 1.0
    return float((r[label].sum() - npos * (npos + 1) / 2) / (npos * nneg))


@torch.no_grad()
def collect(lifter, refiner, loader, dev, reg, max_samples=0):
    hm = torch.tensor([cfg.output_hm_shape[2], cfg.output_hm_shape[1]], dtype=torch.float32, device=dev)
    regt = torch.as_tensor(reg, dtype=torch.float32, device=dev)
    out = {k: [] for k in ('log_b', 'spread', 'miss', 'e3d', 'e3d_pa', 'e2d', 'v2d')}
    seen = 0
    for bi, (inputs, targets, meta) in enumerate(loader):
        if max_samples and seen >= max_samples:
            break
        img = inputs['img'].to(dev).float()
        B = img.shape[0]
        j2 = inputs['joints'].to(dev).float()[..., :2]
        jm = inputs['joints_mask'].to(dev).float().reshape(B, 17)
        cam = inputs['cam_bbox'].to(dev).float()
        gt17 = torch.einsum('jk,bkc->bjc', regt, targets['smpl_mesh_cam'].to(dev).float())
        gt17 = gt17 - gt17[:, 0:1]
        g2 = targets['orig_joint_img'].to(dev).float()[..., :2] / hm * 2 - 1      # GT 2D [-1,1] crop

        nc = int(getattr(lifter, 'num_ctx', 0))
        xy, log_b, f8, fj = refiner_inputs(refiner, img, j2, jm, 17 + nc)
        cond7 = ImgLifter.make_cond7(xy, log_b, lifter_det(jm, nc), cam)
        gen = torch.Generator(device=dev)
        gen.manual_seed(int(args.seed) + bi)                                      # giong eval_img_lifter
        hyp = lifter.sample(cond7, fj, f8, K=args.K, S=args.S, eta=args.eta, generator=gen)   # (B,K,17,3)
        hyp = hyp - hyp[:, :, 0:1]                                               # root-relative tung gia thuyet
        pred = aggregate(hyp, xy, log_b, 'mean')
        pred = pred - pred[:, 0:1]

        # do phan tan K gia thuyet: khoang cach TB toi trung binh gia thuyet (mm), tung khop
        mu = hyp.mean(1, keepdim=True)
        spread = torch.norm(hyp - mu, dim=-1).mean(1) * 1000                     # (B,17)

        out['log_b'].append(log_b[:, :17].cpu().numpy())
        out['spread'].append(spread.cpu().numpy())
        out['miss'].append((jm <= 0).float().cpu().numpy())
        out['e3d'].append((torch.norm(pred - gt17, dim=-1) * 1000).cpu().numpy())
        out['e3d_pa'].append(pa_joint_errors(pred, gt17).cpu().numpy())
        out['e2d'].append((torch.norm(xy[:, :17] - g2, dim=-1) * 128.0).cpu().numpy())   # px crop 256
        # GT 2D chi tin duoc o khop nam trong crop (orig_joint_trunc) -> mask cho loi 2D
        out['v2d'].append((meta['orig_joint_trunc'].float().reshape(B, 17) > 0).float().numpy())
        seen += B
    return {k: np.concatenate(v, 0) for k, v in out.items()}


def report(label, d, bad_q):
    N = d['e3d'].shape[0]
    # tin hieu ket hop: trung binh hang (toan tap) cua log_b va spread
    rb = rankdata(d['log_b'].reshape(-1)).reshape(N, 17)          # 1D -> hang toan tap
    rs = rankdata(d['spread'].reshape(-1)).reshape(N, 17)
    d = dict(d, comb=(rb + rs) / 2)
    sigs = [('log_b', 'log_b Refiner 2D'), ('spread', 'spread K gia thuyet'), ('comb', 'ket hop (rank TB)'),
            ('miss', 'detector sot (0/1)')]
    errs = [('e3d', 'loi 3D'), ('e3d_pa', 'loi 3D PA'), ('e2d', 'loi 2D Refiner')]
    bad = d['e3d_pa'] >= np.quantile(d['e3d_pa'], bad_q)

    lines = ['=' * 104, f'{label} | n mau = {N} | K={args.K} S={args.S} | ty le khop sot {100 * d["miss"].mean():.1f}%'
             f' | khop xau = e3d_pa >= {np.quantile(d["e3d_pa"], bad_q):.1f} mm (phan vi {int(100 * bad_q)}%)',
             '=' * 104,
             f'{"tin hieu":24s} | {"vs loi":14s} | {"[A] gop":>8s} | {"[B] trong mau":>13s} | '
             f'{"[C] trong loai khop":>19s} | {"[D] AUROC xau":>13s}']
    for sk, sn in sigs:
        s = d[sk]
        for ek, en in errs:
            e = d[ek]
            # loi 2D: chi dung khop co GT 2D hop le (trong crop); [B] chi tren mau du 17 khop hop le
            v = d['v2d'] > 0 if ek == 'e2d' else np.ones_like(e, dtype=bool)
            A = spearman(s[v], e[v])
            rows = v.all(1)
            if sk == 'miss' or rows.sum() < 2:
                Bm = float('nan')                                               # 0/1 trong mau: hang vo nghia
            else:
                Bm = float(np.nanmean(pearson_rows(rankdata(s[rows], 1), rankdata(e[rows], 1))))
            Cm = float(np.nanmean([spearman(s[v[:, j], j], e[v[:, j], j]) if v[:, j].sum() > 2 else np.nan
                                   for j in range(17)]))
            D = auroc(s.reshape(-1), bad.reshape(-1)) if ek == 'e3d_pa' else float('nan')
            ds = '' if np.isnan(D) else f'{D:13.3f}'
            lines.append(f'{sn:24s} | {en:14s} | {A:8.3f} | {Bm:13.3f} | {Cm:19.3f} | {ds:>13s}')
        lines.append('-' * 104)

    # [E] loi TB theo quintile tin hieu (toan tap)
    lines.append('[E] loi 3D PA TB (mm) theo 5 nhom tin hieu, thap -> cao:')
    for sk, sn in sigs[:3]:
        s = d[sk].reshape(-1)
        e = d['e3d_pa'].reshape(-1)
        qs = np.quantile(s, [0.2, 0.4, 0.6, 0.8])
        g = np.digitize(s, qs)
        vals = ' '.join(f'{e[g == i].mean():6.1f}' for i in range(5))
        lines.append(f'   {sn:24s} {vals}')
    m = d['miss'].reshape(-1) > 0
    lines.append(f'   {"detector sot / co":24s} {d["e3d_pa"].reshape(-1)[m].mean():6.1f} / '
                 f'{d["e3d_pa"].reshape(-1)[~m].mean():6.1f}')

    # theo tung khop: loi TB va Spearman [C] cua comb
    lines.append('[F] theo khop: loi 3D PA TB | spread TB | Spearman(comb, loi 3D PA) qua cac mau')
    for j in range(17):
        lines.append(f'   {j:2d} {J17[j]:10s} {d["e3d_pa"][:, j].mean():6.1f} mm | {d["spread"][:, j].mean():6.1f} mm | '
                     f'{spearman(d["comb"][:, j], d["e3d_pa"][:, j]):+.3f}')
    return '\n'.join(lines)


def main():
    dev = torch.device('cuda')
    lifter, ck = load_img_lifter(args.ckpt, dev)
    refiner = load_frozen_refiner(args.refiner_ckpt or ck['refiner_ckpt'], dev)
    bs = args.batch_size or cfg.TEST.batch_size
    os.makedirs(cfg.output_dir, exist_ok=True)
    for name in [s.strip() for s in args.sets.split(',') if s.strip()]:
        cfg.DATASET.pw3d_ocpc_test_only = (args.ocpc_mode == 'test') and name in ('3dpw-oc', '3dpw-pc')
        label = name + (' (chi seq test)' if cfg.DATASET.pw3d_ocpc_test_only else '')
        ds = get_test_dataset(name, None)
        reg = ds.h36m_joint_regressor
        if args.max_samples and args.max_samples < len(ds):
            ds = Subset(ds, list(range(args.max_samples)))
        loader = DataLoader(ds, batch_size=bs, shuffle=False, num_workers=cfg.DATASET.workers)
        d = collect(lifter, refiner, loader, dev, reg, args.max_samples)
        print(report(label, d, args.bad_q), flush=True)
        out = os.path.join(cfg.output_dir, f'corr_uncert_{name}.npz')
        np.savez_compressed(out, **d)                                           # luu de phan tich them
        print(f'===> Luu {out}')


if __name__ == '__main__':
    main()
