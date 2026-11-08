"""
tools/analyze_corr_npz.py -- (2026-11-08) Phan tich lai .npz cua main/corr_uncert_lifter.py, CHI CPU (numpy).

Van de: spread / loi root-relative deu tang theo khoang cach toi Pelvis (Pelvis spread = 0) -> Spearman "trong mau"
[B] bi thoi phong vi tin hieu chi can "biet" loai khop. Script nay LOAI anh huong loai khop:
  moi khop j: doi tin hieu va loi thanh PHAN VI trong chinh loai khop j (qua toan bo mau cua tap).
  -> gia tri 0.9 = "khop nay bat dinh / loi hon 90% cac mau khac CUNG loai khop".
Bo Pelvis (spread = 0 theo dinh nghia), con 16 khop.

Bao cao moi tap:
  [B*]  Spearman trong tung mau giua phan vi tin hieu va phan vi loi (16 khop), trung binh qua mau
  [D*]  AUROC phat hien khop "xau bat thuong" = loi >= phan vi bad_q CUA LOAI KHOP DO
  [P@3] trong moi mau: 3 khop bat dinh nhat (phan vi) co nam trong 3 khop loi nhat (phan vi) khong
        -> precision@3, so voi ngau nhien = 3/16 = 0.1875
Nguong doc (dat truoc): [B*] >= ~0.3 va [D*] >= ~0.7 -> du co so cho hypergraph dan dat boi do bat dinh.

Chay:
  python tools/analyze_corr_npz.py experiment/_eval/corr_uncert_lifter_11-08_10-48-46
  (them --bad_q 0.8)
"""
import argparse
import glob
import os

import numpy as np

parser = argparse.ArgumentParser()
parser.add_argument('npz_dir', type=str, help='thu muc chua corr_uncert_*.npz')
parser.add_argument('--bad_q', type=float, default=0.8)
args = parser.parse_args()

J17 = ['Pelvis', 'RHip', 'RKnee', 'RAnkle', 'LHip', 'LKnee', 'LAnkle', 'Torso', 'Neck', 'Nose', 'Head',
       'LShoulder', 'LElbow', 'LWrist', 'RShoulder', 'RElbow', 'RWrist']
KEEP = list(range(1, 17))                      # bo Pelvis


def rank1d(x):
    """Hang 1D, tie lay hang trung binh."""
    _, inv, cnt = np.unique(x, return_inverse=True, return_counts=True)
    start = np.concatenate([[0], np.cumsum(cnt)[:-1]])
    return (start + (cnt - 1) / 2.0)[inv.reshape(-1)].astype(np.float64)


def pct_per_joint(x):
    """(N,J) -> phan vi [0,1] trong tung cot (tung loai khop)."""
    N = x.shape[0]
    out = np.empty_like(x, dtype=np.float64)
    for j in range(x.shape[1]):
        out[:, j] = rank1d(x[:, j]) / max(N - 1, 1)
    return out


def rank_rows(x):
    """Hang trong tung hang (mau). Tin hieu lien tuc -> argsort du."""
    return np.argsort(np.argsort(x, axis=1, kind='stable'), axis=1, kind='stable').astype(np.float64)


def pearson_rows(a, b):
    a = a - a.mean(1, keepdims=True)
    b = b - b.mean(1, keepdims=True)
    den = np.sqrt((a ** 2).sum(1) * (b ** 2).sum(1))
    with np.errstate(invalid='ignore', divide='ignore'):
        return (a * b).sum(1) / den


def auroc(score, label):
    label = label.astype(bool)
    npos, nneg = int(label.sum()), int((~label).sum())
    if npos == 0 or nneg == 0:
        return float('nan')
    r = rank1d(score) + 1.0
    return float((r[label].sum() - npos * (npos + 1) / 2) / (npos * nneg))


def prec_at_k(sig, err, k=3):
    """Ti le trung giua top-k tin hieu va top-k loi trong tung mau."""
    ts = np.argsort(-sig, axis=1)[:, :k]
    te = np.argsort(-err, axis=1)[:, :k]
    hit = [len(set(a) & set(b)) for a, b in zip(ts, te)]
    return float(np.mean(hit)) / k


def analyze(path):
    d = np.load(path)
    name = os.path.basename(path).replace('corr_uncert_', '').replace('.npz', '')
    N = d['e3d'].shape[0]
    sig = {'spread': pct_per_joint(d['spread'][:, KEEP]),
           'log_b': pct_per_joint(d['log_b'][:, KEEP])}
    sig['comb'] = (sig['spread'] + sig['log_b']) / 2
    err = {'e3d': pct_per_joint(d['e3d'][:, KEEP]),
           'e3d_pa': pct_per_joint(d['e3d_pa'][:, KEEP])}

    lines = ['=' * 92, f'{name} | n mau = {N} | 16 khop (bo Pelvis) | tin hieu & loi = phan vi TRONG loai khop',
             '=' * 92,
             f'{"tin hieu":10s} | {"vs loi":8s} | {"[B*] trong mau":>14s} | {"[D*] AUROC xau":>14s} | '
             f'{"[P@3]":>7s} (ngau nhien 0.188)']
    for sk in ('spread', 'log_b', 'comb'):
        for ek in ('e3d', 'e3d_pa'):
            s, e = sig[sk], err[ek]
            Bs = float(np.nanmean(pearson_rows(rank_rows(s), rank_rows(e))))
            bad = e >= args.bad_q                                              # top (1-bad_q) cua tung loai khop
            D = auroc(s.reshape(-1), bad.reshape(-1))
            P = prec_at_k(s, e, 3)
            lines.append(f'{sk:10s} | {ek:8s} | {Bs:14.3f} | {D:14.3f} | {P:7.3f}')
        lines.append('-' * 92)

    # theo khop: AUROC (comb vs e3d_pa) -> khop nao tin hieu dung duoc
    s, e = sig['comb'], err['e3d_pa']
    row = []
    for i, j in enumerate(KEEP):
        row.append(f'{J17[j]}={auroc(s[:, i], e[:, i] >= args.bad_q):.2f}')
    lines.append('AUROC theo khop (comb vs e3d_pa): ' + ', '.join(row))
    return '\n'.join(lines)


if __name__ == '__main__':
    files = sorted(glob.glob(os.path.join(args.npz_dir, 'corr_uncert_*.npz')))
    assert files, f'Khong thay corr_uncert_*.npz trong {args.npz_dir}'
    for f in files:
        print(analyze(f), flush=True)
