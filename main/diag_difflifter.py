"""
main/diag_difflifter.py
Chan doan DiffLifter tren 1 checkpoint (KHONG train, khong sua checkpoint).

Chay:
  python main/diag_difflifter.py --cfg config/finetune_difflifter.yaml --gpu 0 \
      --ckpt experiment/finetune_difflifter/latest_epoch.bin

In ra (log cung ghi vao <output_dir>/diag.txt va diag_summary.json):
  A. Thong ke 2D/3D cua tap train          -> goi y DIFFLIFT.scale (1/std 3D)
  B. Moi tap test: MPJPE mean-K / median-K / K=1 / min-K, PA, per-joint, spread
  C. Hieu chuan spread <-> loi             -> spread co bao hieu loi that khong
  D. Khop nhin ro (mask=1) vs khop mat (mask=0), loi theo so khop mat / mau
  E. Sai so pred_x0 theo t + 2 doi chung   -> mang co thuc su dung x_t khong
  F. Quet sampler S x K x eta (S=1 ~ hoi quy thuan tu 2D, khong co khu nhieu)
  G. Che nhan tao theo chi                 -> model co hoan thanh khop mat khong, spread co tang khong
  H. Toc do

Luu y: moi so lieu o E, F, G, H dung chung 1 tap con NGAU NHIEN (--sub_batches batch) cua --main_set.
Noise DDIM co dinh theo (seed, chi so batch) -> cac cau hinh so sanh duoc cung nhau.
"""

import os
import sys
sys.path.append('./lib')
sys.path.append('./')

os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"

import argparse
import contextlib
import json
import time

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

# Import qua 'core.*' / 'utils.*' (KHONG qua 'lib.core.*') de dung chung 1 module config voi dataset.
from core.config import update_config, cfg
from utils.jotr_dataset import get_train_dataset, get_test_dataset
from models.diff_lifter import DiffLifter, aggregate_jpma

JOINT_NAMES = ['Pelvis', 'R_Hip', 'R_Knee', 'R_Ankle', 'L_Hip', 'L_Knee', 'L_Ankle', 'Torso',
               'Neck', 'Nose', 'Head_top', 'L_Shoulder', 'L_Elbow', 'L_Wrist',
               'R_Shoulder', 'R_Elbow', 'R_Wrist']
LIMBS = {'R_leg': [1, 2, 3], 'L_leg': [4, 5, 6], 'L_arm': [11, 12, 13], 'R_arm': [14, 15, 16]}

# (S, K, eta): S = so buoc DDIM, K = so gia thuyet, eta = muc ngau nhien cua DDIM
SWEEP = [(1, 1, 1.0), (1, 5, 1.0), (3, 5, 1.0), (5, 1, 1.0), (5, 5, 1.0), (5, 5, 0.0),
         (5, 10, 1.0), (5, 20, 1.0), (10, 5, 1.0), (10, 5, 0.0), (10, 20, 1.0)]

LOG_PATH = None


def log(msg=''):
    print(msg)
    if LOG_PATH:
        with open(LOG_PATH, 'a', encoding='utf-8') as f:
            f.write(str(msg) + '\n')


def parse_args():
    p = argparse.ArgumentParser(description='Chan doan DiffLifter')
    p.add_argument('--cfg', type=str, default='config/finetune_difflifter.yaml')
    p.add_argument('--gpu', type=str, default='0')
    p.add_argument('--ckpt', type=str, default='',
                   help='Mac dinh: <cfg.DIFFLIFT.save_dir>/latest_epoch.bin')
    p.add_argument('--sets', type=str, default='3dpw,3dpw-oc,3dpw-pc,3dpw-crowd,Human36M',
                   help='Cac tap test cho muc B, C, D (tap loi se bi bo qua, chay tiep tap sau)')
    p.add_argument('--main_set', type=str, default='3dpw', help='Tap cho muc E, F, G, H')
    p.add_argument('--batch_size', type=int, default=256)
    p.add_argument('--max_batches', type=int, default=0, help='<=0: toan bo moi tap (muc B, C, D)')
    p.add_argument('--sub_batches', type=int, default=40,
                   help='So batch ngau nhien cache cho muc E, F, G, H')
    p.add_argument('--stats_sets', type=str, default='Human36M,MuCo,3dpw-train')
    p.add_argument('--stats_batches', type=int, default=30)
    p.add_argument('--skip_stats', action='store_true')
    p.add_argument('--skip_sweep', action='store_true')
    p.add_argument('--seed', type=int, default=0)
    return p.parse_args()


# ============================================================
# Tien ich
# ============================================================
def pa_mpjpe_batch(pred, gt):
    """Procrustes theo batch. pred, gt: (N, 17, 3). Tra ve loi PA tung mau (N,), met."""
    mu_p = pred.mean(1, keepdim=True)
    mu_g = gt.mean(1, keepdim=True)
    p = pred - mu_p
    g = gt - mu_g
    var_p = (p ** 2).sum(dim=(1, 2)).clamp(min=1e-8)
    K = p.transpose(1, 2) @ g
    U, S, Vh = torch.linalg.svd(K)
    V = Vh.transpose(1, 2)
    d = torch.sign(torch.det(V @ U.transpose(1, 2)))
    D = torch.diag_embed(torch.stack([torch.ones_like(d), torch.ones_like(d), d], dim=-1))
    R = V @ D @ U.transpose(1, 2)
    scale = (S * torch.diagonal(D, dim1=1, dim2=2)).sum(-1) / var_p
    aligned = scale[:, None, None] * (p @ R.transpose(1, 2)) + mu_g
    return torch.norm(aligned - gt, dim=-1).mean(-1)


def make_cond(inp, device):
    """Dieu kien 2D: [x_norm, y_norm, mask] giong finetune_difflifter.evaluate. (B, 17, 3)"""
    return torch.cat([inp['joints'][..., :2].float(), inp['joints_mask'].float()], dim=-1).to(device)


def get_gt(targets, reg, device):
    """GT 3D root-relative, met: orig_joint_cam neu co, nguoc lai H36M regressor x smpl_mesh_cam."""
    if 'orig_joint_cam' in targets:
        gt = targets['orig_joint_cam'].to(device).float()
    else:
        mesh = targets['smpl_mesh_cam'].to(device).float()
        gt = torch.matmul(reg[None], mesh)
    return gt - gt[:, 0:1, :]


def spread_of(preds):
    """preds (B, K, 17, 3) -> (B, 17) met. K = 1 -> 0 (std khong xac dinh)."""
    if preds.shape[1] < 2:
        return torch.zeros(preds.shape[0], preds.shape[2], device=preds.device)
    return preds.std(dim=1).norm(dim=-1)


@contextlib.contextmanager
def sampler(model, S=None, K=None, eta=None):
    """Doi tam thoi S / K / eta cua DiffLifter roi tra lai."""
    old = (model.sampling_timesteps, model.num_proposals, model.ddim_sampling_eta)
    if S is not None:
        model.sampling_timesteps = S
    if K is not None:
        model.num_proposals = K
    if eta is not None:
        model.ddim_sampling_eta = eta
    try:
        yield
    finally:
        model.sampling_timesteps, model.num_proposals, model.ddim_sampling_eta = old


@torch.no_grad()
def sample(model, cond, bi, seed):
    """K gia thuyet, noise co dinh theo (seed, bi). Tra ve (B, K, 17, 3) root-relative."""
    gen = torch.Generator(device=cond.device)
    gen.manual_seed(seed * 100003 + bi)
    _, _, preds = model(cond, generator=gen)
    return preds - preds[:, :, 0:1, :]


def metrics_from_preds(preds, gt, cond=None):
    mean = preds.mean(1)
    errs_k = torch.norm(preds - gt.unsqueeze(1), dim=-1).mean(-1)          # (B, K)
    # JPMA: moi khop chon gia thuyet khop 2D nhat (can cond); khong co cond -> bang mean
    jp = aggregate_jpma(preds, cond) if cond is not None else mean
    return {
        'jpma': torch.norm(jp - gt, dim=-1).mean(-1),                      # (B,)
        'err_j': torch.norm(mean - gt, dim=-1),                            # (B, 17) loi cua mean
        'pa': pa_mpjpe_batch(mean, gt),                                    # (B,)
        'mink': errs_k.min(1).values,                                      # (B,)
        'k1': errs_k[:, 0],                                                # (B,) gia thuyet dau tien
        'med': torch.norm(preds.median(1).values - gt, dim=-1).mean(-1),   # (B,)
        'spr_j': spread_of(preds),                                         # (B, 17)
    }


def make_loader(name, args, shuffle=False, seed=0):
    ds = get_test_dataset(name, None)
    gen = torch.Generator().manual_seed(seed) if shuffle else None
    loader = DataLoader(ds, batch_size=args.batch_size, shuffle=shuffle, num_workers=cfg.DATASET.workers,
                        pin_memory=True, generator=gen)
    reg = torch.as_tensor(ds.dataset.h36m_joint_regressor, dtype=torch.float32)
    return ds, loader, reg


def build_model(ckpt, device):
    dc = ckpt['difflift_cfg']
    # aggregate luon 'mean' o day: diag tu tinh ca mean lan JPMA tu K gia thuyet
    model = DiffLifter(num_joints=17, embed_dim=dc['embed_dim'], depth=dc['depth'], num_heads=dc['num_heads'],
                       timesteps=dc['timesteps'], sampling_timesteps=dc['sampling_timesteps'],
                       num_proposals=dc['num_proposals'], scale=dc['scale'], drop_path_rate=0.0,
                       cond_mode=dc.get('cond_mode', 'input'), aggregate='mean').to(device)
    model.load_state_dict(ckpt['model_state'], strict=True)
    model.eval()
    return model


def pearson(a, b):
    a = a.double() - a.double().mean()
    b = b.double() - b.double().mean()
    return float((a * b).sum() / (a.norm() * b.norm()).clamp(min=1e-12))


def rank(x):
    return torch.argsort(torch.argsort(x)).float()


# ============================================================
# A. Thong ke du lieu train
# ============================================================
def report_stats(args):
    log('\n' + '=' * 70)
    log('A. THONG KE TAP TRAIN (co augmentation nhu luc train)')
    log('=' * 70)
    for name in [s.strip() for s in args.stats_sets.split(',') if s.strip()]:
        try:
            ds = get_train_dataset(name, None)
            loader = DataLoader(ds, batch_size=512, shuffle=True, num_workers=cfg.DATASET.workers)
            g3, xy2, mk = [], [], []
            for bi, (inp, tgt, meta) in enumerate(loader):
                if bi >= args.stats_batches:
                    break
                g = tgt['orig_joint_cam'].float()
                g3.append(g - g[:, :1])
                xy2.append(inp['joints'][..., :2].float())
                mk.append(inp['joints_mask'].float()[..., 0])
            g = torch.cat(g3)
            xy = torch.cat(xy2)
            m = torch.cat(mk)
            body = g[:, 1:]
            std_all = body.reshape(-1).std().item()
            log(f'[{name}] n={g.shape[0]}')
            log(f'  3D (m): std gop (bo root)={std_all:.3f} | std theo truc={body.reshape(-1, 3).std(0).tolist()} '
                f'| |max|={g.abs().max().item():.2f} | ti le |toa do|>1.1 (bi clamp)={(body.abs() > 1.1).float().mean().item():.4f}')
            log('  3D std theo khop (m): ' + ', '.join(f'{k}={v:.2f}' for k, v in
                                                       zip(JOINT_NAMES, g.std(0).norm(dim=-1).tolist())))
            log(f'  => goi y scale ~ 1/std = {1.0 / std_all:.2f}  (DIFFLIFT.scale hien tai se thay o phan dau log)')
            v = xy[m > 0]
            log(f'  2D (chuan hoa): mean={v.mean(0).tolist()} | std={v.std(0).tolist()} '
                f'| ti le |x|>1={(v.abs() > 1).float().mean().item():.4f} | ti le khop mask=0={(m == 0).float().mean().item():.3f}')
        except Exception as e:
            log(f'[{name}] LOI: {type(e).__name__}: {e}')


# ============================================================
# B, C, D. Danh gia day du 1 tap test
# ============================================================
def eval_set(model, name, args, device):
    ds, loader, reg = make_loader(name, args)
    reg = reg.to(device)
    acc = {k: [] for k in ('jpma', 'err_j', 'pa', 'mink', 'k1', 'med', 'spr_j', 'msk_j')}
    for bi, (inp, tgt, meta) in enumerate(tqdm(loader, desc=name)):
        if args.max_batches > 0 and bi >= args.max_batches:
            break
        cond = make_cond(inp, device)
        gt = get_gt(tgt, reg, device)
        m = metrics_from_preds(sample(model, cond, bi, args.seed), gt, cond)
        for k, v in m.items():
            acc[k].append(v.cpu())
        acc['msk_j'].append(cond[..., 2].cpu())
    return {k: torch.cat(v) for k, v in acc.items()}


def report_main(name, r):
    mm = lambda x: float(x.mean()) * 1000
    err_s = r['err_j'].mean(1)
    spr = r['spr_j'][:, 1:]
    err = r['err_j'][:, 1:]
    log(f"\n[{name}] n={err_s.shape[0]}")
    log(f"  MPJPE (mm): mean-K={mm(err_s):.2f} | JPMA={mm(r['jpma']):.2f} | median-K={mm(r['med']):.2f} "
        f"| K=1 (gia thuyet dau)={mm(r['k1']):.2f} "
        f"| min-K={mm(r['mink']):.2f} | PA(mean)={mm(r['pa']):.2f}")
    log(f"  spread TB (bo root)={float(spr.mean()) * 1000:.2f} mm | loi TB (bo root)={float(err.mean()) * 1000:.2f} mm "
        f"| spread/loi={float(spr.mean() / err.mean()):.3f}")
    log('  per-joint MPJPE (mm): ' + ', '.join(f'{k}={v:.0f}' for k, v in
                                               zip(JOINT_NAMES, (r['err_j'].mean(0) * 1000).tolist())))
    log('  per-joint spread (mm): ' + ', '.join(f'{k}={v:.0f}' for k, v in
                                                zip(JOINT_NAMES, (r['spr_j'].mean(0) * 1000).tolist())))


def report_calibration(r):
    e = r['err_j'][:, 1:].reshape(-1)
    s = r['spr_j'][:, 1:].reshape(-1)
    log(f"  [C] hieu chuan tung khop (bo root): Pearson={pearson(s, e):.3f} | Spearman={pearson(rank(s), rank(e)):.3f}")
    es = r['err_j'][:, 1:].mean(1)
    ss = r['spr_j'][:, 1:].mean(1)
    sp_sample = pearson(rank(ss), rank(es))
    log(f"  [C] hieu chuan tung mau: Pearson={pearson(ss, es):.3f} | Spearman={sp_sample:.3f}")
    q = torch.quantile(s, torch.tensor([0.25, 0.5, 0.75], dtype=s.dtype))
    edges = [float('-inf'), q[0].item(), q[1].item(), q[2].item(), float('inf')]
    for i, nm in enumerate(['Q1 (spread thap)', 'Q2', 'Q3', 'Q4 (spread cao)']):
        sel = (s > edges[i]) & (s <= edges[i + 1])
        if sel.any():
            log(f"      {nm}: spread {float(s[sel].mean()) * 1000:.1f} mm | loi {float(e[sel].mean()) * 1000:.1f} mm "
                f"| n={int(sel.sum())}")
    return pearson(rank(s), rank(e))


def report_visibility(r):
    m = r['msk_j'][:, 1:] > 0
    e = r['err_j'][:, 1:]
    s = r['spr_j'][:, 1:]
    log(f"  [D] ti le khop mask=0 (bo root): {(~m).float().mean().item() * 100:.1f}%")
    if m.any():
        log(f"      khop nhin ro (mask=1): loi {float(e[m].mean()) * 1000:.1f} mm | spread {float(s[m].mean()) * 1000:.1f} mm")
    if (~m).any():
        log(f"      khop mat (mask=0)    : loi {float(e[~m].mean()) * 1000:.1f} mm | spread {float(s[~m].mean()) * 1000:.1f} mm")
    nmiss = (r['msk_j'] == 0).sum(1)
    for lo, hi in [(0, 0), (1, 1), (2, 3), (4, 17)]:
        sel = (nmiss >= lo) & (nmiss <= hi)
        if sel.any():
            log(f"      mau co {lo}-{hi} khop mat: MPJPE {float(r['err_j'][sel].mean(1).mean()) * 1000:.1f} mm | n={int(sel.sum())}")


# ============================================================
# E. Sai so theo t va doi chung x_t
# ============================================================
@torch.no_grad()
def cache_subset(name, args, device):
    ds, loader, reg = make_loader(name, args, shuffle=True, seed=args.seed)
    reg = reg.to(device)
    out = []
    for bi, (inp, tgt, meta) in enumerate(loader):
        if bi >= args.sub_batches:
            break
        out.append((make_cond(inp, device), get_gt(tgt, reg, device)))
    return out


@torch.no_grad()
def report_t_curve(model, cache, seed):
    s = model.scale
    ts = [999, 799, 599, 399, 199, 99, 0]
    sig = torch.cat([g[:, 1:].reshape(-1) for _, g in cache]).std().item()
    log('\n' + '=' * 70)
    log('E. SAI SO pred_x0 THEO t (x_t tao tu GT + nhieu; 3 dieu kien)')
    log('=' * 70)
    log(f'  std gop cua GT 3D (bo root) = {sig:.3f} m | scale = {s} | SNR = sqrt(a/(1-a)) * std * scale')
    log('  cot "dung": x_t dung cua mau | "mau khac": x_t cua mau ke ben | "zero": x_t = 0')
    log('  Neu "dung" ~ "mau khac" ~ "zero" => mang BO QUA x_t o muc t do (chi hoi quy tu 2D).')
    res = {t: np.zeros(4) for t in ts}
    for bi, (cond, gt) in enumerate(cache):
        B = gt.shape[0]
        x0 = (gt * s).clamp(-1.1 * s, 1.1 * s)
        g = torch.Generator(device=x0.device)
        g.manual_seed(seed * 100003 + bi)
        noise = torch.randn(x0.shape, device=x0.device, generator=g)
        for t in ts:
            tt = torch.full((B,), t, device=x0.device, dtype=torch.long)
            x_in = model.q_sample(x0, tt, noise).clamp(-1.1 * s, 1.1 * s) / s

            def err(xi):
                p = model._denoise(cond, xi, tt)
                return torch.norm(p - gt, dim=-1).mean(-1).sum().item()
            res[t] += np.array([err(x_in), err(x_in.roll(1, dims=0)), err(torch.zeros_like(x_in)), B])
    log(f"  {'t':>5} | {'alpha_bar':>9} | {'SNR':>6} | {'dung':>8} | {'mau khac':>8} | {'zero':>8}   (MPJPE mm)")
    for t in ts:
        a = model.alphas_cumprod[t].item()
        snr = (a / max(1 - a, 1e-8)) ** 0.5 * sig * s
        n = res[t][3]
        log(f"  {t:>5} | {a:>9.4f} | {snr:>6.2f} | {res[t][0] / n * 1000:>8.1f} | {res[t][1] / n * 1000:>8.1f} "
            f"| {res[t][2] / n * 1000:>8.1f}")


# ============================================================
# F. Quet sampler
# ============================================================
def report_sweep(model, cache, seed):
    log('\n' + '=' * 70)
    log('F. QUET SAMPLER (tap con ngau nhien). S=1: chi 1 buoc tai t=999 ~ hoi quy thuan tu 2D')
    log('=' * 70)
    d = (model.sampling_timesteps, model.num_proposals, model.ddim_sampling_eta)
    log(f"  {'S':>3} {'K':>3} {'eta':>4} | {'MPJPE(mean)':>11} | {'JPMA':>6} | {'PA':>6} | {'min-K':>6} "
        f"| {'spread':>6}   (mm)")
    for S, K, eta in SWEEP:
        tot = np.zeros(6)
        with sampler(model, S, K, eta):
            for bi, (cond, gt) in enumerate(cache):
                m = metrics_from_preds(sample(model, cond, bi, seed), gt, cond)
                tot += np.array([m['err_j'].mean(1).sum().item(), m['pa'].sum().item(), m['mink'].sum().item(),
                                 m['spr_j'][:, 1:].mean(1).sum().item(), m['jpma'].sum().item(), gt.shape[0]])
        n = tot[5]
        mark = '  <- mac dinh checkpoint' if (S, K, eta) == (d[0], d[1], d[2]) else ''
        log(f"  {S:>3} {K:>3} {eta:>4.1f} | {tot[0] / n * 1000:>11.2f} | {tot[4] / n * 1000:>6.2f} "
            f"| {tot[1] / n * 1000:>6.2f} | {tot[2] / n * 1000:>6.2f} | {tot[3] / n * 1000:>6.2f}{mark}")


# ============================================================
# G. Che nhan tao theo chi
# ============================================================
def agg_pose(preds, cond, agg):
    """Gop K gia thuyet: 'mean' hoac 'jpma'."""
    return aggregate_jpma(preds, cond) if agg == 'jpma' else preds.mean(1)


@torch.no_grad()
def report_limb_drop(model, cache, seed, tag='sampler mac dinh', agg='mean'):
    log('\n' + '=' * 70)
    log(f'G. CHE NHAN TAO THEO CHI [{tag}] (xy = 0, mask = 0 cho ca chi; chi tinh mau ma chi do ban dau nhin ro)')
    log('=' * 70)
    acc = {g: np.zeros(7) for g in LIMBS}   # n, eb, ed, sb, sd, ob, od
    for bi, (cond, gt) in enumerate(cache):
        pb = sample(model, cond, bi, seed)
        eb = torch.norm(agg_pose(pb, cond, agg) - gt, dim=-1)
        sb = spread_of(pb)
        for gname, idx in LIMBS.items():
            other = [j for j in range(1, 17) if j not in idx]
            vis = (cond[:, idx, 2] > 0).all(dim=1)
            if vis.sum() == 0:
                continue
            c2 = cond.clone()
            c2[:, idx, :] = 0
            pd = sample(model, c2, bi, seed)
            ed = torch.norm(agg_pose(pd, c2, agg) - gt, dim=-1)
            sd = spread_of(pd)
            acc[gname] += np.array([
                vis.sum().item(),
                eb[vis][:, idx].mean(1).sum().item(), ed[vis][:, idx].mean(1).sum().item(),
                sb[vis][:, idx].mean(1).sum().item(), sd[vis][:, idx].mean(1).sum().item(),
                eb[vis][:, other].mean(1).sum().item(), ed[vis][:, other].mean(1).sum().item()])
    log('  Doc: loi chi bi che tang bao nhieu (Delta); spread chi bi che co tang khong; loi cac khop con lai co doi khong.')
    for gname, a in acc.items():
        n = a[0]
        if n == 0:
            log(f'  {gname}: khong co mau nao')
            continue
        v = a[1:] / n * 1000
        log(f"  {gname:>6} (n={int(n)}): loi chi {v[0]:.1f} -> {v[1]:.1f} mm (Delta {v[1] - v[0]:+.1f}) "
            f"| spread chi {v[2]:.1f} -> {v[3]:.1f} mm | loi cac khop con lai {v[4]:.1f} -> {v[5]:.1f} mm (Delta {v[5] - v[4]:+.1f})")


# ============================================================
# H. Toc do
# ============================================================
@torch.no_grad()
def report_speed(model, cache):
    log('\n' + '=' * 70)
    log('H. TOC DO (1 batch co dinh)')
    log('=' * 70)
    cond, _ = cache[0]
    B = cond.shape[0]
    n_params = sum(p.numel() for p in model.parameters())
    log(f'  so tham so: {n_params / 1e6:.2f} M | batch = {B}')
    for S, K in [(1, 1), (5, 5)]:
        with sampler(model, S, K, None):
            for _ in range(2):
                sample(model, cond, 0, 0)
            if cond.is_cuda:
                torch.cuda.synchronize()
            t0 = time.time()
            n = 5
            for _ in range(n):
                sample(model, cond, 0, 0)
            if cond.is_cuda:
                torch.cuda.synchronize()
            dt = (time.time() - t0) / n
        log(f'  S={S} K={K}: {dt * 1000:.1f} ms/batch | {dt / B * 1e6:.1f} us/mau')


# ============================================================
# Main
# ============================================================
def main():
    global LOG_PATH
    args = parse_args()
    if args.gpu:
        os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
    update_config(args.cfg)
    cfg.DATASET.lift_only = True      # khong doc anh, tra thang H36M-17
    LOG_PATH = os.path.join(cfg.output_dir, 'diag.txt')

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    ckpt_path = args.ckpt or os.path.join(cfg.DIFFLIFT.save_dir, 'latest_epoch.bin')
    ckpt = torch.load(ckpt_path, map_location='cpu')
    model = build_model(ckpt, device)

    log('=' * 70)
    log(f'DIAG DiffLifter | {time.strftime("%Y-%m-%d %H:%M:%S")} | device={device}')
    log(f'ckpt={ckpt_path} | epoch={ckpt.get("epoch", "?")} | best_val_mpjpe(ckpt)={ckpt.get("best_val_mpjpe", "?")}')
    log(f'difflift_cfg={ckpt["difflift_cfg"]}')
    log(f'train_list={list(cfg.DATASET.train_list)} | h36m/pw3d/muco 2D = '
        f'{cfg.DATASET.h36m_2d_source}/{cfg.DATASET.pw3d_2d_source}/{cfg.DATASET.muco_2d_source} '
        f'| halpe26_conf_thr={cfg.DATASET.halpe26_conf_thr}')
    log(f'sampler mac dinh: S={model.sampling_timesteps} K={model.num_proposals} eta={model.ddim_sampling_eta}')
    log('=' * 70)

    if not args.skip_stats:
        report_stats(args)

    summary = {}
    log('\n' + '=' * 70)
    log('B, C, D. DANH GIA TUNG TAP TEST (sampler mac dinh cua checkpoint)')
    log('=' * 70)
    for name in [s.strip() for s in args.sets.split(',') if s.strip()]:
        try:
            r = eval_set(model, name, args, device)
            report_main(name, r)
            sp = report_calibration(r)
            report_visibility(r)
            summary[name] = {
                'n': int(r['err_j'].shape[0]),
                'mpjpe_mean': float(r['err_j'].mean(1).mean()) * 1000,
                'mpjpe_jpma': float(r['jpma'].mean()) * 1000,
                'mpjpe_median': float(r['med'].mean()) * 1000,
                'mpjpe_k1': float(r['k1'].mean()) * 1000,
                'mink': float(r['mink'].mean()) * 1000,
                'pa': float(r['pa'].mean()) * 1000,
                'spread': float(r['spr_j'][:, 1:].mean()) * 1000,
                'spearman_joint': sp,
            }
        except Exception as e:
            log(f'[{name}] LOI: {type(e).__name__}: {e}')

    try:
        cache = cache_subset(args.main_set, args, device)
        log(f'\n[sub] {args.main_set}: {sum(c.shape[0] for c, _ in cache)} mau ngau nhien cho muc E, F, G, H')
        report_t_curve(model, cache, args.seed)
        if not args.skip_sweep:
            report_sweep(model, cache, args.seed)
        report_limb_drop(model, cache, args.seed, tag='sampler mac dinh, gop mean')
        report_limb_drop(model, cache, args.seed, tag='sampler mac dinh, gop JPMA', agg='jpma')
        # Duong chuan hoi quy thuan (S=1, K=1): so voi dong tren de biet DIFFUSION co giup khi mat khop khong
        # (neu loi chi bi che / Delta giong nhau -> loi ich den tu kenh conf + augmentation, khong phai diffusion)
        with sampler(model, 1, 1, None):
            report_limb_drop(model, cache, args.seed, tag='S=1 K=1 = hoi quy thuan')
        report_speed(model, cache)
    except Exception as e:
        log(f'[sub {args.main_set}] LOI: {type(e).__name__}: {e}')

    log('\n' + '=' * 70)
    log('TOM TAT (mm)')
    log('=' * 70)
    log(f"  {'tap':>12} | {'n':>6} | {'mean-K':>7} | {'JPMA':>7} | {'K=1':>7} | {'min-K':>7} | {'PA':>6} "
        f"| {'spread':>6} | {'Spearman':>8}")
    for name, v in summary.items():
        log(f"  {name:>12} | {v['n']:>6} | {v['mpjpe_mean']:>7.2f} | {v['mpjpe_jpma']:>7.2f} "
            f"| {v['mpjpe_k1']:>7.2f} | {v['mink']:>7.2f} "
            f"| {v['pa']:>6.2f} | {v['spread']:>6.2f} | {v['spearman_joint']:>8.3f}")
    with open(os.path.join(cfg.output_dir, 'diag_summary.json'), 'w', encoding='utf-8') as f:
        json.dump({'ckpt': ckpt_path, 'epoch': ckpt.get('epoch'), 'summary': summary}, f, indent=2)
    log(f'\nLog: {LOG_PATH}')


if __name__ == '__main__':
    main()
