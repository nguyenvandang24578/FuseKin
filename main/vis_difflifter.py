"""
main/vis_difflifter.py
Visual cho DiffLifter (KHONG train): xem nhieu Gauss tren pose 3D trong nhu the nao va mang lam gi voi no.

Chay:
  python main/vis_difflifter.py --cfg config/finetune_difflifter.yaml --gpu 0 \
      --ckpt experiment/finetune_difflifter/latest_epoch.bin --set 3dpw --num_vis 6
  # chi xem mau bi mat khop (che khuat):  them  --min_missing 2

Anh duoc luu vao <vis_dir>/difflifter/ (vis_dir cua thu muc thi nghiem _eval):
  forward_XX.jpg : tung muc t.  Hang tren = x_t MANG THAY (GT + nhieu, xanh duong, GT xam mo).
                   Hang duoi = pred_x0 cua mang tu x_t do (do) so voi GT (xam).
  trace_XX.jpg   : DDIM tung buoc luc inference.  Hang tren = x vao mang (gia thuyet 0).
                   Hang duoi = pred_x0 cua K gia thuyet chong len nhau (moi gia thuyet 1 mau).
                   Cot cuoi = ket qua cuoi (mean do, GT xanh la / K gia thuyet).
  hist.jpg       : phan bo tin hieu (GT * scale) / nhieu / x_t o tung t -> thay SNR va anh huong cua scale.

Cach doc nhanh:
  - forward: o t lon x_t la "dam day" khong con hinh nguoi; neu hang duoi van ra pose dung thi mang chi dua vao 2D.
  - trace: neu o buoc dau cac mau cua K gia thuyet TRUNG nhau => chua co da dang tu x_t (hypothesis sup).
  - hist: neu duong tin hieu (hep) nam gon trong duong nhieu (rong) o hau het t => scale qua nho.
"""

import os
import sys
sys.path.append('./lib')
sys.path.append('./')

os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"

import argparse

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from core.config import update_config, cfg
from utils.jotr_dataset import get_test_dataset
# Dung lai ham cua diag_difflifter.py (cung thu muc main/)
from diag_difflifter import build_model, make_cond, get_gt, JOINT_NAMES

H36M_SKELETON = [
    (0, 1), (1, 2), (2, 3),           # chan phai
    (0, 4), (4, 5), (5, 6),           # chan trai
    (0, 7), (7, 8), (8, 9), (9, 10),  # cot song, dau
    (8, 14), (14, 15), (15, 16),      # tay phai
    (8, 11), (11, 12), (12, 13),      # tay trai
]
LIM = 1.1   # met (cung muc clamp cua DiffLifter khi scale = 1)


def parse_args():
    p = argparse.ArgumentParser(description='Visual DiffLifter')
    p.add_argument('--cfg', type=str, default='config/finetune_difflifter.yaml')
    p.add_argument('--gpu', type=str, default='0')
    p.add_argument('--ckpt', type=str, default='', help='Mac dinh: <cfg.DIFFLIFT.save_dir>/latest_epoch.bin')
    p.add_argument('--set', type=str, default='3dpw')
    p.add_argument('--num_vis', type=int, default=6, help='So mau ve moi loai anh')
    p.add_argument('--pool', type=int, default=2048, help='So mau ngau nhien lay de chon mau ve va ve histogram')
    p.add_argument('--min_missing', type=int, default=0, help='Chi ve mau co it nhat N khop mask=0')
    p.add_argument('--ts', type=str, default='999,799,599,399,199,99,0', help='Cac t cho forward_*.jpg va hist.jpg')
    p.add_argument('--seed', type=int, default=0)
    return p.parse_args()


def npy(x):
    return x.detach().cpu().numpy()


# ============================================================
# Ve
# ============================================================
def draw_skel(ax, j3d, color, lw=1.5, s=8, alpha=1.0):
    """j3d: numpy (17, 3), camera-space (Y huong xuong) -> lat Y de nguoi dung thang."""
    x, y, z = j3d[:, 0], -j3d[:, 1], j3d[:, 2]
    for a, b in H36M_SKELETON:
        ax.plot([x[a], x[b]], [z[a], z[b]], [y[a], y[b]], c=color, lw=lw, alpha=alpha)
    ax.scatter(x, z, y, c=[color], s=s, alpha=alpha)
    ax.set_xlim(-LIM, LIM)
    ax.set_ylim(-LIM, LIM)
    ax.set_zlim(-LIM, LIM)
    ax.set_xticklabels([])
    ax.set_yticklabels([])
    ax.set_zticklabels([])
    ax.set_box_aspect((1, 1, 1))


def draw_2d(ax, cond_i):
    """cond_i: numpy (17, 3) = [x, y, mask] trong [-1, 1]. Khop mask=0 khong ve."""
    xy, m = cond_i[:, :2], cond_i[:, 2] > 0
    for a, b in H36M_SKELETON:
        if m[a] and m[b]:
            ax.plot([xy[a, 0], xy[b, 0]], [xy[a, 1], xy[b, 1]], c='tab:orange', lw=1.5)
    ax.scatter(xy[m, 0], xy[m, 1], c='tab:green', s=12)
    ax.set_xlim(-1, 1)
    ax.set_ylim(1, -1)          # anh: y huong xuong
    ax.set_aspect('equal')
    ax.set_xticks([])
    ax.set_yticks([])
    miss = [JOINT_NAMES[j] for j in range(17) if not m[j]]
    ax.set_title(f'2D vao | mat {len(miss)} khop', fontsize=9)
    if miss:
        ax.set_xlabel(','.join(miss), fontsize=6)


# ============================================================
# Theo doi DDIM tung buoc (sao chep dung logic DiffLifter.forward_inference, chi them ghi lai)
# ============================================================
@torch.no_grad()
def trace_sampling(model, cond, seed):
    dev = cond.device
    gen = torch.Generator(device=dev)
    gen.manual_seed(seed * 100003)            # giong diag_difflifter.sample voi bi = 0
    B, K, s = cond.shape[0], model.num_proposals, model.scale
    T, S, eta = model.num_timesteps, model.sampling_timesteps, model.ddim_sampling_eta
    times = torch.linspace(-1, T - 1, steps=S + 1)
    times = list(reversed(times.int().tolist()))
    pairs = list(zip(times[:-1], times[1:]))

    x = torch.randn(B, K, model.num_joints, 3, device=dev, generator=gen)
    rec = []
    for time, time_next in pairs:
        t_cond = torch.full((B,), time, device=dev, dtype=torch.long)
        x_in = x.clamp(-LIM * s, LIM * s) / s
        pred = model._denoise(cond, x_in, t_cond)
        pred_s = (pred * s).clamp(-LIM * s, LIM * s)
        rec.append({'t': time, 'x_in': x_in, 'pred': pred_s / s})
        if time_next < 0:
            x = pred_s
            break
        alpha = model.alphas_cumprod[time]
        alpha_next = model.alphas_cumprod[time_next]
        sigma = eta * ((1 - alpha / alpha_next) * (1 - alpha_next) / (1 - alpha)).sqrt()
        c = (1 - alpha_next - sigma ** 2).sqrt()
        pred_noise = (x - pred_s * alpha.sqrt()) / (1 - alpha + 1e-8).sqrt()
        noise = torch.randn(x.shape, device=dev, dtype=x.dtype, generator=gen)
        x = pred_s * alpha_next.sqrt() + c * pred_noise + sigma * noise
    return rec, x / s


# ============================================================
# Anh 1: forward (GT + nhieu o cac t) va pred_x0 cua mang
# ============================================================
@torch.no_grad()
def fig_forward(model, cond_i, gt_i, noise_i, ts, sig, path, title):
    s = model.scale
    dev = cond_i.device
    ncol = 1 + len(ts)
    fig = plt.figure(figsize=(2.6 * ncol, 5.8))
    draw_2d(fig.add_subplot(2, ncol, 1), npy(cond_i))
    ax = fig.add_subplot(2, ncol, ncol + 1, projection='3d')
    draw_skel(ax, npy(gt_i), 'tab:green')
    ax.set_title('GT 3D', fontsize=9)

    x0 = (gt_i * s).clamp(-LIM * s, LIM * s)[None]
    for k, t in enumerate(ts):
        tt = torch.full((1,), t, device=dev, dtype=torch.long)
        x_in = model.q_sample(x0, tt, noise_i[None]).clamp(-LIM * s, LIM * s) / s
        pred = model._denoise(cond_i[None], x_in, tt)[0]
        a = model.alphas_cumprod[t].item()
        snr = (a / max(1 - a, 1e-8)) ** 0.5 * sig * s
        err = torch.norm(pred - gt_i, dim=-1).mean().item() * 1000
        ax1 = fig.add_subplot(2, ncol, 2 + k, projection='3d')
        draw_skel(ax1, npy(gt_i), 'lightgray', lw=1.0, s=3)
        draw_skel(ax1, npy(x_in[0]), 'tab:blue')
        ax1.set_title(f't={t}  SNR={snr:.2f}\nx_t mang thay', fontsize=8)
        ax2 = fig.add_subplot(2, ncol, ncol + 2 + k, projection='3d')
        draw_skel(ax2, npy(gt_i), 'lightgray', lw=1.0, s=3)
        draw_skel(ax2, npy(pred), 'tab:red')
        ax2.set_title(f'pred_x0 | {err:.0f} mm', fontsize=8)
    fig.suptitle(title, fontsize=10)
    plt.tight_layout()
    plt.savefig(path, dpi=85, bbox_inches='tight')
    plt.close(fig)


# ============================================================
# Anh 2: DDIM tung buoc
# ============================================================
def fig_trace(rec, final, gt_i, i, path, title):
    S = len(rec)
    K = final.shape[1]
    ncol = S + 1
    colors = plt.cm.tab10(np.arange(K) % 10)
    gt_np = npy(gt_i)
    fig = plt.figure(figsize=(2.8 * ncol, 5.8))
    for c, r in enumerate(rec):
        ax1 = fig.add_subplot(2, ncol, c + 1, projection='3d')
        draw_skel(ax1, gt_np, 'lightgray', lw=1.0, s=3)
        draw_skel(ax1, npy(r['x_in'][i, 0]), 'tab:blue')
        ax1.set_title(f'buoc {c + 1}  t={r["t"]}\nx vao mang (gia thuyet 0)', fontsize=8)
        ax2 = fig.add_subplot(2, ncol, ncol + c + 1, projection='3d')
        draw_skel(ax2, gt_np, 'lightgray', lw=1.0, s=3)
        for k in range(K):
            draw_skel(ax2, npy(r['pred'][i, k]), colors[k], lw=1.0, s=3, alpha=0.85)
        ax2.set_title(f'pred_x0 cua {K} gia thuyet', fontsize=8)
    ax1 = fig.add_subplot(2, ncol, ncol, projection='3d')
    draw_skel(ax1, gt_np, 'tab:green')
    draw_skel(ax1, npy(final[i].mean(0)), 'tab:red')
    ax1.set_title('cuoi: GT (xanh la) + mean (do)', fontsize=8)
    ax2 = fig.add_subplot(2, ncol, 2 * ncol, projection='3d')
    draw_skel(ax2, gt_np, 'lightgray', lw=1.0, s=3)
    for k in range(K):
        draw_skel(ax2, npy(final[i, k]), colors[k], lw=1.0, s=3, alpha=0.85)
    ax2.set_title(f'cuoi: {K} gia thuyet', fontsize=8)
    fig.suptitle(title, fontsize=10)
    plt.tight_layout()
    plt.savefig(path, dpi=85, bbox_inches='tight')
    plt.close(fig)


# ============================================================
# Anh 3: histogram tin hieu / nhieu / x_t
# ============================================================
@torch.no_grad()
def fig_hist(model, gt_pool, ts, path, seed):
    s = model.scale
    dev = gt_pool.device
    g = torch.Generator(device=dev)
    g.manual_seed(seed)
    x0 = (gt_pool[:, 1:] * s).clamp(-LIM * s, LIM * s)          # bo root (luon = 0)
    eps = torch.randn(x0.shape, device=dev, generator=g)
    fig, axes = plt.subplots(1, len(ts), figsize=(3.3 * len(ts), 3.2), squeeze=False)
    print(f'  {"t":>5} | {"alpha_bar":>9} | {"std tin hieu":>12} | {"std nhieu":>9} | {"ti le":>6}   (khong gian scaled)')
    for ax, t in zip(axes[0], ts):
        a = model.alphas_cumprod[t].item()
        sig_part = (a ** 0.5) * x0
        noi_part = ((1 - a) ** 0.5) * eps
        xt = sig_part + noi_part
        for v, lab, col in [(sig_part, 'tin hieu', 'tab:green'), (noi_part, 'nhieu', 'tab:red'),
                            (xt, 'x_t', 'tab:blue')]:
            ax.hist(npy(v).reshape(-1), bins=80, range=(-3, 3), density=True, histtype='step', label=lab, color=col)
        for edge in (-LIM * s, LIM * s):
            ax.axvline(edge, color='k', ls='--', lw=0.6)
        ax.set_title(f't={t}  a={a:.3f}', fontsize=9)
        ax.set_yticks([])
        ratio = sig_part.std().item() / max(noi_part.std().item(), 1e-8)
        print(f'  {t:>5} | {a:>9.4f} | {sig_part.std().item():>12.3f} | {noi_part.std().item():>9.3f} | {ratio:>6.2f}')
    axes[0][0].legend(fontsize=7)
    fig.suptitle(f'Phan bo (scale={s}); net dut = muc clamp +-{LIM * s:.1f}', fontsize=10)
    plt.tight_layout()
    plt.savefig(path, dpi=85, bbox_inches='tight')
    plt.close(fig)


# ============================================================
# Main
# ============================================================
def main():
    args = parse_args()
    if args.gpu:
        os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
    update_config(args.cfg)
    cfg.DATASET.lift_only = True

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    ckpt_path = args.ckpt or os.path.join(cfg.DIFFLIFT.save_dir, 'latest_epoch.bin')
    ckpt = torch.load(ckpt_path, map_location='cpu')
    model = build_model(ckpt, device)
    ts = [int(v) for v in args.ts.split(',') if v.strip() != '']
    for t in ts:
        assert 0 <= t < model.num_timesteps, f't={t} ngoai [0, {model.num_timesteps - 1}]'
    print(f'ckpt={ckpt_path} | epoch={ckpt.get("epoch", "?")} | scale={model.scale} | '
          f'S={model.sampling_timesteps} K={model.num_proposals} eta={model.ddim_sampling_eta}')

    # ---- Lay pool mau ngau nhien ----
    ds = get_test_dataset(args.set, None)
    reg = torch.as_tensor(ds.dataset.h36m_joint_regressor, dtype=torch.float32, device=device)
    rng = np.random.RandomState(args.seed)
    pool_idx = rng.choice(len(ds), size=min(len(ds), args.pool), replace=False)
    loader = DataLoader(Subset(ds, pool_idx.tolist()), batch_size=256, shuffle=False,
                        num_workers=min(4, cfg.DATASET.workers))
    conds, gts = [], []
    for inp, tgt, meta in loader:
        conds.append(make_cond(inp, device))
        gts.append(get_gt(tgt, reg, device))
    cond_pool = torch.cat(conds)
    gt_pool = torch.cat(gts)

    nmiss = (cond_pool[..., 2] == 0).sum(1)
    cand = torch.nonzero(nmiss >= args.min_missing).flatten().tolist()
    if len(cand) < args.num_vis:
        print(f'CANH BAO: chi co {len(cand)} mau co >= {args.min_missing} khop mat trong pool {len(cond_pool)}; '
              f'lay ngau nhien thay the')
        cand = list(range(len(cond_pool)))
    sel = cand[:args.num_vis]
    cond_v, gt_v = cond_pool[sel], gt_pool[sel]
    n = cond_v.shape[0]

    out_dir = os.path.join(cfg.vis_dir, 'difflifter')
    os.makedirs(out_dir, exist_ok=True)
    sig = gt_pool[:, 1:].reshape(-1).std().item()
    print(f'pool={len(cond_pool)} mau cua {args.set} | std gop GT 3D (bo root) = {sig:.3f} m | ve {n} mau -> {out_dir}')

    # ---- Anh 3: histogram ----
    print('\n[hist] so sanh tin hieu va nhieu:')
    fig_hist(model, gt_pool, ts, os.path.join(out_dir, 'hist.jpg'), args.seed)

    # ---- Anh 1: forward ----
    g = torch.Generator(device=device)
    g.manual_seed(args.seed)
    noise_v = torch.randn(n, 17, 3, device=device, generator=g)    # CUNG noise cho moi t -> thay qua trinh lam hong dan
    # ---- Anh 2: trace DDIM ----
    rec, final = trace_sampling(model, cond_v, args.seed)

    # Kiem tra ban sao logic khop voi model.forward (cung seed)
    with torch.no_grad():
        g2 = torch.Generator(device=device)
        g2.manual_seed(args.seed * 100003)
        _, _, ref = model(cond_v, generator=g2)
    ref = ref - ref[:, :, 0:1]
    fin = final - final[:, :, 0:1]
    print(f'\n[kiem tra] trace_sampling vs model.forward (cung seed): max |sai khac| = '
          f'{(ref - fin).abs().max().item():.2e} (ky vong ~0)')

    mean = fin.mean(1)
    err = torch.norm(mean - gt_v, dim=-1).mean(-1) * 1000
    mink = torch.norm(fin - gt_v.unsqueeze(1), dim=-1).mean(-1).min(1).values * 1000
    spr = (fin.std(1).norm(-1)[:, 1:].mean(1) * 1000) if fin.shape[1] > 1 else torch.zeros(n, device=device)

    for i in range(n):
        miss = int((cond_v[i, :, 2] == 0).sum().item())
        title = (f'{args.set} | mau {i} | mat {miss} khop | MPJPE(mean)={err[i]:.0f} mm | min-K={mink[i]:.0f} mm '
                 f'| spread={spr[i]:.0f} mm')
        fig_forward(model, cond_v[i], gt_v[i], noise_v[i], ts, sig,
                    os.path.join(out_dir, f'forward_{i:02d}.jpg'), title)
        fig_trace(rec, final, gt_v[i], i, os.path.join(out_dir, f'trace_{i:02d}.jpg'), title)
        print(f'  mau {i}: mat {miss} khop | MPJPE {err[i]:.0f} mm | min-K {mink[i]:.0f} mm | spread {spr[i]:.0f} mm')

    print(f'\nXong. Anh o: {out_dir}')


if __name__ == '__main__':
    main()
