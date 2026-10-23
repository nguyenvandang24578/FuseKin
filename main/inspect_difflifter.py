"""
main/inspect_difflifter.py
Kiem tra DiffLifter tren 1 tap test (mac dinh 3dpw): SO LIEU + HINH, KHONG train.

SO LIEU (in ra + ghi <out_dir>/inspect_log.txt):
  1. MPJPE / N-MPJPE (chi chinh kich thuoc) / R-MPJPE (chi xoay) / PA-MPJPE (xoay + kich thuoc)
     -> MPJPE - N-MPJPE lon: loi chu yeu do KICH THUOC nguoi (gia thuyet hoc thuoc chieu cao dien vien)
     -> MPJPE - R-MPJPE lon: loi chu yeu do HUONG XOAY tong the (do nghieng camera)
  2. Ti le kich thuoc du doan / GT (tong do dai xuong), toan tap + THEO TUNG SEQUENCE
     -> sequence nao lech deu (vd 0.92 hoac 1.08) = mo hinh doan sai chieu cao nguoi do
  3. Loi theo truc x / y / z (z = do sau) va spread theo truc
  4. Spread co y nghia khong: Spearman(spread, loi) tung khop, min-K vs mean-K
  5. Gia thuyet rieng le co hop ly khong: do lech do dai xuong GIUA K gia thuyet (mm)
  6. Tu kiem tra: quy dao DDIM chep lai trong file nay == model.forward_inference (sai khac ~0)

HINH (<out_dir>/vis/*.png), moi mau 1 hang 4 o:
  [anh + 2D, vong tron = spread] [nhin truoc: K gia thuyet + GT] [nhin ngang (do sau)] [nhin ngang SAU KHI che tay phai]
  + hinh quy dao DDIM (x0 du doan qua tung buoc) cho vai mau.
  Chon mau: spread cao nhat, score 2D thap nhat, ngau nhien.

Chay:
  python main/inspect_difflifter.py --cfg config/finetune_difflifter_v3_mix.yaml --gpu 0 \
      --ckpt experiment/finetune_difflifter_v3_mix/best_epoch.bin
"""
import os
import sys
sys.path.append('./lib')
sys.path.append('./')

os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"

import argparse
from collections import defaultdict

import cv2
import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from core.config import update_config, cfg
from utils.jotr_dataset import get_test_dataset
from models.diff_lifter import H36M_PARENTS
from diag_difflifter import build_model, get_gt, JOINT_NAMES

BONES = [(j, p) for j, p in enumerate(H36M_PARENTS) if p >= 0]       # 16 xuong
R_ARM = [15, 16]                                                       # R_Elbow, R_Wrist (che nhan tao)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--cfg', required=True)
    p.add_argument('--ckpt', required=True)
    p.add_argument('--gpu', default='0')
    p.add_argument('--set', default='3dpw')
    p.add_argument('--batch_size', type=int, default=512)
    p.add_argument('--seed', type=int, default=0)
    p.add_argument('--num_vis', type=int, default=4, help='so mau moi nhom (spread cao / score thap / ngau nhien)')
    p.add_argument('--K_vis', type=int, default=30, help='so gia thuyet khi VE (nhieu hon de thay phan phoi)')
    p.add_argument('--S', type=int, default=5, help='so buoc DDIM cho so lieu + hinh (checkpoint moi luu S=1 -> '
                                                     'gia thuyet gan trung nhau, khong thay phan phoi)')
    p.add_argument('--K', type=int, default=5, help='so gia thuyet cho so lieu (muc 4, 5 can K > 1)')
    p.add_argument('--out_dir', default='', help='mac dinh <thu muc ckpt>/inspect_<set>')
    return p.parse_args()


LOG = []


def log(s=''):
    print(s)
    LOG.append(s)


# ------------------------------------------------------------------
# Can chinh
# ------------------------------------------------------------------
def align(p, g, use_rot, use_scale):
    """p, g: (N, J, 3) root-relative. Tra ve p da can chinh vao g.
    Chi scale: khong tru tam (N-MPJPE kieu Rhodin). Co xoay: tru tam roi Kabsch (+ scale neu use_scale)."""
    if not use_rot:
        s = (p * g).sum((1, 2)) / (p * p).sum((1, 2)).clamp(min=1e-8)
        return s[:, None, None] * p
    mp, mg = p.mean(1, keepdim=True), g.mean(1, keepdim=True)
    pc, gc = p - mp, g - mg
    H = pc.transpose(1, 2) @ gc
    U, S, Vh = torch.linalg.svd(H)
    V = Vh.transpose(1, 2)
    d = torch.sign(torch.det(V @ U.transpose(1, 2)))
    D = torch.diag_embed(torch.stack([torch.ones_like(d), torch.ones_like(d), d], -1))
    R = V @ D @ U.transpose(1, 2)
    s = (S * torch.diagonal(D, dim1=1, dim2=2)).sum(-1) / (pc ** 2).sum((1, 2)).clamp(min=1e-8) \
        if use_scale else torch.ones_like(d)
    return s[:, None, None] * (pc @ R.transpose(1, 2)) + mg


def bone_len(x):
    """x: (..., 17, 3) -> (..., 16)"""
    return torch.stack([torch.norm(x[..., j, :] - x[..., p, :], dim=-1) for j, p in BONES], -1)


def rank(x):
    return torch.argsort(torch.argsort(x)).float()


def spearman(a, b):
    a, b = rank(a), rank(b)
    a, b = a - a.mean(), b - b.mean()
    return float((a * b).sum() / (a.norm() * b.norm()).clamp(min=1e-8))


# ------------------------------------------------------------------
# Quy dao DDIM: chep DUNG logic DiffLifter.forward_inference, them ghi lai tung buoc
# ------------------------------------------------------------------
@torch.no_grad()
def ddim_traj(model, xy_conf, gen):
    B, K, J = xy_conf.shape[0], model.num_proposals, model.num_joints
    T, S, eta, sc = model.num_timesteps, model.sampling_timesteps, model.ddim_sampling_eta, model.scale
    times = list(reversed(torch.linspace(-1, T - 1, steps=S + 1).int().tolist()))
    x = torch.randn(B, K, J, 3, device=xy_conf.device, generator=gen)
    x0s = []
    for time, time_next in zip(times[:-1], times[1:]):
        t_cond = torch.full((B,), time, device=xy_conf.device, dtype=torch.long)
        x_in = x.clamp(-1.1 * sc, 1.1 * sc) / sc
        pred_x0 = model._denoise(xy_conf, x_in, t_cond)
        x0s.append(pred_x0)
        p_sc = (pred_x0 * sc).clamp(-1.1 * sc, 1.1 * sc)
        if time_next < 0:
            x = p_sc
            break
        a, an = model.alphas_cumprod[time], model.alphas_cumprod[time_next]
        sigma = eta * ((1 - a / an) * (1 - an) / (1 - a)).clamp(min=0).sqrt()
        c = (1 - an - sigma ** 2).clamp(min=0).sqrt()
        pred_noise = (x - p_sc * a.sqrt()) / (1 - a + 1e-8).sqrt()
        noise = torch.randn(x.shape, device=x.device, dtype=x.dtype, generator=gen)
        x = p_sc * an.sqrt() + c * pred_noise + sigma * noise
    return x / sc, x0s


def make_cond(inp, device):
    cam = inp['cam_bbox'].float()[:, None, :].expand(-1, inp['joints'].shape[1], -1)
    return torch.cat([inp['joints'][..., :2].float(), inp['joints_mask'].float(), cam], dim=-1).to(device)


def corrupt_arm(cond):
    """Che tay phai giong that: dich 2D (0.15, -0.10) + score ve nguong + 0.05 (neu soft conf)."""
    c2 = cond.clone()
    c2[:, R_ARM, 0] = (c2[:, R_ARM, 0] + 0.15).clamp(-1, 1)
    c2[:, R_ARM, 1] = (c2[:, R_ARM, 1] - 0.10).clamp(-1, 1)
    if cfg.DATASET.lift_soft_conf:
        c2[:, R_ARM, 2] = torch.clamp(c2[:, R_ARM, 2], max=cfg.DATASET.halpe26_conf_thr + 0.05)
    return c2


# ------------------------------------------------------------------
# Ve
# ------------------------------------------------------------------
def draw_skel(ax, P, ix, iy, color, lw, alpha):
    for j, p in BONES:
        ax.plot([P[j, ix], P[p, ix]], [-P[j, iy], -P[p, iy]], color=color, lw=lw, alpha=alpha)


def panel_3d(ax, hyps, gt, ix, title):
    for h in hyps:
        draw_skel(ax, h * 1000, ix, 1, 'tab:blue', 0.8, 0.25)
    draw_skel(ax, hyps.mean(0) * 1000, ix, 1, 'tab:red', 2.0, 0.9)
    draw_skel(ax, gt * 1000, ix, 1, 'black', 2.0, 0.9)
    ax.set_title(title, fontsize=9)
    ax.set_aspect('equal')
    ax.set_xlabel('x (mm)' if ix == 0 else 'z = do sau (mm)', fontsize=8)
    ax.tick_params(labelsize=7)


def panel_img(ax, img_path, raw2d, spread_mm, hyp_mean):
    img = cv2.imread(img_path)
    if img is None:
        ax.set_title('khong doc duoc anh', fontsize=8)
        ax.axis('off')
        return
    img = img[..., ::-1]
    xy, sc = raw2d[:, :2], raw2d[:, 2]
    # px / mm xap xi: chieu cao 2D (px) / chieu cao 3D du doan theo truc y (mm)
    h2d = np.ptp(xy[:, 1]) + 1e-6
    h3d = np.ptp(hyp_mean[:, 1] * 1000) + 1e-6
    px_mm = h2d / h3d
    x0, x1 = xy[:, 0].min(), xy[:, 0].max()
    y0, y1 = xy[:, 1].min(), xy[:, 1].max()
    m = 0.25 * max(x1 - x0, y1 - y0)
    ax.imshow(img)
    for j, p in BONES:
        ax.plot([xy[j, 0], xy[p, 0]], [xy[j, 1], xy[p, 1]], color='lime', lw=1.2)
    for j in range(1, 17):
        ax.add_patch(plt.Circle(xy[j], spread_mm[j] * px_mm, color='red', fill=False, lw=1.2))
        if sc[j] < 0.6:
            ax.text(xy[j, 0], xy[j, 1], f'{sc[j]:.2f}', color='yellow', fontsize=7)
    ax.set_xlim(x0 - m, x1 + m)
    ax.set_ylim(y1 + m, y0 - m)
    ax.set_title('2D Halpe (vang = score < 0.6), do = spread', fontsize=8)
    ax.axis('off')


# ------------------------------------------------------------------
def main():
    args = parse_args()
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
    update_config(args.cfg)
    cfg.DATASET.lift_only = True
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    out_dir = args.out_dir or os.path.join(os.path.dirname(args.ckpt), f'inspect_{args.set}')
    os.makedirs(os.path.join(out_dir, 'vis'), exist_ok=True)

    ckpt = torch.load(args.ckpt, map_location=device)
    model = build_model(ckpt, device)
    model.sampling_timesteps, model.num_proposals = args.S, args.K   # ghi de sampler luu trong checkpoint
    log(f'ckpt={args.ckpt}')
    log(f'difflift_cfg={ckpt["difflift_cfg"]}')
    log(f'soft_conf (cfg)={cfg.DATASET.lift_soft_conf} | S={model.sampling_timesteps} | K={model.num_proposals}')

    ds = get_test_dataset(args.set, None)
    ds.keep_vis_info = True
    loader = DataLoader(ds, batch_size=args.batch_size, shuffle=False, num_workers=cfg.DATASET.workers)
    reg = torch.as_tensor(ds.dataset.h36m_joint_regressor, dtype=torch.float32, device=device)

    C, P, G, paths, raws = [], [], [], [], []
    max_diff = 0.0
    for bi, (inp, tgt, meta) in enumerate(tqdm(loader, desc=args.set)):
        cond = make_cond(inp, device)
        gt = get_gt(tgt, reg, device)
        g1 = torch.Generator(device=device); g1.manual_seed(args.seed * 100000 + bi)
        preds, _ = ddim_traj(model, cond, g1)
        if bi == 0:
            # Tu kiem tra: quy dao chep lai phai trung forward_inference
            g2 = torch.Generator(device=device); g2.manual_seed(args.seed * 100000 + bi)
            _, _, ref = model(cond, generator=g2)
            max_diff = float((ref - preds).abs().max())
        preds = preds - preds[:, :, :1]
        C.append(cond.cpu()); P.append(preds.cpu()); G.append(gt.cpu())
        paths += list(meta['img_path'])
        raws.append(meta['raw_joint_coord_img'].float())
    C, P, G, R2 = torch.cat(C), torch.cat(P), torch.cat(G), torch.cat(raws)
    mean = P.mean(1)
    N, K = P.shape[:2]
    mm = lambda e: float(e.mean()) * 1000

    log('\n' + '=' * 70)
    log(f'[{args.set}] n={N} | K={K}')
    log(f'  [6] tu kiem tra quy dao DDIM: max|sai khac| = {max_diff:.2e} m (phai ~0)')
    log('=' * 70)
    # 1. Can chinh
    e = lambda p: torch.norm(p - G, dim=-1).mean(1)
    e_raw, e_n = e(mean), e(align(mean, G, False, True))
    e_r, e_pa = e(align(mean, G, True, False)), e(align(mean, G, True, True))
    log('  [1] MPJPE theo kieu can chinh (mm, 17 khop, gop mean-K):')
    log(f'      MPJPE={mm(e_raw):.2f} | N-MPJPE (chi kich thuoc)={mm(e_n):.2f} | '
        f'R-MPJPE (chi xoay)={mm(e_r):.2f} | PA (xoay+kich thuoc)={mm(e_pa):.2f}')
    log(f'      phan do kich thuoc = {mm(e_raw) - mm(e_n):.2f} mm | phan do xoay = {mm(e_raw) - mm(e_r):.2f} mm')
    # 2. Kich thuoc
    ratio = bone_len(mean).sum(-1) / bone_len(G).sum(-1)
    log(f'  [2] ti le kich thuoc du doan/GT: TB={float(ratio.mean()):.3f} | std={float(ratio.std()):.3f} '
        f'| ti le mau lech >5%={float(((ratio - 1).abs() > 0.05).float().mean()) * 100:.1f}%')
    seq = [p.replace('\\', '/').split('/')[-2] for p in paths]
    by = defaultdict(list)
    for i, s in enumerate(seq):
        by[s].append(i)
    log('      theo sequence (ti le TB | MPJPE | N-MPJPE | so mau):')
    for s in sorted(by):
        ix = torch.tensor(by[s])
        log(f'        {s:<28} {float(ratio[ix].mean()):.3f} | {mm(e_raw[ix]):6.1f} | {mm(e_n[ix]):6.1f} | {len(ix)}')
    # 3. Theo truc
    d = (mean - G)[:, 1:].abs().reshape(-1, 3).mean(0) * 1000
    sp_ax = P.std(1)[:, 1:].reshape(-1, 3).mean(0) * 1000
    log(f'  [3] loi TB theo truc (mm): x={d[0]:.1f} y={d[1]:.1f} z(sau)={d[2]:.1f} '
        f'| spread theo truc: x={sp_ax[0]:.1f} y={sp_ax[1]:.1f} z={sp_ax[2]:.1f}')
    # 4. Spread
    spr = P.std(1).norm(dim=-1)                                 # (N, 17)
    err_j = torch.norm(mean - G, dim=-1)
    mink = torch.norm(P - G[:, None], dim=-1).mean(-1).min(1).values
    log(f'  [4] mean-K={mm(e_raw):.2f} | min-K={mm(mink):.2f} | spread TB={mm(spr[:, 1:]):.2f} mm | '
        f'Spearman(spread, loi) tung khop={spearman(spr[:, 1:].reshape(-1), err_j[:, 1:].reshape(-1)):.3f}')
    log('      Spearman theo khop: ' + ', '.join(
        f'{JOINT_NAMES[j]}={spearman(spr[:, j], err_j[:, j]):.2f}' for j in range(1, 17)))
    # 5. Hop ly tung gia thuyet
    bl = bone_len(P)                                            # (N, K, 16)
    log(f'  [5] do lech do dai xuong giua K gia thuyet: TB={float(bl.std(1).mean()) * 1000:.1f} mm '
        f'| max xuong TB={float(bl.std(1).mean(0).max()) * 1000:.1f} mm '
        f'(nho = moi gia thuyet la 1 bo xuong on dinh)')
    log(f'      loi tung gia thuyet TB={mm(torch.norm(P - G[:, None], dim=-1).mean(-1)):.2f} vs gop mean-K={mm(e_raw):.2f}')

    # ---------------- HINH ----------------
    rng = np.random.RandomState(args.seed)
    nv = args.num_vis
    groups = {
        'spread_cao': torch.argsort(spr[:, 1:].mean(1), descending=True)[:nv].tolist(),
        'score_thap': torch.argsort(torch.where(C[:, 1:, 2] > 0, C[:, 1:, 2], torch.ones_like(C[:, 1:, 2])).min(1).values)[:nv].tolist(),
        'ngau_nhien': rng.choice(N, nv, replace=False).tolist(),
    }
    K0 = model.num_proposals
    model.num_proposals = args.K_vis
    for gname, idxs in groups.items():
        cond = C[idxs].to(device)
        gen = torch.Generator(device=device); gen.manual_seed(args.seed)
        hyp, x0s = ddim_traj(model, cond, gen)
        gen = torch.Generator(device=device); gen.manual_seed(args.seed)
        hyp_oc, _ = ddim_traj(model, corrupt_arm(cond), gen)
        hyp, hyp_oc = (hyp - hyp[:, :, :1]).cpu().numpy(), (hyp_oc - hyp_oc[:, :, :1]).cpu().numpy()
        fig, axes = plt.subplots(len(idxs), 4, figsize=(18, 4.5 * len(idxs)))
        axes = np.atleast_2d(axes)
        for r, i in enumerate(idxs):
            gt = G[i].numpy()
            s_mm = hyp[r].std(0)
            s_mm = np.linalg.norm(s_mm, axis=-1) * 1000
            em = np.linalg.norm(hyp[r].mean(0) - gt, axis=-1).mean() * 1000
            panel_img(axes[r, 0], paths[i], R2[i].numpy(), s_mm, hyp[r].mean(0))
            panel_3d(axes[r, 1], hyp[r], gt, 0, f'nhin truoc | xanh={args.K_vis} gia thuyet, do=TB, den=GT | loi {em:.0f} mm')
            panel_3d(axes[r, 2], hyp[r], gt, 2, 'nhin ngang (do sau)')
            s_oc = np.linalg.norm(hyp_oc[r][:, R_ARM].std(0), axis=-1).mean() * 1000
            s_b = np.linalg.norm(hyp[r][:, R_ARM].std(0), axis=-1).mean() * 1000
            panel_3d(axes[r, 3], hyp_oc[r], gt, 2, f'nhin ngang SAU KHI che tay phai | spread tay {s_b:.0f} -> {s_oc:.0f} mm')
            axes[r, 0].set_ylabel(os.path.basename(os.path.dirname(paths[i])), fontsize=8)
        fig.tight_layout()
        fp = os.path.join(out_dir, 'vis', f'{gname}.png')
        fig.savefig(fp, dpi=90)
        plt.close(fig)
        log(f'  [vis] {fp}')
        if gname == 'ngau_nhien':
            # Quy dao DDIM: x0 du doan o tung buoc (nhin ngang), 2 mau dau
            S = len(x0s)
            fig, axes = plt.subplots(2, S, figsize=(3.6 * S, 7), squeeze=False)
            for r in range(min(2, len(idxs))):
                for k in range(S):
                    h = (x0s[k][r] - x0s[k][r][:, :1]).cpu().numpy()
                    panel_3d(axes[r, k], h, G[idxs[r]].numpy(), 2, f'buoc {k + 1}/{S}: x0 du doan')
            fig.tight_layout()
            fp = os.path.join(out_dir, 'vis', 'quy_dao_ddim.png')
            fig.savefig(fp, dpi=90)
            plt.close(fig)
            log(f'  [vis] {fp}')
    model.num_proposals = K0

    with open(os.path.join(out_dir, 'inspect_log.txt'), 'w') as f:
        f.write('\n'.join(LOG) + '\n')
    log(f'\nLog: {os.path.join(out_dir, "inspect_log.txt")}')


if __name__ == '__main__':
    main()
