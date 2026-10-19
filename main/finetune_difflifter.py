"""
main/finetune_difflifter.py
Train DiffLifter (single-frame diffusion 2D->3D lifter) doc lap.

Theo khung LiftTrainer cua base.py + finetune_motionbert.py:
  - Dataset: jotr_dataset (Human36M17Dataset), giong finetune_motionbert
  - Config: cfg.DIFFLIFT (them vao config.py)
  - Model: DiffLifter tu lib/models/diff_lifter.py
  - Loss: MPJPE truc tiep tren pred_x0
  - Checkpoint: luu/nap giong finetune_motionbert.py

Chay:
  python main/finetune_difflifter.py --cfg config/finetune_difflifter.yaml --gpu 0
"""

import os
import sys
sys.path.append('./lib')
sys.path.append('./')

os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"

import argparse
import copy
import random
import time

import numpy as np
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm

from core.config import update_config, cfg
from utils.jotr_dataset import get_train_dataset, get_test_dataset
from data_final.dataset import MultipleDatasets
from models.diff_lifter import DiffLifter


# ============================================================
# Args
# ============================================================

def parse_args():
    parser = argparse.ArgumentParser(description='Train DiffLifter')
    parser.add_argument('--cfg', type=str, default='config/finetune_difflifter.yaml')
    parser.add_argument('--gpu', type=str, default='0')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--eval_only', action='store_true',
                        help='Khong train, chi eval checkpoint')
    parser.add_argument('--ckpt', type=str, default='',
                        help='Checkpoint de eval khi --eval_only')
    return parser.parse_args()


# ============================================================
# Loss: MPJPE (giong base.py CoordLoss nhung don gian hon)
# ============================================================

def mpjpe(pred, gt):
    """pred, gt: (B, 17, 3). Tra ve scalar (met)."""
    return torch.norm(pred - gt, dim=-1).mean()


def mpjpe_masked(pred, gt, valid):
    """valid: (B, 17, 1) binary mask."""
    err = torch.norm(pred - gt, dim=-1)   # (B, 17)
    w = valid[..., 0]
    return (err * w).sum() / w.sum().clamp(min=1.0)


def pa_mpjpe_batch(pred, gt):
    """Procrustes-aligned MPJPE theo batch. pred, gt: (N, 17, 3) tensor.
    Tra ve sai so PA TUNG MAU (N,) don vi met (giong batch_pa_errors cua finetune_motionbert)."""
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


# ============================================================
# Data helpers -- copy va mo rong tu finetune_motionbert.py
# ============================================================

# Chi xa (2 khop) + khop goc + chi doi xung, thu tu H36M-17
OCC_GROUPS = [([2, 3], 1, [5, 6]),        # R_Knee, R_Ankle  | goc R_Hip      | doi xung L
              ([5, 6], 4, [2, 3]),        # L_Knee, L_Ankle  | goc L_Hip
              ([12, 13], 11, [15, 16]),   # L_Elbow, L_Wrist | goc L_Shoulder
              ([15, 16], 14, [12, 13])]   # R_Elbow, R_Wrist | goc R_Shoulder


def occ_corrupt(xy, conf, dl_cfg):
    """
    Che khuat GIONG THAT cho detector 2D: khop bi che thuong VAN CO toa do (sai cho) + score hoi thap,
    khong phai bien mat. Moi chi (xac suat occ_aug_prob, chi ap dung khi ca chi dang nhin ro) chon 1 trong 4:
      0 dich ca doan chi (detector bam vao vat khac)
      1 lay vi tri chi doi xung trai/phai (nham trai-phai) -- neu chi doi xung khong nhin ro thi doi sang 0
      2 dinh vao khop goc (vai/hong) -- detector "bo cuoc"
      3 xoa (xy = 0, conf = 0) nhu cu
    Score (neu lift_soft_conf): ha ve gan nguong: thr + (conf - thr) * U(0, 0.5). Mask nhi phan: giu nguyen.
    xy, conf: (B, 17, 2), (B, 17, 1) -> sua tai cho va tra ve.
    """
    p = getattr(dl_cfg, 'occ_aug_prob', 0.0)
    if p <= 0:
        return xy, conf
    std = getattr(dl_cfg, 'occ_shift_std', 0.15)
    soft = cfg.DATASET.lift_soft_conf
    thr = cfg.DATASET.halpe26_conf_thr
    B, dev = xy.shape[0], xy.device
    xy0, conf0 = xy.clone(), conf.clone()
    ar = torch.arange(B, device=dev)
    for g, parent, mirror in OCC_GROUPS:
        sel = (torch.rand(B, device=dev) < p) & (conf0[:, g, 0].min(1).values > 0)
        if not sel.any():
            continue
        mode = torch.randint(0, 4, (B,), device=dev)
        mirror_ok = conf0[:, mirror, 0].min(1).values > 0
        mode = torch.where((mode == 1) & ~mirror_ok, torch.zeros_like(mode), mode)
        cur = xy0[:, g]                                                        # (B, 2, 2)
        small = lambda: torch.randn_like(cur) * std * 0.25
        cand = torch.stack([
            cur + torch.randn(B, 1, 2, device=dev) * std + small(),            # 0 dich
            xy0[:, mirror] + small(),                                          # 1 doi xung
            xy0[:, parent:parent + 1].expand_as(cur) + small(),                # 2 dinh khop goc
            torch.zeros_like(cur),                                             # 3 xoa
        ], dim=1)                                                              # (B, 4, 2, 2)
        new_xy = cand[ar, mode].clamp(-1, 1)
        if soft:
            new_c = thr + (conf0[:, g] - thr).clamp(min=0) * torch.rand(B, 1, 1, device=dev) * 0.5
        else:
            new_c = conf0[:, g].clone()
        new_c = torch.where((mode == 3)[:, None, None], torch.zeros_like(new_c), new_c)
        m = sel[:, None, None]
        xy[:, g] = torch.where(m, new_xy, xy[:, g])
        conf[:, g] = torch.where(m, new_c, conf[:, g])
    return xy, conf


def make_xy_conf(inputs_b, device, dl_cfg):
    """
    Lay xy_conf tu batch:
      xy_conf: (B, 17, 3) = [x_norm, y_norm, conf]
    Va ap dung occlusion augmentation khi train.
    """
    joints = inputs_b['joints'].to(device).float()     # (B, 17, 3): [x, y, mask]
    xy   = joints[..., :2]                             # (B, 17, 2)
    conf = inputs_b['joints_mask'].to(device).float()  # (B, 17, 1)

    # 1. Drop khop ngau nhien (gia lap khop bi an)
    drop_prob = getattr(dl_cfg, 'joint_drop_prob', 0.0)
    if drop_prob > 0:
        drop = (torch.rand_like(conf) < drop_prob) & (conf > 0)
        drop[:, 0] = False    # giu Pelvis (root)
        keep = (~drop).float()
        xy   = xy   * keep
        conf = conf * keep

    # 2. Drop ca chi (tay, chan) -- gia lap che khuat theo nhom khop
    limb_prob = getattr(dl_cfg, 'limb_drop_prob', 0.0)
    if limb_prob > 0:
        # H36M-17 indices: R_leg=[1,2,3], L_leg=[4,5,6], L_arm=[11,12,13], R_arm=[14,15,16]
        limb_groups = [[1, 2, 3], [4, 5, 6], [11, 12, 13], [14, 15, 16]]
        for group in limb_groups:
            mask_drop = (torch.rand(xy.shape[0], 1, device=device) < limb_prob)  # (B,1)
            mask_drop = mask_drop.expand(-1, len(group))   # (B, len_group)
            xy[:, group, :]   = xy[:, group, :]   * (~mask_drop).unsqueeze(-1).float()
            conf[:, group, :] = conf[:, group, :] * (~mask_drop).unsqueeze(-1).float()

    # 2b. Che khuat giong that (2D sai cho + score thap), mac dinh TAT (occ_aug_prob = 0)
    xy, conf = occ_corrupt(xy, conf, dl_cfg)

    # 3. Jitter nho tren toa do 2D
    jitter_std = getattr(dl_cfg, 'jitter_std', 0.0)
    if jitter_std > 0:
        noise = torch.randn_like(xy) * jitter_std
        xy = xy + noise * (conf > 0)

    return torch.cat([xy, conf], dim=-1)   # (B, 17, 3)


def make_gt3d(targets_b, device):
    """Lay GT 3D root-relative, met. Tra ve (B, 17, 3)."""
    gt = targets_b['orig_joint_cam'].to(device).float()   # (B, 17, 3) met
    gt = gt - gt[:, 0:1, :]                               # root-relative
    return gt


# ============================================================
# Train epoch -- theo pattern LiftTrainer.train() + finetune_motionbert.train_epoch()
# ============================================================

@torch.no_grad()
def ema_update(ema, model, decay, step):
    """EMA trong so: ema = d * ema + (1 - d) * model. Warmup d nho luc dau de EMA bat kip."""
    d = min(decay, (1 + step) / (10 + step))
    for e, m in zip(ema.state_dict().values(), model.state_dict().values()):
        if e.dtype.is_floating_point:
            e.mul_(d).add_(m.detach(), alpha=1 - d)
        else:
            e.copy_(m)


def train_epoch(model, loader, optimizer, device, dl_cfg, epoch, ema=None, step0=0):
    model.train()
    total_loss = 0.0
    n = 0
    iters = getattr(dl_cfg, 'iters_per_epoch', 0)
    total = min(len(loader), iters) if iters > 0 else len(loader)

    pbar = tqdm(enumerate(loader), total=total,
                desc=f'Epoch {epoch} [train]')
    for i, (inputs_b, targets_b, meta_b) in pbar:
        if iters > 0 and i >= iters:
            break

        xy_conf = make_xy_conf(inputs_b, device, dl_cfg)  # (B,17,3)
        gt_3d   = make_gt3d(targets_b, device)             # (B,17,3)

        # --- Forward + loss ---
        pred_x0 = model(xy_conf, gt_3d)    # (B,17,3) -- training forward

        # valid mask (dung neu co)
        if 'orig_joint_valid' in meta_b:
            valid = meta_b['orig_joint_valid'].to(device).float()  # (B,17,1)
            loss = mpjpe_masked(pred_x0, gt_3d, valid)
        else:
            loss = mpjpe(pred_x0, gt_3d)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if ema is not None:
            ema_update(ema, model, dl_cfg.ema_decay, step0 + n)

        total_loss += loss.item()
        n += 1

        if i % getattr(dl_cfg, 'print_freq', 50) == 0:
            pbar.set_postfix({'loss_mm': f'{loss.item()*1000:.2f}'})

        if i == 0 and epoch <= 2:
            print(f'\n[Epoch {epoch} batch 0] xy_conf range: '
                  f'{xy_conf.min().item():.3f}..{xy_conf.max().item():.3f} | '
                  f'gt_3d range: {gt_3d.min().item():.3f}..{gt_3d.max().item():.3f} | '
                  f'pred_x0 range: {pred_x0.min().item():.3f}..{pred_x0.max().item():.3f}')

    return total_loss / max(n, 1), n


# ============================================================
# Evaluate -- theo pattern LiftTester.test()
# ============================================================

@torch.no_grad()
def evaluate(model, loader, h36m_regressor, device, desc='Val', seed=0):
    """
    GT 3D (giong finetune_motionbert.evaluate):
      - 3DPW: H36M regressor nhan smpl_mesh_cam
      - Human36M: targets['orig_joint_cam'] (mocap, met)
    Root-relative. Tra ve (mm, trung binh theo MAU):
      mpjpe    -- trung binh K hypotheses (chi so chinh, so duoc voi MotionBERT/PMCE)
      pa_mpjpe
      min_k    -- hypothesis tot nhat (can tren, KHONG dung de so sanh voi lifter khac)
    seed: nhieu DDIM co dinh moi lan eval -> so sanh giua cac epoch khong bi nhieu ngau nhien.
    """
    model.eval()
    reg = torch.as_tensor(h36m_regressor, dtype=torch.float32, device=device)  # (17, 6890)
    gen = torch.Generator(device=device)
    gen.manual_seed(seed)

    err_sum, pa_sum, mink_sum, n = 0.0, 0.0, 0.0, 0

    pbar = tqdm(loader, desc=desc)
    for inputs_b, targets_b, meta_b in pbar:
        # 2D condition: toa do crop da chuan hoa [-1,1] (Human36M17Dataset), KHONG tru root
        xy_conf = torch.cat([
            inputs_b['joints'][..., :2].float(),
            inputs_b['joints_mask'].float(),
        ], dim=-1).to(device)   # (B, 17, 3)

        # GT
        if 'orig_joint_cam' in targets_b:
            gt = targets_b['orig_joint_cam'].to(device).float()
        else:
            mesh_gt = targets_b['smpl_mesh_cam'].to(device).float()   # (B, 6890, 3)
            gt = torch.matmul(reg[None], mesh_gt)                       # (B, 17, 3)
        gt = gt - gt[:, 0:1, :]    # root-relative

        # Inference: K hypotheses (model da tru root ben trong)
        mean_pose, spread, all_preds = model(xy_conf, generator=gen)
        # mean_pose: (B,17,3), all_preds: (B,K,17,3)
        mean_pose = mean_pose - mean_pose[:, 0:1, :]
        all_preds = all_preds - all_preds[:, :, 0:1, :]

        # MPJPE (mean)
        err_sum += torch.norm(mean_pose - gt, dim=-1).mean(-1).sum().item()

        # PA-MPJPE (theo batch)
        pa_sum += pa_mpjpe_batch(mean_pose, gt).sum().item()

        # min-of-K: (B,K)
        errs_k = torch.norm(all_preds - gt.unsqueeze(1), dim=-1).mean(-1)
        mink_sum += errs_k.min(dim=1).values.sum().item()

        n += mean_pose.shape[0]

    return {
        'mpjpe':    err_sum / max(n, 1) * 1000,
        'pa_mpjpe': pa_sum  / max(n, 1) * 1000,
        'min_k':    mink_sum / max(n, 1) * 1000,
        'n': n,
    }


# ============================================================
# Checkpoint helpers
# ============================================================

def save_ckpt(path, model, optimizer, epoch, best_val, dl_cfg, ema=None, step=0):
    # Co EMA: 'model_state' = trong so EMA (diag / FuseKin nap thang), 'raw_model_state' = trong so goc (de resume)
    extra = {'raw_model_state': model.state_dict(), 'step': step} if ema is not None else {}
    torch.save({
        'epoch': epoch,
        'model_state': (ema if ema is not None else model).state_dict(),
        **extra,
        'optimizer_state': optimizer.state_dict(),
        'best_val_mpjpe': best_val,
        'difflift_cfg': {
            'embed_dim': dl_cfg.embed_dim,
            'depth':     dl_cfg.depth,
            'num_heads': dl_cfg.num_heads,
            'timesteps': dl_cfg.timesteps,
            'sampling_timesteps': dl_cfg.sampling_timesteps,
            'num_proposals': dl_cfg.num_proposals,
            'scale':     dl_cfg.scale,
            'cond_mode': getattr(dl_cfg, 'cond_mode', 'input'),
            'aggregate': getattr(dl_cfg, 'aggregate', 'mean'),
            'soft_conf': bool(cfg.DATASET.lift_soft_conf),
            'conf_attn': bool(getattr(dl_cfg, 'conf_attn', False)),
            'skel_attn': bool(getattr(dl_cfg, 'skel_attn', False)),
            'mask_token': bool(getattr(dl_cfg, 'mask_token', False)),
            'self_cond': bool(getattr(dl_cfg, 'self_cond', False)),
            'ema_decay': float(getattr(dl_cfg, 'ema_decay', 0.0)),
        },
    }, path)


def load_ckpt(path, model, optimizer=None, device='cpu', ema=None):
    ckpt = torch.load(path, map_location=device)
    if ema is not None and 'raw_model_state' in ckpt:
        # resume co EMA: model = trong so goc, ema = trong so EMA
        model.load_state_dict(ckpt['raw_model_state'])
        ema.load_state_dict(ckpt['model_state'])
    else:
        model.load_state_dict(ckpt['model_state'])
        if ema is not None:
            ema.load_state_dict(ckpt['model_state'])
    if optimizer is not None and 'optimizer_state' in ckpt:
        optimizer.load_state_dict(ckpt['optimizer_state'])
    return ckpt.get('epoch', 0), ckpt.get('best_val_mpjpe', float('inf'))


# ============================================================
# Main
# ============================================================

def main():
    opts = parse_args()

    if opts.gpu:
        os.environ['CUDA_VISIBLE_DEVICES'] = opts.gpu

    update_config(opts.cfg)
    dl_cfg = cfg.DIFFLIFT
    # Phai dat TRUOC khi tao bat ky dataset nao (ca nhanh eval_only)
    cfg.DATASET.lift_only = True   # khong doc anh, tra thang H36M-17

    random.seed(opts.seed)
    np.random.seed(opts.seed)
    torch.manual_seed(opts.seed)
    torch.cuda.manual_seed_all(opts.seed)
    torch.backends.cudnn.deterministic = True

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Device: {device} | cfg: {opts.cfg}')

    save_dir = dl_cfg.save_dir
    os.makedirs(save_dir, exist_ok=True)
    log_path = os.path.join(save_dir, 'log.txt')

    def log(msg):
        print(msg)
        with open(log_path, 'a') as f:
            f.write(msg + '\n')

    log(f'===== DiffLifter Train | {time.strftime("%Y-%m-%d %H:%M:%S")} =====')
    log(f'train_list={list(cfg.DATASET.train_list)} | val_set={dl_cfg.val_set}')
    log(f'embed_dim={dl_cfg.embed_dim} | depth={dl_cfg.depth} | '
        f'T={dl_cfg.timesteps} | S={dl_cfg.sampling_timesteps} | K={dl_cfg.num_proposals}')
    log(f'lr={dl_cfg.lr} | batch={dl_cfg.batch_size} | epochs={dl_cfg.epochs}')
    log(f'scale={dl_cfg.scale} | cond_mode={getattr(dl_cfg, "cond_mode", "input")} | '
        f'aggregate={getattr(dl_cfg, "aggregate", "mean")}')

    # --- Model ---
    model = DiffLifter(
        num_joints=17,
        embed_dim=dl_cfg.embed_dim,
        depth=dl_cfg.depth,
        num_heads=dl_cfg.num_heads,
        timesteps=dl_cfg.timesteps,
        sampling_timesteps=dl_cfg.sampling_timesteps,
        num_proposals=dl_cfg.num_proposals,
        scale=dl_cfg.scale,
        drop_path_rate=0.1,
        cond_mode=getattr(dl_cfg, 'cond_mode', 'input'),
        aggregate=getattr(dl_cfg, 'aggregate', 'mean'),
        conf_attn=getattr(dl_cfg, 'conf_attn', False),
        skel_attn=getattr(dl_cfg, 'skel_attn', False),
        mask_token=getattr(dl_cfg, 'mask_token', False),
        self_cond=getattr(dl_cfg, 'self_cond', False),
    ).to(device)

    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    log(f'Model params: {n_params/1e6:.2f} M')

    optimizer = optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=dl_cfg.lr,
        weight_decay=getattr(dl_cfg, 'weight_decay', 0.01),
    )

    # EMA (tat khi ema_decay = 0)
    ema = None
    if getattr(dl_cfg, 'ema_decay', 0.0) > 0:
        ema = copy.deepcopy(model).eval()
        for q in ema.parameters():
            q.requires_grad_(False)
    log(f'ema_decay={getattr(dl_cfg, "ema_decay", 0.0)} | occ_aug_prob={getattr(dl_cfg, "occ_aug_prob", 0.0)} | '
        f'soft_conf={cfg.DATASET.lift_soft_conf} | conf_attn={getattr(dl_cfg, "conf_attn", False)} | '
        f'skel_attn={getattr(dl_cfg, "skel_attn", False)} | mask_token={getattr(dl_cfg, "mask_token", False)} | '
        f'self_cond={getattr(dl_cfg, "self_cond", False)} | balance_datasets={getattr(dl_cfg, "balance_datasets", True)}')

    start_epoch = 0
    step = 0
    best_val = float('inf')
    best_path  = os.path.join(save_dir, 'best_epoch.bin')
    latest_path = os.path.join(save_dir, 'latest_epoch.bin')

    # Resume
    if getattr(dl_cfg, 'resume', '') and os.path.exists(dl_cfg.resume):
        start_epoch, best_val = load_ckpt(dl_cfg.resume, model, optimizer, device, ema=ema)
        step = 10 ** 6   # resume: bo warmup EMA
        log(f'Resumed from {dl_cfg.resume} (epoch {start_epoch})')

    # --- Eval only ---
    if opts.eval_only:
        ckpt_path = opts.ckpt or best_path
        load_ckpt(ckpt_path, model, device=device)
        log(f'EVAL ONLY: {ckpt_path}')
        for name in [dl_cfg.val_set] + list(cfg.DATASET.test_list):
            try:
                ds = get_test_dataset(name, None)
                ld = DataLoader(ds, batch_size=dl_cfg.batch_size,
                                shuffle=False, num_workers=cfg.DATASET.workers)
                res = evaluate(model, ld, ds.dataset.h36m_joint_regressor, device,
                               desc=f'Eval {name}')
                log(f"[Eval {name}] MPJPE: {res['mpjpe']:.2f} mm | PA: {res['pa_mpjpe']:.2f} mm | "
                    f"min-K: {res['min_k']:.2f} mm (n={res['n']})")
            except Exception as e:
                log(f'[Eval {name}] LOI: {type(e).__name__}: {e}')
        return

    # --- Datasets ---
    train_names = list(cfg.DATASET.train_list)
    train_sets = [get_train_dataset(name, opts) for name in train_names]
    for name, ds in zip(train_names, train_sets):
        log(f'# train {name}: {len(ds)}')

    balance = getattr(dl_cfg, 'balance_datasets', True)
    train_dataset = train_sets[0] if len(train_sets) == 1 else \
        MultipleDatasets(train_sets, make_same_len=balance)
    log(f'# train total: {len(train_dataset)}')

    train_loader = DataLoader(
        dataset=train_dataset,
        batch_size=dl_cfg.batch_size,
        shuffle=True,
        num_workers=cfg.DATASET.workers,
        pin_memory=True,
        persistent_workers=cfg.DATASET.workers > 0,
    )

    val_ds = get_test_dataset(dl_cfg.val_set, None)
    val_loader = DataLoader(val_ds, batch_size=dl_cfg.batch_size,
                            shuffle=False, num_workers=cfg.DATASET.workers,
                            pin_memory=True)
    log(f'# val {dl_cfg.val_set}: {len(val_ds)}')
    val_reg = val_ds.dataset.h36m_joint_regressor

    # Tap danh gia them: moi epoch (epoch_eval_sets) va moi full_eval_freq epoch (full_eval_sets)
    extra_loaders = {}
    for name in list(getattr(dl_cfg, 'epoch_eval_sets', [])) + list(getattr(dl_cfg, 'full_eval_sets', [])):
        if name in extra_loaders or name == dl_cfg.val_set:
            continue
        try:
            ds_e = get_test_dataset(name, None)
            extra_loaders[name] = (DataLoader(ds_e, batch_size=dl_cfg.batch_size, shuffle=False,
                                              num_workers=cfg.DATASET.workers, pin_memory=True),
                                   ds_e.dataset.h36m_joint_regressor)
            log(f'# eval them {name}: {len(ds_e)}')
        except Exception as e:
            log(f'# eval them {name}: LOI {type(e).__name__}: {e}')

    def eval_extra(names, eval_model, epoch_no):
        for name in names:
            if name not in extra_loaders:
                continue
            ld, reg = extra_loaders[name]
            r = evaluate(eval_model, ld, reg, device, desc=f'Eval {name}')
            log(f'    [Ep {epoch_no} | {name}] MPJPE: {r["mpjpe"]:.2f} mm | PA: {r["pa_mpjpe"]:.2f} mm | '
                f'min-K: {r["min_k"]:.2f} mm (n={r["n"]})')

    # --- Train loop (theo pattern LiftTrainer.train + LiftTester.test) ---
    lr_decay = dl_cfg.lr_decay
    # Resume: tiep tuc lich LR tu epoch da dung (khong quay ve lr goc)
    lr = dl_cfg.lr * (lr_decay ** start_epoch)
    for pg in optimizer.param_groups:
        pg['lr'] = lr

    for epoch in range(start_epoch, dl_cfg.epochs):
        t0 = time.time()

        train_loss, n_it = train_epoch(model, train_loader, optimizer, device, dl_cfg, epoch + 1,
                                       ema=ema, step0=step)
        step += n_it
        eval_model = ema if ema is not None else model
        val_res    = evaluate(eval_model, val_loader, val_reg, device, desc=f'Val {dl_cfg.val_set}')

        elapsed = (time.time() - t0) / 60
        log(f'[Epoch {epoch+1}/{dl_cfg.epochs}] {elapsed:.1f}m | '
            f'Train loss: {train_loss*1000:.2f} mm | '
            f'Val MPJPE: {val_res["mpjpe"]:.2f} mm | '
            f'PA: {val_res["pa_mpjpe"]:.2f} mm | '
            f'min-K: {val_res["min_k"]:.2f} mm | '
            f'LR: {lr:.6f}')

        # Danh gia them (chi log, KHONG dung de chon best)
        eval_extra(list(getattr(dl_cfg, 'epoch_eval_sets', [])), eval_model, epoch + 1)
        if (epoch + 1) % getattr(dl_cfg, 'full_eval_freq', 5) == 0:
            eval_extra(list(getattr(dl_cfg, 'full_eval_sets', [])), eval_model, epoch + 1)

        # LR decay
        lr *= lr_decay
        for pg in optimizer.param_groups:
            pg['lr'] = lr

        # Checkpoint: cap nhat best TRUOC de latest luu dung best_val (resume khong bi lech)
        if val_res['mpjpe'] < best_val:
            best_val = val_res['mpjpe']
            save_ckpt(best_path, model, optimizer, epoch + 1, best_val, dl_cfg, ema, step)
            log(f'--> Best model saved (Val MPJPE={best_val:.2f} mm)')

        save_ckpt(latest_path, model, optimizer, epoch + 1, best_val, dl_cfg, ema, step)

        if (epoch + 1) % getattr(dl_cfg, 'save_freq', 10) == 0:
            ep_path = os.path.join(save_dir, f'epoch_{epoch+1}.bin')
            save_ckpt(ep_path, model, optimizer, epoch + 1, best_val, dl_cfg, ema, step)

    # --- Final test ---
    log('===== Final test with best checkpoint =====')
    if os.path.exists(best_path):
        load_ckpt(best_path, model, device=device)
    for test_name in list(cfg.DATASET.test_list):
        try:
            test_ds = get_test_dataset(test_name, None)
            test_loader = DataLoader(test_ds, batch_size=dl_cfg.batch_size,
                                     shuffle=False, num_workers=cfg.DATASET.workers)
            res = evaluate(model, test_loader, test_ds.dataset.h36m_joint_regressor, device,
                           desc=f'Test {test_name}')
            log(f'[Test {test_name}] MPJPE: {res["mpjpe"]:.2f} mm | '
                f'PA: {res["pa_mpjpe"]:.2f} mm | min-K: {res["min_k"]:.2f} mm')
        except Exception as e:
            log(f'[Test {test_name}] LOI: {type(e).__name__}: {e}')

    print('Done.')


if __name__ == '__main__':
    main()
