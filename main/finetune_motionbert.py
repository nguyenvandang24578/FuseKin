import os, sys
sys.path.append('./lib')
sys.path.append('./')

os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"  # cho Apple Accelerate/vecLib
os.environ["NUMEXPR_NUM_THREADS"] = "1"
import argparse
import random
import time

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm

# Import qua 'core.*' / 'utils.*' (KHONG qua 'lib.core.*'): dataset cung import 'core.config',
# neu import 'lib.core.config' thi Python tao 2 module config khac nhau -> yml khong toi duoc dataset.
from core.config import update_config, cfg
from utils.jotr_dataset import get_train_dataset, get_test_dataset
from utils.h36m_adapter import drop_joints_2d, h36m_joint_indices
from data_final.dataset import MultipleDatasets
from models.DSTformer import DSTformer
from MotionBERT.lib.model.loss import loss_velocity, \
    loss_limb_var, loss_limb_gt, loss_angle, loss_angle_velocity


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cfg", type=str, default="config/finetune_motionbert_pw3d.yaml", help="Path to the unified config file (DATASET + MODEL + MOTIONBERT).")
    parser.add_argument('--pretrained', default='', type=str, help='pretrained checkpoint path, ghi de cfg.MOTIONBERT.pretrained (e.g. MotionBERT/checkpoint/pretrain/MB_release.bin)')
    parser.add_argument('--gpu', type=str, default='0,1', help='assign multi-gpus by comma concat, e.g. "0,1" or "0,1,2,3"')
    parser.add_argument('--seed', type=int, default=123)
    parser.add_argument('--eval_only', action='store_true',
                        help='Khong train, chi danh gia checkpoint tren val_set + test_list cua yml')
    parser.add_argument('--ckpt', type=str, default='',
                        help='Checkpoint de danh gia khi --eval_only. Mac dinh: <save_dir>/best_epoch.bin')
    parser.add_argument('--num_vis', type=int, default=10,
                        help='So mau ve hinh cho MOI tap test (0 = tat). Luu o <cfg.vis_dir>/<ten tap>/')
    opts = parser.parse_args()
    return opts


# ============================================================
# Loss co mask (khi moi khop hop le thi bang dung loss_mpjpe / n_mpjpe cua MotionBERT)
# ============================================================
def masked_mpjpe(pred, gt, valid):
    """pred, gt: (B, F, J, 3); valid: (B, F, J, 1). Trung binh khoang cach tren cac khop hop le."""
    err = torch.norm(pred - gt, dim=-1)          # (B, F, J)
    w = valid[..., 0]
    return (err * w).sum() / w.sum().clamp(min=1.0)


def masked_n_mpjpe(pred, gt, valid):
    """MPJPE sau khi can scale toi uu cho moi frame (giong n_mpjpe), chi tinh tren khop hop le."""
    norm_pred = torch.sum(valid * pred ** 2, dim=(2, 3), keepdim=True)
    norm_tgt = torch.sum(valid * gt * pred, dim=(2, 3), keepdim=True)
    scale = norm_tgt / norm_pred.clamp(min=1e-8)
    return masked_mpjpe(scale * pred, gt, valid)


def batch_pa_errors(pred, gt):
    """Procrustes theo batch. pred, gt: (N, J, 3) tensor. Tra ve sai so PA tung mau (N,)."""
    mu_p = pred.mean(1, keepdim=True)
    mu_g = gt.mean(1, keepdim=True)
    p = pred - mu_p
    g = gt - mu_g
    var_p = (p ** 2).sum(dim=(1, 2)).clamp(min=1e-8)
    K = p.transpose(1, 2) @ g                      # (N, 3, 3)
    U, S, Vh = torch.linalg.svd(K)
    V = Vh.transpose(1, 2)
    d = torch.sign(torch.det(V @ U.transpose(1, 2)))
    D = torch.diag_embed(torch.stack([torch.ones_like(d), torch.ones_like(d), d], dim=-1))
    R = V @ D @ U.transpose(1, 2)
    scale = (S * torch.diagonal(D, dim1=1, dim2=2)).sum(-1) / var_p
    aligned = scale[:, None, None] * (p @ R.transpose(1, 2)) + mu_g
    return torch.norm(aligned - gt, dim=-1).mean(-1)


# ============================================================
# Chuan bi dau vao MotionBERT (PHAI giong ARTS.lift_2d_to_3d)
# ============================================================
def make_mb_input(xy, conf, rootrel, drop_idx):
    """xy: (B, 17, 2), conf: (B, 17, 1) -> (B, 1, 17, 3).
    Thu tu giong ARTS.lift_2d_to_3d: tru root (neu bat) -> xoa khop co dinh."""
    if rootrel:
        xy = xy - xy[:, 0:1, :]
    xy, conf = drop_joints_2d(xy, conf, drop_idx)
    return torch.cat([xy, conf], dim=-1).unsqueeze(1)


def normalize_hm_xy(xy):
    """Toa do trong khong gian output_hm_shape -> [-1, 1] (giong Human36M17Dataset)."""
    xy = xy.clone()
    xy[..., 0] = xy[..., 0] / cfg.output_hm_shape[2] * 2 - 1
    xy[..., 1] = xy[..., 1] / cfg.output_hm_shape[1] * 2 - 1
    return xy


def get_train_2d(inputs_b, targets_b, meta_b, mb_cfg, device):
    """Lay 2D dau vao luc train: 2D nhieu cua dataset, co the thay bang GT 2D sach (gt2d_prob)
    va xoa ngau nhien khop (joint_drop_prob). Tra ve xy (B,17,2), conf (B,17,1)."""
    xy = inputs_b['joints'][..., :2].to(device).float()
    conf = inputs_b['joints_mask'].to(device).float()

    gt2d_prob = mb_cfg.gt2d_prob
    if gt2d_prob > 0:
        gt_xy = normalize_hm_xy(targets_b['orig_joint_img'][..., :2].to(device).float())
        gt_mask = (meta_b['orig_joint_valid'] * meta_b['orig_joint_trunc']).to(device).float()
        gt_xy = gt_xy * (gt_mask > 0)
        use_gt = (torch.rand(xy.shape[0], 1, 1, device=device) < gt2d_prob)
        xy = torch.where(use_gt, gt_xy, xy)
        conf = torch.where(use_gt, gt_mask, conf)

    drop_prob = mb_cfg.joint_drop_prob
    if drop_prob > 0:
        drop = (torch.rand_like(conf) < drop_prob) & (conf > 0)
        drop[:, 0] = False   # khong xoa Pelvis (goc toa do)
        keep = (~drop).float()
        xy = xy * keep
        conf = conf * keep
    return xy, conf


def train_epoch(mb_cfg, model, train_loader, optimizer, device, rootrel, drop_idx):
    model.train()
    losses = {
        '3d_pos': 0.0, '3d_scale': 0.0, '3d_velocity': 0.0,
        'lv': 0.0, 'lg': 0.0, 'angle': 0.0, 'angle_velocity': 0.0, 'total': 0.0
    }
    n_batches = 0
    iters = mb_cfg.iters_per_epoch
    total = min(len(train_loader), iters) if iters > 0 else len(train_loader)

    for i, (inputs_b, targets_b, meta_b) in tqdm(enumerate(train_loader), total=total):
        if iters > 0 and i >= iters:
            break
        xy, conf = get_train_2d(inputs_b, targets_b, meta_b, mb_cfg, device)
        target_3d = targets_b['orig_joint_cam'].to(device).float()          # (B, 17, 3) met
        valid_3d = meta_b['orig_joint_valid'].to(device).float() if 'orig_joint_valid' in meta_b \
            else torch.ones_like(target_3d[..., :1])

        # Static single frame (F=1), khop voi ARTS.lift_2d_to_3d luc dung that
        with torch.no_grad():
            mb_input = make_mb_input(xy, conf, rootrel, drop_idx)            # (B, 1, 17, 3)
            target_3d_seq = target_3d.unsqueeze(1)
            target_3d_seq = target_3d_seq - target_3d_seq[:, :, 0:1, :]      # root-relative, met
            valid_seq = valid_3d.unsqueeze(1)                                # (B, 1, 17, 1)

        predicted_3d = model(mb_input)  # (B, 1, 17, 3)

        if i == 0:
            print(f"\n--- Batch {i} ---")
            print(f"target_3d_seq range: min={target_3d_seq.min().item():.2f}, max={target_3d_seq.max().item():.2f}")
            print(f"predicted_3d range : min={predicted_3d.min().item():.2f}, max={predicted_3d.max().item():.2f}")
            print(f"mb_input range     : min={mb_input.min().item():.2f}, max={mb_input.max().item():.2f}")
            print(f"ti le khop hop le dau vao: {mb_input[..., 2].mean().item():.3f} | ti le khop 3D hop le: {valid_seq.mean().item():.3f}")
            print("--------------------\n")

        optimizer.zero_grad()

        # Loss theo MotionBERT train.py; pos/scale co mask theo khop 3D hop le
        loss_3d_pos   = masked_mpjpe(predicted_3d, target_3d_seq, valid_seq)
        loss_3d_scale = masked_n_mpjpe(predicted_3d, target_3d_seq, valid_seq)
        loss_3d_vel   = loss_velocity(predicted_3d, target_3d_seq)   # = 0 khi F=1
        loss_lv       = loss_limb_var(predicted_3d)
        loss_lg       = loss_limb_gt(predicted_3d, target_3d_seq)
        loss_a        = loss_angle(predicted_3d, target_3d_seq)
        loss_av       = loss_angle_velocity(predicted_3d, target_3d_seq)

        loss_total = loss_3d_pos + \
                     mb_cfg.lambda_scale       * loss_3d_scale + \
                     mb_cfg.lambda_3d_velocity * loss_3d_vel + \
                     mb_cfg.lambda_lv          * loss_lv + \
                     mb_cfg.lambda_lg          * loss_lg + \
                     mb_cfg.lambda_a           * loss_a  + \
                     mb_cfg.lambda_av          * loss_av

        loss_total.backward()
        optimizer.step()

        losses['3d_pos']          += loss_3d_pos.item()
        losses['3d_scale']        += loss_3d_scale.item()
        losses['3d_velocity']     += loss_3d_vel.item()
        losses['lv']              += loss_lv.item()
        losses['lg']              += loss_lg.item()
        losses['angle']           += loss_a.item()
        losses['angle_velocity']  += loss_av.item()
        losses['total']           += loss_total.item()
        n_batches += 1

    for k in losses:
        losses[k] /= max(n_batches, 1)
    return losses


def evaluate(model, loader, h36m_regressor, device, rootrel, drop_idx, desc):
    """Lift tren tap val/test. GT 3D:
      - 3DPW: H36M regressor tren smpl_mesh_cam;
      - Human36M: targets['orig_joint_cam'] (mocap, met).
    Root-relative. Tra ve MPJPE va PA-MPJPE (mm), trung binh theo MAU (khong theo batch)."""
    model.eval()
    reg = torch.as_tensor(h36m_regressor, dtype=torch.float32, device=device)  # (17, 6890)
    err_sum, pa_sum, n = 0.0, 0.0, 0
    with torch.no_grad():
        for inputs_b, targets_b, meta_b in tqdm(loader, desc=desc):
            xy = inputs_b['joints'][..., :2].to(device).float()
            conf = inputs_b['joints_mask'].to(device).float()
            if 'orig_joint_cam' in targets_b:
                gt = targets_b['orig_joint_cam'].to(device).float()             # (B, 17, 3)
            else:
                mesh_gt = targets_b['smpl_mesh_cam'].to(device).float()        # (B, 6890, 3)
                gt = torch.matmul(reg[None], mesh_gt)                           # (B, 17, 3)
            gt = gt - gt[:, 0:1, :]

            pred = model(make_mb_input(xy, conf, rootrel, drop_idx))[:, 0]     # (B, 17, 3)
            pred = pred - pred[:, 0:1, :]

            err_sum += torch.norm(pred - gt, dim=-1).mean(-1).sum().item()
            pa_sum += batch_pa_errors(pred, gt).sum().item()
            n += pred.shape[0]
    return {'mpjpe': err_sum / max(n, 1) * 1000, 'pa_mpjpe': pa_sum / max(n, 1) * 1000, 'n': n}


def make_eval_loader(name, mb_cfg):
    dataset = get_test_dataset(name, None)
    loader = DataLoader(dataset=dataset, batch_size=mb_cfg.batch_size, shuffle=False,
                        num_workers=cfg.DATASET.workers, pin_memory=True)
    return dataset, loader


def main():
    opts = parse_args()

    if opts.gpu:
        os.environ['CUDA_VISIBLE_DEVICES'] = str(opts.gpu)
        print(f"Work on GPU(s): {opts.gpu}")

    update_config(opts.cfg)
    mb_cfg = cfg.MOTIONBERT

    random.seed(opts.seed)
    np.random.seed(opts.seed)
    torch.manual_seed(opts.seed)

    num_gpus = torch.cuda.device_count()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device} (Total GPUs active: {num_gpus})")

    rootrel = cfg.MODEL.motionbert_2d_rootrel
    drop_names = list(cfg.MODEL.motionbert_drop_joints)
    drop_idx = h36m_joint_indices(drop_names)
    pretrained = opts.pretrained or mb_cfg.pretrained
    save_dir = mb_cfg.save_dir
    os.makedirs(save_dir, exist_ok=True)
    log_path = os.path.join(save_dir, 'log.txt')

    def log(msg):
        print(msg)
        with open(log_path, 'a') as f:
            f.write(msg + '\n')

    log(f"===== Finetune MotionBERT | cfg={opts.cfg} | {time.strftime('%Y-%m-%d %H:%M:%S')} =====")
    log(f"train_list={list(cfg.DATASET.train_list)} | val_set={mb_cfg.val_set} | test_list={list(cfg.DATASET.test_list)}")
    log(f"lift_only={cfg.DATASET.lift_only} | rootrel_2d={rootrel} | drop_joints={drop_names} | "
        f"gt2d_prob={mb_cfg.gt2d_prob} | joint_drop_prob={mb_cfg.joint_drop_prob}")
    log(f"pretrained={pretrained or '(ngau nhien)'} | batch={mb_cfg.batch_size} | iters/epoch={mb_cfg.iters_per_epoch} | "
        f"epochs={mb_cfg.epochs} | lr={mb_cfg.learning_rate} | lr_decay={mb_cfg.lr_decay}")

    # ---------------- Chi danh gia (khong train) ----------------
    if opts.eval_only:
        ckpt_path = opts.ckpt or os.path.join(save_dir, 'best_epoch.bin')
        model = build_model(mb_cfg)
        ckpt = torch.load(ckpt_path, map_location=lambda storage, loc: storage)
        model.load_state_dict(ckpt['model_pos'], strict=True)
        log(f"===== EVAL ONLY: {ckpt_path} (epoch {ckpt.get('epoch', '?')}) =====")
        run_tests(model, [mb_cfg.val_set] + list(cfg.DATASET.test_list), mb_cfg, device, rootrel, drop_idx, log,
                  num_vis=opts.num_vis)
        return

    # ---------------- Data ----------------
    train_names = list(cfg.DATASET.train_list)
    train_sets = [get_train_dataset(name, opts) for name in train_names]
    for name, ds in zip(train_names, train_sets):
        log(f"# train {name}: {len(ds)}")
    # Tron nhieu dataset: balance_datasets=True -> lay mau ngang nhau (make_same_len);
    # False -> noi thang, moi epoch duyet moi mau dung 1 lan
    train_dataset = train_sets[0] if len(train_sets) == 1 else \
        MultipleDatasets(train_sets, make_same_len=mb_cfg.balance_datasets)
    log(f"# train tong: {len(train_dataset)} mau | balance_datasets={mb_cfg.balance_datasets}")
    workers = cfg.DATASET.workers
    train_loader = DataLoader(
        dataset=train_dataset,
        batch_size=mb_cfg.batch_size,
        shuffle=True,
        num_workers=workers,
        pin_memory=True,
        persistent_workers=workers > 0,
    )

    val_dataset, val_loader = make_eval_loader(mb_cfg.val_set, mb_cfg)
    log(f"# val {mb_cfg.val_set}: {len(val_dataset)}")
    h36m_regressor = val_dataset.dataset.h36m_joint_regressor

    # ---------------- Model ----------------
    model = build_model(mb_cfg)

    if pretrained:
        print(f"Loading pretrained weights from {pretrained}...")
        checkpoint = torch.load(pretrained, map_location=lambda storage, loc: storage)
        model.load_state_dict(checkpoint['model_pos'], strict=True)
        print("Pretrained weights loaded SUCCESSFULLY!")
    else:
        print("WARNING: pretrained is empty! Model is initializing with random weights!")

    lr = mb_cfg.learning_rate
    optimizer = optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=lr,
        weight_decay=mb_cfg.weight_decay
    )
    lr_decay = mb_cfg.lr_decay

    # Thong tin di kem checkpoint de biet ckpt nay can cau hinh nao khi dung trong Teacher/Student
    ckpt_meta = {
        'train_list': train_names,
        'val_set': mb_cfg.val_set,
        'motionbert_2d_rootrel': rootrel,
        'motionbert_drop_joints': drop_names,
        'gt2d_prob': mb_cfg.gt2d_prob,
        'joint_drop_prob': mb_cfg.joint_drop_prob,
        'pretrained': pretrained,
    }

    def save(path, epoch, min_loss):
        torch.save({
            'epoch': epoch,
            'lr': lr,
            'optimizer': optimizer.state_dict(),
            'model_pos': model.state_dict(),
            'min_loss': min_loss,
            'fusekin_meta': ckpt_meta,
        }, path)

    # ---------------- Train ----------------
    print(f"Starting finetuning for {mb_cfg.epochs} epochs...")
    min_loss = float('inf')
    best_path = os.path.join(save_dir, 'best_epoch.bin')

    for epoch in range(mb_cfg.epochs):
        start_time = time.time()

        losses = train_epoch(mb_cfg, model, train_loader, optimizer, device, rootrel, drop_idx)
        val = evaluate(model, val_loader, h36m_regressor, device, rootrel, drop_idx, desc=f'Val {mb_cfg.val_set}')

        elapsed = (time.time() - start_time) / 60
        log(f"[Epoch {epoch+1}/{mb_cfg.epochs}] Time: {elapsed:.2f}m | LR: {lr:.6f} | "
            f"Train MPJPE: {losses['3d_pos']*1000:.2f} mm (total {losses['total']:.5f}) | "
            f"Val({mb_cfg.val_set}) MPJPE: {val['mpjpe']:.2f} mm, PA: {val['pa_mpjpe']:.2f} mm")

        lr *= lr_decay
        for param_group in optimizer.param_groups:
            param_group['lr'] *= lr_decay

        save(os.path.join(save_dir, 'latest_epoch.bin'), epoch + 1, min_loss)
        if (epoch + 1) % mb_cfg.checkpoint_frequency == 0:
            save(os.path.join(save_dir, f'epoch_{epoch+1}.bin'), epoch + 1, min_loss)

        # val_loss tinh bang met nhu ban cu (min_loss trong checkpoint giu don vi cu)
        val_loss_m = val['mpjpe'] / 1000
        if val_loss_m < min_loss:
            min_loss = val_loss_m
            save(best_path, epoch + 1, min_loss)
            log(f"--> Saved new best model (Val MPJPE={val['mpjpe']:.2f} mm) to {best_path}")

    # ---------------- Test 1 lan voi checkpoint cuoi (MOTIONBERT.final_ckpt: 'best' | 'latest') ----------------
    final_kind = mb_cfg.get('final_ckpt', 'best')
    if final_kind not in ('best', 'latest'):
        raise ValueError(f"MOTIONBERT.final_ckpt khong hop le: {final_kind}")
    final_path = best_path if final_kind == 'best' else os.path.join(save_dir, 'latest_epoch.bin')
    if os.path.exists(final_path):
        best = torch.load(final_path, map_location=lambda storage, loc: storage)
        model.load_state_dict(best['model_pos'], strict=True)
        log(f"===== Test voi {final_kind} checkpoint (epoch {best['epoch']}): {final_path} =====")
        run_tests(model, list(cfg.DATASET.test_list), mb_cfg, device, rootrel, drop_idx, log,
                  num_vis=opts.num_vis)

    print("Finetuning completed.")


def build_model(mb_cfg):
    print("Initializing MotionBERT (DSTformer)...")
    model = DSTformer(
        dim_in=3,
        dim_out=3,
        dim_feat=mb_cfg.dim_feat,
        dim_rep=mb_cfg.dim_rep,
        depth=mb_cfg.depth,
        num_heads=mb_cfg.num_heads,
        mlp_ratio=mb_cfg.mlp_ratio,
        norm_layer=nn.LayerNorm,
        maxlen=mb_cfg.maxlen,
        num_joints=mb_cfg.num_joints,
        att_fuse=mb_cfg.att_fuse
    )
    if torch.cuda.is_available():
        model = nn.DataParallel(model)
        model = model.cuda()
        print(f"Model wrapped in DataParallel on {torch.cuda.device_count()} GPU(s)")
    return model


def run_tests(model, names, mb_cfg, device, rootrel, drop_idx, log, num_vis=0):
    """Danh gia lan luot tung tap (+ ve num_vis mau moi tap).
    Mot tap loi (vd thieu file annotation) thi ghi loi va chay tiep tap sau."""
    for name in names:
        try:
            test_dataset, test_loader = make_eval_loader(name, mb_cfg)
            res = evaluate(model, test_loader, test_dataset.dataset.h36m_joint_regressor, device,
                           rootrel, drop_idx, desc=f'Test {name}')
            log(f"[Test {name}] MPJPE: {res['mpjpe']:.2f} mm | PA-MPJPE: {res['pa_mpjpe']:.2f} mm | n={res['n']}")
        except Exception as e:
            log(f"[Test {name}] LOI: {type(e).__name__}: {e}")
            continue
        if num_vis > 0:
            try:
                out_dir = os.path.join(cfg.vis_dir, name)
                visualize_samples(model, test_dataset, name, num_vis, out_dir, device, rootrel, drop_idx)
                log(f"[Vis {name}] {num_vis} anh -> {out_dir}")
            except Exception as e:
                log(f"[Vis {name}] LOI: {type(e).__name__}: {e}")


# ============================================================
# Ve hinh (gop tu main/debug_motionbert.py, dung DUNG dau vao nhu luc train/eval:
# make_mb_input -> F=1, rootrel theo cfg, xoa khop theo cfg.MODEL.motionbert_drop_joints)
# ============================================================
H36M_SKELETON = [
    (0, 1), (1, 2), (2, 3),           # chan phai
    (0, 4), (4, 5), (5, 6),           # chan trai
    (0, 7), (7, 8), (8, 9), (9, 10),  # cot song, dau
    (8, 14), (14, 15), (15, 16),      # tay phai
    (8, 11), (11, 12), (12, 13),      # tay trai
]


def _draw_2d(img_bgr, joints_px, used):
    """Chi ve khop thuc su dua vao MotionBERT (used[j] = 1). Khop bi xoa co toa do 0 nen khong ve."""
    import cv2
    for p1, p2 in H36M_SKELETON:
        if used[p1] and used[p2]:
            cv2.line(img_bgr, tuple(int(v) for v in joints_px[p1]), tuple(int(v) for v in joints_px[p2]), (255, 255, 0), 2)
    for j, (x, y) in enumerate(joints_px):
        if used[j]:
            cv2.circle(img_bgr, (int(x), int(y)), 4, (0, 255, 0), -1)
            cv2.putText(img_bgr, str(j), (int(x) + 5, int(y) + 5), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 1)
    return img_bgr


def _draw_3d(ax, joint_3d, color_joint, color_bone, title=None, set_limits=True):
    x, y, z = joint_3d[:, 0], -joint_3d[:, 1], joint_3d[:, 2]   # lat Y (anh Y huong xuong -> 3D Y huong len)
    ax.scatter(x, z, y, c=color_joint, marker='o', s=20)
    for p1, p2 in H36M_SKELETON:
        ax.plot([x[p1], x[p2]], [z[p1], z[p2]], [y[p1], y[p2]], c=color_bone)
    if set_limits:
        r = 1.0   # met, du cho nguoi dung (root-relative)
        ax.set_xlim(-r, r); ax.set_ylim(-r, r); ax.set_zlim(-r, r)
        ax.set_xlabel('X'); ax.set_ylabel('Z (sau)'); ax.set_zlabel('Y')
    if title:
        ax.set_title(title, fontsize=10)


def visualize_samples(model, dataset, name, num, out_dir, device, rootrel, drop_idx):
    """Ve `num` mau trai deu trong tap, moi mau 1 anh 4 o:
    [Anh crop + 2D dau vao] | [GT 3D] | [Pred 3D] | [Chong GT (xanh) + Pred (do)]."""
    import cv2
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    os.makedirs(out_dir, exist_ok=True)
    model.eval()
    reg = dataset.dataset.h36m_joint_regressor
    W, H = cfg.input_img_shape[1], cfg.input_img_shape[0]
    dataset.keep_vis_info = True
    try:
        idxs = np.linspace(0, len(dataset) - 1, num).astype(int)
        for k, idx in enumerate(idxs):
            inputs, targets, meta = dataset[int(idx)]
            xy = torch.as_tensor(np.asarray(inputs['joints'])[:, :2], dtype=torch.float32)[None].to(device)
            conf = torch.as_tensor(np.asarray(inputs['joints_mask']), dtype=torch.float32).reshape(1, -1, 1).to(device)
            mb_input = make_mb_input(xy, conf, rootrel, drop_idx)
            with torch.no_grad():
                pred = model(mb_input)[0, 0].cpu().numpy()
            used = (mb_input[0, 0, :, 2].cpu().numpy() > 0).astype(int)   # khop thuc su dua vao model

            if 'orig_joint_cam' in targets:
                gt = np.asarray(targets['orig_joint_cam'], dtype=np.float32)
            else:
                gt = np.dot(reg, np.asarray(targets['smpl_mesh_cam'], dtype=np.float32))
            # regressor la float64 -> ep ca 2 ve float32 (torch matmul khong tron kieu duoc)
            gt = (gt - gt[0:1]).astype(np.float32)
            pred = (pred - pred[0:1]).astype(np.float32)
            err = np.linalg.norm(pred - gt, axis=-1)
            pa = batch_pa_errors(torch.from_numpy(pred)[None], torch.from_numpy(gt)[None])[0].item()

            # Anh crop (giong luc train) + 2D dau vao. Khong co duong dan anh -> nen den.
            img_bgr = np.zeros((H, W, 3), dtype=np.uint8)
            img_path = meta.get('img_path', None)
            if img_path is not None and os.path.isfile(img_path):
                raw = cv2.imread(img_path)
                if raw is not None:
                    img_bgr = cv2.warpAffine(raw, np.asarray(meta['img2bb_trans'], dtype=np.float32), (W, H),
                                             flags=cv2.INTER_LINEAR)
            joints_px = (np.asarray(inputs['joints'])[:, :2] + 1) / 2.0 * np.array([W, H])
            img_bgr = _draw_2d(img_bgr.copy(), joints_px, used)

            fig = plt.figure(figsize=(24, 6))
            ax1 = fig.add_subplot(141)
            ax1.imshow(cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)); ax1.axis('off')
            ax1.set_title(f'Input 2D ({used.sum()}/17 khop dua vao model; khop bi xoa khong ve)', fontsize=10)
            _draw_3d(fig.add_subplot(142, projection='3d'), gt, 'green', 'blue', 'GT 3D (root-rel)')
            _draw_3d(fig.add_subplot(143, projection='3d'), pred, 'red', 'orange', 'Pred 3D (root-rel)')
            ax4 = fig.add_subplot(144, projection='3d')
            _draw_3d(ax4, gt, 'green', 'blue')
            _draw_3d(ax4, pred, 'red', 'orange', 'Chong: GT xanh / Pred cam', set_limits=False)
            worst = int(err.argmax())
            plt.suptitle(f'{name} | idx {int(idx)} | MPJPE {err.mean()*1000:.1f} mm | PA {pa*1000:.1f} mm | '
                         f'khop sai nhat: {worst} ({err[worst]*1000:.0f} mm)', fontsize=12, y=1.02)
            plt.tight_layout()
            plt.savefig(os.path.join(out_dir, f'{name}_{k:02d}_idx{int(idx)}.jpg'), dpi=100, bbox_inches='tight')
            plt.close(fig)
    finally:
        dataset.keep_vis_info = False


if __name__ == '__main__':
    main()
