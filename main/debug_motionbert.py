"""
Debug script: Chay MotionBERT DSTformer tren data thuc te va so sanh ket qua voi GT.
Luu moi sample thanh 1 anh gom 3 cot:
  [Input Image + 2D KP] | [GT 3D Skeleton] | [MotionBERT Predicted 3D]

Chay tu thu muc goc:
    python main/debug_motionbert.py \
        --cfg ./config/train_init_mesh.yaml \
        --gpu 0 \
        --checkpoint /path/to/best_epoch.bin \
        --num_samples 10
"""
import os, sys
sys.path.append('./lib')
sys.path.append('./')

import argparse
import warnings
warnings.filterwarnings("ignore")

parser = argparse.ArgumentParser()
parser.add_argument('--cfg',         type=str, default='./config/train_init_mesh.yaml')
parser.add_argument('--gpu',         type=str, default='0')
parser.add_argument('--checkpoint',  type=str, default='',
                    help='Path to MotionBERT pose3d checkpoint (best_epoch.bin)')
parser.add_argument('--num_samples', type=int, default=10,
                    help='Number of samples to visualize')
parser.add_argument('--out_dir',     type=str, default='debug_motionbert_vis',
                    help='Output directory for images')
args = parser.parse_args()

from core.config import cfg, update_config
update_config(args.cfg)

os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu

import torch
import numpy as np
import cv2
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from functools import partial
import torch.nn as nn
import __init_path

from utils.jotr_dataset import get_train_dataset
from torch.utils.data import DataLoader

# -----------------------------------------------------------------
# MotionBERT DSTformer -- built exactly as in
#   MotionBERT/lib/utils/learning.py :: load_backbone()
# -----------------------------------------------------------------
from models.DSTformer import DSTformer

# Hyperparams matching configs/pretrain/MB_lite.yaml in MotionBERT repo (MotionBERT-Lite)
_MB_DIM_IN     = 3
_MB_DIM_OUT    = 3       # Direct 3D joint output
_MB_DIM_FEAT   = 256     # MB-Lite uses 256, full MotionBERT uses 512
_MB_DIM_REP    = 512
_MB_DEPTH      = 5
_MB_NUM_HEADS  = 8
_MB_MLP_RATIO  = 4       # MB-Lite uses 4, full MotionBERT uses 2
_MB_MAXLEN     = 243
_MB_NUM_JOINTS = 17
_MB_ATT_FUSE   = True


def build_motionbert():
    """Build DSTformer exactly as in MotionBERT/lib/utils/learning.py :: load_backbone()"""
    model = DSTformer(
        dim_in=_MB_DIM_IN,
        dim_out=_MB_DIM_OUT,
        dim_feat=_MB_DIM_FEAT,
        dim_rep=_MB_DIM_REP,
        depth=_MB_DEPTH,
        num_heads=_MB_NUM_HEADS,
        mlp_ratio=_MB_MLP_RATIO,
        norm_layer=partial(nn.LayerNorm, eps=1e-6),
        maxlen=_MB_MAXLEN,
        num_joints=_MB_NUM_JOINTS,
        att_fuse=_MB_ATT_FUSE,
    )
    return model


def load_checkpoint(model, chk_path):
    """Load checkpoint exactly as in MotionBERT/lib/utils/learning.py :: load_pretrained_weights()"""
    checkpoint = torch.load(chk_path, map_location='cpu')
    # MotionBERT saves under key 'model_pos' (ref: MotionBERT/train.py :: save_checkpoint)
    state_dict = checkpoint['model_pos']
    new_state_dict = {}
    for k, v in state_dict.items():
        # Strip DataParallel prefix if present
        if k.startswith('module.'):
            k = k[7:]
        new_state_dict[k] = v
    missing, unexpected = model.load_state_dict(new_state_dict, strict=False)
    print(f'  Checkpoint loaded. Missing keys: {len(missing)}, Unexpected: {len(unexpected)}')
    return model


# -----------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------
# H36M 17-joint skeleton edges (same as debug_dataloader.py)
SKELETON = [
    (0, 1), (1, 2), (2, 3),           # R_Leg
    (0, 4), (4, 5), (5, 6),           # L_Leg
    (0, 7), (7, 8), (8, 9), (9, 10),  # Spine & Head
    (8, 14), (14, 15), (15, 16),      # R_Arm
    (8, 11), (11, 12), (12, 13),      # L_Arm
]


def draw_2d_skeleton(img_bgr, joints_px, joints_mask):
    """Draw 2D skeleton on image in-place. Matches debug_dataloader.py style."""
    for (p1, p2) in SKELETON:
        v1, v2 = int(joints_mask[p1, 0]), int(joints_mask[p2, 0])
        if v1 == 1 and v2 == 1:
            x1, y1 = int(joints_px[p1, 0]), int(joints_px[p1, 1])
            x2, y2 = int(joints_px[p2, 0]), int(joints_px[p2, 1])
            cv2.line(img_bgr, (x1, y1), (x2, y2), (255, 255, 0), thickness=2)
    for i in range(len(joints_px)):
        x, y = int(joints_px[i, 0]), int(joints_px[i, 1])
        valid = int(joints_mask[i, 0])
        color = (0, 255, 0) if valid == 1 else (0, 0, 255)
        cv2.circle(img_bgr, (x, y), radius=4, color=color, thickness=-1)
        cv2.putText(img_bgr, str(i), (x + 5, y + 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)
    return img_bgr


def draw_3d_skeleton(ax, joint_3d, color_joint='r', color_bone='b', title=''):
    """Draw 3D skeleton on a matplotlib 3D axes. Matches debug_dataloader.py style."""
    x =  joint_3d[:, 0]
    y = -joint_3d[:, 1]   # Flip Y (image Y-down -> 3D Y-up)
    z =  joint_3d[:, 2]

    ax.scatter(x, z, y, c=color_joint, marker='o', s=20)
    for (p1, p2) in SKELETON:
        ax.plot([x[p1], x[p2]], [z[p1], z[p2]], [y[p1], y[p2]], c=color_bone)

    # Equal scale
    all_coords = np.stack([x, y, z], axis=0)
    ranges = all_coords.max(axis=1) - all_coords.min(axis=1)
    max_range = ranges.max() / 2.0
    mids = (all_coords.max(axis=1) + all_coords.min(axis=1)) * 0.5
    ax.set_xlim(mids[0] - max_range, mids[0] + max_range)
    ax.set_ylim(mids[2] - max_range, mids[2] + max_range)
    ax.set_zlim(mids[1] - max_range, mids[1] + max_range)
    ax.view_init(elev=15, azim=-90)
    ax.set_xlabel('X')
    ax.set_ylabel('Depth Z')
    ax.set_zlabel('Y')
    ax.set_title(title, fontsize=10)


def run_motionbert(model, joints_2d_batch, device):
    """
    Run MotionBERT inference on a batch.
    joints_2d_batch: (B, J, 2) tensor -- normalised 2D joints from dataloader

    MotionBERT input convention (ref: MotionBERT/train.py :: train_epoch):
      shape = (N, T, J, C)  where C=3 (x, y, confidence)
    Our dataloader gives C=2, so we append a confidence column of 1.0.

    Returns: (B, J, 3) numpy array -- predicted 3D joints
    """
    B, J, C = joints_2d_batch.shape
    # Always take x,y (first 2 channels) then append confidence=1
    # MotionBERT always expects exactly 3 channels: (x, y, confidence)
    xy = joints_2d_batch[..., :2]                              # (B, J, 2)
    conf = torch.ones(B, J, 1, device=device)
    pose2d_3ch = torch.cat([xy, conf], dim=-1)                 # (B, J, 3)

    # Duplicate single frame to 243 to activate full temporal embedding
    mb_input = pose2d_3ch.unsqueeze(1).repeat(1, _MB_MAXLEN, 1, 1)  # (B, 243, J, 3)

    with torch.no_grad():
        pose3d_seq = model(mb_input)    # (B, 243, J, 3) -- DSTformer.forward()

    # Take centre frame (index 121), matching MotionBERT data_stride=81 convention
    centre = _MB_MAXLEN // 2            # = 121
    pose3d = pose3d_seq[:, centre, :, :]  # (B, J, 3)
    return pose3d.cpu().numpy()


# -----------------------------------------------------------------
# Main
# -----------------------------------------------------------------
os.makedirs(args.out_dir, exist_ok=True)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# 1. Build & load MotionBERT
print(f"\n{'='*55}")
print("  Building MotionBERT DSTformer ...")
print(f"{'='*55}")
model_mb = build_motionbert().to(device)
model_mb.eval()

if args.checkpoint:
    print(f"  Loading checkpoint: {args.checkpoint}")
    model_mb = load_checkpoint(model_mb, args.checkpoint)
else:
    print("  [WARNING] No checkpoint -- running with random weights!")
    print("  Use --checkpoint /path/to/best_epoch.bin for pretrained results.")

# Count parameters
n_params = sum(p.numel() for p in model_mb.parameters())
print(f"  DSTformer parameters: {n_params/1e6:.1f}M")

# 2. Dataset & Dataloader
dataset_name = cfg.DATASET.train_list[0]
print(f"\n  Dataset : {dataset_name}")
dataset = get_train_dataset(dataset_name, args)
loader  = DataLoader(dataset, batch_size=4, shuffle=False, num_workers=0)
print(f"  Samples : {len(dataset)}")

# 3. Run inference & visualize
print(f"\n{'='*55}")
print(f"  Visualizing {args.num_samples} samples ...")
print(f"  Output  : {args.out_dir}/")
print(f"{'='*55}")

img_count = 0
for batch_idx, (inputs_b, targets_b, meta_b) in enumerate(loader):
    if img_count >= args.num_samples:
        break

    joints_2d_batch = inputs_b['joints'].to(device)             # (B, J, 2)
    pred_3d_batch   = run_motionbert(model_mb, joints_2d_batch, device)  # (B, J, 3)

    batch_size = inputs_b['img'].shape[0]
    for b in range(batch_size):
        if img_count >= args.num_samples:
            break

        # -- Prepare 2D image -------------------------------
        img_tensor  = inputs_b['img'][b].numpy()               # (C, H, W)
        img_np      = np.transpose(img_tensor, (1, 2, 0))
        img_np      = (img_np * 255).astype(np.uint8)
        img_bgr     = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)

        joints      = inputs_b['joints'][b].numpy()            # (J, 2)
        joints_mask = inputs_b['joints_mask'][b].numpy()       # (J, 1)
        joints_px   = (joints + 1) / 2.0 * 256.0              # Denorm -> pixel
        img_bgr     = draw_2d_skeleton(img_bgr, joints_px, joints_mask)

        # -- Build 3-panel figure ---------------------------
        fig = plt.figure(figsize=(18, 6))

        # Panel 1: Input image + 2D skeleton
        ax1 = fig.add_subplot(131)
        ax1.imshow(cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB))
        ax1.axis('off')
        ax1.set_title('Input Image + 2D KP', fontsize=11)

        # Panel 2: GT 3D skeleton (green / blue)
        ax2 = fig.add_subplot(132, projection='3d')
        if 'orig_joint_cam' in targets_b:
            gt_3d = targets_b['orig_joint_cam'][b].numpy()    # (J, 3)
            draw_3d_skeleton(ax2, gt_3d,
                             color_joint='green', color_bone='blue',
                             title='GT 3D (orig_joint_cam)')
        else:
            ax2.set_title('GT not available')

        # Panel 3: MotionBERT predicted 3D (red / orange)
        ax3 = fig.add_subplot(133, projection='3d')
        pred_3d = pred_3d_batch[b]                             # (J, 3)
        draw_3d_skeleton(ax3, pred_3d,
                         color_joint='red', color_bone='orange',
                         title='MotionBERT Predicted 3D')

        plt.suptitle(f'Sample {img_count:03d}', fontsize=13, y=1.02)
        plt.tight_layout()
        out_path = os.path.join(args.out_dir, f'mb_compare_{img_count:03d}.jpg')
        plt.savefig(out_path, dpi=100, bbox_inches='tight')
        plt.close(fig)

        print(f"  [{img_count+1:>2}/{args.num_samples}] Saved: {out_path}")
        img_count += 1

print(f"\n{'='*55}")
print(f"  DONE! {img_count} images saved to: {args.out_dir}/")
print(f"{'='*55}\n")





