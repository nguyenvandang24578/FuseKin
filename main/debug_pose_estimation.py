"""
Debug script: Chay GraphormerNet tren data thuc te va so sanh ket qua voi GT.
Luu moi sample thanh 1 anh gom 4 cot:
  [Input Image + 2D KP] | [GT 3D Skeleton] | [Pred 3D] | [Top-Down Overlap]

Chay tu thu muc goc:
    python main/debug_pose_estimation.py \
        --cfg ./config/train_init_mesh.yaml \
        --gpu 0 \
        --checkpoint /path/to/posenet.pth \
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
                    help='Path to Pose_Estimation checkpoint')
parser.add_argument('--num_samples', type=int, default=10,
                    help='Number of samples to visualize')
parser.add_argument('--out_dir',     type=str, default='debug_pose_est_vis',
                    help='Output directory for images')
parser.add_argument('--split',       type=str, default='train', choices=['train', 'test'],
                    help='Dataset split to visualize (train or test)')
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
import __init_path

from utils.jotr_dataset import get_train_dataset, get_test_dataset
from torch.utils.data import DataLoader

# Import models
from models.backbones.resnet import ResNetBackbone
from models.Pose_Estimation import get_model

# -----------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------
SKELETON = [
    (0, 1), (1, 2), (2, 3),           # R_Leg
    (0, 4), (4, 5), (5, 6),           # L_Leg
    (0, 7), (7, 8), (8, 9), (9, 10),  # Spine & Head
    (8, 14), (14, 15), (15, 16),      # R_Arm
    (8, 11), (11, 12), (12, 13),      # L_Arm
]

def draw_2d_skeleton(img_bgr, joints_px, joints_mask):
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
    x =  joint_3d[:, 0]
    y = -joint_3d[:, 1]
    z =  joint_3d[:, 2]

    ax.scatter(x, z, y, c=color_joint, marker='o', s=20)
    for (p1, p2) in SKELETON:
        ax.plot([x[p1], x[p2]], [z[p1], z[p2]], [y[p1], y[p2]], c=color_bone)

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

def run_pose_estimation(model, backbone, img_tensor, joints_2d_batch, device):
    """
    Run Pose_Estimation on a batch.
    img_tensor: (B, 3, H, W)
    joints_2d_batch: (B, J, 2) tensor -- normalised 2D joints from dataloader
    
    Returns: (B, J, 3) numpy array -- predicted 3D joints
    """
    T = 16  # usually 16

    # 1. Chạy backbone để sinh img_feat
    _, global_feature = backbone(img_tensor)
    global_feature = global_feature.view(global_feature.size(0), -1) # Flatten to (B, 2048)
    
    # 2. Nhân bản feature ra T frames
    img_feat = global_feature.unsqueeze(1).repeat(1, T, 1) # (B, T, 2048)
    
    # 3. Chuẩn bị x (joints 2D) theo yêu cầu:
    # B1: Đưa từ [-1, 1] về không gian Pixel của ảnh crop (thường là 256x256)
    img_w, img_h = img_tensor.shape[3], img_tensor.shape[2]  # Lấy w, h thực tế từ img_tensor (thường 256)
    xy_px = (joints_2d_batch[..., :2] + 1.0) / 2.0 * img_w
    
    # B2: Chuẩn hóa lại bằng hàm normalize_screen_coordinates (sử dụng toán tử của PyTorch)
    # X / w * 2 - np.array([1, h / w])
    offset = torch.tensor([1.0, img_h / img_w], device=device, dtype=xy_px.dtype)
    xy_norm = (xy_px / img_w) * 2.0 - offset
    
    # B3: Đưa về root-relative (trừ đi gốc tọa độ pelvis)
    xy_norm = xy_norm - xy_norm[:, 0:1, :]
    x = xy_norm.unsqueeze(1).repeat(1, T, 1, 1) # (B, T, J, 2)
    print(f"pose 2D có shape là {x.shape}")
    print(f"img feat có shape là {img_feat.shape}")
    with torch.no_grad():
        pose3d = model(x, img_feat) # (B, J, 3)
        
    return pose3d.cpu().numpy()

# -----------------------------------------------------------------
# Main
# -----------------------------------------------------------------
if __name__ == '__main__':
    os.makedirs(args.out_dir, exist_ok=True)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    print(f"\n{'='*55}")
    print("  Building GraphormerNet & ResNetBackbone ...")
    print(f"{'='*55}")

    backbone = ResNetBackbone(cfg.MODEL.resnet_type).to(device)
    backbone.eval()

    # Khởi tạo mô hình Pose_Estimation (pretrained=False vì ta tự load checkpoint)
    model = get_model(pretrained=False).to(device)
    model.eval()

    if args.checkpoint:
        print(f"  Loading checkpoint: {args.checkpoint}")
        checkpoint = torch.load(args.checkpoint, map_location='cpu')
        state_dict = checkpoint.get('model_state_dict', checkpoint)
        new_state_dict = {}
        for k, v in state_dict.items():
            if k.startswith('module.'):
                k = k[7:]
            new_state_dict[k] = v
        model.load_state_dict(new_state_dict, strict=False)
    else:
        print("  [WARNING] No checkpoint -- running with random weights!")
        print("  Use --checkpoint /path/to/posenet.pth for pretrained results.")

    if args.split == 'train':
        dataset_name = cfg.DATASET.train_list[0]
        print(f"\n  Dataset (Train) : {dataset_name}")
        dataset = get_train_dataset(dataset_name, args)
    else:
        dataset_name = cfg.DATASET.test_list[0]
        print(f"\n  Dataset (Test) : {dataset_name}")
        dataset = get_test_dataset(dataset_name, args)

    loader = DataLoader(dataset, batch_size=4, shuffle=False, num_workers=0)
    print(f"  Samples : {len(dataset)}")

    print(f"\n{'='*55}")
    print(f"  Visualizing {args.num_samples} samples ...")
    print(f"  Output  : {args.out_dir}/")
    print(f"{'='*55}")

    img_count = 0
    for batch_idx, (inputs_b, targets_b, meta_b) in enumerate(loader):
        if img_count >= args.num_samples:
            break

        img_tensor = inputs_b['img'].to(device)
        joints_2d = inputs_b['joints'].to(device)
        
        pred_3d_batch = run_pose_estimation(model, backbone, img_tensor, joints_2d, device)

        batch_size = img_tensor.shape[0]
        for b in range(batch_size):
            if img_count >= args.num_samples:
                break

            # -- Prepare 2D image -------------------------------
            img_t  = inputs_b['img'][b].numpy()
            img_np = np.transpose(img_t, (1, 2, 0))
            img_np = (img_np * 255).astype(np.uint8)
            img_bgr = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)

            joints = inputs_b['joints'][b].numpy()
            mask   = inputs_b['joints_mask'][b].numpy()
            joints_px = (joints + 1) / 2.0 * 256.0
            img_bgr = draw_2d_skeleton(img_bgr, joints_px, mask)

            # -- Build 4-panel figure ---------------------------
            fig = plt.figure(figsize=(24, 6))

            # Panel 1: Input image + 2D skeleton
            ax1 = fig.add_subplot(141)
            ax1.imshow(cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB))
            ax1.axis('off')
            ax1.set_title('Input Image + 2D KP', fontsize=11)

            pred_3d = pred_3d_batch[b]
            gt_3d = None
            if 'orig_joint_cam' in targets_b:
                gt_3d = targets_b['orig_joint_cam'][b].numpy()
            elif 'smpl_mesh_cam' in targets_b:
                mesh = targets_b['smpl_mesh_cam'][b].numpy()
                gt_3d = np.dot(dataset.dataset.h36m_joint_regressor, mesh)
                
            if gt_3d is not None:
                print(f"\n[Sample {img_count:03d}] So sánh tọa độ:")
                print(f" - Mẫu GT khớp 0 (pelvis): {gt_3d[0]}")
                print(f" - Mẫu Pred khớp 0 (pelvis): {pred_3d[0]}")

                gt_root = gt_3d[0:1, :]
                pred_root = pred_3d[0:1, :]
                
                gt_rel = gt_3d - gt_root
                pred_rel = pred_3d - pred_root
                
                gt_bone_len = np.linalg.norm(gt_rel[1] - gt_rel[0])
                pred_bone_len = np.linalg.norm(pred_rel[1] - pred_rel[0])
                print(f" - Chiều dài xương đùi (khớp 1-0): GT = {gt_bone_len:.4f}, Pred = {pred_bone_len:.4f}")

                if pred_bone_len > 10 * gt_bone_len:
                    print(" -> Phát hiện Pred dùng đơn vị Millimet, GT dùng đơn vị Mét! Tự động chia Pred cho 1000...")
                    pred_rel = pred_rel / 1000.0
                    
                mpjpe = np.sqrt(np.sum((pred_rel - gt_rel) ** 2, axis=1)).mean()
                print(f" -> MPJPE (đã Root-Relative & đồng bộ Scale): {mpjpe:.4f} mét ({mpjpe*1000:.2f} mm)")

            # Panel 2: GT 3D skeleton
            ax2 = fig.add_subplot(142, projection='3d')
            if gt_3d is not None:
                draw_3d_skeleton(ax2, gt_rel, color_joint='green', color_bone='blue', title='GT 3D (Root Relative)')
            else:
                ax2.set_title('GT not available')

            # Panel 3: Pred 3D
            ax3 = fig.add_subplot(143, projection='3d')
            draw_3d_skeleton(ax3, pred_rel if gt_3d is not None else pred_3d, color_joint='red', color_bone='orange', title='Pred 3D (Root Relative)')

            # Panel 4: OVERLAP TOP-DOWN VIEW
            ax4 = fig.add_subplot(144, projection='3d')
            if gt_3d is not None:
                draw_3d_skeleton(ax4, gt_rel, color_joint='green', color_bone='blue', title='Top-Down Overlap (GT=Green, Pred=Red)')
                joints_x = pred_rel[:, 0]
                joints_y = -pred_rel[:, 1]
                joints_z = pred_rel[:, 2]
                ax4.scatter(joints_x, joints_z, joints_y, c='red', s=20)
                for bone in SKELETON:
                    ax4.plot([joints_x[bone[0]], joints_x[bone[1]]],
                             [joints_z[bone[0]], joints_z[bone[1]]],
                             [joints_y[bone[0]], joints_y[bone[1]]], color='orange')
                ax4.view_init(elev=90, azim=-90)
                ax4.set_title('Top-Down Overlap (GT=Blue, Pred=Orange)', fontsize=11)
            else:
                ax4.set_title('Top-Down not available')

            plt.suptitle(f'Sample {img_count:03d}', fontsize=13, y=1.02)
            plt.tight_layout()
            out_path = os.path.join(args.out_dir, f'pose_est_compare_{img_count:03d}.jpg')
            plt.savefig(out_path, dpi=100, bbox_inches='tight')
            plt.close(fig)

            print(f"  [{img_count+1:>2}/{args.num_samples}] Saved: {out_path}")
            img_count += 1

    print(f"\n{'='*55}")
    print(f"  DONE! {img_count} images saved to: {args.out_dir}/")
    print(f"{'='*55}\n")
