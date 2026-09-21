"""
Debug script: In ra shapes + sample values cua dataloader.
Chay tu thu muc main/:
    python debug_dataloader.py --cfg ../config/train_mesh_h36m.yml --gpu 0
"""
import os, sys
sys.path.append('./lib')

import argparse
import warnings
warnings.filterwarnings("ignore")

parser = argparse.ArgumentParser()
parser.add_argument('--cfg',  type=str, default='../config/train_mesh_h36m.yml')
parser.add_argument('--gpu',  type=str, default='0')
parser.add_argument('--num_batches', type=int, default=2, help='So batch muon inspect')
args = parser.parse_args()

from core.config import cfg, update_config
update_config(args.cfg)

os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu

import torch
import numpy as np
import __init_path
from utils.jotr_dataset import get_train_dataset
from torch.utils.data import DataLoader

# -------------------------------------------------
# 1. Tao dataset + dataloader
# -------------------------------------------------
dataset_name = cfg.DATASET.train_list[0]
print(f"\n{'='*50}")
print(f"  Dataset: {dataset_name}")
print(f"{'='*50}")

dataset = get_train_dataset(dataset_name, args)
loader  = DataLoader(dataset, batch_size=4, shuffle=False, num_workers=0)

print(f"  Tong so samples: {len(dataset)}")

# -------------------------------------------------
# 2. Lay 1 sample don le de xem cau truc
# -------------------------------------------------
print(f"\n{'-'*50}")
print("  [__getitem__] Cau truc 1 sample:")
print(f"{'-'*50}")

inputs, targets, meta = dataset[0]

def print_dict_shapes(d, title):
    print(f"\n  >> {title}:")
    for k, v in d.items():
        if hasattr(v, 'shape'):
            arr = np.array(v)
            print(f"     {k:30s} shape={arr.shape}  dtype={arr.dtype}  "
                  f"min={arr.min():.4f}  max={arr.max():.4f}")
        else:
            print(f"     {k:30s} = {v}")

print_dict_shapes(inputs,  "inputs")
print_dict_shapes(targets, "targets")
print_dict_shapes(meta,    "meta_info")

# -------------------------------------------------
# 3. Chay qua vai batch, in shapes cua batch
# -------------------------------------------------
print(f"\n{'-'*50}")
print(f"  [DataLoader] Shapes qua {args.num_batches} batch (batch_size=4):")
print(f"{'-'*50}")

for batch_idx, (inputs_b, targets_b, meta_b) in enumerate(loader):
    if batch_idx >= args.num_batches:
        break

    print(f"\n  -- Batch {batch_idx} --")
    for k, v in inputs_b.items():
        if torch.is_tensor(v):
            print(f"     inputs['{k}']:  {tuple(v.shape)}  dtype={v.dtype}  "
                  f"min={v.min():.3f}  max={v.max():.3f}")

    for k, v in targets_b.items():
        if torch.is_tensor(v):
            print(f"     targets['{k}']: {tuple(v.shape)}  dtype={v.dtype}  "
                  f"min={v.min():.3f}  max={v.max():.3f}")

    for k, v in meta_b.items():
        if torch.is_tensor(v):
            print(f"     meta['{k}']:    {tuple(v.shape)}  dtype={v.dtype}  "
                  f"min={v.min():.3f}  max={v.max():.3f}")
        else:
            print(f"     meta['{k}']:    {type(v).__name__}")

# -------------------------------------------------
# 4. In sample values chi tiet cho key quan trong
# -------------------------------------------------
print(f"\n{'-'*50}")
print("  [Chi tiet] Sample dau tien trong batch 0:")
print(f"{'-'*50}")

for batch_idx, (inputs_b, targets_b, meta_b) in enumerate(loader):
    img = inputs_b['img'][0]   # (C, H, W)
    print(f"\n  img:           shape={tuple(img.shape)}  "
          f"mean={img.mean():.3f}  std={img.std():.3f}")

    joints = inputs_b['joints'][0]
    print(f"  joints (2D):   shape={tuple(joints.shape)}")
    print(f"     5 joints dau:\n{joints[:5].numpy()}")

    if 'orig_joint_cam' in targets_b:
        cam = targets_b['orig_joint_cam'][0]
        print(f"  orig_joint_cam: shape={tuple(cam.shape)}")
        print(f"     5 joints dau (mm):\n{cam[:5].numpy()}")

    if 'is_3D' in meta_b:
        print(f"  is_3D (batch):  {meta_b['is_3D'].numpy()}")

    if 'orig_joint_valid' in meta_b:
        valid = meta_b['orig_joint_valid'][0]
        print(f"  orig_joint_valid: shape={tuple(valid.shape)}  "
              f"sum={valid.sum().item():.0f}/{valid.numel()}")

    break

print(f"\n{'='*50}")
print("  DONE - Debug hoan thanh!")
print(f"{'='*50}\n")

# -------------------------------------------------
# 5. Visualizing 2D Keypoints + Mask
# -------------------------------------------------
import cv2

print(f"\n{'-'*50}")
print("  [Visualizing] Luu anh ket qua de kiem tra 2D keypoints")
print(f"{'-'*50}")

img_count = 0
for batch_idx, (inputs_b, targets_b, meta_b) in enumerate(loader):
    batch_size = inputs_b['img'].shape[0]
    for b in range(batch_size):
        if img_count >= 10:
            break
            
        # Lay anh thu b cua batch (C, H, W)
        img_tensor = inputs_b['img'][b].numpy()  # float32 in [0, 1]
        
        # Chuyen (C, H, W) -> (H, W, C), va scale len [0, 255]
        img_np = np.transpose(img_tensor, (1, 2, 0))
        img_np = (img_np * 255).astype(np.uint8)
        
        # Chuyen RGB (cua ToTensor) sang BGR de luu bang cv2
        img_bgr = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)
        
        # Lay joints 2D va mask
        joints = inputs_b['joints'][b].numpy()       # (17, 2) in [-1, 1]
        joints_mask = inputs_b['joints_mask'][b].numpy() # (17, 1) in {0, 1}

        # Denormalize joints tu [-1, 1] ve pixel [0, 256]
        joints_px = (joints + 1) / 2.0 * 256.0
        
        # Ve tung diem len anh
        for i in range(len(joints_px)):
            x, y = int(joints_px[i, 0]), int(joints_px[i, 1])
            valid = int(joints_mask[i, 0])
            
            # Color: Xanh la neu valid (1), Do neu invalid/bi che (0)
            color = (0, 255, 0) if valid == 1 else (0, 0, 255)
            
            # Ve hinh tron
            cv2.circle(img_bgr, (x, y), radius=4, color=color, thickness=-1)
            
            # In so thu tu cua khop de de theo doi
            cv2.putText(img_bgr, str(i), (x+5, y+5), cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)
            
        out_path = f'debug_vis_{img_count}.jpg'
        cv2.imwrite(out_path, img_bgr)
        
        # --- 3D SKELETON VISUALIZATION ---
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        
        fig = plt.figure(figsize=(6, 6))
        ax = fig.add_subplot(111, projection='3d')
        
        if 'orig_joint_cam' in targets_b:
            joint_3d = targets_b['orig_joint_cam'][b].numpy() # (17, 3)
            
            # H36M Skeleton Edges (17 joints)
            skeleton = [
                (0, 1), (1, 2), (2, 3), # R_Leg
                (0, 4), (4, 5), (5, 6), # L_Leg
                (0, 7), (7, 8), (8, 9), (9, 10), # Spine & Head
                (8, 14), (14, 15), (15, 16), # R_Arm
                (8, 11), (11, 12), (12, 13)  # L_Arm
            ]
            
            # Extract x, y, z
            # Note: in camera coords, Y is down. For visualization, we flip Y.
            x = joint_3d[:, 0]
            y = -joint_3d[:, 1]
            z = joint_3d[:, 2]
            
            # Plot joints
            ax.scatter(x, z, y, c='r', marker='o', s=20)
            
            # Plot bones
            for edge in skeleton:
                p1, p2 = edge
                ax.plot([x[p1], x[p2]], [z[p1], z[p2]], [y[p1], y[p2]], c='b')
                
            # Set labels
            ax.set_xlabel('X')
            ax.set_ylabel('Depth (Z)')
            ax.set_zlabel('Y (Flipped)')
            ax.set_title('3D Skeleton (Camera Coords)')
            
            # Make axes scale equal
            max_range = np.array([x.max()-x.min(), y.max()-y.min(), z.max()-z.min()]).max() / 2.0
            mid_x = (x.max()+x.min()) * 0.5
            mid_y = (y.max()+y.min()) * 0.5
            mid_z = (z.max()+z.min()) * 0.5
            ax.set_xlim(mid_x - max_range, mid_x + max_range)
            ax.set_ylim(mid_z - max_range, mid_z + max_range)
            ax.set_zlim(mid_y - max_range, mid_y + max_range)
            
            # Save 3D figure
            out_3d_path = f'debug_vis_3d_{img_count}.jpg'
            plt.savefig(out_3d_path)
            plt.close(fig)
            print(f"  Da luu anh 2D vao {out_path} va 3D vao {out_3d_path}")
        else:
            print(f"  Da luu anh 2D vao file: {out_path}")
        
        img_count += 1

    if img_count >= 10:
        break

print(f"\n{'='*50}")
print("  HOAN THANH VISUALIZE!")
print(f"{'='*50}\n")
