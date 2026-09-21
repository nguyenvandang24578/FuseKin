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

for batch_idx, (inputs_b, targets_b, meta_b) in enumerate(loader):
    # Lay anh dau tien cua batch (C, H, W)
    img_tensor = inputs_b['img'][0].numpy()  # float32 in [0, 1]
    
    # Chuyen (C, H, W) -> (H, W, C), va scale len [0, 255]
    img_np = np.transpose(img_tensor, (1, 2, 0))
    img_np = (img_np * 255).astype(np.uint8)
    
    # Chuyen RGB (cua ToTensor) sang BGR de luu bang cv2
    img_bgr = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)
    
    # Lay joints 2D va mask
    joints = inputs_b['joints'][0].numpy()       # (17, 2) in [-1, 1]
    joints_mask = inputs_b['joints_mask'][0].numpy() # (17, 1) in {0, 1}

    # Denormalize joints tu [-1, 1] ve pixel [0, 256]
    # Vi output_hm_shape = 64, va scale cua anh = 256 => factor = 256
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
        
    out_path = 'debug_vis.jpg'
    cv2.imwrite(out_path, img_bgr)
    print(f"  Da luu anh truc quan hoa vao file: {out_path} ! Ban co the mo de xem.")
    
    break

print(f"\n{'='*50}")
print("  HOAN THANH VISUALIZE!")
print(f"{'='*50}\n")
