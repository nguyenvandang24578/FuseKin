import os
import sys
import argparse
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'lib'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

from core.config import update_config, cfg
from models.Multimodel import Pose2Mesh
from utils.jotr_dataset import get_train_dataset

import numpy as np

def to_tensor(x, device):
    if isinstance(x, np.ndarray):
        return torch.from_numpy(x).float().unsqueeze(0).to(device)
    elif torch.is_tensor(x):
        return x.float().unsqueeze(0).to(device)
    return x

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--cfg', type=str, default='experiment/mesh_3dpw.yaml')
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--out_dir', type=str, default='logs/server_checks')
    parser.add_argument('--real_batch', action='store_true', help='Use a real batch from 3DPW')
    return parser.parse_args()

def main():
    args = parse_args()
    os.makedirs(args.out_dir, exist_ok=True)
    log_file = os.path.join(args.out_dir, 'check_grad_paths.log')

    def log(msg):
        print(msg)
        with open(log_file, 'a', encoding='utf-8') as f:
            f.write(msg + '\n')

    with open(log_file, 'w', encoding='utf-8') as f:
        f.write("=== check_grad_paths ===\n")

    try:
        update_config(args.cfg)
        cfg.MODEL.REFINER = 'diffusion'
        model = Pose2Mesh(cfg).to(args.device)
        model.train()
        
        if args.real_batch:
            log("Using REAL batch from dataset...")
            ds = get_train_dataset('3dpw-train', args)
            inputs, targets, meta = ds[0]
            
            # Prepare inputs with unsqueeze to add batch dim
            input_image = to_tensor(inputs['img'], args.device)
            input_pose = to_tensor(inputs['joints'], args.device)
            gt_pose_6d = torch.randn(1, 24, 6, device=args.device) # Dummy for diffusion loss
            kp2d = to_tensor(targets['orig_joint_img'][:17, :2], args.device)
            
            orig_trunc = meta['orig_joint_trunc'][:17]
            if isinstance(orig_trunc, np.ndarray):
                kp_conf = torch.from_numpy(orig_trunc).float().unsqueeze(0).squeeze(-1).to(args.device)
            else:
                kp_conf = orig_trunc.float().unsqueeze(0).squeeze(-1).to(args.device)
                
            pose_valid_mask = torch.ones(1, 24, device=args.device)
        else:
            log("Using FAKE random tensors...")
            B = 4
            input_image = torch.randn(B, 3, 256, 256, device=args.device)
            input_pose = torch.randn(B, 17, 3, device=args.device)
            gt_pose_6d = torch.randn(B, 24, 6, device=args.device)
            kp2d = torch.randn(B, 17, 2, device=args.device)
            kp_conf = torch.ones(B, 17, device=args.device)
            pose_valid_mask = torch.ones(B, 24, device=args.device)
        
        out = model(
            input_image, input_pose, is_train=True, use_gt_3d=True,
            gt_pose_6d=gt_pose_6d, kp2d=kp2d, kp_conf=kp_conf,
            pose_valid_mask=pose_valid_mask
        )
        
        # Test 1: Full diff_loss backward
        log("\n--- TEST 1: Backward from diff_loss ---")
        model.zero_grad()
        loss = out['diff_loss'].mean()
        loss.backward(retain_graph=True)
        
        diff_no_grad = []
        diff_has_grad = False
        cam_shape_has_grad = False
        fusion_has_grad = False
        requires_but_no_grad = []
        
        for name, p in model.named_parameters():
            if 'diffusion' in name:
                if p.grad is not None:
                    diff_has_grad = True
                else:
                    diff_no_grad.append(name)
            elif 'cam_head' in name or 'shape' in name:
                if p.grad is not None and p.grad.abs().sum() > 0:
                    cam_shape_has_grad = True
            elif 'fusion' in name:
                if p.grad is not None and p.grad.abs().sum() > 0:
                    fusion_has_grad = True
                    
            if p.requires_grad and p.grad is None:
                requires_but_no_grad.append(name)
                
        log(f"Diffusion params received gradient? {diff_has_grad}")
        if diff_no_grad:
            log(f"WARNING: some diffusion params have NO grad: {diff_no_grad}")
            
        log(f"Cam/Shape branches received gradient from diff_loss? {cam_shape_has_grad} (expected False)")
        log(f"Fusion received gradient from diff_loss? {fusion_has_grad} (expected False)")
        log(f"Params with requires_grad=True but grad=None: {len(requires_but_no_grad)}")
        
        # Test 2: Check body_joint_proj loss isolation
        log("\n--- TEST 2: Backward from body_joint_proj (via smpl_mesh_cam_proj) ---")
        model.zero_grad()
        
        # Mock projection loss logic using smpl_mesh_cam_proj
        cam = out['cam_param']
        scale = cam[:, 0:1, None]
        trans = cam[:, 1:3, None].transpose(1, 2)
        # using J_regressor from model
        pred_pose_17 = torch.matmul(model.smpl_layer.J_regressor[None, :, :], out['smpl_mesh_cam_proj'])
        proj_2d = scale * pred_pose_17[:, :, :2] + trans
        loss2 = proj_2d.sum()
        loss2.backward()
        
        diff_grad_after_proj = False
        cam_head_grad_norm = 0.0
        shape_head_grad_norm = 0.0
        
        for name, p in model.named_parameters():
            if p.grad is not None:
                grad_norm = p.grad.norm().item()
                if 'diffusion' in name and grad_norm > 0:
                    diff_grad_after_proj = True
                if 'cam_head' in name:
                    cam_head_grad_norm += grad_norm
                if 'shape' in name:
                    shape_head_grad_norm += grad_norm
                
        log(f"Diffusion received gradient from cam/proj loss? {diff_grad_after_proj} (expected False)")
        log(f"cam_head gradient norm: {cam_head_grad_norm:.4f} (expected > 0)")
        log(f"shape_head gradient norm: {shape_head_grad_norm:.4f} (expected > 0)")
        
        if diff_has_grad and not diff_no_grad and not cam_shape_has_grad and not diff_grad_after_proj and cam_head_grad_norm > 0 and shape_head_grad_norm > 0:
            log("\nPASS - Gradient paths are isolated correctly.")
        else:
            log("\nFAIL - Gradient leaking or not propagating correctly.")
            sys.exit(1)
            
    except Exception as e:
        import traceback
        log(f"FAIL - Exception occurred:\n{traceback.format_exc()}")
        sys.exit(1)

if __name__ == "__main__":
    main()
