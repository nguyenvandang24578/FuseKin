"""check_forward_shapes.py – Verify model output shapes in train & eval modes.

Claims to check
===============
1. All tensor outputs have expected shapes: pose (B,72), shape (B,10),
   cam (B,3), mesh (B,6890,3), joint_proj (B,30,2), pred_pose_6d_refined (B,24,6).
2. No NaN/Inf in any output.
3. Works in all keypoint modes (GT, noisy, detector).

Previous bugs
=============
- inputs['joints'] is numpy → crash on .unsqueeze().
- Pose2Mesh(cfg) wrong constructor (takes num_joint, embed_dim).
- PASS was unconditional as long as no exception.
"""

import os
import sys
import argparse
import torch
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'lib'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

from core.config import update_config, cfg
from models.ARTS import get_model as get_arts_model
from models.smpl_hyperdiff import axis_angle_to_rot6d


def dict_to_tensor(d, device):
    """Convert all numpy arrays in a dict to tensors, adding batch dim."""
    out = {}
    for k, v in d.items():
        if isinstance(v, np.ndarray):
            out[k] = torch.from_numpy(v).float().unsqueeze(0).to(device)
        elif isinstance(v, (float, int, bool)):
            out[k] = torch.tensor([v], device=device).float()
        elif torch.is_tensor(v):
            out[k] = v.float().unsqueeze(0).to(device)
        else:
            out[k] = v
    return out


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--cfg', type=str, default='config/train_init_mesh.yaml')
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--out_dir', type=str, default='logs/server_checks')
    parser.add_argument('--real_batch', action='store_true',
                        help='Use a real batch from 3DPW')
    return parser.parse_args()


def main():
    args = parse_args()
    os.makedirs(args.out_dir, exist_ok=True)
    log_file = os.path.join(args.out_dir, 'check_forward_shapes.log')

    def log(msg):
        print(msg)
        with open(log_file, 'a', encoding='utf-8') as f:
            f.write(msg + '\n')

    with open(log_file, 'w', encoding='utf-8') as f:
        f.write("=== check_forward_shapes ===\n")

    all_pass = True
    try:
        update_config(args.cfg)
        cfg.MODEL.REFINER = 'diffusion'
        device = args.device

        model = get_arts_model(num_joint=17, embed_dim=cfg.MODEL.hpe_dim).to(device)
        
        log(f"Constructed ARTS model. embed_dim={cfg.MODEL.hpe_dim}")

        # ---- Prepare inputs ----
        if args.real_batch:
            log("Using REAL batch from dataset...")
            from utils.jotr_dataset import get_train_dataset
            ds = get_train_dataset('3dpw-train', args)

            B = 4
            imgs, poses, gt6ds, kp2ds, confs, masks = [], [], [], [], [], []
            for b in range(B):
                inputs, targets, meta = ds[b]
                inp_t = dict_to_tensor(inputs, device)
                tgt_t = dict_to_tensor(targets, device)
                met_t = dict_to_tensor(meta, device)

                imgs.append(inp_t['img'])
                poses.append(inp_t['joints'][:, :17])

                # GT 6D
                pp = tgt_t['pose_param'].reshape(-1, 3)
                g6d = axis_angle_to_rot6d(pp).reshape(1, 24, 6)
                gt6ds.append(g6d)

                # kp2d
                kp = tgt_t['orig_joint_img'][:, :17, :2].clone()
                kp[..., 0] = kp[..., 0] / cfg.output_hm_shape[2] * 2 - 1
                kp[..., 1] = kp[..., 1] / cfg.output_hm_shape[1] * 2 - 1
                kp2ds.append(kp)

                c = met_t['orig_joint_trunc'][:, :17]
                if c.dim() == 3:
                    c = c.squeeze(-1)
                confs.append(c)

                fv = met_t['fit_param_valid']
                if fv.shape[-1] == 72:
                    m = fv.reshape(1, 24, 3)[:, :, 0]
                else:
                    m = torch.ones(1, 24, device=device)
                masks.append(m)

            input_image = torch.cat(imgs, 0)
            input_pose = torch.cat(poses, 0)
            gt_pose_6d = torch.cat(gt6ds, 0)
            kp2d = torch.cat(kp2ds, 0)
            kp_conf = torch.cat(confs, 0)
            pose_valid_mask = torch.cat(masks, 0)
        else:
            log("Using FAKE random tensors...")
            B = 2
            input_image = torch.randn(B, 3, 256, 256, device=device)
            input_pose = torch.randn(B, 17, 3, device=device)
            gt_pose_6d = torch.randn(B, 24, 6, device=device)
            kp2d = torch.randn(B, 17, 2, device=device).clamp(-1, 1)
            kp_conf = torch.ones(B, 17, device=device)
            pose_valid_mask = torch.ones(B, 24, device=device)

        # ---- Expected shapes ----
        expected_shapes = {
            'joint_proj': (B, 30, 2),
            'joint_cam': (B, 30, 3),
            'smpl_mesh_cam': (B, 6890, 3),
            'smpl_mesh_cam_proj': (B, 6890, 3),
            'smpl_pose': (B, 72),
            'smpl_shape': (B, 10),
            'cam_param': (B, 3),
        }
        train_extra = {
            'pred_pose_6d_refined': (B, 24, 6),
        }

        # Define 3 modes:
        # Mode 1: Train, GT 3D, GT kp2d (Noisy)
        # Mode 2: Train, lift 2D (detector), detector kp2d
        # Mode 3: Eval, lift 2D, detector kp2d
        modes = [
            ("Train GT", True, True, kp2d + torch.randn_like(kp2d)*0.1),
            ("Train Detector", True, False, kp2d),
            ("Eval Detector", False, False, kp2d)
        ]

        for mode_name, is_train, use_gt_3d, mode_kp2d in modes:
            log(f"\n--- {mode_name} mode (B={B}) ---")
            if is_train:
                model.train()
            else:
                model.eval()

            with torch.set_grad_enabled(is_train):
                out = model(
                    input_image, input_pose, is_train=is_train, use_gt_3d=use_gt_3d,
                    gt_pose_6d=gt_pose_6d if is_train else None, 
                    kp2d=mode_kp2d, kp_conf=kp_conf,
                    pose_valid_mask=pose_valid_mask if is_train else None
                )

            for k, v in out.items():
                if torch.is_tensor(v):
                    shape = tuple(v.shape)
                    has_nan = torch.isnan(v).any().item()
                    has_inf = torch.isinf(v).any().item()
                    log(f"  {k:>25}: {str(list(shape)):>20}  "
                        f"nan={has_nan}  inf={has_inf}")

                    # Check expected shape
                    exp = expected_shapes.get(k) or (train_extra.get(k) if is_train else None)
                    if exp and shape != exp:
                        log(f"    SHAPE MISMATCH: expected {exp}")
                        all_pass = False
                    if has_nan or has_inf:
                        log(f"    FAIL: NaN or Inf detected!")
                        all_pass = False
                else:
                    log(f"  {k:>25}: {type(v).__name__} = {v}")

            # Check diff_loss exists in train
            if is_train:
                if 'diff_loss' not in out:
                    log("  FAIL: diff_loss missing from train output!")
                    all_pass = False
                else:
                    dl = out['diff_loss']
                    log(f"  diff_loss: {dl.item():.4f}")

        # ---- Verdict ----
        if all_pass:
            log("\nPASS - All shapes correct, no NaN/Inf.")
        else:
            log("\nFAIL - Shape or NaN/Inf issues detected.")
            sys.exit(1)

    except Exception as e:
        import traceback
        log(f"FAIL - Exception occurred:\n{traceback.format_exc()}")
        sys.exit(1)


if __name__ == "__main__":
    main()
