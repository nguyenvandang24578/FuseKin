"""check_grad_paths.py – Verify gradient isolation between losses.

Claims to check
===============
For each loss (diff_loss, body_joint_proj, smpl_shape), check which parameter
groups receive gradients. Expected:
  diff_loss       → diffusion YES, cam/shape/Teacher NO
  body_joint_proj → diffusion NO (detach), cam_head YES, shape/fuse YES, Teacher YES
  smpl_shape      → diffusion NO, shape_head/fuse YES, Teacher YES

Previous bugs
=============
- inputs['joints'] is numpy → crash on .unsqueeze(). All dataset outputs
  are numpy, need conversion.
- Pose2Mesh(cfg) is wrong constructor call (takes num_joint, embed_dim).
- Only tested diff_loss + proj_loss. Did not test smpl_shape or per-loss
  backward in isolation.
- "model.smpl_layer.J_regressor" does not exist; it's model.joint_regressor_t.
- forward(input_pose, input_image, ...) → should be forward(joints, img_feats, ...).
"""

import os
import sys
import argparse
import torch
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'lib'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

from core.config import update_config, cfg
from models.Multimodel import Pose2Mesh
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


def classify_param(name):
    """Classify a named parameter into a module group."""
    if 'diffusion.' in name:
        return 'diffusion'
    elif 'cam_head' in name:
        return 'cam_head'
    elif 'shape_head' in name:
        return 'shape_head'
    elif 'fuse_shape' in name:
        return 'fuse_shape'
    elif 'shape_embed' in name:
        return 'shape_embed'
    elif 'shape_token' in name:
        return 'shape_token'
    elif 'fusion.' in name:
        return 'teacher/fusion'
    elif 'vposer' in name:
        return 'vposer(frozen)'
    else:
        return 'other'


def grad_status(p):
    """Return 'GRAD>0', 'ZERO', or 'NONE' for a parameter."""
    if p.grad is None:
        return 'NONE'
    if p.grad.norm().item() > 1e-12:
        return 'GRAD>0'
    return 'ZERO'


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
    log_file = os.path.join(args.out_dir, 'check_grad_paths.log')

    def log(msg):
        print(msg)
        with open(log_file, 'a', encoding='utf-8') as f:
            f.write(msg + '\n')

    with open(log_file, 'w', encoding='utf-8') as f:
        f.write("=== check_grad_paths ===\n")

    all_pass = True
    try:
        update_config(args.cfg)
        cfg.MODEL.REFINER = 'diffusion'
        device = args.device

        model = Pose2Mesh(num_joint=17, embed_dim=cfg.MODEL.hpe_dim).to(device)
        model.train()

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

                # GT 6D from pose_param
                pp = tgt_t['pose_param'].reshape(-1, 3)
                g6d = axis_angle_to_rot6d(pp).reshape(1, 24, 6)
                gt6ds.append(g6d)

                # kp2d from orig_joint_img (heatmap->[-1,1])
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
            B = 4
            input_image = torch.randn(B, 3, 256, 256, device=device)
            input_pose = torch.randn(B, 17, 3, device=device)
            gt_pose_6d = torch.randn(B, 24, 6, device=device)
            kp2d = torch.randn(B, 17, 2, device=device).clamp(-1, 1)
            kp_conf = torch.ones(B, 17, device=device)
            pose_valid_mask = torch.ones(B, 24, device=device)

        log(f"Input shapes: img={list(input_image.shape)}, "
            f"pose={list(input_pose.shape)}, gt6d={list(gt_pose_6d.shape)}")

        # ---- Forward pass ----
        # Pose2Mesh.forward(joints, img_feats, is_train, ...)
        out = model(
            input_pose, input_image, is_train=True,
            gt_pose_6d=gt_pose_6d, kp2d=kp2d, kp_conf=kp_conf,
            pose_valid_mask=pose_valid_mask
        )

        # ---- Collect all trainable parameter groups ----
        groups = {}
        for name, p in model.named_parameters():
            if not p.requires_grad:
                continue
            g = classify_param(name)
            if g not in groups:
                groups[g] = []
            groups[g].append((name, p))

        log(f"\nTrainable parameter groups: {sorted(groups.keys())}")
        for g, params in sorted(groups.items()):
            log(f"  {g}: {len(params)} params")

        # ---- Define expected gradient table ----
        # loss_name -> group -> expected status
        expected = {
            'diff_loss': {
                'diffusion': 'GRAD>0',
                'cam_head': 'NONE',
                'shape_head': 'NONE',
                'fuse_shape': 'NONE',
                'shape_embed': 'NONE',
                'shape_token': 'NONE',
                'teacher/fusion': 'NONE',
            },
            'body_joint_proj': {
                'diffusion': 'NONE',  # detached
                'cam_head': 'GRAD>0',
                'shape_head': 'GRAD>0',
                'fuse_shape': 'GRAD>0',
                'teacher/fusion': 'GRAD>0',
            },
            'smpl_shape': {
                'diffusion': 'NONE',
                'cam_head': 'NONE',
                'shape_head': 'GRAD>0',
                'fuse_shape': 'GRAD>0',
                'teacher/fusion': 'GRAD>0',
            },
        }

        # ---- Build losses ----
        losses = {}

        # diff_loss
        losses['diff_loss'] = out['diff_loss']

        # body_joint_proj: replicate the exact computation from base.py
        cam = out['cam_param']
        J_reg = model.joint_regressor_t
        mesh_proj = out['smpl_mesh_cam_proj']
        pred_j17 = torch.matmul(J_reg[None, :, :].expand(B, -1, -1), mesh_proj)
        # Weak-perspective: scale * xy + trans
        scale = cam[:, 0:1, None]
        trans = cam[:, 1:3, None].transpose(1, 2)
        proj_2d = scale * pred_j17[:, :, :2] + trans
        # Simple L1 target (just need a loss that flows grad)
        losses['body_joint_proj'] = proj_2d.abs().mean()

        # smpl_shape
        losses['smpl_shape'] = out['smpl_shape'].abs().mean()

        # ---- Per-loss backward + check ----
        results_table = {}  # loss -> group -> actual status

        for loss_name in ['diff_loss', 'body_joint_proj', 'smpl_shape']:
            log(f"\n{'='*60}")
            log(f"Backward from: {loss_name}")
            log(f"{'='*60}")

            model.zero_grad()
            loss_val = losses[loss_name]
            if loss_val.dim() == 0:
                loss_val.backward(retain_graph=True)
            else:
                loss_val.mean().backward(retain_graph=True)

            results_table[loss_name] = {}
            for g, params in sorted(groups.items()):
                statuses = [grad_status(p) for _, p in params]
                # Aggregate: if any param has GRAD>0, group = GRAD>0
                if any(s == 'GRAD>0' for s in statuses):
                    group_status = 'GRAD>0'
                elif any(s == 'ZERO' for s in statuses):
                    group_status = 'ZERO'
                else:
                    group_status = 'NONE'

                results_table[loss_name][g] = group_status

                # Print norms for non-None
                norms = [p.grad.norm().item() for _, p in params
                         if p.grad is not None]
                if norms:
                    log(f"  {g:>20}: {group_status:>7}  "
                        f"(norms: min={min(norms):.2e}, max={max(norms):.2e})")
                else:
                    log(f"  {g:>20}: {group_status}")

        # ---- Check expectations ----
        log(f"\n{'='*60}")
        log("EXPECTATION TABLE")
        log(f"{'='*60}")
        log(f"{'Loss':>20} | {'Group':>20} | {'Expected':>8} | {'Actual':>8} | {'Match':>5}")
        log("-" * 75)

        mismatches = []
        for loss_name, exp_groups in expected.items():
            for g, exp_status in exp_groups.items():
                if g not in results_table.get(loss_name, {}):
                    actual = 'N/A'
                    match = False
                else:
                    actual = results_table[loss_name][g]
                    # NONE and ZERO both mean "no useful gradient"
                    # GRAD>0 means gradient is flowing
                    if exp_status == 'NONE':
                        match = actual in ('NONE', 'ZERO')
                    elif exp_status == 'GRAD>0':
                        match = actual == 'GRAD>0'
                    else:
                        match = actual == exp_status

                sym = '✓' if match else '✗'
                log(f"{loss_name:>20} | {g:>20} | {exp_status:>8} | {actual:>8} | {sym:>5}")
                if not match:
                    mismatches.append((loss_name, g, exp_status, actual))

        # ---- List requires_grad=True but grad=None after full backward ----
        log(f"\n--- Params with requires_grad=True but never got gradient ---")
        model.zero_grad()
        total_loss = sum(l.mean() for l in losses.values())
        total_loss.backward()
        no_grad_list = []
        for name, p in model.named_parameters():
            if p.requires_grad and p.grad is None:
                no_grad_list.append(name)
        if no_grad_list:
            log(f"  {len(no_grad_list)} params have no grad after total backward:")
            for n in no_grad_list[:20]:
                log(f"    {n}")
            if len(no_grad_list) > 20:
                log(f"    ... and {len(no_grad_list)-20} more")
        else:
            log("  All trainable params received gradient. ✓")

        # ---- Verdict ----
        if mismatches:
            log(f"\nFAIL - {len(mismatches)} expectation mismatches:")
            for m in mismatches:
                log(f"  {m[0]}/{m[1]}: expected {m[2]}, got {m[3]}")
            all_pass = False
        else:
            log(f"\nAll expectations matched.")

        if all_pass:
            log("PASS - Gradient paths are isolated correctly.")
        else:
            log("FAIL - Gradient isolation violated.")
            sys.exit(1)

    except Exception as e:
        import traceback
        log(f"FAIL - Exception occurred:\n{traceback.format_exc()}")
        sys.exit(1)


if __name__ == "__main__":
    main()
