"""check_grad_paths.py – Verify gradient isolation between losses.

Claims to check
===============
For each loss (diff_loss, body_joint_proj, smpl_shape), check which parameter
groups receive gradients. Expected:
  diff_loss       → diffusion YES, cam/shape/Teacher NO
  body_joint_proj → diffusion NO (detach), cam_head YES, shape/fuse YES, Teacher YES
  smpl_shape      → diffusion NO, shape_head/fuse YES, Teacher YES

Added Negative Control: Set cfg.LOSS.DETACH_POSE_FOR_PROJ=False. Then body_joint_proj
MUST propagate gradient to diffusion. If not, the test is invalid.
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
from core.loss import JOTRCoordLoss


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


def run_tests(model, input_pose, input_image, gt_pose_6d, kp2d, kp_conf, pose_valid_mask, gt_orig_joint_img, orig_joint_trunc, log):
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

    # ---- Forward pass ----
    out = model.forward_arts(
        input_image, input_pose, is_train=True, use_gt_3d=True,
        gt_pose_6d=gt_pose_6d, kp2d=kp2d, kp_conf=kp_conf,
        pose_valid_mask=pose_valid_mask
    )

    # ---- Build losses ----
    losses = {}

    # diff_loss
    losses['diff_loss'] = out['diff_loss']

    # body_joint_proj (exactly like base.py)
    cam = out['cam_param']
    scale = cam[:, 0:1, None]
    trans = cam[:, 1:3, None].transpose(1, 2)
    
    # We use model.pose_mesh_coevo.joint_regressor_t because that's what MULTIMODEL uses for mesh -> 17 joints
    J_reg = model.pose_mesh_coevo.joint_regressor_t
    pred_pose_17 = torch.matmul(J_reg[None, :, :], out['smpl_mesh_cam_proj'])
    
    proj_2d = scale * pred_pose_17[:, :, :2] + trans
    proj_pixel = (proj_2d + 1.0) * 0.5 * cfg.input_img_shape[0]
    pred_joint_proj = proj_pixel * (cfg.output_hm_shape[1] / cfg.input_img_shape[0])
    
    jotr_coord_loss = JOTRCoordLoss().to(input_image.device)
    loss_body_joint_proj = jotr_coord_loss(
        pred_joint_proj,
        gt_orig_joint_img[:, :, :2],
        orig_joint_trunc
    ).mean()
    
    losses['body_joint_proj'] = loss_body_joint_proj

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
    return results_table


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

        # Build ARTS model (which includes backbone + Multimodel)
        model = get_arts_model(num_joint=17, embed_dim=cfg.MODEL.hpe_dim).to(device)
        model.train()
        
        log("Constructed ARTS model successfully.")

        # ---- Prepare inputs ----
        if args.real_batch:
            log("Using REAL batch from dataset...")
            from utils.jotr_dataset import get_train_dataset
            ds = get_train_dataset('3dpw-train', args)

            B = 4
            imgs, poses, gt6ds, kp2ds, confs, masks = [], [], [], [], [], []
            gt_orig_imgs, orig_truncs = [], []
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
                
                gt_orig_imgs.append(tgt_t['orig_joint_img'])
                orig_truncs.append(met_t['orig_joint_trunc'])

            input_image = torch.cat(imgs, 0)
            input_pose = torch.cat(poses, 0)
            gt_pose_6d = torch.cat(gt6ds, 0)
            kp2d = torch.cat(kp2ds, 0)
            kp_conf = torch.cat(confs, 0)
            pose_valid_mask = torch.cat(masks, 0)
            gt_orig_joint_img = torch.cat(gt_orig_imgs, 0)
            orig_joint_trunc = torch.cat(orig_truncs, 0)
        else:
            log("Using FAKE random tensors...")
            B = 4
            input_image = torch.randn(B, 3, 256, 256, device=device)
            input_pose = torch.randn(B, 17, 3, device=device)
            gt_pose_6d = torch.randn(B, 24, 6, device=device)
            kp2d = torch.randn(B, 17, 2, device=device).clamp(-1, 1)
            kp_conf = torch.ones(B, 17, device=device)
            pose_valid_mask = torch.ones(B, 24, device=device)
            gt_orig_joint_img = torch.randn(B, 30, 3, device=device)
            orig_joint_trunc = torch.ones(B, 30, 1, device=device)

        log(f"Input shapes: img={list(input_image.shape)}, "
            f"pose={list(input_pose.shape)}, gt6d={list(gt_pose_6d.shape)}")

        
        # Test 1: Normal mode (DETACH_POSE_FOR_PROJ depends on config, usually True)
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
        
        cfg.LOSS.DETACH_POSE_FOR_PROJ = True
        log(f"\n--- NORMAL MODE (cfg.LOSS.DETACH_POSE_FOR_PROJ=True) ---")
        results_table = run_tests(model, input_pose, input_image, gt_pose_6d, kp2d, kp_conf, pose_valid_mask, gt_orig_joint_img, orig_joint_trunc, log)

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

        # Test 2: Negative Control
        log(f"\n--- NEGATIVE CONTROL (cfg.LOSS.DETACH_POSE_FOR_PROJ=False) ---")
        cfg.LOSS.DETACH_POSE_FOR_PROJ = False
        res_negative = run_tests(model, input_pose, input_image, gt_pose_6d, kp2d, kp_conf, pose_valid_mask, gt_orig_joint_img, orig_joint_trunc, log)
        
        diff_proj_actual = res_negative['body_joint_proj'].get('diffusion', 'NONE')
        if diff_proj_actual == 'GRAD>0':
            log(f"  ✓ Negative control passed: body_joint_proj propagated gradient to diffusion (got {diff_proj_actual}).")
        else:
            log(f"  ✗ Negative control FAILED: expected body_joint_proj to propagate gradient to diffusion, got {diff_proj_actual}.")
            all_pass = False

        # ---- Verdict ----
        if mismatches:
            log(f"\nFAIL - {len(mismatches)} expectation mismatches in NORMAL mode:")
            for m in mismatches:
                log(f"  {m[0]}/{m[1]}: expected {m[2]}, got {m[3]}")
            all_pass = False
        else:
            log(f"\nAll expectations matched in NORMAL mode.")

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
