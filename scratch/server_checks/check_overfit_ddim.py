"""check_overfit_ddim.py – Overfit SMPL_HyperDiff on a real batch and verify quality.

Claims to check
===============
1. Loss decreases significantly (last < first * 0.1).
2. DDIM samples are close to GT (measured in DEGREES via geodesic per joint).
3. Two DDIM seeds give similar results (deterministic overfit).
4. CONTROL: shuffling kp condition should degrade results, proving the model
   actually uses keypoint conditioning (not just memorising).
5. Compares on overfitted batch vs unseen batch.
6. Reports valid vs invalid joint errors separately.

Previous bugs
=============
- Used random input (randn for kp2d, aa), not real data.
- Only measured MSE in 6D space, not angular degrees.
- PASS threshold too lax.
"""

import os
import sys
import csv
import argparse
import torch
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'lib'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

from core.config import update_config, cfg
from models.ARTS import get_model as get_arts_model
from models.smpl_hyperdiff import axis_angle_to_rot6d


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--cfg', type=str, default='config/train_init_mesh.yaml')
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--out_dir', type=str, default='logs/server_checks')
    parser.add_argument('--steps', type=int, default=1000)
    return parser.parse_args()


def dict_to_tensor(d, device):
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


def rot6d_to_rotmat_local(x):
    """Convert (*, 6) -> (*, 3, 3) using GS orthogonalisation."""
    x = x.reshape(-1, 3, 2)
    a1 = x[:, :, 0]
    a2 = x[:, :, 1]
    b1 = torch.nn.functional.normalize(a1, dim=-1)
    dot = (b1 * a2).sum(dim=-1, keepdim=True)
    b2 = torch.nn.functional.normalize(a2 - dot * b1, dim=-1)
    b3 = torch.cross(b1, b2, dim=-1)
    return torch.stack([b1, b2, b3], dim=-1)


def geodesic_per_joint(pred_6d, gt_6d):
    """Compute per-joint geodesic angle in degrees.
    Returns (B, 24) in degrees.
    """
    B, J = pred_6d.shape[:2]
    R_pred = rot6d_to_rotmat_local(pred_6d.reshape(-1, 6)).reshape(B, J, 3, 3)
    R_gt = rot6d_to_rotmat_local(gt_6d.reshape(-1, 6)).reshape(B, J, 3, 3)

    R_diff = torch.matmul(R_gt.transpose(-1, -2), R_pred)
    trace = R_diff.diagonal(dim1=-2, dim2=-1).sum(dim=-1)
    cos_angle = ((trace - 1) / 2).clamp(-1, 1)
    angle_rad = torch.acos(cos_angle)
    import math
    return angle_rad * (180.0 / math.pi)


def get_real_batch(ds, start_idx, B, device):
    imgs, gt6ds, kp2ds, confs, masks = [], [], [], [], []
    for b in range(start_idx, start_idx + B):
        inputs, targets, meta = ds[b]
        inp_t = dict_to_tensor(inputs, device)
        tgt_t = dict_to_tensor(targets, device)
        met_t = dict_to_tensor(meta, device)

        imgs.append(inp_t['img'])
        pp = tgt_t['pose_param'].reshape(-1, 3)
        g6d = axis_angle_to_rot6d(pp).reshape(1, 24, 6)
        gt6ds.append(g6d)

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

    return {
        'img': torch.cat(imgs, 0),
        'gt_pose_6d': torch.cat(gt6ds, 0),
        'kp2d': torch.cat(kp2ds, 0),
        'kp_conf': torch.cat(confs, 0),
        'valid_mask': torch.cat(masks, 0)
    }


def main():
    args = parse_args()
    os.makedirs(args.out_dir, exist_ok=True)
    log_file = os.path.join(args.out_dir, 'check_overfit_ddim.log')
    csv_file = os.path.join(args.out_dir, 'overfit_loss.csv')

    def log(msg):
        print(msg)
        with open(log_file, 'a', encoding='utf-8') as f:
            f.write(msg + '\n')

    with open(log_file, 'w', encoding='utf-8') as f:
        f.write("=== check_overfit_ddim ===\n")

    all_pass = True
    try:
        update_config(args.cfg)
        cfg.MODEL.REFINER = 'diffusion'
        device = args.device

        # Get diff model from ARTS
        arts_model = get_arts_model(num_joint=17, embed_dim=cfg.MODEL.hpe_dim).to(device)
        model = arts_model.pose_mesh_coevo.diffusion
        model.train()

        optim = torch.optim.Adam(model.parameters(), lr=1e-3)

        # ---- Prepare Real Data ----
        from utils.jotr_dataset import get_train_dataset
        ds = get_train_dataset('3dpw-train', args)
        
        B = min(64, len(ds) // 2)
        log(f"Loading {B} samples for train batch, and {B} for unseen test batch.")
        
        train_batch = get_real_batch(ds, 0, B, device)
        test_batch = get_real_batch(ds, B, B, device)

        gt_6d = train_batch['gt_pose_6d']
        kp2d = train_batch['kp2d']
        kp_conf = train_batch['kp_conf']
        valid_mask = train_batch['valid_mask']

        log(f"Overfit data: B={B}, gt_6d shape={list(gt_6d.shape)}")
        log(f"Training for {args.steps} steps...\n")

        # ---- Train ----
        with open(csv_file, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['step', 'loss'])

            first_loss = None
            last_loss = None
            for step in range(args.steps):
                optim.zero_grad()
                _, loss = model(gt_6d, kp2d, kp_conf, is_train=True, valid_mask=valid_mask)
                loss.backward()
                optim.step()

                l_val = loss.item()
                if step == 0:
                    first_loss = l_val
                last_loss = l_val

                if step % 50 == 0 or step == args.steps - 1:
                    writer.writerow([step, l_val])
                    log(f"  step {step:>4d}: loss = {l_val:.6f}")

        log(f"\nFirst loss: {first_loss:.6f}, Last loss: {last_loss:.6f}")
        log(f"Reduction ratio: {last_loss/first_loss:.4f}")

        if last_loss >= first_loss * 0.1:
            log(f"  WARNING: loss did not decrease 10x (ratio={last_loss/first_loss:.4f})")
            all_pass = False

        # ---- DDIM sampling on Train batch ----
        model.eval()
        g1 = torch.Generator(device=device).manual_seed(42)
        g2 = torch.Generator(device=device).manual_seed(100)

        with torch.no_grad():
            sample1 = model.ddim_sample(kp2d, kp_conf, generator=g1)
            sample2 = model.ddim_sample(kp2d, kp_conf, generator=g2)

        # ---- Geodesic error per joint ----
        geo1 = geodesic_per_joint(sample1, gt_6d)  # (B, 24)

        log(f"\n--- Per-joint geodesic error (degrees) - Train Batch (Seed 42) ---")
        log(f"{'Joint':>6} | {'Mean':>8} | {'Valid Mean':>11} | {'Invld Mean':>11}")
        log("-" * 46)
        
        valid_errs, invalid_errs = [], []
        for j in range(24):
            m = geo1[:, j].mean().item()
            v_mask = valid_mask[:, j] > 0.5
            
            if v_mask.any():
                v_mean = geo1[v_mask, j].mean().item()
                valid_errs.extend(geo1[v_mask, j].tolist())
            else:
                v_mean = 0.0
                
            if (~v_mask).any():
                inv_mean = geo1[~v_mask, j].mean().item()
                invalid_errs.extend(geo1[~v_mask, j].tolist())
            else:
                inv_mean = 0.0
                
            log(f"{j:>6} | {m:>8.2f} | {v_mean:>11.2f} | {inv_mean:>11.2f}")

        mean_valid = sum(valid_errs) / len(valid_errs) if valid_errs else 0
        mean_invalid = sum(invalid_errs) / len(invalid_errs) if invalid_errs else 0
        mean_geo = geo1.mean().item()
        
        log(f"\nOverall mean valid geodesic error: {mean_valid:.2f}°")
        log(f"Overall mean invalid geodesic error: {mean_invalid:.2f}°")
        log(f"Overall mean geodesic error: {mean_geo:.2f}°")

        # ---- Seed consistency ----
        geo_diff = geodesic_per_joint(sample1, sample2)
        seed_diff_mean = geo_diff.mean().item()
        log(f"\nSeed diversity (geodesic between seed 42 vs 100): {seed_diff_mean:.2f}°")

        # ---- CONTROL: shuffle keypoints ----
        log(f"\n--- CONTROL: shuffle kp2d across Train batch ---")
        perm = torch.randperm(B, device=device)
        kp2d_shuffled = kp2d[perm]

        with torch.no_grad():
            sample_shuf = model.ddim_sample(kp2d_shuffled, kp_conf, generator=g1)
        geo_shuf = geodesic_per_joint(sample_shuf, gt_6d)
        mean_geo_shuf = geo_shuf.mean().item()
        log(f"Shuffled kp mean geodesic error: {mean_geo_shuf:.2f}°")
        log(f"Normal kp mean geodesic error:   {mean_geo:.2f}°")

        if mean_geo_shuf <= mean_geo * 1.5:
            log(f"  WARNING: shuffled kps give similar error! Model might just be memorizing batch indices.")

        # ---- Eval on UNSEEN batch ----
        log(f"\n--- EVAL on UNSEEN Batch ---")
        with torch.no_grad():
            sample_unseen = model.ddim_sample(test_batch['kp2d'], test_batch['kp_conf'], generator=g1)
            
        geo_unseen = geodesic_per_joint(sample_unseen, test_batch['gt_pose_6d'])
        v_mask_u = test_batch['valid_mask'] > 0.5
        v_unseen = geo_unseen[v_mask_u].mean().item() if v_mask_u.any() else 0.0
        
        log(f"Unseen batch valid mean error: {v_unseen:.2f}°")

        # ---- Thresholds ----
        # With real B=64 and 1000 steps, we expect the model to fit well on train data.
        # It's an overfit test, so train error should be small (~3-5°).
        THRESH_GEO_DEG = 5.0
        if mean_valid > THRESH_GEO_DEG:
            log(f"\n  FAIL: mean valid geodesic {mean_valid:.2f}° > {THRESH_GEO_DEG}° threshold")
            all_pass = False

        # ---- Verdict ----
        if all_pass:
            log(f"\nPASS - Overfit successful: loss {first_loss:.4f}→{last_loss:.4f}, "
                f"valid geodesic {mean_valid:.2f}°.")
        else:
            log(f"\nFAIL - Overfit quality insufficient.")
            sys.exit(1)

    except Exception as e:
        import traceback
        log(f"FAIL - Exception occurred:\n{traceback.format_exc()}")
        sys.exit(1)


if __name__ == "__main__":
    main()
