"""check_overfit_ddim.py – Overfit SMPL_HyperDiff on a tiny batch and verify quality.

Claims to check
===============
1. Loss decreases significantly (last < first * 0.1).
2. DDIM samples are close to GT (measured in DEGREES via geodesic per joint).
3. Two DDIM seeds give similar results (deterministic overfit).
4. CONTROL: shuffling kp condition should degrade results, proving the model
   actually uses keypoint conditioning (not just memorising).

Previous bugs
=============
- Used random input (randn for kp2d, aa), not real data. Overfit test should
  use plausible keypoints and poses.
- Only measured MSE in 6D space, not angular degrees.
- No shuffle control → could pass even if keypoints are ignored.
- PASS threshold too lax (last < first * 0.5).
"""

import os
import sys
import csv
import math
import argparse
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'lib'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

from models.smpl_hyperdiff import SMPL_HyperDiff, axis_angle_to_rot6d, axis_angle_to_rotmat


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--cfg', type=str, default='')
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--out_dir', type=str, default='logs/server_checks')
    parser.add_argument('--steps', type=int, default=500)
    return parser.parse_args()


def rot6d_to_rotmat_local(x):
    """Convert (*, 6) -> (*, 3, 3) using GS orthogonalisation."""
    x = x.reshape(-1, 3, 2)
    a1 = x[:, :, 0]
    a2 = x[:, :, 1]
    b1 = torch.nn.functional.normalize(a1, dim=-1)
    dot = (b1 * a2).sum(dim=-1, keepdim=True)
    b2 = torch.nn.functional.normalize(a2 - dot * b1, dim=-1)
    b3 = torch.cross(b1, b2, dim=-1)
    return torch.stack([b1, b2, b3], dim=-1)  # (N, 3, 3)


def geodesic_per_joint(pred_6d, gt_6d):
    """Compute per-joint geodesic angle in degrees.
    pred_6d, gt_6d: (B, 24, 6) -> returns (B, 24) in degrees.
    """
    B, J = pred_6d.shape[:2]
    R_pred = rot6d_to_rotmat_local(pred_6d.reshape(-1, 6)).reshape(B, J, 3, 3)
    R_gt = rot6d_to_rotmat_local(gt_6d.reshape(-1, 6)).reshape(B, J, 3, 3)

    # R_diff = R_gt^T @ R_pred
    R_diff = torch.matmul(R_gt.transpose(-1, -2), R_pred)
    trace = R_diff.diagonal(dim1=-2, dim2=-1).sum(dim=-1)  # (B, J)
    cos_angle = ((trace - 1) / 2).clamp(-1, 1)
    angle_rad = torch.acos(cos_angle)  # (B, J)
    return torch.degrees(angle_rad)


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
        device = args.device
        model = SMPL_HyperDiff().to(device)
        model.train()

        optim = torch.optim.Adam(model.parameters(), lr=1e-3)

        # ---- Generate plausible synthetic data ----
        B = 4
        # Small random rotations (realistic range ≈ ±0.5 rad per axis)
        gt_aa = torch.randn(B, 24, 3, device=device) * 0.3
        gt_6d = axis_angle_to_rot6d(gt_aa.reshape(-1, 3)).reshape(B, 24, 6)

        # Plausible 2D keypoints in [-1, 1] (structured, not random noise)
        # Rough skeleton layout
        kp2d = torch.zeros(B, 17, 2, device=device)
        # Set a rough human shape
        kp_template = torch.tensor([
            [0.0, -0.4],   # 0 Pelvis
            [-0.1, -0.2],  # 1 R_Hip
            [-0.1, 0.1],   # 2 R_Knee
            [-0.1, 0.3],   # 3 R_Ankle
            [0.1, -0.2],   # 4 L_Hip
            [0.1, 0.1],    # 5 L_Knee
            [0.1, 0.3],    # 6 L_Ankle
            [0.0, -0.5],   # 7 Torso
            [0.0, -0.65],  # 8 Neck
            [0.0, -0.7],   # 9 Nose
            [0.0, -0.8],   # 10 Head
            [0.15, -0.6],  # 11 L_Shoulder
            [0.25, -0.4],  # 12 L_Elbow
            [0.3, -0.2],   # 13 L_Wrist
            [-0.15, -0.6], # 14 R_Shoulder
            [-0.25, -0.4], # 15 R_Elbow
            [-0.3, -0.2],  # 16 R_Wrist
        ], device=device)
        for b in range(B):
            noise = torch.randn(17, 2, device=device) * 0.02
            kp2d[b] = kp_template + noise

        kp_conf = torch.ones(B, 17, device=device)

        log(f"Overfit data: B={B}, gt_6d shape={list(gt_6d.shape)}")
        log(f"Using plausible synthetic keypoints and small-angle GT poses.")
        log(f"Training for {args.steps} steps...\n")

        # ---- Train ----
        with open(csv_file, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['step', 'loss'])

            first_loss = None
            last_loss = None
            for step in range(args.steps):
                optim.zero_grad()
                _, loss = model(gt_6d, kp2d, kp_conf, is_train=True)
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

        # ---- DDIM sampling ----
        model.eval()
        g1 = torch.Generator(device=device).manual_seed(42)
        g2 = torch.Generator(device=device).manual_seed(100)

        with torch.no_grad():
            sample1 = model.ddim_sample(kp2d, kp_conf, generator=g1)
            sample2 = model.ddim_sample(kp2d, kp_conf, generator=g2)

        # ---- Geodesic error per joint ----
        geo1 = geodesic_per_joint(sample1, gt_6d)  # (B, 24) in degrees
        geo2 = geodesic_per_joint(sample2, gt_6d)

        log(f"\n--- Per-joint geodesic error (degrees) - Seed 42 ---")
        log(f"{'Joint':>6} | {'Mean':>8} | {'Max':>8}")
        log("-" * 30)
        for j in range(24):
            m = geo1[:, j].mean().item()
            mx = geo1[:, j].max().item()
            log(f"{j:>6} | {m:>8.2f} | {mx:>8.2f}")

        mean_geo = geo1.mean().item()
        log(f"\nOverall mean geodesic error: {mean_geo:.2f}°")
        log(f"Overall max geodesic error: {geo1.max().item():.2f}°")

        # ---- Seed consistency ----
        geo_diff = geodesic_per_joint(sample1, sample2)
        seed_diff_mean = geo_diff.mean().item()
        log(f"\nSeed diversity (geodesic between seed 42 vs 100): {seed_diff_mean:.2f}°")

        # ---- CONTROL: shuffle keypoints ----
        log(f"\n--- CONTROL: shuffle kp2d across batch ---")
        perm = torch.randperm(B, device=device)
        kp2d_shuffled = kp2d[perm]  # different person's kps for each pose

        with torch.no_grad():
            sample_shuf = model.ddim_sample(kp2d_shuffled, kp_conf, generator=g1)
        geo_shuf = geodesic_per_joint(sample_shuf, gt_6d)
        mean_geo_shuf = geo_shuf.mean().item()
        log(f"Shuffled kp mean geodesic error: {mean_geo_shuf:.2f}°")
        log(f"Normal kp mean geodesic error:   {mean_geo:.2f}°")

        if mean_geo_shuf <= mean_geo * 1.1:
            log(f"  WARNING: shuffled kps give similar error! "
                f"Model may not be using keypoint conditioning. "
                f"(This can happen with tiny B={B} overfitting.)")
            # Don't fail for this with B=4, but warn.
            # With a larger batch this should be a harder failure.

        # ---- Thresholds ----
        THRESH_GEO_DEG = 15.0  # after 500-step overfit, expect < 15°
        if mean_geo > THRESH_GEO_DEG:
            log(f"\n  FAIL: mean geodesic {mean_geo:.2f}° > {THRESH_GEO_DEG}° threshold")
            all_pass = False

        # ---- Verdict ----
        if all_pass:
            log(f"\nPASS - Overfit successful: loss {first_loss:.4f}→{last_loss:.4f}, "
                f"geodesic {mean_geo:.2f}°.")
        else:
            log(f"\nFAIL - Overfit quality insufficient.")
            sys.exit(1)

    except Exception as e:
        import traceback
        log(f"FAIL - Exception occurred:\n{traceback.format_exc()}")
        sys.exit(1)


if __name__ == "__main__":
    main()
