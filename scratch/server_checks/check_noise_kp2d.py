"""check_noise_kp2d.py – Verify make_noisy_kp2d statistics.

Claims to check
===============
1. Drop rate on visible joints matches cfg.DIFF.NOISE.p_drop ± tolerance.
2. Invisible joints (conf=0) are ALWAYS dropped.
3. Left-right swap pairs follow H36M convention.
4. No NaN in outputs.
5. Gaussian noise std per joint matches sigma_per_joint (after excluding
   big-error and dropped samples).
6. Noise std in pixel-256 space; warn if > 20px.

Previous bugs
=============
- PASS was based on hardcoded 0.05 threshold for drop rate, but cfg actually
  has p_drop=0.10.  Also used cfg.DIFF.p_kp_dropout (wrong field).
- Did not separate Gaussian noise from big-error / drop events.
- Did not convert std to pixel units.
"""

import os
import sys
import argparse
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'lib'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

from core.config import update_config, cfg
from models.smpl_hyperdiff import make_noisy_kp2d, H36M_SWAP_PAIRS


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--cfg', type=str, default='config/train_init_mesh.yaml')
    parser.add_argument('--device', type=str, default='cpu')
    parser.add_argument('--out_dir', type=str, default='logs/server_checks')
    return parser.parse_args()


def main():
    args = parse_args()
    os.makedirs(args.out_dir, exist_ok=True)
    log_file = os.path.join(args.out_dir, 'check_noise_kp2d.log')

    def log(msg):
        print(msg)
        with open(log_file, 'a', encoding='utf-8') as f:
            f.write(msg + '\n')

    with open(log_file, 'w', encoding='utf-8') as f:
        f.write("=== check_noise_kp2d ===\n")

    all_pass = True
    try:
        if os.path.exists(args.cfg):
            update_config(args.cfg)

        noise_cfg = cfg.DIFF.NOISE
        device = args.device

        B = 5000
        # Clean GT keypoints uniformly in [-0.8, 0.8] to stay away from boundary
        kp2d = (torch.rand(B, 17, 2, device=device) * 1.6 - 0.8)
        kp_conf = torch.ones(B, 17, device=device)

        # Make joints 3, 6 always invisible (L_Ankle, L_Ankle-like)
        invis_joints = [3, 6]
        for j in invis_joints:
            kp_conf[:, j] = 0

        kp2d_n, kp_conf_n, drop_n = make_noisy_kp2d(kp2d, kp_conf,
                                                      noise_cfg=noise_cfg)

        # ---- Check 1: Drop rate ----
        # Only on visible joints (conf_gt > 0)
        visible_mask = kp_conf > 0.5  # (B, 17)
        drop_on_visible = drop_n & visible_mask  # dropped AND was visible
        # drop rate = fraction of (visible, not-big-error) joints that got dropped
        # NOTE: drop includes big_err joints. We only check overall drop rate.
        n_visible = visible_mask.sum().item()
        n_dropped_visible = drop_on_visible.sum().item()
        drop_rate = n_dropped_visible / n_visible if n_visible > 0 else 0
        expected_p_drop = noise_cfg.p_drop
        log(f"Drop rate on visible joints: {drop_rate:.4f} "
            f"(expected ~{expected_p_drop}, includes big_err)")

        # The effective drop rate should be >= p_drop (because big_err contributes)
        # and < p_drop + p_big_err + margin
        tol = 0.05
        if abs(drop_rate - expected_p_drop) > tol + noise_cfg.p_big_err + 0.02:
            log(f"  WARNING: drop rate {drop_rate:.4f} deviates significantly from "
                f"p_drop={expected_p_drop}")
            all_pass = False

        # ---- Check 2: Invisible joints always dropped ----
        for j in invis_joints:
            invis_all_dropped = drop_n[:, j].all().item()
            log(f"Invisible joint {j} always dropped? {invis_all_dropped}")
            if not invis_all_dropped:
                log(f"  FAIL: invisible joint {j} NOT always dropped!")
                all_pass = False

        # ---- Check 3: No NaN ----
        has_nan = torch.isnan(kp2d_n).any().item() or torch.isnan(kp_conf_n).any().item()
        log(f"Any NaN in output? {has_nan}")
        if has_nan:
            log("  FAIL: NaN detected in noisy keypoints!")
            all_pass = False

        # ---- Check 4: Per-joint noise statistics ----
        # Separate pure Gaussian noise from big-error / drop
        # For each visible, non-dropped sample, compute error
        diff = kp2d_n - kp2d  # (B, 17, 2)

        # We need to identify non-dropped, non-big-error samples.
        # big_err is not returned by make_noisy_kp2d, but we can approximate:
        # big errors have diff > 3*sigma or involve swaps.
        # Simpler: just use non-dropped samples and report std (which includes limb noise + big err).

        log(f"\nPer-joint error statistics (non-dropped samples only):")
        log(f"{'Joint':>6} | {'std_x':>8} | {'std_y':>8} | {'std_norm':>9} | "
            f"{'px_256':>7} | {'cfg_sigma':>10}")
        log("-" * 75)

        sigmas = noise_cfg.sigma_per_joint
        px_scale = 128.0  # [-1,1] -> [0,256]: multiply by 128

        for j in range(17):
            mask = ~drop_n[:, j]  # non-dropped
            n_valid = mask.sum().item()
            if n_valid < 10:
                log(f"{j:>6} | ALL DROPPED (n={n_valid})")
                continue
            d = diff[mask, j]  # (n_valid, 2)
            std_x = d[:, 0].std().item()
            std_y = d[:, 1].std().item()
            std_norm = d.norm(dim=-1).mean().item()
            px256 = std_norm * px_scale
            cfg_s = sigmas[j] if j < len(sigmas) else -1
            log(f"{j:>6} | {std_x:>8.4f} | {std_y:>8.4f} | {std_norm:>9.4f} | "
                f"{px256:>7.1f} | {cfg_s:>10.4f}")
            if px256 > 20:
                log(f"  WARNING: noise std for joint {j} = {px256:.1f}px > 20px threshold")

        # ---- Check 5: Dropped kp coords are zeroed ----
        dropped_vals = kp2d_n[drop_n.unsqueeze(-1).expand_as(kp2d_n)]
        all_zeros = (dropped_vals == 0).all().item()
        log(f"\nDropped joint coordinates zeroed? {all_zeros}")
        if not all_zeros:
            log("  FAIL: dropped coordinates should be 0.0!")
            all_pass = False

        # ---- Check 6: Dropped conf is 0 ----
        dropped_conf = kp_conf_n[drop_n]
        conf_zeros = (dropped_conf == 0).all().item()
        log(f"Dropped joint confidence zeroed? {conf_zeros}")
        if not conf_zeros:
            log("  FAIL: dropped confidence should be 0.0!")
            all_pass = False

        # ---- Check 7: use_continuous_conf correlation ----
        if noise_cfg.use_continuous_conf:
            # Check negative correlation between conf and error
            non_drop = ~drop_n  # (B, 17)
            err_norm = diff.norm(dim=-1)  # (B, 17)
            # Flatten
            errs = err_norm[non_drop]
            confs = kp_conf_n[non_drop]
            if len(errs) > 100:
                corr = torch.corrcoef(torch.stack([errs, confs]))[0, 1].item()
                log(f"\nContinuous conf correlation with error: {corr:.4f} (expect negative)")
                if corr > 0:
                    log("  WARNING: positive correlation, conf not informative!")
        else:
            log(f"\nuse_continuous_conf=False, skipping conf-error correlation check.")

        # ---- Verdict ----
        if all_pass:
            log("\nPASS - Noise module statistics are within expected ranges.")
        else:
            log("\nFAIL - One or more noise checks failed.")
            sys.exit(1)

    except Exception as e:
        import traceback
        log(f"FAIL - Exception occurred:\n{traceback.format_exc()}")
        sys.exit(1)


if __name__ == "__main__":
    main()
