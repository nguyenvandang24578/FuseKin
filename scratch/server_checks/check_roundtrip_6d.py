"""check_roundtrip_6d.py – Verify 6D rotation round-trip consistency.

Claims to check
===============
1. axis_angle -> rot6d -> axis_angle  is lossless  (geodesic < threshold)
2. rot6d_to_rotmat(6d) == rodrigues(aa)             (element-wise < 1e-3)
3. 6D layout is [r00, r01, r10, r11, r20, r21]      (element-wise < 1e-5)

Reports separately for angle groups:
- near 0 (mag < 0.1)
- near pi (mag > pi - 0.1)
- mid (0.1 <= mag <= pi - 0.1)
"""

import os
import sys
import argparse
import torch
import math

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'lib'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

from models.smpl_hyperdiff import axis_angle_to_rot6d, axis_angle_to_rotmat
from utils.transforms import rot6d_to_axis_angle
from geometry import rot6d_to_rotmat


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--cfg', type=str, default='')
    parser.add_argument('--device', type=str, default='cpu')
    parser.add_argument('--out_dir', type=str, default='logs/server_checks')
    return parser.parse_args()


def geodesic_angle(R1, R2):
    R_diff = torch.bmm(R1.transpose(1, 2), R2)
    trace = R_diff.diagonal(dim1=1, dim2=2).sum(dim=1)
    cos_angle = ((trace - 1.0) / 2.0).clamp(-1.0, 1.0)
    return torch.acos(cos_angle)


def main():
    args = parse_args()
    os.makedirs(args.out_dir, exist_ok=True)
    log_file = os.path.join(args.out_dir, 'check_roundtrip_6d.log')

    def log(msg):
        print(msg)
        with open(log_file, 'a', encoding='utf-8') as f:
            f.write(msg + '\n')

    with open(log_file, 'w', encoding='utf-8') as f:
        f.write("=== check_roundtrip_6d ===\n")

    all_pass = True
    try:
        device = args.device
        N = 3000

        # Sample covering [0, pi)
        aa_dir = torch.randn(N, 3, device=device)
        aa_dir = aa_dir / (aa_dir.norm(dim=-1, keepdim=True) + 1e-8)

        # 1000 near zero, 1000 near pi, 1000 mid
        aa_mag = torch.zeros(N, 1, device=device)
        aa_mag[:1000] = torch.rand(1000, 1, device=device) * 0.1
        aa_mag[1000:2000] = math.pi - torch.rand(1000, 1, device=device) * 0.1
        aa_mag[2000:] = 0.1 + torch.rand(1000, 1, device=device) * (math.pi - 0.2)

        aa = aa_dir * aa_mag

        r6d = axis_angle_to_rot6d(aa)
        aa_back = rot6d_to_axis_angle(r6d)

        R_orig = axis_angle_to_rotmat(aa)
        R_back = axis_angle_to_rotmat(aa_back)
        geo = geodesic_angle(R_orig, R_back)

        log("\n--- [Test i] aa->6d->aa Geodesic Error by Group ---")
        
        groups = [
            ("Near Zero (<0.1)", aa_mag.squeeze(-1) < 0.1, 1e-4),
            ("Near Pi (>pi-0.1)", aa_mag.squeeze(-1) > (math.pi - 0.1), 1e-2),
            ("Mid", (aa_mag.squeeze(-1) >= 0.1) & (aa_mag.squeeze(-1) <= (math.pi - 0.1)), 1e-4)
        ]
        
        log(f"{'Group':>20} | {'Max Geo(rad)':>12} | {'Threshold':>10} | {'Status':>6}")
        log("-" * 55)
        for name, mask, thresh in groups:
            if mask.sum() > 0:
                geo_g = geo[mask]
                max_err = geo_g.max().item()
                status = "PASS" if max_err <= thresh else "FAIL"
                if max_err > thresh:
                    all_pass = False
                log(f"{name:>20} | {max_err:>12.2e} | {thresh:>10.0e} | {status:>6}")

        # Test (ii) rotmat check
        R_from_6d = rot6d_to_rotmat(r6d)
        if R_from_6d.shape != R_orig.shape:
            R_from_6d = R_from_6d.reshape(N, 3, 3)
        err_mat = (R_from_6d - R_orig).abs().max().item()
        
        log(f"\n--- [Test ii] rotmat(6d) vs rotmat(aa) ---")
        log(f"Max abs err: {err_mat:.2e} (Threshold: 1e-3)")
        if err_mat > 1e-3:
            log("FAIL: rotmat mismatch")
            all_pass = False

        # Test (iii) Layout check
        col0 = R_orig[:, :, 0]
        col1 = R_orig[:, :, 1]
        expected_6d = torch.stack([
            col0[:, 0], col1[:, 0],
            col0[:, 1], col1[:, 1],
            col0[:, 2], col1[:, 2],
        ], dim=-1)
        err_layout = (r6d - expected_6d).abs().max().item()
        
        log(f"\n--- [Test iii] Layout [r00,r01,r10,r11,r20,r21] ---")
        log(f"Max err: {err_layout:.2e} (Threshold: 1e-5)")
        if err_layout > 1e-5:
            log("FAIL: layout mismatch")
            all_pass = False

        log(f"\n[Test iv] Ran on device='{device}' without errors.")

        if all_pass:
            log("\nPASS - All round-trip and consistency checks passed within strict thresholds.")
        else:
            log("\nFAIL - One or more strict thresholds were exceeded.")
            sys.exit(1)

    except Exception as e:
        import traceback
        log(f"FAIL - Exception occurred:\n{traceback.format_exc()}")
        sys.exit(1)


if __name__ == "__main__":
    main()
