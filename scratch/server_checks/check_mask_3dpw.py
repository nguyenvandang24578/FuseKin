"""check_mask_3dpw.py – Gather fit_param_valid mask statistics from 3DPW.

Claims to check
===============
This is a STATISTICS script. It reports which of the 24 SMPL joints have
mask=0 across the training set, and prints per-joint GT pose parameter
statistics (std, min, max in radians→degrees) on a subsample of ~500 items
to assess whether masked joints have real GT data.

Previous bugs
=============
- IndexError: iterated 30 joints but mask is (24*3) → index 24 OOB for
  SMPL original 24 joints.
- Scanned all 22735 samples including full image loading → ~1 hour.
- Used smpl.joints_name[:24] which is actually 30 SMPL joints (index 0..23
  maps to 'Pelvis'..'R_Hand' but smpl.joints_name has 30 entries; slice [:24]
  is correct but the old code used j_idx >= 24 somewhere).

Fix strategy
============
- Read the mask-setting code directly: PW3D.dataset.py line ~389-398 sets
  smpl_param_valid = ones((24,3)) then zeros out 8 joints by name:
  L/R_Ankle, L/R_Toe, L/R_Wrist, L/R_Hand.
  So the mask is deterministic per-sample (always same 8 joints = 0).
- Only load ~500 random samples (via random indices) for statistics.
- Convert axis-angle std to degrees for readability.
- Write mask_stats.json.
"""

import os
import sys
import json
import argparse
import random
import math
import numpy as np
from tqdm import tqdm

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'lib'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

from core.config import update_config, cfg


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--cfg', type=str, default='config/train_init_mesh.yaml')
    parser.add_argument('--device', type=str, default='cpu')
    parser.add_argument('--out_dir', type=str, default='logs/server_checks')
    parser.add_argument('--n_samples', type=int, default=500,
                        help='Number of random samples to inspect')
    parser.add_argument('--fast', action='store_true', default=True,
                        help='Limit samples (default: True)')
    return parser.parse_args()


# SMPL 24 original joint names (from utils/smpl.py)
SMPL_24_NAMES = (
    'Pelvis', 'L_Hip', 'R_Hip', 'Torso', 'L_Knee', 'R_Knee', 'Spine',
    'L_Ankle', 'R_Ankle', 'Chest', 'L_Toe', 'R_Toe', 'Neck', 'L_Thorax',
    'R_Thorax', 'Head', 'L_Shoulder', 'R_Shoulder', 'L_Elbow', 'R_Elbow',
    'L_Wrist', 'R_Wrist', 'L_Hand', 'R_Hand',
)

# The 8 joints hardcoded as mask=0 in PW3D dataset.py (line ~397)
MASK_ZERO_JOINTS = ('L_Ankle', 'R_Ankle', 'L_Toe', 'R_Toe',
                     'L_Wrist', 'R_Wrist', 'L_Hand', 'R_Hand')


def main():
    args = parse_args()
    os.makedirs(args.out_dir, exist_ok=True)
    log_file = os.path.join(args.out_dir, 'check_mask_3dpw.log')
    json_file = os.path.join(args.out_dir, 'mask_stats.json')

    def log(msg):
        print(msg)
        with open(log_file, 'a', encoding='utf-8') as f:
            f.write(msg + '\n')

    with open(log_file, 'w', encoding='utf-8') as f:
        f.write("=== check_mask_3dpw ===\n")

    try:
        if os.path.exists(args.cfg):
            update_config(args.cfg)

        # ---- Report what the dataset code does ----
        log("--- Source code analysis (PW3D dataset.py) ---")
        log("File: data_final/PW3D/dataset.py, lines ~388-399")
        log("  smpl_param_valid = np.ones((24, 3), dtype=np.float32)")
        log("  Unless FORCE_FULL_FIT_MASK_3DPW=True, sets mask=0 for:")
        for name in MASK_ZERO_JOINTS:
            idx = SMPL_24_NAMES.index(name)
            log(f"    joint {idx:2d} ({name})")
        log(f"  FORCE_FULL_FIT_MASK_3DPW = {cfg.DATASET.FORCE_FULL_FIT_MASK_3DPW}")
        log("")

        # ---- Load dataset and subsample ----
        from utils.jotr_dataset import get_train_dataset
        train_ds = get_train_dataset('3dpw-train', args)
        total = len(train_ds)
        n_samples = min(args.n_samples, total) if args.fast else total
        indices = sorted(random.sample(range(total), n_samples))
        log(f"Dataset size: {total}, sampling {n_samples} items for statistics.\n")

        # ---- Collect mask and pose stats ----
        mask_sum_24 = np.zeros(24, dtype=np.float64)
        poses_list = []  # each (72,) or (24,3)
        count = 0

        for i in tqdm(indices, desc="Sampling"):
            inputs, targets, meta = train_ds[i]

            # fit_param_valid is (72,) numpy from dataset -> Human36M17Dataset
            # passes it through unchanged (it's not a joint key or mask key).
            mask = np.asarray(meta['fit_param_valid'])  # (72,) for orig 24 joints
            if mask.shape[0] == 72:
                mask_24 = mask.reshape(24, 3)[:, 0]  # take first of each triplet
            else:
                # Unexpected shape
                log(f"  WARNING: fit_param_valid shape={mask.shape}, expected (72,)")
                mask_24 = np.ones(24)

            mask_sum_24 += mask_24

            pose = np.asarray(targets['pose_param'])  # (72,)
            poses_list.append(pose)
            count += 1

        # ---- Compute statistics ----
        poses_arr = np.stack(poses_list)  # (n_samples, 72)
        poses_24x3 = poses_arr.reshape(-1, 24, 3)  # (n_samples, 24, 3)

        stats = {
            'total_samples_in_dataset': total,
            'samples_inspected': count,
            'force_full_mask': bool(cfg.DATASET.FORCE_FULL_FIT_MASK_3DPW),
            'joints': {},
        }

        log(f"{'Joint':>3} {'Name':>12} | {'mask=1':>8}/{count:<6} | "
            f"{'aa_std(deg)':>11} | {'aa_min(deg)':>11} | {'aa_max(deg)':>11} | "
            f"{'is_masked':>9}")
        log("-" * 100)

        for j in range(24):
            name = SMPL_24_NAMES[j]
            valid_count = mask_sum_24[j]
            ratio = valid_count / count if count > 0 else 0

            # Axis-angle for this joint: (n_samples, 3)
            aa_j = poses_24x3[:, j, :]  # (n_samples, 3)
            aa_norms = np.linalg.norm(aa_j, axis=-1)  # angle magnitudes

            aa_std_deg = float(np.degrees(aa_norms.std()))
            aa_min_deg = float(np.degrees(aa_norms.min()))
            aa_max_deg = float(np.degrees(aa_norms.max()))

            is_masked = name in MASK_ZERO_JOINTS

            log(f"{j:>3} {name:>12} | {valid_count:>8.0f}/{count:<6} | "
                f"{aa_std_deg:>11.2f} | {aa_min_deg:>11.2f} | {aa_max_deg:>11.2f} | "
                f"{'YES' if is_masked else 'no':>9}")

            stats['joints'][name] = {
                'index': j,
                'valid_count': float(valid_count),
                'total': count,
                'ratio': ratio,
                'is_hardcoded_zero': is_masked,
                'aa_angle_std_deg': aa_std_deg,
                'aa_angle_min_deg': aa_min_deg,
                'aa_angle_max_deg': aa_max_deg,
            }

        # ---- Summary ----
        masked_joints = [n for n in SMPL_24_NAMES
                         if stats['joints'][n]['is_hardcoded_zero']]
        log(f"\nSummary: {len(masked_joints)} joints have mask=0: {masked_joints}")

        # Check if any masked joint actually has significant GT motion
        for name in masked_joints:
            s = stats['joints'][name]
            if s['aa_angle_std_deg'] > 5.0:
                log(f"  NOTE: {name} has std={s['aa_angle_std_deg']:.1f}° despite mask=0. "
                    f"GT data IS present. Consider enabling FORCE_FULL_FIT_MASK_3DPW.")

        with open(json_file, 'w') as f:
            json.dump(stats, f, indent=2)
        log(f"\nWrote mask_stats.json to {json_file}")
        log("PASS - Mask statistics collected successfully.")

    except Exception as e:
        import traceback
        log(f"FAIL - Exception occurred:\n{traceback.format_exc()}")
        sys.exit(1)


if __name__ == "__main__":
    main()
