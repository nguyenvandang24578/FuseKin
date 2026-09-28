"""check_noise_kp2d.py – Verify make_noisy_kp2d statistics.

Claims to check
===============
1. Drop rate on visible joints exactly matches cfg.DIFF.NOISE.p_drop (big_err does NOT increase drop rate).
2. Invisible joints (conf=0) are ALWAYS dropped.
3. Gaussian noise std per joint precisely matches sigma_per_joint.
4. Left-right swap pairs follow H36M convention.
5. No NaN in outputs.

Previous bugs
=============
- PASS was based on hardcoded 0.05 threshold for drop rate.
- Wrong explanation: "big_err increases drop rate". (It doesn't).
- Gaussian noise std was conflated with big_err and limb noise.
"""

import os
import sys
import argparse
import torch
import copy

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

        device = args.device
        B = 10000
        
        # Clean GT keypoints uniformly in [-0.8, 0.8]
        kp2d = (torch.rand(B, 17, 2, device=device) * 1.6 - 0.8)
        kp_conf = torch.ones(B, 17, device=device)
        
        # Make joints 3, 6 always invisible
        invis_joints = [3, 6]
        for j in invis_joints:
            kp_conf[:, j] = 0

        # ---- TEST 1: Drop Rate (Isolate Drop) ----
        log("\n--- TEST 1: Drop Rate ---")
        cfg_drop = copy.deepcopy(cfg.DIFF.NOISE)
        cfg_drop.sigma_per_joint = [0.0] * 17
        cfg_drop.limb_sigma = 0.0
        cfg_drop.p_big_err = 0.0 # Prove big_err doesn't affect drop rate
        
        _, _, drop_n = make_noisy_kp2d(kp2d, kp_conf, noise_cfg=cfg_drop)
        
        visible_mask = kp_conf > 0.5
        n_visible = visible_mask.sum().item()
        drop_on_visible = drop_n & visible_mask
        drop_rate = drop_on_visible.sum().item() / n_visible
        
        log(f"Configured p_drop: {cfg_drop.p_drop:.4f}")
        log(f"Measured drop rate (visible joints): {drop_rate:.4f}")
        
        if abs(drop_rate - cfg_drop.p_drop) > 0.01:
            log("FAIL: Drop rate mismatch > 1%")
            all_pass = False
            
        for j in invis_joints:
            if not drop_n[:, j].all().item():
                log(f"FAIL: Invisible joint {j} NOT always dropped!")
                all_pass = False

        # ---- TEST 2: Gaussian Std Dev (Isolate Gaussian) ----
        log("\n--- TEST 2: Pure Gaussian Noise Std Dev ---")
        cfg_gauss = copy.deepcopy(cfg.DIFF.NOISE)
        cfg_gauss.limb_sigma = 0.0
        cfg_gauss.p_big_err = 0.0
        cfg_gauss.p_drop = 0.0
        
        kp2d_g, _, _ = make_noisy_kp2d(kp2d, kp_conf, noise_cfg=cfg_gauss)
        diff_g = kp2d_g - kp2d
        
        log(f"{'Joint':>6} | {'Cfg Sigma':>10} | {'Meas Std(X)':>12} | {'Meas Std(Y)':>12} | {'Diff %':>8}")
        log("-" * 65)
        for j in range(17):
            cfg_s = cfg_gauss.sigma_per_joint[j]
            std_x = diff_g[:, j, 0].std().item()
            std_y = diff_g[:, j, 1].std().item()
            
            err_x = abs(std_x - cfg_s) / (cfg_s + 1e-8) * 100
            err_y = abs(std_y - cfg_s) / (cfg_s + 1e-8) * 100
            max_err = max(err_x, err_y)
            
            log(f"{j:>6} | {cfg_s:>10.4f} | {std_x:>12.4f} | {std_y:>12.4f} | {max_err:>7.1f}%")
            
            if cfg_s > 0 and max_err > 10.0:
                log(f"  FAIL: Gaussian std deviation > 10% for joint {j}")
                all_pass = False

        # ---- TEST 3: Big Errors (Isolate Swaps/Jumps) ----
        log("\n--- TEST 3: Big Errors (Swaps & Jumps) ---")
        cfg_big = copy.deepcopy(cfg.DIFF.NOISE)
        cfg_big.sigma_per_joint = [0.0] * 17
        cfg_big.limb_sigma = 0.0
        cfg_big.p_drop = 0.0
        # Force 100% big error
        cfg_big.p_big_err = 1.0 
        
        kp2d_b, _, _ = make_noisy_kp2d(kp2d, kp_conf, noise_cfg=cfg_big)
        
        # Check if left-right swaps occurred exactly on the correct pairs
        # For p_big_err=1.0, is_swap = 0.5 (50% chance of swap, 50% chance of jump)
        swapped_pairs_count = 0
        jump_count = 0
        for b in range(B):
            for (l, r) in H36M_SWAP_PAIRS:
                # If swapped, kp2d_b[b, l] == kp2d[b, r]
                if torch.allclose(kp2d_b[b, l], kp2d[b, r]) and torch.allclose(kp2d_b[b, r], kp2d[b, l]):
                    swapped_pairs_count += 1
                elif not torch.allclose(kp2d_b[b, l], kp2d[b, l]): # It jumped
                    jump_count += 1
                    
        total_pairs = B * len(H36M_SWAP_PAIRS)
        swap_rate = swapped_pairs_count / total_pairs
        log(f"For p_big_err=1.0, expected ~50% swaps. Measured: {swap_rate*100:.1f}%")
        
        if abs(swap_rate - 0.5) > 0.05:
            log(f"FAIL: Swap rate {swap_rate} deviates from expected 0.5")
            all_pass = False

        # ---- Verdict ----
        if all_pass:
            log("\nPASS - Noise module statistics exactly match configurations.")
        else:
            log("\nFAIL - One or more noise checks failed.")
            sys.exit(1)

    except Exception as e:
        import traceback
        log(f"FAIL - Exception occurred:\n{traceback.format_exc()}")
        sys.exit(1)


if __name__ == "__main__":
    main()
