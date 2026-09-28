import os
import sys
import argparse
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'lib'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

from models.smpl_hyperdiff import make_noisy_kp2d
from core.config import cfg

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--cfg', type=str, default='')
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu')
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

    try:
        B = 1000
        device = args.device
        
        # Clean GT
        kp2d = torch.randn(B, 17, 2, device=device).clamp(-1, 1)
        kp_conf = torch.ones(B, 17, device=device)
        
        # Make joint 5 always invisible
        kp_conf[:, 5] = 0
        
        kp2d_n, kp_conf_n, drop_n = make_noisy_kp2d(kp2d, kp_conf)
        
        # Check drop rate
        drop_rate = drop_n[:, 0].float().mean().item()
        log(f"Drop rate on visible joint: {drop_rate:.4f} (expected ~{cfg.DIFF.NOISE.drop_rate})")
        
        # Check invisible joint drop
        invis_dropped = drop_n[:, 5].all().item()
        log(f"Invisible joint (5) always dropped? {invis_dropped}")
        
        # Check error distribution
        # Only consider joints that were not dropped
        valid_mask = ~drop_n
        diff = (kp2d_n - kp2d)
        
        log(f"\n{'Joint':<10} | {'Mean Err (X)':<12} | {'Std Err (X)':<12} | {'Mean Err (Y)':<12} | {'Std Err (Y)':<12}")
        log("-" * 75)
        for j in range(17):
            j_mask = valid_mask[:, j]
            if j_mask.sum() == 0:
                log(f"{j:<10} | ALL DROPPED")
            else:
                d_x = diff[j_mask, j, 0]
                d_y = diff[j_mask, j, 1]
                log(f"{j:<10} | {d_x.mean().item():.4f}       | {d_x.std().item():.4f}       | {d_y.mean().item():.4f}       | {d_y.std().item():.4f}")

        if invis_dropped and 0.05 < drop_rate < 0.2:
            log("\nPASS - Noise module works properly.")
        else:
            log("\nFAIL - Noise module acting suspiciously.")
            sys.exit(1)
            
    except Exception as e:
        import traceback
        log(f"FAIL - Exception occurred:\n{traceback.format_exc()}")
        sys.exit(1)

if __name__ == "__main__":
    main()
