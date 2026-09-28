import os
import sys
import csv
import argparse
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'lib'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

from models.smpl_hyperdiff import SMPL_HyperDiff, axis_angle_to_rot6d

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--cfg', type=str, default='')
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--out_dir', type=str, default='logs/server_checks')
    return parser.parse_args()

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

    try:
        device = args.device
        model = SMPL_HyperDiff().to(device)
        model.train()
        
        optim = torch.optim.Adam(model.parameters(), lr=1e-3)
        B = 2
        gt_aa = torch.randn(B, 24, 3, device=device) * 0.1
        gt_6d = axis_angle_to_rot6d(gt_aa.view(-1, 3)).view(B, 24, 6)
        kp2d = torch.randn(B, 17, 2, device=device).clamp(-1, 1)
        kp_conf = torch.ones(B, 17, device=device)
        
        log(f"Starting overfit for 300 steps...")
        with open(csv_file, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['step', 'loss'])
            
            first_loss = None
            last_loss = None
            for step in range(300):
                optim.zero_grad()
                _, loss = model(gt_6d, kp2d, kp_conf, is_train=True)
                loss.backward()
                optim.step()
                
                l_val = loss.item()
                if step == 0: first_loss = l_val
                last_loss = l_val
                
                if step % 20 == 0:
                    writer.writerow([step, l_val])
                    
        log(f"First loss: {first_loss:.4f}, Last loss: {last_loss:.4f}")
        
        # DDIM sample
        model.eval()
        g1 = torch.Generator(device=device).manual_seed(42)
        g2 = torch.Generator(device=device).manual_seed(100)
        
        with torch.no_grad():
            sample1 = model.ddim_sample(kp2d, kp_conf, generator=g1)
            sample2 = model.ddim_sample(kp2d, kp_conf, generator=g2)
            
        diff = (sample1 - sample2).abs().max().item()
        err_vs_gt = (sample1 - gt_6d).abs().mean().item()
        
        log(f"Sample diff between 2 seeds (diversity check): {diff:.4e}")
        log(f"Sample error vs GT after overfit: {err_vs_gt:.4e}")
        
        if last_loss < first_loss * 0.5:
            log("PASS - Model can overfit successfully.")
        else:
            log("FAIL - Model failed to overfit.")
            sys.exit(1)
            
    except Exception as e:
        import traceback
        log(f"FAIL - Exception occurred:\n{traceback.format_exc()}")
        sys.exit(1)

if __name__ == "__main__":
    main()
