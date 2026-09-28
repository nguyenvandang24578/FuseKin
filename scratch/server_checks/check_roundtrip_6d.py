import os
import sys
import argparse
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'lib'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

from models.smpl_hyperdiff import axis_angle_to_rot6d, axis_angle_to_rotmat
from utils.transforms import rot6d_to_axis_angle
from geometry import rot6d_to_rotmat, rodrigues

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--cfg', type=str, default='')
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu')
    parser.add_argument('--out_dir', type=str, default='logs/server_checks')
    return parser.parse_args()

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

    try:
        N = 2000
        device = args.device
        aa = torch.randn(N, 3, device=device)
        
        # i) aa -> 6d -> aa
        r6d = axis_angle_to_rot6d(aa)
        aa_back = rot6d_to_axis_angle(r6d)
        err_aa = (aa - aa_back).norm(dim=-1).max().item()
        
        # ii) rot6d_to_rotmat(6d) == rodrigues(aa)
        r_from_6d = rot6d_to_rotmat(r6d).reshape(N, 3, 3)
        r_from_aa = rodrigues(aa).reshape(N, 3, 3)
        err_mat = (r_from_6d - r_from_aa).abs().max().item()
        
        # iii) layout check
        # r6d should be [r00, r01, r10, r11, r20, r21]
        col0 = r_from_aa[:, :, 0] # [r00, r10, r20]
        col1 = r_from_aa[:, :, 1] # [r01, r11, r21]
        expected_6d = torch.stack([col0[:,0], col1[:,0], col0[:,1], col1[:,1], col0[:,2], col1[:,2]], dim=-1)
        err_layout = (r6d - expected_6d).abs().max().item()
        
        log(f"Error aa roundtrip: {err_aa:.2e}")
        log(f"Error matrix:       {err_mat:.2e}")
        log(f"Error layout:       {err_layout:.2e}")
        
        if err_aa < 1e-3 and err_mat < 1e-3 and err_layout < 1e-5:
            log("PASS - Roundtrip and matrix consistency are perfect.")
        else:
            log(f"FAIL - Errors too large! err_aa={err_aa}, err_mat={err_mat}, err_layout={err_layout}")
            sys.exit(1)
            
    except Exception as e:
        import traceback
        log(f"FAIL - Exception occurred:\n{traceback.format_exc()}")
        sys.exit(1)

if __name__ == "__main__":
    main()
