import os
import sys
import argparse
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'lib'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

from core.config import update_config, cfg
from models.Multimodel import Pose2Mesh
from utils.jotr_dataset import get_train_dataset

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--cfg', type=str, default='experiment/mesh_3dpw.yaml')
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--out_dir', type=str, default='logs/server_checks')
    parser.add_argument('--real_batch', action='store_true', help='Use a real batch from 3DPW')
    return parser.parse_args()

def main():
    args = parse_args()
    os.makedirs(args.out_dir, exist_ok=True)
    log_file = os.path.join(args.out_dir, 'check_forward_shapes.log')

    def log(msg):
        print(msg)
        with open(log_file, 'a', encoding='utf-8') as f:
            f.write(msg + '\n')

    with open(log_file, 'w', encoding='utf-8') as f:
        f.write("=== check_forward_shapes ===\n")

    try:
        update_config(args.cfg)
        cfg.MODEL.REFINER = 'diffusion'
        model = Pose2Mesh(cfg).to(args.device)
        
        if args.real_batch:
            log("Using REAL batch from dataset...")
            ds = get_train_dataset('3dpw-train', args)
            inputs, targets, meta = ds[0]
            
            input_image = inputs['img'].unsqueeze(0).to(args.device)
            input_pose = inputs['joints'].unsqueeze(0).to(args.device)
            gt_pose_6d = torch.randn(1, 24, 6, device=args.device)
            kp2d = targets['orig_joint_img'][:17, :2].unsqueeze(0).to(args.device)
            kp_conf = meta['orig_joint_trunc'][:17].unsqueeze(0).squeeze(-1).to(args.device)
            pose_valid_mask = torch.ones(1, 24, device=args.device)
        else:
            log("Using FAKE random tensors...")
            B = 2
            input_image = torch.randn(B, 3, 256, 256, device=args.device)
            input_pose = torch.randn(B, 17, 3, device=args.device)
            
            gt_pose_6d = torch.randn(B, 24, 6, device=args.device)
            kp2d = torch.randn(B, 17, 2, device=args.device)
            kp_conf = torch.ones(B, 17, device=args.device)
            pose_valid_mask = torch.ones(B, 24, device=args.device)
        
        def run_check(mode_name, is_train):
            log(f"\nRunning {mode_name} mode...")
            out = model(
                input_image, input_pose, is_train=is_train, use_gt_3d=is_train,
                gt_pose_6d=gt_pose_6d, kp2d=kp2d, kp_conf=kp_conf,
                pose_valid_mask=pose_valid_mask
            )
            for k, v in out.items():
                if torch.is_tensor(v):
                    log(f"  {k}: shape {list(v.shape)}, has_nan: {torch.isnan(v).any().item()}, has_inf: {torch.isinf(v).any().item()}")
                else:
                    log(f"  {k}: {type(v)}")
                    
        model.train()
        run_check("TRAIN", True)
        
        model.eval()
        with torch.no_grad():
            run_check("EVAL", False)
            
        log("\nPASS - Shapes are correct, no NaNs or Infs.")
            
    except Exception as e:
        import traceback
        log(f"FAIL - Exception occurred:\n{traceback.format_exc()}")
        sys.exit(1)

if __name__ == "__main__":
    main()
