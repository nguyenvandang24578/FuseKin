import os
import sys
import json
import argparse
import torch
import numpy as np
from PIL import Image, ImageDraw

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'lib'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

from core.config import update_config, cfg
from utils.jotr_dataset import get_train_dataset, get_test_dataset

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--cfg', type=str, default='experiment/mesh_3dpw.yaml')
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--out_dir', type=str, default='logs/server_checks')
    return parser.parse_args()

def draw_overlay(img_tensor, kp2d, kp_conf, out_path):
    # img_tensor: (3, H, W) in [0, 1] usually, normalized. 
    # Actually JOTR transforms to tensor might have normalize. Let's un-normalize roughly.
    img_np = img_tensor.permute(1, 2, 0).cpu().numpy()
    img_np = (img_np * np.array([0.229, 0.224, 0.225]) + np.array([0.485, 0.456, 0.406]))
    img_np = np.clip(img_np * 255, 0, 255).astype(np.uint8)
    img_pil = Image.fromarray(img_np)
    draw = ImageDraw.Draw(img_pil)
    
    H, W = img_np.shape[:2]
    
    for i in range(17):
        x, y = kp2d[i]
        c = kp_conf[i]
        
        # Depending on input scale, might be heatmap [0, 64) or [-1, 1] or pixel [0, H]
        # JOTR dataset typically outputs heatmap coords [0, output_hm_shape]
        if x < 100 and y < 100:
            px = x / cfg.output_hm_shape[2] * W
            py = y / cfg.output_hm_shape[1] * H
        else:
            px, py = x, y
            
        color = 'green' if c > 0.5 else 'red'
        draw.ellipse((px-2, py-2, px+2, py+2), fill=color)
        draw.text((px+4, py+4), str(i), fill=color)
        
    img_pil.save(out_path)

def main():
    args = parse_args()
    out_img_dir = os.path.join(args.out_dir, 'overlay')
    os.makedirs(out_img_dir, exist_ok=True)
    log_file = os.path.join(args.out_dir, 'check_overlay.log')

    def log(msg):
        print(msg)
        with open(log_file, 'a', encoding='utf-8') as f:
            f.write(msg + '\n')

    with open(log_file, 'w', encoding='utf-8') as f:
        f.write("=== check_overlay ===\n")

    try:
        if os.path.exists(args.cfg):
            update_config(args.cfg)
            
        train_ds = get_train_dataset('3dpw-train', args)
        test_ds = get_test_dataset('3dpw', args)
        
        # Train (orig_joint_img)
        log("Processing Train dataset (16 samples)...")
        for i in range(16):
            inputs, targets, meta = train_ds[i]
            img = inputs['img']
            kp2d = targets['orig_joint_img'][:, :2]
            conf = meta['orig_joint_trunc'].squeeze(-1)
            draw_overlay(img, kp2d, conf, os.path.join(out_img_dir, f'train_{i:02d}.png'))
            
        # Test (inputs['joints'])
        log("Processing Test dataset (16 samples)...")
        for i in range(16):
            inputs, targets, meta = test_ds[i]
            img = inputs['img']
            kp2d = inputs['joints'][:, :2]
            
            # test inputs['joints'] might be pixel coords if det, or heatmap coords
            # conf might be inputs['joints_mask'] or joints[:, 2]
            if inputs['joints'].shape[1] > 2:
                conf = inputs['joints'][:, 2]
            else:
                conf = inputs.get('joints_mask', np.ones(17)).squeeze()
                
            draw_overlay(img, kp2d, conf, os.path.join(out_img_dir, f'test_{i:02d}.png'))
            
        log("PASS - Overlays generated successfully.")
    except Exception as e:
        import traceback
        log(f"FAIL - Exception occurred:\n{traceback.format_exc()}")
        sys.exit(1)

if __name__ == "__main__":
    main()
