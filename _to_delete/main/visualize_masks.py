import os, sys
sys.path.append('./lib')
sys.path.append('./')
import argparse
import numpy as np
import torch
import cv2
import matplotlib.pyplot as plt

from core.config import cfg, update_config
from core.base import get_dataloader
from utils.transforms import cam2pixel

def denormalize_image(img_tensor):
    # img_tensor shape: (3, H, W)
    mean = np.array([0.485, 0.456, 0.406]).reshape(3, 1, 1)
    std = np.array([0.229, 0.224, 0.225]).reshape(3, 1, 1)
    img = img_tensor.cpu().numpy()
    img = img * std + mean
    img = np.clip(img, 0, 1)
    img = (img * 255).astype(np.uint8)
    return img.transpose(1, 2, 0) # (H, W, 3)

def main():
    parser = argparse.ArgumentParser(description='Visualize Masks')
    parser.add_argument('--cfg', type=str, default='config/train_mesh_3dpw.yml', help='experiment configure file name')
    parser.add_argument('--debug', action='store_true', default=True, help='reduce dataset items')
    args, _ = parser.parse_known_args()
    
    update_config(args.cfg)
    
    # Force batch size to 1 for easier visualization
    cfg.TRAIN.batch_size = 1
    
    dataset_names = cfg.DATASET.train_list
    _, batch_generator = get_dataloader(args, dataset_names, is_train=True)
    
    num_saved = 0
    max_save = 5 # Số lượng ảnh có mask bạn muốn lưu
    
    for i, (inputs, targets, meta) in enumerate(batch_generator):
        print(f"Checking batch {i}...")
        input_image = inputs['img'][0] # (3, H, W)
        
        # We need 2D coordinates for orig_joint to plot on image. 
        # Usually dataset provides them in 'orig_joint_img' or we can project them.
        # Let's check targets and meta
        if 'orig_joint_img' in targets:
            orig_joint_img = targets['orig_joint_img'][0].numpy()
        else:
            print("No orig_joint_img found, using center as fallback or let's try projecting")
            # This is just a fallback, usually orig_joint_img is present
            orig_joint_img = np.zeros((24, 3))
            
        orig_joint_valid = meta['orig_joint_valid'][0].numpy() # shape usually (J, 1) or (J)
        fit_joint_trunc = meta['fit_joint_trunc'][0].numpy()
        
        # Lọc: Chỉ lấy những ảnh CÓ mask (nghĩa là có ít nhất 1 điểm bị lỗi / bị che / bị cắt)
        # Tức là tồn tại ít nhất 1 giá trị <= 0 trong orig_joint_valid hoặc fit_joint_trunc
        if np.all(orig_joint_valid > 0) and np.all(fit_joint_trunc > 0):
            continue
            
        print(f"Found image with masks! Saving as {num_saved + 1}/{max_save}")
        
        img_np = denormalize_image(input_image)
        
        # Scale coordinates from heatmap size back to input image size
        # Assuming orig_joint_img is in output_hm_shape (e.g. 64x64) and img_np is in input_img_shape (e.g. 256x256)
        scale_x = cfg.input_img_shape[1] / cfg.output_hm_shape[2] if len(cfg.output_hm_shape) == 3 else cfg.input_img_shape[1] / cfg.output_hm_shape[1]
        scale_y = cfg.input_img_shape[0] / cfg.output_hm_shape[1] if len(cfg.output_hm_shape) == 3 else cfg.input_img_shape[0] / cfg.output_hm_shape[0]

        fig, axes = plt.subplots(1, 2, figsize=(10, 5))
        
        # Plot orig_joint_valid
        axes[0].imshow(img_np)
        axes[0].set_title("orig_joint_valid\nGreen=Valid, Red=Invalid")
        for j in range(orig_joint_img.shape[0]):
            x, y = orig_joint_img[j, 0] * scale_x, orig_joint_img[j, 1] * scale_y
            if x > 0 and y > 0 and x < img_np.shape[1] and y < img_np.shape[0]:
                is_valid = orig_joint_valid[j]
                color = 'green' if np.all(is_valid) else 'red'
                axes[0].scatter(x, y, c=color, s=20)
                
        # Plot fit_joint_trunc
        axes[1].imshow(img_np)
        axes[1].set_title("fit_joint_trunc\nGreen=Not Truncated, Red=Truncated")
        # For fit_joint_trunc we might need fit_joint_img, but we can just use orig_joint_img for demonstration
        for j in range(orig_joint_img.shape[0]):
            x, y = orig_joint_img[j, 0] * scale_x, orig_joint_img[j, 1] * scale_y
            if x > 0 and y > 0 and x < img_np.shape[1] and y < img_np.shape[0]:
                is_trunc = fit_joint_trunc[j]
                # trunc == 1 usually means truncated, or maybe 0 means valid. Let's assume 1 = truncated (bad), 0 = ok
                # Or wait, usually mask=1 means valid. Let's check color based on mask > 0.
                color = 'green' if np.all(is_trunc > 0) else 'red' 
                axes[1].scatter(x, y, c=color, s=20)
                
        out_name = f'visualize_masks_output_{num_saved}.png'
        plt.savefig(out_name)
        plt.close(fig)
        print(f"Saved visualization to {out_name}")
        
        num_saved += 1
        if num_saved >= max_save:
            break

if __name__ == '__main__':
    main()
