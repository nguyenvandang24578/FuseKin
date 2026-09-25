import sys
import os
import os.path as osp
import cv2
import torch
import numpy as np
import random
import argparse

sys.path.append(osp.abspath('.'))
sys.path.append(osp.abspath('./lib'))

from core.config import cfg
from utils.jotr_dataset import get_train_dataset, get_test_dataset
from utils.h36m_adapter import HUMAN36M_JOINTS

# Skeleton 17 khớp (H36M)
H36M_SKELETON = [
    (0, 1), (1, 2), (2, 3), # Chân phải
    (0, 4), (4, 5), (5, 6), # Chân trái
    (0, 7), (7, 8), (8, 9), (9, 10), # Cột sống và đầu
    (8, 14), (14, 15), (15, 16), # Tay phải
    (8, 11), (11, 12), (12, 13) # Tay trái
]

def draw_skeleton(img, points, mask, color, label, offset):
    # points: (17, 2), mask: (17, 1)
    
    # Vẽ xương (Edges)
    for edge in H36M_SKELETON:
        j1, j2 = edge
        if mask[j1] > 0.5 and mask[j2] > 0.5:
            pt1 = (int(points[j1, 0]), int(points[j1, 1]))
            pt2 = (int(points[j2, 0]), int(points[j2, 1]))
            cv2.line(img, pt1, pt2, color, 2)
            
    # Vẽ điểm (Nodes)
    for i in range(len(points)):
        pt = (int(points[i, 0]), int(points[i, 1]))
        if mask[i] > 0.5:
            cv2.circle(img, pt, 4, color, -1)
        else:
            # Điểm bị mask (bị cắt khỏi ảnh hoặc bị che) vẽ màu xám
            cv2.circle(img, pt, 4, (128, 128, 128), -1)
            
    cv2.putText(img, label, offset, cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)

def main():
    random.seed(42)
    np.random.seed(42)
    
    datasets_to_check = [
        ('3dpw-train', get_train_dataset),
        ('3dpw', get_test_dataset)
    ]
    
    for data_name, get_func in datasets_to_check:
        print(f"\n=====================================")
        print(f"Đang tải dataset: {data_name}")
        print(f"=====================================")
        # Khởi tạo dataset trực tiếp thông qua wrapper
        dataset = get_func(data_name, None)
        
        out_dir = f'debug_vis/{data_name.replace("-", "_")}'
        os.makedirs(out_dir, exist_ok=True)
        
        num_samples = min(16, len(dataset))
        indices = random.sample(range(len(dataset)), num_samples)
        
        print(f"Sẽ lưu {num_samples} ảnh vào thư mục {out_dir}/")
        
        for idx in indices:
            # Gọi trực tiếp __getitem__ y như DataLoader
            inputs, targets, meta = dataset[idx]
            
            # Lấy thông tin metadata
            raw_data = dataset.dataset.datalist[idx]
            ann_id = meta.get('aid', raw_data.get('ann_id', idx))
            img_path = meta.get('img_path', raw_data.get('img_path', 'unknown'))
            
            if isinstance(ann_id, torch.Tensor): ann_id = ann_id.item()
            elif isinstance(ann_id, (list, tuple, np.ndarray)): ann_id = ann_id[0]
            
            if isinstance(img_path, torch.Tensor): img_path = img_path.item()
            elif isinstance(img_path, (list, tuple, np.ndarray)): img_path = img_path[0]
            
            # ==========================================================
            # 1. DENORMALIZE ẢNH
            # Công thức gốc trong dataset.py (dòng 290): img = self.transform(img.astype(np.float32))/255.
            # => Nghịch đảo: img * 255.0
            # ==========================================================
            img_tensor = inputs['img']
            img_np = img_tensor.numpy().transpose(1, 2, 0)
            img_bgr = (img_np * 255).astype(np.uint8)
            img_bgr = cv2.cvtColor(img_bgr, cv2.COLOR_RGB2BGR)
            
            # ==========================================================
            # 2. DENORMALIZE INPUTS['JOINTS'] (Input 2D OpenPose/HRNet)
            # Công thức gốc trong jotr_dataset.py (dòng 84-85):
            # inputs['joints'][..., 0] = inputs['joints'][..., 0] / cfg.output_hm_shape[2] * 2 - 1
            # => Nghịch đảo về hệ Pixel (256x256): (x + 1) / 2 * input_img_shape
            # ==========================================================
            input_joints = inputs['joints'].numpy()
            input_mask = inputs['joints_mask'].numpy()
            
            input_joints_pixel = np.zeros_like(input_joints)
            input_joints_pixel[:, 0] = (input_joints[:, 0] + 1.0) / 2.0 * cfg.input_img_shape[1]
            input_joints_pixel[:, 1] = (input_joints[:, 1] + 1.0) / 2.0 * cfg.input_img_shape[0]
            
            # ==========================================================
            # 3. DENORMALIZE TARGETS['ORIG_JOINT_IMG'] (GT 2D SMPL)
            # Công thức gốc trong dataset.py (dòng 358-359):
            # h36m_coord_img[:, 0] = h36m_coord_img[:, 0] / cfg.input_img_shape[1] * cfg.output_hm_shape[2]
            # => Nghịch đảo về hệ Pixel (256x256): x / output_hm_shape * input_img_shape
            # ==========================================================
            has_gt = 'orig_joint_img' in targets
            gt_joints_pixel = None
            gt_mask = None
            if has_gt:
                gt_joints = targets['orig_joint_img'].numpy()
                gt_mask = meta['orig_joint_trunc'].numpy()
                gt_joints_pixel = np.zeros_like(gt_joints)
                gt_joints_pixel[:, 0] = gt_joints[:, 0] / cfg.output_hm_shape[2] * cfg.input_img_shape[1]
                gt_joints_pixel[:, 1] = gt_joints[:, 1] / cfg.output_hm_shape[1] * cfg.input_img_shape[0]
            
            # Tính số liệu hiển thị ra console
            valid_input = np.sum(input_mask > 0.5)
            min_x, max_x = np.min(input_joints_pixel[:, 0]), np.max(input_joints_pixel[:, 0])
            min_y, max_y = np.min(input_joints_pixel[:, 1]), np.max(input_joints_pixel[:, 1])
            
            print(f"--- Sample ann_id={ann_id} ---")
            print(f"  Valid Input Joints (mask=1): {valid_input}/17")
            print(f"  Input Coords Min/Max: X=[{min_x:.1f}, {max_x:.1f}], Y=[{min_y:.1f}, {max_y:.1f}]")
            
            if has_gt:
                valid_both = (input_mask > 0.5) & (gt_mask > 0.5)
                if np.any(valid_both):
                    error = np.linalg.norm(input_joints_pixel[:, :2] - gt_joints_pixel[:, :2], axis=1)
                    mean_error = np.mean(error[valid_both.flatten()])
                    print(f"  Mean Error (Input vs GT): {mean_error:.2f} pixels (trên ảnh {cfg.input_img_shape[0]}x{cfg.input_img_shape[1]})")
                else:
                    print(f"  Mean Error: N/A (không có điểm nào chung hợp lệ)")
            
            # Vẽ hình
            if has_gt:
                draw_skeleton(img_bgr, gt_joints_pixel, gt_mask, (0, 255, 0), "GT 2D (Green)", offset=(10, 20))
            draw_skeleton(img_bgr, input_joints_pixel, input_mask, (0, 0, 255), "Input 2D (Red)", offset=(10, 40))
            
            # Chèn text
            cv2.putText(img_bgr, f"ann_id: {ann_id}", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            cv2.putText(img_bgr, f"split: {data_name}", (10, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            short_path = str(img_path).split('/')[-1] if isinstance(img_path, str) else str(img_path)
            cv2.putText(img_bgr, f"file: {short_path}", (10, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            
            cv2.imwrite(f"{out_dir}/{ann_id}.jpg", img_bgr)

if __name__ == '__main__':
    main()
