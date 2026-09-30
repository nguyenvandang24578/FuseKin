import torch
import matplotlib.pyplot as plt
import numpy as np
import sys
import os

# Đảm bảo import được lib từ thư mục gốc
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from lib.models.smpl_hyperdiff import SMPL_HyperDiff, axis_angle_to_rot6d

def main():
    print("Khởi tạo mô hình Diffusion...")
    model = SMPL_HyperDiff(num_timesteps=1000)
    
    # 1. Tạo x0 giả lập: 2048 mẫu để histogram mượt
    B = 2048
    print(f"Tạo {B} mẫu x0 giả lập (6D pose)...")
    axis_angle = torch.randn(B, 24, 3) * 0.5 
    x0 = axis_angle_to_rot6d(axis_angle)
    
    # 2. Chọn các mốc thời gian để quan sát
    timesteps = [0, 200, 500, 800, 999]
    fig, axes = plt.subplots(1, 5, figsize=(20, 4))
    
    for i, t_val in enumerate(timesteps):
        t = torch.full((B,), t_val, dtype=torch.long)
        
        # Thêm nhiễu
        x_t = model.q_sample(x0, t)
        vals = x_t.flatten().numpy()
        
        # Vẽ Histogram (cột xanh)
        axes[i].hist(vals, bins=60, density=True, alpha=0.6, color='dodgerblue')
        
        # Vẽ đường phân phối chuẩn N(0,1) lý thuyết (đường cong đen)
        x_axis = np.linspace(-4, 4, 100)
        p = np.exp(-0.5 * x_axis**2) / np.sqrt(2 * np.pi)
        axes[i].plot(x_axis, p, 'k', linewidth=2, label='$\mathcal{N}(0,1)$')
        
        axes[i].set_title(f't = {t_val}')
        axes[i].set_xlim(-4, 4)
        if i == 0:
            axes[i].legend()

    plt.tight_layout()
    out_path = os.path.join(os.path.dirname(__file__), 'xt_dist_visual.png')
    plt.savefig(out_path)
    print(f"Đã lưu biểu đồ tại: {out_path}")

if __name__ == '__main__':
    main()
