import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from graph import Graph
# 1. Khởi tạo đối tượng Graph từ code bạn đã sửa
# Đảm bảo bạn đã đổi num_node = 24 và cập nhật inward_ori_index theo SMPL
g = Graph(hyper_joints=3, labeling_mode='virtual_ensemble')

# 2. Lấy ma trận A (Shape thường là 8 x 27 x 27)
matrix_A = g.A 

# 3. Vẽ ma trận của nhánh đầu tiên (Subset 0)
def plot_adjacency_matrix(A, num_joints=24):
    total_v = A.shape[1] # Thường là 27 (24 thực + 3 ảo)
    
    plt.figure(figsize=(10, 8))
    
    # Sử dụng heatmap để vẽ
    sns.heatmap(A[0], cmap="YlGnBu", cbar=True, square=True)
    
    # Vẽ đường kẻ phân tách vùng khớp thực và khớp ảo để dễ nhìn
    plt.axhline(y=num_joints, color='red', linestyle='--', label='Virtual Boundary')
    plt.axvline(x=num_joints, color='red', linestyle='--')
    
    plt.title(f"Ma trận kề Hyper-GCN (Nhánh 1)\nKhớp thực: {num_joints} | Khớp ảo: {total_v - num_joints}")
    plt.xlabel("Joint Index")
    plt.ylabel("Joint Index")
    plt.show()

# Gọi hàm vẽ
plot_adjacency_matrix(matrix_A, num_joints=24)