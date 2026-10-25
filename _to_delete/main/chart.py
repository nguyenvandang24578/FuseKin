import matplotlib.pyplot as plt
import numpy as np

# 1. Cài đặt font chữ chuẩn học thuật (Times New Roman) giống ảnh của bạn
plt.rcParams.update({
    'font.size': 14,
    'font.family': 'serif',
    'font.serif': ['Times New Roman']
})

# 2. DỮ LIỆU CỦA BẠN (Cái ruột xịn)
categories = ['Torso', 'Head/Neck', 'Arms', 'Legs']
baseline_errors = [15.7, 60.8, 79.3, 61.8] # ARTS
ours_errors     = [15.1, 60.1, 78.5, 60.2] # TKR-HMR

x = np.arange(len(categories))
width = 0.35

# 3. Khởi tạo biểu đồ
fig, ax = plt.subplots(figsize=(8, 5))

# Vẽ cột
rects1 = ax.bar(x - width/2, baseline_errors, width, 
                label='ARTS (Baseline)', color='#4C72B0', edgecolor='black', linewidth=1)
rects2 = ax.bar(x + width/2, ours_errors, width, 
                label='TKR-HMR (Ours)', color='#DD8452', edgecolor='black', linewidth=1)

# Ghi số trực tiếp lên đầu mỗi cột
ax.bar_label(rects1, padding=3, fmt='%.1f', fontsize=12)
ax.bar_label(rects2, padding=3, fmt='%.1f', fontsize=12)

# Thiết lập trục và nhãn
ax.set_ylabel('MPJPE (mm) $\downarrow$', fontsize=15, labelpad=10)
ax.set_xticks(x)
ax.set_xticklabels(categories)
ax.legend(loc='upper left', framealpha=0.9, edgecolor='black')

# Nới trục Y để số không bị đè vào viền trên
ax.set_ylim(0, 95)

# ---------------------------------------------------------
# 4. ÁP DỤNG "CÁI KHUNG" SẠCH SẼ NHƯ ẢNH BẠN GỬI YÊU CẦU
# ---------------------------------------------------------
# Xóa các vạch chia độ lởm chởm ở trục X và Y
ax.tick_params(axis='both', which='both', length=0) 

# Tắt hoàn toàn lưới (grid) bên trong để "ruột" trống trơn sạch sẽ
ax.grid(False)

# Làm dày 4 cạnh viền (khung hộp) lên cho cứng cáp
for spine in ax.spines.values():
    spine.set_linewidth(1.2)
    spine.set_color('black')

# 5. Lưu ảnh
plt.tight_layout()
plt.savefig('per_joint_error_clean.pdf', dpi=300, bbox_inches='tight')
plt.savefig('per_joint_error_clean.png', dpi=300, bbox_inches='tight')

print("Đã xong! Lưu file per_joint_error_clean thành công.")