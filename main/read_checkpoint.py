import torch
import os

def extract_and_save_F_block(checkpoint_A_path, output_F_path):
    print(f"Đọc checkpoint gốc: {checkpoint_A_path}")
    checkpoint = torch.load(checkpoint_A_path, map_location='cpu')
    state_dict_A = checkpoint['model_state_dict']
    
    # Tiền tố cần loại bỏ
    prefix_F = 'pose_mesh_coevo.residual.'
    state_dict_F = {}
    
    for key, value in state_dict_A.items():
        clean_key = key.replace('module.', '')
        
        if clean_key.startswith(prefix_F):
            # SỬA DÒNG NÀY: Thay vì giữ nguyên clean_key, ta xóa bỏ prefix_F đi
            core_key = clean_key.replace(prefix_F, '')
            state_dict_F[core_key] = value
            
    print(f"Đã trích xuất {len(state_dict_F)} tham số.")
    
    # Tạo thư mục lưu trữ nếu chưa tồn tại
    os.makedirs(os.path.dirname(output_F_path), exist_ok=True)
    
    # LƯU THÀNH FILE MỚI
    torch.save(state_dict_F, output_F_path)
    print(f"Đã lưu checkpoint khối F thành công tại: {output_F_path}")

# --- Thực thi ---
extract_and_save_F_block(
    checkpoint_A_path='./experiment/ARTS_Demo_Model/checkpoint/best.pth.tar', 
    output_F_path='./experiment/ARTS_Demo_Model/checkpoint/F_block_only.pth'
)