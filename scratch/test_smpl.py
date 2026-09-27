import sys
import torch
import numpy as np

# Adjust paths to match FuseKin working directory
sys.path.append('./lib')

from utils.smpl import SMPL as SMPLOld
from models.smpl_mps import SMPL as SMPLNew

def main():
    print("Loading SMPL models...")
    # Khởi tạo model cũ (smplpytorch)
    human_model = SMPLOld()
    old_smpl = human_model.layer['neutral']
    joint_regressor_old = torch.from_numpy(human_model.joint_regressor).float()

    # Khởi tạo model mới (smplx)
    new_smpl = SMPLNew('data_final/base_data', create_transl=False)

    batch_size = 2
    # Fake random inputs: Pose (72D), Shape (10D), Trans (3D)
    smpl_pose = torch.randn(batch_size, 72)
    smpl_shape = torch.randn(batch_size, 10)
    smpl_trans = torch.randn(batch_size, 3)

    print("Running forwards...")
    # 1. Chạy bản CŨ
    with torch.no_grad():
        mesh_old, _ = old_smpl(smpl_pose, smpl_shape, smpl_trans)
        joint_cam_old = torch.bmm(
            joint_regressor_old[None, :, :].repeat(batch_size, 1, 1),
            mesh_old
        )
        root_cam_old = joint_cam_old[:, 0, None, :]
        mesh_old_root_relative = mesh_old - root_cam_old

    # 2. Chạy bản MỚI
    with torch.no_grad():
        out_new = new_smpl(
            betas=smpl_shape,
            body_pose=smpl_pose[:, 3:],
            global_orient=smpl_pose[:, :3],
            transl=smpl_trans,
            pose2rot=True
        )
        mesh_new = out_new.vertices
        joint_cam_new = out_new.joints
        # Root joint index cho 'OP MidHip' là 8
        root_cam_new = joint_cam_new[:, 8, None, :]
        mesh_new_root_relative = mesh_new - root_cam_new

    # 3. So sánh
    print("\n--- KẾT QUẢ SO SÁNH ---")
    
    mesh_diff = torch.abs(mesh_old - mesh_new)
    print(f"1. Độ lệch Lưới MESH nguyên bản (Absolute):")
    print(f"   - Lệch tối đa (Max diff): {mesh_diff.max().item():.6f}")
    print(f"   - Lệch trung bình (Mean diff): {mesh_diff.mean().item():.6f}")
    
    mesh_rel_diff = torch.abs(mesh_old_root_relative - mesh_new_root_relative)
    print(f"\n2. Độ lệch Lưới MESH sau khi trừ Root (Root-Relative):")
    print(f"   - Lệch tối đa (Max diff): {mesh_rel_diff.max().item():.6f}")
    print(f"   - Lệch trung bình (Mean diff): {mesh_rel_diff.mean().item():.6f}")

    root_diff = torch.abs(root_cam_old - root_cam_new)
    print(f"\n3. Độ lệch của Root Joint (Pelvis cũ vs OP MidHip mới):")
    print(f"   - Lệch tối đa (Max diff): {root_diff.max().item():.6f}")

    if mesh_rel_diff.max().item() > 1e-4:
        print("\n=> KẾT LUẬN: Lưới Mesh sinh ra từ 2 model có sự chênh lệch ĐÁNG KỂ!")
        print("   Sự khác biệt này khiến ma trận J_regressor_h36m trong quá trình tính loss nội suy ra các khớp bị sai lệch.")
        print("   Bạn nên dùng lại utils.smpl cho module Pose2Mesh để tương thích 100% với J_regressor_h36m.")
    else:
        print("\n=> KẾT LUẬN: Lưới Mesh sinh ra từ 2 model hoàn toàn giống nhau (chênh lệch cực nhỏ do sai số float).")
        print("   Nguyên nhân kết quả tệ hơn nằm ở một nơi khác trong pipeline.")

    print("\n--- KIỂM TRA TRỰC TIẾP MA TRẬN REGRESSOR (PELVIS) ---")
    row_old = joint_regressor_old[0] # (6890,)
    # smplx lưu regressor dưới dạng tensor (nếu sparse thì cần convert về dense)
    j_reg_new = new_smpl.J_regressor
    if j_reg_new.is_sparse:
        j_reg_new = j_reg_new.to_dense()
    row_new = j_reg_new[0] # (6890,)
    
    diff_reg = torch.abs(row_old - row_new)
    print(f"Max Diff giữa 2 hàng Pelvis: {diff_reg.max().item():.6f}")
    print(f"Sum Diff giữa 2 hàng Pelvis: {diff_reg.sum().item():.6f}")
    
    non_zero_old = (row_old != 0).sum().item()
    non_zero_new = (row_new != 0).sum().item()
    print(f"Số lượng Non-Zero ở bản CŨ (smplpytorch): {non_zero_old}")
    print(f"Số lượng Non-Zero ở bản MỚI (smplx): {non_zero_new}")

if __name__ == '__main__':
    main()
