"""
So sánh đầy đủ giữa 2 wrapper SMPL: utils.smpl (smplpytorch) vs models.smpl_mps (smplx).

Sửa lại bug của lần test trước: KHÔNG tự tính lại joint bằng cách nhân
J_regressor với mesh đã posed (đó chỉ là xấp xỉ). Thay vào đó lấy đúng
joint "chuẩn" mà mỗi model tự trả ra qua forward kinematics.

Chạy 2 kịch bản:
  A. Pose ngẫu nhiên Gaussian (torch.randn) - biên độ lớn, phi thực tế
  B. Pose thực tế - biên độ nhỏ, giống pose người thật

In ra đầy đủ số liệu để xác định chính xác nguồn gốc sai lệch.
"""
import sys
import torch
import numpy as np

sys.path.append('./lib')

from utils.smpl import SMPL as SMPLOld
from models.smpl_mps import SMPL as SMPLNew


def compare_regressors(human_model, new_smpl):
    """So sánh trực tiếp J_regressor của 2 bên (chỉ 24 khớp gốc)."""
    print("=" * 70)
    print("BƯỚC 1: SO SÁNH MA TRẬN J_REGRESSOR (24 khớp gốc)")
    print("=" * 70)

    reg_old = torch.from_numpy(human_model.joint_regressor).float()[:24]  # (24, 6890)

    # J_regressor nội bộ của smplx nằm trong buffer 'J_regressor'
    reg_new = new_smpl.J_regressor.float()  # (24, 6890) thường

    if reg_old.shape != reg_new.shape:
        print(f"[CẢNH BÁO] Shape khác nhau: old={reg_old.shape}, new={reg_new.shape}")
    else:
        diff = (reg_old - reg_new).abs()
        print(f"Max diff toàn ma trận 24 khớp : {diff.max().item():.8f}")
        print(f"Sum diff toàn ma trận 24 khớp : {diff.sum().item():.8f}")

    # riêng Pelvis (row 0)
    row_old = reg_old[0]
    row_new = reg_new[0]
    print(f"\n-- Riêng row Pelvis (index 0) --")
    print(f"Max diff        : {(row_old - row_new).abs().max().item():.8f}")
    print(f"Non-zero (old)  : {(row_old != 0).sum().item()}")
    print(f"Non-zero (new)  : {(row_new != 0).sum().item()}")
    print()


def run_case(name, old_smpl, new_smpl, joint_regressor_old, smpl_pose, smpl_shape, smpl_trans):
    print("=" * 70)
    print(f"KỊCH BẢN: {name}")
    print(f"Pose abs-mean = {smpl_pose.abs().mean().item():.4f} rad "
          f"(~{np.degrees(smpl_pose.abs().mean().item()):.1f} deg trung bình mỗi chiều)")
    print("=" * 70)

    with torch.no_grad():
        # ---- Bản CŨ: dùng joints CÓ SẴN từ forward, KHÔNG tự tính lại ----
        mesh_old, joint_native_old = old_smpl(smpl_pose, smpl_shape, smpl_trans)
        root_native_old = joint_native_old[:, 0, None, :]
        mesh_old_rootrel = mesh_old - root_native_old

        # ---- (để đối chiếu) cách tính SAI ở lần test trước: regressor x mesh posed ----
        joint_cam_old_approx = torch.bmm(
            joint_regressor_old[None, :, :].repeat(smpl_pose.shape[0], 1, 1),
            mesh_old
        )
        root_old_approx = joint_cam_old_approx[:, 0, None, :]

        # ---- Bản MỚI: smplx ----
        out_new = new_smpl(
            betas=smpl_shape,
            body_pose=smpl_pose[:, 3:],
            global_orient=smpl_pose[:, :3],
            transl=smpl_trans,
            pose2rot=True,
        )
        mesh_new = out_new.vertices
        joint_cam_new = out_new.joints
        root_cam_new = joint_cam_new[:, 8, None, :]  # OP MidHip
        mesh_new_rootrel = mesh_new - root_cam_new

    # ---- So sánh ----
    mesh_abs_diff = (mesh_old - mesh_new).abs()
    print(f"\n1. Mesh tuyệt đối (absolute), old vs new:")
    print(f"   Max diff  : {mesh_abs_diff.max().item():.8f}")
    print(f"   Mean diff : {mesh_abs_diff.mean().item():.8f}")

    root_native_vs_new = (root_native_old - root_cam_new).abs()
    print(f"\n2. Root joint: 'joints' CÓ SẴN của bản cũ  vs  OP MidHip của bản mới:")
    print(f"   Max diff  : {root_native_vs_new.max().item():.8f}")
    print(f"   Mean diff : {root_native_vs_new.mean().item():.8f}")

    root_approx_vs_native = (root_old_approx - root_native_old).abs()
    print(f"\n   (Đối chiếu) Root 'xấp xỉ' (regressor x mesh_posed)  vs  Root 'chuẩn' (joints có sẵn), CÙNG 1 model cũ:")
    print(f"   Max diff  : {root_approx_vs_native.max().item():.8f}")
    print(f"   -> Đây chính là sai số do posedirs gây ra khi tự tính lại joint từ mesh đã posed.")

    mesh_rel_diff = (mesh_old_rootrel - mesh_new_rootrel).abs()
    print(f"\n3. Mesh SAU KHI trừ root (dùng joint 'chuẩn' cả 2 bên):")
    print(f"   Max diff  : {mesh_rel_diff.max().item():.8f}")
    print(f"   Mean diff : {mesh_rel_diff.mean().item():.8f}")

    print()
    if mesh_rel_diff.max().item() < 1e-4:
        print(">> KẾT LUẬN kịch bản này: 2 model TƯƠNG ĐƯƠNG HOÀN TOÀN (khi dùng joint chuẩn).")
    else:
        print(">> KẾT LUẬN kịch bản này: vẫn còn lệch đáng kể dù đã dùng joint chuẩn "
              "-> cần xem lại joint_map / gender / num_betas.")
    print()
    return mesh_rel_diff.max().item()


def main():
    torch.manual_seed(0)
    print("Loading SMPL models...")
    human_model = SMPLOld()
    old_smpl = human_model.layer['neutral']
    joint_regressor_old = torch.from_numpy(human_model.joint_regressor).float()

    new_smpl = SMPLNew('data_final/base_data', create_transl=False)

    compare_regressors(human_model, new_smpl)

    batch_size = 4

    # ---- Kịch bản A: pose ngẫu nhiên Gaussian thuần (biên độ lớn, phi thực tế) ----
    smpl_pose_A = torch.randn(batch_size, 72)
    smpl_shape_A = torch.randn(batch_size, 10)
    smpl_trans_A = torch.randn(batch_size, 3)
    max_diff_A = run_case(
        "A. Pose ngẫu nhiên Gaussian (torch.randn, phi thực tế)",
        old_smpl, new_smpl, joint_regressor_old,
        smpl_pose_A, smpl_shape_A, smpl_trans_A,
    )

    # ---- Kịch bản B: pose thực tế, biên độ nhỏ (giống pose người thật, root = 0) ----
    smpl_pose_B = torch.randn(batch_size, 72) * 0.15  # ~8-9 deg mỗi chiều, hợp lý hơn nhiều
    smpl_pose_B[:, :3] = 0.0  # global_orient = 0 để tách riêng ảnh hưởng của posedirs thân mình
    smpl_shape_B = torch.randn(batch_size, 10) * 0.5
    smpl_trans_B = torch.zeros(batch_size, 3)
    max_diff_B = run_case(
        "B. Pose thực tế, biên độ nhỏ (giống pose người thật)",
        old_smpl, new_smpl, joint_regressor_old,
        smpl_pose_B, smpl_shape_B, smpl_trans_B,
    )

    print("=" * 70)
    print("TỔNG KẾT")
    print("=" * 70)
    print(f"Max diff mesh root-relative (kịch bản A - pose cực đoan) : {max_diff_A:.6f}")
    print(f"Max diff mesh root-relative (kịch bản B - pose thực tế)  : {max_diff_B:.6f}")
    print()
    print("Diễn giải:")
    print("- Nếu diff ở BƯỚC 1 (J_regressor) ~ 0        -> 2 file .pkl dùng chung 1 regressor Pelvis, không phải nguồn gây lệch.")
    print("- Nếu mục '2.' (joint chuẩn old vs new) ~ 0   -> 2 model hoàn toàn tương đương ở mức joint.")
    print("- Nếu mục 'Đối chiếu' (approx vs native) LỚN  -> xác nhận: sai số ở LẦN TEST TRƯỚC là do")
    print("  cách tính SAI (regressor x mesh đã posed), không phải do model/asset khác nhau.")
    print("- Nếu mục '3.' (mesh root-relative) ~ 0 ở CẢ 2 kịch bản -> lỗi 'tệ hơn' trong thực tế Pose2Mesh")
    print("  chắc chắn nằm ở nơi KHÁC trong pipeline (J_regressor_h36m/loss/gender/eval mapping),")
    print("  không phải ở bước forward SMPL này.")


if __name__ == '__main__':
    main()