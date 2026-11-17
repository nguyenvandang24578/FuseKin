"""
lib/models/hybrik_ik_grad.py (2026-11-17, nguoi dung duyet, PLAN_H1H2.md muc 6, MODEL.v9_ik_grad) -- IK HybrIK KHA VI.

Ban sua tu batch_inverse_kinematics_transform (lib/models/hybrik_lbs.py, chep ARTS; KHONG sua file do):
  - Huong xuong con cua moi khop = xuong DU DOAN (khop con - khop cha cua chinh khung xuong dau vao, 'orig_vec'),
    KHONG detach, BO nhanh big_diff (> 15 mm) -> cung mot cong thuc cho train va test, moi khop co gradient.
    (Ban goc: rel_pose_skeleton bi .detach() va khi lech > 15 mm thi thay bang orig_vec da detach -> cat gradient.)
    Dung xuong du doan thay vi (khop con du doan - khop cha dung theo do dai template) giong nhanh naive calc_swing
    cua HybrIK: khong de sai do dai xuong cha keo lech huong xuong con.
  - Huong Pelvis: giai tich Rodrigues 2 buoc (giong batch_get_pelvis_orient nhanh train=True), KHONG SVD, KHONG .cpu().
  - Spine3: trong FuseKin children_map[9] = Neck (12) (hybrik_smpl._children_map ghi de -3) -> Spine3 la khop 1 con
    nhu cac khop khac, KHONG di qua nhanh 3 con / SVD -> khong can xu ly rieng.
  - Chuan hoa vector bang sqrt(|v|^2 + eps^2) (gradient huu han tai 0), khong assert NaN, khong gan in-place.
Swing (Rodrigues tu xuong T-pose sang xuong du doan, trong he khop cha) + twist (phis cos / sin quanh truc xuong
T-pose) giong ban goc. Tra rot_mats LOCAL (B, 24, 3, 3), khop 0 = huong toan cuc.
"""
import torch

from models.hybrik_lbs import blend_shapes, vertices2joints
from models.hybrik_smpl import LEAF_VERTS


def _safe_norm(v, eps=1e-6):
    """v (B, 3, 1) -> (B, 1, 1), sqrt(|v|^2 + eps^2)."""
    return torch.sqrt((v * v).sum(dim=1, keepdim=True) + eps * eps)


def _skew(a):
    """a (B, 3, 1) -> ma tran doi xung lech (B, 3, 3)."""
    B = a.shape[0]
    z = torch.zeros(B, 1, 1, dtype=a.dtype, device=a.device)
    rx, ry, rz = a[:, 0:1], a[:, 1:2], a[:, 2:3]
    return torch.cat([z, -rz, ry, rz, z, -rx, -ry, rx, z], dim=1).view(B, 3, 3)


def _vec2rot(v_rest, v_final):
    """Xoay nho nhat dua v_rest (B, 3, 1) ve huong v_final (B, 3, 1) (Rodrigues, giong vectors2rotmat)."""
    nr, nf = _safe_norm(v_rest), _safe_norm(v_final)
    axis = torch.cross(v_rest, v_final, dim=1)
    na = _safe_norm(axis)
    cos = (v_rest * v_final).sum(dim=1, keepdim=True) / (nr * nf)
    sin = na / (nr * nf)
    K = _skew(axis / na)
    ident = torch.eye(3, dtype=v_rest.dtype, device=v_rest.device)[None]
    return ident + sin * K + (1 - cos) * torch.bmm(K, K)


def _pelvis_orient(rel_pose, rel_rest, parents, spine_child):
    """Giong batch_get_pelvis_orient (giai tich): xoay xuong Spine1 ve dung huong, roi xoay quanh no de trung
    binh 2 xuong hong (chieu len mat phang vuong goc Spine1) trung huong. rel_* (B, 29, 3, 1)."""
    hips = [i for i in range(1, len(parents)) if parents[i] == 0 and i != spine_child]
    sp_f, sp_r = rel_pose[:, spine_child], rel_rest[:, spine_child]
    sp_dir = sp_f / _safe_norm(sp_f)
    R_spine = _vec2rot(sp_r, sp_f)
    c_f = sum(rel_pose[:, i] for i in hips) / len(hips)
    c_r = torch.matmul(R_spine, sum(rel_rest[:, i] for i in hips) / len(hips))
    c_f = c_f - (c_f * sp_dir).sum(dim=1, keepdim=True) * sp_dir
    c_r = c_r - (c_r * sp_dir).sum(dim=1, keepdim=True) * sp_dir
    return torch.matmul(_vec2rot(c_r, c_f), R_spine)


def ik_grad(pose_skeleton, phis, rest_pose, children, parents):
    """pose_skeleton (B, 29, 3) met; phis (B, 23, 2) (cos, sin) twist; rest_pose (B, 29, 3) khop T-pose (theo beta);
    children, parents: tensor (29,) nhu HybrIKSMPL.children_map / parents. -> rot_mats LOCAL (B, 24, 3, 3)."""
    B = pose_skeleton.shape[0]
    ch = [int(c) for c in children.tolist()]
    par = [int(p) for p in parents.tolist()]
    assert ch[0] == 3, 'children[0] phai la Spine1 (3)'
    pidx = torch.as_tensor(par[1:], device=pose_skeleton.device, dtype=torch.long)
    # Vector xuong (con - cha); goc giu nguyen (khong dung). Khong gan in-place.
    rel_pose = torch.cat([pose_skeleton[:, :1], pose_skeleton[:, 1:] - pose_skeleton[:, pidx]], dim=1).unsqueeze(-1)
    rel_rest = torch.cat([rest_pose[:, :1], rest_pose[:, 1:] - rest_pose[:, pidx]], dim=1).unsqueeze(-1)
    ph = phis / torch.sqrt((phis * phis).sum(dim=2, keepdim=True) + 1e-12)
    ident = torch.eye(3, dtype=pose_skeleton.dtype, device=pose_skeleton.device)[None]

    g = _pelvis_orient(rel_pose, rel_rest, par, ch[0])
    chain, local = [g], [g]
    for i in range(1, len(par)):
        if ch[i] == -1:                                          # diem la (24..28): khong co goc rieng
            continue
        assert ch[i] >= 0, f'khop {i} co children {ch[i]} (nhanh 3 con chua ho tro)'
        Rp = chain[par[i]]
        c = ch[i]
        v_f = torch.matmul(Rp.transpose(1, 2), rel_pose[:, c])   # xuong du doan, trong he khop cha
        v_r = rel_rest[:, c]
        R_swing = _vec2rot(v_r, v_f)
        # twist quanh truc xuong T-pose (giong ban goc)
        K = _skew(v_r / _safe_norm(v_r))
        cos = ph[:, i - 1, 0].view(B, 1, 1)
        sin = ph[:, i - 1, 1].view(B, 1, 1)
        R_spin = ident + sin * K + (1 - cos) * torch.bmm(K, K)
        R = torch.matmul(R_swing, R_spin)
        chain.append(torch.matmul(Rp, R))
        local.append(R)
    assert len(local) == 24, f'IK tra {len(local)} goc, can 24'
    return torch.stack(local, dim=1)


def hybrik_ik_grad(hk, joints29, betas, phis):
    """Giong HybrIKSMPL.ik (hk = HybrIKSMPL) nhung dung ik_grad: (B, 29, 3) + betas (B, 10) + phis (B, 23, 2)
    -> rot_mats (B, 24, 3, 3)."""
    dt = joints29.dtype
    v_shaped = hk.v_template[None] + blend_shapes(betas.to(dt), hk.shapedirs)
    rest_J = torch.cat([vertices2joints(hk.J_regressor, v_shaped), v_shaped[:, LEAF_VERTS]], dim=1)   # (B, 29, 3)
    return ik_grad(joints29, phis, rest_J, hk.children_map, hk.parents)
