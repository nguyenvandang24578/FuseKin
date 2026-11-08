"""
lib/models/hybrik_smpl.py -- (2026-11-08) Lop SMPL nguoc kieu HybrIK cho FuseKin:
  29 diem 3D (24 khop SMPL + 5 diem dau chi) + beta + twist (phis) -> goc xoay (IK giai tich) -> mesh.

Dung lai ham hybrik() chep tu ARTS (lib/models/hybrik_lbs.py), KHONG nap them file SMPL:
buffer (v_template, shapedirs, posedirs, J_regressor, weights) lay tu lop smplpytorch san co cua FuseKin
(utils.smpl.SMPL().layer['neutral']).

Thu tu 29 diem (giong HybrIK / ARTS SMPL_layer):
  0..23 = khop SMPL chuan, 24 = dinh dau (dinh 411), 25 = giua ban tay trai (2445), 26 = giua ban tay phai (5905),
  27 = ngon chan cai trai (3216), 28 = ngon chan cai phai (6617).
"""
import numpy as np
import torch
import torch.nn as nn

from models.hybrik_lbs import hybrik, vertices2joints

# Cha cua 24 khop SMPL chuan (kintree_table[0]) + 5 diem dau chi (nhu ARTS SMPL_layer)
SMPL_PARENTS_24 = [-1, 0, 0, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 9, 9, 12, 13, 14, 16, 17, 18, 19, 20, 21]
LEAF_PARENTS = [15, 22, 23, 10, 11]
LEAF_VERTS = [411, 2445, 5905, 3216, 6617]
LEAF_IDX = [24, 25, 26, 27, 28]
SPINE3_IDX = 9
NECK_IDX = 12


def _children_map(parents):
    """Giong SMPL_layer._parents_to_children cua ARTS (num_joints = 29)."""
    children = torch.ones_like(parents) * -1
    for i in range(parents.shape[0]):
        if children[parents[i]] < 0:
            children[parents[i]] = i
    for i in LEAF_IDX:
        children[i] = -1
    children[SPINE3_IDX] = -3
    children[0] = 3
    children[SPINE3_IDX] = NECK_IDX
    return children


def _buf(layer, *names):
    """Lay buffer dau tien ton tai trong lop smplpytorch (ten co the khac nhau giua phien ban)."""
    for n in names:
        v = getattr(layer, n, None)
        if v is not None:
            return v
    raise AttributeError(f'Lop SMPL khong co buffer nao trong {names}')


class HybrIKSMPL(nn.Module):
    def __init__(self, smpl_layer, h36m_regressor):
        """smpl_layer: lop smplpytorch (utils.smpl.SMPL().layer['neutral']); h36m_regressor: (17, 6890) numpy."""
        super().__init__()
        f32 = torch.float32
        v_template = torch.as_tensor(_buf(smpl_layer, 'th_v_template', 'v_template'), dtype=f32).reshape(-1, 3)
        shapedirs = torch.as_tensor(_buf(smpl_layer, 'th_shapedirs', 'shapedirs'), dtype=f32)       # (V, 3, 10)
        posedirs = torch.as_tensor(_buf(smpl_layer, 'th_posedirs', 'posedirs'), dtype=f32)        # (V, 3, 207)
        J_reg = torch.as_tensor(_buf(smpl_layer, 'th_J_regressor', 'J_regressor'), dtype=f32)     # (24, V)
        weights = torch.as_tensor(_buf(smpl_layer, 'th_weights', 'lbs_weights'), dtype=f32)       # (V, 24)
        V = v_template.shape[0]
        assert shapedirs.shape[:2] == (V, 3) and J_reg.shape == (24, V) and weights.shape == (V, 24), \
            f'Kich thuoc buffer SMPL la: shapedirs {tuple(shapedirs.shape)}, J {tuple(J_reg.shape)}, w {tuple(weights.shape)}'
        if posedirs.dim() == 3:                                    # (V, 3, 207) -> (207, V*3) nhu ARTS
            posedirs = posedirs.reshape(-1, posedirs.shape[-1]).t().contiguous()
        assert posedirs.shape == (207, V * 3), f'posedirs {tuple(posedirs.shape)}'
        # kiem tra cay dong hoc cua lop smplpytorch (neu co) khop voi cay chuan
        kp = getattr(smpl_layer, 'kintree_parents', None)
        if kp is not None:
            kp = [int(x) for x in list(kp)]
            assert kp[1:24] == SMPL_PARENTS_24[1:], f'kintree_parents khac SMPL chuan: {kp}'

        parents = torch.tensor(SMPL_PARENTS_24 + LEAF_PARENTS, dtype=torch.long)
        self.register_buffer('v_template', v_template)
        self.register_buffer('shapedirs', shapedirs[:, :, :10].contiguous())
        self.register_buffer('posedirs', posedirs)
        self.register_buffer('J_regressor', J_reg)
        self.register_buffer('J_regressor_h36m', torch.as_tensor(h36m_regressor, dtype=f32))
        self.register_buffer('lbs_weights', weights)
        self.register_buffer('parents', parents)
        self.register_buffer('children_map', _children_map(parents))

    def joints29_from_mesh(self, verts):
        """Mesh (B, V, 3) -> 29 diem (24 = J_regressor x mesh, 5 = dinh dau chi). Dung de tao muc tieu / khop M."""
        j24 = vertices2joints(self.J_regressor, verts)
        return torch.cat([j24, verts[:, LEAF_VERTS]], dim=1)

    def forward(self, joints29, betas=None, phis=None):
        """joints29 (B, 29, 3) met; betas (B, 10) (None = shape trung binh); phis (B, 23, 2) (None = twist 0).
        Tra dict: vertices (B, V, 3), joints24 (B, 24, 3), rot_mats (B, 24, 3, 3), joints_h36m (B, 17, 3).
        Chi dung nhanh test cua HybrIK (train=False) -> IK giai tich, kha vi."""
        B = joints29.shape[0]
        dev, dt = joints29.device, joints29.dtype
        if betas is None:
            betas = torch.zeros(B, 10, device=dev, dtype=dt)
        if phis is None:                                           # twist = 0 <=> (cos, sin) = (1, 0)
            phis = torch.zeros(B, 23, 2, device=dev, dtype=dt)
            phis[..., 0] = 1.0
        verts, joints24, rot_mats, j_h36m = hybrik(
            betas, None, joints29, phis,
            self.v_template, self.shapedirs, self.posedirs,
            self.J_regressor, self.J_regressor_h36m, self.parents, self.children_map,
            self.lbs_weights, dtype=dt, train=False, leaf_thetas=None)
        return {'vertices': verts, 'joints24': joints24, 'rot_mats': rot_mats.reshape(B, 24, 3, 3),
                'joints_h36m': j_h36m}
