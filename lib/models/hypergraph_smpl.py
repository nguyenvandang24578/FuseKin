"""
lib/models/hypergraph_smpl.py -- (2026-11-12) HyperGCN tren 24 khop SMPL (token query cua Student).

KHONG sua hypergraph.py / HGraph.py (giong HyperDiff): lop HyperGCNSMPL24 KE THUA HyperGCN, chi ghi de 2 ham tao ma tran
sieu canh part / body cho 24 khop SMPL; cong thuc chuan hoa (DV^-1/2 H W DE^-1 H^T DV^-1/2), cac nhanh part / body /
joint, BN, ReLU giu nguyen. Block HGraphBlockSMPL24 chep cau truc HGraphBlock (norm, mixer, MLP, layer scale 1e-5).

Sieu canh: BAM SAT 10 part / 5 body cua H36M trong hypergraph.py (cung thu tu), them cac khop SMPL khong co trong
H36M vao dung chuoi chi: cot song (spine1/2/3), xuong don (collar), ban chan (foot), ban tay (hand).
SMPL-24: 0 Pelvis, 1 L_Hip, 2 R_Hip, 3 Spine1, 4 L_Knee, 5 R_Knee, 6 Spine2, 7 L_Ankle, 8 R_Ankle, 9 Spine3,
         10 L_Foot, 11 R_Foot, 12 Neck, 13 L_Collar, 14 R_Collar, 15 Head, 16 L_Shoulder, 17 R_Shoulder,
         18 L_Elbow, 19 R_Elbow, 20 L_Wrist, 21 R_Wrist, 22 L_Hand, 23 R_Hand.
Ve kiem tra: tools/vis_hyperedges.py.
"""
import numpy as np
import torch
import torch.nn as nn
from timm.models.layers import DropPath

from models.hypergraph import HyperGCN
from models.HGraph import MLP

SMPL24_NAMES = ('Pelvis', 'L_Hip', 'R_Hip', 'Spine1', 'L_Knee', 'R_Knee', 'Spine2', 'L_Ankle', 'R_Ankle', 'Spine3',
                'L_Foot', 'R_Foot', 'Neck', 'L_Collar', 'R_Collar', 'Head', 'L_Shoulder', 'R_Shoulder',
                'L_Elbow', 'R_Elbow', 'L_Wrist', 'R_Wrist', 'L_Hand', 'R_Hand')
SMPL24_PARENTS = (-1, 0, 0, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 9, 9, 12, 13, 14, 16, 17, 18, 19, 20, 21)

# 10 part, cung thu tu voi H36M (hypergraph.py): than, co-dau, dui P, cang chan P, dui T, cang chan T,
# vai-canh tay P, vai-canh tay T, cang tay P, cang tay T
SMPL24_PARTS = [
    [0, 3, 6, 9],        # 1 than        (H36M: Pelvis, Torso, Neck)
    [9, 12, 15],         # 2 co - dau    (Neck, Nose, Head)
    [0, 2, 5],           # 3 dui P       (Pelvis, RHip, RKnee)
    [5, 8, 11],          # 4 cang chan P (RKnee, RAnkle) + ban chan
    [0, 1, 4],           # 5 dui T
    [4, 7, 10],          # 6 cang chan T
    [9, 14, 17, 19],     # 7 vai - canh tay P (Neck, RShoulder, RElbow) + xuong don
    [9, 13, 16, 18],     # 8 vai - canh tay T
    [19, 21, 23],        # 9 cang tay P  (RElbow, RWrist) + ban tay
    [18, 20, 22],        # 10 cang tay T
]
# 5 body, cung thu tu voi H36M: than, tay P, tay T, chan P, chan T
SMPL24_BODY = [
    [0, 1, 2, 3, 6, 9, 12, 13, 14, 15, 16, 17],   # than (gom hong, vai, dau - giong H36M)
    [17, 19, 21, 23],                              # tay P
    [16, 18, 20, 22],                              # tay T
    [2, 5, 8, 11],                                 # chan P
    [1, 4, 7, 10],                                 # chan T
]
# Do thi khop (nhanh joint cua mixer): cay cha - con SMPL, 2 chieu
SMPL24_CONNECTIONS = {j: [] for j in range(24)}
for _j, _p in enumerate(SMPL24_PARENTS):
    if _p >= 0:
        SMPL24_CONNECTIONS[_j].append(_p)
        SMPL24_CONNECTIONS[_p].append(_j)


def hyper_adj(groups, num_nodes):
    """Ma tran sieu canh chuan hoa G = DV^-1/2 H W DE^-1 H^T DV^-1/2 (W = 1), GIONG cong thuc trong hypergraph.py."""
    H = np.zeros((num_nodes, len(groups)))
    for e, js in enumerate(groups):
        H[js, e] = 1.0
    DV, DE = H.sum(1), H.sum(0)
    assert (DV > 0).all(), f'Khop khong thuoc sieu canh nao: {np.where(DV == 0)[0].tolist()}'
    G = np.diag(DV ** -0.5) @ H @ np.diag(1.0 / DE) @ H.T @ np.diag(DV ** -0.5)
    return G


class HyperGCNSMPL24(HyperGCN):
    """HyperGCN cua HyperDiff tren 24 khop SMPL: chi doi ma tran sieu canh part / body."""
    def __init__(self, dim_in, dim_out, neighbour_num=4):
        super().__init__(dim_in, dim_out, num_nodes=24, neighbour_num=neighbour_num, use_partscale=True,
                         use_bodyscale=True, connections=SMPL24_CONNECTIONS, dataset='smpl24')

    def _init_part_adj(self, variable_weight=False):
        return hyper_adj(SMPL24_PARTS, self.num_nodes)

    def _init_body_adj(self, variable_weight=False):
        return hyper_adj(SMPL24_BODY, self.num_nodes)


class HGraphBlockSMPL24(nn.Module):
    """Chep cau truc HGraphBlock (HGraph.py): x + ls1 * mixer(LN(x)); x + ls2 * MLP(LN(x)). x: (B, 1, 24, C)."""
    def __init__(self, dim, mlp_ratio=4., act_layer=nn.GELU, drop=0., drop_path=0., layer_scale_init_value=1e-5):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.mixer = HyperGCNSMPL24(dim, dim)
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = MLP(in_features=dim, hidden_features=int(dim * mlp_ratio), act_layer=act_layer, drop=drop)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.layer_scale_1 = nn.Parameter(layer_scale_init_value * torch.ones(dim), requires_grad=True)
        self.layer_scale_2 = nn.Parameter(layer_scale_init_value * torch.ones(dim), requires_grad=True)

    def forward(self, x):
        x = x + self.drop_path(self.layer_scale_1.unsqueeze(0).unsqueeze(0) * self.mixer(self.norm1(x)))
        x = x + self.drop_path(self.layer_scale_2.unsqueeze(0).unsqueeze(0) * self.mlp(self.norm2(x)))
        return x


def create_smpl_layers(dim, n_layers=3, mlp_ratio=4., layer_scale_init_value=1e-5):
    """n_layers block HyperGCN SMPL-24, cung sieu tham so voi spatial_hypers cua Student (drop 0, layer scale 1e-5)."""
    return nn.Sequential(*[HGraphBlockSMPL24(dim, mlp_ratio, layer_scale_init_value=layer_scale_init_value)
                           for _ in range(n_layers)])
