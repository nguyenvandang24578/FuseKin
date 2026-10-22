"""
lib/models/hyp_selector.py
HypSelector: dung dac trung anh chon gia thuyet DiffLifter THEO TUNG KHOP (cach A', oracle: lifter 72 -> 40 mm).

  hyp (B, K, J, 3) gia thuyet 3D root-relative (DiffLifter dong bang, S buoc, K gia thuyet)
  feature_map (B, C, h, w) dac trung anh backbone; pose_2d (B, J, 2) khung crop [-1, 1]; mask (B, J, 1) conf
  -> diem (B, K, J) -> softmax theo K -> pose (B, J, 3) = trung binh co trong so tung khop

Token (gia thuyet k, khop j) = MLP([p, p - mean_K(p), p - p_cha] * 3)       -- hinh dang gia thuyet quanh khop
                              + Linear(grid_sample(feature_map, 2D khop j))  -- anh tai khop (khop mat -> vector rieng)
                              + embedding khop j
-> 1 lop Transformer giua 17 khop TRONG cung 1 gia thuyet (ngu canh ca tu the) -> Linear -> diem.
Lop diem cuoi khoi tao 0 -> luc dau trong so deu = trung binh K gia thuyet (~ ban S=1, diag muc F).
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

from models.diff_lifter import H36M_PARENTS


class HypSelector(nn.Module):
    def __init__(self, num_joints=17, img_dim=2048, dim=256, num_heads=8, pos_scale=3.0):
        super().__init__()
        self.pos_scale = pos_scale       # toa do met (std ~0.3) -> ~1, giong scale cua DiffLifter
        self.register_buffer('parents', torch.tensor([max(p, 0) for p in H36M_PARENTS]), persistent=False)
        self.hyp_mlp = nn.Sequential(nn.Linear(9, dim), nn.GELU(), nn.Linear(dim, dim))
        self.img_proj = nn.Sequential(nn.Linear(img_dim, dim), nn.LayerNorm(dim))
        self.missing_img = nn.Parameter(torch.zeros(dim))         # khop mask = 0: khong doc anh tai (0, 0)
        self.joint_emb = nn.Parameter(torch.zeros(1, num_joints, dim))
        nn.init.trunc_normal_(self.joint_emb, std=0.02)
        self.ctx = nn.TransformerEncoderLayer(d_model=dim, nhead=num_heads, dim_feedforward=dim * 2,
                                              dropout=0.0, batch_first=True, norm_first=True)
        self.head = nn.Sequential(nn.LayerNorm(dim), nn.Linear(dim, 1))
        nn.init.zeros_(self.head[-1].weight)
        nn.init.zeros_(self.head[-1].bias)

    def forward(self, hyp, feature_map, pose_2d, mask):
        B, K, J, _ = hyp.shape
        # --- hinh dang gia thuyet ---
        rel_mean = hyp - hyp.mean(dim=1, keepdim=True)                       # khac trung binh K o dau
        bone = hyp - hyp[:, :, self.parents]                                 # huong xuong toi khop cha
        h = self.hyp_mlp(torch.cat([hyp, rel_mean, bone], dim=-1) * self.pos_scale)   # (B, K, J, D)
        # --- anh tai vi tri 2D cua khop ---
        grid = pose_2d[:, :, None, :].to(feature_map.dtype)                  # (B, J, 1, 2), (x, y) trong [-1, 1]
        img = F.grid_sample(feature_map, grid, mode='bilinear', align_corners=False)[..., 0].transpose(1, 2)
        vis = (mask.reshape(B, J, 1) > 0).to(h.dtype)
        img = self.img_proj(img) * vis + self.missing_img * (1 - vis)         # (B, J, D)
        x = h + img[:, None] + self.joint_emb[:, None]                       # (B, K, J, D)
        # --- ngu canh giua cac khop trong cung 1 gia thuyet ---
        x = self.ctx(x.reshape(B * K, J, -1)).reshape(B, K, J, -1)
        w = self.head(x)[..., 0].softmax(dim=1)                              # (B, K, J), tong theo K = 1
        pose = (w[..., None] * hyp).sum(dim=1)                               # (B, J, 3)
        return pose - pose[:, 0:1], w
