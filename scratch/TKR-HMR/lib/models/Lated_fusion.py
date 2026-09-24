import os, sys
sys.path.append('./lib')

import numpy as np
import torch
import os.path as osp
import torch.nn as nn
import torch.nn.functional as F
from timm.models.layers import DropPath
from timm.models.vision_transformer import _cfg, Mlp

from core.config import cfg
from einops import rearrange

BASE_DATA_DIR = cfg.DATASET.BASE_DATA_DIR
def build_spatial_mask(num_joints=24):
    smpl_parents = [
        -1, 0, 0, 0,   # 0:Pelvis, 1:L_Hip, 2:R_Hip, 3:Spine1
         1, 2, 3,       # 4:L_Knee, 5:R_Knee, 6:Spine2
         4, 5, 6,       # 7:L_Ankle, 8:R_Ankle, 9:Spine3
         7, 8, 9,       # 10:L_Foot, 11:R_Foot, 12:Neck
         9, 9, 12,      # 13:L_Collar, 14:R_Collar, 15:Head  ← sửa ở đây
        13, 14,         # 16:L_Shoulder, 17:R_Shoulder
        16, 17,         # 18:L_Elbow, 19:R_Elbow
        18, 19,         # 20:L_Wrist, 21:R_Wrist
        20, 21          # 22:L_Hand, 23:R_Hand
    ]
    dist = torch.full((num_joints, num_joints), float('inf'))
    for i in range(num_joints):
            dist[i, i] = 0.0
            
    # Khoảng cách 1-hop (Cha - Con) = 1
    for i in range(1, num_joints):
        p = smpl_parents[i]
        dist[i, p] = 1.0
        dist[p, i] = 1.0
        
    # 3. Thuật toán Floyd-Warshall để tìm đường đi ngắn nhất cho mọi cặp
    for k in range(num_joints):
        for i in range(num_joints):
            for j in range(num_joints):
                if dist[i, j] > dist[i, k] + dist[k, j]:
                    dist[i, j] = dist[i, k] + dist[k, j]
                        
    return dist
class AdaLayerNorm(nn.Module):
    def __init__(self, num_features, eps=1e-6):
        super(AdaLayerNorm, self).__init__()
        self.mlp_gamma = nn.Linear(2048, num_features)
        self.mlp_beta = nn.Linear(2048, num_features)
        self.eps = eps

    def forward(self, x, img_feat):
        size = x.size()
        mean = x.mean(-1, keepdim=True)
        std = x.std(-1, keepdim=True)
        gamma = self.mlp_gamma(img_feat).view(size[0], 1, -1).expand(size)
        beta = self.mlp_beta(img_feat).view(size[0], 1, -1).expand(size)
        return gamma * (x - mean) / (std + self.eps) + beta
class Block(nn.Module):

    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, drop=0., attn_drop=0.,
                 drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = Attention(dim, num_heads=num_heads, qkv_bias=qkv_bias, attn_drop=attn_drop, proj_drop=drop)
        # NOTE: drop path for stochastic depth, we shall see if this is better than dropout here
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

    def forward(self, x):  # ← thêm mask=None
        attn_out, attn_map = self.attn(self.norm1(x))  # ← truyền mask
        x = x + self.drop_path(attn_out)
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x, attn_map
    
class Attention(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)
        dist_matrix = build_spatial_mask(24)
        
        # Khởi tạo Bias: Khớp càng xa nhau trên cây -> Giá trị âm càng lớn -> Attention ban đầu càng nhỏ
        # Hệ số -0.5 là một hyperparameter (decay rate) để tránh việc phạt quá nặng.
        init_bias = -0.5 * dist_matrix 
        
        # Biến ma trận này thành nn.Parameter để Mạng TỰ DO CẬP NHẬT trong lúc train!
        self.spatial_bias = nn.Parameter(init_bias)
    def forward(self, x):  # ← thêm mask=None
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)

        attn = (q @ k.transpose(-2, -1)) * self.scale  # (B, H, N, N)

        attn = attn + self.spatial_bias.unsqueeze(0).unsqueeze(0)

        attn = attn.softmax(dim=-1)
        # score_expected = attn[0, 0, 4, 1].item()   # L_Knee → L_Hip (nên > 0)
        # score_forbidden = attn[0, 0, 4, 22].item() # L_Knee → L_Hand (nên ≈ 0)
        # print(f"[DEBUG] L_Knee→L_Hip: {score_expected:.6f} | L_Knee→L_Hand: {score_forbidden:.6f}")
        # assert score_forbidden < 1e-3, "MASK BỊ LỖI: khớp không liên quan vẫn có attention!"        
        
        attn_map = attn.clone().detach()
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x, attn_map

# ==============================================================================
# 2. KHỐI CROSS-ATTENTION THỜI GIAN VỚI GAUSSIAN MASK (SDPA)
# ==============================================================================
class MaskedTemporalCrossAttentionBlock(nn.Module):
    def __init__(self, embed_dim=512, num_heads=8, drop=0.1):
        super().__init__()
        self.num_heads = num_heads
        
        # Norms
        self.norm_q = nn.LayerNorm(embed_dim)
        self.norm_k = nn.LayerNorm(embed_dim)
        self.norm_v = nn.LayerNorm(embed_dim)
        
        # Projections
        self.wq = nn.Linear(embed_dim, embed_dim)
        self.wk = nn.Linear(embed_dim, embed_dim)
        self.wv = nn.Linear(embed_dim, embed_dim)
        self.proj = nn.Linear(embed_dim, embed_dim)
        
        self.mlp = Mlp(in_features=embed_dim, hidden_features=int(embed_dim*4), act_layer=nn.GELU, drop=drop)
        self.norm_mlp = nn.LayerNorm(embed_dim)

    def forward(self, q, kv, attn_bias):
        """
        q, kv: (B*J_group, T, D) - CHÚ Ý: Trục T bây giờ mới đúng là trục Sequence!
        sigma: (B*J_group, 1, T, 1)
        """
        N, T, D = q.shape
        
        # Pre-norm
        q_norm = self.norm_q(q)
        k_norm = self.norm_k(kv)
        v_norm = self.norm_v(kv)
        
        # Project & Reshape cho Multi-Head
        q_h = rearrange(self.wq(q_norm), 'n t (h d) -> n h t d', h=self.num_heads)
        k_h = rearrange(self.wk(k_norm), 'n t (h d) -> n h t d', h=self.num_heads)
        v_h = rearrange(self.wv(v_norm), 'n t (h d) -> n h t d', h=self.num_heads)

        # -----------------------------------------------------
        # TÍNH TOÁN GAUSSIAN MASK TRỰC TIẾP TRÊN GPU
        # -----------------------------------------------------
        bias = attn_bias.unsqueeze(0).unsqueeze(0)   # (1, 1, T, T)

        # -----------------------------------------------------
        # FLASH CROSS-ATTENTION
        # -----------------------------------------------------
        attn_out = F.scaled_dot_product_attention(
            q_h, k_h, v_h, 
            attn_mask=bias, 
            dropout_p=0.1 if self.training else 0.0
        )
        
        # Output Projection & Residual
        out = rearrange(attn_out, 'n h t d -> n t (h d)')
        out = q + self.proj(out)
        
        # MLP Block
        out = out + self.mlp(self.norm_mlp(out))
        return out

# ==============================================================================
# 3. TRÁI TIM CỦA V3: KINEMATIC TEMPORAL ATTENTION
# ==============================================================================
class KinematicTemporalAttention(nn.Module):
    def __init__(self, embed_dim=512, seq_len=16, spatial_mask=None):
        super().__init__()
        self.seq_len = seq_len
        positions = torch.arange(seq_len)
        dist = positions.unsqueeze(1) - positions.unsqueeze(0)   # (T, T)
        dist_idx = dist + (seq_len - 1)                           # (T, T), range [0, 2T-2]
        self.register_buffer('dist_idx', dist_idx)
        # 1. NHÓM KHỚP XƯƠNG
        self.global_joints = [0,1,2,3,4,5,6]               
        self.mid_joints    = [7,8,9,10,11,12,13,14,15,16,17] 
        self.local_joints  = [18,19,20,21,22,23]             
        
        self.temporal_pe = nn.Parameter(torch.zeros(1, seq_len, 1, embed_dim))
        self.spatial_pe = nn.Parameter(torch.zeros(1, 1, 24, embed_dim))
        nn.init.trunc_normal_(self.temporal_pe, std=0.02)
        nn.init.trunc_normal_(self.spatial_pe, std=0.02)
        
        # 2. KHỞI TẠO 3 MẠNG GATING ĐỘC LẬP CHO 3 NHÓM
        self.temporal_bias_global = nn.Parameter(torch.zeros(2 * seq_len - 1))  # (2T-1,)
        self.temporal_bias_mid    = nn.Parameter(torch.zeros(2 * seq_len - 1))
        self.temporal_bias_local  = nn.Parameter(torch.zeros(2 * seq_len - 1))
        
        # 3. KHỞI TẠO 3 KHỐI THỜI GIAN ĐỘC LẬP
        self.attn_global = MaskedTemporalCrossAttentionBlock(embed_dim)
        self.attn_mid    = MaskedTemporalCrossAttentionBlock(embed_dim)
        self.attn_local  = MaskedTemporalCrossAttentionBlock(embed_dim)
        
        # 4. KHỐI TRUYỀN TIN KHÔNG GIAN (Giữ nguyên Block của bạn)
        # Giả định class Block() và AdaLayerNorm đã được định nghĩa ở trên trong file
        self.joint_sa = Block(dim=embed_dim, num_heads=8, mlp_ratio=4., qkv_bias=True)
    def get_temporal_bias(self, bias_table):
        return bias_table[self.dist_idx] 
    def forward(self, smpl_tokens, spin_kv):
        B, T, N_joints, D = smpl_tokens.shape

        # --- Cộng PE ---
        spin_kv = spin_kv + self.temporal_pe + self.spatial_pe

        # BƯỚC 1: TÍNH TEMPORAL BIAS MATRIX CHO TỪNG NHÓM KHỚP
        bias_g = self.get_temporal_bias(self.temporal_bias_global)
        bias_m = self.get_temporal_bias(self.temporal_bias_mid)
        bias_l = self.get_temporal_bias(self.temporal_bias_local)

        # BƯỚC 2: TÁCH NHÓM & RESHAPE CHO TRỤC THỜI GIAN
        def prepare_group(idx):
            q  = rearrange(smpl_tokens[:, :, idx, :], 'b t j d -> (b j) t d')
            kv = rearrange(spin_kv[:, :, idx, :],     'b t j d -> (b j) t d')
            return q, kv

        q_g, kv_g = prepare_group(self.global_joints)
        q_m, kv_m = prepare_group(self.mid_joints)
        q_l, kv_l = prepare_group(self.local_joints)

        # BƯỚC 3: TEMPORAL CROSS-ATTENTION TỪNG NHÓM VỚI LEARNABLE BIAS
        out_g = self.attn_global(q_g, kv_g, attn_bias=bias_g)
        out_m = self.attn_mid   (q_m, kv_m, attn_bias=bias_m)
        out_l = self.attn_local (q_l, kv_l, attn_bias=bias_l)

# BƯỚC 4: RÁP LẠI & TRUYỀN TIN KHÔNG GIAN
        global_out = rearrange(out_g, '(b j) t d -> b t j d', b=B)
        mid_out    = rearrange(out_m, '(b j) t d -> b t j d', b=B)
        local_out  = rearrange(out_l, '(b j) t d -> b t j d', b=B)

        # CÁCH TỐT NHẤT: Dùng clone() và gán theo đúng index của từng nhóm
        out_assembled = smpl_tokens.clone()
        out_assembled[:, :, self.global_joints, :] += global_out
        out_assembled[:, :, self.mid_joints, :]    += mid_out
        out_assembled[:, :, self.local_joints, :]  += local_out

        # Làm phẳng để đưa vào Spatial Attention
        out_assembled_flat = out_assembled.reshape(B * T, N_joints, D)
        out_final, att_map = self.joint_sa(out_assembled_flat)
        
        bias_dict = {
            'global': bias_g.detach().cpu(),
            'mid':    bias_m.detach().cpu(),
            'local':  bias_l.detach().cpu(),
        }
        return out_final.reshape(B, T, N_joints, D), att_map, bias_dict
def get_model(num_joint, embed_dim, depth):
    model = KinematicTemporalAttention(embed_dim, depth)
    return model
