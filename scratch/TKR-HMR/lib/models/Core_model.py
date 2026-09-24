# ============================================================
# 3. CROSS-ATTENTION: Pose Tokens ← Image Features
#    *** ĐÂY LÀ NOVELTY CHÍNH ***
# ============================================================
import os
import sys
sys.path.append('./lib')
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from typing import Tuple, Type, Optional
from einops import rearrange
from models.common import LayerNorm2d, MLPBlock
from timm.models.layers import DropPath, to_2tuple, trunc_normal_
from timm.models.vision_transformer import _cfg, Mlp
from functools import partial
from core.config import cfg

# ============================================================
# 1. NATIVE PYTORCH ATTENTION (thay thế xformers)
# ============================================================
class Attention(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0.):
        super().__init__()

        self.num_heads = num_heads
        head_dim = dim // num_heads
        assert dim % num_heads == 0
        self.scale = qk_scale or head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]  
        
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)

        return x


class Block(nn.Module):
    def __init__(self, dim, num_heads, mlp_hidden_dim, qkv_bias=False, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = Attention(dim, num_heads=num_heads, qkv_bias=qkv_bias, \
            qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

    def forward(self, x):
        x = x + self.drop_path(self.attn(self.norm1(x)))
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x


class Transformer(nn.Module):
    def __init__(self, depth=4, embed_dim=512, mlp_hidden_dim=2048, length=cfg.DATASET.seqlen, h=8):
        super().__init__()
        drop_rate = 0.1
        drop_path_rate = 0.2
        attn_drop_rate = 0.

        self.pos_embed = nn.Parameter(torch.zeros(1, length, embed_dim))

        norm_layer = partial(nn.LayerNorm, eps=1e-6)
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]  

        self.blocks = nn.ModuleList([
            Block(dim=embed_dim, num_heads=h, mlp_hidden_dim=mlp_hidden_dim, qkv_bias=True, qk_scale=None,
                  drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[i], norm_layer=norm_layer)
            for i in range(depth)]) 
        self.norm = norm_layer(embed_dim)

    def forward(self, x):
        x = x + self.pos_embed
        for blk in self.blocks:
            x = blk(x)
        x = self.norm(x)
        return x

class CrossAttention(nn.Module):
    def __init__(self, dim, v_dim, kv_num, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.num_heads = num_heads
        self.kv_num = kv_num
        head_dim = dim // num_heads
        # NOTE scale factor was wrong in my original version, can set manually to be compat with prev weights
        self.scale = qk_scale or head_dim ** -0.5

        self.wq = nn.Linear(dim, dim, bias=qkv_bias)
        self.wk = nn.Linear(dim, dim, bias=qkv_bias)
        self.wv = nn.Linear(v_dim, v_dim, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(v_dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, xq, xk, xv):

        B, N, C = xq.shape
        v_dim = xv.shape[-1]
        q = self.wq(xq).reshape(B, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)  # [B,N1,C] -> [B,N1,H,(C/H)] -> [B,H,N1,(C/H)]
        k = self.wk(xk).reshape(B, self.kv_num, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)  # [B,N2,C] -> [B,N2,H,(C/H)] -> [B,H,N2,(C/H)]
        v = self.wv(xv).reshape(B, self.kv_num, self.num_heads, v_dim // self.num_heads).permute(0, 2, 1, 3)  # [B,N2,C] -> [B,N2,H,(C/H)] -> [B,H,N2,(C/H)]

        attn = (q @ k.transpose(-2, -1)) * self.scale  # [B,H,N1,(C/H)] @ [B,H,(C/H),N2] -> [B,H,N1,N2]
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, v_dim)   # [B,H,N1,N2] @ [B,H,N2,(C/H)] -> [B,H,N1,(C/H)] -> [B,N1,H,(C/H)] -> [B,N1,C]
        x = self.proj(x)
        x = self.proj_drop(x)
        return x

class CrossAttentionBlock(nn.Module):
    def __init__(self, q_dim, k_dim, v_dim, kv_num, num_heads, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop=0.2, 
                 attn_drop=0.2, drop_path=0.2, act_layer=nn.GELU, norm_layer=nn.LayerNorm, has_mlp=True):
        super().__init__()
        self.normq = norm_layer(q_dim)
        self.normk = norm_layer(k_dim)
        self.normv = norm_layer(v_dim)
        self.kv_num = kv_num
        self.attn = CrossAttention(q_dim, v_dim, kv_num = kv_num, num_heads=num_heads, qkv_bias=qkv_bias, 
                                   qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop)
        # NOTE: drop path for stochastic depth, we shall see if this is better than dropout here
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.has_mlp = has_mlp
        if has_mlp:
            self.norm2 = norm_layer(q_dim)
            mlp_hidden_dim = int(q_dim * mlp_ratio)
            self.mlp = Mlp(in_features=q_dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

    def forward(self, xq, xk, xv):
        xq = xq + self.drop_path(self.attn(self.normq(xq), self.normk(xk), self.normv(xv)))
        if self.has_mlp:
            xq = xq + self.drop_path(self.mlp(self.norm2(xq)))

        return xq


# ============================================================
# 2. STATIC KINEMATIC GCN
#    (thay thế C2KT dynamic graph bị degenerate)
# ============================================================
def build_smpl_adjacency() -> Tensor:
    """
    Tạo normalized adjacency matrix từ SMPL kinematic tree (24 joints).
    Áp dụng symmetric normalization: D^{-1/2} * A * D^{-1/2}
    → Stable gradient, không bị saturate như softmax(-1e9) cũ.
    """
    parents = [
        -1, 0, 0, 0, 1, 2, 3, 4, 5, 6,
         7, 8, 9, 9, 9,12,13,14,16,17,
        18,19,20,21
    ]
    adj = torch.zeros(24, 24)
    for i, p in enumerate(parents):
        adj[i, i] = 1.0          # self-loop
        if p != -1:
            adj[i, p] = 1.0
            adj[p, i] = 1.0      # undirected

    # Symmetric normalization chuẩn GCN
    deg     = adj.sum(dim=-1, keepdim=True).clamp(min=1e-6)
    deg_inv = deg.pow(-0.5)
    A_norm  = deg_inv * adj * deg_inv.T   # (24, 24)
    return A_norm


class StaticKinematicGCN(nn.Module):
    def __init__(self, embed_dim: int = 512, num_joints: int = 24):
        super().__init__()
        # register_buffer: tự lên đúng device, không cần .cuda() tay
        self.register_buffer("A_norm", build_smpl_adjacency())

        self.W_v  = nn.Linear(embed_dim, embed_dim)
        self.W_o  = nn.Linear(embed_dim, embed_dim)
        self.norm = nn.LayerNorm(embed_dim)
        self.ffn  = nn.Sequential(
            nn.LayerNorm(embed_dim),
            nn.Linear(embed_dim, embed_dim * 2),
            nn.GELU(),
            nn.Linear(embed_dim * 2, embed_dim),
        )

    def forward(self, pose_tokens: Tensor) -> Tensor:
        """
        Args:
            pose_tokens : (B, T, J, D)
        Returns:
            pose_tokens : (B, T, J, D)
        """
        V     = self.W_v(pose_tokens)              # (B, T, J, D)
        X_gcn = torch.matmul(self.A_norm, V)       # (B, T, J, D)
        X_gcn = self.W_o(X_gcn)
        x     = self.norm(pose_tokens + X_gcn)     # residual + LN
        x     = x + self.ffn(x)                    # FFN + residual
        return x
class KinematicGAT(nn.Module):
    def __init__(self, embed_dim: int = 512, num_joints: int = 24, num_heads: int = 4):
        super().__init__()
        assert embed_dim % num_heads == 0, \
            f"embed_dim ({embed_dim}) phải chia hết cho num_heads ({num_heads})"

        A = build_smpl_adjacency()  # (24, 24) binary tensor
        mask = torch.zeros(num_joints, num_joints)
        mask[A == 0] = float('-inf')
        self.register_buffer("connectivity_mask", mask)  # auto move to device

        self.num_heads = num_heads
        self.head_dim  = embed_dim // num_heads  # e.g. 512//4 = 128

        self.linear_proj    = nn.Linear(embed_dim, embed_dim, bias=False)
        self.scoring_fn_src = nn.Parameter(torch.Tensor(1, num_heads, self.head_dim))
        self.scoring_fn_tgt = nn.Parameter(torch.Tensor(1, num_heads, self.head_dim))
        self.leaky_relu     = nn.LeakyReLU(0.2)
        self.softmax        = nn.Softmax(dim=-1)

        self.W_o  = nn.Linear(embed_dim, embed_dim)
        self.norm = nn.LayerNorm(embed_dim)
        self.ffn  = nn.Sequential(
            nn.LayerNorm(embed_dim),
            nn.Linear(embed_dim, embed_dim * 2),
            nn.GELU(),
            nn.Linear(embed_dim * 2, embed_dim),
        )

        nn.init.xavier_uniform_(self.linear_proj.weight)
        nn.init.xavier_uniform_(self.scoring_fn_src)
        nn.init.xavier_uniform_(self.scoring_fn_tgt)

        self.attn_weights = None  # cache để visualize

    def forward(self, pose_tokens: Tensor, return_attn: bool = False):
        """
        Args:
            pose_tokens: (B, T, J, D)
        Returns:
            x: (B, T, J, D)
            attn (optional): (B, T, NH, J, J)
        """
        B, T, J, D = pose_tokens.shape
        N = B * T

        x = pose_tokens.view(N, J, D)  # (N, J, D)

        # 1. Linear projection → (N, J, NH, head_dim)
        proj = self.linear_proj(x).view(N, J, self.num_heads, self.head_dim)

        # 2. Tính scores: (N, J, NH) → element-wise với scoring vector
        score_src = (proj * self.scoring_fn_src).sum(dim=-1)  # (N, J, NH)
        score_tgt = (proj * self.scoring_fn_tgt).sum(dim=-1)  # (N, J, NH)

        # Broadcast: (N, NH, J, 1) + (N, NH, 1, J) → (N, NH, J, J)
        scores = self.leaky_relu(
            score_src.permute(0, 2, 1).unsqueeze(-1) +   # (N, NH, J, 1)
            score_tgt.permute(0, 2, 1).unsqueeze(-2)      # (N, NH, 1, J)
        )

        # 3. Mask non-edges (-inf → softmax = 0)
        # connectivity_mask: (J, J) → broadcast sang (N, NH, J, J)
        scores = scores + self.connectivity_mask         # (N, NH, J, J)
        attn = self.softmax(scores)                      # (N, NH, J, J)
        self.attn_weights = attn.detach()

        # 4. Aggregate features
        proj_t = proj.permute(0, 2, 1, 3)               # (N, NH, J, head_dim)
        out = attn @ proj_t                              # (N, NH, J, head_dim)
        out = out.permute(0, 2, 1, 3).reshape(N, J, D)  # (N, J, D)

        # 5. Output projection + residual + FFN
        out = self.W_o(out)
        x   = self.norm(x + out)         # residual + LayerNorm
        x   = x + self.ffn(x)            # FFN + residual
        x   = x.view(B, T, J, D)

        if return_attn:
            return x, attn.view(B, T, self.num_heads, J, J)
        return x
class AttentionGCN(nn.Module):
    def __init__(self, embed_dim, num_joints):
        super().__init__()
        # Step 1: Self-Attention toàn cục
        self.attn = nn.MultiheadAttention(
            embed_dim, num_heads=8, batch_first=True
        )
        self.attn_norm = nn.LayerNorm(embed_dim)

        # Step 2: GCN theo kinematic tree
        self.gcn = KinematicGAT(embed_dim=embed_dim, num_joints=24, num_heads=4)
    def forward(self, pose_tokens):
        # pose_tokens: (B, T, J, D)
        B, T, J, D = pose_tokens.shape

        # Attention dọc trục J
        x = pose_tokens.view(B * T, J, D)
        attn_out, _ = self.attn(x, x, x)
        x = self.attn_norm(x + attn_out)
        x = x.view(B, T, J, D)

        # GCN refine
        x, gat_attn = self.gcn(x, return_attn=True)  # ← x đã qua Attention
        return x 

class DeepAttentionGCN(nn.Module):
    def __init__(self, embed_dim: int = 512, num_joints: int = 24, num_layers: int = 3):
        """
        Khởi tạo mạng GCN sâu với nhiều lớp AttentionGCN.
        
        Args:
            embed_dim: Số chiều đặc trưng (Dimension).
            num_joints: Số lượng khớp xương (thường là 24 với SMPL).
            num_layers: Số lớp AttentionGCN muốn xếp chồng (mặc định là 3).
        """
        super().__init__()
        self.num_layers = num_layers
        
        # Sử dụng nn.ModuleList để PyTorch có thể theo dõi và cập nhật 
        # parameters (trọng số) của tất cả các lớp trong quá trình training.
        self.layers = nn.ModuleList([
            AttentionGCN(embed_dim, num_joints) for _ in range(num_layers)
        ])
        
        # (Tùy chọn) Thêm một lớp LayerNorm ở cuối để chuẩn hóa output
        self.final_norm = nn.LayerNorm(embed_dim)

    def forward(self, pose_tokens: torch.Tensor) -> torch.Tensor:
        """
        Luồng truyền dữ liệu.
        
        Args:
            pose_tokens: Tensor đầu vào có kích thước (B, T, J, D)
                         B: Batch size
                         T: Time (Số khung hình)
                         J: Joints (Số khớp)
                         D: Dimension (Chiều đặc trưng)
        Returns:
            Tensor đã qua xử lý, kích thước (B, T, J, D)
        """
        x = pose_tokens
        attn_w = None

        # Truyền dữ liệu tuần tự qua từng lớp AttentionGCN
        for i, layer in enumerate(self.layers):
            x  = layer(x)
            
        # Chuẩn hóa lần cuối trước khi trả về
        x = self.final_norm(x)
        
        return x 
# ============================================================
# 3. CROSS-ATTENTION: Pose Tokens ← Image Features
#    *** ĐÂY LÀ NOVELTY CHÍNH ***
# ============================================================
class PoseImageCrossAttention(nn.Module):
    """
    Cross-attention với Joint-Aware Global Features:
      - Thay vì broadcast (copy) 1 global feature cho 24 khớp.
      - Dùng MLP để chiếu Global Feature (D) thành (J * D).
      - Giúp mỗi khớp nhận được ngữ cảnh ảnh (K, V) phù hợp với vị trí và đặc tính của nó.
    """
    def __init__(self, embed_dim: int = 512, seq_len: int = 16, num_joints: int = 24):
        super().__init__()
        self.num_joints = num_joints
        
        self.CAB = CrossAttentionBlock(q_dim=embed_dim, k_dim=embed_dim, v_dim=embed_dim, kv_num = seq_len, num_heads=8, mlp_ratio=4., qkv_bias=True,
                                                drop=0.1, attn_drop=0.1, drop_path=0.25, has_mlp=True)

    def forward(self, pose_tokens: Tensor, img_feats: Tensor, pose_pe) -> Tensor:
        """
        Args:
            pose_tokens : (B, T, J, D) - Query từ SMPL joints
            img_feats   : (B, T, D)    - Global image features từ backbone
        Returns:
            pose_tokens : (B, T, J, D)
        """

        pose_tokens = pose_tokens + pose_pe
        global_output = self.CAB(pose_tokens, img_feats, img_feats)

        # Trả về shape ban đầu
        return global_output
