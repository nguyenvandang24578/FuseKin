"""
lib/models/diff_lifter.py
DiffLifter: D3DP adapted cho single-frame 2D->3D lifting.
  - Bo toan bo Temporal branch cua MixSTE2 -> SpatialMixSTE
  - Giu nguyen diffusion framework (cosine schedule, DDIM, pred_x0)
  - Them conf channel (occlusion mask) vao input: in_chans=3 thay vi 2
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from functools import partial
from collections import namedtuple

try:
    from timm.models.layers import DropPath, trunc_normal_
except ImportError:
    from timm.layers import DropPath, trunc_normal_

__all__ = ["DiffLifter", "aggregate_jpma"]

ModelPrediction = namedtuple('ModelPrediction', ['pred_noise', 'pred_x0'])


# ============================================================
# Diffusion schedule utilities (copy tu D3DP/common/diffusionpose.py)
# ============================================================

def cosine_beta_schedule(timesteps, s=0.008):
    steps = timesteps + 1
    x = torch.linspace(0, timesteps, steps, dtype=torch.float64)
    alphas_cumprod = torch.cos(((x / timesteps) + s) / (1 + s) * math.pi * 0.5) ** 2
    alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
    betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
    return torch.clip(betas, 0, 0.999)


def extract(a, t, x_shape):
    """Extract t-index values from buffer a for a batch."""
    batch_size = t.shape[0]
    out = a.gather(-1, t)
    return out.reshape(batch_size, *((1,) * (len(x_shape) - 1)))


# ============================================================
# SpatialMixSTE building blocks (copy tu D3DP/common/mixste.py)
# ============================================================

class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None,
                 act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class Attention(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None,
                 attn_drop=0., proj_drop=0.):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x, attn_bias=None):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads,
                                   C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        attn = (q @ k.transpose(-2, -1)) * self.scale
        if attn_bias is not None:
            attn = attn + attn_bias          # (B, H, 1, N): bias theo KEY (khop nguon)
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)
        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class Block(nn.Module):
    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, qk_scale=None,
                 drop=0., attn_drop=0., drop_path=0., act_layer=nn.GELU,
                 norm_layer=nn.LayerNorm):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = Attention(dim, num_heads=num_heads, qkv_bias=qkv_bias,
                              qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim,
                       act_layer=act_layer, drop=drop)

    def forward(self, x):
        x = x + self.drop_path(self.attn(self.norm1(x)))
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x


class AdaLNBlock(nn.Module):
    """
    Block Transformer co dieu kien adaLN THEO TUNG KHOP (kieu DiT, nhung vector dieu kien rieng moi token).
    c (N, 17, dim) = embedding 2D cua khop + pos + t -> shift/scale/gate cho attn va mlp.
    Lop sinh tham so khoi tao = 0 -> gate = 0 -> block ban dau la identity (train on dinh).
    Muc dich: 2D duoc dua vao MOI block (khong chi lop dau), mang luon doi chieu x_t voi 2D.
    """

    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, qk_scale=None,
                 drop=0., attn_drop=0., drop_path=0., act_layer=nn.GELU, conf_attn=False,
                 num_hops=0):
        super().__init__()
        # skel_attn: bias attention hoc duoc theo so canh tren khung xuong (kieu Graphormer), moi head 1 gia tri
        # cho moi khoang cach. Khoi tao 0 -> ban dau giong block cu.
        self.hop_bias = nn.Embedding(num_hops, num_heads) if num_hops > 0 else None
        if self.hop_bias is not None:
            nn.init.zeros_(self.hop_bias.weight)
        # conf_attn: moi head hoc 1 he so w_h, bias attention = w_h * log(conf cua khop KEY).
        # w_h khoi tao 0 -> ban dau giong het block cu. w_h > 0 -> khop score thap / bi che it duoc tham khao.
        self.conf_w = nn.Parameter(torch.zeros(num_heads)) if conf_attn else None
        self.norm1 = nn.LayerNorm(dim, elementwise_affine=False, eps=1e-6)
        self.attn = Attention(dim, num_heads=num_heads, qkv_bias=qkv_bias,
                              qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = nn.LayerNorm(dim, elementwise_affine=False, eps=1e-6)
        self.mlp = Mlp(in_features=dim, hidden_features=int(dim * mlp_ratio),
                       act_layer=act_layer, drop=drop)
        self.ada = nn.Sequential(nn.SiLU(), nn.Linear(dim, 6 * dim))
        nn.init.zeros_(self.ada[-1].weight)
        nn.init.zeros_(self.ada[-1].bias)

    def forward(self, x, c, log_conf=None, hops=None):
        sh1, sc1, g1, sh2, sc2, g2 = self.ada(c).chunk(6, dim=-1)    # moi cai (N, 17, dim)
        bias = None
        if self.conf_w is not None and log_conf is not None:
            bias = self.conf_w[None, :, None, None] * log_conf[:, None, None, :]   # (B, H, 1, 17)
        if self.hop_bias is not None and hops is not None:
            hb = self.hop_bias(hops).permute(2, 0, 1)[None]                         # (1, H, 17, 17)
            bias = hb if bias is None else bias + hb
        x = x + self.drop_path(g1 * self.attn(self.norm1(x) * (1 + sc1) + sh1, attn_bias=bias))
        x = x + self.drop_path(g2 * self.mlp(self.norm2(x) * (1 + sc2) + sh2))
        return x


# Cha cua tung khop H36M-17 (0 Pelvis, 1 RHip, 2 RKnee, 3 RAnk, 4 LHip, 5 LKnee, 6 LAnk, 7 Torso,
# 8 Neck, 9 Nose, 10 Head, 11 LSho, 12 LElb, 13 LWri, 14 RSho, 15 RElb, 16 RWri)
H36M_PARENTS = [-1, 0, 1, 2, 0, 4, 5, 0, 7, 8, 9, 8, 11, 12, 8, 14, 15]


def skeleton_hops(parents=H36M_PARENTS):
    """Ma tran so canh ngan nhat giua 2 khop tren khung xuong (J, J), long."""
    J = len(parents)
    adj = [[] for _ in range(J)]
    for j, p in enumerate(parents):
        if p >= 0:
            adj[j].append(p)
            adj[p].append(j)
    hops = torch.zeros(J, J, dtype=torch.long)
    for s in range(J):
        dist = [-1] * J
        dist[s] = 0
        q = [s]
        while q:
            u = q.pop(0)
            for v in adj[u]:
                if dist[v] < 0:
                    dist[v] = dist[u] + 1
                    q.append(v)
        hops[s] = torch.tensor(dist)
    return hops


class SinusoidalPositionEmbeddings(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, time):
        device = time.device
        half_dim = self.dim // 2
        embeddings = math.log(10000) / (half_dim - 1)
        embeddings = torch.exp(torch.arange(half_dim, device=device) * -embeddings)
        embeddings = time[:, None] * embeddings[None, :]
        embeddings = torch.cat((embeddings.sin(), embeddings.cos()), dim=-1)
        return embeddings.float()


# ============================================================
# SpatialMixSTE -- MixSTE2 bo toan bo Temporal branch
# ============================================================

class SpatialMixSTE(nn.Module):
    """
    Spatial-only Transformer denoiser.
    in_chans=3: [x_norm, y_norm, conf/mask]   (them 1 so voi D3DP goc in_chans=2)
    embed_dim_ratio: hidden dim
    depth: so STE blocks
    """

    def __init__(self, num_joints=17, in_chans=3, embed_dim_ratio=256, depth=4,
                 num_heads=8, mlp_ratio=2., qkv_bias=True, qk_scale=None,
                 drop_rate=0., attn_drop_rate=0., drop_path_rate=0.1,
                 norm_layer=None, cond_mode='input', conf_attn=False, skel_attn=False,
                 mask_token=False, self_cond=False):
        super().__init__()
        norm_layer = norm_layer or partial(nn.LayerNorm, eps=1e-6)
        # 'input' (mac dinh, hanh vi cu): 2D chi ghep voi x_t o lop embedding dau.
        # 'adaln': them 2D + t vao MOI block qua AdaLNBlock (theo tung khop).
        if cond_mode not in ('input', 'adaln'):
            raise ValueError(f"cond_mode khong hop le: {cond_mode}")
        self.cond_mode = cond_mode
        if conf_attn and cond_mode != 'adaln':
            raise ValueError("conf_attn chi dung voi cond_mode='adaln'")
        self.conf_attn = conf_attn
        if (skel_attn or mask_token) and cond_mode != 'adaln':
            raise ValueError("skel_attn / mask_token chi dung voi cond_mode='adaln'")
        self.skel_attn = skel_attn
        self.mask_token = mask_token
        # self_cond: token dau vao them 3 kenh = du doan x0 cua buoc DDIM truoc (0 o buoc dau / khi train khong dung)
        self.self_cond = self_cond

        # Input: concat(2D+conf, noisy_3D) = in_chans + 3
        self.Spatial_patch_to_embedding = nn.Linear(in_chans + 3 + (3 if self_cond else 0), embed_dim_ratio)
        # mask_token: vector hoc duoc cong vao token + dieu kien cua khop KHONG co 2D (conf = 0),
        # thay vi de mang tu hieu xy = (0, 0) (= tam crop, mot vi tri co that).
        if mask_token:
            self.mask_in = nn.Parameter(torch.zeros(1, 1, embed_dim_ratio))
            self.mask_cond = nn.Parameter(torch.zeros(1, 1, embed_dim_ratio))
            trunc_normal_(self.mask_in, std=.02)
            trunc_normal_(self.mask_cond, std=.02)
        if skel_attn:
            hops = skeleton_hops()
            self.register_buffer('hops', hops, persistent=False)
            n_hops = int(hops.max()) + 1
        else:
            n_hops = 0
        self.Spatial_pos_embed = nn.Parameter(torch.zeros(1, num_joints, embed_dim_ratio))

        self.pos_drop = nn.Dropout(p=drop_rate)

        # Diffusion timestep embedding (copy tu MixSTE2)
        self.time_mlp = nn.Sequential(
            SinusoidalPositionEmbeddings(embed_dim_ratio),
            nn.Linear(embed_dim_ratio, embed_dim_ratio * 2),
            nn.GELU(),
            nn.Linear(embed_dim_ratio * 2, embed_dim_ratio),
        )

        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]
        self.block_depth = depth

        # Chi Spatial Attention blocks (khong co TTEblocks)
        if cond_mode == 'adaln':
            # Embedding dieu kien rieng cho 2D (x, y, conf) -> vector dieu kien moi khop
            self.cond_embed = nn.Linear(in_chans, embed_dim_ratio)
            self.STEblocks = nn.ModuleList([
                AdaLNBlock(dim=embed_dim_ratio, num_heads=num_heads, mlp_ratio=mlp_ratio,
                           qkv_bias=qkv_bias, qk_scale=qk_scale, drop=drop_rate,
                           attn_drop=attn_drop_rate, drop_path=dpr[i], conf_attn=conf_attn,
                           num_hops=n_hops)
                for i in range(depth)
            ])
        else:
            self.STEblocks = nn.ModuleList([
                Block(dim=embed_dim_ratio, num_heads=num_heads, mlp_ratio=mlp_ratio,
                      qkv_bias=qkv_bias, qk_scale=qk_scale, drop=drop_rate,
                      attn_drop=attn_drop_rate, drop_path=dpr[i], norm_layer=norm_layer)
                for i in range(depth)
            ])
        self.Spatial_norm = norm_layer(embed_dim_ratio)

        # Output head
        self.head = nn.Sequential(
            nn.LayerNorm(embed_dim_ratio),
            nn.Linear(embed_dim_ratio, 3),
        )

        trunc_normal_(self.Spatial_pos_embed, std=.02)

    def forward(self, x_2d, x_3d, t, x_sc=None):
        """
        Train:  x_2d (B,17,3), x_3d (B,17,3),   t (B,)  -> (B,17,3)
        Infer:  x_2d (B,17,3), x_3d (B,K,17,3), t (B,)  -> (B,K,17,3)
        x_2d = [x_norm, y_norm, conf]
        x_sc: du doan x0 truoc do (cung shape x_3d, met) khi self_cond; None -> 0
        """
        inference_mode = x_3d.dim() == 4  # (B,K,17,3)
        if self.self_cond:
            x_3d = torch.cat([x_3d, torch.zeros_like(x_3d) if x_sc is None else x_sc], dim=-1)

        if inference_mode:
            B, K, N, _ = x_3d.shape
            x_2d_exp = x_2d.unsqueeze(1).expand(-1, K, -1, -1)    # (B,K,17,3)
            x = torch.cat([x_2d_exp, x_3d], dim=-1)               # (B,K,17,6)
            x = x.reshape(B * K, N, x.shape[-1])
            t_exp = t.unsqueeze(1).expand(-1, K).reshape(B * K)    # (B*K,)
            c2d = x_2d_exp.reshape(B * K, N, x_2d.shape[-1])       # (B*K,17,3) cho adaLN
        else:
            B, N, _ = x_3d.shape
            x = torch.cat([x_2d, x_3d], dim=-1)                   # (B,17,6)
            t_exp = t
            K = None
            c2d = x_2d

        # --- Spatial embedding ---
        x = self.Spatial_patch_to_embedding(x)           # (..., 17, embed_dim)
        missing = (c2d[..., 2:3] <= 0).to(x.dtype) if self.mask_token else None   # (..., 17, 1)
        if missing is not None:
            x = x + missing * self.mask_in
        x = x + self.Spatial_pos_embed
        time_embed = self.time_mlp(t_exp)                # (..., embed_dim)
        x = x + time_embed[:, None, :]                   # broadcast len 17 joints
        x = self.pos_drop(x)

        # --- Chi Spatial Attention (khong TTE) ---
        if self.cond_mode == 'adaln':
            # Vector dieu kien theo tung khop: 2D cua khop + vi tri khop + t
            c = self.cond_embed(c2d) + self.Spatial_pos_embed + time_embed[:, None, :]
            # log(conf) cua tung khop (conf = 0 -> log(1e-3) ~ -6.9): dung lam bias attention neu conf_attn
            if missing is not None:
                c = c + missing * self.mask_cond
            log_conf = torch.log(c2d[..., 2].clamp(min=1e-3)) if self.conf_attn else None
            hops = self.hops if self.skel_attn else None
            for blk in self.STEblocks:
                x = blk(x, c, log_conf, hops)
        else:
            for blk in self.STEblocks:
                x = blk(x)
        x = self.Spatial_norm(x)

        # --- Head ---
        x = self.head(x)   # (..., 17, 3)

        if inference_mode:
            x = x.reshape(B, K, N, 3)
        return x


# ============================================================
# DiffLifter -- Diffusion wrapper (D3DP adapted, single-frame)
# ============================================================

def aggregate_jpma(all_preds, xy_conf):
    """
    Gop K gia thuyet THEO TUNG KHOP bang do khop 2D (y tuong JPMA cua D3DP, ban weak-perspective).
      all_preds: (B, K, J, 3) root-relative, met
      xy_conf:   (B, J, 3) [x, y, conf] toa do crop [-1, 1]
    1) Khop scale s + offset t (least squares, chi tren khop nhin ro) tu (X, Y) cua pose TRUNG BINH sang 2D.
       Dung chung (s, t) cho moi gia thuyet (giong 1 camera chung).
    2) Moi khop nhin ro: chon gia thuyet co loi chieu 2D nho nhat.
       Khop mat (conf = 0): khong co 2D de doi chieu -> lay trung binh K.
    Khong can thong so camera. Tra ve (B, J, 3) root-relative.
    """
    B, K, J, _ = all_preds.shape
    mean = all_preds.mean(dim=1)                                        # (B, J, 3)
    if K == 1:
        return mean
    p2d = xy_conf[..., :2]                                              # (B, J, 2)
    w = (xy_conf[..., 2] > 0).float()                                   # (B, J)
    P = mean[..., :2]                                                   # (B, J, 2)
    wsum = w.sum(dim=1).clamp(min=1.0)[:, None, None]                   # (B, 1, 1)
    Pm = (w[..., None] * P).sum(dim=1, keepdim=True) / wsum             # (B, 1, 2)
    pm = (w[..., None] * p2d).sum(dim=1, keepdim=True) / wsum           # (B, 1, 2)
    Pc, pc = P - Pm, p2d - pm
    num = (w * (Pc * pc).sum(-1)).sum(dim=1)                            # (B,)
    den = (w * (Pc ** 2).sum(-1)).sum(dim=1).clamp(min=1e-8)           # (B,)
    s = (num / den)[:, None, None]                                      # (B, 1, 1)
    t = pm - s * Pm                                                     # (B, 1, 2)
    proj = s[:, None] * all_preds[..., :2] + t[:, None]                 # (B, K, J, 2)
    err = torch.norm(proj - p2d[:, None], dim=-1)                       # (B, K, J)
    best = err.argmin(dim=1)                                            # (B, J)
    idx = best[:, None, :, None].expand(B, 1, J, 3)
    sel = torch.gather(all_preds, 1, idx).squeeze(1)                    # (B, J, 3)
    out = torch.where((w > 0)[..., None], sel, mean)
    return out - out[:, 0:1, :]


class DiffLifter(nn.Module):
    """
    Single-frame 2D->3D Diffusion Lifter.
    Tai dung diffusion framework cua D3DP (pred_x0, cosine schedule, DDIM).
    Backbone: SpatialMixSTE (bo Temporal branch cua MixSTE2).

    Args:
        num_joints:        so khop (17 cho H36M)
        embed_dim:         chieu an cua Transformer (mac dinh 256)
        depth:             so Spatial Attention blocks (mac dinh 4)
        num_heads:         so attention heads
        timesteps:         T diffusion timesteps khi train (mac dinh 1000)
        sampling_timesteps: S DDIM steps khi inference (mac dinh 5)
        num_proposals:     K hypotheses khi inference (mac dinh 5)
        scale:             he so scale 3D truoc khi them noise (mac dinh 1.0)
        drop_path_rate:    stochastic depth rate
    """

    def __init__(self, num_joints=17, embed_dim=256, depth=4, num_heads=8,
                 timesteps=1000, sampling_timesteps=5, num_proposals=5,
                 scale=1.0, drop_path_rate=0.1, cond_mode='input', aggregate='mean', conf_attn=False,
                 drop_rate=0.0,
                 skel_attn=False, mask_token=False, self_cond=False):
        super().__init__()

        self.num_joints = num_joints
        self.num_proposals = num_proposals
        self.scale = scale
        # Cach gop K gia thuyet luc inference: 'mean' (mac dinh, hanh vi cu) | 'jpma' (xem aggregate_jpma)
        if aggregate not in ('mean', 'jpma'):
            raise ValueError(f"aggregate khong hop le: {aggregate}")
        self.aggregate = aggregate

        # --- Diffusion schedule (giong D3DP hoan toan) ---
        betas = cosine_beta_schedule(timesteps)
        alphas = 1. - betas
        alphas_cumprod = torch.cumprod(alphas, dim=0)
        alphas_cumprod_prev = F.pad(alphas_cumprod[:-1], (1, 0), value=1.)

        self.num_timesteps = int(betas.shape[0])
        self.sampling_timesteps = sampling_timesteps
        assert self.sampling_timesteps <= self.num_timesteps
        self.ddim_sampling_eta = 1.0  # stochastic DDIM (eta=1 ~ DDPM)

        # Tinh bang float64 roi ep ve float32 (giong D3DP register_buffer .to(float32)).
        # Neu giu float64: extract() -> x_t float64 -> Linear float32 bao loi dtype.
        def _reg(name, val):
            self.register_buffer(name, val.to(torch.float32))

        _reg('betas', betas)
        _reg('alphas_cumprod', alphas_cumprod)
        _reg('alphas_cumprod_prev', alphas_cumprod_prev)
        _reg('sqrt_alphas_cumprod', torch.sqrt(alphas_cumprod))
        _reg('sqrt_one_minus_alphas_cumprod', torch.sqrt(1. - alphas_cumprod))
        _reg('sqrt_recip_alphas_cumprod', torch.sqrt(1. / alphas_cumprod))
        _reg('sqrt_recipm1_alphas_cumprod', torch.sqrt(1. / alphas_cumprod - 1))

        # --- Denoiser backbone ---
        self.denoiser = SpatialMixSTE(
            num_joints=num_joints,
            in_chans=3,               # [x_norm, y_norm, conf]
            embed_dim_ratio=embed_dim,
            depth=depth,
            num_heads=num_heads,
            drop_path_rate=drop_path_rate,
            drop_rate=drop_rate,
            cond_mode=cond_mode,
            conf_attn=conf_attn,
            skel_attn=skel_attn,
            mask_token=mask_token,
            self_cond=self_cond,
        )
        self.self_cond = self_cond

    # ------------------------------------------------------------------
    # Forward diffusion: q(x_t | x_0)
    # ------------------------------------------------------------------
    def q_sample(self, x_start, t, noise=None):
        if noise is None:
            noise = torch.randn_like(x_start)
        sqrt_alpha = extract(self.sqrt_alphas_cumprod, t, x_start.shape)
        sqrt_one_minus = extract(self.sqrt_one_minus_alphas_cumprod, t, x_start.shape)
        return sqrt_alpha * x_start + sqrt_one_minus * noise

    def predict_noise_from_x0(self, x_t, t, x0):
        return (
            extract(self.sqrt_recip_alphas_cumprod, t, x_t.shape) * x_t - x0
        ) / extract(self.sqrt_recipm1_alphas_cumprod, t, x_t.shape)

    # ------------------------------------------------------------------
    # Goi denoiser: tra ve pred_x0 don vi MET, root-relative (khop 0 = 0)
    # ------------------------------------------------------------------
    def _denoise(self, xy_conf, x_in, t, x_sc=None):
        """
        x_in: x_t da clamp va chia scale (giong D3DP model_predictions).
        Output tho cua denoiser duoc hieu la toa do met (giong D3DP: loss so truc tiep voi GT met).
        Tru root giong PMCE (pred - pred[:, :1]) -> GT root-relative, khop Pelvis luon = 0.
        """
        pred = self.denoiser(xy_conf, x_in, t, x_sc)
        return pred - pred[..., 0:1, :]

    # ------------------------------------------------------------------
    # Training forward
    # ------------------------------------------------------------------
    def forward_train(self, xy_conf, pose_3d):
        """
        xy_conf: (B, 17, 3)  [x_norm, y_norm, conf]  -- dau vao 2D + occlusion
        pose_3d: (B, 17, 3)  GT 3D root-relative, met
        Returns: pred_x0 (B, 17, 3)  -- du doan toa do 3D
        """
        B = pose_3d.shape[0]
        device = pose_3d.device

        # GT root-relative -> dam bao root = 0 (phong khi dau vao chua tru root)
        pose_3d = pose_3d - pose_3d[:, 0:1, :]
        x0 = pose_3d * self.scale
        x0 = x0.clamp(-1.1 * self.scale, 1.1 * self.scale)

        # Random timestep moi sample (giong D3DP prepare_targets)
        t = torch.randint(0, self.num_timesteps, (B,), device=device).long()
        noise = torch.randn_like(x0)
        x_t = self.q_sample(x0, t, noise)
        x_in = x_t.clamp(-1.1 * self.scale, 1.1 * self.scale) / self.scale

        # Self-conditioning (Chen et al. 2022): 50% batch, tinh x0 lan 1 (khong grad) roi dua vao lam goi y.
        # Luc inference, goi y = x0 du doan o buoc DDIM truoc -> cac buoc sau co the sua dan.
        x_sc = None
        if self.self_cond and torch.rand(()) < 0.5:
            with torch.no_grad():
                x_sc = self._denoise(xy_conf, x_in, t).detach()
        # Output da la met (KHONG nhan scale): truoc day nhan scale -> sai don vi khi scale != 1
        pred_x0 = self._denoise(xy_conf, x_in, t, x_sc)    # (B, 17, 3) met, root-relative
        return pred_x0

    # ------------------------------------------------------------------
    # Inference: DDIM sampling, K hypotheses song song
    # ------------------------------------------------------------------
    @torch.no_grad()
    def forward_inference(self, xy_conf, generator=None):
        """
        xy_conf: (B, 17, 3)
        generator: torch.Generator (tuy chon) -> nhieu co dinh, eval lap lai duoc giua cac epoch
        Returns:
            mean_pose: (B, 17, 3)  -- trung binh K hypotheses, root-rel, met
            spread:    (B, 17)     -- std uncertainty moi khop
            all_preds: (B, K, 17, 3)
        """
        B = xy_conf.shape[0]
        K = self.num_proposals
        device = xy_conf.device

        T = self.num_timesteps
        S = self.sampling_timesteps
        eta = self.ddim_sampling_eta

        # DDIM time pairs: T -> 0 trong S buoc
        times = torch.linspace(-1, T - 1, steps=S + 1)
        times = list(reversed(times.int().tolist()))
        time_pairs = list(zip(times[:-1], times[1:]))

        # Khoi tao K hypotheses song song: (B, K, 17, 3)
        x = torch.randn(B, K, self.num_joints, 3, device=device, generator=generator)

        x_sc = None    # self_cond: du doan x0 cua buoc truoc
        for time, time_next in time_pairs:
            t_cond = torch.full((B,), time, device=device, dtype=torch.long)

            # Denoiser forward: x (B,K,17,3) -> pred_x0 (B,K,17,3)
            x_in = x.clamp(-1.1 * self.scale, 1.1 * self.scale) / self.scale
            pred_x0 = self._denoise(xy_conf, x_in, t_cond, x_sc)    # (B,K,17,3) met, root-rel
            if self.self_cond:
                x_sc = pred_x0
            pred_x0_scaled = (pred_x0 * self.scale).clamp(
                -1.1 * self.scale, 1.1 * self.scale)

            if time_next < 0:
                x = pred_x0_scaled
                break

            alpha = self.alphas_cumprod[time]
            alpha_next = self.alphas_cumprod[time_next]
            sigma = eta * ((1 - alpha / alpha_next) * (1 - alpha_next) / (1 - alpha)).clamp(min=0).sqrt()
            # clamp(min=0): khi alpha ~ 0 (t=999) va eta=1, 1 - alpha_next - sigma^2 ~ 1e-8 co the
            # am nho do lam tron float32 -> sqrt ra NaN (gap o S=3). Ve ly thuyet bieu thuc nay >= 0.
            c = (1 - alpha_next - sigma ** 2).clamp(min=0).sqrt()

            # pred_noise tu pred_x0 (giong D3DP predict_noise_from_start)
            pred_noise = (x - pred_x0_scaled * alpha.sqrt()) / (1 - alpha + 1e-8).sqrt()

            noise = torch.randn(x.shape, device=device, dtype=x.dtype, generator=generator)
            x = pred_x0_scaled * alpha_next.sqrt() + c * pred_noise + sigma * noise

        all_preds = x / self.scale              # (B, K, 17, 3) -- met
        # Pose dau ra: trung binh K (mac dinh) hoac chon theo khop 2D (aggregate='jpma')
        if self.aggregate == 'jpma':
            mean_pose = aggregate_jpma(all_preds, xy_conf)    # (B, 17, 3)
        else:
            mean_pose = all_preds.mean(dim=1)                 # (B, 17, 3)
        # K=1: std khong xac dinh (NaN + warning) -> spread = 0
        if K > 1:
            spread = all_preds.std(dim=1).norm(dim=-1)  # (B, 17)
        else:
            spread = torch.zeros(all_preds.shape[0], self.num_joints, device=device)
        return mean_pose, spread, all_preds

    # ------------------------------------------------------------------
    # Unified forward
    # ------------------------------------------------------------------
    def forward(self, xy_conf, pose_3d=None, generator=None):
        """
        Train: forward(xy_conf, pose_3d) -> pred_x0 (B,17,3) met, root-relative
        Eval:  forward(xy_conf)          -> (mean_pose, spread, all_preds), root-relative
        """
        if self.training:
            assert pose_3d is not None, "pose_3d bat buoc khi training"
            return self.forward_train(xy_conf, pose_3d)
        else:
            return self.forward_inference(xy_conf, generator=generator)
