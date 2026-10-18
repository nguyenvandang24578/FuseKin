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

__all__ = ["DiffLifter"]

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

    def forward(self, x):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads,
                                   C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        attn = (q @ k.transpose(-2, -1)) * self.scale
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
                 norm_layer=None):
        super().__init__()
        norm_layer = norm_layer or partial(nn.LayerNorm, eps=1e-6)

        # Input: concat(2D+conf, noisy_3D) = in_chans + 3
        self.Spatial_patch_to_embedding = nn.Linear(in_chans + 3, embed_dim_ratio)
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

    def forward(self, x_2d, x_3d, t):
        """
        Train:  x_2d (B,17,3), x_3d (B,17,3),   t (B,)  -> (B,17,3)
        Infer:  x_2d (B,17,3), x_3d (B,K,17,3), t (B,)  -> (B,K,17,3)
        x_2d = [x_norm, y_norm, conf]
        """
        inference_mode = x_3d.dim() == 4  # (B,K,17,3)

        if inference_mode:
            B, K, N, _ = x_3d.shape
            x_2d_exp = x_2d.unsqueeze(1).expand(-1, K, -1, -1)    # (B,K,17,3)
            x = torch.cat([x_2d_exp, x_3d], dim=-1)               # (B,K,17,6)
            x = x.reshape(B * K, N, 6)
            t_exp = t.unsqueeze(1).expand(-1, K).reshape(B * K)    # (B*K,)
        else:
            B, N, _ = x_3d.shape
            x = torch.cat([x_2d, x_3d], dim=-1)                   # (B,17,6)
            t_exp = t
            K = None

        # --- Spatial embedding ---
        x = self.Spatial_patch_to_embedding(x)           # (..., 17, embed_dim)
        x = x + self.Spatial_pos_embed
        time_embed = self.time_mlp(t_exp)                # (..., embed_dim)
        x = x + time_embed[:, None, :]                   # broadcast len 17 joints
        x = self.pos_drop(x)

        # --- Chi Spatial Attention (khong TTE) ---
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
                 scale=1.0, drop_path_rate=0.1):
        super().__init__()

        self.num_joints = num_joints
        self.num_proposals = num_proposals
        self.scale = scale

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
        )

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
    def _denoise(self, xy_conf, x_in, t):
        """
        x_in: x_t da clamp va chia scale (giong D3DP model_predictions).
        Output tho cua denoiser duoc hieu la toa do met (giong D3DP: loss so truc tiep voi GT met).
        Tru root giong PMCE (pred - pred[:, :1]) -> GT root-relative, khop Pelvis luon = 0.
        """
        pred = self.denoiser(xy_conf, x_in, t)
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

        # Output da la met (KHONG nhan scale): truoc day nhan scale -> sai don vi khi scale != 1
        pred_x0 = self._denoise(xy_conf, x_in, t)    # (B, 17, 3) met, root-relative
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

        for time, time_next in time_pairs:
            t_cond = torch.full((B,), time, device=device, dtype=torch.long)

            # Denoiser forward: x (B,K,17,3) -> pred_x0 (B,K,17,3)
            x_in = x.clamp(-1.1 * self.scale, 1.1 * self.scale) / self.scale
            pred_x0 = self._denoise(xy_conf, x_in, t_cond)    # (B,K,17,3) met, root-rel
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
        mean_pose = all_preds.mean(dim=1)       # (B, 17, 3)
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
