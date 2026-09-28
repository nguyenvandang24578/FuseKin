"""
SMPL_HyperDiff (v2): Conditional Diffusion with Hypergraph for SMPL 6D Pose.

Replaces HYPERGCv2 + VPoser in Pose2Mesh with a denoising diffusion model
that directly predicts 24x6D rotation parameters conditioned on 2D keypoints.

Architecture
============
  x0 = GT 24x6D rotations (absolute, not residual)
  Forward diffusion: q(x_t | x0) with cosine schedule, T=1000
  Denoiser per layer:
      adaLN-Zero HyperGCN -> [optional joint self-attn] -> CrossAttn(kp2D)
      -> adaLN-Zero FFN
  Inference: DDIM (default 10 steps) from pure noise, optional multi-sample
             averaging.

Changes vs. v1  (tags [Fx] are used in comments below)
======================================================
  [F1] apply_big_errors rewritten: swap and jump are now disjoint events,
       big_err mask is exact.
  [F2] Context attention: enable_context_attn() is defined; missing
       initialisation raises instead of spamming warnings.
  [F3] train/eval unified on self.training; x0 optional; ddim_sample forces
       eval mode (and restores it).
  [F4] Bone feature is zeroed when the parent joint is missing.
  [F5] Extra losses: geodesic / L1 on rotation matrices (Gram-Schmidt),
       optional forward-kinematics joint loss, optional Min-SNR weighting.
       Geometric losses only apply to small-t samples.
  [F6] Multi-sample DDIM (num_samples) + generator used for every random draw.
  [F7] LayerScale replaced by adaLN-Zero gates; redundant t_emb add removed.
  [F8] Self-conditioning (train + DDIM).
  [F9] Ablation switches: use_hyper (hyperedges on/off), use_joint_sa.
  [F10] Pelvis-relative keypoint feature added to the tokenizer.
  [F11] Misc: dead buffers removed, dim asserts, timm.layers import, EMA
        helper, param groups without weight-decay for gates/scalars/embeds.

NOTE: intentionally NOT applied -> clamping x0 to [-1, 1] during sampling.
"""

import copy
import math
import warnings

import torch
import torch.nn as nn
import torch.nn.functional as F

try:                                   # [F11] timm>=0.9
    from timm.layers import DropPath
except ImportError:                    # older timm
    from timm.models.layers import DropPath

from core.config import cfg
from models.Core_model import CrossAttentionBlock


def _cfg(name, default):
    """Read optional cfg.DIFF.<name>, fall back to default if missing."""
    return getattr(cfg.DIFF, name, default)


# ====================================================================
# Constants
# ====================================================================

# SMPL 24-joint kinematic tree  (parent index, -1 = root)
SMPL_PARENT = [
    -1,  0,  0,  0,  1,  2,  3,  4,  5,  6,
     7,  8,  9,  9,  9, 12, 13, 14, 16, 17,
    18, 19, 20, 21,
]

# H36M 17-joint kinematic tree
# 0:Pelvis 1:R_Hip 2:R_Knee 3:R_Ankle 4:L_Hip 5:L_Knee 6:L_Ankle
# 7:Torso 8:Neck 9:Nose 10:Head 11:L_Shoulder 12:L_Elbow 13:L_Wrist
# 14:R_Shoulder 15:R_Elbow 16:R_Wrist
H36M_PARENT = [-1, 0, 1, 2, 0, 4, 5, 0, 7, 8, 9, 8, 11, 12, 8, 14, 15]

# Left-right swap pairs for big-error noise (H36M indexing)
H36M_SWAP_PAIRS = [(1, 4), (2, 5), (3, 6), (11, 14), (12, 15), (13, 16)]

# Limb groups for correlated noise
H36M_LIMB_GROUPS = [
    [11, 12, 13],   # Left arm
    [14, 15, 16],   # Right arm
    [4,  5,  6],    # Left leg
    [1,  2,  3],    # Right leg
]

# Confidence below this is treated as "keypoint missing"
CONF_EPS = 1e-3


# ====================================================================
# Rotation helpers
# ====================================================================

def axis_angle_to_rotmat(aa):
    """Rodrigues: (*, 3) axis-angle -> (*, 3, 3) rotation matrix.

    For theta ~ 0 the axis becomes ~0, K ~ 0 and R -> I automatically,
    so no special-case branch is needed.
    """
    orig_shape = aa.shape[:-1]
    aa = aa.reshape(-1, 3)
    theta = aa.norm(dim=-1, keepdim=True).clamp(min=1e-8)       # (N, 1)
    axis = aa / theta
    cos_t = torch.cos(theta).unsqueeze(-1)                       # (N, 1, 1)
    sin_t = torch.sin(theta).unsqueeze(-1)
    kx, ky, kz = axis.unbind(-1)                                 # (N,)
    zeros = torch.zeros_like(kx)
    K = torch.stack([zeros, -kz, ky,
                     kz, zeros, -kx,
                     -ky, kx, zeros], dim=-1).reshape(-1, 3, 3)
    I = torch.eye(3, device=aa.device, dtype=aa.dtype).unsqueeze(0)
    R = I + sin_t * K + (1 - cos_t) * (K @ K)
    return R.reshape(*orig_shape, 3, 3)


def axis_angle_to_rot6d(aa):
    """(*, 3) axis-angle -> (*, 6) rot6d.

    Convention: 6D = first two **columns** of the rotation matrix flattened
    row-major -> [r00, r01, r10, r11, r20, r21]. Matches
    ``rot6d_to_axis_angle()`` in ``utils/transforms.py`` (``.view(-1,3,2)``).
    """
    orig_shape = aa.shape[:-1]
    R = axis_angle_to_rotmat(aa.reshape(-1, 3))
    rot6d = R[:, :, :2].reshape(-1, 6)
    return rot6d.reshape(*orig_shape, 6)


def rot6d_to_rotmat(x):
    """(*, 6) -> (*, 3, 3) via Gram-Schmidt (same layout as above)."""
    shp = x.shape[:-1]
    m = x.reshape(-1, 3, 2)
    a1, a2 = m[:, :, 0], m[:, :, 1]
    b1 = F.normalize(a1, dim=-1, eps=1e-6)
    b2 = F.normalize(a2 - (b1 * a2).sum(-1, keepdim=True) * b1,
                     dim=-1, eps=1e-6)
    b3 = torch.cross(b1, b2, dim=-1)
    R = torch.stack([b1, b2, b3], dim=-1)                        # columns
    return R.reshape(*shp, 3, 3)


def geodesic_distance(R1, R2):
    """Angle (radians) between two batches of rotation matrices."""
    Rt = R1.float() @ R2.float().transpose(-1, -2)
    tr = Rt.diagonal(dim1=-2, dim2=-1).sum(-1)
    cos = ((tr - 1.0) * 0.5).clamp(-1 + 1e-6, 1 - 1e-6)
    return torch.acos(cos)


# ====================================================================
# Hypergraph construction
# ====================================================================

def build_smpl_kinematic_adj(num_joints=24):
    """Symmetric-normalised adjacency from SMPL kinematic tree."""
    adj = torch.zeros(num_joints, num_joints)
    for i, p in enumerate(SMPL_PARENT):
        adj[i, i] = 1.0
        if p >= 0:
            adj[i, p] = 1.0
            adj[p, i] = 1.0
    deg = adj.sum(dim=-1).clamp(min=1e-6)
    d_inv_sqrt = deg.pow(-0.5)
    return d_inv_sqrt.unsqueeze(-1) * adj * d_inv_sqrt.unsqueeze(0)


def _build_H_part(n=24):
    """10 part-level hyperedges  (from design_smpl_hyperdiff.md §2)."""
    H = torch.zeros(n, 10)
    edges = {
        0: [0, 3, 6],              # Lower spine
        1: [9, 12, 13, 14],        # Upper spine
        2: [12, 15],               # Head
        3: [13, 16, 18],           # L arm upper
        4: [18, 20, 22],           # L arm lower
        5: [14, 17, 19],           # R arm upper
        6: [19, 21, 23],           # R arm lower
        7: [1, 4, 7, 10],          # L leg
        8: [2, 5, 8, 11],          # R leg
        9: [0, 1, 2],              # Hip
    }
    for ei, joints in edges.items():
        for j in joints:
            H[j, ei] = 1.0
    return H


def _build_H_body(n=24):
    """5 body-level hyperedges  (from design_smpl_hyperdiff.md §2)."""
    H = torch.zeros(n, 5)
    edges = {
        0: [0, 3, 6, 9, 12, 13, 14, 15],   # Torso + Head
        1: [13, 16, 18, 20, 22],             # L arm
        2: [14, 17, 19, 21, 23],             # R arm
        3: [1, 4, 7, 10],                    # L leg
        4: [2, 5, 8, 11],                    # R leg
    }
    for ei, joints in edges.items():
        for j in joints:
            H[j, ei] = 1.0
    return H


def _compute_G(H, eps=1e-8):
    """G = Dv^{-1/2}  H  De^{-1}  H^T  Dv^{-1/2}"""
    Dv = H.sum(dim=-1) + eps       # (N,)
    De = H.sum(dim=0) + eps        # (E,)
    H_De = H * De.pow(-1).unsqueeze(0)
    G = H_De @ H.T
    d = Dv.pow(-0.5)
    return d.unsqueeze(-1) * G * d.unsqueeze(0)


# ====================================================================
# Diffusion schedule helpers
# ====================================================================

def cosine_beta_schedule(T, s=0.008):
    """Nichol & Dhariwal 2021 cosine schedule."""
    x = torch.linspace(0, T, T + 1, dtype=torch.float64)
    ac = torch.cos(((x / T) + s) / (1 + s) * math.pi * 0.5) ** 2
    ac = ac / ac[0]
    betas = 1.0 - ac[1:] / ac[:-1]
    return betas.clamp(0.0001, 0.9999).float()


def _extract(a, t, x_shape):
    """Gather from schedule *a* at timestep *t*, reshape broadcastable."""
    out = a.gather(-1, t)
    return out.reshape(t.shape[0], *((1,) * (len(x_shape) - 1)))


# ====================================================================
# 2D keypoint noise augmentation
# ====================================================================

def limb_correlated_noise(kp, limb_sigma):
    """Shared displacement per limb group -> (B,17,2)."""
    noise = torch.zeros_like(kp)
    for group in H36M_LIMB_GROUPS:
        disp = torch.randn(kp.shape[0], 1, 2,
                           device=kp.device, dtype=kp.dtype) * limb_sigma
        noise[:, group] = noise[:, group] + disp        # broadcast over group
    return noise


def apply_big_errors(kp, p_big_err):
    """Simulate gross detector failures.   [F1]

    Two *disjoint* failure types, each joint hit with prob ~ p_big_err / 2
    (so the total per-joint error rate is ~ p_big_err):

      * left/right swap : decided per SYMMETRIC PAIR; both joints of the pair
                          exchange positions.
      * random jump     : decided per joint; only joints that were NOT
                          swapped can jump (no overwriting).

    Joints that are neither in a swap pair nor selected for a jump are never
    flagged, so the returned mask is exact.

    Returns: (kp_out, big_err_mask)  shapes (B,J,2), (B,J) bool
    """
    B, J, _ = kp.shape
    dev = kp.device
    p_each = 0.5 * p_big_err

    kp_out = kp.clone()
    big_err = torch.zeros(B, J, dtype=torch.bool, device=dev)

    # ---- 1) swaps (pairs are disjoint, so reading from `kp` is safe) ----
    swap_flag = torch.rand(B, len(H36M_SWAP_PAIRS), device=dev) < p_each
    for k, (l, r) in enumerate(H36M_SWAP_PAIRS):
        m = swap_flag[:, k]
        mm = m.unsqueeze(-1)
        kp_out[:, l] = torch.where(mm, kp[:, r], kp[:, l])
        kp_out[:, r] = torch.where(mm, kp[:, l], kp[:, r])
        big_err[:, l] |= m
        big_err[:, r] |= m

    # ---- 2) jumps, only for joints that were not swapped ----
    jump = (torch.rand(B, J, device=dev) < p_each) & ~big_err
    rand_pos = torch.rand(B, J, 2, device=dev, dtype=kp.dtype) * 2 - 1
    kp_out = torch.where(jump.unsqueeze(-1), rand_pos, kp_out)
    big_err = big_err | jump
    return kp_out, big_err


def make_noisy_kp2d(kp_gt, conf_gt, noise_cfg=None, warmup_scale=1.0):
    """Simulate detector noise on GT 2D keypoints for training.

    Args
    ----
    kp_gt     : (B, 17, 2)  clean GT keypoints in [-1, 1]
    conf_gt   : (B, 17)     binary confidence {0, 1}
    noise_cfg : edict - from ``cfg.DIFF.NOISE``
    warmup_scale : float - curriculum multiplier (0.3 -> 1.0)

    Returns
    -------
    kp, conf, drop_mask    (all same shapes as inputs)
    """
    if noise_cfg is None:
        noise_cfg = cfg.DIFF.NOISE

    sigma = torch.tensor(noise_cfg.sigma_per_joint,
                         device=kp_gt.device, dtype=kp_gt.dtype)
    sigma = sigma[None, :, None] * warmup_scale         # (1,17,1)

    valid = conf_gt > 0.5                                # (B,17)

    # --- per-joint Gaussian noise + limb-correlated noise ---
    noise = torch.randn_like(kp_gt) * sigma
    noise = noise + limb_correlated_noise(
        kp_gt, noise_cfg.limb_sigma * warmup_scale)
    kp = kp_gt + noise

    # --- big errors (swap / jump) ---
    kp, big_err = apply_big_errors(kp, noise_cfg.p_big_err)

    # --- random drop (+ always drop invalid GT joints) ---
    drop = (torch.rand_like(conf_gt) < noise_cfg.p_drop) | ~valid

    # --- confidence ---
    if noise_cfg.use_continuous_conf:
        err = (kp - kp_gt).norm(dim=-1)
        conf = (1.0 - noise_cfg.k * err
                + 0.05 * torch.randn_like(err)).clamp(0, 1)
        # detectors are often over-confident on gross errors
        conf = torch.where(big_err, conf.clamp(min=0.6), conf)
    else:
        conf = (~drop).float()

    kp = kp.masked_fill(drop.unsqueeze(-1), 0.0)
    conf = conf.masked_fill(drop, 0.0)
    return kp, conf, drop


# ====================================================================
# Sub-modules
# ====================================================================

class SinusoidalTimeEmbed(nn.Module):
    """Sinusoidal positional embedding for timestep."""
    def __init__(self, dim):
        super().__init__()
        assert dim % 2 == 0 and dim >= 4, "time-embed dim must be even (>=4)"
        self.dim = dim

    def forward(self, t):
        half = self.dim // 2
        emb = math.log(10_000) / (half - 1)
        emb = torch.exp(torch.arange(half, device=t.device) * -emb)
        emb = t[:, None].float() * emb[None, :]
        return torch.cat([emb.sin(), emb.cos()], dim=-1)     # (B, dim)


class KeypointTokenizer(nn.Module):
    """[x, y, conf, bone(2), pelvis-rel(2)] -> Linear -> dim + joint PE.

    [F4] bone  = kp - kp[parent], zeroed if the joint is root OR its parent
         is missing (conf <= CONF_EPS).
    [F10] pelvis-rel = kp - kp[pelvis], zeroed if the pelvis is missing.
    """
    KP_INPUT_DIM = 7

    def __init__(self, dim, num_kp=17):
        super().__init__()
        self.proj = nn.Linear(self.KP_INPUT_DIM, dim)
        self.joint_embed = nn.Embedding(num_kp, dim)
        # learned "no-keypoint" token for condition dropout
        self.no_kp_token = nn.Parameter(torch.randn(1, num_kp, dim) * 0.02)
        parent = torch.tensor([max(0, p) for p in H36M_PARENT],
                              dtype=torch.long)
        rmask = torch.tensor([float(p >= 0) for p in H36M_PARENT])
        self.register_buffer('parent_idx', parent)
        self.register_buffer('root_mask', rmask)

    def forward(self, kp2d, conf):
        """kp2d : (B,17,2)   conf : (B,17)  ->  (B,17,dim)"""
        valid = (conf > CONF_EPS).to(kp2d.dtype)                     # (B,17)

        parent_ok = valid[:, self.parent_idx] * self.root_mask[None]  # (B,17)
        bone = (kp2d - kp2d[:, self.parent_idx]) * parent_ok.unsqueeze(-1)

        pelvis_ok = valid[:, 0:1].unsqueeze(-1)                       # (B,1,1)
        rel = (kp2d - kp2d[:, 0:1]) * pelvis_ok * valid.unsqueeze(-1)

        feat = torch.cat([kp2d, conf.unsqueeze(-1), bone, rel], dim=-1)
        return self.proj(feat) + self.joint_embed.weight.unsqueeze(0)

    @staticmethod
    def compute_attn_bias(conf, num_heads):
        """log(conf + eps) -> (B, H, 1, 17) additive attention bias."""
        bias = torch.log(conf + CONF_EPS)                            # (B,17)
        return bias[:, None, None, :].expand(-1, num_heads, -1, -1)


# ====================================================================
# SMPL HyperGCN block  (spatial + part + body)
# ====================================================================

class SMPL_HyperGCNBlock(nn.Module):
    def __init__(self, dim, num_joints=24, use_hyper=True):
        super().__init__()
        self.use_hyper = use_hyper           # [F9] ablation switch
        self.register_buffer('A_spatial', build_smpl_kinematic_adj(num_joints))
        self.conv_s = nn.Linear(dim, dim)
        self.a1 = nn.Parameter(torch.tensor(1.0))
        if use_hyper:
            self.register_buffer('G_part', _compute_G(_build_H_part(num_joints)))
            self.register_buffer('G_body', _compute_G(_build_H_body(num_joints)))
            self.conv_p = nn.Linear(dim, dim)
            self.conv_b = nn.Linear(dim, dim)
            self.a2 = nn.Parameter(torch.tensor(1.0))
            self.a3 = nn.Parameter(torch.tensor(1.0))

    def forward(self, x):
        """x : (B, 24, dim) -> (B, 24, dim)"""
        out = F.softplus(self.a1) * self.conv_s(self.A_spatial @ x)
        if self.use_hyper:
            out = out + F.softplus(self.a2) * self.conv_p(self.G_part @ x)
            out = out + F.softplus(self.a3) * self.conv_b(self.G_body @ x)
        return out


# ====================================================================
# Denoiser layer
# ====================================================================

def _make_adaln_zero(dim):
    """time-emb -> (shift, scale, gate); zero-init => layer starts as identity."""
    lin = nn.Linear(dim, dim * 3)
    nn.init.zeros_(lin.weight)
    nn.init.zeros_(lin.bias)
    return nn.Sequential(nn.SiLU(), lin)


def _modulate(x, norm, ada, t_emb):
    shift, scale, gate = ada(t_emb).chunk(3, dim=-1)                 # (B,dim)
    h = norm(x) * (1 + scale[:, None, :]) + shift[:, None, :]
    return h, gate[:, None, :]


class DenoiserLayer(nn.Module):
    """adaLN-Zero HyperGCN -> [joint self-attn] -> CrossAttn(kp) -> adaLN-Zero FFN.

    [F7] adaLN-Zero (shift, scale, gate; zero-init) replaces LayerScale, so the
         gate is conditioned on the timestep and the layer starts as identity.
    """
    def __init__(self, dim, num_kp=17, num_heads=8, mlp_ratio=4.0,
                 drop_path=0.1, use_hyper=True, use_joint_sa=False,
                 drop=0.1):
        super().__init__()
        self.dim, self.num_heads = dim, num_heads
        self.mlp_ratio, self.drop_path_rate = mlp_ratio, drop_path

        # --- HyperGCN ---
        self.norm1 = nn.LayerNorm(dim, elementwise_affine=False)
        self.ada1 = _make_adaln_zero(dim)
        self.hgcn = SMPL_HyperGCNBlock(dim, use_hyper=use_hyper)
        self.dp1 = DropPath(drop_path) if drop_path > 0 else nn.Identity()

        # --- optional joint self-attention [F9] ---
        self.joint_sa = None
        if use_joint_sa:
            self.norm_sa = nn.LayerNorm(dim, elementwise_affine=False)
            self.ada_sa = _make_adaln_zero(dim)
            self.joint_sa = nn.MultiheadAttention(
                dim, num_heads, dropout=drop, batch_first=True)
            self.dp_sa = DropPath(drop_path) if drop_path > 0 else nn.Identity()

        # --- cross-attention to keypoint tokens ---
        self.cross_attn = CrossAttentionBlock(
            q_dim=dim, k_dim=dim, v_dim=dim, kv_num=num_kp,
            num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=True,
            drop=drop, attn_drop=0., drop_path=drop_path, has_mlp=False,
        )

        # --- FFN ---
        self.norm2 = nn.LayerNorm(dim, elementwise_affine=False)
        self.ada2 = _make_adaln_zero(dim)
        hid = int(dim * mlp_ratio)
        self.ffn = nn.Sequential(
            nn.Linear(dim, hid), nn.GELU(), nn.Dropout(drop),
            nn.Linear(hid, dim), nn.Dropout(drop),
        )
        self.dp2 = DropPath(drop_path) if drop_path > 0 else nn.Identity()

        # context slot (future: cross-attn to img/joint features)   [F2]
        self.context_attn = None

    def enable_context_attn(self, kv_num, ctx_dim=None):
        """Create the optional context cross-attention branch."""
        ctx_dim = ctx_dim or self.dim
        self.context_attn = CrossAttentionBlock(
            q_dim=self.dim, k_dim=ctx_dim, v_dim=ctx_dim, kv_num=kv_num,
            num_heads=self.num_heads, mlp_ratio=self.mlp_ratio,
            qkv_bias=True, drop=0.1, attn_drop=0.,
            drop_path=self.drop_path_rate, has_mlp=False,
        )

    def forward(self, x, kp_tok, kp_bias, t_emb, ctx=None):
        """
        x       : (B,24,dim)
        kp_tok  : (B,17,dim)
        kp_bias : (B,H,1,17)
        t_emb   : (B,dim)
        ctx     : optional (B,N,dim)
        """
        # 1) HyperGCN
        h, g = _modulate(x, self.norm1, self.ada1, t_emb)
        x = x + self.dp1(g * self.hgcn(h))

        # 2) optional joint self-attention
        if self.joint_sa is not None:
            h, g = _modulate(x, self.norm_sa, self.ada_sa, t_emb)
            a, _ = self.joint_sa(h, h, h, need_weights=False)
            x = x + self.dp_sa(g * a)

        # 3) cross-attn to keypoints (block has its own norm + residual)
        x = self.cross_attn(x, kp_tok, kp_tok, attn_bias=kp_bias)

        # 4) optional context cross-attn
        if ctx is not None:
            if self.context_attn is None:
                raise RuntimeError(
                    "context was provided but context_attn is not "
                    "initialised. Call model.enable_context_attn(kv_num) "
                    "first.")
            x = self.context_attn(x, ctx, ctx)

        # 5) FFN
        h, g = _modulate(x, self.norm2, self.ada2, t_emb)
        x = x + self.dp2(g * self.ffn(h))
        return x


# ====================================================================
# SMPL_HyperDiff  (main module)
# ====================================================================

class SMPL_HyperDiff(nn.Module):
    """Conditional Diffusion with Hypergraph for SMPL 6D Pose.

    Parameters are read from ``cfg.DIFF`` by default; constructor kwargs
    override them. New optional cfg.DIFF keys (all have defaults):

        use_hyper=True, use_joint_sa=False, use_self_cond=True,
        min_snr_gamma=0.0 (0 = off), w_rot=0.1, rot_loss_type='geodesic'|'l1',
        geo_t_frac=0.5, w_fk=0.0, t_sampling='uniform'|'sqrt', dropout=0.1
    """

    def __init__(self, **kw):
        super().__init__()
        g = lambda k, d: kw.get(k, _cfg(k, d))          # noqa: E731

        C = kw.get('dim_feat', cfg.DIFF.dim_feat)
        Cr = kw.get('dim_rep', cfg.DIFF.dim_rep)
        T = kw.get('num_timesteps', cfg.DIFF.num_timesteps)
        St = kw.get('sampling_timesteps', cfg.DIFF.sampling_timesteps)
        eta = kw.get('ddim_eta', cfg.DIFF.ddim_eta)
        nL = kw.get('n_layers', cfg.DIFF.n_layers)
        nH = kw.get('num_heads', cfg.DIFF.num_heads)
        mr = kw.get('mlp_ratio', cfg.DIFF.mlp_ratio)
        dp = kw.get('drop_path_rate', cfg.DIFF.drop_path_rate)
        lt = kw.get('loss_type', cfg.DIFF.loss_type)
        pk = kw.get('p_kp_dropout', cfg.DIFF.p_kp_dropout)

        assert C % nH == 0, "dim_feat must be divisible by num_heads"
        assert C % 2 == 0, "dim_feat must be even"

        self.num_joints = 24
        self.dim_out = 6
        self.dim_feat = C
        self.T = T
        self.St = St
        self.eta = eta
        self.loss_type = lt
        self.p_kp_drop = pk
        self.num_heads = nH

        # new options
        self.use_self_cond = bool(g('use_self_cond', True))
        self.min_snr_gamma = float(g('min_snr_gamma', 0.0))
        self.w_rot = float(g('w_rot', 0.1))
        self.rot_loss_type = g('rot_loss_type', 'geodesic')
        self.geo_t_frac = float(g('geo_t_frac', 0.5))
        self.w_fk = float(g('w_fk', 0.0))
        self.t_sampling = g('t_sampling', 'uniform')
        drop = float(g('dropout', 0.1))
        use_hyper = bool(g('use_hyper', True))
        use_joint_sa = bool(g('use_joint_sa', False))

        # optional forward-kinematics head: rotmat (B,24,3,3) -> joints (B,K,3)
        # If an nn.Module is assigned it is registered (moves with .to()).
        self.fk_fn = None

        # ---- cosine schedule (only buffers that are actually used) ----
        betas = cosine_beta_schedule(T)
        ac = torch.cumprod(1.0 - betas, dim=0)
        for name, val in [
            ('ac', ac),
            ('sqrt_ac', ac.sqrt()),
            ('sqrt_1mac', (1 - ac).sqrt()),
            ('rsqrt_ac', ac.rsqrt()),
            ('rsqrt_m1', (1 / ac - 1).sqrt()),
        ]:
            self.register_buffer(name, val)

        # ---- denoiser components ----
        in_dim = self.dim_out * (2 if self.use_self_cond else 1)
        self.pose_emb = nn.Linear(in_dim, C)        # [x_t | x_self_cond]
        self.node_pe = nn.Embedding(self.num_joints, C)
        self.time_mlp = nn.Sequential(
            SinusoidalTimeEmbed(C),
            nn.Linear(C, C * 4), nn.GELU(), nn.Linear(C * 4, C),
        )
        self.kp_tok = KeypointTokenizer(C, num_kp=17)

        dpr = [x.item() for x in torch.linspace(0, dp, nL)]
        self.layers = nn.ModuleList([
            DenoiserLayer(C, 17, nH, mr, dpr[i],
                          use_hyper=use_hyper, use_joint_sa=use_joint_sa,
                          drop=drop)
            for i in range(nL)
        ])
        self.final_norm = nn.LayerNorm(C)
        self.rep_logit = nn.Sequential(nn.Linear(C, Cr), nn.Tanh())
        self.head = nn.Linear(Cr, self.dim_out)

    # ----- optional hooks -----

    def set_fk_fn(self, fn):
        """Register an FK function (e.g. wrapper around the SMPL layer)."""
        self.fk_fn = fn

    def enable_context_attn(self, kv_num, ctx_dim=None):
        """Enable the context cross-attention branch in every layer.  [F2]"""
        for layer in self.layers:
            layer.enable_context_attn(kv_num, ctx_dim)

    # ----- diffusion primitives -----

    def q_sample(self, x0, t, noise=None):
        if noise is None:
            noise = torch.randn_like(x0)
        return (_extract(self.sqrt_ac, t, x0.shape) * x0
                + _extract(self.sqrt_1mac, t, x0.shape) * noise)

    def _pred_noise(self, x_t, t, pred_x0):
        return ((_extract(self.rsqrt_ac, t, x_t.shape) * x_t - pred_x0)
                / _extract(self.rsqrt_m1, t, x_t.shape))

    # ----- denoiser network -----

    def _denoise(self, x_t, t, kp_tokens, kp_bias, ctx=None, x_self=None):
        """x_self: previous x0 estimate for self-conditioning (or None)."""
        if self.use_self_cond:
            if x_self is None:
                x_self = torch.zeros_like(x_t)
            inp = torch.cat([x_t, x_self], dim=-1)                # (B,24,12)
        else:
            inp = x_t
        x = self.pose_emb(inp) + self.node_pe.weight[None]
        t_emb = self.time_mlp(t)          # time enters only through adaLN
        for layer in self.layers:
            x = layer(x, kp_tokens, kp_bias, t_emb, ctx)
        return self.head(self.rep_logit(self.final_norm(x)))       # (B,24,6)

    def _prepare_kp(self, kp2d, kp_conf, dropout=False):
        """Build kp tokens + attn bias, optionally with condition dropout."""
        B = kp2d.shape[0]
        tokens = self.kp_tok(kp2d, kp_conf)
        if dropout and self.training and self.p_kp_drop > 0:
            drop = torch.rand(B, device=kp2d.device) < self.p_kp_drop
            null = self.kp_tok.no_kp_token.expand(B, -1, -1)
            tokens = torch.where(drop[:, None, None], null, tokens)
            kp_conf = torch.where(drop[:, None],
                                  torch.zeros_like(kp_conf), kp_conf)
        bias = KeypointTokenizer.compute_attn_bias(kp_conf, self.num_heads)
        return tokens, bias

    # ----- loss helpers -----

    @staticmethod
    def _reduce(per_j, valid_mask):
        """(B,24) -> (B,) masked mean over joints."""
        if valid_mask is None:
            return per_j.mean(-1)
        vm = valid_mask.to(per_j.dtype)
        return (per_j * vm).sum(-1) / (vm.sum(-1) + 1e-8)

    def _snr_weight(self, t):
        """Min-SNR-gamma weight for x0-prediction, normalised to (0,1]."""
        if self.min_snr_gamma <= 0:
            return torch.ones_like(t, dtype=torch.float32)
        ac = self.ac[t]
        snr = ac / (1 - ac).clamp(min=1e-8)
        return snr.clamp(max=self.min_snr_gamma) / self.min_snr_gamma

    def _sample_t(self, B, device):
        if self.t_sampling == 'sqrt':          # density ~ t  (more high-noise)
            u = torch.rand(B, device=device)
            return (self.T * u.sqrt()).long().clamp(max=self.T - 1)
        return torch.randint(0, self.T, (B,), device=device)

    # ----- forward -----

    def forward(self, x0=None, kp2d=None, kp_conf=None, is_train=None,
                context=None, valid_mask=None, gt_joints=None,
                return_loss_dict=False):
        """
        Args
        ----
        x0         : (B,24,6) clean GT 6D  (training only; None in eval)
        kp2d       : (B,17,2) in [-1,1]
        kp_conf    : (B,17)
        is_train   : DEPRECATED - mode is taken from ``self.training``.  [F3]
        context    : optional (B,N,dim), needs enable_context_attn()
        valid_mask : (B,24) joint validity mask for the loss
        gt_joints  : optional (B,K,3) GT 3D joints for the FK loss

        Returns
        -------
        train -> (pred_x0, diff_loss[, loss_dict])   diff_loss is a scalar
        eval  -> pred_x0  (B,24,6)
        """
        assert kp2d is not None and kp_conf is not None
        if is_train is not None and bool(is_train) != self.training:
            warnings.warn("is_train disagrees with model.training; "
                          "using model.training.", stacklevel=2)

        if not self.training:
            return self.ddim_sample(kp2d, kp_conf, context=context)

        assert x0 is not None, "x0 is required in training mode"
        B, device = x0.shape[0], x0.device

        kp_tokens, kp_bias = self._prepare_kp(kp2d, kp_conf, dropout=True)

        t = self._sample_t(B, device)
        x_t = self.q_sample(x0, t)

        # [F8] self-conditioning: 50% of the time feed a detached estimate
        x_self = None
        if self.use_self_cond and torch.rand(1).item() < 0.5:
            with torch.no_grad():
                x_self = self._denoise(x_t, t, kp_tokens, kp_bias,
                                       context).detach()

        pred_x0 = self._denoise(x_t, t, kp_tokens, kp_bias, context, x_self)

        # ---- main 6D loss (with optional Min-SNR weighting) ----
        if self.loss_type == 'l1':
            per_j = F.l1_loss(pred_x0, x0, reduction='none').mean(-1)
        else:
            per_j = F.mse_loss(pred_x0, x0, reduction='none').mean(-1)
        loss_6d = (self._reduce(per_j, valid_mask)
                   * self._snr_weight(t)).mean()
        loss = loss_6d
        ld = {'loss_6d': loss_6d.detach()}

        # ---- [F5] geometric losses, only on small-t samples ----
        need_rot = self.w_rot > 0
        need_fk = (self.w_fk > 0 and self.fk_fn is not None
                   and gt_joints is not None)
        if need_rot or need_fk:
            gate = (t < int(self.geo_t_frac * self.T)).float()       # (B,)
            denom = gate.sum().clamp(min=1.0)
            R_pred = rot6d_to_rotmat(pred_x0)

            if need_rot:
                R_gt = rot6d_to_rotmat(x0)
                if self.rot_loss_type == 'l1':
                    per_j_rot = (R_pred - R_gt).abs().mean(dim=(-1, -2))
                else:
                    per_j_rot = geodesic_distance(R_pred, R_gt)
                l_rot = (gate * self._reduce(per_j_rot, valid_mask)).sum() / denom
                loss = loss + self.w_rot * l_rot
                ld['loss_rot'] = l_rot.detach()

            if need_fk:
                j_pred = self.fk_fn(R_pred)
                l_fk = (gate * (j_pred - gt_joints).abs().mean(dim=(-1, -2))
                        ).sum() / denom
                loss = loss + self.w_fk * l_fk
                ld['loss_fk'] = l_fk.detach()

        if return_loss_dict:
            return pred_x0, loss, ld
        return pred_x0, loss

    # ----- DDIM sampling -----

    @torch.no_grad()
    def ddim_sample(self, kp2d, kp_conf, context=None, generator=None,
                    num_samples=1, return_all=False):
        """DDIM sampling.   [F3][F6][F8]

        num_samples > 1 : draw several hypotheses per input and average the
                          6D outputs (Gram-Schmidt later restores validity).
        return_all      : return (B, S, 24, 6) instead of the mean.
        """
        was_training = self.training
        self.eval()                                   # [F3] no dropout at test
        try:
            B = kp2d.shape[0]
            device = kp2d.device
            S = int(num_samples)
            if S > 1:
                kp2d = kp2d.repeat_interleave(S, dim=0)
                kp_conf = kp_conf.repeat_interleave(S, dim=0)
                if context is not None:
                    context = context.repeat_interleave(S, dim=0)
            Bn = kp2d.shape[0]
            shape = (Bn, self.num_joints, self.dim_out)

            kp_tokens, kp_bias = self._prepare_kp(kp2d, kp_conf, dropout=False)

            times = torch.linspace(-1, self.T - 1, steps=self.St + 1,
                                   device=device)
            times = list(reversed(times.int().tolist()))
            pairs = list(zip(times[:-1], times[1:]))

            x = torch.randn(shape, device=device, generator=generator)
            x_self = None

            for t_now, t_next in pairs:
                tb = torch.full((Bn,), t_now, device=device, dtype=torch.long)
                p0 = self._denoise(x, tb, kp_tokens, kp_bias, context, x_self)
                if self.use_self_cond:
                    x_self = p0

                if t_next < 0:
                    x = p0
                    break

                pn = self._pred_noise(x, tb, p0)
                a_now = self.ac[t_now]
                a_nxt = self.ac[t_next]
                sig = self.eta * ((1 - a_now / a_nxt) *
                                  (1 - a_nxt) / (1 - a_now)).sqrt()
                c = (1 - a_nxt - sig ** 2).sqrt()
                noise = (torch.randn(x.shape, device=device,
                                     generator=generator)
                         if self.eta > 0 else 0.0)
                x = p0 * a_nxt.sqrt() + c * pn + sig * noise

            x = x.view(B, S, self.num_joints, self.dim_out)
            if return_all:
                return x
            return x.mean(dim=1)                       # (B,24,6)
        finally:
            if was_training:
                self.train()

    # ----- optimiser helper -----

    def get_param_groups(self, weight_decay):
        """[F11] No weight decay for biases/norms/scalars/gates/embeddings."""
        decay, no_decay = [], []
        skip_names = ('node_pe', 'joint_embed', 'no_kp_token')
        for n, p in self.named_parameters():
            if not p.requires_grad:
                continue
            if p.ndim <= 1 or any(s in n for s in skip_names):
                no_decay.append(p)
            else:
                decay.append(p)
        return [{'params': decay, 'weight_decay': weight_decay},
                {'params': no_decay, 'weight_decay': 0.0}]


# ====================================================================
# EMA of weights   [F11]
# ====================================================================

class EMA:
    """Exponential moving average of parameters.

    ema = EMA(model, 0.999)
    ... after each optimizer.step():  ema.update(model)
    ... for evaluation:
        backup = ema.apply_to(model); evaluate(model); ema.restore(model, backup)
    (pass ``model.module`` if the model is wrapped in DataParallel)
    """
    def __init__(self, model, decay=0.999):
        self.decay = decay
        self.shadow = {n: p.detach().clone()
                       for n, p in model.named_parameters() if p.requires_grad}

    @torch.no_grad()
    def update(self, model):
        for n, p in model.named_parameters():
            if n in self.shadow:
                self.shadow[n].mul_(self.decay).add_(p.detach(),
                                                     alpha=1 - self.decay)

    @torch.no_grad()
    def apply_to(self, model):
        backup = {}
        for n, p in model.named_parameters():
            if n in self.shadow:
                backup[n] = p.detach().clone()
                p.copy_(self.shadow[n])
        return backup

    @torch.no_grad()
    def restore(self, model, backup):
        for n, p in model.named_parameters():
            if n in backup:
                p.copy_(backup[n])