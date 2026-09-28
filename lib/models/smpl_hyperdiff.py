"""
SMPL_HyperDiff: Conditional Diffusion with Hypergraph for SMPL 6D Pose.

Replaces HYPERGCv2 + VPoser in Pose2Mesh with a denoising diffusion model
that directly predicts 24×6D rotation parameters conditioned on 2D keypoints.

Architecture
============
  x₀ = GT 24×6D rotations (absolute, not residual)
  Forward diffusion: q(x_t | x₀) with cosine schedule, T=1000
  Denoiser:  per-layer → AdaLN-HyperGCN → CrossAttn(kp2D tokens) → AdaLN-FFN
  Inference: DDIM 10-step from pure noise
"""

import math
import warnings

import torch
import torch.nn as nn
import torch.nn.functional as F
from timm.models.layers import DropPath

from core.config import cfg
from models.Core_model import CrossAttentionBlock

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

# ====================================================================
# Rotation 6D  ↔  axis-angle  conversion
# ====================================================================

def axis_angle_to_rotmat(aa):
    """Rodrigues: (*, 3) axis-angle → (*, 3, 3) rotation matrix."""
    orig_shape = aa.shape[:-1]
    aa = aa.reshape(-1, 3)
    theta = aa.norm(dim=-1, keepdim=True).clamp(min=1e-8)       # (N, 1)
    axis = aa / theta                                            # (N, 3)
    cos_t = torch.cos(theta).unsqueeze(-1)                       # (N, 1, 1)
    sin_t = torch.sin(theta).unsqueeze(-1)                       # (N, 1, 1)
    kx, ky, kz = axis[:, 0:1], axis[:, 1:2], axis[:, 2:3]
    zeros = torch.zeros_like(kx)
    K = torch.cat([zeros, -kz, ky,
                   kz, zeros, -kx,
                   -ky, kx, zeros], dim=-1).reshape(-1, 3, 3)
    I = torch.eye(3, device=aa.device, dtype=aa.dtype).unsqueeze(0)
    R = I + sin_t * K + (1 - cos_t) * (K @ K)
    near_zero = (theta.squeeze(-1) < 1e-6)
    if near_zero.any():
        R[near_zero] = I[0]
    return R.reshape(*orig_shape, 3, 3)


def axis_angle_to_rot6d(aa):
    """(*, 3) axis-angle → (*, 6) rot6d.

    Convention: 6D = first two **columns** of rotation matrix flattened in
    row-major order →  [r00, r01, r10, r11, r20, r21].
    This matches ``rot6d_to_axis_angle()`` in ``utils/transforms.py`` which
    does ``.view(-1, 3, 2)`` to recover columns 0 and 1.
    """
    orig_shape = aa.shape[:-1]
    R = axis_angle_to_rotmat(aa.reshape(-1, 3))        # (N, 3, 3)
    rot6d = R[:, :, :2].reshape(-1, 6)                 # first 2 cols, flatten
    return rot6d.reshape(*orig_shape, 6)


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
    De = H.sum(dim=0)  + eps       # (E,)
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
    """Shared displacement per limb group → (B,17,2)."""
    noise = torch.zeros_like(kp)
    for group in H36M_LIMB_GROUPS:
        disp = torch.randn(kp.shape[0], 1, 2,
                           device=kp.device, dtype=kp.dtype) * limb_sigma
        for j in group:
            noise[:, j] = noise[:, j] + disp.squeeze(1)
    return noise


def apply_big_errors(kp, p_big_err):
    """Left-right swap or random-position jump per joint.

    Returns: (kp_out, big_err_mask)
    """
    B, J, _ = kp.shape
    big_err = torch.rand(B, J, device=kp.device) < p_big_err
    kp_out = kp.clone()
    is_swap = torch.rand(B, J, device=kp.device) < 0.5

    for left, right in H36M_SWAP_PAIRS:
        do = (big_err[:, left] | big_err[:, right]) & \
             (is_swap[:, left] | is_swap[:, right])
        m = do.unsqueeze(-1)
        tmp = kp_out[:, left].clone()
        kp_out[:, left]  = torch.where(m, kp_out[:, right], kp_out[:, left])
        kp_out[:, right] = torch.where(m, tmp, kp_out[:, right])
        big_err[:, left]  = big_err[:, left]  | do
        big_err[:, right] = big_err[:, right] | do

    jump_mask = big_err & ~is_swap
    rand_pos  = torch.rand(B, J, 2, device=kp.device, dtype=kp.dtype) * 2 - 1
    kp_out = torch.where(jump_mask.unsqueeze(-1), rand_pos, kp_out)
    return kp_out, big_err


def make_noisy_kp2d(kp_gt, conf_gt, noise_cfg=None, warmup_scale=1.0):
    """Simulate detector noise on GT 2D keypoints for training.

    Args
    ----
    kp_gt     : (B, 17, 2)  clean GT keypoints in [-1, 1]
    conf_gt   : (B, 17)     binary confidence {0, 1}
    noise_cfg : edict – from ``cfg.DIFF.NOISE``
    warmup_scale : float – curriculum multiplier (0.3→1.0)

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

    # --- per-joint Gaussian noise ---
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
        err  = (kp - kp_gt).norm(dim=-1)
        conf = (1.0 - noise_cfg.k * err
                + 0.05 * torch.randn_like(err)).clamp(0, 1)
        conf = torch.where(big_err, conf.clamp(min=0.6), conf)
    else:
        conf = (~drop).float()

    kp   = kp.masked_fill(drop.unsqueeze(-1), 0.0)
    conf = conf.masked_fill(drop, 0.0)
    return kp, conf, drop


# ====================================================================
# Sub-modules
# ====================================================================

class SinusoidalTimeEmbed(nn.Module):
    """Sinusoidal positional embedding for timestep."""
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, t):
        half = self.dim // 2
        emb  = math.log(10_000) / (half - 1)
        emb  = torch.exp(torch.arange(half, device=t.device) * -emb)
        emb  = t[:, None].float() * emb[None, :]
        return torch.cat([emb.sin(), emb.cos()], dim=-1)     # (B, dim)


class KeypointTokenizer(nn.Module):
    """[x, y, conf, bone_x, bone_y] → Linear → dim  + joint PE."""

    def __init__(self, dim, num_kp=17, kp_input_dim=5):
        super().__init__()
        self.proj = nn.Linear(kp_input_dim, dim)
        self.joint_embed = nn.Embedding(num_kp, dim)
        # "no-keypoint" learned token for condition-dropout
        self.no_kp_token = nn.Parameter(torch.randn(1, num_kp, dim) * 0.02)
        # H36M bone helpers (parent, root mask)
        parent = torch.tensor([max(0, p) for p in H36M_PARENT], dtype=torch.long)
        rmask  = torch.tensor([float(p >= 0) for p in H36M_PARENT])
        self.register_buffer('parent_idx', parent)
        self.register_buffer('root_mask', rmask)

    def forward(self, kp2d, conf):
        """
        kp2d : (B,17,2)   conf : (B,17)
        Returns (B,17,dim)
        """
        bone = (kp2d - kp2d[:, self.parent_idx]) * self.root_mask[None, :, None]
        feat = torch.cat([kp2d, conf.unsqueeze(-1), bone], dim=-1)   # (B,17,5)
        return self.proj(feat) + self.joint_embed.weight.unsqueeze(0)

    @staticmethod
    def compute_attn_bias(conf, num_heads):
        """log(conf + ε) → (B, H, 1, 17)  additive attention bias."""
        bias = torch.log(conf + 1e-3)            # (B,17)
        return bias[:, None, None, :].expand(-1, num_heads, -1, -1)


# ====================================================================
# SMPL HyperGCN block  (spatial + part + body)
# ====================================================================

class SMPL_HyperGCNBlock(nn.Module):
    def __init__(self, dim, num_joints=24):
        super().__init__()
        self.register_buffer('A_spatial', build_smpl_kinematic_adj(num_joints))
        self.register_buffer('G_part',    _compute_G(_build_H_part(num_joints)))
        self.register_buffer('G_body',    _compute_G(_build_H_body(num_joints)))
        self.conv_s = nn.Linear(dim, dim)
        self.conv_p = nn.Linear(dim, dim)
        self.conv_b = nn.Linear(dim, dim)
        self.a1 = nn.Parameter(torch.tensor(1.0))
        self.a2 = nn.Parameter(torch.tensor(1.0))
        self.a3 = nn.Parameter(torch.tensor(1.0))

    def forward(self, x):
        """x : (B, 24, dim) → (B, 24, dim)"""
        a1 = F.softplus(self.a1)
        a2 = F.softplus(self.a2)
        a3 = F.softplus(self.a3)
        return (a1 * self.conv_s(self.A_spatial @ x) +
                a2 * self.conv_p(self.G_part    @ x) +
                a3 * self.conv_b(self.G_body    @ x))


# ====================================================================
# Denoiser layer
# ====================================================================

class DenoiserLayer(nn.Module):
    """AdaLN-HyperGCN → CrossAttn(kp) → AdaLN-FFN.

    Time embedding modulates HyperGCN and FFN via AdaLN (scale+shift).
    CrossAttentionBlock uses its own internal LayerNorm.
    """
    def __init__(self, dim, num_kp=17, num_heads=8,
                 mlp_ratio=4.0, drop_path=0.1, ls_init=1e-5):
        super().__init__()
        # --- HyperGCN + AdaLN ---
        self.norm1   = nn.LayerNorm(dim)
        self.adaln1  = nn.Sequential(nn.SiLU(), nn.Linear(dim, dim * 2))
        self.hgcn    = SMPL_HyperGCNBlock(dim)
        self.ls1     = nn.Parameter(ls_init * torch.ones(dim))
        self.dp1     = DropPath(drop_path) if drop_path > 0 else nn.Identity()

        # --- Cross-attention to keypoint tokens (reuse CrossAttentionBlock) ---
        self.cross_attn = CrossAttentionBlock(
            q_dim=dim, k_dim=dim, v_dim=dim, kv_num=num_kp,
            num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=True,
            drop=0.1, attn_drop=0., drop_path=drop_path, has_mlp=False,
        )

        # --- FFN + AdaLN ---
        self.norm2   = nn.LayerNorm(dim)
        self.adaln2  = nn.Sequential(nn.SiLU(), nn.Linear(dim, dim * 2))
        hid = int(dim * mlp_ratio)
        self.ffn     = nn.Sequential(
            nn.Linear(dim, hid), nn.GELU(), nn.Dropout(0.1),
            nn.Linear(hid, dim), nn.Dropout(0.1),
        )
        self.ls2     = nn.Parameter(ls_init * torch.ones(dim))
        self.dp2     = DropPath(drop_path) if drop_path > 0 else nn.Identity()

        # context slot placeholder (future: cross-attn to img_out+joint_out)
        self.context_attn = None

    # helpers
    @staticmethod
    def _adaln(x, norm, adaln, t_emb):
        s, sh = adaln(t_emb).chunk(2, dim=-1)           # (B, dim) each
        return norm(x) * (1 + s[:, None, :]) + sh[:, None, :]

    def forward(self, x, kp_tok, kp_bias, t_emb, ctx=None):
        """
        x       : (B,24,dim)
        kp_tok  : (B,17,dim)
        kp_bias : (B,H,1,17)
        t_emb   : (B,dim)
        ctx     : optional (B,N,dim)
        """
        # 1) HyperGCN
        x = x + self.dp1(self.ls1 * self.hgcn(
            self._adaln(x, self.norm1, self.adaln1, t_emb)))

        # 2) Cross-attn to keypoints  (CAB has own norm + residual)
        x = self.cross_attn(x, kp_tok, kp_tok, attn_bias=kp_bias)

        # 3) Optional context cross-attn (not yet instantiated)
        if ctx is not None:
            if self.context_attn is not None:
                x = self.context_attn(x, ctx, ctx)
            else:
                warnings.warn(
                    "context provided but context_attn not initialised. "
                    "Call enable_context_attn() first.", stacklevel=2)

        # 4) FFN
        x = x + self.dp2(self.ls2 * self.ffn(
            self._adaln(x, self.norm2, self.adaln2, t_emb)))
        return x


# ====================================================================
# SMPL_HyperDiff  (main module)
# ====================================================================

class SMPL_HyperDiff(nn.Module):
    """Conditional Diffusion with Hypergraph for SMPL 6D Pose.

    Parameters are read from ``cfg.DIFF`` by default; constructor kwargs
    override them.
    """

    def __init__(self, **kw):
        super().__init__()
        # ---- read hyper-params ----
        C  = kw.get('dim_feat',  cfg.DIFF.dim_feat)
        Cr = kw.get('dim_rep',   cfg.DIFF.dim_rep)
        T  = kw.get('num_timesteps',    cfg.DIFF.num_timesteps)
        St = kw.get('sampling_timesteps', cfg.DIFF.sampling_timesteps)
        eta = kw.get('ddim_eta', cfg.DIFF.ddim_eta)
        nL = kw.get('n_layers',  cfg.DIFF.n_layers)
        nH = kw.get('num_heads', cfg.DIFF.num_heads)
        mr = kw.get('mlp_ratio', cfg.DIFF.mlp_ratio)
        dp = kw.get('drop_path_rate', cfg.DIFF.drop_path_rate)
        ls = kw.get('layer_scale_init', cfg.DIFF.layer_scale_init)
        lt = kw.get('loss_type', cfg.DIFF.loss_type)
        pk = kw.get('p_kp_dropout', cfg.DIFF.p_kp_dropout)

        self.num_joints = 24
        self.dim_out    = 6
        self.dim_feat   = C
        self.T          = T
        self.St         = St
        self.eta        = eta
        self.loss_type  = lt
        self.p_kp_drop  = pk
        self.num_heads  = nH

        # ---- cosine schedule ----
        betas = cosine_beta_schedule(T)
        ac    = torch.cumprod(1.0 - betas, dim=0)
        acp   = F.pad(ac[:-1], (1, 0), value=1.0)

        for name, val in [
            ('betas', betas), ('ac', ac), ('acp', acp),
            ('sqrt_ac',   ac.sqrt()),
            ('sqrt_1mac', (1 - ac).sqrt()),
            ('rsqrt_ac',  ac.rsqrt()),
            ('rsqrt_m1',  (1 / ac - 1).sqrt()),
        ]:
            self.register_buffer(name, val)

        # ---- denoiser components ----
        self.pose_emb  = nn.Linear(self.dim_out, C)
        self.node_pe   = nn.Embedding(self.num_joints, C)
        self.time_mlp  = nn.Sequential(
            SinusoidalTimeEmbed(C),
            nn.Linear(C, C * 4), nn.GELU(), nn.Linear(C * 4, C),
        )
        self.kp_tok    = KeypointTokenizer(C, num_kp=17)

        dpr = [x.item() for x in torch.linspace(0, dp, nL)]
        self.layers = nn.ModuleList([
            DenoiserLayer(C, 17, nH, mr, dpr[i], ls) for i in range(nL)
        ])
        self.final_norm = nn.LayerNorm(C)
        self.rep_logit  = nn.Sequential(nn.Linear(C, Cr), nn.Tanh())
        self.head       = nn.Linear(Cr, self.dim_out)

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

    def _denoise(self, x_t, t, kp_tokens, kp_bias, ctx=None):
        x = self.pose_emb(x_t) + self.node_pe.weight[None]
        t_emb = self.time_mlp(t)                              # (B, C)
        x = x + t_emb[:, None, :]
        for layer in self.layers:
            x = layer(x, kp_tokens, kp_bias, t_emb, ctx)
        return self.head(self.rep_logit(self.final_norm(x)))   # (B,24,6)

    def _prepare_kp(self, kp2d, kp_conf, dropout=False):
        """Build kp tokens + attn bias, optionally with condition dropout."""
        B = kp2d.shape[0]
        tokens = self.kp_tok(kp2d, kp_conf)
        if dropout and self.training and self.p_kp_drop > 0:
            drop = torch.rand(B, device=kp2d.device) < self.p_kp_drop
            null = self.kp_tok.no_kp_token.expand(B, -1, -1)
            tokens   = torch.where(drop[:, None, None], null, tokens)
            kp_conf  = torch.where(drop[:, None],
                                   torch.zeros_like(kp_conf), kp_conf)
        bias = KeypointTokenizer.compute_attn_bias(kp_conf, self.num_heads)
        return tokens, bias

    # ----- training forward -----

    def forward(self, x0, kp2d, kp_conf, is_train=True,
                context=None, valid_mask=None):
        """
        Args
        ----
        x0         : (B,24,6) clean GT 6D (training only)
        kp2d       : (B,17,2) in [-1,1]
        kp_conf    : (B,17)
        context    : optional (B,N,dim) — reserved for future
        valid_mask : (B,24) joint validity mask for loss masking

        Returns
        -------
        train  → (pred_x0, diff_loss)
        eval   → pred_x0
        """
        if not is_train:
            return self.ddim_sample(kp2d, kp_conf, context=context)

        B = x0.shape[0]
        device = x0.device

        # condition tokens
        kp_tokens, kp_bias = self._prepare_kp(kp2d, kp_conf, dropout=True)

        # sample t
        if getattr(cfg.DIFF, 't_sampling', 'uniform') == 'sqrt':
            u = torch.rand(B, device=device)
            t = (self.T * u.sqrt()).long().clamp(max=self.T - 1)
        else:
            t = torch.randint(0, self.T, (B,), device=device)

        # forward diffusion + predict x0
        x_t = self.q_sample(x0, t)
        pred_x0 = self._denoise(x_t, t, kp_tokens, kp_bias, context)

        # loss
        if self.loss_type == 'l1':
            per_j = F.l1_loss(pred_x0, x0, reduction='none').mean(-1)
        else:
            per_j = F.mse_loss(pred_x0, x0, reduction='none').mean(-1)  # (B,24)

        if valid_mask is not None:
            diff_loss = (per_j * valid_mask).sum() / (valid_mask.sum() + 1e-8)
        else:
            diff_loss = per_j.mean()

        return pred_x0, diff_loss

    # ----- DDIM sampling -----

    @torch.no_grad()
    def ddim_sample(self, kp2d, kp_conf, context=None, generator=None):
        B = kp2d.shape[0]
        device = kp2d.device
        shape  = (B, self.num_joints, self.dim_out)

        kp_tokens, kp_bias = self._prepare_kp(kp2d, kp_conf, dropout=False)

        times = torch.linspace(-1, self.T - 1, steps=self.St + 1, device=device)
        times = list(reversed(times.int().tolist()))
        pairs = list(zip(times[:-1], times[1:]))

        x = (torch.randn(shape, device=device, generator=generator)
             if generator else torch.randn(shape, device=device))

        for t_now, t_next in pairs:
            tb = torch.full((B,), t_now, device=device, dtype=torch.long)
            p0 = self._denoise(x, tb, kp_tokens, kp_bias, context)

            if t_next < 0:
                x = p0
                break

            pn    = self._pred_noise(x, tb, p0)
            a_now = self.ac[t_now]
            a_nxt = self.ac[t_next]
            sig   = self.eta * ((1 - a_now / a_nxt) *
                                (1 - a_nxt) / (1 - a_now)).sqrt()
            c     = (1 - a_nxt - sig ** 2).sqrt()
            noise = torch.randn_like(x) if self.eta > 0 else 0.0
            x     = p0 * a_nxt.sqrt() + c * pn + sig * noise

        return x
