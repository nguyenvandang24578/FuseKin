"""
lib/models/img_lifter.py -- ImgLifter (2026-11-04): diffusion 2D -> 3D CO ANH, thay DiffLifter cu.

Dau vao (moi nguoi, moi anh) lay tu Refiner 2D (dong bang, lib/models/refiner2d.py):
  xy (B,17,2) 2D da sua, dinh nghia mesh, [-1,1] | log_b (B,17) do bat dinh (o luoi 64) | det (B,17) detector co khop
  cam_bbox (B,3) | f8 (B,2048,8,8) dac trung anh toan cuc | feat_j (B,17,C) dac trung anh tai tung khop
Dieu kien tung khop c_j = Linear([x, y, log_b, det, cam_bbox]) + Linear(feat_j).
  [1] cong vao token khop o dau vao; [2] adaLN-Zero (shift / scale / gate tu c_j + t) o MOI block;
  [3] cross-attention tu 17 token khop toi 64 token anh (f8) o MOI block.
Diffusion giong DiffLifter cu: cosine T=1000, du doan x0, scale pose x3, DDIM eta=1; loss MPJPE (L2) tren x0.
Suy luan: K gia thuyet x S buoc DDIM, gop 'mean' hoac 'jpma' (moi khop chon gia thuyet chieu 2D khop xy nhat;
khop co b > jpma_max_b giu trung binh). Output: (B,17,3) met, root-relative tai Pelvis, dinh nghia khop mesh.
"""
import torch
import torch.nn as nn

from models.diff_lifter import cosine_beta_schedule, extract, Mlp, SinusoidalPositionEmbeddings

try:
    from timm.models.layers import DropPath, trunc_normal_
except ImportError:
    from timm.layers import DropPath, trunc_normal_

COND_DIM = 7     # [x, y, log_b, det, cam_bbox(3)]


def modulate(x, shift, scale):
    return x * (1 + scale) + shift


class ImgLifterBlock(nn.Module):
    """adaLN-Zero (theo tung khop) -> self-attn khop | cross-attn anh | FFN."""

    def __init__(self, dim, num_heads, mlp_ratio=2.0, drop_path=0.0):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim, elementwise_affine=False, eps=1e-6)
        self.self_attn = nn.MultiheadAttention(dim, num_heads, batch_first=True)
        self.norm2 = nn.LayerNorm(dim, elementwise_affine=False, eps=1e-6)
        self.cross_attn = nn.MultiheadAttention(dim, num_heads, batch_first=True)
        self.norm3 = nn.LayerNorm(dim, elementwise_affine=False, eps=1e-6)
        self.mlp = Mlp(in_features=dim, hidden_features=int(dim * mlp_ratio))
        self.drop_path = DropPath(drop_path) if drop_path > 0 else nn.Identity()
        self.ada = nn.Sequential(nn.SiLU(), nn.Linear(dim, 9 * dim))
        nn.init.zeros_(self.ada[-1].weight)      # adaLN-Zero: luc dau moi block = identity
        nn.init.zeros_(self.ada[-1].bias)

    def forward(self, x, cond, img_tok):
        """x (N,17,D) token khop; cond (N,17,D) dieu kien tung khop (+ t); img_tok (N,64,D)."""
        s1, c1, g1, s2, c2, g2, s3, c3, g3 = self.ada(cond).chunk(9, dim=-1)
        h = modulate(self.norm1(x), s1, c1)
        x = x + self.drop_path(g1 * self.self_attn(h, h, h, need_weights=False)[0])
        h = modulate(self.norm2(x), s2, c2)
        x = x + self.drop_path(g2 * self.cross_attn(h, img_tok, img_tok, need_weights=False)[0])
        h = modulate(self.norm3(x), s3, c3)
        x = x + self.drop_path(g3 * self.mlp(h))
        return x


class ImgLifterDenoiser(nn.Module):
    def __init__(self, num_joints=17, dim=512, depth=8, num_heads=8, mlp_ratio=2.0, drop_path=0.1,
                 img_ch=2048, feat_j_ch=256, img_tokens=64):
        super().__init__()
        self.num_joints = num_joints
        self.x_embed = nn.Linear(3, dim)
        self.cond_embed = nn.Sequential(nn.Linear(COND_DIM, dim), nn.GELU(), nn.Linear(dim, dim))
        self.featj_embed = nn.Sequential(nn.LayerNorm(feat_j_ch), nn.Linear(feat_j_ch, dim))
        self.joint_pos = nn.Parameter(torch.zeros(1, num_joints, dim))
        self.img_embed = nn.Sequential(nn.LayerNorm(img_ch), nn.Linear(img_ch, dim))
        self.img_pos = nn.Parameter(torch.zeros(1, img_tokens, dim))
        self.time_mlp = nn.Sequential(SinusoidalPositionEmbeddings(dim), nn.Linear(dim, dim * 2), nn.GELU(),
                                      nn.Linear(dim * 2, dim))
        dpr = [v.item() for v in torch.linspace(0, drop_path, depth)]
        self.blocks = nn.ModuleList([ImgLifterBlock(dim, num_heads, mlp_ratio, dpr[i]) for i in range(depth)])
        self.final_norm = nn.LayerNorm(dim, elementwise_affine=False, eps=1e-6)
        self.final_ada = nn.Sequential(nn.SiLU(), nn.Linear(dim, 2 * dim))
        self.head = nn.Linear(dim, 3)
        nn.init.zeros_(self.final_ada[-1].weight)
        nn.init.zeros_(self.final_ada[-1].bias)
        trunc_normal_(self.joint_pos, std=0.02)
        trunc_normal_(self.img_pos, std=0.02)

    def encode_cond(self, cond7, feat_j, f8):
        """Tinh 1 lan cho moi anh (dung lai o moi buoc DDIM / moi gia thuyet).
        cond7 (B,17,7), feat_j (B,17,C), f8 (B,2048,8,8) -> c_j (B,17,D), img_tok (B,64,D)."""
        c_j = self.cond_embed(cond7) + self.featj_embed(feat_j)
        img = f8.flatten(2).transpose(1, 2)                                 # (B, 64, 2048)
        img_tok = self.img_embed(img) + self.img_pos
        return c_j, img_tok

    def forward(self, x_t, t, c_j, img_tok):
        """x_t (N,17,3) da chia scale; t (N,); c_j (N,17,D); img_tok (N,64,D) -> x0 (N,17,3) met."""
        x = self.x_embed(x_t) + c_j + self.joint_pos                        # [1] dieu kien o dau vao
        cond = c_j + self.time_mlp(t)[:, None, :]                           # [2] adaLN: dieu kien khop + t
        for blk in self.blocks:
            x = blk(x, cond, img_tok)                                       # [3] cross-attn anh trong blk
        shift, scale = self.final_ada(cond).chunk(2, dim=-1)
        return self.head(modulate(self.final_norm(x), shift, scale))


class ImgLifter(nn.Module):
    def __init__(self, num_joints=17, dim=512, depth=8, num_heads=8, mlp_ratio=2.0, drop_path=0.1,
                 timesteps=1000, scale=3.0, feat_j_ch=256):
        super().__init__()
        self.num_joints = num_joints
        self.scale = float(scale)
        self.denoiser = ImgLifterDenoiser(num_joints, dim, depth, num_heads, mlp_ratio, drop_path,
                                          feat_j_ch=feat_j_ch)
        betas = cosine_beta_schedule(timesteps)
        ac = torch.cumprod(1. - betas, dim=0)
        self.num_timesteps = int(betas.shape[0])
        self.register_buffer('alphas_cumprod', ac.to(torch.float32))
        self.register_buffer('sqrt_alphas_cumprod', torch.sqrt(ac).to(torch.float32))
        self.register_buffer('sqrt_one_minus_alphas_cumprod', torch.sqrt(1. - ac).to(torch.float32))

    @staticmethod
    def make_cond7(xy, log_b, det, cam_bbox):
        """(B,17,2), (B,17), (B,17), (B,3) -> (B,17,7)."""
        J = xy.shape[1]
        return torch.cat([xy, log_b.unsqueeze(-1), det.float().unsqueeze(-1),
                          cam_bbox[:, None, :].expand(-1, J, -1)], dim=-1)

    def _x0(self, x_t_scaled, t, c_j, img_tok):
        p = self.denoiser(x_t_scaled, t, c_j, img_tok)
        return p - p[:, 0:1, :]                                             # root-relative, met

    def loss(self, pose_3d, cond7, feat_j, f8, valid=None):
        """pose_3d (B,17,3) GT met root-relative. valid (B,) 0/1. Tra (loss MPJPE met, pred_x0)."""
        B = pose_3d.shape[0]
        x0 = ((pose_3d - pose_3d[:, 0:1]) * self.scale).clamp(-1.1 * self.scale, 1.1 * self.scale)
        t = torch.randint(0, self.num_timesteps, (B,), device=pose_3d.device).long()
        noise = torch.randn_like(x0)
        x_t = extract(self.sqrt_alphas_cumprod, t, x0.shape) * x0 + \
            extract(self.sqrt_one_minus_alphas_cumprod, t, x0.shape) * noise
        c_j, img_tok = self.denoiser.encode_cond(cond7, feat_j, f8)
        pred = self._x0(x_t.clamp(-1.1 * self.scale, 1.1 * self.scale) / self.scale, t, c_j, img_tok)
        err = torch.norm(pred - (pose_3d - pose_3d[:, 0:1]), dim=-1).mean(-1)   # (B,)
        if valid is None:
            return err.mean(), pred
        v = valid.float()
        return (err * v).sum() / v.sum().clamp_min(1.0), pred

    @torch.no_grad()
    def sample(self, cond7, feat_j, f8, K=10, S=4, eta=1.0, generator=None):
        """DDIM, K gia thuyet song song. Tra all_preds (B,K,17,3) met root-relative."""
        B, J = cond7.shape[:2]
        dev = cond7.device
        c_j, img_tok = self.denoiser.encode_cond(cond7, feat_j, f8)
        c_j = c_j.repeat_interleave(K, dim=0)                               # (B*K,17,D)
        img_tok = img_tok.repeat_interleave(K, dim=0)
        T = self.num_timesteps
        times = list(reversed(torch.linspace(-1, T - 1, steps=S + 1).int().tolist()))
        x = torch.randn(B * K, J, 3, device=dev, generator=generator)
        sc = self.scale
        for time, time_next in zip(times[:-1], times[1:]):
            t = torch.full((B * K,), time, device=dev, dtype=torch.long)
            x0 = (self._x0(x.clamp(-1.1 * sc, 1.1 * sc) / sc, t, c_j, img_tok) * sc).clamp(-1.1 * sc, 1.1 * sc)
            if time_next < 0:
                x = x0
                break
            a, an = self.alphas_cumprod[time], self.alphas_cumprod[time_next]
            sigma = eta * ((1 - a / an) * (1 - an) / (1 - a)).clamp(min=0).sqrt()
            c = (1 - an - sigma ** 2).clamp(min=0).sqrt()
            eps = (x - x0 * a.sqrt()) / (1 - a + 1e-8).sqrt()
            noise = torch.randn(x.shape, device=dev, dtype=x.dtype, generator=generator)
            x = x0 * an.sqrt() + c * eps + sigma * noise
        return (x / sc).reshape(B, K, J, 3)


def aggregate(all_preds, xy, log_b, mode='mean', jpma_max_b=3.0):
    """all_preds (B,K,17,3); xy (B,17,2) [-1,1] (2D Refiner); log_b (B,17).
    'mean': trung binh K. 'jpma': fit weak-perspective (s, t) tu XY cua pose trung binh sang xy (trong so 1 / b^2),
    moi khop co b <= jpma_max_b chon gia thuyet co loi chieu 2D nho nhat; con lai giu trung binh."""
    mean = all_preds.mean(1)
    if mode == 'mean' or all_preds.shape[1] == 1:
        return mean - mean[:, 0:1]
    B, K, J, _ = all_preds.shape
    b = torch.exp(log_b)
    w = 1.0 / b.clamp_min(0.25) ** 2
    P = mean[..., :2]
    ws = w.sum(1, keepdim=True)[..., None]
    Pm = (w[..., None] * P).sum(1, keepdim=True) / ws
    pm = (w[..., None] * xy).sum(1, keepdim=True) / ws
    Pc, pc = P - Pm, xy - pm
    s = ((w * (Pc * pc).sum(-1)).sum(1) / (w * (Pc ** 2).sum(-1)).sum(1).clamp_min(1e-8))[:, None, None]
    t = pm - s * Pm
    proj = s[:, None] * all_preds[..., :2] + t[:, None]                    # (B,K,17,2)
    err = torch.norm(proj - xy[:, None], dim=-1)                            # (B,K,17)
    best = err.argmin(1)
    sel = torch.gather(all_preds, 1, best[:, None, :, None].expand(B, 1, J, 3)).squeeze(1)
    out = torch.where((b <= jpma_max_b)[..., None], sel, mean)
    return out - out[:, 0:1]
