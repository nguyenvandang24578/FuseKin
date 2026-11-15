"""
lib/models/sd_encoder/blocks.py -- Khoi co ban cua Stable Diffusion 1.5 (ban toi gian, 2026-11-15).

Chep lai tu DPMesh/ldm (CompVis latent-diffusion, MIT) + DPMesh/cldm (ControlNet, Apache-2.0), CHI giu phan forward
can de trich dac trung. GIU NGUYEN ten module / tham so de nap thang control_sd15_openpose.pth.
Khac ban goc (khong doi cong thuc):
  - bo xformers / pytorch_lightning / omegaconf (khong can cai them goi);
  - gradient checkpointing dung torch.utils.checkpoint (use_reentrant=False), CHI ap cho ResBlock
    (khoi transformer co luu ban do attention ra ngoai -> khong checkpoint de tranh tinh lai / luu trung);
  - attention khong can luu ban do dung F.scaled_dot_product_attention (cung cong thuc softmax(QK^T / sqrt(d)) V);
  - cross-attention cua U-Net luu ban do attention khi attn_store duoc gan (thay register_attention_control cua DPMesh).
Cau hinh SD 1.5 (cldm_v15.yaml): use_scale_shift_norm False, resblock_updown False, conv_resample True,
use_linear_in_transformer False, transformer_depth 1, legacy False, dropout 0.
"""
import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint as torch_checkpoint


def zero_module(module):
    """Dat toan bo tham so ve 0 (giong ldm util.zero_module)."""
    for p in module.parameters():
        p.detach().zero_()
    return module


class GroupNorm32(nn.GroupNorm):
    """GroupNorm tinh o float32 roi tra ve dtype dau vao (giong ldm util.GroupNorm32)."""
    def forward(self, x):
        return super().forward(x.float()).type(x.dtype)


def normalization(channels):
    return GroupNorm32(32, channels)


def timestep_embedding(timesteps, dim, max_period=10000):
    """Embedding sin / cos cua buoc t (giong ldm util.timestep_embedding, repeat_only=False)."""
    half = dim // 2
    freqs = torch.exp(-math.log(max_period) * torch.arange(start=0, end=half, dtype=torch.float32) / half
                      ).to(device=timesteps.device)
    args = timesteps[:, None].float() * freqs[None]
    emb = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
    if dim % 2:
        emb = torch.cat([emb, torch.zeros_like(emb[:, :1])], dim=-1)
    return emb


class TimestepBlock(nn.Module):
    """Module nhan them embedding thoi gian emb (forward(x, emb))."""


class TimestepEmbedSequential(nn.Sequential):
    """Sequential truyen emb cho TimestepBlock, context cho SpatialTransformer (giong openaimodel)."""
    def forward(self, x, emb, context=None):
        for layer in self:
            if isinstance(layer, TimestepBlock):
                x = layer(x, emb)
            elif isinstance(layer, SpatialTransformer):
                x = layer(x, context)
            else:
                x = layer(x)
        return x


class Upsample(nn.Module):
    """Nearest x2 + conv 3x3 (use_conv True, dims 2). Ten tham so: conv.*"""
    def __init__(self, channels, out_channels=None):
        super().__init__()
        self.channels = channels
        self.out_channels = out_channels or channels
        self.conv = nn.Conv2d(self.channels, self.out_channels, 3, padding=1)

    def forward(self, x):
        assert x.shape[1] == self.channels
        x = F.interpolate(x, scale_factor=2, mode='nearest')
        return self.conv(x)


class Downsample(nn.Module):
    """Conv 3x3 stride 2 (use_conv True, dims 2). Ten tham so: op.*"""
    def __init__(self, channels, out_channels=None):
        super().__init__()
        self.channels = channels
        self.out_channels = out_channels or channels
        self.op = nn.Conv2d(self.channels, self.out_channels, 3, stride=2, padding=1)

    def forward(self, x):
        assert x.shape[1] == self.channels
        return self.op(x)


class ResBlock(TimestepBlock):
    """ResBlock cua U-Net (khong scale-shift, khong up / down). Ten tham so giong openaimodel.ResBlock:
    in_layers.{0 norm, 2 conv}, emb_layers.1, out_layers.{0 norm, 3 conv}, skip_connection (conv 1x1 neu doi kenh)."""
    def __init__(self, channels, emb_channels, out_channels=None, use_checkpoint=False):
        super().__init__()
        self.channels = channels
        self.out_channels = out_channels or channels
        self.use_checkpoint = use_checkpoint
        self.in_layers = nn.Sequential(
            normalization(channels),
            nn.SiLU(),
            nn.Conv2d(channels, self.out_channels, 3, padding=1),
        )
        self.emb_layers = nn.Sequential(
            nn.SiLU(),
            nn.Linear(emb_channels, self.out_channels),
        )
        self.out_layers = nn.Sequential(
            normalization(self.out_channels),
            nn.SiLU(),
            nn.Dropout(p=0.0),
            zero_module(nn.Conv2d(self.out_channels, self.out_channels, 3, padding=1)),
        )
        if self.out_channels == channels:
            self.skip_connection = nn.Identity()
        else:
            self.skip_connection = nn.Conv2d(channels, self.out_channels, 1)

    def forward(self, x, emb):
        if self.use_checkpoint and torch.is_grad_enabled():
            return torch_checkpoint(self._forward, x, emb, use_reentrant=False)
        return self._forward(x, emb)

    def _forward(self, x, emb):
        h = self.in_layers(x)
        emb_out = self.emb_layers(emb).type(h.dtype)
        while len(emb_out.shape) < len(h.shape):
            emb_out = emb_out[..., None]
        h = h + emb_out
        h = self.out_layers(h)
        return self.skip_connection(x) + h


class AttnStore:
    """Noi luu ban do cross-attention cua U-Net trong 1 lan forward (thay AttentionStore cua DPMesh).
    Chi luu khi so token query <= max_q (DPMesh: max_size = (256 / 8) // 2 = 16 -> 16 x 16 = 256 token);
    ban do da trung binh theo head: (B, Nq, Nk)."""
    def __init__(self, max_q=256):
        self.max_q = int(max_q)
        self.maps = []

    def wants(self, nq):
        return nq <= self.max_q

    def add(self, attn):
        self.maps.append(attn)

    def clear(self):
        self.maps = []


class CrossAttention(nn.Module):
    """Attention nhieu head (giong ldm attention.CrossAttention). Ten tham so: to_q, to_k, to_v, to_out.0.
    context None -> self-attention. attn_store (AttnStore, gan tu ngoai) khac None va la cross-attention co
    Nq <= max_q -> tinh tuong minh softmax(QK^T * scale) de luu ban do (trung binh theo head, giong
    register_attention_control cua DPMesh); con lai dung scaled_dot_product_attention (cung ket qua)."""
    def __init__(self, query_dim, context_dim=None, heads=8, dim_head=64, dropout=0.0):
        super().__init__()
        inner_dim = dim_head * heads
        context_dim = context_dim if context_dim is not None else query_dim
        self.scale = dim_head ** -0.5
        self.heads = heads
        self.to_q = nn.Linear(query_dim, inner_dim, bias=False)
        self.to_k = nn.Linear(context_dim, inner_dim, bias=False)
        self.to_v = nn.Linear(context_dim, inner_dim, bias=False)
        self.to_out = nn.Sequential(nn.Linear(inner_dim, query_dim), nn.Dropout(dropout))
        self.attn_store = None

    def forward(self, x, context=None):
        h = self.heads
        is_cross = context is not None
        ctx = context if is_cross else x
        q = self.to_q(x)
        k = self.to_k(ctx)
        v = self.to_v(ctx)
        B, Nq, _ = q.shape
        Nk = k.shape[1]
        # 'b n (h d) -> b h n d' (giong rearrange 'b n (h d) -> (b h) n d' cua ldm)
        q = q.reshape(B, Nq, h, -1).transpose(1, 2)
        k = k.reshape(B, Nk, h, -1).transpose(1, 2)
        v = v.reshape(B, Nk, h, -1).transpose(1, 2)
        if is_cross and self.attn_store is not None and self.attn_store.wants(Nq):
            sim = torch.einsum('bhid,bhjd->bhij', q, k) * self.scale
            attn = sim.softmax(dim=-1)
            self.attn_store.add(attn.mean(1))                             # (B, Nq, Nk), trung binh theo head
            out = torch.einsum('bhij,bhjd->bhid', attn.to(v.dtype), v)
        else:
            out = F.scaled_dot_product_attention(q, k, v)                 # scale mac dinh = dim_head ** -0.5
        out = out.transpose(1, 2).reshape(B, Nq, -1)                       # 'b h n d -> b n (h d)'
        return self.to_out(out)


class GEGLU(nn.Module):
    """Ten tham so: proj (Linear dim_in -> 2 dim_out)."""
    def __init__(self, dim_in, dim_out):
        super().__init__()
        self.proj = nn.Linear(dim_in, dim_out * 2)

    def forward(self, x):
        x, gate = self.proj(x).chunk(2, dim=-1)
        return x * F.gelu(gate)


class FeedForward(nn.Module):
    """glu=True (SD): net = [GEGLU(dim, 4 dim), Dropout, Linear(4 dim, dim)]. Ten tham so: net.0.proj, net.2."""
    def __init__(self, dim, mult=4, dropout=0.0):
        super().__init__()
        inner_dim = int(dim * mult)
        self.net = nn.Sequential(GEGLU(dim, inner_dim), nn.Dropout(dropout), nn.Linear(inner_dim, dim))

    def forward(self, x):
        return self.net(x)


class BasicTransformerBlock(nn.Module):
    """attn1 (self) -> attn2 (cross voi context) -> ff, moi buoc co LayerNorm truoc va residual."""
    def __init__(self, dim, n_heads, d_head, context_dim=None):
        super().__init__()
        self.attn1 = CrossAttention(query_dim=dim, heads=n_heads, dim_head=d_head)
        self.ff = FeedForward(dim)
        self.attn2 = CrossAttention(query_dim=dim, context_dim=context_dim, heads=n_heads, dim_head=d_head)
        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)
        self.norm3 = nn.LayerNorm(dim)

    def forward(self, x, context=None):
        x = self.attn1(self.norm1(x)) + x
        x = self.attn2(self.norm2(x), context=context) + x
        x = self.ff(self.norm3(x)) + x
        return x


class SpatialTransformer(nn.Module):
    """use_linear False, depth 1 (SD 1.5). Ten tham so: norm, proj_in (conv 1x1), transformer_blocks.0.*, proj_out."""
    def __init__(self, in_channels, n_heads, d_head, context_dim=None):
        super().__init__()
        self.in_channels = in_channels
        inner_dim = n_heads * d_head
        self.norm = nn.GroupNorm(num_groups=32, num_channels=in_channels, eps=1e-6, affine=True)
        self.proj_in = nn.Conv2d(in_channels, inner_dim, kernel_size=1, stride=1, padding=0)
        self.transformer_blocks = nn.ModuleList([BasicTransformerBlock(inner_dim, n_heads, d_head,
                                                                       context_dim=context_dim)])
        self.proj_out = zero_module(nn.Conv2d(inner_dim, in_channels, kernel_size=1, stride=1, padding=0))

    def forward(self, x, context=None):
        b, c, h, w = x.shape
        x_in = x
        x = self.norm(x)
        x = self.proj_in(x)
        x = x.flatten(2).transpose(1, 2).contiguous()                      # 'b c h w -> b (h w) c'
        for block in self.transformer_blocks:
            x = block(x, context=context)
        x = x.transpose(1, 2).reshape(b, -1, h, w).contiguous()            # 'b (h w) c -> b c h w'
        x = self.proj_out(x)
        return x + x_in
