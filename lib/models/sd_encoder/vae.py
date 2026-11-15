"""
lib/models/sd_encoder/vae.py -- Encoder VAE (AutoencoderKL) cua SD 1.5, ban toi gian (2026-11-15).

Chep tu DPMesh/ldm/modules/diffusionmodules/model.py (Encoder) + ldm/models/autoencoder.py (quant_conv, encode).
Chi giu nhanh encode -> mode() = trung binh cua phan phoi Gauss (giong VPDEncoder.forward cua DPMesh:
encode(x).mode()). Bo decoder. Ten tham so: encoder.*, quant_conv.* (= 'first_stage_model.' trong checkpoint SD).
ddconfig SD 1.5: ch 128, ch_mult (1,2,4,4), num_res_blocks 2, attn_resolutions [], z_channels 4, double_z True.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


def nonlinearity(x):
    return x * torch.sigmoid(x)                                            # swish


def Normalize(in_channels, num_groups=32):
    return nn.GroupNorm(num_groups=num_groups, num_channels=in_channels, eps=1e-6, affine=True)


class Downsample(nn.Module):
    """with_conv True: pad (0,1,0,1) roi conv 3x3 stride 2 padding 0. Ten tham so: conv.*"""
    def __init__(self, in_channels):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, in_channels, kernel_size=3, stride=2, padding=0)

    def forward(self, x):
        x = F.pad(x, (0, 1, 0, 1), mode='constant', value=0)
        return self.conv(x)


class ResnetBlock(nn.Module):
    """temb_channels = 0 (encoder VAE) -> khong co temb_proj. Ten tham so: norm1, conv1, norm2, conv2, nin_shortcut."""
    def __init__(self, in_channels, out_channels=None):
        super().__init__()
        self.in_channels = in_channels
        out_channels = in_channels if out_channels is None else out_channels
        self.out_channels = out_channels
        self.norm1 = Normalize(in_channels)
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1)
        self.norm2 = Normalize(out_channels)
        self.dropout = nn.Dropout(0.0)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1)
        if self.in_channels != self.out_channels:
            self.nin_shortcut = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, padding=0)

    def forward(self, x):
        h = self.conv1(nonlinearity(self.norm1(x)))
        h = self.conv2(self.dropout(nonlinearity(self.norm2(h))))
        if self.in_channels != self.out_channels:
            x = self.nin_shortcut(x)
        return x + h


class AttnBlock(nn.Module):
    """Self-attention 1 head tren luoi (attn_type 'vanilla'). Ten tham so: norm, q, k, v, proj_out.
    Ban goc dung bmm + softmax(q k * c^-0.5); o day dung scaled_dot_product_attention (cung cong thuc)."""
    def __init__(self, in_channels):
        super().__init__()
        self.in_channels = in_channels
        self.norm = Normalize(in_channels)
        self.q = nn.Conv2d(in_channels, in_channels, kernel_size=1, stride=1, padding=0)
        self.k = nn.Conv2d(in_channels, in_channels, kernel_size=1, stride=1, padding=0)
        self.v = nn.Conv2d(in_channels, in_channels, kernel_size=1, stride=1, padding=0)
        self.proj_out = nn.Conv2d(in_channels, in_channels, kernel_size=1, stride=1, padding=0)

    def forward(self, x):
        h_ = self.norm(x)
        q, k, v = self.q(h_), self.k(h_), self.v(h_)
        b, c, h, w = q.shape
        q = q.reshape(b, 1, c, h * w).transpose(2, 3)                       # (b, 1, hw, c)
        k = k.reshape(b, 1, c, h * w).transpose(2, 3)
        v = v.reshape(b, 1, c, h * w).transpose(2, 3)
        out = F.scaled_dot_product_attention(q, k, v)                      # scale = c ** -0.5 (giong ban goc)
        out = out.transpose(2, 3).reshape(b, c, h, w)
        return x + self.proj_out(out)


class Encoder(nn.Module):
    def __init__(self, ch=128, ch_mult=(1, 2, 4, 4), num_res_blocks=2, in_channels=3, z_channels=4, double_z=True):
        super().__init__()
        self.ch = ch
        self.num_resolutions = len(ch_mult)
        self.num_res_blocks = num_res_blocks
        self.conv_in = nn.Conv2d(in_channels, ch, kernel_size=3, stride=1, padding=1)
        in_ch_mult = (1,) + tuple(ch_mult)
        self.down = nn.ModuleList()
        block_in = ch
        for i_level in range(self.num_resolutions):
            block = nn.ModuleList()
            block_in = ch * in_ch_mult[i_level]
            block_out = ch * ch_mult[i_level]
            for _ in range(self.num_res_blocks):
                block.append(ResnetBlock(block_in, block_out))
                block_in = block_out
            down = nn.Module()
            down.block = block
            down.attn = nn.ModuleList()                                    # attn_resolutions [] -> rong
            if i_level != self.num_resolutions - 1:
                down.downsample = Downsample(block_in)
            self.down.append(down)
        self.mid = nn.Module()
        self.mid.block_1 = ResnetBlock(block_in, block_in)
        self.mid.attn_1 = AttnBlock(block_in)
        self.mid.block_2 = ResnetBlock(block_in, block_in)
        self.norm_out = Normalize(block_in)
        self.conv_out = nn.Conv2d(block_in, 2 * z_channels if double_z else z_channels,
                                  kernel_size=3, stride=1, padding=1)

    def forward(self, x):
        h = self.conv_in(x)
        for i_level in range(self.num_resolutions):
            for i_block in range(self.num_res_blocks):
                h = self.down[i_level].block[i_block](h)
            if i_level != self.num_resolutions - 1:
                h = self.down[i_level].downsample(h)
        h = self.mid.block_1(h)
        h = self.mid.attn_1(h)
        h = self.mid.block_2(h)
        h = nonlinearity(self.norm_out(h))
        return self.conv_out(h)


class SDVAEEncoder(nn.Module):
    """x (B,3,H,W) -> latent mode (B,4,H/8,W/8): moments = quant_conv(encoder(x)); mean = 4 kenh dau
    (DiagonalGaussianDistribution.mode() = mean; clamp logvar khong anh huong mean)."""
    def __init__(self, embed_dim=4, z_channels=4):
        super().__init__()
        self.encoder = Encoder(z_channels=z_channels, double_z=True)
        self.quant_conv = nn.Conv2d(2 * z_channels, 2 * embed_dim, 1)

    def forward(self, x):
        moments = self.quant_conv(self.encoder(x))
        mean, _ = torch.chunk(moments, 2, dim=1)
        return mean
