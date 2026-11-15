"""
lib/models/sd_encoder/unet.py -- U-Net SD 1.5 (lay dac trung nhieu tang) + ControlNet (ban toi gian, 2026-11-15).

SDUNet  = cldm.DPControlledUnet / ldm UNetModel cua DPMesh, voi forward cua register_hier_output (module_utils.py):
          KHONG co no_grad o nhanh encoder (de gradient toi LoRA), cong control vao skip, tra dac trung output_blocks
          1, 4, 7 + dau ra cuoi; KHONG dung lop 'out' (dau ra eps) -> bo khi xay.
ControlNet = cldm.ControlNet cua DPMesh: conv dau nhan latent NOI heatmap (in_channels + hint_channels),
          KHONG co input_hint_block (DPMesh da comment), zero_convs sau moi input block + middle_block_out.
Ten module giong het ban goc -> nap duoc 'model.diffusion_model.*' / 'control_model.*' cua control_sd15_openpose.pth.
"""
import torch
import torch.nn as nn

from .blocks import (TimestepEmbedSequential, ResBlock, Downsample, Upsample, SpatialTransformer,
                     timestep_embedding, zero_module)

# cldm_v15.yaml (unet_config / control_stage_config)
SD15 = dict(model_channels=320, channel_mult=(1, 2, 4, 4), num_res_blocks=2, attention_resolutions=(4, 2, 1),
            num_heads=8, context_dim=768)


def _st(ch, c):
    # legacy False, num_head_channels -1 -> dim_head = ch // num_heads
    return SpatialTransformer(ch, c['num_heads'], ch // c['num_heads'], context_dim=c['context_dim'])


def _build_input_blocks(in_channels, emb_dim, use_checkpoint, c=SD15):
    """input_blocks giong UNetModel / ControlNet. Tra (input_blocks, input_block_chans, ch cuoi, ds cuoi)."""
    mc = c['model_channels']
    blocks = nn.ModuleList([TimestepEmbedSequential(nn.Conv2d(in_channels, mc, 3, padding=1))])
    chans = [mc]
    ch, ds = mc, 1
    for level, mult in enumerate(c['channel_mult']):
        for _ in range(c['num_res_blocks']):
            layers = [ResBlock(ch, emb_dim, out_channels=mult * mc, use_checkpoint=use_checkpoint)]
            ch = mult * mc
            if ds in c['attention_resolutions']:
                layers.append(_st(ch, c))
            blocks.append(TimestepEmbedSequential(*layers))
            chans.append(ch)
        if level != len(c['channel_mult']) - 1:
            blocks.append(TimestepEmbedSequential(Downsample(ch, out_channels=ch)))
            chans.append(ch)
            ds *= 2
    return blocks, chans, ch, ds


def _build_middle(ch, emb_dim, use_checkpoint, c=SD15):
    return TimestepEmbedSequential(
        ResBlock(ch, emb_dim, use_checkpoint=use_checkpoint),
        _st(ch, c),
        ResBlock(ch, emb_dim, use_checkpoint=use_checkpoint),
    )


def _time_embed(mc, emb_dim):
    return nn.Sequential(nn.Linear(mc, emb_dim), nn.SiLU(), nn.Linear(emb_dim, emb_dim))


class SDUNet(nn.Module):
    """U-Net SD 1.5 (in_channels 4), khong co lop 'out'.
    forward(x (B,4,H,W), t (B,), context (B,N,768), control list | None) ->
      [h_out1, h_out4, h_out7, h_cuoi] = voi latent 32x32: [1280@4x4, 1280@8x8, 640@16x16, 320@32x32]."""
    def __init__(self, use_checkpoint=False, c=SD15):
        super().__init__()
        mc = c['model_channels']
        emb_dim = mc * 4
        self.model_channels = mc
        self.time_embed = _time_embed(mc, emb_dim)
        self.input_blocks, chans, ch, ds = _build_input_blocks(4, emb_dim, use_checkpoint, c)
        self.middle_block = _build_middle(ch, emb_dim, use_checkpoint, c)
        self.output_blocks = nn.ModuleList([])
        for level, mult in list(enumerate(c['channel_mult']))[::-1]:
            for i in range(c['num_res_blocks'] + 1):
                ich = chans.pop()
                layers = [ResBlock(ch + ich, emb_dim, out_channels=mc * mult, use_checkpoint=use_checkpoint)]
                ch = mc * mult
                if ds in c['attention_resolutions']:
                    layers.append(_st(ch, c))
                if level and i == c['num_res_blocks']:
                    layers.append(Upsample(ch, out_channels=ch))
                    ds //= 2
                self.output_blocks.append(TimestepEmbedSequential(*layers))

    def forward(self, x, timesteps, context=None, control=None):
        emb = self.time_embed(timestep_embedding(timesteps, self.model_channels))
        hs = []
        h = x
        for module in self.input_blocks:
            h = module(h, emb, context)
            hs.append(h)
        h = self.middle_block(h, emb, context)
        if control is not None:
            control = list(control)                                        # khong sua list cua ben goi
            h = h + control.pop()                                          # middle_block_out
        out_list = []
        for i_out, module in enumerate(self.output_blocks):
            skip = hs.pop()
            if control is not None:
                skip = skip + control.pop()
            h = torch.cat([h, skip], dim=1)
            h = module(h, emb, context)
            if i_out in (1, 4, 7):
                out_list.append(h)
        out_list.append(h)
        return out_list


class ControlNet(nn.Module):
    """ControlNet cua DPMesh: x = cat(latent, hint) -> input_blocks (+ zero_conv moi block) -> middle (+ middle_block_out).
    forward(x (B,4,H,W), hint (B,hint_channels,H,W), t, context) -> list 13 tensor (12 input block + middle)."""
    def __init__(self, hint_channels, in_channels=4, use_checkpoint=False, c=SD15):
        super().__init__()
        mc = c['model_channels']
        emb_dim = mc * 4
        self.model_channels = mc
        self.in_channels = in_channels
        self.hint_channels = hint_channels
        self.time_embed = _time_embed(mc, emb_dim)
        self.input_blocks, chans, ch, _ = _build_input_blocks(in_channels + hint_channels, emb_dim,
                                                              use_checkpoint, c)
        self.zero_convs = nn.ModuleList([self._zero_conv(cc) for cc in chans])
        self.middle_block = _build_middle(ch, emb_dim, use_checkpoint, c)
        self.middle_block_out = self._zero_conv(ch)

    @staticmethod
    def _zero_conv(channels):
        return TimestepEmbedSequential(zero_module(nn.Conv2d(channels, channels, 1, padding=0)))

    def forward(self, x, hint, timesteps, context=None):
        emb = self.time_embed(timestep_embedding(timesteps, self.model_channels))
        h = torch.cat((x, hint), dim=1)
        outs = []
        for module, zero_conv in zip(self.input_blocks, self.zero_convs):
            h = module(h, emb, context)
            outs.append(zero_conv(h, emb, context))
        h = self.middle_block(h, emb, context)
        outs.append(self.middle_block_out(h, emb, context))
        return outs
