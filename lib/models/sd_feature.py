"""
lib/models/sd_feature.py -- SDFeatureEncoder (2026-11-15): dac trung anh f8 (B,2048,8,8) tu Stable Diffusion 1.5 +
ControlNet theo loi DPMesh (common/vpd/vpdencoder_useattn.py + common/model.py), thay f8 ResNet-50 cua Refiner
cho ImgLifter (IMGLIFT.sd_enable). Decoder ImgLifter giu nguyen.

  anh 256 RGB [0,1] -> VAE encoder (dong bang, no_grad) -> latent mode (B,4,32,32) x latent_scale
  2D (xy Refiner [-1,1], J = 17 khop H36M) -> heatmap Gauss J x 32 x 32 (sigma 2.5, khop det = 0 -> 0)
                                          -> prompt: Linear(2 -> 768) tren toa do khop / 32 (context cross-attn)
  ControlNet(cat(latent, heatmap), t = 1, context) -> 13 tensor cong vao U-Net (input skip + middle)
  U-Net (dong bang + LoRA r) -> output_blocks 1, 4, 7 + cuoi; + ban do cross-attention (up_cross + down_cross,
          trung binh theo head va theo lop) o 8x8 va 16x16 (J kenh)
  -> outmodules (prior [320, 640 + J, 1280 + 1280 + J]) -> f8 (B, 2048, 8, 8)

Khac DPMesh (co chu dich, ghi trong experiment.md):
  - J = 17 (H36M) thay 30 -> conv dau ControlNet (4 + 17 -> 320) va outmodules khoi tao moi;
    sd_ctrl_in_init 'copy_latent': chep 4 kenh latent tu conv dau ControlNet goc (DPMesh: bo trong so, ngau nhien);
  - khong CLIP (DPMesh khoi tao nhung khong dung), khong pytorch_lightning;
  - trong so dong bang luu bf16 khi sd_bf16 (chay trong autocast bf16), tham so train giu fp32.
Muc train ControlNet (sd_train_control): 'none' = A0 (giong DPMesh, ControlNet dong bang) | 'in_zero' = A (train
conv dau + zero_convs + middle_block_out) | 'full' = B (train ca ControlNet).
"""
import os

import torch
import torch.nn as nn
import torch.nn.functional as F

from models.sd_encoder.blocks import AttnStore, CrossAttention
from models.sd_encoder.unet import SDUNet, ControlNet
from models.sd_encoder.vae import SDVAEEncoder
from models.sd_encoder.lora import inject_trainable_lora


def _load_raw_state(path):
    """Doc checkpoint SD (.pth / .ckpt / .safetensors) -> dict ten -> tensor (bo boc 'state_dict' neu co)."""
    if path.endswith('.safetensors'):
        from safetensors.torch import load_file
        sd = load_file(path, device='cpu')
    else:
        try:
            sd = torch.load(path, map_location='cpu', weights_only=True)
        except Exception:
            sd = torch.load(path, map_location='cpu', weights_only=False)
    if isinstance(sd, dict) and 'state_dict' in sd and isinstance(sd['state_dict'], dict):
        sd = sd['state_dict']
    return sd


def _sub(sd, prefix):
    return {k[len(prefix):]: v for k, v in sd.items() if k.startswith(prefix)}


class SDFeatureEncoder(nn.Module):
    def __init__(self, num_joints=17, lora_rank=64, hm_size=32, hm_sigma=2.5, hm_mask_det=True, out_dim=2048,
                 train_control='none', ctrl_in_init='copy_latent', latent_scale=1.0, img_pm1=False,
                 bf16=True, use_checkpoint=False):
        super().__init__()
        assert train_control in ('none', 'in_zero', 'full'), f'sd_train_control khong hop le: {train_control}'
        assert ctrl_in_init in ('copy_latent', 'random'), f'sd_ctrl_in_init khong hop le: {ctrl_in_init}'
        J = int(num_joints)
        self.num_joints = J
        self.lora_rank = int(lora_rank)
        self.hm_size = int(hm_size)
        self.hm_sigma = float(hm_sigma)
        self.hm_mask_det = bool(hm_mask_det)
        self.train_control = train_control
        self.ctrl_in_init = ctrl_in_init
        self.latent_scale = float(latent_scale)
        self.img_pm1 = bool(img_pm1)
        self.bf16 = bool(bf16)

        # outmodules: giong VPDEncoder, ldm_prior = [320, 640 + J, 1280 + 1280 + J] (DPMesh J = 30: [320, 670, 2590])
        prior = [320, 640 + J, 1280 + 1280 + J]
        self.ldm_prior = prior
        self.outmodules = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(prior[0], prior[0], 3, stride=2, padding=1),
                nn.GroupNorm(16, prior[0]),
                nn.ReLU(),
                nn.Conv2d(prior[0], prior[0], 3, stride=2, padding=1),
            ),
            nn.Sequential(
                nn.Conv2d(prior[1], prior[1], 3, stride=2, padding=1),
            ),
            nn.Sequential(
                nn.Conv2d(sum(prior), out_dim, 1),
                nn.GroupNorm(16, out_dim),
                nn.ReLU(),
            ),
        ])
        self.outmodules.apply(self._init_weights)                          # VPDEncoder._init_weights
        self.kp_lifting = nn.Linear(2, 768)                                # common/model.py: prompt toa do khop

        self.vae = SDVAEEncoder()
        self.unet = SDUNet(use_checkpoint=use_checkpoint)
        self.control = ControlNet(hint_channels=J, use_checkpoint=use_checkpoint)

        # Ban do cross-attention: chi U-Net, chi input_blocks (down) + output_blocks (up), giong
        # attn_selector 'up_cross+down_cross' cua UNetWrapper (bo middle). <= 256 token = 8x8 va 16x16.
        self._store = AttnStore(max_q=256)
        for part in (self.unet.input_blocks, self.unet.output_blocks):
            for m in part.modules():
                if isinstance(m, CrossAttention):
                    m.attn_store = self._store
        self._trainable_names = []
        self.lora_params = []

    @staticmethod
    def _init_weights(m):
        if isinstance(m, (nn.Conv2d, nn.Linear)):
            nn.init.trunc_normal_(m.weight, std=.02)
            nn.init.constant_(m.bias, 0)

    # ------------------------------------------------------------------ khoi tao / dong bang
    def load_pretrained(self, path):
        """Nap U-Net, ControlNet, VAE encoder tu checkpoint ControlNet 1.0 (control_sd15_openpose.pth).
        Bo: lop 'out' cua U-Net, decoder VAE, CLIP, input_hint_block. Conv dau ControlNet: kich thuoc doi
        (4 -> 4 + J) -> 'copy_latent' chep 4 kenh latent + bias, kenh heatmap giu khoi tao mac dinh.
        Tra chuoi bao cao (so key) de log. Goi TRUOC finalize()."""
        assert os.path.isfile(path), f'Khong thay checkpoint SD: {path}'
        sd = _load_raw_state(path)
        u = {k: v for k, v in _sub(sd, 'model.diffusion_model.').items() if not k.startswith('out.')}
        c = {k: v for k, v in _sub(sd, 'control_model.').items() if not k.startswith('input_hint_block.')}
        v = {k: x for k, x in _sub(sd, 'first_stage_model.').items()
             if k.startswith('encoder.') or k.startswith('quant_conv.')}
        assert u and c and v, ('Checkpoint phai co du model.diffusion_model.* / control_model.* / first_stage_model.* '
                               '(ban ControlNet 1.0 day du, vd control_sd15_openpose.pth ~5.7 GB)')
        mu, uu = self.unet.load_state_dict(u, strict=False)
        assert not mu and not uu, f'U-Net lech key: thieu {mu[:5]} ({len(mu)}), thua {uu[:5]} ({len(uu)})'
        mv, uv = self.vae.load_state_dict(v, strict=False)
        assert not mv and not uv, f'VAE lech key: thieu {mv[:5]} ({len(mv)}), thua {uv[:5]} ({len(uv)})'
        w_in = c.pop('input_blocks.0.0.weight')
        b_in = c.pop('input_blocks.0.0.bias')
        mc, uc = self.control.load_state_dict(c, strict=False)
        assert set(mc) == {'input_blocks.0.0.weight', 'input_blocks.0.0.bias'} and not uc, \
            f'ControlNet lech key: thieu {mc[:5]} ({len(mc)}), thua {uc[:5]} ({len(uc)})'
        conv = self.control.input_blocks[0][0]
        assert w_in.shape[1] == conv.in_channels - self.num_joints, f'conv dau ControlNet goc {tuple(w_in.shape)}'
        if self.ctrl_in_init == 'copy_latent':
            with torch.no_grad():
                conv.weight[:, :w_in.shape[1]].copy_(w_in)
                conv.bias.copy_(b_in)
        n_skip = len(sd) - len(u) - len(c) - 2 - len(v)
        return (f'SD nap {path}: U-Net {len(u)} key, ControlNet {len(c) + 2} key (conv dau {self.ctrl_in_init}, '
                f'goc {tuple(w_in.shape)} -> {tuple(conv.weight.shape)}), VAE encoder {len(v)} key; '
                f'bo {n_skip} key (out U-Net, decoder VAE, CLIP, input_hint_block, ...)')

    def finalize(self):
        """Dong bang + LoRA + chon tham so train theo train_control; doi trong so dong bang sang bf16 (bf16).
        Goi SAU load_pretrained (giong DPMesh: U-Net requires_grad False roi inject_trainable_lora)."""
        for p in self.parameters():
            p.requires_grad = False
        self.lora_params, _ = inject_trainable_lora(self.unet, r=self.lora_rank)
        for p in self.lora_params:
            p.requires_grad = True
        for p in list(self.outmodules.parameters()) + list(self.kp_lifting.parameters()):
            p.requires_grad = True
        if self.train_control == 'full':
            for p in self.control.parameters():
                p.requires_grad = True
        elif self.train_control == 'in_zero':
            for m in (self.control.input_blocks[0], self.control.zero_convs, self.control.middle_block_out):
                for p in m.parameters():
                    p.requires_grad = True
        if self.bf16:
            for p in self.parameters():
                if not p.requires_grad:
                    p.data = p.data.to(torch.bfloat16)
        self._trainable_names = [n for n, p in self.named_parameters() if p.requires_grad]
        return self

    def trainable_parameters(self):
        d = dict(self.named_parameters())
        return [d[n] for n in self._trainable_names]

    def trainable_state(self):
        """Chi tham so train (LoRA, outmodules, kp_lifting [+ ControlNet]) -> luu checkpoint nhe."""
        d = dict(self.named_parameters())
        return {n: d[n].detach().clone().cpu() for n in self._trainable_names}

    def load_trainable_state(self, state):
        d = dict(self.named_parameters())
        assert set(state) == set(self._trainable_names), \
            (f'sd_state lech tham so train: thieu {sorted(set(self._trainable_names) - set(state))[:5]}, '
             f'thua {sorted(set(state) - set(self._trainable_names))[:5]}')
        with torch.no_grad():
            for n, v in state.items():
                d[n].copy_(v.to(d[n].device, d[n].dtype))

    def num_params(self):
        tot = sum(p.numel() for p in self.parameters())
        tr = sum(p.numel() for p in self.trainable_parameters())
        return tot, tr

    # ------------------------------------------------------------------ forward
    def make_heatmap(self, g, det):
        """g (B,J,2) toa do o luoi hm_size (tam o i = i), det (B,J) -> (B,J,H,W) Gauss sigma (giong
        make_2d_gaussian_heatmap cua DPMesh), khop det = 0 -> 0 neu hm_mask_det."""
        S = self.hm_size
        ar = torch.arange(S, device=g.device, dtype=torch.float32)
        yy, xx = torch.meshgrid(ar, ar, indexing='ij')
        x = g[..., 0, None, None].float()
        y = g[..., 1, None, None].float()
        heat = torch.exp(-((xx - x) / self.hm_sigma) ** 2 / 2 - ((yy - y) / self.hm_sigma) ** 2 / 2)
        if self.hm_mask_det:
            heat = heat * (det.float() > 0).float()[..., None, None]
        return heat

    def _attn_maps(self, B):
        """Gop ban do cross-attention da luu theo do phan giai: trung binh cac lop -> (B,J,s,s) cho s = 8, 16."""
        groups = {}
        for a in self._store.maps:                                         # (B, Nq, J)
            s = int(round(a.shape[1] ** 0.5))
            groups.setdefault(s, []).append(a.float().transpose(1, 2).reshape(B, a.shape[2], s, s))
        self._store.clear()
        assert 8 in groups and 16 in groups, f'Thieu ban do attention 8x8 / 16x16: co {sorted(groups)}'
        return torch.stack(groups[8]).mean(0), torch.stack(groups[16]).mean(0)

    def forward(self, img, xy, det):
        """img (B,3,256,256) RGB [0,1]; xy (B,J,2) [-1,1] (quy uoc Refiner); det (B,J) -> f8 (B,2048,8,8).
        Goi trong torch.autocast(bf16) khi self.bf16 (trong so dong bang la bf16)."""
        B = img.shape[0]
        J = self.num_joints
        assert xy.shape[1] == J, f'xy co {xy.shape[1]} khop, can {J}'
        x = img * 2 - 1 if self.img_pm1 else img                           # DPMesh: anh [0,1] (ToTensor / 255)
        with torch.no_grad():
            lat = self.vae(x) * self.latent_scale                          # DPMesh: encode(x).mode(), khong nhan scale
        S = self.hm_size
        assert lat.shape[-1] == S and lat.shape[-2] == S, f'latent {tuple(lat.shape)} != luoi heatmap {S}'
        g = (xy.float() + 1) * 0.5 * S - 0.5                               # [-1,1] -> o luoi S (giong Refiner)
        heat = self.make_heatmap(g, det).to(lat.dtype)
        context = self.kp_lifting(g / S)                                   # DPMesh: joints / input_hm_shape
        t = torch.ones((B,), device=img.device, dtype=torch.long)          # DPMesh: t = 1, 1 lan forward
        self._store.clear()
        control = self.control(lat, heat, t, context)
        outs = self.unet(lat, t, context, control=control)                 # [1280@4, 1280@8, 640@16, 320@32]
        a8, a16 = self._attn_maps(B)
        outs[1] = torch.cat([outs[1], a8.to(outs[1].dtype)], dim=1)        # 1280 + J @ 8
        outs[2] = torch.cat([outs[2], a16.to(outs[2].dtype)], dim=1)       # 640 + J @ 16
        o = outs[::-1]                                                     # [320@32, 657@16, 1297@8, 1280@4]
        feats = [o[0], o[1], torch.cat([o[2], F.interpolate(o[3], scale_factor=2)], dim=1)]
        y = torch.cat([self.outmodules[0](feats[0]), self.outmodules[1](feats[1]), feats[2]], dim=1)
        return self.outmodules[2](y)                                       # (B, 2048, 8, 8)


def build_sd_encoder(arch, sd_ckpt, log=print):
    """arch = img_lifter_tools.sd_arch() (hoac ck['sd_arch']). Xay -> nap SD goc -> finalize. Tra mo hinh (CPU)."""
    m = SDFeatureEncoder(num_joints=int(arch['num_joints']), lora_rank=int(arch['lora_rank']),
                         hm_size=int(arch['hm_size']), hm_sigma=float(arch['hm_sigma']),
                         hm_mask_det=bool(arch['hm_mask_det']), train_control=str(arch['train_control']),
                         ctrl_in_init=str(arch['ctrl_in_init']), latent_scale=float(arch['latent_scale']),
                         img_pm1=bool(arch['img_pm1']), bf16=bool(arch['bf16']),
                         use_checkpoint=bool(arch.get('grad_ckpt', False)))
    log(m.load_pretrained(sd_ckpt))
    m.finalize()
    tot, tr = m.num_params()
    log(f'SDFeatureEncoder: {tot / 1e6:.1f}M tham so, train {tr / 1e6:.2f}M (LoRA r{m.lora_rank} + outmodules + '
        f'kp_lifting, ControlNet {m.train_control}) | prior {m.ldm_prior} | bf16 {m.bf16}')
    return m
