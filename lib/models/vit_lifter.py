"""
lib/models/vit_lifter.py (2026-11-18, nguoi dung duyet) -- Lifter diffusion 2D ViTPose + anh -> 3D, KHONG Refiner.

Ly do: Refiner 2D sinh ra de sua loi OpenPose; voi ViTPose thi bo Refiner. Anh van dua vao lifter qua ResNet-50
SPIN DONG BANG (khong heatmap 2D vao backbone: 2D chi la dieu kien cua diffusion).

  anh crop 256 -> ResNet-50 SPIN (dong bang, BN eval) -> f16 (B,1024,16,16), f8 (B,2048,8,8)
    feat_j (B,17,1024) = lay mau song tuyen f16 tai 2D ViTPose (khop thieu -> 0)
    f8 -> 64 token anh cho cross-attn trong moi block (nhu ImgLifter)
    g = TB(f8) (B,2048) -> Linear -> cong vao dieu kien MOI token khop (y tuong imgfeat_embed cua GraphormerNet)
  dieu kien tung khop cond7 = [x, y, log_b, det, cam_bbox]: xy ViTPose [-1,1], det = conf > nguong,
    log_b = -log(max(conf, eps)) (do bat dinh gia tu conf; cong Markov co tham so hoc nen chi can don dieu)

Denoiser, lich nhieu, loss (MPJPE x0 + xuong + goc), lay mau DDIM K gia thuyet: dung nguyen ImgLifter
(lib/models/img_lifter.py, KHONG sua). Chi thay denoiser bang lop con co them g (GlobImgDenoiser).
Dau vao chinh cua diffusion: pose 3D GT (17 khop, root-relative) cong nhieu luc train; tu nhieu thuan luc test.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

from models.backbones.resnet import ResNetBackbone
from models.img_lifter import ImgLifter, ImgLifterDenoiser

F16_CH = 1024     # so kenh layer3 ResNet-50 (stride 16)
F8_CH = 2048      # so kenh layer4 ResNet-50 (stride 32)


class FrozenSpinEncoder(nn.Module):
    """ResNet-50 SPIN dong bang. Khong luu vao checkpoint lifter (nap lai tu file SPIN luc test)."""

    def __init__(self, pretrained='spin'):
        super().__init__()
        self.backbone = ResNetBackbone(50)
        self.backbone.load_pretrained(pretrained)          # bat chuan hoa ImageNet trong backbone (anh [0,1])
        for p in self.parameters():
            p.requires_grad = False
        self.eval()

    def train(self, mode=True):
        # luon eval: BN dung thong ke SPIN, khong cap nhat
        return super().train(False)

    @torch.no_grad()
    def forward(self, img, xy, det):
        """img (B,3,256,256) [0,1]; xy (B,17,2) [-1,1] khung crop; det (B,17) 0/1
        -> f8 (B,2048,8,8), feat_j (B,17,1024)."""
        f8, _, f16 = self.backbone(img, return_f16=True)
        grid = xy.to(f16.dtype)[:, :, None, :]                                  # (B,17,1,2)
        fj = F.grid_sample(f16, grid, mode='bilinear', padding_mode='zeros', align_corners=False)
        fj = fj.squeeze(-1).transpose(1, 2) * det[..., None].to(f16.dtype)      # (B,17,1024), khop thieu = 0
        return f8.float(), fj.float()


class GlobImgDenoiser(ImgLifterDenoiser):
    """ImgLifterDenoiser + vector anh toan cuc cong vao dieu kien moi token (glob_img=False -> y het goc)."""

    def __init__(self, *args, glob_img=False, img_ch=F8_CH, **kwargs):
        super().__init__(*args, img_ch=img_ch, **kwargs)
        self.glob_img = bool(glob_img)
        if self.glob_img:
            dim = self.joint_pos.shape[-1]
            self.glob_embed = nn.Sequential(nn.LayerNorm(img_ch), nn.Linear(img_ch, dim))

    def encode_cond(self, cond7, feat_j, f8):
        c_j, img_tok = super().encode_cond(cond7, feat_j, f8)
        if self.glob_img:
            # c_j vao ca dau vao token lan adaLN cua moi block -> g anh huong moi khop o moi lop
            c_j = c_j + self.glob_embed(f8.mean(dim=(2, 3)))[:, None, :]
        return c_j, img_tok


class VitLifter(ImgLifter):
    """ImgLifter (17 khop, num_ctx = 0) voi denoiser GlobImgDenoiser va feat_j 1024 kenh."""

    def __init__(self, num_joints=17, dim=512, depth=8, num_heads=8, mlp_ratio=2.0, drop_path=0.1, timesteps=1000,
                 scale=3.0, feat_j_ch=F16_CH, markov=False, markov_gate_b=False, markov_iters=0,
                 markov_gate_positive=False, bone_paper=False, markov_msg_norm=False, glob_img=False):
        super().__init__(num_joints=num_joints, dim=dim, depth=depth, num_heads=num_heads, mlp_ratio=mlp_ratio,
                         drop_path=drop_path, timesteps=timesteps, scale=scale, feat_j_ch=feat_j_ch, num_ctx=0,
                         markov=markov, markov_gate_b=markov_gate_b, markov_iters=markov_iters,
                         markov_gate_positive=markov_gate_positive, bone_paper=bone_paper,
                         markov_msg_norm=markov_msg_norm)
        # thay denoiser goc bang lop con (cung tham so + glob_embed)
        self.denoiser = GlobImgDenoiser(num_joints, dim, depth, num_heads, mlp_ratio, drop_path,
                                        feat_j_ch=feat_j_ch, num_ctx=0, markov=markov, markov_gate_b=markov_gate_b,
                                        markov_iters=markov_iters, markov_gate_positive=markov_gate_positive,
                                        markov_msg_norm=markov_msg_norm, glob_img=glob_img)
        self.glob_img = bool(glob_img)


def vit_inputs(encoder, img, joints, jm, cam, logb_eps=0.01):
    """joints (B,17,3) [x, y, conf] ([-1,1], khop thieu = 0), jm (B,17) mask (conf > nguong, sau input_drop),
    cam (B,3) -> cond7 (B,17,7), feat_j (B,17,1024), f8 (B,2048,8,8), xy (B,17,2), log_b (B,17)."""
    det = (jm > 0).float()
    xy = joints[..., :2] * det[..., None]
    conf = joints[..., 2].clamp(0.0, 1.0) * det
    log_b = -torch.log(conf.clamp_min(float(logb_eps)))
    f8, fj = encoder(img, xy, det)
    cond7 = ImgLifter.make_cond7(xy, log_b, det, cam)
    return cond7, fj, f8, xy, log_b
