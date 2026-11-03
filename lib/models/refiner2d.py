"""
lib/models/refiner2d.py -- Refiner 2D (2026-11-03): anh crop + 2D detector -> 2D da sua cua DUNG nguoi + do bat dinh.

Y tuong: 2D detector (OpenPose / HHRNet luc test, GT + PoseFix kieu JOTR luc train) cho biet DANG dung nguoi nao;
anh cho biet khop nao thuc su thuoc nguoi do (nham nguoi, bi che, bo sot). Refiner du doan lai heatmap 17 khop H36M
o luoi 64x64 (khung output_hm_shape), lay toa do bang soft-argmax (kha vi) va do bat dinh tung khop (Laplace b).

  crop 256 -> ResNet-50 (SPIN) -- stride 4: + heatmap 2D detector (heat_fuse, khoi tao 0, giong [A]) --> layer4 8x8
           -> 3 x [ConvT 4x4 s2 + BN + ReLU] -> 64x64 -> Conv1x1 -> logits (B, 17, 64, 64)
           -> softmax tren luoi -> xy (soft-argmax) ; dac trung ky vong tai khop (detach) -> MLP -> log_b

Quy uoc toa do: [-1, 1] giong Human36M17Dataset; o luoi g <-> p = (g + 0.5) / 64 * 2 - 1 (giong ARTS.make_heatmaps).
log_b, sai so trong loss: don vi o luoi 64 (1 o = 4 px crop 256).
Dung boi main/train_refiner2d.py, main/eval_refiner2d.py. Khong anh huong Teacher / Student.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

from models.ARTS import make_heatmaps
from models.backbones.resnet import ResNetBackbone

H36M_NAMES = ('Pelvis', 'R_Hip', 'R_Knee', 'R_Ankle', 'L_Hip', 'L_Knee', 'L_Ankle', 'Torso', 'Neck', 'Nose',
              'Head_top', 'L_Shoulder', 'L_Elbow', 'L_Wrist', 'R_Shoulder', 'R_Elbow', 'R_Wrist')
# 12 khop tay chan co cung ten o moi dinh dang (giong main/compare_2d_sources.py LIMB) -> thuoc do 2D it phu thuoc
# dinh nghia khop (hong / dau / than khac nhau giua OpenPose, annotation, H36M regressor)
LIMB_NAMES = ('L_Shoulder', 'R_Shoulder', 'L_Elbow', 'R_Elbow', 'L_Wrist', 'R_Wrist',
              'L_Hip', 'R_Hip', 'L_Knee', 'R_Knee', 'L_Ankle', 'R_Ankle')
LIMB_IDX = tuple(H36M_NAMES.index(n) for n in LIMB_NAMES)


class Refiner2D(nn.Module):
    def __init__(self, num_joints=17, deconv_ch=256, heat_sigma=2.0, pretrained='spin', bb_bn_eval=True):
        super().__init__()
        self.num_joints = num_joints
        self.heat_sigma = float(heat_sigma)
        self.bb_bn_eval = bool(bb_bn_eval)
        self.backbone = ResNetBackbone(50)
        if pretrained:
            self.backbone.load_pretrained(pretrained)
        self.backbone.enable_heatmap_fusion(num_joints)       # conv khoi tao 0 -> luc dau = backbone pretrain
        layers, cin = [], self.backbone.out_channels
        for _ in range(3):
            layers += [nn.ConvTranspose2d(cin, deconv_ch, kernel_size=4, stride=2, padding=1, bias=False),
                       nn.BatchNorm2d(deconv_ch), nn.ReLU(inplace=True)]
            cin = deconv_ch
        self.deconv = nn.Sequential(*layers)
        self.final = nn.Conv2d(deconv_ch, num_joints, kernel_size=1)
        # Head do bat dinh: [dac trung ky vong tai khop, co khop dau vao, |du doan - dau vao|, dinh xac suat] (detach)
        self.unc = nn.Sequential(nn.Linear(deconv_ch + 3, 128), nn.ReLU(inplace=True), nn.Linear(128, 1))
        for m in self.deconv.modules():
            if isinstance(m, nn.ConvTranspose2d):
                nn.init.normal_(m.weight, std=0.001)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
        nn.init.normal_(self.final.weight, std=0.001)
        nn.init.zeros_(self.final.bias)

    def train(self, mode=True):
        super().train(mode)
        if self.bb_bn_eval:
            # Giu thong ke BN pretrain cua backbone (trong so van duoc train), giong ARTS [A]
            for m in self.backbone.modules():
                if isinstance(m, nn.BatchNorm2d):
                    m.eval()
        return self

    def head_parameters(self):
        """Tham so KHONG thuoc backbone (tru heat_fuse: coi nhu head, lr day du)."""
        bb = {id(p) for n, p in self.backbone.named_parameters() if not n.startswith('heat_fuse.')}
        return [p for p in self.parameters() if id(p) not in bb]

    def backbone_parameters(self):
        return [p for n, p in self.backbone.named_parameters() if not n.startswith('heat_fuse.')]

    def forward(self, img, pose_2d, joints_mask):
        """img (B,3,256,256) RGB [0,1]; pose_2d (B,J,2|3) [-1,1]; joints_mask (B,J,1) (0 = khong co khop)."""
        B, J = pose_2d.shape[0], self.num_joints
        p2 = pose_2d[..., :2].float()
        m = (joints_mask.reshape(B, J).float() > 0).float()
        heat = make_heatmaps(p2, m.unsqueeze(-1), size=img.shape[-1] // 4, sigma=self.heat_sigma)
        f8 = self.backbone(img, heatmap=heat)[0]                                   # (B, 2048, 8, 8)
        x = self.deconv(f8)                                                        # (B, C, 64, 64)
        logits = self.final(x)                                                     # (B, J, H, W)
        H, W = logits.shape[-2:]
        prob = F.softmax(logits.flatten(2).float(), dim=-1)                        # (B, J, H*W)
        gy, gx = torch.meshgrid(torch.arange(H, device=img.device, dtype=prob.dtype),
                                torch.arange(W, device=img.device, dtype=prob.dtype), indexing='ij')
        xg = (prob * gx.reshape(1, 1, -1)).sum(-1)
        yg = (prob * gy.reshape(1, 1, -1)).sum(-1)
        xy_grid = torch.stack([xg, yg], -1)                                        # (B, J, 2) don vi o luoi
        scale = torch.tensor([W, H], device=img.device, dtype=prob.dtype)
        xy = (xy_grid + 0.5) / scale * 2 - 1                                       # [-1, 1]

        # Do bat dinh: chi hoc head unc (dau vao detach -> NLL khong keo backbone / heatmap)
        feat_j = torch.einsum('bjn,bcn->bjc', prob.detach(), x.detach().flatten(2).float())
        in_grid = (p2 + 1) * 0.5 * scale - 0.5
        dist = torch.norm(xy_grid.detach() - in_grid, dim=-1) * m                  # 0 khi khong co khop dau vao
        peak = prob.detach().amax(-1)
        unc_in = torch.cat([feat_j, m.unsqueeze(-1), dist.unsqueeze(-1), peak.unsqueeze(-1)], -1)
        log_b = self.unc(unc_in).squeeze(-1).clamp(-6.0, 6.0)                     # (B, J), log(o luoi)
        return {'logits': logits, 'xy': xy, 'xy_grid': xy_grid, 'log_b': log_b}


def refiner_losses(out, target_xy, valid, sigma=2.0, w_coord=1.0, w_heat=0.1, w_nll=0.1):
    """target_xy (B,J,2) [-1,1] (GT 2D sach); valid (B,J) 0/1. Tra (loss tong, dict log).
    coord: loi Euclid soft-argmax (o luoi); heat: CE giua softmax heatmap va Gauss muc tieu chuan hoa (sigma o luoi);
    nll: Laplace |e| / b + log b voi e detach -> chi hoc log_b. Trong so co dinh, KHONG qua AWL."""
    logits = out['logits']
    H, W = logits.shape[-2:]
    scale = torch.tensor([W, H], device=logits.device, dtype=torch.float32)
    tg = (target_xy.float() + 1) * 0.5 * scale - 0.5                               # (B, J, 2) o luoi
    valid = valid.float()
    n = valid.sum().clamp_min(1.0)
    err = torch.norm(out['xy_grid'].float() - tg, dim=-1)                          # (B, J)
    l_coord = (err * valid).sum() / n

    gx = torch.arange(W, device=logits.device, dtype=torch.float32)
    gy = torch.arange(H, device=logits.device, dtype=torch.float32)
    ex = torch.exp(-(gx.view(1, 1, -1) - tg[..., 0:1]) ** 2 / (2 * sigma ** 2))    # (B, J, W)
    ey = torch.exp(-(gy.view(1, 1, -1) - tg[..., 1:2]) ** 2 / (2 * sigma ** 2))    # (B, J, H)
    tgt = (ey.unsqueeze(-1) * ex.unsqueeze(-2)).flatten(2)                         # (B, J, H*W)
    tgt = tgt / tgt.sum(-1, keepdim=True).clamp_min(1e-12)
    ce = -(tgt * F.log_softmax(logits.flatten(2).float(), dim=-1)).sum(-1)       # (B, J)
    l_heat = (ce * valid).sum() / n

    log_b = out['log_b'].float()
    nll = err.detach() * torch.exp(-log_b) + log_b
    l_nll = (nll * valid).sum() / n

    loss = w_coord * l_coord + w_heat * l_heat + w_nll * l_nll
    return loss, {'coord': float(l_coord), 'heat': float(l_heat), 'nll': float(l_nll),
                  'err_px': float(l_coord) * 256.0 / W}


def person_size(gt_xy):
    """Kich thuoc nguoi = canh lon nhat bbox cua GT 2D (17 khop), cung don vi voi gt_xy. (B,)"""
    ext = gt_xy.amax(1) - gt_xy.amin(1)
    return ext.amax(-1).clamp_min(1e-6)


def far_counts(pred_xy, det, gt_xy, gt_valid, far_ratio=0.2, idx=LIMB_IDX):
    """Dem tren cac khop idx (mac dinh 12 khop tay chan). pred_xy, gt_xy (B,J,2) cung khung; det (B,J) = nguon co khop;
    gt_valid (B,J) = khop GT trong khung. Tra dict tensor (B, len(idx)): e (loi / kich thuoc nguoi), det, far, miss."""
    idx = list(idx)
    size = person_size(gt_xy)
    e = torch.norm(pred_xy[:, idx] - gt_xy[:, idx], dim=-1) / size[:, None]
    gv = gt_valid[:, idx].float()
    d = det[:, idx].float() * gv
    far = (e > far_ratio).float() * d
    miss = (1 - det[:, idx].float()) * gv
    return {'e': e, 'det': d, 'far': far, 'miss': miss, 'gt': gv}
