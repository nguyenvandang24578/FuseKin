"""
lib/models/fusekin_v8.py -- FuseKin v8, moc M0 (2026-11-13, nguoi dung duyet): mang END-TO-END, ANH LAM CHINH.

  anh + heatmap 2D detector (tiem o stride 4, nhu 3DCrowdNet / JOTR) -> ResNet-50 (train end-to-end, BN train)
  -> F (2048 x 8 x 8) -> Conv 1x1 (D kenh) + PE sin 2D -> Transformer encoder (v8_enc_layers lop)
  -> Transformer decoder (v8_dec_layers lop): 3 token SMPL (pose, shape, camera) + 17 query khop H36M
  -> sau MOI lop decoder: hoi quy phan du theo tang (pose 24 x 6D, beta 10, camera 3, khop 3D 17 x 3) -> SMPL.

M0 = moc tai tao JOTR KHONG co phan 3D cua ho (bang 4 JOTR, dong "flatting / none"): khong khoi dac trung 3D,
khong lay mau dac trung tai khop, khong contrastive. Cung KHONG co phan cua FuseKin (lifter, HyperGCN, IK):
cac phan nay them o M1+ (moi phan 1 key, mac dinh tat).

Giao dien giong ARTS -> Student_Trainer / Tester / test_student_3dpw_sets dung nguyen:
  forward(image, joints, is_train, joints_mask, pose_2d=None, cam_bbox=None) -> dict output nhu Student.
  - refine_history: output SMPL cua cac lop decoder truoc lop cuoi (base._hard_losses giam sat ca chung).
  - ctx_3d: khop 3D cua 17 query (lop cuoi), giam sat bang loss ctx (MODEL.v5_ctx_loss_w) trong Student_Trainer.
  - lifted_joints_3d = ctx_3d (detach): cot LIFT luc danh gia = do chinh xac cua query khop.
Thuoc tinh smpl_model BAT BUOC (base._joint_log_sigma doc net.smpl_model).
"""
import math

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from core.config import cfg
from models.backbones.resnet import ResNetBackbone
from models.ARTS import make_heatmaps
from models.hybrik_smpl import rotmat_to_aa_safe

SMPL_MEAN_PARAMS = 'data_final/base_data/smpl_mean_params.npz'
N_SMPL_TOK = 3     # pose, shape, camera
N_JOINT = 17       # query khop H36M-17


def rot6d_to_rotmat(x):
    """(N, 6) -> (N, 3, 3). Quy uoc SPIN / smpl_mean_params: x.view(-1, 3, 2), cot 0 = a1, cot 1 = a2."""
    x = x.reshape(-1, 3, 2)
    b1 = F.normalize(x[..., 0], dim=-1)
    a2 = x[..., 1]
    b2 = F.normalize(a2 - (b1 * a2).sum(-1, keepdim=True) * b1, dim=-1)
    b3 = torch.cross(b1, b2, dim=-1)
    return torch.stack([b1, b2, b3], dim=-1)


def sine_pos_2d(h, w, dim, device, dtype, temperature=10000.0):
    """PE sin 2D kieu DETR (normalize, scale 2 pi) -> (h * w, dim); thu tu o = hang * w + cot (giong flatten(2))."""
    npf = dim // 2
    y = (torch.arange(h, device=device, dtype=torch.float32) + 1) / h * 2 * math.pi
    x = (torch.arange(w, device=device, dtype=torch.float32) + 1) / w * 2 * math.pi
    gy, gx = torch.meshgrid(y, x, indexing='ij')                                   # (h, w)
    dim_t = torch.arange(npf, device=device, dtype=torch.float32)
    dim_t = temperature ** (2 * torch.div(dim_t, 2, rounding_mode='floor') / npf)
    px = gx[..., None] / dim_t
    py = gy[..., None] / dim_t
    px = torch.stack([px[..., 0::2].sin(), px[..., 1::2].cos()], dim=-1).flatten(-2)
    py = torch.stack([py[..., 0::2].sin(), py[..., 1::2].cos()], dim=-1).flatten(-2)
    return torch.cat([py, px], dim=-1).reshape(h * w, dim).to(dtype)


class EncLayer(nn.Module):
    """Lop encoder post-norm giong JOTR (normalize_before = False): self-attn -> FFN."""

    def __init__(self, d, nhead=8, ff=2048, drop=0.1):
        super().__init__()
        self.sa = nn.MultiheadAttention(d, nhead, dropout=drop, batch_first=True)
        self.l1, self.l2 = nn.Linear(d, ff), nn.Linear(ff, d)
        self.n1, self.n2 = nn.LayerNorm(d), nn.LayerNorm(d)
        self.d0, self.d1, self.d2 = nn.Dropout(drop), nn.Dropout(drop), nn.Dropout(drop)

    def forward(self, x, pos):
        q = k = x + pos
        x = self.n1(x + self.d1(self.sa(q, k, x, need_weights=False)[0]))
        x = self.n2(x + self.d2(self.l2(self.d0(F.relu(self.l1(x))))))
        return x


class DecLayer(nn.Module):
    """Lop decoder post-norm giong JOTR: self-attn giua query -> cross-attn toi token anh -> FFN."""

    def __init__(self, d, nhead=8, ff=2048, drop=0.1):
        super().__init__()
        self.sa = nn.MultiheadAttention(d, nhead, dropout=drop, batch_first=True)
        self.ca = nn.MultiheadAttention(d, nhead, dropout=drop, batch_first=True)
        self.l1, self.l2 = nn.Linear(d, ff), nn.Linear(ff, d)
        self.n1, self.n2, self.n3 = nn.LayerNorm(d), nn.LayerNorm(d), nn.LayerNorm(d)
        self.d0, self.d1, self.d2, self.d3 = nn.Dropout(drop), nn.Dropout(drop), nn.Dropout(drop), nn.Dropout(drop)

    def forward(self, tgt, qpos, mem, mpos):
        q = k = tgt + qpos
        tgt = self.n1(tgt + self.d1(self.sa(q, k, tgt, need_weights=False)[0]))
        tgt = self.n2(tgt + self.d2(self.ca(tgt + qpos, mem + mpos, mem, need_weights=False)[0]))
        tgt = self.n3(tgt + self.d3(self.l2(self.d0(F.relu(self.l1(tgt))))))
        return tgt


class CascadeHead(nn.Module):
    """Hoi quy phan du theo tang (giong cascade_fc cua JOTR): lop 0 du doan tu token; lop l > 0 du doan phan du tu
    [token, du doan truoc]. init (1, k) | None: lop 0 = init + Linear(token) (vd pose trung binh)."""

    def __init__(self, d, k, n_layers, init=None):
        super().__init__()
        self.heads = nn.ModuleList([nn.Linear(d, k)] + [nn.Linear(d + k, k) for _ in range(n_layers - 1)])
        for m in self.heads:                                                         # khoi tao nho: luc dau ~ init
            nn.init.normal_(m.weight, std=1e-3)
            nn.init.zeros_(m.bias)
        if init is not None:
            self.register_buffer('init', init.float().reshape(1, k))
        else:
            self.init = None

    def forward(self, i, h, prev):
        if i == 0:
            out = self.heads[0](h)
            return out + self.init if self.init is not None else out
        return prev + self.heads[i](torch.cat([h, prev], dim=-1))


class V8Head(nn.Module):
    """Encoder + decoder + cac head + SMPL (= smpl_model cua V8Net)."""

    def __init__(self, dim=256, n_enc=3, n_dec=3, img_ch=2048):
        super().__init__()
        self.n_dec = int(n_dec)
        self.img_proj = nn.Conv2d(img_ch, dim, 1)
        self.encoder = nn.ModuleList([EncLayer(dim) for _ in range(int(n_enc))])
        self.decoder = nn.ModuleList([DecLayer(dim) for _ in range(self.n_dec)])
        self.dec_norm = nn.LayerNorm(dim)                       # JOTR: norm sau MOI lop decoder, dua vao lop tiep
        self.query_pos = nn.Embedding(N_SMPL_TOK + N_JOINT, dim)
        self.tgt = nn.Parameter(torch.zeros(1, N_SMPL_TOK + N_JOINT, dim))   # noi dung ban dau = 0 (giong JOTR)
        for p in list(self.encoder.parameters()) + list(self.decoder.parameters()):  # JOTR _reset_parameters
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

        mean = np.load(SMPL_MEAN_PARAMS)
        init_pose = torch.from_numpy(mean['pose'][:].astype('float32')).reshape(1, 24 * 6)
        init_shape = torch.from_numpy(mean['shape'][:].astype('float32')).reshape(1, 10)
        self.pose_head = CascadeHead(dim, 24 * 6, self.n_dec, init_pose)
        self.shape_head = CascadeHead(dim, 10, self.n_dec, init_shape)
        self.cam_head = CascadeHead(dim, 3, self.n_dec)
        self.joint_head = CascadeHead(dim, 3, self.n_dec)      # khop 3D root-relative (met), moi query 1 khop

        from utils.smpl import SMPL as SMPLModel
        self.human_model = SMPLModel()
        self.human_model_layer = self.human_model.layer['neutral']
        self.register_buffer('joint_regressor_t', torch.from_numpy(self.human_model.joint_regressor).float())

    # ---------------- SMPL (chep tu Student, cung quy uoc camera / chieu) ----------------
    def get_camera_trans(self, cam_param):
        """cam_param (B, 3) = [tx, ty, gamma] -> cam_trans (B, 3) = [tx, ty, tz]."""
        t_xy = cam_param[:, :2]
        gamma = torch.sigmoid(cam_param[:, 2])
        k_value = math.sqrt(cfg.DATASET.focal[0] * cfg.DATASET.focal[1] * cfg.DATASET.camera_3d_size
                            * cfg.DATASET.camera_3d_size / (cfg.input_img_shape[0] * cfg.input_img_shape[1]))
        return torch.cat([t_xy, (k_value * gamma)[:, None]], dim=1)

    def get_coord(self, smpl_pose, smpl_shape, smpl_trans):
        B = smpl_pose.shape[0]
        mesh_cam, _ = self.human_model_layer(smpl_pose, smpl_shape, smpl_trans)
        joint_cam = torch.bmm(self.joint_regressor_t[None].repeat(B, 1, 1), mesh_cam)
        x = joint_cam[:, :, 0] / (joint_cam[:, :, 2] + 1e-4) * cfg.DATASET.focal[0] + cfg.DATASET.princpt[0]
        y = joint_cam[:, :, 1] / (joint_cam[:, :, 2] + 1e-4) * cfg.DATASET.focal[1] + cfg.DATASET.princpt[1]
        x = x / cfg.input_img_shape[1] * cfg.output_hm_shape[2]
        y = y / cfg.input_img_shape[0] * cfg.output_hm_shape[1]
        joint_proj = torch.stack((x, y), 2)
        mesh_cam_render = mesh_cam.clone()
        root_cam = joint_cam[:, self.human_model.root_joint_idx, None, :]
        return joint_proj, joint_cam - root_cam, mesh_cam - root_cam, mesh_cam_render

    def smpl_outputs(self, pose6d, shape, cam_raw, is_train):
        """pose6d (B, 144) + shape (B, 10) + camera tho (B, 3) -> dict output nhu Student._smpl_outputs."""
        B = pose6d.shape[0]
        aa = rotmat_to_aa_safe(rot6d_to_rotmat(pose6d.float().reshape(B * 24, 6))).reshape(B, 24, 3)
        aa = torch.cat([aa[:, :22], torch.zeros_like(aa[:, 22:])], dim=1).reshape(B, 72)   # ban tay = 0 (nhu Student)
        shape = shape.float()
        cam_trans = self.get_camera_trans(cam_raw.float())
        joint_proj, joint_cam, mesh_cam, mesh_render = self.get_coord(aa, shape, cam_trans)
        if is_train:                                    # loss chieu 2D: detach pose (giu shape, camera), nhu Student
            _, _, _, mesh_render_det = self.get_coord(aa.detach(), shape, cam_trans)
        else:
            mesh_render_det = mesh_render
        return {'joint_proj': joint_proj, 'joint_cam': joint_cam, 'smpl_mesh_cam': mesh_cam,
                'smpl_mesh_cam_render': mesh_render, 'smpl_mesh_cam_render_det': mesh_render_det,
                'smpl_pose': aa, 'smpl_shape': shape, 'cam_param': cam_trans}

    def forward(self, fmap, is_train=True):
        B, _, h, w = fmap.shape
        mem = self.img_proj(fmap).flatten(2).transpose(1, 2)                          # (B, h*w, D)
        mpos = sine_pos_2d(h, w, mem.shape[-1], mem.device, mem.dtype)[None]          # (1, h*w, D)
        for layer in self.encoder:
            mem = layer(mem, mpos)
        qpos = self.query_pos.weight[None]                                           # (1, 20, D)
        tgt = self.tgt.expand(B, -1, -1)
        pose = shape = cam = joints = None
        per_layer = []
        for i, layer in enumerate(self.decoder):
            tgt = self.dec_norm(layer(tgt, qpos, mem, mpos))
            pose = self.pose_head(i, tgt[:, 0], pose)
            shape = self.shape_head(i, tgt[:, 1], shape)
            cam = self.cam_head(i, tgt[:, 2], cam)
            joints = self.joint_head(i, tgt[:, N_SMPL_TOK:], joints)                 # (B, 17, 3)
            per_layer.append((pose, shape, cam))
        result = self.smpl_outputs(pose, shape, cam, is_train)
        # Cac lop truoc lop cuoi -> refine_history (base._hard_losses giam sat voi trong so v3_refine_gamma)
        result['refine_history'] = [self.smpl_outputs(p, s, c, is_train) for (p, s, c) in per_layer[:-1]] \
            if is_train else []
        j3d = joints - joints[:, 0:1]                                                # root-relative tai Pelvis
        result['ctx_3d'] = j3d                                                        # loss ctx (v5_ctx_loss_w)
        result['lifted_joints_3d'] = j3d.detach()                                     # cot LIFT = query khop
        result['privileged_3d'] = j3d                                                 # tat bang v7_no_privileged
        result['feat_joint'] = tgt[:, N_SMPL_TOK:]
        result['feat_global'] = tgt[:, :N_SMPL_TOK].mean(1)
        return result


class V8Net(nn.Module):
    """Vo boc FuseKin v8 (thay ARTS khi MODEL.v8_e2e): backbone + heatmap 2D + V8Head."""

    def __init__(self):
        super().__init__()
        self.mode = 'v8'
        self.backbone = ResNetBackbone(cfg.MODEL.resnet_type)                       # BN thuong (train), nhu JOTR
        self.backbone.enable_heatmap_fusion(N_JOINT)                                 # conv khoi tao 0
        bb_src = cfg.MODEL.get('backbone_pretrained', '')
        if bb_src:
            self.backbone.load_pretrained(
                bb_src, spin_checkpoint=cfg.MODEL.get('spin_checkpoint',
                                                      'data_final/base_data/spin_model_checkpoint.pth.tar'))
        else:
            print('[CANH BAO] V8: backbone KHONG nap pretrain -> dac trung anh ngau nhien.')
        self.smpl_model = V8Head(dim=int(cfg.MODEL.v8_dim), n_enc=int(cfg.MODEL.v8_enc_layers),
                                 n_dec=int(cfg.MODEL.v8_dec_layers))
        self.joint_drop = float(cfg.MODEL.v8_joint_drop)
        print(f'===> FuseKin v8 (M0): dim {cfg.MODEL.v8_dim} | encoder {cfg.MODEL.v8_enc_layers} lop | decoder '
              f'{cfg.MODEL.v8_dec_layers} lop | che khop 2D luc train {self.joint_drop}')

    def forward(self, image, joints, is_train=True, joints_mask=None, pose_2d=None, cam_bbox=None):
        """joints = 2D detector (B, 17, 2|3) trong [-1, 1]; joints_mask (B, 17, 1). pose_2d / cam_bbox: khong dung."""
        B = image.shape[0]
        p2 = joints[:, 0] if joints.dim() == 4 else joints
        if joints_mask is None:
            m = (p2[..., 2:3] > 0).float() if p2.shape[-1] >= 3 else torch.ones_like(p2[..., :1])
        else:
            m = joints_mask.reshape(B, N_JOINT, 1).float()
        if self.training and self.joint_drop > 0:
            # Giong JOTR: che ngau nhien mot phan khop 2D dau vao luc train -> mang phai doc anh de bu
            m = m * (torch.rand_like(m) >= self.joint_drop).float()
        heat = make_heatmaps(p2, m, size=image.shape[-1] // 4, sigma=float(cfg.MODEL.v3_heatmap_sigma))
        fmap = self.backbone(image, heatmap=heat)[0]                                 # (B, 2048, 8, 8)
        return self.smpl_model(fmap, is_train=is_train)
